#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLUSTER_ROOT = Path.home() / "cluster"
DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_OUTPUT_NAME = "cluster_queue_snapshot.json"
DEFAULT_SUMMARY_NAME = "cluster_queue_summary.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture machine-readable cluster queue state.")
    parser.add_argument(
        "command",
        nargs="?",
        default="update",
        choices=("update",),
        help="Subcommand. Only 'update' is currently supported.",
    )
    parser.add_argument("--cluster-root", default=str(DEFAULT_CLUSTER_ROOT), help="Path to ~/cluster.")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help="Redis host used by the cluster queue.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_NAME, help="Snapshot JSON path.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_NAME, help="Summary markdown path.")
    parser.add_argument("--stdout-summary", action="store_true", help="Print the JSON payload after writing it.")
    return parser.parse_args()


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def redis_raw_lines(redis_host: str, *args: str) -> list[str]:
    result = run_capture(["redis-cli", "-h", redis_host, "--raw", *args])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def redis_hash(redis_host: str, key: str) -> dict[str, str]:
    lines = redis_raw_lines(redis_host, "HGETALL", key)
    if not lines:
        return {}
    out: dict[str, str] = {}
    for idx in range(0, len(lines) - 1, 2):
        out[lines[idx]] = lines[idx + 1]
    return out


def load_status_json(cluster_root: Path) -> dict[str, Any]:
    status_script = cluster_root / "status.sh"
    if not status_script.exists():
        return {"generated_at_utc": now_utc(), "hosts": [], "locks": [], "source": "missing_status_script"}
    result = run_capture(["bash", str(status_script), "--json"])
    if result.returncode != 0:
        return {
            "generated_at_utc": now_utc(),
            "hosts": [],
            "locks": [],
            "source": "status_script_failed",
            "stderr": result.stderr.strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "generated_at_utc": now_utc(),
            "hosts": [],
            "locks": [],
            "source": "invalid_status_json",
            "stdout_excerpt": result.stdout[:500],
        }
    if not isinstance(payload, dict):
        return {"generated_at_utc": now_utc(), "hosts": [], "locks": [], "source": "status_json_not_object"}
    payload.setdefault("generated_at_utc", now_utc())
    payload.setdefault("hosts", [])
    payload.setdefault("locks", [])
    payload["source"] = "status_sh_json"
    return payload


def collect_jobs(redis_host: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for key in redis_raw_lines(redis_host, "KEYS", "cluster:job:*"):
        job_id = key.split("cluster:job:", 1)[-1]
        payload = redis_hash(redis_host, key)
        payload["job_id"] = job_id
        payload["key"] = key
        jobs.append(payload)
    jobs.sort(key=lambda item: str(item.get("started", "")), reverse=True)
    return jobs


def index_by_host(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        host = str(row.get(key_name, "") or "")
        if host:
            out.setdefault(host, row)
    return out


def build_cluster_snapshot(*, cluster_root: Path, redis_host: str) -> dict[str, Any]:
    status_payload = load_status_json(cluster_root)
    locks = status_payload.get("locks", [])
    if not isinstance(locks, list):
        locks = []
    hosts = status_payload.get("hosts", [])
    if not isinstance(hosts, list):
        hosts = []
    jobs = collect_jobs(redis_host)

    lock_by_host = index_by_host([row for row in locks if isinstance(row, dict)], "host")
    jobs_by_host = index_by_host([row for row in jobs if isinstance(row, dict)], "host")
    jobs_by_id = {
        str(row.get("job_id", "") or ""): row
        for row in jobs
        if isinstance(row, dict) and row.get("job_id")
    }
    enriched_hosts: list[dict[str, Any]] = []
    for raw in hosts:
        if not isinstance(raw, dict):
            continue
        host = str(raw.get("host", "") or "")
        row = dict(raw)
        lock_row = lock_by_host.get(host, {})
        lock_owner = str(lock_row.get("owner", "") or row.get("lock_owner", "") or "")
        job_row = jobs_by_id.get(lock_owner, {}) if lock_owner else {}
        if not job_row and not lock_owner:
            job_row = jobs_by_host.get(host, {})
        row.setdefault("lock_owner", lock_row.get("owner", ""))
        row.setdefault("lock_ttl", lock_row.get("ttl"))
        row["active_job_id"] = str(job_row.get("job_id", "") or "")
        row["active_job_status"] = str(job_row.get("status", "") or "")
        row["active_script"] = str(job_row.get("script", "") or "")
        row["active_remote_dir"] = str(job_row.get("remote_dir", "") or "")
        enriched_hosts.append(row)

    return {
        "generated_at_utc": now_utc(),
        "cluster_root": cluster_root.expanduser().resolve().as_posix(),
        "redis_host": redis_host,
        "status_source": status_payload.get("source", "unknown"),
        "hosts": enriched_hosts,
        "locks": locks,
        "jobs": jobs,
    }


def summarize_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, "") or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    hosts = [row for row in payload.get("hosts", []) if isinstance(row, dict)]
    jobs = [row for row in payload.get("jobs", []) if isinstance(row, dict)]
    locks = [row for row in payload.get("locks", []) if isinstance(row, dict)]
    lines: list[str] = []
    lines.append("# Cluster Queue Summary")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload.get('generated_at_utc','')}`")
    lines.append(f"- cluster_root: `{payload.get('cluster_root','')}`")
    lines.append(f"- redis_host: `{payload.get('redis_host','')}`")
    lines.append(f"- status_source: `{payload.get('status_source','')}`")
    lines.append(f"- host_count: `{len(hosts)}`")
    lines.append(f"- lock_count: `{len(locks)}`")
    lines.append(f"- job_count: `{len(jobs)}`")
    lines.append("")
    lines.append("Host statuses:")
    for status, count in summarize_counts(hosts, "status").items():
        lines.append(f"- `{status or 'unknown'}`: `{count}`")
    lines.append("")
    lines.append("Job statuses:")
    for status, count in summarize_counts(jobs, "status").items():
        lines.append(f"- `{status or 'unknown'}`: `{count}`")
    lines.append("")
    lines.append("Locked hosts:")
    for row in hosts:
        owner = str(row.get("lock_owner", "") or "")
        if not owner:
            continue
        lines.append(
            f"- `{row.get('host','')}`: owner=`{owner}` ttl=`{row.get('lock_ttl','')}` "
            f"job=`{row.get('active_job_id','')}` status=`{row.get('active_job_status','')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_snapshot(output_path: Path, summary_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(summary_path, payload)


def main() -> None:
    args = parse_args()
    cluster_root = Path(args.cluster_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    summary_output = Path(args.summary_output).expanduser().resolve()
    payload = build_cluster_snapshot(cluster_root=cluster_root, redis_host=args.redis_host)
    write_snapshot(output, summary_output, payload)
    if args.stdout_summary:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
