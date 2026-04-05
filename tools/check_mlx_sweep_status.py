#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import re
import subprocess
from pathlib import Path


DISPATCH_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+(?:serial_)?dispatch config=(?P<config>\S+)\s+host=(?P<host>\S+)(?:\s+attempt=(?P<attempt>\d+)/(?P<total>\d+)|\s+run_slug=\S+\s+index=(?P<serial_idx>\d+)/(?P<serial_total>\d+))"
)
JOB_RE = re.compile(r"^\[(?P<job>job_[^\]]+)\]\s+(?P<msg>.*)$")
CLAIMED_RE = re.compile(r"^Claimed (?P<host>\S+)")
RUNNING_RE = re.compile(r"^Running on (?P<host>\S+)\.\.\.")
RELEASED_RE = re.compile(r"^Released (?P<host>\S+)")
FAILED_RE = re.compile(r"^dispatch_failed config=(?P<config>\S+)\s+attempts=(?P<attempts>\d+)")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def remote_config_is_running(host: str, config_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ssh", host, f"pgrep -af {config_name!r} >/dev/null && echo 1 || echo 0"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return False
    return result.stdout.strip() == "1"


def summarize_sweep(iteration_dir: Path, *, check_remote: bool) -> list[dict[str, object]]:
    manifest_path = iteration_dir / "manifest.json"
    dispatch_path = iteration_dir / "dispatch.out"
    if not dispatch_path.exists():
        launch_log = iteration_dir / "launch.nohup.log"
        if launch_log.exists():
            dispatch_path = launch_log
    manifest = load_json(manifest_path)
    runs = manifest.get("runs", [])
    summary: dict[str, dict[str, object]] = {}
    for run in runs:
        config_name = Path(str(run["config_path"])).name
        summary[config_name] = {
            "config": config_name,
            "run_slug": run.get("run_slug"),
            "run_id": run.get("run_id"),
            "attempt": 0,
            "attempt_total": 0,
            "job_id": "",
            "host": "",
            "status": "prepared",
        }

    job_to_config: dict[str, str] = {}
    pending_configs: deque[str] = deque()
    if dispatch_path.exists():
        for raw_line in dispatch_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = DISPATCH_RE.match(line)
            if m:
                config_name = m.group("config")
                pending_configs.append(config_name)
                row = summary.setdefault(config_name, {"config": config_name})
                attempt = m.group("attempt") or m.group("serial_idx") or "1"
                total = m.group("total") or m.group("serial_total") or "1"
                row["attempt"] = int(attempt)
                row["attempt_total"] = int(total)
                row["status"] = "dispatching"
                continue
            m = JOB_RE.match(line)
            if m:
                job_id = m.group("job")
                if job_id not in job_to_config and pending_configs:
                    job_to_config[job_id] = pending_configs.popleft()
                if job_id not in job_to_config:
                    continue
                config_name = job_to_config[job_id]
                row = summary.setdefault(config_name, {"config": config_name})
                row["job_id"] = job_id
                msg = m.group("msg")
                claimed = CLAIMED_RE.match(msg)
                if claimed:
                    row["host"] = claimed.group("host")
                    row["status"] = "claimed"
                    continue
                running = RUNNING_RE.match(msg)
                if running:
                    row["host"] = running.group("host")
                    row["status"] = "running"
                    continue
                released = RELEASED_RE.match(msg)
                if released:
                    row["host"] = released.group("host")
                    row["status"] = "released"
                    continue
            m = FAILED_RE.match(line)
            if m:
                config_name = m.group("config")
                row = summary.setdefault(config_name, {"config": config_name})
                row["status"] = "dispatch_failed"
                continue

    results_summary = iteration_dir / "results_summary.md"
    for row in summary.values():
        if results_summary.exists():
            row["status"] = "summarized"
        elif check_remote and row.get("host") and remote_config_is_running(str(row["host"]), str(row["config"])):
            row["status"] = "running"
        elif row.get("status") == "released":
            row["status"] = "stopped"

    rows = list(summary.values())
    rows.sort(key=lambda item: str(item.get("config")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize live/completed MLX sweep status from a generated sweep dir.")
    parser.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json and dispatch.out")
    parser.add_argument("--check-remote", action="store_true", help="SSH to claimed hosts and verify whether each config is still running")
    args = parser.parse_args()

    iteration_dir = Path(args.iteration_dir).resolve()
    rows = summarize_sweep(iteration_dir, check_remote=args.check_remote)
    observed_results_path = iteration_dir / "observed_results.json"
    observed_runs = {}
    if observed_results_path.exists():
        try:
            observed_payload = load_json(observed_results_path)
            raw_runs = observed_payload.get("runs", {})
            if isinstance(raw_runs, dict):
                observed_runs = raw_runs
        except (OSError, json.JSONDecodeError):
            observed_runs = {}
    for row in rows:
        observed_suffix = ""
        for run_slug, observed in observed_runs.items():
            if not isinstance(observed, dict):
                continue
            if run_slug != row.get("run_slug"):
                continue
            final_exact = observed.get("final_int8_zlib_roundtrip_exact")
            if isinstance(final_exact, dict) and "val_bpb" in final_exact:
                observed_suffix = f"\tbpb={final_exact['val_bpb']}"
            break
        print(
            f"{row.get('config')}\t{row.get('status')}\t"
            f"host={row.get('host','')}\tjob={row.get('job_id','')}\t"
            f"attempt={row.get('attempt',0)}/{row.get('attempt_total',0)}\t"
            f"run_id={row.get('run_id','')}{observed_suffix}"
        )


if __name__ == "__main__":
    main()
