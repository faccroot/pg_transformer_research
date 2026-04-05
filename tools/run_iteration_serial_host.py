#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from run_iteration_saved_diagnostics import build_run_entries, load_json
except ModuleNotFoundError:
    from tools.run_iteration_saved_diagnostics import build_run_entries, load_json


ROOT = Path(__file__).resolve().parents[1]
RECOVER_TOOL = ROOT / "tools" / "recover_iteration_cluster_artifacts.py"
OBSERVED_TOOL = ROOT / "tools" / "update_iteration_observed_results.py"
RUNTIME_TOOL = ROOT / "tools" / "update_iteration_runtime_index.py"
BRANCH_MEMORY_TOOL = ROOT / "tools" / "branch_memoryctl.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a generated MLX sweep serially on one Mini host, with recovery and observed-results refresh."
    )
    p.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json and configs/")
    p.add_argument("--host", required=True, help="Mini host to run on, e.g. mini09")
    p.add_argument(
        "--skip-observed-finals",
        action="store_true",
        help="Skip runs already marked observed_final in observed_results.json",
    )
    p.add_argument(
        "--run-slug",
        action="append",
        default=[],
        help="Restrict execution to specific run_slug values. Can be passed multiple times.",
    )
    p.add_argument(
        "--start-at-slug",
        default="",
        help="Start execution from this run_slug onward in manifest order.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue dispatching later runs even if an earlier run fails.",
    )
    p.add_argument(
        "--recover-skip-existing",
        action="store_true",
        help="Pass --skip-existing to artifact recovery after each run.",
    )
    p.add_argument(
        "--stdout-summary",
        action="store_true",
        help="Print a compact JSON summary after completion.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return p.parse_args()


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_dispatch_line(dispatch_path: Path, line: str) -> None:
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    with dispatch_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def load_observed_results(iteration_dir: Path) -> dict[str, object]:
    observed_path = iteration_dir / "observed_results.json"
    if not observed_path.exists():
        return {}
    try:
        return json.loads(observed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def observed_run_status(observed: dict[str, object], run_slug: str) -> str:
    runs = observed.get("runs")
    if not isinstance(runs, dict):
        return ""
    payload = runs.get(run_slug)
    if not isinstance(payload, dict):
        return ""
    status = payload.get("status")
    return status if isinstance(status, str) else ""


def select_entries(
    entries,
    *,
    observed: dict[str, object],
    run_slugs: set[str],
    start_at_slug: str,
    skip_observed_finals: bool,
):
    selected = []
    started = not start_at_slug
    for entry in entries:
        if not started:
            if entry.run_slug == start_at_slug:
                started = True
            else:
                continue
        if run_slugs and entry.run_slug not in run_slugs:
            continue
        if skip_observed_finals and observed_run_status(observed, entry.run_slug) == "observed_final":
            continue
        selected.append(entry)
    return selected


def dispatch_command(manifest: dict[str, object], entry, host: str) -> list[str]:
    dispatch_script = str(manifest["dispatch_script"])
    wrapper_script = str(manifest["wrapper_script"])
    script_path = str(manifest.get("script") or (ROOT / "train_gpt_mlx.py"))
    cmd = [
        "bash",
        dispatch_script,
        "--host",
        host,
        wrapper_script,
        script_path,
        str(entry.config_path),
    ]
    cmd.extend(str(item) for item in manifest.get("support_files", []))
    return cmd


def run_subprocess(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> int:
    if dry_run:
        print("DRY-RUN", " ".join(shlex.quote(part) for part in cmd))
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=False)
    return int(proc.returncode)


def post_run_commands(iteration_dir: Path, host: str, *, recover_skip_existing: bool) -> list[list[str]]:
    recover_cmd = [sys.executable, str(RECOVER_TOOL), str(iteration_dir), "--host", host]
    if recover_skip_existing:
        recover_cmd.append("--skip-existing")
    observed_cmd = [sys.executable, str(OBSERVED_TOOL), str(iteration_dir)]
    runtime_cmd = [sys.executable, str(RUNTIME_TOOL), "update", str(iteration_dir)]
    branch_memory_cmd = [sys.executable, str(BRANCH_MEMORY_TOOL), "--direct", "ingest"]
    return [recover_cmd, observed_cmd, runtime_cmd, branch_memory_cmd]


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    manifest = load_json(iteration_dir / "manifest.json")
    entries = build_run_entries(iteration_dir, manifest)
    observed = load_observed_results(iteration_dir)
    dispatch_path = iteration_dir / "dispatch.out"

    requested_slugs = {slug for slug in args.run_slug if slug}
    selected = select_entries(
        entries,
        observed=observed,
        run_slugs=requested_slugs,
        start_at_slug=args.start_at_slug,
        skip_observed_finals=bool(args.skip_observed_finals),
    )
    if not selected:
        summary = {
            "iteration_dir": str(iteration_dir),
            "host": args.host,
            "selected_run_count": 0,
            "status": "nothing_selected",
        }
        if args.stdout_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return

    failures: list[dict[str, object]] = []
    completed: list[str] = []
    for index, entry in enumerate(selected, start=1):
        append_dispatch_line(
            dispatch_path,
            f"[{now_local_iso()}] serial_dispatch config={entry.config_name} host={args.host} "
            f"run_slug={entry.run_slug} index={index}/{len(selected)}",
        )
        cmd = dispatch_command(manifest, entry, args.host)
        rc = run_subprocess(cmd, dry_run=bool(args.dry_run), cwd=iteration_dir)
        append_dispatch_line(
            dispatch_path,
            f"[{now_local_iso()}] serial_dispatch_result config={entry.config_name} host={args.host} "
            f"run_slug={entry.run_slug} rc={rc}",
        )
        if rc != 0:
            failures.append({"run_slug": entry.run_slug, "config": entry.config_name, "returncode": rc})
            if not args.continue_on_error:
                break
        for post_cmd in post_run_commands(iteration_dir, args.host, recover_skip_existing=bool(args.recover_skip_existing)):
            post_rc = run_subprocess(post_cmd, dry_run=bool(args.dry_run), cwd=iteration_dir)
            append_dispatch_line(
                dispatch_path,
                f"[{now_local_iso()}] serial_post_run tool={Path(post_cmd[1]).name} rc={post_rc}",
            )
        if rc == 0:
            completed.append(entry.run_slug)

    if args.stdout_summary:
        print(
            json.dumps(
                {
                    "iteration_dir": str(iteration_dir),
                    "host": args.host,
                    "completed": completed,
                    "failures": failures,
                    "selected_run_count": len(selected),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
