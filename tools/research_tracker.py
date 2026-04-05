#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"
ITERATIONS_DIR = RESEARCH_DIR / "iterations"
ARCHIVE_DIR = ITERATIONS_DIR / "archive"
ITERATION_INDEX = ITERATIONS_DIR / "iteration_index.jsonl"
RUN_INDEX = ITERATIONS_DIR / "run_index.jsonl"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "iteration"


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def ensure_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    ensure_jsonl(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_metrics(values: list[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metric must look like key=value, got {value!r}")
        key, metric_value = value.split("=", 1)
        key = key.strip()
        metric_value = metric_value.strip()
        if not key:
            raise ValueError(f"metric key is empty in {value!r}")
        metrics[key] = metric_value
    return metrics


def iteration_note_content(entry: dict) -> str:
    tags = ", ".join(entry["tags"]) if entry["tags"] else "none"
    parent = entry["parent_iteration_id"] or "none"
    return f"""# {entry["iteration_id"]}

## Summary
- Slug: `{entry["slug"]}`
- Status: `{entry["status"]}`
- Component: `{entry["component"]}`
- Script: `{entry["script"]}`
- Parent iteration: `{parent}`
- Tags: `{tags}`
- Base branch: `{entry["base_branch"] or "unknown"}`
- Base commit: `{entry["base_commit"] or "unknown"}`
- Created at (UTC): `{entry["created_at_utc"]}`

## Hypothesis
{entry["hypothesis"]}

## Planned changes
- Fill in the concrete code changes before running the experiment.

## Expected signal
- State the metric movement or failure mode this iteration is testing.

## Actual outcome
- Fill in after runs complete.

## Follow-up
- Decide whether to extend, revert, or branch this line of work.
"""


def create_iteration(args: argparse.Namespace) -> None:
    slug = slugify(args.slug)
    timestamp = datetime.now(timezone.utc)
    iteration_id = f"iter_{timestamp.strftime('%Y%m%d_%H%M%S')}_{slug}"
    note_dir = ARCHIVE_DIR / timestamp.strftime("%Y") / iteration_id
    note_dir.mkdir(parents=True, exist_ok=True)
    note_path = note_dir / "README.md"

    entry = {
        "base_branch": git_value("branch", "--show-current"),
        "base_commit": git_value("rev-parse", "HEAD"),
        "component": args.component,
        "created_at_utc": now_utc(),
        "hypothesis": args.hypothesis.strip(),
        "iteration_id": iteration_id,
        "note_path": note_path.relative_to(ROOT).as_posix(),
        "parent_iteration_id": args.parent_iteration_id,
        "script": args.script,
        "slug": slug,
        "status": args.status,
        "tags": sorted(set(args.tag)),
    }
    note_path.write_text(iteration_note_content(entry), encoding="utf-8")
    append_jsonl(ITERATION_INDEX, entry)
    print(json.dumps(entry, indent=2, sort_keys=True))


def log_run(args: argparse.Namespace) -> None:
    entry = {
        "artifact_path": args.artifact_path,
        "branch": git_value("branch", "--show-current"),
        "command": args.command,
        "commit": git_value("rev-parse", "HEAD"),
        "created_at_utc": now_utc(),
        "dataset": args.dataset,
        "host": args.host,
        "iteration_id": args.iteration_id,
        "log_path": args.log_path,
        "metrics": parse_metrics(args.metric),
        "notes": args.notes,
        "run_id": args.run_id,
        "script": args.script,
        "status": args.status,
        "tags": sorted(set(args.tag)),
        "train_shards": args.train_shards,
    }
    append_jsonl(RUN_INDEX, entry)
    print(json.dumps(entry, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track parameter-golf iterations and runs in JSONL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_iteration = subparsers.add_parser("new-iteration", help="Create an iteration entry and note folder.")
    new_iteration.add_argument("--slug", required=True, help="Short human-readable label for the idea.")
    new_iteration.add_argument("--component", required=True, help="Primary subsystem under test.")
    new_iteration.add_argument("--hypothesis", required=True, help="What you expect to learn or improve.")
    new_iteration.add_argument("--script", default="train_gpt_mlx.py", help="Primary script for the iteration.")
    new_iteration.add_argument("--status", default="planned", help="Initial status, for example planned or running.")
    new_iteration.add_argument("--parent-iteration-id", help="Optional parent iteration ID.")
    new_iteration.add_argument("--tag", action="append", default=[], help="Optional tag, repeatable.")
    new_iteration.set_defaults(func=create_iteration)

    run = subparsers.add_parser("log-run", help="Append a run entry.")
    run.add_argument("--iteration-id", required=True, help="Iteration ID from iteration_index.jsonl.")
    run.add_argument("--run-id", required=True, help="External or script RUN_ID.")
    run.add_argument("--script", default="train_gpt_mlx.py", help="Script used for the run.")
    run.add_argument("--status", default="launched", help="Run status.")
    run.add_argument("--host", help="Machine or cluster node.")
    run.add_argument("--dataset", default="fineweb10B_sp1024", help="Dataset variant.")
    run.add_argument("--train-shards", type=int, help="Number of train shards used.")
    run.add_argument("--command", help="Exact command line if you want it tracked.")
    run.add_argument("--log-path", help="Path to the log file.")
    run.add_argument("--artifact-path", help="Path to the saved artifact.")
    run.add_argument("--notes", help="Free-form short note.")
    run.add_argument("--metric", action="append", default=[], help="Metric in key=value form, repeatable.")
    run.add_argument("--tag", action="append", default=[], help="Optional tag, repeatable.")
    run.set_defaults(func=log_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

