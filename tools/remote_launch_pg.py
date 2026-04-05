#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Remote wrapper for run_mini_pg.py. "
            "Extra positional file args are ignored by this wrapper but force dispatch.sh "
            "to stage the files into the remote job directory."
        )
    )
    p.add_argument("--run-script", required=True, help="Basename/path of copied run_mini_pg_job.py")
    p.add_argument(
        "--support-files",
        nargs="*",
        default=[],
        help="Ignored by this wrapper; used only to force dispatch.sh to stage files into the remote job dir.",
    )
    p.add_argument("run_args", nargs=argparse.REMAINDER, help="Arguments forwarded to run_mini_pg_job.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_script = Path(args.run_script)
    if not run_script.exists():
        raise FileNotFoundError(f"run script not found in remote job dir: {run_script}")
    forwarded = list(args.run_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd = [sys.executable, str(run_script), *forwarded]
    subprocess.run(cmd, cwd=Path.cwd(), check=True)


if __name__ == "__main__":
    main()
