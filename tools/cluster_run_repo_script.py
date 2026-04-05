#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: cluster_run_repo_script.py <repo_script.py> [args...]", file=sys.stderr)
        return 2

    repo_script = sys.argv[1]
    job_cwd = Path.cwd()
    repo_root = Path.home() / "transformer_research" / "parameter-golf"
    target = repo_root / repo_script
    if not target.is_file():
        print(f"Repo script not found: {target}", file=sys.stderr)
        return 2

    forwarded_args: list[str] = []
    for arg in sys.argv[2:]:
        candidate = job_cwd / arg
        if candidate.is_file():
            forwarded_args.append(str(candidate))
            continue
        if "=" in arg:
            key, value = arg.split("=", 1)
            candidate = job_cwd / value
            if candidate.is_file():
                forwarded_args.append(f"{key}={candidate}")
                continue
        forwarded_args.append(arg)

    cmd = [sys.executable, str(target), *forwarded_args]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
