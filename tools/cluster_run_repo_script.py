#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: cluster_run_repo_script.py <repo_script.py> [args...]", file=sys.stderr)
        return 2

    filtered_argv: list[str] = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--copy-mod":
            if i + 1 >= len(sys.argv):
                print("--copy-mod requires a path", file=sys.stderr)
                return 2
            i += 2
            continue
        filtered_argv.append(arg)
        i += 1

    if not filtered_argv:
        print("Usage: cluster_run_repo_script.py <repo_script.py> [args...]", file=sys.stderr)
        return 2

    repo_script = filtered_argv[0]
    job_cwd = Path.cwd()
    repo_root = Path.home() / "transformer_research" / "parameter-golf"
    repo_target = repo_root / repo_script
    copied_target = job_cwd / Path(repo_script).name
    if repo_target.is_file():
        target = repo_target
    elif copied_target.is_file():
        target = copied_target
    else:
        print(f"Repo script not found in repo or job dir: {repo_target} / {copied_target}", file=sys.stderr)
        return 2

    forwarded_args: list[str] = []
    for arg in filtered_argv[1:]:
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

    env = os.environ.copy()
    py_entries = [str(job_cwd), str(repo_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        py_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_entries)

    cmd = [sys.executable, str(target), *forwarded_args]
    return subprocess.call(cmd, cwd=repo_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
