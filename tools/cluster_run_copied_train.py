#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--copy-mod", action="append", default=[])
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_dir = Path.cwd()
    target = job_dir / args.script
    config = job_dir / args.config
    if not target.is_file():
        print(f"Missing copied script: {target}", file=sys.stderr)
        return 2
    if not config.is_file():
        print(f"Missing copied config: {config}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(target), "--config", str(config), *args.passthrough]
    return subprocess.call(cmd, cwd=job_dir)


if __name__ == "__main__":
    raise SystemExit(main())
