#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a staged Python script with argv loaded from a staged JSON file."
    )
    parser.add_argument("script", help="Path to the staged Python entrypoint")
    parser.add_argument("args_json", help="Path to a staged JSON file containing argv items")
    parser.add_argument(
        "support_files",
        nargs="*",
        help="Additional staged support files. Ignored at runtime; present only so dispatch.sh copies them.",
    )
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    args_path = Path(args.args_json).resolve()
    argv_payload = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(argv_payload, list) or not all(isinstance(item, str) for item in argv_payload):
        raise SystemExit(f"Args payload must be a JSON string list: {args_path}")

    os.chdir(script_path.parent)
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *argv_payload]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
