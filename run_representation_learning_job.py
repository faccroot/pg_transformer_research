#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "numpy",
    "sentencepiece",
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
)


def missing_modules() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def ensure_requirements(requirements_path: Path) -> None:
    missing = missing_modules()
    if not missing:
        print("representation_learning_job: requirements already satisfied", flush=True)
        return
    print(
        "representation_learning_job: installing missing modules "
        + ",".join(missing)
        + f" from {requirements_path}",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install extraction deps if needed, then run a staged representation-learning pipeline script."
    )
    parser.add_argument("script", help="Path to the staged Python entrypoint")
    parser.add_argument("args_json", help="Path to a staged JSON file containing argv items")
    parser.add_argument(
        "--requirements",
        default="requirements-representation-learning.txt",
        help="Staged requirements file used to install missing deps",
    )
    args, support_files = parser.parse_known_args()

    script_path = Path(args.script).resolve()
    args_path = Path(args.args_json).resolve()
    requirements_path = Path(args.requirements).resolve()
    argv_payload = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(argv_payload, list) or not all(isinstance(item, str) for item in argv_payload):
        raise SystemExit(f"Args payload must be a JSON string list: {args_path}")

    ensure_requirements(requirements_path)

    os.chdir(script_path.parent)
    sys.path.insert(0, str(script_path.parent))
    if support_files:
        print(f"representation_learning_job: staged_support_files={len(support_files)}", flush=True)
    sys.argv = [str(script_path), *argv_payload]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
