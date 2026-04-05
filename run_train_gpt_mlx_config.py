#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def resolve_trainer_script(default_script: Path, config_path: Path) -> Path:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_script
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return default_script

    candidates: list[Path] = []
    for key in ("trainer_script", "trainer_script_source"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(Path(value.strip()))

    stage_root = config_path.parent.parent
    for candidate in candidates:
        if candidate.is_absolute():
            staged_name = (stage_root / candidate.name).resolve()
            if staged_name.exists():
                return staged_name
            if candidate.exists():
                return candidate.resolve()
            continue
        for root in (stage_root, config_path.parent, default_script.parent):
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
    return default_script


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a staged trainer copy with a staged JSON config."
    )
    parser.add_argument("script", help="Path to the staged train_gpt_mlx.py")
    parser.add_argument("config", help="Path to the staged config JSON")
    parser.add_argument(
        "support_files",
        nargs="*",
        help="Additional staged support files. These are ignored at runtime and exist only so dispatch.sh copies them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    script_path = resolve_trainer_script(Path(args.script).resolve(), config_path)
    os.chdir(script_path.parent)
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), "--config", str(config_path)]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
