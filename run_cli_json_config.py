#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def resolve_script(default_script: Path, config_path: Path) -> Path:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_script
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return default_script

    candidates: list[Path] = []
    for key in ("script_source", "trainer_script_source", "script", "trainer_script"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(Path(value.strip()))

    stage_root = config_path.parent.parent
    for candidate in candidates:
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            staged_name = (stage_root / candidate.name).resolve()
            if staged_name.exists():
                return staged_name
            continue
        for root in (stage_root, config_path.parent, default_script.parent):
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
    return default_script


def append_cli_args(argv: list[str], key: str, value) -> None:
    flag = f"--{key.replace('_', '-')}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, list):
        for item in value:
            append_cli_args(argv, key, item)
        return
    argv.extend([flag, str(value)])


def append_positionals(argv: list[str], values) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise SystemExit("config.positionals must be a JSON array when provided")
    for value in values:
        if value is None:
            raise SystemExit("config.positionals entries must not be null")
        if isinstance(value, list):
            raise SystemExit("config.positionals entries must be scalars, not arrays")
        argv.append(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a staged CLI script from a staged JSON config.")
    parser.add_argument("script", help="Path to the staged Python script")
    parser.add_argument("config", help="Path to the staged config JSON")
    parser.add_argument(
        "support_files",
        nargs="*",
        help="Additional staged support files. Ignored at runtime; only present so dispatch copies them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    script_path = resolve_script(Path(args.script).resolve(), config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    env = payload.get("env", {})
    cli_args = payload.get("args", {})
    positionals = payload.get("positionals", [])
    if not isinstance(env, dict):
        raise SystemExit("config.env must be a JSON object")
    if not isinstance(cli_args, dict):
        raise SystemExit("config.args must be a JSON object")

    for key, value in env.items():
        os.environ[str(key)] = str(value)

    argv = [str(script_path)]
    append_positionals(argv, positionals)
    for key, value in cli_args.items():
        append_cli_args(argv, str(key), value)

    os.chdir(script_path.parent)
    sys.path.insert(0, str(script_path.parent))
    sys.argv = argv
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
