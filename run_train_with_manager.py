#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
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
    for key in ("trainer_script", "trainer_script_source", "script", "script_source"):
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


def load_config_env(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", payload)
    if not isinstance(env_payload, dict):
        raise SystemExit(f"Config env payload must be a JSON object: {config_path}")
    env: dict[str, str] = {}
    for key, value in env_payload.items():
        if value is None:
            continue
        env[str(key)] = str(value)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trainer alongside optional helper workers.")
    parser.add_argument("trainer_script", nargs="?", help="Path to train_gpt.py or train_gpt_mlx.py")
    parser.add_argument("config", nargs="?", help="Path to JSON config with env payload")
    parser.add_argument(
        "support_files",
        nargs="*",
        help="Additional staged support files. Ignored at runtime; present so dispatch copies them.",
    )
    parser.add_argument("--trainer-script", dest="trainer_script_flag", help="Path to train_gpt.py or train_gpt_mlx.py")
    parser.add_argument("--config", dest="config_flag", help="Path to JSON config with env payload")
    parser.add_argument(
        "--manager-script",
        default=str(Path(__file__).with_name("student_manager_worker.py")),
        help="Path to the manager worker script",
    )
    parser.add_argument(
        "--manager-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval for the manager",
    )
    parser.add_argument(
        "--teacher-hidden-worker-script",
        default=str(Path(__file__).with_name("teacher_hidden_cache_worker.py")),
        help="Path to the teacher hidden cache worker script",
    )
    parser.add_argument(
        "--teacher-hidden-worker-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval for the teacher hidden cache worker",
    )
    args = parser.parse_args()

    trainer_arg = args.trainer_script_flag or args.trainer_script
    config_arg = args.config_flag or args.config
    if not trainer_arg or not config_arg:
        raise SystemExit("Provide trainer_script and config")

    config_path = Path(config_arg).resolve()
    trainer_script = resolve_trainer_script(Path(trainer_arg).resolve(), config_path)
    manager_script = Path(args.manager_script).resolve()
    env = os.environ.copy()
    env.update(load_config_env(config_path))

    run_id = env.get("RUN_ID", str(uuid.uuid4()))
    snapshot_bus_dir = env.get("STUDENT_SNAPSHOT_DIR", "").strip()
    if not snapshot_bus_dir:
        raise SystemExit("Config must set STUDENT_SNAPSHOT_DIR for manager mode")

    start_manager = bool(int(env.get("START_MANAGER", "1")))
    start_teacher_hidden_worker = bool(int(env.get("START_TEACHER_HIDDEN_WORKER", "0")))
    env["RUN_ID"] = run_id
    env["EXTERNAL_CONTROLLER_ENABLED"] = "1" if start_manager else "0"
    env.setdefault("EXTERNAL_CONTROLLER_REFRESH_EVERY", "10")

    manager_env = env.copy()
    manager_env["RUN_ID"] = run_id
    manager_env["SNAPSHOT_BUS_DIR"] = snapshot_bus_dir
    manager_env["MANAGER_POLL_SECONDS"] = str(max(args.manager_poll_seconds, 0.1))
    teacher_hidden_worker_script = Path(args.teacher_hidden_worker_script).resolve()
    teacher_hidden_env = env.copy()
    teacher_hidden_env["RUN_ID"] = run_id
    teacher_hidden_env["SNAPSHOT_BUS_DIR"] = snapshot_bus_dir
    teacher_hidden_env["TEACHER_HIDDEN_WORKER_POLL_SECONDS"] = str(max(args.teacher_hidden_worker_poll_seconds, 0.1))
    teacher_hidden_env.setdefault("HELPER_NAME", "teacher_hidden_worker")

    trainer_cmd = [sys.executable, str(trainer_script), "--config", str(config_path)]
    manager_cmd = [sys.executable, str(manager_script)]
    teacher_hidden_cmd = [sys.executable, str(teacher_hidden_worker_script)]

    manager_proc = None
    teacher_hidden_proc = None
    print(
        f"helper_launch run_id={run_id} trainer={trainer_script.name} "
        f"start_manager={int(start_manager)} start_teacher_hidden_worker={int(start_teacher_hidden_worker)}",
        flush=True,
    )
    if start_manager:
        manager_proc = subprocess.Popen(manager_cmd, env=manager_env, cwd=str(manager_script.parent))
        print(f"helper_launch manager_pid={manager_proc.pid} script={manager_script.name}", flush=True)
    if start_teacher_hidden_worker:
        teacher_hidden_proc = subprocess.Popen(
            teacher_hidden_cmd,
            env=teacher_hidden_env,
            cwd=str(teacher_hidden_worker_script.parent),
        )
        print(
            f"helper_launch teacher_hidden_worker_pid={teacher_hidden_proc.pid} "
            f"script={teacher_hidden_worker_script.name}",
            flush=True,
        )
    try:
        trainer_proc = subprocess.Popen(trainer_cmd, env=env, cwd=str(trainer_script.parent))
        print(f"helper_launch trainer_pid={trainer_proc.pid}", flush=True)
        trainer_rc = trainer_proc.wait()
    finally:
        if teacher_hidden_proc is not None:
            teacher_hidden_proc.terminate()
            try:
                teacher_hidden_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                teacher_hidden_proc.kill()
                teacher_hidden_proc.wait(timeout=5.0)
        if manager_proc is not None:
            manager_proc.terminate()
            try:
                manager_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                manager_proc.kill()
                manager_proc.wait(timeout=5.0)

    if trainer_rc != 0:
        raise SystemExit(trainer_rc)


if __name__ == "__main__":
    main()
