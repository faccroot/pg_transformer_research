#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFY_PYTHON = ROOT / ".venv-mlx-cpu" / "bin" / "python"
DEFAULT_VERIFY_SCRIPT = ROOT / "tools" / "verify_hardmax_execution_trace.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate a trace-pretrain export checkpoint/config pair and run held-out execution verification."
    )
    parser.add_argument("iteration_dir", help="Generated iteration directory containing configs and artifacts/logs.")
    parser.add_argument("--run-slug", default="", help="Run slug from config metadata.")
    parser.add_argument("--run-id", default="", help="Run id from config args/env.")
    parser.add_argument("--artifact", default="", help="Optional explicit full-model checkpoint path.")
    parser.add_argument("--config-json", default="", help="Optional explicit config JSON path.")
    parser.add_argument("--label", default="", help="Optional label passed through to the verifier.")
    parser.add_argument("--result-json", default="", help="Optional output path; defaults under trace_execution_verification/.")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--rollout-max-sequences", type=int, default=64)
    parser.add_argument("--teacher-forced-modes", default="")
    parser.add_argument("--rollout-modes", default="")
    parser.add_argument("--verify-python", default=str(DEFAULT_VERIFY_PYTHON))
    parser.add_argument("--verify-script", default=str(DEFAULT_VERIFY_SCRIPT))
    return parser.parse_args()


def load_config_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config payload must be a dict, got {type(payload)!r}")
    return payload


def executable_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def discover_config(iteration_dir: Path, run_slug: str, run_id: str) -> Path:
    configs_dir = iteration_dir / "configs"
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Missing configs dir: {configs_dir}")
    matches: list[Path] = []
    for config_path in sorted(configs_dir.glob("*.json")):
        payload = load_config_payload(config_path)
        metadata = payload.get("metadata", {})
        env = payload.get("env", {})
        args = payload.get("args", {})
        if not isinstance(metadata, dict) or not isinstance(env, dict) or not isinstance(args, dict):
            continue
        meta_slug = str(metadata.get("run_slug", ""))
        cfg_run_id = str(args.get("run-id", "") or env.get("RUN_ID", ""))
        if run_slug and meta_slug != run_slug:
            continue
        if run_id and cfg_run_id != run_id:
            continue
        matches.append(config_path)
    if not matches:
        raise FileNotFoundError(
            f"No config matched run_slug={run_slug!r} run_id={run_id!r} under {configs_dir}"
        )
    if len(matches) > 1 and not run_slug and not run_id:
        raise RuntimeError(
            f"Multiple configs found under {configs_dir}; specify --run-slug or --run-id. "
            f"Matches: {[str(p.name) for p in matches]}"
        )
    return matches[0]


def artifact_candidates(iteration_dir: Path, run_slug: str, run_id: str) -> list[Path]:
    roots = [iteration_dir / "artifacts", iteration_dir / "logs", iteration_dir]
    patterns = []
    if run_id:
        patterns.extend(
            [
                f"*{run_id}*_trace_pretrain_model.npz",
                f"*{run_id}*model*.npz",
            ]
        )
    if run_slug:
        patterns.extend(
            [
                f"*{run_slug}*_trace_pretrain_model.npz",
                f"*{run_slug}*model*.npz",
            ]
        )
    patterns.extend(["*_trace_pretrain_model.npz"])
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path not in seen:
                    found.append(path)
                    seen.add(path)
    return found


def discover_artifact(iteration_dir: Path, config_path: Path, run_slug: str, run_id: str) -> Path:
    candidates = artifact_candidates(iteration_dir, run_slug, run_id)
    if not candidates:
        raise FileNotFoundError(
            f"No full trace-pretrain model checkpoint found under {iteration_dir}. "
            "Expected a *_trace_pretrain_model.npz artifact from the patched exporter."
        )
    if len(candidates) > 1:
        exact = [path for path in candidates if run_id and run_id in path.name]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(
            f"Multiple artifacts found for {config_path.name}; pass --artifact explicitly. "
            f"Matches: {[path.name for path in candidates]}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve() if args.config_json else discover_config(
        iteration_dir,
        str(args.run_slug),
        str(args.run_id),
    )
    payload = load_config_payload(config_path)
    metadata = payload.get("metadata", {})
    env = payload.get("env", {})
    cfg_args = payload.get("args", {})
    meta_slug = str(metadata.get("run_slug", "")) if isinstance(metadata, dict) else ""
    cfg_run_id = (
        str(cfg_args.get("run-id", "") or env.get("RUN_ID", ""))
        if isinstance(cfg_args, dict) and isinstance(env, dict)
        else ""
    )
    artifact_path = Path(args.artifact).expanduser().resolve() if args.artifact else discover_artifact(
        iteration_dir,
        config_path,
        meta_slug or str(args.run_slug),
        cfg_run_id or str(args.run_id),
    )
    verify_python = executable_path(str(args.verify_python))
    verify_script = Path(args.verify_script).expanduser().resolve()
    label = str(args.label or meta_slug or artifact_path.stem)
    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else iteration_dir / "trace_execution_verification" / f"{(meta_slug or artifact_path.stem)}.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(verify_python),
        str(verify_script),
        "--config",
        str(config_path),
        "--artifact",
        str(artifact_path),
        "--result-json",
        str(result_path),
        "--label",
        label,
        "--split",
        str(args.split),
        "--max-sequences",
        str(int(args.max_sequences)),
        "--rollout-max-sequences",
        str(int(args.rollout_max_sequences)),
    ]
    if args.teacher_forced_modes:
        cmd.extend(["--teacher-forced-modes", str(args.teacher_forced_modes)])
    if args.rollout_modes:
        cmd.extend(["--rollout-modes", str(args.rollout_modes)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
