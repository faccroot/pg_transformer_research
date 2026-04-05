#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISPATCH = Path.home() / "cluster" / "dispatch.sh"
JOB_RE = re.compile(r"^\[(job_\d+_\d+)\]\s+Claimed\s+(mini\d{2})\b", re.MULTILINE)


TOOL_SPECS = {
    "controller": {
        "script": ROOT / "tools" / "analyze_hardmax_structural_controller.py",
        "output_flag": "--result-json",
        "default_remote_output": "hardmax_controller_summary.json",
    },
    "causal": {
        "script": ROOT / "tools" / "eval_saved_hardmax_causal_ablation.py",
        "output_flag": "--result-json",
        "default_remote_output": "causal_ablation.json",
    },
    "factors": {
        "script": ROOT / "tools" / "analyze_saved_logit_factors.py",
        "output_flag": "--summary-json",
        "default_remote_output": "logit_factors_summary.json",
    },
    "residual": {
        "script": ROOT / "tools" / "analyze_residual_autocorrelation.py",
        "output_flag": "--result-json",
        "default_remote_output": "residual_autocorrelation.json",
    },
    "face": {
        "script": ROOT / "tools" / "export_hardmax_face_trace.py",
        "output_flag": "--output-json",
        "default_remote_output": "face_trace.json",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dispatch a saved-artifact hardmax analysis tool to a Mini and copy the JSON result back."
    )
    p.add_argument("mode", choices=sorted(TOOL_SPECS), help="Which analysis tool to run remotely.")
    p.add_argument("--artifact", required=True, help="Saved artifact path.")
    p.add_argument("--config-json", required=True, help="Config JSON path.")
    p.add_argument("--output-json", required=True, help="Local destination for the copied JSON result.")
    p.add_argument("--host", default="", help="Optional target Mini alias.")
    p.add_argument("--dispatch", default=str(DEFAULT_DISPATCH), help="dispatch.sh path.")
    p.add_argument("--tokenizer-path", default="", help="Optional tokenizer path to stage and pass through.")
    p.add_argument("--repo-bundle", default="", help="Optional existing repo bundle tar/tgz. If omitted, one is built.")
    p.add_argument("--keep-bundle", action="store_true", help="Do not delete the auto-built repo bundle.")
    p.add_argument("--dry-run", action="store_true", help="Print the dispatch command and planned local output path.")
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument to forward to the remote tool. Can be repeated. Use `--extra-arg=--flag` for forwarded flags.",
    )
    return p.parse_args()


def build_repo_bundle(bundle_path: Path) -> None:
    include_patterns = [
        "*.py",
        "tools/*.py",
    ]
    files: list[Path] = []
    for pattern in include_patterns:
        files.extend(ROOT.glob(pattern))
    files = sorted({path.resolve() for path in files if path.is_file()})
    if not files:
        raise ValueError(f"No files selected for repo bundle under {ROOT}")
    with tarfile.open(bundle_path, "w:gz") as tf:
        for path in files:
            arcname = Path("parameter-golf") / path.relative_to(ROOT)
            tf.add(path, arcname=str(arcname))


def dispatch_and_capture(cmd: list[str]) -> tuple[str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = ((proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise RuntimeError(f"dispatch failed with code {proc.returncode}\n{combined}")
    match = JOB_RE.search(combined)
    if not match:
        raise RuntimeError(f"Could not parse job id / host from dispatch output:\n{combined}")
    return match.group(1), match.group(2)


def scp_copy(remote_spec: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", remote_spec, str(dest_path)], check=True)


def main() -> None:
    args = parse_args()
    spec = TOOL_SPECS[args.mode]
    dispatch_path = Path(args.dispatch).expanduser().resolve()
    if not dispatch_path.is_file():
        raise FileNotFoundError(f"Dispatch script not found: {dispatch_path}")

    artifact_path = Path(args.artifact).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else None
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")
    if tokenizer_path is not None and not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="remote_hardmax_analysis_"))
    bundle_owned = False
    try:
        if args.repo_bundle:
            bundle_path = Path(args.repo_bundle).expanduser().resolve()
            if not bundle_path.is_file():
                raise FileNotFoundError(f"Repo bundle not found: {bundle_path}")
        else:
            bundle_path = temp_dir / "parameter-golf-current.tgz"
            build_repo_bundle(bundle_path)
            bundle_owned = True

        remote_output_name = spec["default_remote_output"]
        cmd = [str(dispatch_path)]
        if args.host:
            cmd.extend(["--host", args.host])
        cmd.extend(
            [
                str(spec["script"]),
                "--artifact",
                str(artifact_path),
                "--config-json",
                str(config_path),
                "--repo-bundle",
                str(bundle_path),
                spec["output_flag"],
                remote_output_name,
            ]
        )
        if tokenizer_path is not None:
            cmd.extend(["--tokenizer-path", str(tokenizer_path)])
        cmd.extend(args.extra_arg)

        if args.dry_run:
            print("DRY-RUN", " ".join(cmd))
            print(f"DRY-RUN local_output_json={output_json}")
            return

        job_id, host = dispatch_and_capture(cmd)
        remote_result = f"{host}:~/jobs/{job_id}/{remote_output_name}"
        scp_copy(remote_result, output_json)
        print(
            f"remote_analysis_ok mode={args.mode} host={host} job_id={job_id} "
            f"output_json={output_json}"
        )
    finally:
        if not args.keep_bundle and bundle_owned:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
