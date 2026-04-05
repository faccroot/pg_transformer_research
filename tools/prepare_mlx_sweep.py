#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ITERATION_INDEX = ROOT / "research" / "iterations" / "iteration_index.jsonl"
RUN_INDEX = ROOT / "research" / "iterations" / "run_index.jsonl"
DEFAULT_SCRIPT = ROOT / "train_gpt_mlx.py"
DEFAULT_DISPATCH = Path.home() / "cluster" / "dispatch_parallel.sh"
DEFAULT_QUEUE_DISPATCH = Path.home() / "cluster" / "dispatch.sh"
DEFAULT_WRAPPER = ROOT / "run_train_gpt_mlx_config.py"
ASSIGNMENT_RE = re.compile(r"^\s+\[(mini\d{2})\]\s+config=(\S+)\s*$")
RESULT_RE = re.compile(r"^\[(mini\d{2})\]\s+(SUCCESS|FAILED)\b")
SWEEP_ID_RE = re.compile(r"^\[([^\]]+)\]\s+Script:")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    chars: list[str] = []
    last_dash = False
    for ch in value.strip().lower():
        if ch.isalnum():
            chars.append(ch)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    slug = "".join(chars).strip("-")
    return slug or "sweep"


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep spec must be a JSON object: {path}")
    return payload


def resolve_path(path_like: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def maybe_resolve_existing_file(path_like: str | Path, *, extra_roots: list[Path] | None = None) -> Path | None:
    candidate = Path(path_like).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate.resolve())
    else:
        roots = list(extra_roots or [])
        roots.extend([ROOT])
        for root in roots:
            candidates.append((root / candidate).resolve())
    for resolved in candidates:
        if resolved.is_file():
            return resolved
    return None


def repo_local_module_path(module_name: str) -> Path | None:
    if not module_name or "." in module_name:
        module_name = module_name.split(".", 1)[0]
    file_candidate = (ROOT / f"{module_name}.py").resolve()
    if file_candidate.is_file():
        return file_candidate
    package_candidate = (ROOT / module_name / "__init__.py").resolve()
    if package_candidate.is_file():
        return package_candidate
    return None


def discover_repo_local_support_files(entrypoints: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen_files: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_files or not resolved.is_file():
            return
        seen_files.add(resolved)
        try:
            source = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        try:
            tree = ast.parse(source, filename=str(resolved))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            module_name: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".", 1)[0]
                    target = repo_local_module_path(module_name)
                    if target is not None and target != resolved:
                        discovered.append(target)
                        visit(target)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module.split(".", 1)[0]
                target = repo_local_module_path(module_name)
                if target is not None and target != resolved:
                    discovered.append(target)
                    visit(target)

    for entry in entrypoints:
        visit(entry)
    return unique_paths(discovered)


def ensure_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    ensure_jsonl(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def find_iteration_note_dir(iteration_id: str | None) -> Path | None:
    if not iteration_id or not ITERATION_INDEX.is_file():
        return None
    for raw_line in ITERATION_INDEX.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("iteration_id") != iteration_id:
            continue
        note_path = entry.get("note_path")
        if isinstance(note_path, str):
            return (ROOT / note_path).resolve().parent
    return None


def merged_run_env(base_env: dict, run_env: dict, default_run_id: str) -> dict:
    env = dict(base_env)
    env.update(run_env)
    env.setdefault("RUN_ID", default_run_id)
    env.setdefault("OUT_DIR", "logs")
    return env


def _stage_queue_value(
    value,
    *,
    spec_path: Path,
    support_files: list[Path],
):
    if isinstance(value, list):
        return [
            _stage_queue_value(item, spec_path=spec_path, support_files=support_files)
            for item in value
        ]
    if not isinstance(value, str):
        return value
    extra_roots = [spec_path.parent, ROOT]
    if "=" in value:
        prefix, maybe_path = value.split("=", 1)
        resolved = maybe_resolve_existing_file(maybe_path, extra_roots=extra_roots)
        if resolved is not None:
            support_files.append(resolved)
            return f"{prefix}={resolved.name}"
    resolved = maybe_resolve_existing_file(value, extra_roots=extra_roots)
    if resolved is not None:
        support_files.append(resolved)
        return resolved.name
    return value


def stage_mapping_file_values_for_queue(
    mapping: dict,
    *,
    spec_path: Path,
    support_files: list[Path],
) -> dict:
    staged = dict(mapping)
    for key, value in list(staged.items()):
        staged[key] = _stage_queue_value(value, spec_path=spec_path, support_files=support_files)
    return staged


def build_output_dir(spec: dict, spec_path: Path, explicit_out_dir: Path | None) -> Path:
    if explicit_out_dir is not None:
        return explicit_out_dir.resolve()
    sweep_slug = slugify(str(spec.get("sweep_slug") or spec_path.stem))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    iteration_dir = find_iteration_note_dir(spec.get("iteration_id"))
    if iteration_dir is not None:
        return iteration_dir / "cluster_sweeps" / f"{stamp}_{sweep_slug}"
    return ROOT / "research" / "iterations" / "generated" / f"{stamp}_{sweep_slug}"


def prepare_sweep(spec: dict, spec_path: Path, out_dir: Path) -> tuple[Path, Path, dict]:
    base_env = spec.get("base_env", {})
    base_args = spec.get("base_args", {})
    runs = spec.get("runs")
    if not isinstance(base_env, dict):
        raise ValueError("base_env must be a JSON object")
    if not isinstance(base_args, dict):
        raise ValueError("base_args must be a JSON object")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty JSON array")

    script_path = resolve_path(spec.get("script") or DEFAULT_SCRIPT)
    if not script_path.is_file():
        raise FileNotFoundError(f"Sweep script not found: {script_path}")

    raw_support_files = spec.get("support_files", [])
    if not isinstance(raw_support_files, list) or not all(isinstance(item, str) for item in raw_support_files):
        raise ValueError("support_files must be a JSON string array when provided")
    support_files = [resolve_path(item) for item in raw_support_files]

    wrapper_script = spec.get("wrapper_script")
    dispatch_mode = str(spec.get("dispatch_mode") or ("queue" if support_files or wrapper_script else "parallel")).strip().lower()
    if dispatch_mode not in {"parallel", "queue"}:
        raise ValueError(f"Unsupported dispatch_mode={dispatch_mode!r}")
    if dispatch_mode == "parallel" and (support_files or wrapper_script):
        raise ValueError("parallel dispatch does not support wrapper_script/support_files; use dispatch_mode='queue'")
    raw_queue_parallelism = spec.get("queue_parallelism", 1)
    if not isinstance(raw_queue_parallelism, int) or raw_queue_parallelism < 1:
        raise ValueError("queue_parallelism must be an integer >= 1 when provided")
    queue_parallelism = raw_queue_parallelism if dispatch_mode == "queue" else 1
    raw_queue_retry_attempts = spec.get("queue_retry_attempts", 8 if queue_parallelism > 1 else 1)
    if not isinstance(raw_queue_retry_attempts, int) or raw_queue_retry_attempts < 1:
        raise ValueError("queue_retry_attempts must be an integer >= 1 when provided")
    queue_retry_attempts = raw_queue_retry_attempts if dispatch_mode == "queue" else 1
    wrapper_default = DEFAULT_WRAPPER if dispatch_mode == "queue" else script_path
    wrapper_path = resolve_path(wrapper_script or wrapper_default)
    if not wrapper_path.is_file():
        raise FileNotFoundError(f"Wrapper script not found: {wrapper_path}")

    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    sweep_slug = slugify(str(spec.get("sweep_slug") or spec_path.stem))
    manifest_runs: list[dict] = []
    run_script_paths: list[Path] = []
    for idx, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Run entry {idx} must be a JSON object")
        run_slug = slugify(str(run.get("slug") or run.get("name") or f"run-{idx:02d}"))
        run_env = run.get("env", {})
        run_args = run.get("args", {})
        if not isinstance(run_env, dict):
            raise ValueError(f"Run {run_slug} env must be a JSON object")
        if not isinstance(run_args, dict):
            raise ValueError(f"Run {run_slug} args must be a JSON object")
        run_script_path = resolve_path(run.get("script") or script_path)
        if not run_script_path.is_file():
            raise FileNotFoundError(f"Run {run_slug} script not found: {run_script_path}")
        run_script_paths.append(run_script_path)
        run_id = str(run.get("run_id") or f"{sweep_slug}_{idx:02d}_{run_slug}")
        merged_env = merged_run_env(base_env, run_env, run_id)
        merged_args = dict(base_args)
        merged_args.update(run_args)
        if dispatch_mode == "queue":
            merged_env = stage_mapping_file_values_for_queue(
                merged_env,
                spec_path=spec_path,
                support_files=support_files,
            )
            merged_args = stage_mapping_file_values_for_queue(
                merged_args,
                spec_path=spec_path,
                support_files=support_files,
            )
        payload = {
            "args": merged_args,
            "env": merged_env,
            "metadata": {
                "created_at_utc": now_utc(),
                "iteration_id": spec.get("iteration_id"),
                "notes": run.get("notes"),
                "run_index": idx,
                "run_slug": run_slug,
                "sweep_slug": sweep_slug,
                "tags": run.get("tags", []),
                "trainer_script": run_script_path.name,
                "trainer_script_source": str(run_script_path),
            },
        }
        config_path = configs_dir / f"{idx:02d}_{run_slug}.json"
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_runs.append(
            {
                "config_path": config_path.relative_to(out_dir).as_posix(),
                "run_id": payload["env"]["RUN_ID"],
                "run_slug": run_slug,
                "notes": run.get("notes"),
                "script": str(run_script_path),
                "tags": run.get("tags", []),
            }
        )

    support_files = unique_paths(support_files + [path for path in run_script_paths if path != script_path])
    if dispatch_mode == "queue":
        support_files = unique_paths(
            support_files
            + discover_repo_local_support_files([script_path, wrapper_path, *support_files, *run_script_paths])
        )

    raw_post_run_diag = spec.get("post_run_hardmax_diagnostics", False)
    if not isinstance(raw_post_run_diag, (bool, dict)):
        raise ValueError("post_run_hardmax_diagnostics must be a boolean or object when provided")
    post_run_diag_cfg = raw_post_run_diag if isinstance(raw_post_run_diag, dict) else {}
    if isinstance(post_run_diag_cfg, dict):
        unknown_diag_keys = set(post_run_diag_cfg) - {
            "output_dir",
            "skip_existing",
            "skip_residual",
            "remote_analyzers",
            "remote_host",
            "remote_dispatch",
            "remote_repo_bundle",
            "remote_keep_bundle",
            "include_causal_ablation",
            "include_logit_factors",
            "val_max_seqs",
            "analysis_max_batches",
            "factor_max_batches",
            "factor_num_factors",
            "factor_top_tokens",
        }
        if unknown_diag_keys:
            raise ValueError(f"Unknown post_run_hardmax_diagnostics keys: {sorted(unknown_diag_keys)}")

    manifest = {
        "config_count": len(manifest_runs),
        "configs_dir": configs_dir.as_posix(),
        "created_at_utc": now_utc(),
        "dispatch_mode": dispatch_mode,
        "dispatch_script": str(
            spec.get("dispatch_script")
            or (DEFAULT_QUEUE_DISPATCH if dispatch_mode == "queue" else DEFAULT_DISPATCH)
        ),
        "iteration_id": spec.get("iteration_id"),
        "script": str(script_path),
        "support_files": [str(path) for path in support_files],
        "source_spec": str(spec_path.resolve()),
        "sweep_slug": sweep_slug,
        "queue_parallelism": queue_parallelism,
        "queue_retry_attempts": queue_retry_attempts,
        "wrapper_script": str(wrapper_path),
        "runs": manifest_runs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    launch_path = out_dir / "launch.sh"
    if dispatch_mode == "parallel":
        launch_command = f"{shlex.quote(str(manifest['dispatch_script']))} {shlex.quote(str(script_path))} {shlex.quote(str(configs_dir))}"
        launch_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + launch_command + "\n",
            encoding="utf-8",
        )
    else:
        wrapper_q = shlex.quote(str(wrapper_path))
        script_q = shlex.quote(str(script_path))
        dispatch_q = shlex.quote(str(manifest["dispatch_script"]))
        recover_tool_q = shlex.quote(str(ROOT / "tools" / "recover_iteration_cluster_artifacts.py"))
        observed_tool_q = shlex.quote(str(ROOT / "tools" / "update_iteration_observed_results.py"))
        iteration_dir_q = shlex.quote(str(out_dir))
        dispatch_log_q = shlex.quote(str(out_dir / "dispatch.out"))
        support_q = " ".join(shlex.quote(str(path)) for path in support_files)
        queue_parallelism_q = shlex.quote(str(queue_parallelism))
        queue_retry_attempts_q = shlex.quote(str(queue_retry_attempts))
        recover_post_args: list[str] = []
        if raw_post_run_diag:
            recover_post_args.append("--run-hardmax-transfer-diagnostics")
            if bool(post_run_diag_cfg.get("skip_existing", True)):
                recover_post_args.append("--diagnostics-skip-existing")
            if bool(post_run_diag_cfg.get("skip_residual", False)):
                recover_post_args.append("--diagnostics-skip-residual")
            if bool(post_run_diag_cfg.get("remote_analyzers", False)):
                recover_post_args.append("--diagnostics-remote-analyzers")
            if bool(post_run_diag_cfg.get("remote_keep_bundle", False)):
                recover_post_args.append("--diagnostics-remote-keep-bundle")
            if bool(post_run_diag_cfg.get("include_causal_ablation", False)):
                recover_post_args.append("--diagnostics-include-causal-ablation")
            if bool(post_run_diag_cfg.get("include_logit_factors", False)):
                recover_post_args.append("--diagnostics-include-logit-factors")
            for key, flag in (
                ("output_dir", "--diagnostics-output-dir"),
                ("remote_host", "--diagnostics-remote-host"),
                ("remote_dispatch", "--diagnostics-remote-dispatch"),
                ("remote_repo_bundle", "--diagnostics-remote-repo-bundle"),
                ("val_max_seqs", "--diagnostics-val-max-seqs"),
                ("analysis_max_batches", "--diagnostics-analysis-max-batches"),
                ("factor_max_batches", "--diagnostics-factor-max-batches"),
                ("factor_num_factors", "--diagnostics-factor-num-factors"),
                ("factor_top_tokens", "--diagnostics-factor-top-tokens"),
            ):
                value = post_run_diag_cfg.get(key)
                if value not in (None, "", False):
                    recover_post_args.extend([flag, shlex.quote(str(value))])
        recover_post_args_q = " ".join(recover_post_args)
        launch_path.write_text(
            (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                f'DISPATCH={dispatch_q}\n'
                f'WRAPPER={wrapper_q}\n'
                f'SCRIPT={script_q}\n'
                f'DISPATCH_LOG={dispatch_log_q}\n'
                f'ITERATION_DIR={iteration_dir_q}\n'
                f'RECOVER_TOOL={recover_tool_q}\n'
                f'OBSERVED_TOOL={observed_tool_q}\n'
                f'CONFIG_DIR={shlex.quote(str(configs_dir))}\n'
                f'QUEUE_PARALLELISM={queue_parallelism_q}\n'
                f'QUEUE_RETRY_ATTEMPTS={queue_retry_attempts_q}\n'
                f'SUPPORT_FILES=({support_q})\n'
                'HOST="${1:-}"\n\n'
                'mkdir -p "$(dirname "$DISPATCH_LOG")"\n'
                'exec > >(tee -a "$DISPATCH_LOG") 2>&1\n\n'
                'configs=("$CONFIG_DIR"/*.json)\n'
                'post_run() {\n'
                f'  python3 "$RECOVER_TOOL" "$ITERATION_DIR" --skip-existing {recover_post_args_q} >/dev/null 2>&1 || true\n'
                '  python3 "$OBSERVED_TOOL" "$ITERATION_DIR" >/dev/null 2>&1 || true\n'
                '}\n\n'
                'dispatch_one() {\n'
                '  local config=$1\n'
                "  local attempt=1\n"
                "  local rc=0\n"
                '  while (( attempt <= QUEUE_RETRY_ATTEMPTS )); do\n'
                '    echo "[$(date -Iseconds)] dispatch config=$(basename "$config") host=${HOST:-auto} attempt=${attempt}/${QUEUE_RETRY_ATTEMPTS}"\n'
                '    if [[ -n "$HOST" ]]; then\n'
                '      if bash "$DISPATCH" --host "$HOST" "$WRAPPER" "$SCRIPT" "$config" "${SUPPORT_FILES[@]}"; then\n'
                "        rc=0\n"
                "      else\n"
                "        rc=$?\n"
                "      fi\n"
                "    else\n"
                '      if bash "$DISPATCH" "$WRAPPER" "$SCRIPT" "$config" "${SUPPORT_FILES[@]}"; then\n'
                "        rc=0\n"
                "      else\n"
                "        rc=$?\n"
                "      fi\n"
                "    fi\n"
                "    if (( rc == 0 )); then\n"
                "      return 0\n"
                "    fi\n"
                '    if (( attempt == QUEUE_RETRY_ATTEMPTS )); then\n'
                '      echo "[$(date -Iseconds)] dispatch_failed config=$(basename "$config") attempts=${QUEUE_RETRY_ATTEMPTS}" >&2\n'
                "      return $rc\n"
                "    fi\n"
                "    sleep $(( attempt < 5 ? attempt : 5 ))\n"
                "    ((attempt+=1))\n"
                "  done\n"
                "  return $rc\n"
                "}\n\n"
                'if (( QUEUE_PARALLELISM <= 1 || ${#configs[@]} <= 1 )); then\n'
                '  for config in "${configs[@]}"; do\n'
                '    dispatch_one "$config"\n'
                "  done\n"
                "  post_run\n"
                "else\n"
                "  pids=()\n"
                "  status=0\n"
                '  for config in "${configs[@]}"; do\n'
                '    dispatch_one "$config" &\n'
                '    pids+=("$!")\n'
                "    if (( ${#pids[@]} >= QUEUE_PARALLELISM )); then\n"
                '      if ! wait "${pids[0]}"; then\n'
                "        status=1\n"
                "      fi\n"
                '      pids=("${pids[@]:1}")\n'
                "    fi\n"
                "  done\n"
                '  for pid in "${pids[@]}"; do\n'
                '    if ! wait "$pid"; then\n'
                "      status=1\n"
                "    fi\n"
                "  done\n"
                "  post_run\n"
                "  exit $status\n"
                "fi\n"
            ),
            encoding="utf-8",
        )
    launch_path = out_dir / "launch.sh"
    launch_path.chmod(0o755)
    return configs_dir, manifest_path, manifest


def maybe_launch(manifest: dict) -> None:
    if str(manifest.get("dispatch_mode", "parallel")) != "parallel":
        raise RuntimeError("Immediate --launch with run logging is only supported for parallel dispatch")
    dispatch_script = Path(str(manifest["dispatch_script"])).expanduser()
    script_path = Path(str(manifest["script"]))
    configs_dir = Path(str(manifest["configs_dir"]))
    proc = subprocess.Popen(
        [str(dispatch_script), str(script_path), str(configs_dir)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to capture dispatcher stdout")

    lines: list[str] = []
    for raw_line in proc.stdout:
        print(raw_line, end="")
        lines.append(raw_line.rstrip("\n"))
    returncode = proc.wait()
    return lines, returncode


def parse_dispatch_output(lines: list[str]) -> tuple[str | None, dict[str, str], dict[str, str]]:
    sweep_id: str | None = None
    assignments: dict[str, str] = {}
    results: dict[str, str] = {}
    for line in lines:
        if sweep_id is None:
            match = SWEEP_ID_RE.match(line)
            if match:
                sweep_id = match.group(1)
        match = ASSIGNMENT_RE.match(line)
        if match:
            assignments[match.group(1)] = match.group(2)
            continue
        match = RESULT_RE.match(line)
        if match:
            results[match.group(1)] = match.group(2).lower()
    return sweep_id, assignments, results


def run_status_from_dispatch(result: str | None) -> str:
    if result == "success":
        return "completed"
    if result == "failed":
        return "failed"
    return "launched"


def log_sweep_runs(manifest: dict, sweep_id: str, assignments: dict[str, str], results: dict[str, str]) -> int:
    configs_dir = Path(str(manifest["configs_dir"]))
    run_lookup = {Path(run["config_path"]).name: run for run in manifest["runs"]}
    branch = git_value("branch", "--show-current")
    commit = git_value("rev-parse", "HEAD")
    count = 0
    for host, config_name in assignments.items():
        run_meta = run_lookup.get(config_name)
        if run_meta is None:
            continue
        config_path = configs_dir / config_name
        config_payload = load_json(config_path)
        env = config_payload.get("env", {})
        metadata = config_payload.get("metadata", {})
        run_id = str(env.get("RUN_ID"))
        out_dir = str(env.get("OUT_DIR", "logs")).strip("./")
        remote_dir = f"~/jobs/{sweep_id}_{host}"
        dataset = Path(str(env.get("DATA_PATH", ""))).name or None
        script_path = Path(str(run_meta.get("script") or manifest["script"]))
        entry = {
            "artifact_path": f"{remote_dir}/{out_dir}/{run_id}_mlx_model.int8.ptz",
            "branch": branch,
            "command": (
                f"source ~/.cluster_env 2>/dev/null; cd {remote_dir} && "
                f"/opt/homebrew/bin/python3 {script_path.name} --config {config_name}"
            ),
            "commit": commit,
            "config_path": config_path.as_posix(),
            "created_at_utc": now_utc(),
            "dataset": dataset,
            "host": host,
            "iteration_id": manifest.get("iteration_id"),
            "job_id": f"{sweep_id}_{host}",
            "log_path": f"{remote_dir}/{out_dir}/{run_id}.txt",
            "metrics": {},
            "notes": metadata.get("notes"),
            "run_id": run_id,
            "script": script_path.name,
            "status": run_status_from_dispatch(results.get(host)),
            "sweep_id": sweep_id,
            "sweep_slug": manifest.get("sweep_slug"),
            "tags": sorted(set(list(metadata.get("tags", [])) + ["cluster-sweep"])),
            "train_shards": env.get("TRAIN_SHARDS"),
        }
        append_jsonl(RUN_INDEX, entry)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or launch a config-driven MLX sweep for the Mac mini cluster.")
    parser.add_argument("spec", type=Path, help="JSON sweep spec")
    parser.add_argument("--out-dir", type=Path, help="Optional output directory for generated configs")
    parser.add_argument("--launch", action="store_true", help="Launch immediately via dispatch_parallel.sh after preparing configs")
    parser.add_argument("--no-log-runs", action="store_true", help="Do not append run_index rows after a launched sweep completes")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    out_dir = build_output_dir(spec, spec_path, args.out_dir)
    configs_dir, manifest_path, manifest = prepare_sweep(spec, spec_path, out_dir)
    summary = {
        "configs_dir": str(configs_dir),
        "manifest_path": str(manifest_path),
        "launch_command": (out_dir / "launch.sh").as_posix(),
        "dispatch_mode": manifest["dispatch_mode"],
        "run_count": manifest["config_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.launch:
        if manifest["dispatch_mode"] != "parallel":
            raise SystemExit("--launch currently supports only dispatch_mode='parallel'; run the generated launch.sh for queue mode")
        lines, returncode = maybe_launch(manifest)
        sweep_id, assignments, results = parse_dispatch_output(lines)
        if not args.no_log_runs and sweep_id is not None and assignments:
            logged = log_sweep_runs(manifest, sweep_id, assignments, results)
            print(json.dumps({"logged_runs": logged, "run_index": str(RUN_INDEX)}, indent=2, sort_keys=True))
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
