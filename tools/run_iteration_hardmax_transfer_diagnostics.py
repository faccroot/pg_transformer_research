#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_iteration_saved_diagnostics import (
    ROOT,
    RunEntry,
    build_run_entries,
    discover_artifact_for_run,
    find_control_entry,
    load_json,
    nested_get,
)


RESIDUAL_WRAPPER = ROOT / "tools" / "run_iteration_saved_diagnostics.py"
CONTROLLER_TOOL = ROOT / "tools" / "analyze_hardmax_structural_controller.py"
CAUSAL_TOOL = ROOT / "tools" / "eval_saved_hardmax_causal_ablation.py"
FACTOR_TOOL = ROOT / "tools" / "analyze_saved_logit_factors.py"
REMOTE_WRAPPER = ROOT / "tools" / "run_remote_saved_hardmax_analysis.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run saved-artifact hardmax transfer diagnostics for a generated sweep, including "
            "residual/controller analysis and optional causal-ablation / logit-factor outputs."
        )
    )
    p.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json and configs/")
    p.add_argument(
        "--output-dir",
        default="",
        help="Directory for combined diagnostics. Defaults to <iteration_dir>/hardmax_transfer_diagnostics",
    )
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional directory to scan recursively for artifacts. Can be passed multiple times.",
    )
    p.add_argument("--control-slug", default="", help="Optional run_slug to treat as the control.")
    p.add_argument("--python", default=sys.executable, help="Python executable to use for analyzer subprocesses.")
    p.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    p.add_argument("--skip-residual", action="store_true", help="Skip residual/prosody diagnostics.")
    p.add_argument("--remote-analyzers", action="store_true", help="Run controller/causal/factor analyzers via Mini dispatch.")
    p.add_argument("--remote-host", default="", help="Optional target Mini alias for remote analyzers.")
    p.add_argument("--remote-dispatch", default="", help="Optional dispatch.sh path for remote analyzers.")
    p.add_argument("--remote-repo-bundle", default="", help="Optional existing repo bundle tar/tgz for remote analyzers.")
    p.add_argument("--remote-keep-bundle", action="store_true", help="Keep auto-built repo bundle for remote analyzers.")
    p.add_argument("--val-max-seqs", type=int, default=1024)
    p.add_argument("--eval-seq-len", type=int, default=1024)
    p.add_argument("--eval-stride", type=int, default=0)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=64)
    p.add_argument("--probe-layers", default="-1")
    p.add_argument("--probe-max-samples", type=int, default=32768)
    p.add_argument("--probe-ridge", type=float, default=1.0)
    p.add_argument("--probe-train-frac", type=float, default=0.7)
    p.add_argument("--transition-window", type=int, default=32)
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--top-k-transitions", type=int, default=10)
    p.add_argument("--preview-radius", type=int, default=24)
    p.add_argument("--top-k-states", type=int, default=8)
    p.add_argument("--include-causal-ablation", action="store_true", help="Emit saved-artifact causal hardmax ablation JSONs.")
    p.add_argument("--causal-ablations", default="", help="Optional comma-separated causal ablation list.")
    p.add_argument("--include-logit-factors", action="store_true", help="Emit saved-artifact logit factor JSONs.")
    p.add_argument("--factor-mode", choices=("logits", "prob_residual"), default="prob_residual")
    p.add_argument("--factor-num-factors", type=int, default=4)
    p.add_argument("--factor-top-tokens", type=int, default=12)
    p.add_argument("--factor-max-batches", type=int, default=32)
    return p.parse_args()


def run_subprocess(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def hardmax_enabled(entry: RunEntry) -> bool:
    payload = load_json(entry.config_path)
    env = payload.get("env", {})
    if not isinstance(env, dict):
        return False
    states = int(env.get("HARDMAX_STRUCT_NUM_STATES", 0) or 0)
    static_adapter = str(env.get("HARDMAX_STRUCT_STATIC_ADAPTER", "0")).strip().lower() in {"1", "true", "yes", "on"}
    return states > 0 or static_adapter


def remote_saved_analysis_command(
    args: argparse.Namespace,
    *,
    mode: str,
    entry: RunEntry,
    artifact_path: Path,
    output_json: Path,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        args.python,
        str(REMOTE_WRAPPER),
        mode,
        "--artifact",
        str(artifact_path),
        "--config-json",
        str(entry.config_path),
        "--output-json",
        str(output_json),
    ]
    if args.remote_host:
        cmd.extend(["--host", str(args.remote_host)])
    if args.remote_dispatch:
        cmd.extend(["--dispatch", str(args.remote_dispatch)])
    if entry.tokenizer_override:
        cmd.extend(["--tokenizer-path", entry.tokenizer_override])
    if args.remote_repo_bundle:
        cmd.extend(["--repo-bundle", str(args.remote_repo_bundle)])
    if args.remote_keep_bundle:
        cmd.append("--keep-bundle")
    for extra_arg in extra_args:
        cmd.append(f"--extra-arg={extra_arg}")
    return cmd


def controller_command(
    args: argparse.Namespace,
    *,
    entry: RunEntry,
    artifact_path: Path,
    result_json: Path,
) -> list[str]:
    if args.remote_analyzers:
        extra_args = [
            "--label",
            entry.run_slug,
            "--cache-variant",
            str(args.cache_variant),
            "--train-shards",
            str(int(args.train_shards)),
            "--analysis-max-batches",
            str(int(args.analysis_max_batches)),
            "--top-k-states",
            str(int(args.top_k_states)),
        ]
        if args.val_max_seqs is not None:
            extra_args.extend(["--val-max-seqs", str(int(args.val_max_seqs))])
        if args.eval_seq_len is not None:
            extra_args.extend(["--eval-seq-len", str(int(args.eval_seq_len))])
        if args.eval_stride:
            extra_args.extend(["--eval-stride", str(int(args.eval_stride))])
        if args.eval_batch_seqs is not None:
            extra_args.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
        return remote_saved_analysis_command(
            args,
            mode="controller",
            entry=entry,
            artifact_path=artifact_path,
            output_json=result_json,
            extra_args=extra_args,
        )
    cmd = [
        args.python,
        str(CONTROLLER_TOOL),
        "--artifact",
        str(artifact_path),
        "--config-json",
        str(entry.config_path),
        "--result-json",
        str(result_json),
        "--label",
        entry.run_slug,
        "--cache-variant",
        str(args.cache_variant),
        "--train-shards",
        str(int(args.train_shards)),
        "--analysis-max-batches",
        str(int(args.analysis_max_batches)),
        "--top-k-states",
        str(int(args.top_k_states)),
    ]
    if entry.tokenizer_override:
        cmd.extend(["--tokenizer-path", entry.tokenizer_override])
    if entry.data_override:
        cmd.extend(["--data-path", entry.data_override])
    if args.val_max_seqs is not None:
        cmd.extend(["--val-max-seqs", str(int(args.val_max_seqs))])
    if args.eval_seq_len is not None:
        cmd.extend(["--eval-seq-len", str(int(args.eval_seq_len))])
    if args.eval_stride:
        cmd.extend(["--eval-stride", str(int(args.eval_stride))])
    if args.eval_batch_seqs is not None:
        cmd.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
    return cmd


def causal_command(
    args: argparse.Namespace,
    *,
    entry: RunEntry,
    artifact_path: Path,
    result_json: Path,
) -> list[str]:
    if args.remote_analyzers:
        extra_args = [
            "--label",
            entry.run_slug,
        ]
        if args.causal_ablations:
            extra_args.extend(["--ablations", str(args.causal_ablations)])
        if args.val_max_seqs is not None:
            extra_args.extend(["--val-max-seqs", str(int(args.val_max_seqs))])
        if args.eval_seq_len is not None:
            extra_args.extend(["--eval-seq-len", str(int(args.eval_seq_len))])
        if args.eval_stride:
            extra_args.extend(["--eval-stride", str(int(args.eval_stride))])
        if args.eval_batch_seqs is not None:
            extra_args.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
        return remote_saved_analysis_command(
            args,
            mode="causal",
            entry=entry,
            artifact_path=artifact_path,
            output_json=result_json,
            extra_args=extra_args,
        )
    cmd = [
        args.python,
        str(CAUSAL_TOOL),
        "--artifact",
        str(artifact_path),
        "--config-json",
        str(entry.config_path),
        "--result-json",
        str(result_json),
        "--label",
        entry.run_slug,
        "--val-max-seqs",
        str(int(args.val_max_seqs)),
        "--eval-seq-len",
        str(int(args.eval_seq_len)),
    ]
    if entry.tokenizer_override:
        cmd.extend(["--tokenizer-path", entry.tokenizer_override])
    if entry.data_override:
        cmd.extend(["--data-path", entry.data_override])
    if args.causal_ablations:
        cmd.extend(["--ablations", str(args.causal_ablations)])
    if args.eval_stride:
        cmd.extend(["--eval-stride", str(int(args.eval_stride))])
    if args.eval_batch_seqs is not None:
        cmd.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
    return cmd


def factor_command(
    args: argparse.Namespace,
    *,
    entry: RunEntry,
    artifact_path: Path,
    result_json: Path,
) -> list[str]:
    if args.remote_analyzers:
        extra_args = [
            "--label",
            entry.run_slug,
            "--mode",
            str(args.factor_mode),
            "--num-factors",
            str(int(args.factor_num_factors)),
            "--top-tokens",
            str(int(args.factor_top_tokens)),
            "--max-batches",
            str(int(args.factor_max_batches)),
        ]
        if args.val_max_seqs is not None:
            extra_args.extend(["--val-max-seqs", str(int(args.val_max_seqs))])
        if args.eval_seq_len is not None:
            extra_args.extend(["--eval-seq-len", str(int(args.eval_seq_len))])
        return remote_saved_analysis_command(
            args,
            mode="factors",
            entry=entry,
            artifact_path=artifact_path,
            output_json=result_json,
            extra_args=extra_args,
        )
    cmd = [
        args.python,
        str(FACTOR_TOOL),
        "--artifact",
        str(artifact_path),
        "--config-json",
        str(entry.config_path),
        "--summary-json",
        str(result_json),
        "--label",
        entry.run_slug,
        "--mode",
        str(args.factor_mode),
        "--num-factors",
        str(int(args.factor_num_factors)),
        "--top-tokens",
        str(int(args.factor_top_tokens)),
        "--max-batches",
        str(int(args.factor_max_batches)),
        "--val-max-seqs",
        str(int(args.val_max_seqs)),
        "--eval-seq-len",
        str(int(args.eval_seq_len)),
    ]
    if entry.tokenizer_override:
        cmd.extend(["--tokenizer-path", entry.tokenizer_override])
    if entry.data_override:
        cmd.extend(["--data-path", entry.data_override])
    return cmd


def format_float(value: object | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def baseline_bpb_from_causal_payload(payload: dict[str, object]) -> float | None:
    for row in payload.get("results", []):
        if isinstance(row, dict) and row.get("ablation") == "baseline" and row.get("val_bpb") is not None:
            return float(row["val_bpb"])
    return None


def write_controller_summary(
    out_path: Path,
    *,
    control_entry: RunEntry,
    result_paths: dict[str, Path],
    artifact_paths: dict[str, Path],
) -> None:
    control_payload = load_json(result_paths[control_entry.run_slug])
    lines: list[str] = []
    lines.append("# Hardmax Controller Diagnostics")
    lines.append("")
    lines.append(f"Control reference: `{control_entry.run_slug}`")
    lines.append("")
    lines.append("Analyzed hardmax runs:")
    for run_slug, result_path in sorted(result_paths.items()):
        lines.append(f"- `{run_slug}`: [{result_path.name}]({result_path.as_posix()})")
    for run_slug, result_path in sorted(result_paths.items()):
        payload = load_json(result_path)
        lines.append("")
        lines.append(f"## {run_slug}")
        lines.append("")
        lines.append(f"- artifact: [{artifact_paths[run_slug].name}]({artifact_paths[run_slug].as_posix()})")
        lines.append(f"- config: [{Path(payload['config_json']).name}]({payload['config_json']})")
        lines.append(f"- mean NLL: `{format_float(payload.get('mean_nll'))}`")
        state_usage = payload.get("state_usage", {})
        controller = payload.get("controller", {})
        transitions = payload.get("state_transitions", {})
        lines.append(f"- used states: `{state_usage.get('used_states', 'n/a')}`")
        lines.append(f"- max state fraction: `{format_float(state_usage.get('max_state_fraction'))}`")
        lines.append(f"- usage perplexity: `{format_float(state_usage.get('usage_perplexity'))}`")
        lines.append(f"- self-transition fraction: `{format_float(transitions.get('self_transition_fraction'))}`")
        lines.append(f"- confidence std: `{format_float(controller.get('confidence_std'))}`")
        lines.append(f"- corr(confidence, NLL): `{format_float(controller.get('corr_confidence_vs_nll'))}`")
        after_sentence = nested_get(controller, ("nll_by_prev_boundary", "after_boundary_ge", "after_sentence", "mean"))
        lines.append(f"- NLL after sentence boundary: `{format_float(after_sentence)}`")
        if run_slug != control_entry.run_slug:
            ctrl_state_usage = control_payload.get("state_usage", {})
            ctrl_controller = control_payload.get("controller", {})
            ctrl_transitions = control_payload.get("state_transitions", {})
            lines.append(f"- delta max state fraction vs control: `{format_float((state_usage.get('max_state_fraction') or 0.0) - (ctrl_state_usage.get('max_state_fraction') or 0.0))}`")
            lines.append(f"- delta usage perplexity vs control: `{format_float((state_usage.get('usage_perplexity') or 0.0) - (ctrl_state_usage.get('usage_perplexity') or 0.0))}`")
            lines.append(f"- delta confidence std vs control: `{format_float((controller.get('confidence_std') or 0.0) - (ctrl_controller.get('confidence_std') or 0.0))}`")
            lines.append(f"- delta self-transition fraction vs control: `{format_float((transitions.get('self_transition_fraction') or 0.0) - (ctrl_transitions.get('self_transition_fraction') or 0.0))}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_causal_summary(
    out_path: Path,
    *,
    control_entry: RunEntry,
    result_paths: dict[str, Path],
) -> None:
    control_payload = load_json(result_paths[control_entry.run_slug])
    control_baseline = baseline_bpb_from_causal_payload(control_payload)
    lines: list[str] = []
    lines.append("# Hardmax Causal Ablation Summary")
    lines.append("")
    lines.append(f"Control reference: `{control_entry.run_slug}`")
    lines.append("")
    for run_slug, result_path in sorted(result_paths.items()):
        payload = load_json(result_path)
        lines.append(f"## {run_slug}")
        baseline_row = next((row for row in payload.get("results", []) if row.get("ablation") == "baseline"), None)
        baseline_bpb = baseline_row.get("val_bpb") if isinstance(baseline_row, dict) else None
        lines.append("")
        lines.append(f"- baseline val_bpb: `{format_float(baseline_bpb)}`")
        if run_slug != control_entry.run_slug:
            delta_vs_control = (float(baseline_bpb) - float(control_baseline)) if baseline_bpb is not None and control_baseline is not None else None
            lines.append(f"- delta baseline vs control: `{format_float(delta_vs_control)}`")
        for row in payload.get("results", []):
            if not isinstance(row, dict) or row.get("ablation") == "baseline":
                continue
            delta = None
            if baseline_bpb is not None and row.get("val_bpb") is not None:
                delta = float(row["val_bpb"]) - float(baseline_bpb)
            lines.append(f"- {row.get('ablation')}: val_bpb `{format_float(row.get('val_bpb'))}`, delta `{format_float(delta)}`")
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_factor_summary(
    out_path: Path,
    *,
    result_paths: dict[str, Path],
) -> None:
    lines: list[str] = []
    lines.append("# Hardmax Logit Factor Summary")
    lines.append("")
    for run_slug, result_path in sorted(result_paths.items()):
        payload = load_json(result_path)
        lines.append(f"## {run_slug}")
        lines.append("")
        lines.append(f"- factor mode: `{payload.get('mode', 'n/a')}`")
        lines.append(f"- num factors: `{payload.get('num_factors', 'n/a')}`")
        factors = payload.get("factors", [])
        for factor in factors[: min(len(factors), 2)]:
            if not isinstance(factor, dict):
                continue
            lines.append(
                f"- factor {factor.get('factor_index', 'n/a')}: "
                f"evr `{format_float(factor.get('explained_variance_ratio'))}`, "
                f"|coord|-NLL corr `{format_float(factor.get('abs_coord_nll_corr'))}`"
            )
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    manifest = load_json(iteration_dir / "manifest.json")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (iteration_dir / "hardmax_transfer_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_residual:
        residual_dir = output_dir / "residual"
        residual_cmd = [
            args.python,
            str(RESIDUAL_WRAPPER),
            str(iteration_dir),
            "--output-dir",
            str(residual_dir),
            "--control-slug",
            str(args.control_slug),
            "--python",
            str(args.python),
            "--val-max-seqs",
            str(int(args.val_max_seqs)),
            "--eval-seq-len",
            str(int(args.eval_seq_len)),
            "--cache-variant",
            str(args.cache_variant),
            "--train-shards",
            str(int(args.train_shards)),
            "--analysis-max-batches",
            str(int(args.analysis_max_batches)),
            "--max-lag",
            str(int(args.max_lag)),
            "--probe-layers",
            str(args.probe_layers),
            "--probe-max-samples",
            str(int(args.probe_max_samples)),
            "--probe-ridge",
            str(float(args.probe_ridge)),
            "--probe-train-frac",
            str(float(args.probe_train_frac)),
            "--transition-window",
            str(int(args.transition_window)),
            "--top-k-tokens",
            str(int(args.top_k_tokens)),
            "--top-k-transitions",
            str(int(args.top_k_transitions)),
            "--preview-radius",
            str(int(args.preview_radius)),
        ]
        if args.skip_existing:
            residual_cmd.append("--skip-existing")
        if args.dry_run:
            residual_cmd.append("--dry-run")
        if args.eval_stride:
            residual_cmd.extend(["--eval-stride", str(int(args.eval_stride))])
        if args.eval_batch_seqs is not None:
            residual_cmd.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
        for root in args.search_root:
            residual_cmd.extend(["--search-root", root])
        run_subprocess(residual_cmd, dry_run=args.dry_run)

    entries = build_run_entries(iteration_dir, manifest)
    control_entry = find_control_entry(entries, args.control_slug)
    search_roots = [iteration_dir]
    search_roots.extend(Path(item).expanduser().resolve() for item in args.search_root)

    controller_dir = output_dir / "controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    result_paths: dict[str, Path] = {}
    artifact_paths: dict[str, Path] = {}
    causal_paths: dict[str, Path] = {}
    factor_paths: dict[str, Path] = {}
    for entry in entries:
        if not hardmax_enabled(entry):
            continue
        artifact_path = discover_artifact_for_run(entry.run_id, entry.run_slug, entry.config_name, search_roots)
        if artifact_path is None:
            continue
        artifact_paths[entry.run_slug] = artifact_path
        result_json = controller_dir / f"{entry.run_slug}.json"
        result_paths[entry.run_slug] = result_json
        if not (args.skip_existing and result_json.exists()):
            run_subprocess(
                controller_command(args, entry=entry, artifact_path=artifact_path, result_json=result_json),
                dry_run=args.dry_run,
            )
        if args.include_causal_ablation:
            causal_dir = output_dir / "causal"
            causal_dir.mkdir(parents=True, exist_ok=True)
            causal_json = causal_dir / f"{entry.run_slug}.json"
            causal_paths[entry.run_slug] = causal_json
            if not (args.skip_existing and causal_json.exists()):
                run_subprocess(
                    causal_command(args, entry=entry, artifact_path=artifact_path, result_json=causal_json),
                    dry_run=args.dry_run,
                )
        if args.include_logit_factors:
            factor_dir = output_dir / "factors"
            factor_dir.mkdir(parents=True, exist_ok=True)
            factor_json = factor_dir / f"{entry.run_slug}.json"
            factor_paths[entry.run_slug] = factor_json
            if not (args.skip_existing and factor_json.exists()):
                run_subprocess(
                    factor_command(args, entry=entry, artifact_path=artifact_path, result_json=factor_json),
                    dry_run=args.dry_run,
                )

    if not args.dry_run and result_paths:
        control_run_slug = control_entry.run_slug
        if control_run_slug not in result_paths:
            # if the control has no hardmax controller, pick the first hardmax run as the baseline summary anchor
            control_run_slug = sorted(result_paths)[0]
            control_entry = next(entry for entry in entries if entry.run_slug == control_run_slug)
        write_controller_summary(
            output_dir / "controller_summary.md",
            control_entry=control_entry,
            result_paths=result_paths,
            artifact_paths=artifact_paths,
        )
    if not args.dry_run and causal_paths:
        control_run_slug = control_entry.run_slug
        if control_run_slug not in causal_paths:
            control_run_slug = sorted(causal_paths)[0]
            control_entry = next(entry for entry in entries if entry.run_slug == control_run_slug)
        write_causal_summary(
            output_dir / "causal_ablation_summary.md",
            control_entry=control_entry,
            result_paths=causal_paths,
        )
    if not args.dry_run and factor_paths:
        write_factor_summary(
            output_dir / "factor_summary.md",
            result_paths=factor_paths,
        )


if __name__ == "__main__":
    main()
