#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.compare_gcca_stitching import compare_gcca_stitching
    from tools.representation_learning.extract_model_representation import read_calibration_records
    from tools.representation_learning.model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch
    from tools.representation_learning.schemas import SharedLatentGeometry
    from tools.representation_learning.verify_model_stitching import (
        _collect_sequence_representations,
        _texts_for_chunk_ids,
        resolve_model_layer_index,
    )
    from tools.representation_learning.verify_model_stitching_cohort import (
        build_stitching_cohort_report_from_resources,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from compare_gcca_stitching import compare_gcca_stitching  # type: ignore[no-redef]
    from extract_model_representation import read_calibration_records  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch  # type: ignore[no-redef]
    from schemas import SharedLatentGeometry  # type: ignore[no-redef]
    from verify_model_stitching import _collect_sequence_representations, _texts_for_chunk_ids, resolve_model_layer_index  # type: ignore[no-redef]
    from verify_model_stitching_cohort import build_stitching_cohort_report_from_resources  # type: ignore[no-redef]


def _resolve_layers(shared_geometry: SharedLatentGeometry, layers: list[int] | None) -> list[int]:
    if not layers:
        return sorted(shared_geometry.layers)
    resolved = sorted({int(layer) for layer in layers})
    missing = [layer for layer in resolved if layer not in shared_geometry.layers]
    if missing:
        raise ValueError(f"Requested layers not present in shared geometry: {missing}")
    return resolved


def _select_common_chunk_ids(
    records: list[dict[str, Any]],
    *,
    shared_geometry: SharedLatentGeometry,
    layers: list[int],
    max_examples: int,
) -> list[str]:
    common_ids: set[str] | None = None
    for layer_idx in layers:
        layer_chunk_ids = set(shared_geometry.layers[layer_idx].chunk_ids)
        common_ids = layer_chunk_ids if common_ids is None else (common_ids & layer_chunk_ids)
    allowed = common_ids or set()
    selected: list[str] = []
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        if chunk_id and chunk_id in allowed:
            selected.append(chunk_id)
        if len(selected) >= max_examples:
            break
    if not selected:
        raise ValueError("No shared chunk_ids available across the requested layers and calibration corpus")
    return selected


def build_stitching_layer_sweep_report(
    *,
    shared_geometry_path: str | Path,
    calibration_jsonl: str | Path,
    output_dir: str | Path,
    layers: list[int] | None = None,
    max_examples: int = 128,
    train_fraction: float = 0.75,
    seed: int = 17,
    batch_size: int = 8,
    max_length: int | None = 1024,
    ridge_lambda: float = 1e-4,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    include_self_pairs: bool = False,
    representation_mode: str = "layer_last_hidden_continuation",
) -> dict[str, Any]:
    shared_geometry = SharedLatentGeometry.load(shared_geometry_path)
    layer_indices = _resolve_layers(shared_geometry, layers)
    records = read_calibration_records(calibration_jsonl, max_examples=0)
    chunk_ids = _select_common_chunk_ids(
        records,
        shared_geometry=shared_geometry,
        layers=layer_indices,
        max_examples=max_examples,
    )
    texts = _texts_for_chunk_ids(records, chunk_ids)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    adapters: dict[str, Any] = {}
    batches: dict[str, SequenceRepresentationBatch] = {}
    model_layer_requests: dict[str, list[int]] = {}
    for model_id in shared_geometry.source_models:
        adapter = HFCausalLMAdapter(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        adapters[model_id] = adapter
        if representation_mode == "layer_last_hidden_continuation":
            requested_layers = sorted({
                resolve_model_layer_index(
                    shared_geometry=shared_geometry,
                    shared_layer=shared_layer_idx,
                    model_num_layers=int(getattr(adapter, "num_layers", 0)),
                )
                for shared_layer_idx in layer_indices
            })
        else:
            requested_layers = []
        model_layer_requests[model_id] = requested_layers
        batches[model_id] = _collect_sequence_representations(
            adapter,
            texts,
            batch_size=batch_size,
            layers=requested_layers,
            capture_full_sequences=representation_mode == "layer_last_hidden_continuation",
            max_length=max_length,
        )

    per_layer: dict[str, Any] = {}
    best_layer_by_target: dict[str, dict[str, Any]] = {}
    for layer_idx in layer_indices:
        layer_dir = output_root / f"layer_{layer_idx:02d}"
        cohort = build_stitching_cohort_report_from_resources(
            shared_geometry=shared_geometry,
            shared_geometry_path=shared_geometry_path,
            calibration_jsonl=calibration_jsonl,
            output_dir=layer_dir,
            shared_layer=layer_idx,
            selected_chunk_ids=chunk_ids,
            adapters=adapters,
            batches=batches,
            train_fraction=train_fraction,
            seed=seed,
            batch_size=batch_size,
            max_length=max_length,
            ridge_lambda=ridge_lambda,
            include_self_pairs=include_self_pairs,
            representation_mode=representation_mode,
        )
        cohort_summary_path = layer_dir / "cohort_summary.json"
        joined = compare_gcca_stitching(
            shared_geometry_path=shared_geometry_path,
            stitching_cohort_report=cohort_summary_path,
        )
        joined_path = layer_dir / "gcca_vs_stitch.json"
        joined_path.write_text(json.dumps(joined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cohort_summary = cohort["summary"]
        per_layer[str(layer_idx)] = {
            "cohort_summary_path": str(cohort_summary_path),
            "gcca_vs_stitch_path": str(joined_path),
            "mean_eval_target_logit_kl_mean": float(cohort_summary["mean_eval_target_logit_kl_mean"]),
            "mean_eval_hidden_cosine_mean": float(cohort_summary["mean_eval_hidden_cosine_mean"]),
            "mean_eval_target_top1_agreement": float(cohort_summary["mean_eval_target_top1_agreement"]),
            "pair_count": int(cohort_summary["pair_count"]),
            "num_examples": int(cohort_summary["num_examples"]),
            "best_source_for_target": cohort_summary["best_source_for_target"],
            "gcca_vs_stitch_summary": joined["summary"],
            "layer_view_residuals": joined["layer_view_residuals"],
        }
        for target_model_id, best in cohort_summary["best_source_for_target"].items():
            candidate = {
                "layer": int(layer_idx),
                "source_model_id": str(best["source_model_id"]),
                "eval_target_logit_kl_mean": float(best["eval_target_logit_kl_mean"]),
                "eval_hidden_cosine_mean": float(best["eval_hidden_cosine_mean"]),
                "eval_target_top1_agreement": float(best["eval_target_top1_agreement"]),
            }
            existing = best_layer_by_target.get(target_model_id)
            if existing is None or (
                candidate["eval_target_logit_kl_mean"],
                -candidate["eval_target_top1_agreement"],
                str(candidate["source_model_id"]),
            ) < (
                float(existing["eval_target_logit_kl_mean"]),
                -float(existing["eval_target_top1_agreement"]),
                str(existing["source_model_id"]),
            ):
                best_layer_by_target[target_model_id] = candidate

    layer_records = [
        {"layer": int(layer), **payload}
        for layer, payload in per_layer.items()
    ]
    best_layer_by_mean_kl = min(layer_records, key=lambda item: float(item["mean_eval_target_logit_kl_mean"]))
    best_layer_by_hidden_cosine = max(layer_records, key=lambda item: float(item["mean_eval_hidden_cosine_mean"]))
    best_layer_by_top1 = max(layer_records, key=lambda item: float(item["mean_eval_target_top1_agreement"]))

    report = {
        "shared_geometry_path": str(Path(shared_geometry_path).resolve()),
        "calibration_jsonl": str(Path(calibration_jsonl).resolve()),
        "source_models": shared_geometry.source_models,
        "layers": per_layer,
        "summary": {
            "layer_count": len(layer_indices),
            "evaluated_layers": layer_indices,
            "selected_chunk_ids": chunk_ids,
            "num_examples": len(chunk_ids),
            "best_layer_by_mean_eval_target_logit_kl": {
                "layer": int(best_layer_by_mean_kl["layer"]),
                "mean_eval_target_logit_kl_mean": float(best_layer_by_mean_kl["mean_eval_target_logit_kl_mean"]),
            },
            "best_layer_by_mean_hidden_cosine": {
                "layer": int(best_layer_by_hidden_cosine["layer"]),
                "mean_eval_hidden_cosine_mean": float(best_layer_by_hidden_cosine["mean_eval_hidden_cosine_mean"]),
            },
            "best_layer_by_mean_top1_agreement": {
                "layer": int(best_layer_by_top1["layer"]),
                "mean_eval_target_top1_agreement": float(best_layer_by_top1["mean_eval_target_top1_agreement"]),
            },
            "best_layer_by_target": best_layer_by_target,
        },
        "metadata": {
            "method": "last_hidden_affine_stitching_layer_sweep" if representation_mode == "final_last_hidden" else "layer_last_hidden_continuation_stitching_layer_sweep",
            "representation_source": "final_last_hidden_only" if representation_mode == "final_last_hidden" else "source_last_position_hidden_at_selected_layer",
            "target_evaluation": "target_lm_head_projection_only" if representation_mode == "final_last_hidden" else "target_remainder_execution_to_logits",
            "shared_layer_role": "chunk_selection_and_join_axis_only" if representation_mode == "final_last_hidden" else "relative_depth_and_join_axis",
            "representation_mode": representation_mode,
            "train_fraction": float(train_fraction),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "max_length": int(max_length) if max_length is not None else None,
            "ridge_lambda": float(ridge_lambda),
            "include_self_pairs": bool(include_self_pairs),
            "chunk_policy": "intersection_across_layers",
        },
    }
    summary_path = output_root / "layer_sweep_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a shared-layer stitching sweep over a SharedLatentGeometry cohort using the same chunk set and forward passes.")
    parser.add_argument("output_dir", help="Output directory for the layer sweep")
    parser.add_argument("--shared-geometry", required=True, help="SharedLatentGeometry artifact defining the cohort")
    parser.add_argument("--calibration-jsonl", required=True, help="Calibration JSONL with chunk_id/text fields")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Optional explicit shared-geometry layers to evaluate")
    parser.add_argument("--max-examples", type=int, default=128, help="Maximum number of shared chunks to evaluate")
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Train split fraction for fitting each stitch map")
    parser.add_argument("--seed", type=int, default=17, help="Random split seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Forward-pass batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Optional tokenizer truncation length")
    parser.add_argument("--ridge-lambda", type=float, default=1e-4, help="Ridge penalty for the affine stitch fit")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading Hugging Face models")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype name passed to the adapters")
    parser.add_argument("--include-self-pairs", action="store_true", help="Also evaluate source==target identity stitching")
    parser.add_argument(
        "--representation-mode",
        choices=["final_last_hidden", "layer_last_hidden_continuation"],
        default="layer_last_hidden_continuation",
        help="Whether to stitch final hidden states directly or continue the target from a selected layer after replacing its last-position hidden state",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_stitching_layer_sweep_report(
        shared_geometry_path=args.shared_geometry,
        calibration_jsonl=args.calibration_jsonl,
        output_dir=args.output_dir,
        layers=args.layers,
        max_examples=args.max_examples,
        train_fraction=args.train_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        ridge_lambda=args.ridge_lambda,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=args.torch_dtype,
        include_self_pairs=bool(args.include_self_pairs),
        representation_mode=args.representation_mode,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
