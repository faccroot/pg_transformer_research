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
    from tools.representation_learning.extract_model_representation import read_calibration_records
    from tools.representation_learning.model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch
    from tools.representation_learning.schemas import SharedLatentGeometry
    from tools.representation_learning.verify_model_stitching import (
        _collect_sequence_representations,
        _select_chunk_ids,
        _texts_for_chunk_ids,
        build_stitching_report_from_batches,
        resolve_model_layer_index,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from extract_model_representation import read_calibration_records  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch  # type: ignore[no-redef]
    from schemas import SharedLatentGeometry  # type: ignore[no-redef]
    from verify_model_stitching import (  # type: ignore[no-redef]
        _collect_sequence_representations,
        _select_chunk_ids,
        _texts_for_chunk_ids,
        build_stitching_report_from_batches,
        resolve_model_layer_index,
    )


def build_stitching_cohort_report(
    *,
    shared_geometry_path: str | Path,
    calibration_jsonl: str | Path,
    output_dir: str | Path,
    shared_layer: int | None = None,
    max_examples: int = 128,
    train_fraction: float = 0.75,
    seed: int = 17,
    batch_size: int = 8,
    max_length: int | None = 1024,
    ridge_lambda: float = 1e-4,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    include_self_pairs: bool = False,
    representation_mode: str = "final_last_hidden",
) -> dict[str, Any]:
    shared_geometry = SharedLatentGeometry.load(shared_geometry_path)
    records = read_calibration_records(calibration_jsonl, max_examples=0)
    effective_shared_layer = int(shared_layer) if shared_layer is not None else max(shared_geometry.layers)
    chunk_ids = _select_chunk_ids(
        records,
        shared_geometry=shared_geometry,
        shared_layer=effective_shared_layer,
        max_examples=max_examples,
    )
    texts = _texts_for_chunk_ids(records, chunk_ids)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters: dict[str, Any] = {}
    batches: dict[str, SequenceRepresentationBatch] = {}
    model_layer_requests: dict[str, list[int]] = {model_id: [] for model_id in shared_geometry.source_models}
    for model_id in shared_geometry.source_models:
        adapter = HFCausalLMAdapter(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        adapters[model_id] = adapter
        if representation_mode == "layer_last_hidden_continuation":
            model_layer_requests[model_id] = [
                resolve_model_layer_index(
                    shared_geometry=shared_geometry,
                    shared_layer=effective_shared_layer,
                    model_num_layers=int(getattr(adapter, "num_layers", 0)),
                )
            ]
        batches[model_id] = _collect_sequence_representations(
            adapter,
            texts,
            batch_size=batch_size,
            layers=model_layer_requests[model_id],
            capture_full_sequences=representation_mode == "layer_last_hidden_continuation",
            max_length=max_length,
        )

    return build_stitching_cohort_report_from_resources(
        shared_geometry=shared_geometry,
        shared_geometry_path=shared_geometry_path,
        calibration_jsonl=calibration_jsonl,
        output_dir=output_dir,
        shared_layer=effective_shared_layer,
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


def build_stitching_cohort_report_from_resources(
    *,
    shared_geometry: SharedLatentGeometry,
    shared_geometry_path: str | Path | None = None,
    calibration_jsonl: str | Path,
    output_dir: str | Path,
    shared_layer: int,
    selected_chunk_ids: list[str],
    adapters: dict[str, Any],
    batches: dict[str, SequenceRepresentationBatch],
    train_fraction: float = 0.75,
    seed: int = 17,
    batch_size: int = 8,
    max_length: int | None = 1024,
    ridge_lambda: float = 1e-4,
    include_self_pairs: bool = False,
    representation_mode: str = "final_last_hidden",
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise: list[dict[str, Any]] = []
    for source_model_id in shared_geometry.source_models:
        for target_model_id in shared_geometry.source_models:
            if not include_self_pairs and source_model_id == target_model_id:
                continue
            report = build_stitching_report_from_batches(
                source_model_id=source_model_id,
                target_model_id=target_model_id,
                calibration_jsonl=calibration_jsonl,
                selected_chunk_ids=selected_chunk_ids,
                source_batch=batches[source_model_id],
                target_batch=batches[target_model_id],
                target_adapter=adapters[target_model_id],
                shared_geometry_path=str(Path(shared_geometry_path).resolve()) if shared_geometry_path is not None else "",
                shared_layer=shared_layer,
                source_layer_idx=resolve_model_layer_index(
                    shared_geometry=shared_geometry,
                    shared_layer=shared_layer,
                    model_num_layers=int(getattr(adapters[source_model_id], "num_layers", shared_layer)),
                ) if representation_mode == "layer_last_hidden_continuation" else None,
                target_layer_idx=resolve_model_layer_index(
                    shared_geometry=shared_geometry,
                    shared_layer=shared_layer,
                    model_num_layers=int(getattr(adapters[target_model_id], "num_layers", shared_layer)),
                ) if representation_mode == "layer_last_hidden_continuation" else None,
                train_fraction=train_fraction,
                seed=seed,
                ridge_lambda=ridge_lambda,
                batch_size=batch_size,
                max_length=max_length,
                representation_mode=representation_mode,
            )
            safe_source = source_model_id.replace("/", "__")
            safe_target = target_model_id.replace("/", "__")
            report_path = output_dir / f"{safe_source}__to__{safe_target}.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            pairwise.append(
                {
                    "source_model_id": source_model_id,
                    "target_model_id": target_model_id,
                    "report_path": str(report_path),
                    "eval_target_logit_kl_mean": float(report["metrics"]["eval"]["target_logit_kl_mean"]),
                    "eval_target_logit_js_mean": float(report["metrics"]["eval"]["target_logit_js_mean"]),
                    "eval_hidden_cosine_mean": float(report["metrics"]["eval"]["hidden_cosine_mean"]),
                    "eval_target_top1_agreement": float(report["metrics"]["eval"]["target_top1_agreement"]),
                }
            )

    best_source_for_target: dict[str, dict[str, Any]] = {}
    by_target: dict[str, list[dict[str, Any]]] = {}
    for record in pairwise:
        by_target.setdefault(str(record["target_model_id"]), []).append(record)
    for target_model_id, records_for_target in by_target.items():
        best = min(
            records_for_target,
            key=lambda item: (
                float(item["eval_target_logit_kl_mean"]),
                -float(item["eval_target_top1_agreement"]),
                str(item["source_model_id"]),
            ),
        )
        best_source_for_target[target_model_id] = {
            "source_model_id": str(best["source_model_id"]),
            "eval_target_logit_kl_mean": float(best["eval_target_logit_kl_mean"]),
            "eval_hidden_cosine_mean": float(best["eval_hidden_cosine_mean"]),
            "eval_target_top1_agreement": float(best["eval_target_top1_agreement"]),
        }

    summary = {
        "shared_geometry_path": str(Path(shared_geometry_path).resolve()) if shared_geometry_path is not None else None,
        "calibration_jsonl": str(Path(calibration_jsonl).resolve()),
        "shared_layer": int(shared_layer),
        "num_examples": len(selected_chunk_ids),
        "pair_count": len(pairwise),
        "mean_eval_target_logit_kl_mean": float(np.mean([record["eval_target_logit_kl_mean"] for record in pairwise])) if pairwise else None,
        "mean_eval_hidden_cosine_mean": float(np.mean([record["eval_hidden_cosine_mean"] for record in pairwise])) if pairwise else None,
        "mean_eval_target_top1_agreement": float(np.mean([record["eval_target_top1_agreement"] for record in pairwise])) if pairwise else None,
        "best_source_for_target": best_source_for_target,
    }
    cohort_report = {
        "source_models": shared_geometry.source_models,
        "selected_chunk_ids": selected_chunk_ids,
        "pairwise": pairwise,
        "summary": summary,
        "metadata": {
            "method": "last_hidden_affine_stitching_cohort" if representation_mode == "final_last_hidden" else "layer_last_hidden_continuation_stitching_cohort",
            "representation_source": "final_last_hidden_only" if representation_mode == "final_last_hidden" else "source_last_position_hidden_at_selected_layer",
            "target_evaluation": "target_lm_head_projection_only" if representation_mode == "final_last_hidden" else "target_remainder_execution_to_logits",
            "shared_layer_role": "chunk_selection_and_metadata_only" if representation_mode == "final_last_hidden" else "relative_depth_and_layer_mapping",
            "representation_mode": representation_mode,
            "train_fraction": float(train_fraction),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "max_length": int(max_length) if max_length is not None else None,
            "ridge_lambda": float(ridge_lambda),
            "include_self_pairs": bool(include_self_pairs),
        },
    }
    summary_path = output_dir / "cohort_summary.json"
    summary_path.write_text(json.dumps(cohort_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cohort_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise stitching verification across all ordered model pairs in a SharedLatentGeometry cohort.")
    parser.add_argument("output_dir", help="Output directory for pairwise stitching reports")
    parser.add_argument("--shared-geometry", required=True, help="SharedLatentGeometry artifact defining the cohort")
    parser.add_argument("--calibration-jsonl", required=True, help="Calibration JSONL with chunk_id/text fields")
    parser.add_argument("--shared-layer", type=int, default=None, help="Optional shared-geometry layer to evaluate")
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
        default="final_last_hidden",
        help="Whether to stitch final hidden states directly or continue the target from a selected layer after replacing its last-position hidden state",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_stitching_cohort_report(
        shared_geometry_path=args.shared_geometry,
        calibration_jsonl=args.calibration_jsonl,
        output_dir=args.output_dir,
        shared_layer=args.shared_layer,
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
