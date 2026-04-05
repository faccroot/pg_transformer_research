#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def parse_artifact_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"artifact spec must look like label=/path/to/artifact.npz, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"artifact spec must contain non-empty label and path, got {spec!r}")
    return label, path


def summarize_paired_deltas(
    baseline_batches: list[float],
    candidate_batches: list[float],
) -> dict[str, float | int]:
    if len(baseline_batches) != len(candidate_batches):
        raise ValueError(
            f"paired summaries require equal-length batch lists, got {len(baseline_batches)} and {len(candidate_batches)}"
        )
    deltas = [candidate - baseline for baseline, candidate in zip(baseline_batches, candidate_batches)]
    wins = sum(1 for delta in deltas if delta < 0.0)
    losses = sum(1 for delta in deltas if delta > 0.0)
    ties = len(deltas) - wins - losses
    mean_delta = sum(deltas) / max(len(deltas), 1)
    variance = sum((delta - mean_delta) ** 2 for delta in deltas) / max(len(deltas), 1)
    return {
        "wins_vs_baseline": wins,
        "losses_vs_baseline": losses,
        "ties_vs_baseline": ties,
        "mean_delta_bpb_vs_baseline": mean_delta,
        "std_delta_bpb_vs_baseline": math.sqrt(max(variance, 0.0)),
    }


def inspect_artifact(path: str | Path) -> dict[str, object]:
    payload = np.load(Path(path), allow_pickle=False)
    kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
    source_models: list[str] = []
    if "source_models" in payload:
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()]
    if kind == "model_representation":
        category = "single_source"
    elif kind == "platonic_geometry":
        category = "merged" if len(source_models) > 1 else "geometry"
    else:
        category = "unknown"
    return {
        "kind": kind,
        "category": category,
        "source_models": source_models,
        "num_source_models": len(source_models),
    }


def rank_results(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    ranking = []
    for label, payload in results.items():
        if label == "baseline":
            continue
        ranking.append(
            {
                "label": label,
                "mean_val_bpb": float(payload["mean_val_bpb"]),
            }
        )
    ranking.sort(key=lambda item: (float(item["mean_val_bpb"]), str(item["label"])))
    return ranking


def select_best_label(
    results: dict[str, dict[str, object]],
    artifact_details: dict[str, dict[str, object]],
    *,
    category: str,
) -> str | None:
    candidates = [
        label
        for label in results
        if label != "baseline" and str(artifact_details.get(label, {}).get("category")) == category
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda label: (float(results[label]["mean_val_bpb"]), label))


def summarize_against_reference(
    candidate_batches_by_label: dict[str, list[float]],
    *,
    reference_label: str,
) -> dict[str, dict[str, float | int]]:
    if reference_label not in candidate_batches_by_label:
        raise ValueError(f"reference label {reference_label!r} missing from candidate batches")
    reference_batches = candidate_batches_by_label[reference_label]
    summary: dict[str, dict[str, float | int]] = {}
    for label, candidate_batches in candidate_batches_by_label.items():
        if label == reference_label:
            continue
        paired = summarize_paired_deltas(reference_batches, candidate_batches)
        summary[label] = {
            "reference_label": reference_label,
            "wins_vs_reference": int(paired["wins_vs_baseline"]),
            "losses_vs_reference": int(paired["losses_vs_baseline"]),
            "ties_vs_reference": int(paired["ties_vs_baseline"]),
            "mean_delta_bpb_vs_reference": float(paired["mean_delta_bpb_vs_baseline"]),
            "std_delta_bpb_vs_reference": float(paired["std_delta_bpb_vs_baseline"]),
        }
    return summary


def apply_env_overrides(parsed: argparse.Namespace) -> None:
    env_map = {
        "DATA_PATH": parsed.data_path,
        "TOKENIZER_PATH": parsed.tokenizer_path,
        "VAL_BATCH_SIZE": str(parsed.val_batch_size),
        "VAL_MAX_SEQS": str(parsed.val_max_seqs),
        "TRAIN_SEQ_LEN": str(parsed.train_seq_len),
        "EVAL_SEQ_LEN": str(parsed.eval_seq_len),
        "EVAL_STRIDE": str(parsed.eval_stride),
        "EVAL_BATCH_SEQS": str(parsed.eval_batch_seqs),
        "VOCAB_SIZE": str(parsed.vocab_size),
        "NUM_LAYERS": str(parsed.num_layers),
        "NUM_LAYER_TEMPLATES": str(parsed.num_layer_templates),
        "MODEL_DIM": str(parsed.model_dim),
        "NUM_HEADS": str(parsed.num_heads),
        "NUM_KV_HEADS": str(parsed.num_kv_heads),
        "MLP_MULT": str(parsed.mlp_mult),
        "TIE_EMBEDDINGS": str(int(parsed.tie_embeddings)),
        "LOGIT_CHUNK_TOKENS": str(parsed.logit_chunk_tokens),
        "SEED": str(parsed.seed),
    }
    for key, value in env_map.items():
        os.environ[key] = value


def eval_batches(
    base,
    mx,
    args,
    model,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
) -> list[float]:
    if args.eval_stride > 0:
        raise ValueError("paired batch summaries are only implemented for non-overlapping eval mode")
    eval_seq_len = args.effective_eval_seq_len
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < eval_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one eval sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"EVAL_SEQ_LEN={eval_seq_len}"
        )
    val_batch_seqs = max(val_batch_tokens // eval_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    batch_bpbs: list[float] = []
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * eval_seq_len
        raw_end = batch_seq_end * eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, eval_seq_len)
        y_np = chunk[1:].reshape(-1, eval_seq_len)
        operator_codes = base.operator_codes_mx_for_numpy_batch(model, x_np)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        batch_loss = model.ce_loss(x, y, operator_codes).astype(mx.float32)
        mx.eval(batch_loss)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype("int16", copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype("int16", copy=False)
        total_tokens = float(y.size)
        total_bytes = float(bytes_np.astype("float64").sum())
        bits_per_token = float(batch_loss.item()) / math.log(2.0)
        batch_bpbs.append(bits_per_token * (total_tokens / total_bytes))
    return batch_bpbs


def evaluate_artifact(
    *,
    base,
    mx,
    args,
    sp,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    priors_path: str | None,
    init_strength: float,
    init_targets: str,
    adapter_mode: str,
) -> dict[str, object]:
    try:
        from tools.representation_learning.runtime_mlx import apply_priors
    except ModuleNotFoundError as exc:
        if exc.name != "tools":
            raise
        from runtime_mlx import apply_priors  # type: ignore[no-redef]

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model = base.make_gpt(args, sp)
    prior_application = None
    if priors_path is not None:
        prior_application = apply_priors(
            model,
            priors_path=priors_path,
            strength=init_strength,
            targets=init_targets,
            adapter_mode=adapter_mode,
        )

    if model.logic_sidecar is not None:
        compiled_loss = lambda x, y, operator_codes: model.ce_loss(x, y, operator_codes)
        compiled_forward_logits = lambda x, operator_codes: model.forward_logits(x, operator_codes)
    else:
        compiled_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y)
        compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits(x)

    val_loss, val_bpb = base.eval_val(
        args,
        model,
        compiled_loss,
        compiled_forward_logits if args.eval_stride > 0 else None,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    batch_bpbs = eval_batches(
        base,
        mx,
        args,
        model,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    return {
        "priors_path": priors_path,
        "prior_application": prior_application,
        "mean_val_loss": float(val_loss),
        "mean_val_bpb": float(val_bpb),
        "num_batches": len(batch_bpbs),
        "batch_bpbs": batch_bpbs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate representation-learning priors on a real MLX GPT architecture at initialization time.",
    )
    parser.add_argument("--output", required=True, help="Path to write the JSON report.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact to evaluate, as label=/absolute/path/to/artifact.npz. Can be repeated.",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--val-batch-size", type=int, default=131072)
    parser.add_argument("--val-max-seqs", type=int, default=256)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-layer-templates", type=int, default=9)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=3)
    parser.add_argument("--tie-embeddings", action="store_true", default=False)
    parser.add_argument("--logit-chunk-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--init-strength", type=float, default=0.5)
    parser.add_argument("--init-targets", default="qk")
    parser.add_argument("--adapter-mode", default="random")
    return parser


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()
    apply_env_overrides(parsed)

    import sentencepiece as spm
    import mlx.core as mx
    import train_gpt_mlx as base

    artifact_specs = [parse_artifact_spec(spec) for spec in parsed.artifact]
    if not artifact_specs:
        raise SystemExit("Provide at least one --artifact label=/path/to/artifact.npz")

    args = base.Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise SystemExit(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    val_seq_len = max(args.train_seq_len, args.effective_eval_seq_len)
    val_tokens = base.limit_validation_tokens(
        base.load_validation_tokens(args.val_files, val_seq_len),
        args.effective_eval_seq_len,
        args.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size
    )

    mx.random.seed(args.seed)

    baseline = evaluate_artifact(
        base=base,
        mx=mx,
        args=args,
        sp=sp,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        priors_path=None,
        init_strength=parsed.init_strength,
        init_targets=parsed.init_targets,
        adapter_mode=parsed.adapter_mode,
    )
    baseline_batch_bpbs = baseline.pop("batch_bpbs")

    results: dict[str, object] = {"baseline": baseline}
    paired: dict[str, object] = {}
    candidate_batches_by_label: dict[str, list[float]] = {"baseline": baseline_batch_bpbs}
    artifact_details: dict[str, dict[str, object]] = {}

    for label, artifact_path in artifact_specs:
        artifact_details[label] = inspect_artifact(artifact_path)
        evaluated = evaluate_artifact(
            base=base,
            mx=mx,
            args=args,
            sp=sp,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            priors_path=artifact_path,
            init_strength=parsed.init_strength,
            init_targets=parsed.init_targets,
            adapter_mode=parsed.adapter_mode,
        )
        candidate_batch_bpbs = evaluated.pop("batch_bpbs")
        results[label] = evaluated
        candidate_batches_by_label[label] = candidate_batch_bpbs
        paired[label] = summarize_paired_deltas(baseline_batch_bpbs, candidate_batch_bpbs)

    ranking = rank_results(results)
    best_single_source_label = select_best_label(results, artifact_details, category="single_source")
    best_merged_label = select_best_label(results, artifact_details, category="merged")
    paired_vs_best_single_source = (
        summarize_against_reference(candidate_batches_by_label, reference_label=best_single_source_label)
        if best_single_source_label is not None
        else {}
    )
    paired_vs_best_merged = (
        summarize_against_reference(candidate_batches_by_label, reference_label=best_merged_label)
        if best_merged_label is not None
        else {}
    )

    output = {
        "artifacts": [
            {
                "label": label,
                "path": path,
                **artifact_details.get(label, {}),
            }
            for label, path in artifact_specs
        ],
        "data_path": args.data_path,
        "tokenizer_path": args.tokenizer_path,
        "val_batch_size": args.val_batch_size,
        "val_max_seqs": args.val_max_seqs,
        "train_seq_len": args.train_seq_len,
        "eval_seq_len": args.effective_eval_seq_len,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "init_strength": parsed.init_strength,
        "init_targets": parsed.init_targets,
        "adapter_mode": parsed.adapter_mode,
        "ranking": ranking,
        "best_single_source_label": best_single_source_label,
        "best_merged_label": best_merged_label,
        "results": results,
        "paired_vs_baseline": paired,
        "paired_vs_best_single_source": paired_vs_best_single_source,
        "paired_vs_best_merged": paired_vs_best_merged,
    }
    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
