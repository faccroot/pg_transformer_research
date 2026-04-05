#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "tools" else SCRIPT_PATH.parent
TOOLS_ROOT = REPO_ROOT / "tools"
PATH_CANDIDATES = [
    REPO_ROOT,
    TOOLS_ROOT,
    Path.home() / "transformer_research" / "parameter-golf",
    Path.home() / "transformer_research" / "parameter-golf" / "tools",
]
for root in PATH_CANDIDATES:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze hardmax structural-controller behavior on a saved artifact.")
    p.add_argument("--artifact", required=True, help="Path to *_mlx_model.npz or *_int8zlib.pklz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument("--result-json", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--data-path", default="")
    p.add_argument("--val-max-seqs", type=int, default=None)
    p.add_argument("--eval-seq-len", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=8)
    p.add_argument("--top-k-states", type=int, default=8)
    p.add_argument("--repo-bundle", default="", help="Optional tar/tgz of current repo Python modules for remote parity")
    return p.parse_args()


def maybe_stage_repo_bundle(bundle_path: str) -> Path | None:
    raw = bundle_path.strip()
    if not raw:
        return None
    bundle = Path(raw).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Repo bundle not found: {bundle}")
    temp_root = Path(tempfile.mkdtemp(prefix="hardmax_controller_bundle_"))
    with tarfile.open(bundle, "r:*") as tf:
        tf.extractall(temp_root)
    candidate_roots = [temp_root / "parameter-golf", temp_root]
    for root in candidate_roots:
        if root.exists():
            tools_root = root / "tools"
            for path in (root, tools_root):
                if path.exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return root
    return temp_root


def load_model_modules():
    import train_gpt_mlx as base_mod

    return importlib.reload(base_mod)


def load_support_modules():
    import eval_saved_structural as evs
    from logic_register_mlx import strip_register_positions
    from text_prosody_features import BOUNDARY_STRENGTH_NAMES, extract_text_prosody_features

    return evs, strip_register_positions, BOUNDARY_STRENGTH_NAMES, extract_text_prosody_features


def build_model(base_mod, sp: spm.SentencePieceProcessor):
    hps = base_mod.Hyperparameters()
    model = base_mod.make_gpt(hps, sp)
    return hps, model


def summarize_masked_scalar(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if arr.shape[0] != keep.shape[0]:
        raise ValueError(f"value count {arr.shape[0]} does not match mask length {keep.shape[0]}")
    selected = arr[keep]
    return {
        "count": int(selected.size),
        "mean": None if selected.size <= 0 else float(selected.mean()),
        "std": None if selected.size <= 0 else float(selected.std()),
        "sum": None if selected.size <= 0 else float(selected.sum()),
    }


def summarize_grouped_scalar(
    values: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    if arr.shape[0] != lab.shape[0]:
        raise ValueError(f"value count {arr.shape[0]} does not match label count {lab.shape[0]}")
    total_sum = float(arr.sum()) if arr.size > 0 else 0.0
    rows: dict[str, object] = {}
    for idx, name in enumerate(label_names):
        mask = lab == int(idx)
        stats = summarize_masked_scalar(arr, mask)
        total_share = 0.0
        if total_sum > 0.0 and stats["sum"] is not None:
            total_share = float(float(stats["sum"]) / total_sum)
        rows[str(name)] = {
            "count": int(stats["count"]),
            "fraction": float(mask.mean()) if mask.size > 0 else 0.0,
            "mean": stats["mean"],
            "std": stats["std"],
            "share": total_share,
        }
    return rows


def summarize_boundary_conditioned_scalar(values: np.ndarray, prev_boundary_ids: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    prev_boundary = np.asarray(prev_boundary_ids, dtype=np.int32).reshape(-1)
    _evs, _strip_register_positions, boundary_strength_names, _extract_text_prosody_features = load_support_modules()
    exact = summarize_grouped_scalar(arr, prev_boundary, boundary_strength_names)
    cumulative: dict[str, object] = {}
    for idx, name in enumerate(boundary_strength_names[1:], start=1):
        mask = prev_boundary >= int(idx)
        stats = summarize_masked_scalar(arr, mask)
        cumulative[f"after_{name}"] = {
            "count": int(stats["count"]),
            "fraction": float(mask.mean()) if mask.size > 0 else 0.0,
            "mean": stats["mean"],
            "std": stats["std"],
        }
    return {
        "by_prev_boundary_exact": exact,
        "after_boundary_ge": cumulative,
    }


def safe_corrcoef(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    a = np.asarray(lhs, dtype=np.float32).reshape(-1)
    b = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if a.shape[0] != b.shape[0] or a.shape[0] <= 1:
        return None
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom <= 1.0e-8:
        return None
    return float((a_c @ b_c) / denom)


def summarize_quantile_buckets(values: np.ndarray, order_by: np.ndarray, prefix: str) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    score = np.asarray(order_by, dtype=np.float32).reshape(-1)
    if arr.shape[0] != score.shape[0] or arr.size <= 0:
        return {}
    quantiles = np.quantile(score, [0.25, 0.5, 0.75]).astype(np.float32)
    bucket_ids = np.digitize(score, quantiles, right=True).astype(np.int32)
    names = tuple(f"{prefix}_q{i}" for i in range(1, 5))
    return summarize_grouped_scalar(arr, bucket_ids, names)


def state_usage_summary(
    state_ids: np.ndarray,
    soft_usage: np.ndarray,
    nll: np.ndarray,
    confidence: np.ndarray,
    budget: np.ndarray,
    *,
    num_states: int,
    top_k_states: int,
) -> dict[str, object]:
    hard_ids = np.asarray(state_ids, dtype=np.int32).reshape(-1)
    soft = np.asarray(soft_usage, dtype=np.float32).reshape(-1, int(num_states))
    nll_arr = np.asarray(nll, dtype=np.float32).reshape(-1)
    conf_arr = np.asarray(confidence, dtype=np.float32).reshape(-1)
    budget_arr = np.asarray(budget, dtype=np.float32).reshape(-1)

    hard_counts = np.bincount(hard_ids, minlength=int(num_states)).astype(np.int64)
    hard_frac = hard_counts.astype(np.float64) / max(int(hard_counts.sum()), 1)
    soft_mean = soft.mean(axis=0) if soft.size > 0 else np.zeros((int(num_states),), dtype=np.float32)

    entropy_nats = float(-(hard_frac[hard_frac > 0.0] * np.log(hard_frac[hard_frac > 0.0])).sum()) if hard_frac.size > 0 else 0.0
    entropy_bits = float(entropy_nats / np.log(2.0)) if entropy_nats > 0.0 else 0.0
    perplexity = float(np.exp(entropy_nats)) if entropy_nats > 0.0 else 1.0

    per_state: list[dict[str, object]] = []
    for state_idx in range(int(num_states)):
        mask = hard_ids == state_idx
        nll_stats = summarize_masked_scalar(nll_arr, mask)
        conf_stats = summarize_masked_scalar(conf_arr, mask)
        budget_stats = summarize_masked_scalar(budget_arr, mask)
        per_state.append(
            {
                "state": int(state_idx),
                "count": int(hard_counts[state_idx]),
                "fraction": float(hard_frac[state_idx]),
                "soft_usage_mean": float(soft_mean[state_idx]),
                "mean_nll": nll_stats["mean"],
                "mean_confidence": conf_stats["mean"],
                "mean_budget": budget_stats["mean"],
            }
        )
    per_state.sort(key=lambda row: int(row["count"]), reverse=True)
    return {
        "num_states": int(num_states),
        "used_states": int((hard_counts > 0).sum()),
        "usage_entropy_bits": entropy_bits,
        "usage_perplexity": perplexity,
        "max_state_fraction": float(hard_frac.max(initial=0.0)),
        "hard_counts": [int(x) for x in hard_counts.tolist()],
        "hard_fractions": [float(x) for x in hard_frac.tolist()],
        "soft_usage_mean": [float(x) for x in soft_mean.tolist()],
        "top_states": per_state[: max(int(top_k_states), 1)],
        "all_states": per_state,
    }


def summarize_transitions(state_ids_batches: list[np.ndarray], num_states: int) -> dict[str, object]:
    matrix = np.zeros((int(num_states), int(num_states)), dtype=np.int64)
    self_count = 0
    total = 0
    segment_lengths: list[int] = []
    for state_ids in state_ids_batches:
        seq = np.asarray(state_ids, dtype=np.int32)
        if seq.ndim == 1:
            seq = seq[None, :]
        for row in seq:
            if row.size <= 1:
                if row.size == 1:
                    segment_lengths.append(1)
                continue
            run = 1
            for prev, cur in zip(row[:-1], row[1:]):
                matrix[int(prev), int(cur)] += 1
                total += 1
                if int(prev) == int(cur):
                    self_count += 1
                    run += 1
                else:
                    segment_lengths.append(run)
                    run = 1
            segment_lengths.append(run)
    return {
        "transition_count": int(total),
        "self_transition_fraction": float(self_count / total) if total > 0 else None,
        "mean_segment_length": None if not segment_lengths else float(np.mean(segment_lengths)),
        "median_segment_length": None if not segment_lengths else float(np.median(np.asarray(segment_lengths, dtype=np.float32))),
        "transition_matrix": matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    maybe_stage_repo_bundle(args.repo_bundle)
    evs, strip_register_positions, _boundary_strength_names, extract_text_prosody_features = load_support_modules()
    evs.apply_config_env(Path(args.config_json), args)
    evs.ensure_dataset_ready(args)
    base_mod = load_model_modules()

    artifact_path = Path(args.artifact).expanduser().resolve()
    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(os.environ["TOKENIZER_PATH"]))
    hps, model = build_model(base_mod, sp)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        hps.val_max_seqs,
    )

    if not hasattr(model, "has_hardmax_structural_controller") or not model.has_hardmax_structural_controller():
        raise ValueError("Model does not expose an active hardmax structural controller")

    model.set_turbo_qat(False, 0.0)
    flat_state = evs.load_flat_state(artifact_path, base_mod)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()

    flat_actual_ids: list[np.ndarray] = []
    flat_nll: list[np.ndarray] = []
    flat_confidence: list[np.ndarray] = []
    flat_budget: list[np.ndarray] = []
    flat_state_ids: list[np.ndarray] = []
    flat_soft_usage: list[np.ndarray] = []
    state_batches: list[np.ndarray] = []

    for batch_idx, (x_np, y_np) in enumerate(evs.iter_eval_batches(base_mod, hps, val_tokens), start=1):
        x = mx.array(x_np, dtype=mx.int32)
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        final_hidden, _captured, aux = model.forward_hidden_with_aux(
            x,
            capture_layers=(),
            operator_codes=operator_codes,
        )
        structural_aux = aux.get("hardmax_structural")
        if not isinstance(structural_aux, dict):
            raise ValueError("Hardmax structural aux outputs are unavailable for this artifact")
        logits = base_mod.logits_from_hidden(model, final_hidden) if hasattr(base_mod, "logits_from_hidden") else (
            final_hidden @ model.tok_emb.weight.astype(final_hidden.dtype).T
            if model.tie_embeddings
            else model.lm_head(final_hidden)
        )
        logits = model.softcap(logits)
        nll = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            mx.array(y_np, dtype=mx.int32),
            reduction="none",
        ).astype(mx.float32)

        confidence = structural_aux["confidence"].astype(mx.float32)
        budget = structural_aux["budget"].astype(mx.float32)
        state_index = structural_aux["state_index"]
        soft_usage = structural_aux["soft_usage"].astype(mx.float32)

        if getattr(model, "num_registers", 0) > 0:
            layout = getattr(model, "register_layout", "prefix")
            stride = getattr(model, "register_stride", 0)
            confidence = strip_register_positions(confidence, model.num_registers, layout=layout, register_stride=stride)
            budget = strip_register_positions(budget, model.num_registers, layout=layout, register_stride=stride)
            state_index = strip_register_positions(state_index, model.num_registers, layout=layout, register_stride=stride)
            soft_usage = strip_register_positions(soft_usage, model.num_registers, layout=layout, register_stride=stride)

        flat_actual_ids.append(y_np.reshape(-1).astype(np.int32, copy=False))
        flat_nll.append(np.asarray(mx.stop_gradient(nll), dtype=np.float32).reshape(-1))
        flat_confidence.append(np.asarray(mx.stop_gradient(confidence), dtype=np.float32).reshape(-1))
        flat_budget.append(np.asarray(mx.stop_gradient(budget), dtype=np.float32).reshape(-1))
        flat_state_ids.append(np.asarray(mx.stop_gradient(state_index), dtype=np.int32).reshape(-1))
        flat_soft_usage.append(np.asarray(mx.stop_gradient(soft_usage), dtype=np.float32).reshape(-1, soft_usage.shape[-1]))
        state_batches.append(np.asarray(mx.stop_gradient(state_index), dtype=np.int32))

        if args.analysis_max_batches > 0 and batch_idx >= args.analysis_max_batches:
            break

    actual_ids = np.concatenate(flat_actual_ids, axis=0)
    nll_flat = np.concatenate(flat_nll, axis=0)
    confidence_flat = np.concatenate(flat_confidence, axis=0)
    budget_flat = np.concatenate(flat_budget, axis=0)
    state_ids_flat = np.concatenate(flat_state_ids, axis=0)
    soft_usage_flat = np.concatenate(flat_soft_usage, axis=0)

    prosody = extract_text_prosody_features(sp, actual_ids)
    prev_boundary = prosody.prev_boundary_strength_ids

    usage = state_usage_summary(
        state_ids_flat,
        soft_usage_flat,
        nll_flat,
        confidence_flat,
        budget_flat,
        num_states=soft_usage_flat.shape[-1],
        top_k_states=args.top_k_states,
    )
    transitions = summarize_transitions(state_batches, soft_usage_flat.shape[-1])

    result = {
        "label": args.label,
        "artifact": str(artifact_path),
        "config_json": str(Path(args.config_json).expanduser().resolve()),
        "analysis_params": {
            "analysis_max_batches": int(args.analysis_max_batches),
            "top_k_states": int(args.top_k_states),
        },
        "positions_analyzed": int(nll_flat.shape[0]),
        "mean_nll": float(nll_flat.mean()) if nll_flat.size > 0 else None,
        "controller": {
            "confidence_mean": float(confidence_flat.mean()) if confidence_flat.size > 0 else None,
            "confidence_std": float(confidence_flat.std()) if confidence_flat.size > 0 else None,
            "budget_mean": float(budget_flat.mean()) if budget_flat.size > 0 else None,
            "budget_std": float(budget_flat.std()) if budget_flat.size > 0 else None,
            "corr_confidence_vs_nll": safe_corrcoef(confidence_flat, nll_flat),
            "corr_budget_vs_nll": safe_corrcoef(budget_flat, nll_flat),
            "corr_confidence_vs_prev_boundary": safe_corrcoef(confidence_flat, prev_boundary),
            "corr_budget_vs_prev_boundary": safe_corrcoef(budget_flat, prev_boundary),
            "nll_by_confidence_quantile": summarize_quantile_buckets(nll_flat, confidence_flat, "conf"),
            "nll_by_budget_quantile": summarize_quantile_buckets(nll_flat, budget_flat, "budget"),
            "confidence_by_prev_boundary": summarize_boundary_conditioned_scalar(confidence_flat, prev_boundary),
            "budget_by_prev_boundary": summarize_boundary_conditioned_scalar(budget_flat, prev_boundary),
            "nll_by_prev_boundary": summarize_boundary_conditioned_scalar(nll_flat, prev_boundary),
        },
        "state_usage": usage,
        "state_transitions": transitions,
    }

    result_path = Path(args.result_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
