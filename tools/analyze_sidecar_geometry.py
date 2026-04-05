#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import eval_saved_sidecar as ess
from lexical_family_utils import build_family_routing_spec, detect_family_codes_np, parse_family_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze operator-conditioned geometric transforms in sidecar states.")
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--val-max-seqs", type=int, default=256)
    p.add_argument("--train-seq-len", type=int, default=512)
    p.add_argument("--eval-seq-len", type=int, default=0)
    p.add_argument("--persistent", type=int, default=None)
    p.add_argument("--mode", choices=("reset", "persistent", "both"), default=None)
    p.add_argument("--persist-group-seqs", type=int, default=1)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--tie-embeddings", type=int, default=0)
    p.add_argument("--logic-dim", type=int, default=0)
    p.add_argument("--logic-layer-index", type=int, default=4)
    p.add_argument("--logic-route-to-next-token", type=int, default=1)
    p.add_argument("--logic-operator-mode", default="not_only")
    p.add_argument("--polarity-detector-enabled", type=int, default=0)
    p.add_argument("--polarity-detector-layer-index", type=int, default=3)
    p.add_argument("--polarity-detector-hidden-dim", type=int, default=64)
    p.add_argument("--polarity-seed-blend", type=float, default=1.0)
    p.add_argument("--polarity-seed-weight", type=float, default=0.0)
    p.add_argument("--polarity-sparse-weight", type=float, default=0.0)
    p.add_argument("--polarity-smooth-weight", type=float, default=0.0)
    p.add_argument("--sidecar-polarity-write", type=int, default=0)
    p.add_argument("--sidecar-polarity-pool", default="max")
    p.add_argument("--sidecar-tap-layer", type=int, default=3)
    p.add_argument("--sidecar-state-dim", type=int, default=64)
    p.add_argument("--sidecar-chunk-size", type=int, default=8)
    p.add_argument("--sidecar-pred-weight", type=float, default=0.05)
    p.add_argument("--sidecar-pred-offset", type=int, default=1)
    p.add_argument("--sidecar-sigreg-weight", type=float, default=0.01)
    p.add_argument("--sidecar-spherical-weight", type=float, default=0.01)
    p.add_argument("--sidecar-sigreg-mode", default="full")
    p.add_argument("--sidecar-weak-sigreg-dim", type=int, default=32)
    p.add_argument("--sidecar-read-rmsnorm", type=int, default=1)
    p.add_argument("--sidecar-summary-mode", default="query")
    p.add_argument("--sidecar-pred-target-mode", default="delta")
    p.add_argument(
        "--trainer-module",
        default="train_gpt_mlx_jepa_sidecar",
        help="Sidecar trainer module to instantiate, e.g. train_gpt_mlx_jepa_sidecar or train_gpt_mlx_jepa_sidecar_ref.",
    )
    p.add_argument("--families", default="all", help="Comma list or preset: operators|all|control|nsm")
    p.add_argument("--after-offsets", default="0,1,2,4,8", help="Comma-separated offsets after operator token.")
    p.add_argument("--sign-threshold", type=float, default=0.10, help="Absolute-value threshold for sign-flip analysis.")
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


def parse_offsets(raw: str) -> tuple[int, ...]:
    offsets: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"after offsets must be non-negative, got {value}")
        offsets.append(value)
    if not offsets:
        raise ValueError("Need at least one after offset")
    return tuple(sorted(set(offsets)))


def mean_centered_radius(mat: np.ndarray) -> float:
    if mat.shape[0] <= 0:
        return 0.0
    centered = mat - mat.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centered, axis=1).mean())


def summarize_transform(
    before: np.ndarray,
    after: np.ndarray,
    *,
    sign_threshold: float,
    top_k: int,
) -> dict[str, object]:
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        raise ValueError(f"expected matching 2D matrices, got before={before.shape}, after={after.shape}")
    samples, dims = before.shape
    result: dict[str, object] = {"samples": int(samples), "state_dim": int(dims)}
    if samples <= 0:
        result["available"] = False
        return result

    delta = after - before
    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    delta_norm = np.linalg.norm(delta, axis=1)
    cosine = np.sum(before * after, axis=1) / np.clip(before_norm * after_norm, 1e-8, None)

    before_var = before.var(axis=0)
    after_var = after.var(axis=0)
    var_ratio = after_var / np.clip(before_var, 1e-8, None)

    before_abs = np.abs(before)
    after_abs = np.abs(after)
    active = (before_abs >= sign_threshold) & (after_abs >= sign_threshold)
    flips = active & ((before * after) < 0.0)
    active_fraction = active.mean(axis=0)
    flip_fraction_overall = flips.mean(axis=0)
    flip_fraction_when_active = flip_fraction_overall / np.clip(active_fraction, 1e-8, None)

    before_mean = before.mean(axis=0)
    after_mean = after.mean(axis=0)
    before_abs_mean = before_abs.mean(axis=0)
    after_abs_mean = after_abs.mean(axis=0)
    mean_delta = delta.mean(axis=0)
    abs_mean_delta = np.abs(mean_delta)

    top_flip_idx = np.argsort(-flip_fraction_overall)[:top_k]
    top_delta_idx = np.argsort(-abs_mean_delta)[:top_k]
    top_var_idx = np.argsort(-var_ratio)[:top_k]

    result.update(
        {
            "available": True,
            "mean_cosine": float(cosine.mean()),
            "mean_delta_norm": float(delta_norm.mean()),
            "mean_norm_before": float(before_norm.mean()),
            "mean_norm_after": float(after_norm.mean()),
            "mean_norm_change": float((after_norm - before_norm).mean()),
            "variance_trace_before": float(before_var.sum()),
            "variance_trace_after": float(after_var.sum()),
            "variance_trace_ratio": float(after_var.sum() / max(float(before_var.sum()), 1e-8)),
            "mean_centered_radius_before": mean_centered_radius(before),
            "mean_centered_radius_after": mean_centered_radius(after),
            "spread_ratio": float(mean_centered_radius(after) / max(mean_centered_radius(before), 1e-8)),
            "mean_active_flip_fraction_per_sample": float(flips.mean(axis=1).mean()),
            "top_by_flip_fraction": [
                {
                    "dim": int(idx),
                    "flip_fraction_overall": float(flip_fraction_overall[idx]),
                    "flip_fraction_when_active": float(flip_fraction_when_active[idx]),
                    "active_fraction": float(active_fraction[idx]),
                    "before_mean": float(before_mean[idx]),
                    "after_mean": float(after_mean[idx]),
                    "before_abs_mean": float(before_abs_mean[idx]),
                    "after_abs_mean": float(after_abs_mean[idx]),
                }
                for idx in top_flip_idx
            ],
            "top_by_abs_mean_delta": [
                {
                    "dim": int(idx),
                    "mean_delta": float(mean_delta[idx]),
                    "before_mean": float(before_mean[idx]),
                    "after_mean": float(after_mean[idx]),
                    "before_abs_mean": float(before_abs_mean[idx]),
                    "after_abs_mean": float(after_abs_mean[idx]),
                }
                for idx in top_delta_idx
            ],
            "top_by_variance_ratio": [
                {
                    "dim": int(idx),
                    "variance_ratio": float(var_ratio[idx]),
                    "variance_before": float(before_var[idx]),
                    "variance_after": float(after_var[idx]),
                    "before_mean": float(before_mean[idx]),
                    "after_mean": float(after_mean[idx]),
                }
                for idx in top_var_idx
            ],
        }
    )
    return result


def main() -> None:
    args = parse_args()
    after_offsets = parse_offsets(args.after_offsets)

    ess.set_env(args)
    os.environ["SIDECAR_STATE_DIM"] = str(int(args.sidecar_state_dim))
    os.environ["SIDECAR_CHUNK_SIZE"] = str(int(args.sidecar_chunk_size))

    base_mod, sidecar_mod = ess.load_modules(args.trainer_module)
    hps = sidecar_mod.Hyperparameters()
    ess.ensure_dataset_ready(args)
    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(args.tokenizer_path))
    flat_state = ess.load_flat_state(Path(args.artifact), base_mod)
    model = sidecar_mod.make_sidecar_gpt(hps, sp)
    model.update(tree_unflatten(list(flat_state.items())))
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()

    family_names = parse_family_list(args.families)
    routing = build_family_routing_spec(sp, hps.vocab_size, family_names)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        args.val_max_seqs,
    )

    seq_len = hps.train_seq_len
    num_seqs = (val_tokens.size - 1) // seq_len

    pair_before: dict[str, dict[int, list[np.ndarray]]] = {
        name: {offset: [] for offset in after_offsets} for name in family_names
    }
    pair_after: dict[str, dict[int, list[np.ndarray]]] = {
        name: {offset: [] for offset in after_offsets} for name in family_names
    }
    family_mentions = {name: 0 for name in family_names}
    state_granularity = "unknown"
    inferred_state_stride: int | None = None

    for seq_idx in range(num_seqs):
        raw_start = seq_idx * seq_len
        chunk = val_tokens[raw_start : raw_start + seq_len + 1]
        x_np = chunk[:-1].reshape(1, seq_len).astype(np.int32, copy=False)
        raw_codes = detect_family_codes_np(x_np, routing)[0]
        x = mx.array(x_np, dtype=mx.int32)
        _final_hidden, _captured, aux = model.forward_hidden_with_aux(x)
        side_states = np.asarray(aux["sidecar_states"].astype(mx.float32), dtype=np.float32)[0]
        state_len = int(side_states.shape[0])
        if state_len <= 0:
            continue
        if state_len == seq_len:
            current_granularity = "token"
            current_stride = 1
            zero_state = np.zeros_like(side_states[:1])
            prev_states = np.concatenate([zero_state, side_states[:-1]], axis=0)

            def before_state_for_positions(positions: np.ndarray) -> np.ndarray:
                return prev_states[positions]

            def after_state_for_positions(positions: np.ndarray, offset: int) -> np.ndarray:
                return side_states[positions + offset]

            def valid_positions(positions: np.ndarray, offset: int) -> np.ndarray:
                return positions[positions + offset < seq_len]
        elif seq_len % state_len == 0:
            current_granularity = "chunk"
            current_stride = seq_len // state_len
            zero_state = np.zeros_like(side_states[:1])
            prev_chunk_states = np.concatenate([zero_state, side_states[:-1]], axis=0)

            def before_state_for_positions(positions: np.ndarray) -> np.ndarray:
                chunk_idx = positions // current_stride
                return prev_chunk_states[chunk_idx]

            def after_state_for_positions(positions: np.ndarray, offset: int) -> np.ndarray:
                token_after = positions + offset
                chunk_after = token_after // current_stride
                return side_states[chunk_after]

            def valid_positions(positions: np.ndarray, offset: int) -> np.ndarray:
                return positions[positions + offset < seq_len]
        else:
            raise ValueError(f"unexpected side_states shape {side_states.shape}, seq_len={seq_len}")

        if state_granularity == "unknown":
            state_granularity = current_granularity
            inferred_state_stride = int(current_stride)
        elif state_granularity != current_granularity or inferred_state_stride != int(current_stride):
            raise ValueError(
                f"inconsistent sidecar state layout across sequences: "
                f"{state_granularity}/{inferred_state_stride} vs {current_granularity}/{current_stride}"
            )

        for code_id, name in enumerate(family_names, start=1):
            positions = np.flatnonzero(raw_codes == code_id)
            family_mentions[name] += int(positions.size)
            if positions.size <= 0:
                continue
            for offset in after_offsets:
                valid = valid_positions(positions, offset)
                if valid.size <= 0:
                    continue
                pair_before[name][offset].append(before_state_for_positions(valid))
                pair_after[name][offset].append(after_state_for_positions(valid, offset))

    family_results: dict[str, object] = {}
    total_pairs = 0
    for name in family_names:
        offset_results: dict[str, object] = {}
        for offset in after_offsets:
            if pair_before[name][offset]:
                before = np.concatenate(pair_before[name][offset], axis=0)
                after = np.concatenate(pair_after[name][offset], axis=0)
                total_pairs += int(before.shape[0])
            else:
                before = np.zeros((0, int(model.sidecar_state_dim)), dtype=np.float32)
                after = np.zeros_like(before)
            offset_results[str(offset)] = summarize_transform(
                before,
                after,
                sign_threshold=float(args.sign_threshold),
                top_k=int(args.top_k),
            )
        family_results[name] = {
            "mentions": int(family_mentions[name]),
            "offset_results": offset_results,
        }

    result = {
        "artifact": str(Path(args.artifact).resolve()),
        "trainer_module": args.trainer_module,
        "tokenizer_path": os.path.expanduser(args.tokenizer_path),
        "data_path": os.path.expanduser(args.data_path),
        "val_max_seqs": int(args.val_max_seqs),
        "train_seq_len": int(seq_len),
        "num_sequences": int(num_seqs),
        "state_granularity": state_granularity,
        "state_stride_tokens": int(inferred_state_stride or 0),
        "families": list(family_names),
        "after_offsets": list(after_offsets),
        "sign_threshold": float(args.sign_threshold),
        "num_pairs_total": int(total_pairs),
        "family_results": family_results,
    }

    out_path = Path(args.result_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
