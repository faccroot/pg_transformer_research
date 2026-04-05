#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    p = argparse.ArgumentParser(description="Analyze operator correlation in JEPA sidecar states.")
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
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--families", default="operators", help="Comma list or preset: operators|all|control|nsm")
    return p.parse_args()


def point_biserial(effect_mask: np.ndarray, values: np.ndarray) -> float:
    mask = np.asarray(effect_mask, dtype=np.bool_)
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or mask.size != vals.size:
        return 0.0
    p = float(mask.mean())
    q = 1.0 - p
    if p <= 0.0 or q <= 0.0:
        return 0.0
    mu1 = float(vals[mask].mean())
    mu0 = float(vals[~mask].mean())
    sigma = float(vals.std())
    if sigma <= 1e-12:
        return 0.0
    return ((mu1 - mu0) / sigma) * math.sqrt(p * q)


def summarize_group(mask: np.ndarray, matrix: np.ndarray, top_k: int) -> dict[str, object]:
    mask = np.asarray(mask, dtype=np.bool_)
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {mat.shape}")
    total = int(mat.shape[0])
    positives = int(mask.sum())
    negatives = total - positives
    out: dict[str, object] = {
        "samples": total,
        "positive_samples": positives,
        "negative_samples": negatives,
    }
    if positives <= 0 or negatives <= 0:
        out["available"] = False
        return out
    pos = mat[mask]
    neg = mat[~mask]
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    delta = pos_mean - neg_mean
    abs_delta = np.abs(delta)
    corr = np.array([point_biserial(mask, mat[:, idx]) for idx in range(mat.shape[1])], dtype=np.float32)
    top_delta_idx = np.argsort(-abs_delta)[:top_k]
    top_corr_idx = np.argsort(-np.abs(corr))[:top_k]
    out["available"] = True
    out["mean_abs_delta"] = float(abs_delta.mean())
    out["top_by_abs_delta"] = [
        {
            "dim": int(idx),
            "delta": float(delta[idx]),
            "positive_mean": float(pos_mean[idx]),
            "negative_mean": float(neg_mean[idx]),
        }
        for idx in top_delta_idx
    ]
    out["top_by_abs_corr"] = [
        {
            "dim": int(idx),
            "corr": float(corr[idx]),
            "positive_mean": float(pos_mean[idx]),
            "negative_mean": float(neg_mean[idx]),
        }
        for idx in top_corr_idx
    ]
    return out


def main() -> None:
    args = parse_args()
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
    chunk_size = int(model.sidecar_chunk_size)
    if seq_len % chunk_size != 0:
        raise ValueError(f"train_seq_len={seq_len} must be divisible by chunk_size={chunk_size}")
    num_seqs = (val_tokens.size - 1) // seq_len

    all_states: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []
    family_any: dict[str, list[np.ndarray]] = {name: [] for name in family_names}
    family_count: dict[str, list[np.ndarray]] = {name: [] for name in family_names}

    for seq_idx in range(num_seqs):
        raw_start = seq_idx * seq_len
        chunk = val_tokens[raw_start : raw_start + seq_len + 1]
        x_np = chunk[:-1].reshape(1, seq_len).astype(np.int32, copy=False)
        raw_codes = detect_family_codes_np(x_np, routing)
        x = mx.array(x_np, dtype=mx.int32)
        _final_hidden, _captured, aux = model.forward_hidden_with_aux(x)
        side_states = np.asarray(aux["sidecar_states"].astype(mx.float32), dtype=np.float32)[0]
        if side_states.shape[0] <= 0:
            continue
        code_chunks = raw_codes.reshape(1, seq_len // chunk_size, chunk_size)[0]
        deltas = np.diff(side_states, axis=0, prepend=np.zeros_like(side_states[:1]))

        all_states.append(side_states)
        all_deltas.append(deltas)
        for code_id, name in enumerate(family_names, start=1):
            family_any[name].append(np.any(code_chunks == code_id, axis=1))
            family_count[name].append(np.sum(code_chunks == code_id, axis=1).astype(np.int32))

    if not all_states:
        raise SystemExit("No sidecar states collected")

    state_mat = np.concatenate(all_states, axis=0)
    delta_mat = np.concatenate(all_deltas, axis=0)
    family_any_flat = {name: np.concatenate(parts, axis=0) for name, parts in family_any.items()}
    family_count_flat = {name: np.concatenate(parts, axis=0) for name, parts in family_count.items()}

    family_results: dict[str, object] = {}
    family_counts_json: dict[str, int] = {}
    for name in family_names:
        mask = family_any_flat[name]
        family_results[name] = {
            "state": summarize_group(mask, state_mat, args.top_k),
            "delta": summarize_group(mask, delta_mat, args.top_k),
        }
        family_counts_json[f"{name}_any"] = int(mask.sum())
        family_counts_json[f"{name}_total_mentions"] = int(family_count_flat[name].sum())

    result = {
        "artifact": str(Path(args.artifact).resolve()),
        "tokenizer_path": os.path.expanduser(args.tokenizer_path),
        "data_path": os.path.expanduser(args.data_path),
        "val_max_seqs": int(args.val_max_seqs),
        "train_seq_len": int(seq_len),
        "sidecar_chunk_size": int(chunk_size),
        "families": list(family_names),
        "num_sequences": int(num_seqs),
        "num_chunks": int(state_mat.shape[0]),
        "state_dim": int(state_mat.shape[1]),
        "family_chunk_counts": family_counts_json,
        "state_norm": {
            "mean": float(np.linalg.norm(state_mat, axis=1).mean()),
            "max": float(np.linalg.norm(state_mat, axis=1).max()),
        },
        "delta_norm": {
            "mean": float(np.linalg.norm(delta_mat, axis=1).mean()),
            "max": float(np.linalg.norm(delta_mat, axis=1).max()),
        },
        "family_results": family_results,
    }

    for legacy_name in ("not", "and", "or"):
        if legacy_name in family_results:
            result[f"{legacy_name}_state"] = family_results[legacy_name]["state"]
            result[f"{legacy_name}_delta"] = family_results[legacy_name]["delta"]

    out_path = Path(args.result_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
