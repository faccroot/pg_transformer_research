#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import zlib
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
    p = argparse.ArgumentParser(description="Analyze token NLL near operator positions for saved MLX artifacts.")
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--model-kind", choices=("baseline", "sidecar", "sidecar_ref"), required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--val-max-seqs", type=int, default=256)
    p.add_argument("--train-seq-len", type=int, default=512)
    p.add_argument("--eval-seq-len", type=int, default=0)
    p.add_argument("--max-distance", type=int, default=8)
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
        help="Sidecar trainer module to instantiate when model-kind is sidecar/sidecar_ref.",
    )
    p.add_argument("--families", default="operators", help="Comma list or preset: operators|all|control|nsm")
    return p.parse_args()


def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_z


def distances_after_code(codes: np.ndarray, code_id: int, max_distance: int) -> np.ndarray:
    out = np.full(codes.shape, -1, dtype=np.int16)
    last = -10**9
    for idx, code in enumerate(codes.tolist()):
        if int(code) == code_id:
            last = idx
        dist = idx - last
        if 0 <= dist <= max_distance:
            out[idx] = dist
    return out


def init_bucket(family_names: tuple[str, ...], max_distance: int) -> dict[str, object]:
    return {
        "overall": {"count": 0, "nll_sum": 0.0},
        "after_any_family": {"count": 0, "nll_sum": 0.0},
        "background": {"count": 0, "nll_sum": 0.0},
        "families": {
            name: {
                "mentions": 0,
                "window_count": 0,
                "window_nll_sum": 0.0,
                "distances": {
                    str(d): {"count": 0, "nll_sum": 0.0}
                    for d in range(max_distance + 1)
                },
            }
            for name in family_names
        },
    }


def update_stats(
    stats: dict[str, object],
    family_names: tuple[str, ...],
    token_nll: np.ndarray,
    codes: np.ndarray,
    max_distance: int,
) -> None:
    token_nll = np.asarray(token_nll, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int32)
    stats["overall"]["count"] += int(token_nll.size)
    stats["overall"]["nll_sum"] += float(token_nll.sum())

    any_window = np.zeros(codes.shape, dtype=np.bool_)
    for code_id, family_name in enumerate(family_names, start=1):
        op_bucket = stats["families"][family_name]
        distances = distances_after_code(codes, code_id, max_distance)
        op_bucket["mentions"] += int(np.sum(codes == code_id))
        mask = distances >= 0
        any_window |= mask
        if np.any(mask):
            op_bucket["window_count"] += int(mask.sum())
            op_bucket["window_nll_sum"] += float(token_nll[mask].sum())
        for d in range(max_distance + 1):
            d_mask = distances == d
            if not np.any(d_mask):
                continue
            d_bucket = op_bucket["distances"][str(d)]
            d_bucket["count"] += int(d_mask.sum())
            d_bucket["nll_sum"] += float(token_nll[d_mask].sum())

    if np.any(any_window):
        stats["after_any_family"]["count"] += int(any_window.sum())
        stats["after_any_family"]["nll_sum"] += float(token_nll[any_window].sum())
    background = ~any_window
    if np.any(background):
        stats["background"]["count"] += int(background.sum())
        stats["background"]["nll_sum"] += float(token_nll[background].sum())


def finalize_stats(stats: dict[str, object], family_names: tuple[str, ...], max_distance: int) -> dict[str, object]:
    def finalize_bucket(bucket: dict[str, float | int]) -> dict[str, float | int | None]:
        count = int(bucket["count"])
        nll_sum = float(bucket["nll_sum"])
        return {
            "count": count,
            "nll_sum": nll_sum,
            "mean_nll": (nll_sum / count) if count > 0 else None,
        }

    out = {
        "overall": finalize_bucket(stats["overall"]),
        "after_any_family": finalize_bucket(stats["after_any_family"]),
        "background": finalize_bucket(stats["background"]),
        "families": {},
    }
    overall_mean = out["overall"]["mean_nll"]
    for family_name in family_names:
        op_bucket = stats["families"][family_name]
        op_out = {
            "mentions": int(op_bucket["mentions"]),
            "window": finalize_bucket({
                "count": op_bucket["window_count"],
                "nll_sum": op_bucket["window_nll_sum"],
            }),
            "distances": {},
        }
        if overall_mean is not None and op_out["window"]["mean_nll"] is not None:
            op_out["window"]["delta_vs_overall"] = op_out["window"]["mean_nll"] - overall_mean
        else:
            op_out["window"]["delta_vs_overall"] = None
        for d in range(max_distance + 1):
            d_bucket = finalize_bucket(op_bucket["distances"][str(d)])
            if overall_mean is not None and d_bucket["mean_nll"] is not None:
                d_bucket["delta_vs_overall"] = d_bucket["mean_nll"] - overall_mean
            else:
                d_bucket["delta_vs_overall"] = None
            op_out["distances"][str(d)] = d_bucket
        out["families"][family_name] = op_out
    for legacy_name in ("not", "and", "or"):
        if legacy_name in out["families"]:
            out.setdefault("operators", {})[legacy_name] = out["families"][legacy_name]
    return out


def load_model(args: argparse.Namespace):
    ess.set_env(args)
    os.environ["SIDECAR_STATE_DIM"] = str(int(args.sidecar_state_dim))
    os.environ["SIDECAR_CHUNK_SIZE"] = str(int(args.sidecar_chunk_size))
    if args.model_kind in {"sidecar", "sidecar_ref"}:
        trainer_module = args.trainer_module
        if args.model_kind == "sidecar_ref" and trainer_module == "train_gpt_mlx_jepa_sidecar":
            trainer_module = "train_gpt_mlx_jepa_sidecar_ref"
        base_mod, sidecar_mod = ess.load_modules(trainer_module)
        model_mod = sidecar_mod
        hps = sidecar_mod.Hyperparameters()
    else:
        import train_gpt_mlx as base_mod

        base_mod = importlib.reload(base_mod)
        sidecar_mod = None
        model_mod = base_mod
        hps = base_mod.Hyperparameters()
    ess.ensure_dataset_ready(args)
    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(args.tokenizer_path))
    artifact_path = Path(args.artifact)
    if artifact_path.suffix == ".ptz":
        flat_state = base_mod.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    else:
        flat_state = ess.load_flat_state(artifact_path, base_mod)
    if args.model_kind == "sidecar":
        model = model_mod.make_sidecar_gpt(hps, sp)
    else:
        model = model_mod.make_gpt(hps, sp)
    model.update(tree_unflatten(list(flat_state.items())))
    if hasattr(model, "set_turbo_qat"):
        model.set_turbo_qat(False, 0.0)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()
    return base_mod, hps, sp, model


def main() -> None:
    args = parse_args()
    base_mod, hps, sp, model = load_model(args)
    family_names = parse_family_list(args.families)
    routing = build_family_routing_spec(sp, hps.vocab_size, family_names)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        args.val_max_seqs,
    )

    seq_len = hps.train_seq_len
    num_seqs = (val_tokens.size - 1) // seq_len
    stats = init_bucket(family_names, args.max_distance)
    carry_state = None
    persistent = args.model_kind == "sidecar" and (
        args.mode == "persistent" or (args.mode is None and int(args.persistent or 0) == 1)
    )

    for seq_idx in range(num_seqs):
        if persistent and args.persist_group_seqs > 0 and seq_idx % int(args.persist_group_seqs) == 0:
            carry_state = None
        raw_start = seq_idx * seq_len
        chunk = val_tokens[raw_start : raw_start + seq_len + 1]
        x_np = chunk[:-1].reshape(1, seq_len).astype(np.int32, copy=False)
        y_np = chunk[1:].astype(np.int32, copy=False)
        raw_codes = detect_family_codes_np(x_np, routing)
        x = mx.array(x_np, dtype=mx.int32)
        op_codes = mx.array(raw_codes.astype(np.int32), dtype=mx.int32)
        if args.model_kind == "sidecar" and persistent:
            logits, carry_state, _aux = model.forward_logits_with_sidecar_state(
                x,
                operator_codes=op_codes,
                initial_sidecar_state=carry_state,
            )
        else:
            logits = model.forward_logits(x, operator_codes=op_codes)
        logits_np = np.asarray(logits.astype(mx.float32), dtype=np.float32)[0]
        log_probs = stable_log_softmax(logits_np)
        token_nll = -log_probs[np.arange(seq_len), y_np]
        update_stats(stats, family_names, token_nll, raw_codes[0], args.max_distance)

    result = {
        "label": args.label,
        "model_kind": args.model_kind,
        "artifact": str(Path(args.artifact).resolve()),
        "tokenizer_path": os.path.expanduser(args.tokenizer_path),
        "data_path": os.path.expanduser(args.data_path),
        "val_max_seqs": int(args.val_max_seqs),
        "train_seq_len": int(seq_len),
        "max_distance": int(args.max_distance),
        "families": list(family_names),
        "num_sequences": int(num_seqs),
        "persistent_eval": bool(persistent),
        "persist_group_seqs": int(args.persist_group_seqs),
        "stats": finalize_stats(stats, family_names, args.max_distance),
    }

    out_path = Path(args.result_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
