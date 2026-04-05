#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    try:
        sys.path.remove(str(root))
    except ValueError:
        pass
for root in (REPO_ROOT, TOOLS_ROOT):
    sys.path.insert(0, str(root))

import analyze_mlx_quant_export as aqe
import search_mlx_rope_gauge as srg
import train_gpt_mlx as tg


SURPRISAL_BUCKET_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 1.0e9)


def parse_gauge_result(
    path: Path | None,
) -> tuple[
    tuple[int, ...] | None,
    dict[int, int] | None,
    dict[tuple[int, int], int] | None,
    np.ndarray | None,
]:
    if path is None:
        return None, None, None, None
    data = json.loads(path.read_text())
    final = data.get("final", {})
    band_angles = final.get("band_angles")
    if band_angles is not None:
        gauge = final.get("gauge", {}) or {}
        layers_raw = gauge.get("layers", [])
        layers = tuple(int(x) for x in layers_raw) if layers_raw else None
        return layers, None, None, np.asarray(band_angles, dtype=np.float32)
    layers_raw = final.get("layers", [])
    seed_by_layer_raw = final.get("seed_by_layer", {}) or {}
    seed_by_block_kv_raw = final.get("seed_by_block_kv", {}) or {}
    layers = tuple(int(x) for x in layers_raw) if layers_raw else None
    seed_by_layer = {int(k): int(v) for k, v in seed_by_layer_raw.items()} if seed_by_layer_raw else None
    seed_by_block_kv = None
    if seed_by_block_kv_raw:
        seed_by_block_kv = {}
        for key, value in seed_by_block_kv_raw.items():
            layer_idx, kv_head_idx = (int(part) for part in str(key).split(":", 1))
            seed_by_block_kv[(layer_idx, kv_head_idx)] = int(value)
    return layers, seed_by_layer, seed_by_block_kv, None


def build_model_from_state(flat_state: dict[str, mx.array]) -> tg.GPT:
    config = aqe.infer_model_config(flat_state)
    model = tg.GPT(
        vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_layers"]),
        num_layer_templates=int(config["num_layer_templates"]),
        dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        mlp_mult=int(config["mlp_mult"]),
        mlp_leaky_slope=float(config.get("mlp_leaky_slope", 0.5)),
        tie_embeddings=bool(config["tie_embeddings"]),
        logit_chunk_tokens=0,
        logit_softcap=float(config["logit_softcap"]),
        rope_base=float(config["rope_base"]),
        tied_embed_init_std=float(config["tied_embed_init_std"]),
        qk_gain_init=float(config["qk_gain_init"]),
    )
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()
    model.set_turbo_qat(False, 0.0)
    return model


def load_proxy_tokens(
    flat_state: dict[str, mx.array],
    data_path: Path,
    tokenizer_path: Path,
    seq_len: int,
    max_seqs: int,
) -> np.ndarray:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    config = aqe.infer_model_config(flat_state)
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    return tg.limit_validation_tokens(
        tg.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), seq_len),
        seq_len,
        max_seqs,
    )


def build_proxy_context(
    flat_state: dict[str, mx.array],
    data_path: Path,
    tokenizer_path: Path,
    seq_len: int,
    max_seqs: int,
    batch_seqs: int,
) -> dict[str, object]:
    return {
        "ref_model": build_model_from_state(flat_state),
        "val_tokens": load_proxy_tokens(flat_state, data_path, tokenizer_path, seq_len, max_seqs),
        "seq_len": int(seq_len),
        "max_seqs": int(max_seqs),
        "batch_seqs": int(batch_seqs),
    }


def make_compare_state(
    flat_state: dict[str, mx.array],
    *,
    scheme: dict[str, object] | None,
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None,
    turbo_prod_patterns: tuple[str, ...] | None,
) -> tuple[dict[str, mx.array], dict[str, object] | None]:
    if scheme is None:
        return dict(flat_state), None
    if scheme["kind"] == "intn":
        quant_obj, stats = aqe.quantize_state_dict_intn(flat_state, int(scheme["bits"]))
        return aqe.dequantize_state_dict_intn(quant_obj), {"scheme": dict(scheme), "stats": stats}
    with aqe.turbo_config(
        flat_state,
        mse_bits=int(scheme["mse_bits"]),
        prod_bits=int(scheme["prod_bits"]),
        block_size=int(scheme["block_size"]),
        rot_seed=int(scheme["rot_seed"]),
        qjl_seed=int(scheme["qjl_seed"]),
        embed_export=turbo_embed_export,
        mse_patterns=turbo_mse_patterns,
        prod_patterns=turbo_prod_patterns,
    ):
        quant_obj, stats = tg.quantize_state_dict_turbo(flat_state)
        return tg.dequantize_state_dict(quant_obj), {"scheme": dict(scheme), "stats": stats}


def analyze_compare_state_proxies(
    ref_model: tg.GPT,
    compare_state: dict[str, mx.array],
    val_tokens: np.ndarray,
    seq_len: int,
    max_seqs: int,
    batch_seqs: int,
) -> dict[str, object]:
    cmp_model = build_model_from_state(compare_state)
    return analyze_output_proxies(ref_model, cmp_model, val_tokens, seq_len, max_seqs, batch_seqs)


def summarize_search_proxy(proxies: dict[str, object]) -> dict[str, float]:
    summary = proxies["summary"]
    mean_ce_delta = float(summary["mean_ce_delta"])
    mean_margin_delta = float(summary["mean_margin_delta"])
    top1_true_rate = float(summary["top1_true_rate"])
    mean_kl = float(summary["mean_kl"])
    mean_logit_rmse = float(summary["mean_logit_rmse"])
    score = mean_ce_delta - 0.5 * mean_margin_delta - 0.25 * top1_true_rate
    return {
        "mean_ce_delta": mean_ce_delta,
        "mean_margin_delta": mean_margin_delta,
        "top1_true_rate": top1_true_rate,
        "mean_kl": mean_kl,
        "mean_logit_rmse": mean_logit_rmse,
        "score": score,
    }


def iter_eval_batches(val_tokens: np.ndarray, seq_len: int, max_seqs: int, batch_seqs: int):
    total = min(max_seqs, (val_tokens.size - 1) // seq_len)
    if total <= 0:
        raise ValueError(f"No eval sequences for seq_len={seq_len}, max_seqs={max_seqs}")
    for start in range(0, total, batch_seqs):
        batch_seq_end = min(start + batch_seqs, total)
        raw_start = start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        yield x_np, y_np


def init_bucket_state() -> dict[str, object]:
    n = len(SURPRISAL_BUCKET_EDGES) - 1
    return {
        "count": np.zeros((n,), dtype=np.int64),
        "ce_delta_sum": np.zeros((n,), dtype=np.float64),
        "kl_sum": np.zeros((n,), dtype=np.float64),
        "rmse_sum": np.zeros((n,), dtype=np.float64),
        "fisher_sum": np.zeros((n,), dtype=np.float64),
        "margin_delta_sum": np.zeros((n,), dtype=np.float64),
        "top1_agree_sum": np.zeros((n,), dtype=np.float64),
        "top1_true_sum": np.zeros((n,), dtype=np.float64),
    }


def summarize_buckets(bucket_state: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx in range(len(SURPRISAL_BUCKET_EDGES) - 1):
        count = int(bucket_state["count"][idx])
        denom = max(count, 1)
        out.append(
            {
                "bucket": [float(SURPRISAL_BUCKET_EDGES[idx]), float(SURPRISAL_BUCKET_EDGES[idx + 1])],
                "count": count,
                "mean_ce_delta": float(bucket_state["ce_delta_sum"][idx] / denom),
                "mean_kl": float(bucket_state["kl_sum"][idx] / denom),
                "mean_logit_rmse": float(bucket_state["rmse_sum"][idx] / denom),
                "mean_fisher_quad": float(bucket_state["fisher_sum"][idx] / denom),
                "mean_margin_delta": float(bucket_state["margin_delta_sum"][idx] / denom),
                "top1_agree_rate": float(bucket_state["top1_agree_sum"][idx] / denom),
                "top1_true_rate": float(bucket_state["top1_true_sum"][idx] / denom),
            }
        )
    return out


def analyze_output_proxies(
    ref_model: tg.GPT,
    cmp_model: tg.GPT,
    val_tokens: np.ndarray,
    seq_len: int,
    max_seqs: int,
    batch_seqs: int,
) -> dict[str, object]:
    counts = 0
    ce_ref_sum = 0.0
    ce_cmp_sum = 0.0
    ce_delta_sum = 0.0
    kl_sum = 0.0
    rmse_sum = 0.0
    fisher_sum = 0.0
    top1_agree_sum = 0.0
    top1_true_sum = 0.0
    margin_delta_sum = 0.0
    bucket_state = init_bucket_state()
    pos_ce_delta_sum = np.zeros((seq_len,), dtype=np.float64)
    pos_count = np.zeros((seq_len,), dtype=np.int64)

    for x_np, y_np in iter_eval_batches(val_tokens, seq_len, max_seqs, batch_seqs):
        operator_codes = tg.operator_codes_mx_for_numpy_batch(ref_model, x_np)
        x = mx.array(x_np, dtype=mx.int32)
        logits_ref = ref_model.forward_logits(x, operator_codes).astype(mx.float32)
        logits_cmp = cmp_model.forward_logits(x, operator_codes).astype(mx.float32)
        mx.eval(logits_ref, logits_cmp)

        ref_np = np.asarray(logits_ref, dtype=np.float32)
        cmp_np = np.asarray(logits_cmp, dtype=np.float32)
        y_idx = y_np[..., None]

        # Stable log-softmax in numpy; vocab is only 1024, so this is cheap enough.
        ref_shift = ref_np - ref_np.max(axis=-1, keepdims=True)
        cmp_shift = cmp_np - cmp_np.max(axis=-1, keepdims=True)
        ref_logsumexp = np.log(np.exp(ref_shift, dtype=np.float64).sum(axis=-1, keepdims=True)).astype(np.float32)
        cmp_logsumexp = np.log(np.exp(cmp_shift, dtype=np.float64).sum(axis=-1, keepdims=True)).astype(np.float32)
        logp_ref = ref_shift - ref_logsumexp
        logp_cmp = cmp_shift - cmp_logsumexp
        p_ref = np.exp(logp_ref, dtype=np.float64).astype(np.float32)

        true_lp_ref = np.take_along_axis(logp_ref, y_idx, axis=-1).squeeze(-1)
        true_lp_cmp = np.take_along_axis(logp_cmp, y_idx, axis=-1).squeeze(-1)
        ce_ref = -true_lp_ref
        ce_cmp = -true_lp_cmp
        ce_delta = ce_cmp - ce_ref

        delta = cmp_np - ref_np
        rmse = np.sqrt(np.mean(np.square(delta, dtype=np.float64), axis=-1))
        fisher_mean = np.sum(p_ref * delta, axis=-1, dtype=np.float64)
        fisher_quad = np.sum(p_ref * np.square(delta, dtype=np.float64), axis=-1) - np.square(fisher_mean)
        kl = np.sum(p_ref * (logp_ref - logp_cmp), axis=-1, dtype=np.float64)

        top1_ref = ref_np.argmax(axis=-1)
        top1_cmp = cmp_np.argmax(axis=-1)
        top1_agree = (top1_ref == top1_cmp).astype(np.float32)
        top1_true = (top1_cmp == y_np).astype(np.float32)

        ref_true = np.take_along_axis(ref_np, y_idx, axis=-1).squeeze(-1)
        cmp_true = np.take_along_axis(cmp_np, y_idx, axis=-1).squeeze(-1)
        ref_other = ref_np.copy()
        cmp_other = cmp_np.copy()
        np.put_along_axis(ref_other, y_idx, -1.0e9, axis=-1)
        np.put_along_axis(cmp_other, y_idx, -1.0e9, axis=-1)
        ref_margin = ref_true - ref_other.max(axis=-1)
        cmp_margin = cmp_true - cmp_other.max(axis=-1)
        margin_delta = cmp_margin - ref_margin

        flat_count = int(ce_ref.size)
        counts += flat_count
        ce_ref_sum += float(ce_ref.astype(np.float64).sum())
        ce_cmp_sum += float(ce_cmp.astype(np.float64).sum())
        ce_delta_sum += float(ce_delta.astype(np.float64).sum())
        kl_sum += float(kl.astype(np.float64).sum())
        rmse_sum += float(rmse.astype(np.float64).sum())
        fisher_sum += float(fisher_quad.astype(np.float64).sum())
        top1_agree_sum += float(top1_agree.astype(np.float64).sum())
        top1_true_sum += float(top1_true.astype(np.float64).sum())
        margin_delta_sum += float(margin_delta.astype(np.float64).sum())

        pos_ce_delta_sum += ce_delta.astype(np.float64).sum(axis=0)
        pos_count += ce_delta.shape[0]

        bucket_idx = np.clip(
            np.searchsorted(np.asarray(SURPRISAL_BUCKET_EDGES[1:-1], dtype=np.float32), ce_ref.reshape(-1), side="right"),
            0,
            len(SURPRISAL_BUCKET_EDGES) - 2,
        )
        ce_delta_flat = ce_delta.reshape(-1)
        kl_flat = kl.reshape(-1)
        rmse_flat = rmse.reshape(-1)
        fisher_flat = fisher_quad.reshape(-1)
        margin_delta_flat = margin_delta.reshape(-1)
        top1_agree_flat = top1_agree.reshape(-1)
        top1_true_flat = top1_true.reshape(-1)
        for idx in range(len(SURPRISAL_BUCKET_EDGES) - 1):
            mask = bucket_idx == idx
            if not np.any(mask):
                continue
            bucket_state["count"][idx] += int(mask.sum())
            bucket_state["ce_delta_sum"][idx] += float(ce_delta_flat[mask].astype(np.float64).sum())
            bucket_state["kl_sum"][idx] += float(kl_flat[mask].astype(np.float64).sum())
            bucket_state["rmse_sum"][idx] += float(rmse_flat[mask].astype(np.float64).sum())
            bucket_state["fisher_sum"][idx] += float(fisher_flat[mask].astype(np.float64).sum())
            bucket_state["margin_delta_sum"][idx] += float(margin_delta_flat[mask].astype(np.float64).sum())
            bucket_state["top1_agree_sum"][idx] += float(top1_agree_flat[mask].astype(np.float64).sum())
            bucket_state["top1_true_sum"][idx] += float(top1_true_flat[mask].astype(np.float64).sum())

    denom = max(counts, 1)
    return {
        "summary": {
            "mean_ce_ref": ce_ref_sum / denom,
            "mean_ce_cmp": ce_cmp_sum / denom,
            "mean_ce_delta": ce_delta_sum / denom,
            "mean_kl": kl_sum / denom,
            "mean_logit_rmse": rmse_sum / denom,
            "mean_fisher_quad": fisher_sum / denom,
            "top1_agree_rate": top1_agree_sum / denom,
            "top1_true_rate": top1_true_sum / denom,
            "mean_margin_delta": margin_delta_sum / denom,
            "num_tokens": counts,
        },
        "surprisal_buckets": summarize_buckets(bucket_state),
        "position_mean_ce_delta": (pos_ce_delta_sum / np.maximum(pos_count, 1)).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze token-output proxy metrics for an MLX checkpoint against a quantized/dequantized compare state.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--compare-scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="none")
    parser.add_argument("--gauge-parameterization", choices=("global_head_phase", "banded_phase", "full_pair_phase"), default="global_head_phase")
    parser.add_argument("--gauge-seed", type=int, default=0)
    parser.add_argument("--gauge-angle-scale", type=float, default=math.pi)
    parser.add_argument("--gauge-num-bands", type=int, default=4)
    parser.add_argument("--gauge-layers")
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--batch-seqs", type=int, default=1)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    gauge_meta = None
    gauge_layers = srg.parse_layers(args.gauge_layers)
    gauge_result_layers, gauge_seed_by_layer, gauge_seed_by_block_kv, gauge_band_angles = parse_gauge_result(args.gauge_result)
    if gauge_result_layers is not None:
        gauge_layers = gauge_result_layers
    if args.gauge_transform != "none":
        if gauge_band_angles is not None:
            flat_state, gauge_meta = srg.apply_rope_gauge_band_angles(
                flat_state,
                band_angles=gauge_band_angles,
                transform=args.gauge_transform,
                layers=gauge_layers,
            )
        else:
            flat_state, gauge_meta = srg.apply_rope_gauge_transform(
                flat_state,
                seed=args.gauge_seed,
                angle_scale=args.gauge_angle_scale,
                transform=args.gauge_transform,
                parameterization=args.gauge_parameterization,
                num_bands=args.gauge_num_bands,
                layers=gauge_layers,
                seed_by_layer=gauge_seed_by_layer,
                seed_by_block_kv=gauge_seed_by_block_kv,
            )

    compare_scheme = None
    if args.compare_scheme.strip():
        schemes = aqe.parse_schemes(args.compare_scheme)
        if len(schemes) != 1:
            raise SystemExit("Expected exactly one compare scheme")
        compare_scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    compare_state, quant_meta = make_compare_state(
        flat_state,
        scheme=compare_scheme,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
    )

    ref_model = build_model_from_state(flat_state)
    cmp_model = build_model_from_state(compare_state)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    config = aqe.infer_model_config(flat_state)
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    val_tokens = tg.limit_validation_tokens(
        tg.load_validation_tokens(str(args.data_path / "fineweb_val_*.bin"), args.seq_len),
        args.seq_len,
        args.max_seqs,
    )
    proxies = analyze_output_proxies(ref_model, cmp_model, val_tokens, args.seq_len, args.max_seqs, args.batch_seqs)

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "compare_scheme": compare_scheme,
        "gauge": gauge_meta,
        "quant_meta": quant_meta,
        "seq_len": args.seq_len,
        "max_seqs": args.max_seqs,
        "batch_seqs": args.batch_seqs,
        "proxies": proxies,
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
