#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import train_gpt_mlx as tg
import analyze_mlx_quant_export as aq


def metric_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "sum_abs": 0.0,
        "dot": 0.0,
        "ref_sq": 0.0,
        "other_sq": 0.0,
        "max_abs": 0.0,
    }


def update_metric(acc: dict[str, float], ref: mx.array, other: mx.array) -> None:
    ref_np = np.asarray(ref.astype(mx.float32), dtype=np.float32)
    other_np = np.asarray(other.astype(mx.float32), dtype=np.float32)
    diff = other_np - ref_np
    acc["count"] += float(diff.size)
    acc["sum"] += float(diff.sum(dtype=np.float64))
    acc["sum_sq"] += float(np.square(diff, dtype=np.float64).sum())
    acc["sum_abs"] += float(np.abs(diff, dtype=np.float64).sum())
    acc["dot"] += float(np.multiply(ref_np, other_np, dtype=np.float64).sum())
    acc["ref_sq"] += float(np.square(ref_np, dtype=np.float64).sum())
    acc["other_sq"] += float(np.square(other_np, dtype=np.float64).sum())
    acc["max_abs"] = max(acc["max_abs"], float(np.abs(diff).max(initial=0.0)))


def finalize_metric(acc: dict[str, float]) -> dict[str, float]:
    denom = max(acc["count"], 1.0)
    return {
        "bias": acc["sum"] / denom,
        "rmse": math.sqrt(acc["sum_sq"] / denom),
        "mae": acc["sum_abs"] / denom,
        "cosine": acc["dot"] / max(math.sqrt(acc["ref_sq"] * acc["other_sq"]), 1e-12),
        "max_abs": acc["max_abs"],
    }


def expand_kv_heads(x: mx.array, num_heads: int, num_kv_heads: int) -> mx.array:
    if num_heads == num_kv_heads:
        return x
    repeats = num_heads // num_kv_heads
    return mx.repeat(x, repeats=repeats, axis=1)


def attention_logits(attn: tg.CausalSelfAttention, x: mx.array) -> mx.array:
    bsz, seqlen, dim = x.shape
    q = attn.c_q(x).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = attn.c_k(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    q = attn.rope(tg.rms_norm(q).astype(tg.COMPUTE_DTYPE))
    k = attn.rope(tg.rms_norm(k).astype(tg.COMPUTE_DTYPE))
    q = q * attn.q_gain.astype(q.dtype)[None, :, None, None]
    k = expand_kv_heads(k, attn.num_heads, attn.num_kv_heads)
    return (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * float(attn.scale)


def build_model(config: dict[str, object]) -> tg.GPT:
    return tg.GPT(
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


def load_probe_tokens(data_path: Path, train_seq_len: int, probe_seqs: int) -> np.ndarray:
    val_tokens = tg.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), train_seq_len)
    return tg.limit_validation_tokens(val_tokens, train_seq_len, probe_seqs)


def quantize_for_scheme(
    flat_state: dict[str, mx.array],
    scheme: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None,
    turbo_prod_patterns: tuple[str, ...] | None,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    if scheme["kind"] == "intn":
        quant_obj, stats = aq.quantize_state_dict_intn(flat_state, int(scheme["bits"]))
        return aq.dequantize_state_dict_intn(quant_obj), {"stats": stats}
    with aq.turbo_config(
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
        return tg.dequantize_state_dict(quant_obj), {"stats": stats}


def analyze_scheme_activations(
    flat_state: dict[str, mx.array],
    config: dict[str, object],
    probe_tokens: np.ndarray,
    train_seq_len: int,
    scheme: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None,
    turbo_prod_patterns: tuple[str, ...] | None,
    eval_ctx: dict[str, object] | None,
) -> dict[str, object]:
    float_model = build_model(config)
    float_model.update(tree_unflatten(list(flat_state.items())))
    float_model.set_turbo_qat(False, 0.0)
    float_model.clear_turbo_cache()

    deq_state, quant_meta = quantize_for_scheme(flat_state, scheme, turbo_embed_export, turbo_mse_patterns, turbo_prod_patterns)
    quant_model = build_model(config)
    quant_model.update(tree_unflatten(list(deq_state.items())))
    quant_model.set_turbo_qat(False, 0.0)
    quant_model.clear_turbo_cache()

    layer_logits: dict[str, dict[str, float]] = {}
    layer_outputs: dict[str, dict[str, float]] = {}
    logits_acc: dict[str, dict[str, float]] = {}
    out_acc: dict[str, dict[str, float]] = {}
    global_logits = metric_accumulator()
    global_out = metric_accumulator()

    total_seqs = (probe_tokens.size - 1) // train_seq_len
    for seq_idx in range(total_seqs):
        raw_start = seq_idx * train_seq_len
        chunk = probe_tokens[raw_start : raw_start + train_seq_len + 1]
        x_ids = mx.array(chunk[:-1].reshape(1, train_seq_len), dtype=mx.int32)
        x_float = tg.rms_norm(float_model.tok_emb(x_ids).astype(tg.COMPUTE_DTYPE))
        x0_float = x_float
        skips: list[mx.array] = []

        for i in range(float_model.num_encoder_layers):
            float_block = float_model.block_for_step(i)
            quant_block = quant_model.block_for_step(i)
            name = f"blocks.{i}"
            mix = float_block.resid_mix.astype(x_float.dtype)
            x_in = mix[0][None, None, :] * x_float + mix[1][None, None, :] * x0_float
            logits_acc.setdefault(name, metric_accumulator())
            out_acc.setdefault(name, metric_accumulator())
            attn_in = float_block.attn_norm(x_in)
            logits_float = attention_logits(float_block.attn, attn_in)
            logits_quant = attention_logits(quant_block.attn, attn_in)
            update_metric(logits_acc[name], logits_float, logits_quant)
            update_metric(global_logits, logits_float, logits_quant)
            out_float = float_block(x_float, x0_float)
            out_quant = quant_block(x_float, x0_float)
            update_metric(out_acc[name], out_float, out_quant)
            update_metric(global_out, out_float, out_quant)
            x_float = out_float
            skips.append(x_float)

        for i in range(float_model.num_decoder_layers):
            if skips:
                x_float = x_float + float_model.skip_weights[i].astype(x_float.dtype)[None, None, :] * skips.pop()
            step_idx = float_model.num_encoder_layers + i
            float_block = float_model.block_for_step(step_idx)
            quant_block = quant_model.block_for_step(step_idx)
            name = f"blocks.{step_idx}"
            mix = float_block.resid_mix.astype(x_float.dtype)
            x_in = mix[0][None, None, :] * x_float + mix[1][None, None, :] * x0_float
            logits_acc.setdefault(name, metric_accumulator())
            out_acc.setdefault(name, metric_accumulator())
            attn_in = float_block.attn_norm(x_in)
            logits_float = attention_logits(float_block.attn, attn_in)
            logits_quant = attention_logits(quant_block.attn, attn_in)
            update_metric(logits_acc[name], logits_float, logits_quant)
            update_metric(global_logits, logits_float, logits_quant)
            out_float = float_block(x_float, x0_float)
            out_quant = quant_block(x_float, x0_float)
            update_metric(out_acc[name], out_float, out_quant)
            update_metric(global_out, out_float, out_quant)
            x_float = out_float

    for name in sorted(logits_acc):
        layer_logits[name] = finalize_metric(logits_acc[name])
        layer_outputs[name] = finalize_metric(out_acc[name])

    result: dict[str, object] = {
        "scheme": dict(scheme),
        "quant_stats": quant_meta["stats"],
        "logits": {"global": finalize_metric(global_logits), "layers": layer_logits},
        "block_outputs": {"global": finalize_metric(global_out), "layers": layer_outputs},
    }
    if eval_ctx is not None:
        val_loss, val_bpb = aq.eval_state(eval_ctx, deq_state)
        result["eval"] = {"val_loss": float(val_loss), "val_bpb": float(val_bpb)}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-wise activation and attention-logit distortion analysis for MLX quantization schemes.")
    parser.add_argument("checkpoint", type=Path, help="Raw MLX .npz checkpoint")
    parser.add_argument("--schemes", default="int8,int6,turbo:3:4:256")
    parser.add_argument("--probe-seqs", type=int, default=8)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, help="Optional tokenizer for capped eval")
    parser.add_argument("--eval-val-max-seqs", type=int, default=0)
    parser.add_argument("--eval-val-batch-size", type=int, default=524288)
    parser.add_argument("--eval-seq-len", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aq.load_flat_state(args.checkpoint)
    config = aq.infer_model_config(flat_state)
    schemes = aq.parse_schemes(args.schemes)
    turbo_mse_patterns = aq.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aq.parse_pattern_list(args.turbo_prod_patterns)
    probe_tokens = load_probe_tokens(args.data_path, args.train_seq_len, args.probe_seqs)
    eval_ctx = None
    if args.tokenizer_path:
        eval_ctx = aq.build_eval_context(
            config,
            args.data_path,
            args.tokenizer_path,
            args.train_seq_len,
            args.eval_val_max_seqs,
            args.eval_val_batch_size,
            args.eval_seq_len,
            args.eval_stride,
            args.eval_batch_seqs,
        )
    results = {
        "checkpoint": str(args.checkpoint),
        "probe_seqs": args.probe_seqs,
        "schemes": [
            analyze_scheme_activations(
                flat_state,
                config,
                probe_tokens,
                args.train_seq_len,
                scheme,
                args.turbo_embed_export,
                turbo_mse_patterns,
                turbo_prod_patterns,
                eval_ctx,
            )
            for scheme in schemes
        ],
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
