#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

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
import export_mlx_taskaware_turbo as emt
import mlx.core as mx
import train_gpt_mlx as tg
import turbo_quant_mlx as tq


def parse_pattern_list(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    parts = tuple(part.strip() for part in text.split(",") if part.strip())
    return parts or ()


def ternary_gaussian_levels() -> tuple[float, float]:
    centroid = math.sqrt(2.0 / math.pi)
    for _ in range(32):
        threshold = 0.5 * centroid
        tail_prob = 0.5 * math.erfc(threshold / math.sqrt(2.0))
        pdf = math.exp(-0.5 * threshold * threshold) / math.sqrt(2.0 * math.pi)
        centroid = pdf / max(tail_prob, 1.0e-12)
    threshold = 0.5 * centroid
    return threshold, centroid


TERNARY_THRESHOLD_STD, TERNARY_CENTROID_STD = ternary_gaussian_levels()


def should_ternarize(name: str, arr: mx.array, target_patterns: tuple[str, ...] | None) -> bool:
    if not mx.issubdtype(arr.dtype, mx.floating):
        return False
    if arr.ndim != 2:
        return False
    if int(arr.size) <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
        return False
    if target_patterns is None:
        return True
    return any(pattern in name for pattern in target_patterns)


def ternary_quantize_dequantize_array(
    arr: mx.array,
    *,
    block_size: int,
    rotate: bool,
    rot_seed: int,
) -> np.ndarray:
    f32 = np.asarray(arr.astype(mx.float32), dtype=np.float32)
    rows, row_dim = f32.shape
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    pad = (-row_dim) % block_size
    flat = np.pad(f32, ((0, 0), (0, pad))) if pad else f32
    blocks = flat.reshape(-1, block_size)
    norms = np.linalg.norm(blocks, axis=-1).astype(np.float32, copy=False)
    unit = np.divide(blocks, np.maximum(norms[:, None], 1.0e-8), out=np.zeros_like(blocks), where=norms[:, None] > 0)
    if rotate:
        rotated = np.asarray(
            tq.rotate_blocks_mx(mx.array(unit), block_size, rot_seed).astype(mx.float32),
            dtype=np.float32,
        )
    else:
        rotated = unit
    threshold = np.float32(TERNARY_THRESHOLD_STD / math.sqrt(block_size))
    centroid = np.float32(TERNARY_CENTROID_STD / math.sqrt(block_size))
    ternary = np.where(rotated > threshold, centroid, np.where(rotated < -threshold, -centroid, 0.0)).astype(
        np.float32,
        copy=False,
    )
    if rotate:
        deq_unit = np.asarray(
            tq.inverse_rotate_blocks_mx(mx.array(ternary), block_size, rot_seed).astype(mx.float32),
            dtype=np.float32,
        )
    else:
        deq_unit = ternary
    deq = np.ascontiguousarray((deq_unit * norms[:, None]).reshape(rows, row_dim + pad)[:, :row_dim])
    return deq.astype(f32.dtype, copy=False)


def ternarize_state(
    flat_state: dict[str, mx.array],
    *,
    block_size: int,
    rotate: bool,
    rot_seed: int,
    target_patterns: tuple[str, ...] | None,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    out: dict[str, mx.array] = {}
    quantized_names: list[str] = []
    param_count = 0
    ternary_param_count = 0
    kept_float_param_count = 0
    for name, arr in flat_state.items():
        param_count += int(arr.size)
        if should_ternarize(name, arr, target_patterns):
            deq = ternary_quantize_dequantize_array(arr, block_size=block_size, rotate=rotate, rot_seed=rot_seed)
            out[name] = mx.array(deq, dtype=arr.dtype)
            quantized_names.append(name)
            ternary_param_count += int(arr.size)
        else:
            out[name] = arr
            if mx.issubdtype(arr.dtype, mx.floating):
                kept_float_param_count += int(arr.size)
    packed2_bytes = int(math.ceil(ternary_param_count * 2 / 8))
    entropy_bytes = float(ternary_param_count * math.log2(3.0) / 8.0)
    return out, {
        "rotate": bool(rotate),
        "rot_seed": int(rot_seed),
        "block_size": int(block_size),
        "quantized_tensor_count": int(len(quantized_names)),
        "quantized_tensor_names": quantized_names,
        "param_count": int(param_count),
        "ternary_param_count": int(ternary_param_count),
        "kept_float_param_count": int(kept_float_param_count),
        "packed2_payload_bytes_estimate": packed2_bytes,
        "entropy_payload_bytes_estimate": entropy_bytes,
        "threshold_std": float(TERNARY_THRESHOLD_STD),
        "centroid_std": float(TERNARY_CENTROID_STD),
    }


def evaluate_state(
    flat_state: dict[str, mx.array],
    *,
    eval_ctx: dict[str, object],
    block_size: int,
    rotate: bool,
    rot_seed: int,
    target_patterns: tuple[str, ...] | None,
) -> dict[str, object]:
    ternary_state, meta = ternarize_state(
        flat_state,
        block_size=block_size,
        rotate=rotate,
        rot_seed=rot_seed,
        target_patterns=target_patterns,
    )
    val_loss, val_bpb = aqe.eval_state(eval_ctx, ternary_state)
    return {
        "meta": meta,
        "eval": {
            "val_loss": float(val_loss),
            "val_bpb": float(val_bpb),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare naive blockwise ternary and Turbo-rotated blockwise ternary on an MLX checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="none")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-max-seqs", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=262144)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--rot-seed", type=int, default=17)
    parser.add_argument(
        "--target-patterns",
        default="attn.c_q.weight,attn.c_k.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight,tok_emb.weight,lm_head.weight",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    flat_state, gauge_meta = emt.load_and_apply_gauge(
        flat_state,
        args.gauge_result,
        gauge_transform=args.gauge_transform,
    )
    config = aqe.infer_model_config(flat_state)
    eval_ctx = aqe.build_eval_context(
        config,
        args.data_path,
        args.tokenizer_path,
        args.train_seq_len,
        args.val_max_seqs,
        args.val_batch_size,
        args.eval_seq_len,
        args.eval_stride,
        args.eval_batch_seqs,
    )
    target_patterns = parse_pattern_list(args.target_patterns)

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "gauge": gauge_meta,
        "block_size": int(args.block_size),
        "rot_seed": int(args.rot_seed),
        "target_patterns": list(target_patterns) if target_patterns is not None else None,
    }
    if not args.skip_baseline:
        base_loss, base_bpb = aqe.eval_state(eval_ctx, flat_state)
        result["baseline"] = {
            "val_loss": float(base_loss),
            "val_bpb": float(base_bpb),
        }

    naive = evaluate_state(
        flat_state,
        eval_ctx=eval_ctx,
        block_size=args.block_size,
        rotate=False,
        rot_seed=args.rot_seed,
        target_patterns=target_patterns,
    )
    rotated = evaluate_state(
        flat_state,
        eval_ctx=eval_ctx,
        block_size=args.block_size,
        rotate=True,
        rot_seed=args.rot_seed,
        target_patterns=target_patterns,
    )
    result["naive_ternary"] = naive
    result["rotated_ternary"] = rotated
    if "baseline" in result:
        result["naive_ternary"]["delta_bpb_vs_baseline"] = float(
            naive["eval"]["val_bpb"] - result["baseline"]["val_bpb"]
        )
        result["rotated_ternary"]["delta_bpb_vs_baseline"] = float(
            rotated["eval"]["val_bpb"] - result["baseline"]["val_bpb"]
        )
    result["delta_bpb_rotated_vs_naive"] = float(
        rotated["eval"]["val_bpb"] - naive["eval"]["val_bpb"]
    )

    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
