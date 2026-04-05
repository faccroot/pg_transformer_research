#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
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
import probe_mlx_turbo_ternary as ptt
import train_gpt_mlx as tg
import turbo_quant_mlx as tq


def should_quantize_2d(
    name: str,
    arr: mx.array,
    target_patterns: tuple[str, ...] | None,
) -> bool:
    if not mx.issubdtype(arr.dtype, mx.floating):
        return False
    if arr.ndim != 2:
        return False
    if int(arr.size) <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
        return False
    if target_patterns is None:
        return True
    return any(pattern in name for pattern in target_patterns)


@contextlib.contextmanager
def configured_turbo(
    *,
    block_size: int,
    mse_bits: int,
    prod_bits: int,
    rot_seed: int,
    qjl_seed: int,
) -> None:
    prior = (
        tq.TURBO_BLOCK_SIZE,
        tq.TURBO_MSE_BITS,
        tq.TURBO_PROD_BITS,
        tq.TURBO_ROT_SEED,
        tq.TURBO_QJL_SEED,
        tq.TURBO_MSE_NAME_PATTERNS,
        tq.TURBO_PROD_NAME_PATTERNS,
    )
    tq.configure(
        block_size=block_size,
        mse_bits=mse_bits,
        prod_bits=prod_bits,
        rot_seed=rot_seed,
        qjl_seed=qjl_seed,
        mse_patterns=(),
        prod_patterns=(),
    )
    try:
        yield
    finally:
        tq.configure(
            block_size=prior[0],
            mse_bits=prior[1],
            prod_bits=prior[2],
            rot_seed=prior[3],
            qjl_seed=prior[4],
            mse_patterns=prior[5],
            prod_patterns=prior[6],
        )


def apply_hybrid_quantization(
    flat_state: dict[str, mx.array],
    *,
    ternary_patterns: tuple[str, ...] | None,
    ternary_rotate: bool,
    ternary_block_size: int,
    ternary_rot_seed: int,
    turbo_patterns: tuple[str, ...] | None,
    turbo_mode: str,
    turbo_total_bits: int,
    turbo_block_size: int,
    turbo_rot_seed: int,
    turbo_qjl_seed: int,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    out: dict[str, mx.array] = {}
    ternary_names: list[str] = []
    turbo_names: list[str] = []
    total_params = 0
    ternary_params = 0
    turbo_params = 0
    kept_float_params = 0
    turbo_payload_bytes = 0

    with configured_turbo(
        block_size=turbo_block_size,
        mse_bits=turbo_total_bits if turbo_mode == "mse" else max(turbo_total_bits - 1, 1),
        prod_bits=turbo_total_bits if turbo_mode == "prod" else max(turbo_total_bits + 1, 2),
        rot_seed=turbo_rot_seed,
        qjl_seed=turbo_qjl_seed,
    ):
        for name, arr in flat_state.items():
            total_params += int(arr.size)
            if ptt.should_ternarize(name, arr, ternary_patterns):
                deq = ptt.ternary_quantize_dequantize_array(
                    arr,
                    block_size=ternary_block_size,
                    rotate=ternary_rotate,
                    rot_seed=ternary_rot_seed,
                )
                out[name] = mx.array(deq, dtype=arr.dtype)
                ternary_names.append(name)
                ternary_params += int(arr.size)
                continue
            if should_quantize_2d(name, arr, turbo_patterns):
                deq, meta = tq.turbo_quantize_dequantize_array(
                    arr,
                    mode=turbo_mode,
                    total_bits=turbo_total_bits,
                    block_size=turbo_block_size,
                )
                out[name] = mx.array(deq, dtype=arr.dtype)
                turbo_names.append(name)
                turbo_params += int(arr.size)
                turbo_payload_bytes += sum(
                    int(value.nbytes)
                    for value in meta.values()
                    if isinstance(value, np.ndarray)
                )
                continue
            out[name] = arr
            if mx.issubdtype(arr.dtype, mx.floating):
                kept_float_params += int(arr.size)

    ternary_entropy_bytes = float(ternary_params * np.log2(3.0) / 8.0)
    ternary_packed2_bytes = int(np.ceil(ternary_params * 2 / 8))
    return out, {
        "ternary": {
            "rotate": bool(ternary_rotate),
            "rot_seed": int(ternary_rot_seed),
            "block_size": int(ternary_block_size),
            "tensor_names": ternary_names,
            "tensor_count": int(len(ternary_names)),
            "param_count": int(ternary_params),
            "entropy_payload_bytes_estimate": ternary_entropy_bytes,
            "packed2_payload_bytes_estimate": ternary_packed2_bytes,
        },
        "turbo": {
            "mode": turbo_mode,
            "total_bits": int(turbo_total_bits),
            "block_size": int(turbo_block_size),
            "rot_seed": int(turbo_rot_seed),
            "qjl_seed": int(turbo_qjl_seed),
            "tensor_names": turbo_names,
            "tensor_count": int(len(turbo_names)),
            "param_count": int(turbo_params),
            "payload_bytes_estimate": int(turbo_payload_bytes),
        },
        "total_param_count": int(total_params),
        "kept_float_param_count": int(kept_float_params),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe hybrid quantization: ternary on one tensor subset, Turbo on another."
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
    parser.add_argument(
        "--ternary-patterns",
        default="attn.c_q.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight,tok_emb.weight,lm_head.weight",
    )
    parser.add_argument("--ternary-rotate", action="store_true")
    parser.add_argument("--ternary-block-size", type=int, default=256)
    parser.add_argument("--ternary-rot-seed", type=int, default=17)
    parser.add_argument("--turbo-patterns", default="attn.c_k.weight")
    parser.add_argument("--turbo-mode", choices=("mse", "prod"), default="prod")
    parser.add_argument("--turbo-total-bits", type=int, default=4)
    parser.add_argument("--turbo-block-size", type=int, default=256)
    parser.add_argument("--turbo-rot-seed", type=int, default=17)
    parser.add_argument("--turbo-qjl-seed", type=int, default=29)
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
    ternary_patterns = ptt.parse_pattern_list(args.ternary_patterns)
    turbo_patterns = ptt.parse_pattern_list(args.turbo_patterns)
    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "gauge": gauge_meta,
        "ternary_patterns": list(ternary_patterns) if ternary_patterns is not None else None,
        "turbo_patterns": list(turbo_patterns) if turbo_patterns is not None else None,
    }
    if not args.skip_baseline:
        base_loss, base_bpb = aqe.eval_state(eval_ctx, flat_state)
        result["baseline"] = {
            "val_loss": float(base_loss),
            "val_bpb": float(base_bpb),
        }

    hybrid_state, hybrid_meta = apply_hybrid_quantization(
        flat_state,
        ternary_patterns=ternary_patterns,
        ternary_rotate=args.ternary_rotate,
        ternary_block_size=args.ternary_block_size,
        ternary_rot_seed=args.ternary_rot_seed,
        turbo_patterns=turbo_patterns,
        turbo_mode=args.turbo_mode,
        turbo_total_bits=args.turbo_total_bits,
        turbo_block_size=args.turbo_block_size,
        turbo_rot_seed=args.turbo_rot_seed,
        turbo_qjl_seed=args.turbo_qjl_seed,
    )
    val_loss, val_bpb = aqe.eval_state(eval_ctx, hybrid_state)
    result["hybrid"] = {
        "meta": hybrid_meta,
        "eval": {
            "val_loss": float(val_loss),
            "val_bpb": float(val_bpb),
        },
    }
    if "baseline" in result:
        result["hybrid"]["delta_bpb_vs_baseline"] = float(val_bpb - float(result["baseline"]["val_bpb"]))

    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
