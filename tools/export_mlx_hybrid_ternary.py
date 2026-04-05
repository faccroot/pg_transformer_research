#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zlib
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
import ternary_quant_mlx as trq
import train_gpt_mlx as tg
import turbo_quant_mlx as tq


def should_2d_quantize(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a hybrid ternary/Turbo/int8 MLX artifact and optionally roundtrip-evaluate it."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="none")
    parser.add_argument(
        "--ternary-patterns",
        default="attn.c_q.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight",
    )
    parser.add_argument("--ternary-rotate", action="store_true")
    parser.add_argument("--ternary-block-size", type=int, default=256)
    parser.add_argument("--ternary-rot-seed", type=int, default=17)
    parser.add_argument("--turbo-patterns", default="__no_match__")
    parser.add_argument("--turbo-mode", choices=("mse", "prod"), default="prod")
    parser.add_argument("--turbo-total-bits", type=int, default=4)
    parser.add_argument("--turbo-block-size", type=int, default=256)
    parser.add_argument("--turbo-rot-seed", type=int, default=17)
    parser.add_argument("--turbo-qjl-seed", type=int, default=29)
    parser.add_argument("--int8-patterns")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--confirm-val-max-seqs", type=int, default=0)
    parser.add_argument("--confirm-val-batch-size", type=int, default=262144)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-stride", type=int, default=0)
    parser.add_argument("--confirm-eval-batch-seqs", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    flat_state, gauge_meta = emt.load_and_apply_gauge(
        flat_state,
        args.gauge_result,
        gauge_transform=args.gauge_transform,
    )
    ternary_patterns = aqe.parse_pattern_list(args.ternary_patterns)
    turbo_patterns = aqe.parse_pattern_list(args.turbo_patterns)
    int8_patterns = aqe.parse_pattern_list(args.int8_patterns)

    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    turbo: dict[str, dict[str, object]] = {}
    ternary: dict[str, dict[str, object]] = {}
    stats = tg.empty_quant_stats()
    stats["payload_bytes"] = 0
    stats["num_ternary_tensors"] = 0
    stats["ternary_payload_bytes"] = 0
    stats["ternary_norm_bytes"] = 0
    stats["ternary_state_bytes"] = 0

    prior_turbo = (
        tq.TURBO_BLOCK_SIZE,
        tq.TURBO_MSE_BITS,
        tq.TURBO_PROD_BITS,
        tq.TURBO_ROT_SEED,
        tq.TURBO_QJL_SEED,
        tq.TURBO_MSE_NAME_PATTERNS,
        tq.TURBO_PROD_NAME_PATTERNS,
    )
    tq.configure(
        block_size=args.turbo_block_size,
        mse_bits=args.turbo_total_bits if args.turbo_mode == "mse" else max(args.turbo_total_bits - 1, 1),
        prod_bits=args.turbo_total_bits if args.turbo_mode == "prod" else max(args.turbo_total_bits + 1, 2),
        rot_seed=args.turbo_rot_seed,
        qjl_seed=args.turbo_qjl_seed,
        mse_patterns=(),
        prod_patterns=(),
    )
    try:
        for name, arr in flat_state.items():
            if any(pattern in name for pattern in tg.SERIALIZATION_SKIP_NAME_PATTERNS):
                continue
            stats["param_count"] += int(arr.size)
            stats["num_tensors"] += 1
            stats["baseline_tensor_bytes"] += int(arr.nbytes)
            if not mx.issubdtype(arr.dtype, mx.floating):
                stats["num_nonfloat_tensors"] += 1
                stats["num_passthrough_tensors"] += 1
                kept = np.ascontiguousarray(np.array(arr))
                passthrough[name] = kept
                stats["passthrough_payload_bytes"] += int(kept.nbytes)
                stats["int8_payload_bytes"] += int(kept.nbytes)
                stats["payload_bytes"] += int(kept.nbytes)
                continue

            if should_2d_quantize(name, arr, ternary_patterns):
                stats["num_float_tensors"] += 1
                stats["num_ternary_tensors"] += 1
                _, meta = trq.ternary_quantize_dequantize_array(
                    arr,
                    block_size=args.ternary_block_size,
                    rotate=args.ternary_rotate,
                    rot_seed=args.ternary_rot_seed,
                )
                ternary[name] = meta
                breakdown = trq.ternary_payload_breakdown(meta)
                stats["ternary_payload_bytes"] += int(breakdown["payload_bytes"])
                stats["ternary_norm_bytes"] += int(breakdown["norm_bytes"])
                stats["ternary_state_bytes"] += int(breakdown["state_bytes"])
                stats["payload_bytes"] += int(breakdown["payload_bytes"])
                continue

            if should_2d_quantize(name, arr, turbo_patterns):
                stats["num_float_tensors"] += 1
                _, meta = tq.turbo_quantize_dequantize_array(
                    arr,
                    mode=args.turbo_mode,
                    total_bits=args.turbo_total_bits,
                    block_size=args.turbo_block_size,
                )
                turbo[name] = meta
                breakdown = aqe.turbo_meta_payload_breakdown(meta)
                stats["turbo_norm_bytes"] += int(breakdown["norm_bytes"])
                stats["payload_bytes"] += int(breakdown["payload_bytes"])
                stats["int8_payload_bytes"] += int(breakdown["payload_bytes"])
                if args.turbo_mode == "mse":
                    stats["num_turbo_mse_tensors"] += 1
                    stats["turbo_mse_payload_bytes"] += int(breakdown["payload_bytes"])
                else:
                    stats["num_turbo_prod_tensors"] += 1
                    stats["turbo_prod_payload_bytes"] += int(breakdown["payload_bytes"])
                    stats["turbo_qjl_bytes"] += int(breakdown["qjl_bytes"])
                continue

            if should_2d_quantize(name, arr, int8_patterns):
                stats["num_float_tensors"] += 1
                stats["num_fallback_int8_tensors"] += 1
                q, s = tg.quantize_float_array(arr)
                if s.ndim > 0:
                    qmeta[name] = {"scheme": "per_row", "axis": 0}
                quantized[name] = q
                scales[name] = s
                dtypes[name] = str(arr.dtype).split(".")[-1]
                stats["fallback_quantized_bytes"] += int(q.nbytes)
                stats["fallback_scale_bytes"] += int(s.nbytes)
                stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
                stats["payload_bytes"] += int(q.nbytes + s.nbytes)
                continue

            kept = tg.keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_passthrough_tensors"] += 1
            stats["passthrough_payload_bytes"] += int(kept.nbytes)
            stats["int8_payload_bytes"] += int(kept.nbytes)
            stats["payload_bytes"] += int(kept.nbytes)
    finally:
        tq.configure(
            block_size=prior_turbo[0],
            mse_bits=prior_turbo[1],
            prod_bits=prior_turbo[2],
            rot_seed=prior_turbo[3],
            qjl_seed=prior_turbo[4],
            mse_patterns=prior_turbo[5],
            prod_patterns=prior_turbo[6],
        )

    quant_obj: dict[str, object] = {
        "__quant_format__": "hybrid_ternary_turbo_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "hybrid_config": {
            "ternary_patterns": list(ternary_patterns) if ternary_patterns is not None else None,
            "ternary_rotate": bool(args.ternary_rotate),
            "ternary_block_size": int(args.ternary_block_size),
            "ternary_rot_seed": int(args.ternary_rot_seed),
            "turbo_patterns": list(turbo_patterns) if turbo_patterns is not None else None,
            "turbo_mode": args.turbo_mode,
            "turbo_total_bits": int(args.turbo_total_bits),
            "turbo_block_size": int(args.turbo_block_size),
            "turbo_rot_seed": int(args.turbo_rot_seed),
            "turbo_qjl_seed": int(args.turbo_qjl_seed),
            "int8_patterns": list(int8_patterns) if int8_patterns is not None else None,
        },
    }
    if qmeta:
        quant_obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        quant_obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if turbo:
        quant_obj["turbo"] = turbo
    if ternary:
        quant_obj["ternary"] = ternary

    deq_state = tg.dequantize_state_dict(quant_obj)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(quant_blob)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "out": str(args.out),
        "artifact_bytes": int(len(quant_blob)),
        "raw_pickle_bytes": int(len(quant_raw)),
        "quant_bytes": aqe.summarize_quant_bytes(quant_obj, stats),
        "stats": stats,
        "gauge": gauge_meta,
        "hybrid_config": quant_obj["hybrid_config"],
    }

    if args.confirm_val_max_seqs > 0:
        if args.data_path is None or args.tokenizer_path is None:
            raise SystemExit("Need --data-path and --tokenizer-path for confirmation")
        config = aqe.infer_model_config(flat_state)
        eval_ctx = aqe.build_eval_context(
            config,
            args.data_path,
            args.tokenizer_path,
            args.train_seq_len,
            args.confirm_val_max_seqs,
            args.confirm_val_batch_size,
            args.confirm_eval_seq_len,
            args.confirm_eval_stride,
            args.confirm_eval_batch_seqs,
        )
        val_loss, val_bpb = aqe.eval_state(eval_ctx, deq_state)
        summary["confirm_eval"] = {"val_loss": float(val_loss), "val_bpb": float(val_bpb)}
        roundtrip_quant_obj = pickle.loads(zlib.decompress(args.out.read_bytes()))
        roundtrip_state = tg.dequantize_state_dict(roundtrip_quant_obj)
        rt_loss, rt_bpb = aqe.eval_state(eval_ctx, roundtrip_state)
        summary["confirm_roundtrip_eval"] = {"val_loss": float(rt_loss), "val_bpb": float(rt_bpb)}

    text = json.dumps(summary, indent=2, sort_keys=False)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
