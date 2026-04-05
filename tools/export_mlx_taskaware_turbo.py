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

import analyze_mlx_output_proxies as aop
import analyze_mlx_quant_export as aqe
import optimize_mlx_turbo_bit_allocation as otba
import search_mlx_rope_gauge as srg
import train_gpt_mlx as tg


def load_and_apply_gauge(
    flat_state: dict[str, object],
    gauge_result: Path | None,
    *,
    gauge_transform: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if gauge_result is None or gauge_transform == "none":
        return flat_state, None
    gauge_layers, seed_by_layer, seed_by_block_kv, band_angles = aop.parse_gauge_result(gauge_result)
    if band_angles is not None:
        return srg.apply_rope_gauge_band_angles(
            flat_state,
            band_angles=band_angles,
            transform=gauge_transform,
            layers=gauge_layers,
        )
    return srg.apply_rope_gauge_transform(
        flat_state,
        seed=0,
        angle_scale=np.pi,
        transform=gauge_transform,
        parameterization="banded_phase",
        num_bands=4,
        layers=gauge_layers,
        seed_by_layer=seed_by_layer,
        seed_by_block_kv=seed_by_block_kv,
    )


def load_codebook_overrides(
    result_path: Path | None,
    *,
    scheme: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, object] | None]:
    if result_path is None:
        return {}, {}, None
    data = json.loads(result_path.read_text())
    final = data.get("final", {})
    codebook = final.get("codebook")
    codebooks_by_tensor = final.get("codebooks_by_tensor")
    if scheme["kind"] != "turbo":
        raise ValueError("Codebook overrides require a Turbo scheme")
    override_key = f"prod:{int(scheme['prod_bits']) - 1}:{int(scheme['block_size'])}"
    global_overrides: dict[str, np.ndarray] = {}
    if codebook is not None:
        global_overrides[override_key] = np.asarray(codebook, dtype=np.float32)
    tensor_overrides: dict[str, dict[str, np.ndarray]] = {}
    if isinstance(codebooks_by_tensor, dict):
        for name, values in codebooks_by_tensor.items():
            tensor_overrides[str(name)] = {
                override_key: np.asarray(values, dtype=np.float32),
            }
        return global_overrides, tensor_overrides, {
            "source": str(result_path),
            "kind": "global_plus_per_tensor_codebook_override"
            if global_overrides
            else "per_tensor_codebook_override",
            "override_keys": [override_key],
            "tensor_names": sorted(tensor_overrides.keys()),
        }
    if codebook is None:
        return {}, {}, {"source": str(result_path), "kind": "none"}
    refine_mode = str(data.get("search", {}).get("refine_mode", "prod"))
    if refine_mode != "prod":
        raise ValueError(f"Unsupported refine_mode={refine_mode}")
    return global_overrides, {}, {
        "source": str(result_path),
        "kind": "global_codebook_override",
        "override_keys": [override_key],
    }


def parse_override_tuple_map(overrides: dict[str, np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    return {
        (key.split(":")[0], int(key.split(":")[1]), int(key.split(":")[2])): np.asarray(value, dtype=np.float32)
        for key, value in overrides.items()
    }


def parse_tensor_override_tuple_map(
    overrides: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[tuple[str, int, int], np.ndarray]]:
    return {
        name: {
            (key.split(":")[0], int(key.split(":")[1]), int(key.split(":")[2])): np.asarray(value, dtype=np.float32)
            for key, value in tensor_map.items()
        }
        for name, tensor_map in overrides.items()
    }


def load_bitalloc_result(
    result_path: Path | None,
) -> tuple[dict[tuple[int, int], int], dict[str, object] | None]:
    if result_path is None:
        return {}, None
    data = json.loads(result_path.read_text())
    final = data.get("final", {})
    allocation_raw = final.get("allocation", {})
    allocation: dict[tuple[int, int], int] = {}
    for key, bits in allocation_raw.items():
        layer, kv_head = key.split(":")
        allocation[(int(layer), int(kv_head))] = int(bits)
    meta = {
        "source": str(result_path),
        "kind": "mixed_bit_k_head_allocation",
        "allocation": {f"{layer}:{kv_head}": int(bits) for (layer, kv_head), bits in sorted(allocation.items())},
        "delta_bpb_vs_baseline_quant": final.get("delta_bpb_vs_baseline_quant"),
        "final_eval": final.get("eval"),
    }
    return allocation, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a task-aware MLX Turbo artifact with optional gauge transform and codebook overrides.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--codebook-result", type=Path)
    parser.add_argument("--bitalloc-result", type=Path)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
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
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1:
        raise SystemExit("Expected exactly one scheme")
    scheme = schemes[0]

    flat_state = aqe.load_flat_state(args.checkpoint)
    flat_state, gauge_meta = load_and_apply_gauge(
        flat_state,
        args.gauge_result,
        gauge_transform=args.gauge_transform,
    )
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    quant_obj, quant_stats, deq_state = aqe.realize_scheme(
        flat_state,
        scheme,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )

    codebook_overrides, tensor_codebook_overrides, codebook_meta = load_codebook_overrides(args.codebook_result, scheme=scheme)
    global_codebook_tuple_overrides = parse_override_tuple_map(codebook_overrides)
    tensor_codebook_tuple_overrides = parse_tensor_override_tuple_map(tensor_codebook_overrides)
    if codebook_overrides:
        quant_obj = dict(quant_obj)
        quant_obj["turbo_codebook_overrides"] = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in codebook_overrides.items()
        }
    if tensor_codebook_overrides:
        quant_obj = dict(quant_obj)
        quant_obj["turbo_codebook_overrides_by_tensor"] = {
            name: {
                key: np.asarray(value, dtype=np.float32)
                for key, value in overrides.items()
            }
            for name, overrides in tensor_codebook_overrides.items()
        }
    if codebook_overrides or tensor_codebook_overrides:
        deq_state = aqe.dequantize_quant_obj(
            quant_obj,
            global_codebook_tuple_overrides,
            tensor_codebook_tuple_overrides,
        )

    bitalloc_allocation, bitalloc_meta = load_bitalloc_result(args.bitalloc_result)
    if bitalloc_allocation:
        all_units = otba.infer_k_units(flat_state)
        units_by_name: dict[str, list[dict[str, int | str]]] = {}
        for unit in all_units:
            units_by_name.setdefault(str(unit["name"]), []).append(unit)
        prod_bits = sorted({int(scheme["prod_bits"]), *bitalloc_allocation.values()})
        slice_library = otba.build_slice_library(
            flat_state,
            all_units,
            prod_bits=tuple(prod_bits),
            block_size=int(scheme["block_size"]),
        )
        quant_obj, quant_stats, deq_state = otba.build_candidate(
            flat_state=flat_state,
            base_quant_obj=quant_obj,
            base_stats=quant_stats,
            base_deq_state=deq_state,
            global_codebook_overrides=global_codebook_tuple_overrides,
            tensor_codebook_overrides=tensor_codebook_tuple_overrides,
            units_by_name=units_by_name,
            slice_library=slice_library,
            allocation=bitalloc_allocation,
            default_prod_bits=int(scheme["prod_bits"]),
        )

    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(quant_blob)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "out": str(args.out),
        "artifact_bytes": int(len(quant_blob)),
        "raw_pickle_bytes": int(len(quant_raw)),
        "quant_bytes": aqe.summarize_quant_bytes(quant_obj, quant_stats),
        "gauge": gauge_meta,
        "codebook": codebook_meta,
        "bitalloc": bitalloc_meta,
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
