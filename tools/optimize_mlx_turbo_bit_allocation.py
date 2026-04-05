#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import analyze_mlx_output_proxies as aop
import analyze_mlx_quant_export as aqe
import export_mlx_taskaware_turbo as emt
import search_mlx_rope_gauge as srg


def parse_int_list(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(values) if values else None


def parse_prod_bits(text: str) -> tuple[int, ...]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("Expected at least one candidate prod bit")
    return tuple(values)


def parse_allocation_map(raw: dict[str, object] | None) -> dict[tuple[int, int], int]:
    allocation: dict[tuple[int, int], int] = {}
    if not isinstance(raw, dict):
        return allocation
    for key, bits in raw.items():
        layer, kv_head = str(key).split(":")
        allocation[(int(layer), int(kv_head))] = int(bits)
    return allocation


def parse_gauge_result(
    flat_state: dict[str, object],
    path: Path | None,
    *,
    transform: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if path is None or transform == "none":
        return flat_state, None
    gauge_layers, gauge_seed_by_layer, gauge_seed_by_block_kv, gauge_band_angles = aop.parse_gauge_result(path)
    if gauge_band_angles is not None:
        return srg.apply_rope_gauge_band_angles(
            flat_state,
            band_angles=gauge_band_angles,
            transform=transform,
            layers=gauge_layers,
        )
    return srg.apply_rope_gauge_transform(
        flat_state,
        seed=0,
        angle_scale=np.pi,
        transform=transform,
        parameterization="banded_phase",
        num_bands=4,
        layers=gauge_layers,
        seed_by_layer=gauge_seed_by_layer,
        seed_by_block_kv=gauge_seed_by_block_kv,
    )


def parse_sensitivity_ranking(path: Path | None) -> dict[tuple[int, int], float]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    ranking: dict[tuple[int, int], float] = {}
    for item in data.get("per_block_kv", []):
        key = (int(item["layer"]), int(item["kv_head"]))
        ranking[key] = float(item.get("ce_loss_from_ablation", 0.0))
    return ranking


def infer_k_units(flat_state: dict[str, object]) -> list[dict[str, int | str]]:
    config = aqe.infer_model_config(flat_state)
    head_dim = int(config["model_dim"]) // int(config["num_heads"])
    attention_blocks = config["attention_blocks"]
    units: list[dict[str, int | str]] = []
    for block in attention_blocks:
        name = str(block["k_name"])
        layer_idx = int(block["index"])
        if name not in flat_state:
            continue
        rows = int(flat_state[name].shape[0])
        num_kv_heads = rows // head_dim
        for kv_head_idx in range(num_kv_heads):
            row_start = kv_head_idx * head_dim
            row_end = row_start + head_dim
            units.append(
                {
                    "name": name,
                    "layer": layer_idx,
                    "kv_head": kv_head_idx,
                    "row_start": row_start,
                    "row_end": row_end,
                }
            )
    return units


def rank_units(
    units: list[dict[str, int | str]],
    sensitivity_ranking: dict[tuple[int, int], float],
    *,
    layers: tuple[int, ...] | None,
    top_k: int,
) -> list[dict[str, int | str]]:
    layer_set = set(layers) if layers is not None else None
    filtered = [
        unit
        for unit in units
        if layer_set is None or int(unit["layer"]) in layer_set
    ]
    filtered.sort(
        key=lambda unit: (
            -float(sensitivity_ranking.get((int(unit["layer"]), int(unit["kv_head"])), 0.0)),
            int(unit["layer"]),
            int(unit["kv_head"]),
        )
    )
    if top_k > 0:
        return filtered[:top_k]
    return filtered


def unit_key(unit: dict[str, int | str]) -> tuple[int, int]:
    return int(unit["layer"]), int(unit["kv_head"])


def build_slice_library(
    flat_state: dict[str, object],
    units: list[dict[str, int | str]],
    *,
    prod_bits: tuple[int, ...],
    block_size: int,
) -> dict[tuple[str, int, int], dict[str, object]]:
    library: dict[tuple[str, int, int], dict[str, object]] = {}
    for unit in units:
        name = str(unit["name"])
        kv_head = int(unit["kv_head"])
        row_start = int(unit["row_start"])
        row_end = int(unit["row_end"])
        weight = flat_state[name]
        slice_arr = weight[row_start:row_end]
        for bits in prod_bits:
            meta, _deq, byte_stats = aqe.quantize_turbo_tensor_row_slices(
                slice_arr,
                [{"row_start": 0, "row_end": int(slice_arr.shape[0]), "mode": "prod", "bits": bits}],
                block_size=block_size,
            )
            part_meta = dict(meta["parts"][0])
            library[(name, kv_head, bits)] = {
                "meta": part_meta,
                "payload_bytes": int(byte_stats["payload_bytes"]),
            }
    return library


def estimate_payload_bytes(quant_obj: dict[str, object]) -> int:
    payload = 0
    for value in quant_obj.get("passthrough", {}).values():
        payload += int(np.asarray(value).nbytes)
    for value in quant_obj.get("quantized", {}).values():
        payload += int(np.asarray(value).nbytes)
    for value in quant_obj.get("scales", {}).values():
        payload += int(np.asarray(value).nbytes)
    for meta in quant_obj.get("turbo", {}).values():
        payload += int(aqe.turbo_meta_payload_breakdown(meta)["payload_bytes"])
    return int(payload)


def build_candidate(
    *,
    flat_state: dict[str, object],
    base_quant_obj: dict[str, object],
    base_stats: dict[str, object],
    base_deq_state: dict[str, object],
    global_codebook_overrides: dict[tuple[str, int, int], np.ndarray],
    tensor_codebook_overrides: dict[str, dict[tuple[str, int, int], np.ndarray]],
    units_by_name: dict[str, list[dict[str, int | str]]],
    slice_library: dict[tuple[str, int, int], dict[str, object]],
    allocation: dict[tuple[int, int], int],
    default_prod_bits: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    quant_obj = dict(base_quant_obj)
    quant_obj["turbo"] = dict(base_quant_obj.get("turbo", {}))
    deq_state = dict(base_deq_state)

    def merged_override_for(name: str) -> dict[tuple[str, int, int], np.ndarray] | None:
        override_map: dict[tuple[str, int, int], np.ndarray] = {}
        override_map.update(global_codebook_overrides)
        per_tensor = tensor_codebook_overrides.get(name)
        if per_tensor is not None:
            override_map.update(per_tensor)
        return override_map or None

    override_names: set[str] = set()
    for name, tensor_units in units_by_name.items():
        for unit in tensor_units:
            key = unit_key(unit)
            if int(allocation.get(key, default_prod_bits)) != int(default_prod_bits):
                override_names.add(name)
                break

    for name in override_names:
        tensor_units = units_by_name[name]
        tensor_units_sorted = sorted(tensor_units, key=lambda item: int(item["kv_head"]))
        parts: list[dict[str, object]] = []
        for unit in tensor_units_sorted:
            key = unit_key(unit)
            bits = int(allocation.get(key, default_prod_bits))
            entry = slice_library[(name, int(unit["kv_head"]), bits)]
            parts.append(entry["meta"])
        quant_obj["turbo"][name] = {
            "scheme": "turbo_sliced_rows_v1",
            "axis": 0,
            "shape": tuple(int(x) for x in flat_state[name].shape),
            "dtype": str(flat_state[name].dtype).split(".")[-1],
            "parts": parts,
        }
        deq_state[name] = aqe.dequantize_turbo_meta(
            quant_obj["turbo"][name],
            merged_override_for(name),
        ).astype(flat_state[name].dtype)
    payload_bytes = estimate_payload_bytes(quant_obj)
    stats = {
        "payload_bytes": int(payload_bytes),
        "int8_payload_bytes": int(payload_bytes),
        "param_count": int(base_stats["param_count"]),
        "baseline_tensor_bytes": int(base_stats["baseline_tensor_bytes"]),
    }
    return quant_obj, stats, deq_state


def summarize_candidate(
    *,
    quant_obj: dict[str, object],
    stats: dict[str, object],
    deq_state: dict[str, object],
    proxy_ctx: dict[str, object],
    baseline_bytes: dict[str, object],
    bytes_weight_per_kib: float,
) -> dict[str, object]:
    bytes_info = aqe.summarize_quant_bytes(quant_obj, stats)
    proxy = aop.summarize_search_proxy(
        aop.analyze_compare_state_proxies(
            proxy_ctx["ref_model"],
            deq_state,
            proxy_ctx["val_tokens"],
            int(proxy_ctx["seq_len"]),
            int(proxy_ctx["max_seqs"]),
            int(proxy_ctx["batch_seqs"]),
        )
    )
    delta_kib = (int(bytes_info["zlib_bytes"]) - int(baseline_bytes["zlib_bytes"])) / 1024.0
    score = float(proxy["mean_ce_delta"]) + float(bytes_weight_per_kib) * float(delta_kib)
    return {
        "bytes": bytes_info,
        "output_proxy": proxy,
        "score": float(score),
    }


def selection_key(candidate: dict[str, object]) -> tuple[float, ...]:
    proxy = candidate["output_proxy"]
    return (
        float(candidate["score"]),
        float(proxy["mean_ce_delta"]),
        -float(proxy["mean_margin_delta"]),
        -float(proxy["top1_true_rate"]),
        int(candidate["bytes"]["zlib_bytes"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedy K-head Turbo bit allocation using true-token CE plus bytes.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--codebook-result", type=Path)
    parser.add_argument("--init-bitalloc-result", type=Path)
    parser.add_argument("--sensitivity-result", type=Path)
    parser.add_argument("--layers")
    parser.add_argument("--top-k-units", type=int, default=12)
    parser.add_argument("--candidate-prod-bits", default="3,4,5")
    parser.add_argument("--bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-pair-rounds", type=int, default=0)
    parser.add_argument("--max-extra-zlib-bytes", type=int, default=0)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--proxy-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-max-seqs", type=int, default=8)
    parser.add_argument("--proxy-batch-seqs", type=int, default=1)
    parser.add_argument("--confirm-val-max-seqs", type=int, default=0)
    parser.add_argument("--confirm-val-batch-size", type=int, default=262144)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-stride", type=int, default=0)
    parser.add_argument("--confirm-eval-batch-seqs", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    flat_state, gauge_meta = parse_gauge_result(flat_state, args.gauge_result, transform=args.gauge_transform)

    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1 or schemes[0]["kind"] != "turbo":
        raise SystemExit("Expected exactly one Turbo scheme")
    scheme = schemes[0]
    prod_bits = parse_prod_bits(args.candidate_prod_bits)
    default_prod_bits = int(scheme["prod_bits"])
    if default_prod_bits not in prod_bits:
        prod_bits = tuple(sorted(set(prod_bits + (default_prod_bits,))))

    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)

    base_quant_obj, base_stats, base_deq_state = aqe.realize_scheme(
        flat_state,
        scheme,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    global_codebook_overrides: dict[tuple[str, int, int], np.ndarray] = {}
    tensor_codebook_overrides: dict[str, dict[tuple[str, int, int], np.ndarray]] = {}
    codebook_meta = None
    if args.codebook_result is not None:
        global_overrides_raw, tensor_overrides_raw, codebook_meta = emt.load_codebook_overrides(
            args.codebook_result,
            scheme=scheme,
        )
        if global_overrides_raw:
            base_quant_obj = dict(base_quant_obj)
            base_quant_obj["turbo_codebook_overrides"] = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in global_overrides_raw.items()
            }
            global_codebook_overrides = {
                (key.split(":")[0], int(key.split(":")[1]), int(key.split(":")[2])): np.asarray(value, dtype=np.float32)
                for key, value in global_overrides_raw.items()
            }
        if tensor_overrides_raw:
            base_quant_obj = dict(base_quant_obj)
            base_quant_obj["turbo_codebook_overrides_by_tensor"] = {
                name: {
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in overrides.items()
                }
                for name, overrides in tensor_overrides_raw.items()
            }
            tensor_codebook_overrides = {
                name: {
                    (key.split(":")[0], int(key.split(":")[1]), int(key.split(":")[2])): np.asarray(value, dtype=np.float32)
                    for key, value in overrides.items()
                }
                for name, overrides in tensor_overrides_raw.items()
            }
        if global_codebook_overrides or tensor_codebook_overrides:
            base_deq_state = aqe.dequantize_quant_obj(
                base_quant_obj,
                global_codebook_overrides,
                tensor_codebook_overrides,
            )
    baseline_bytes = aqe.summarize_quant_bytes(base_quant_obj, base_stats)
    proxy_ctx = aop.build_proxy_context(
        flat_state,
        args.data_path,
        args.tokenizer_path,
        args.proxy_seq_len,
        args.proxy_max_seqs,
        args.proxy_batch_seqs,
    )
    eval_ctx = None
    if args.confirm_val_max_seqs > 0:
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
    baseline = summarize_candidate(
        quant_obj=base_quant_obj,
        stats=base_stats,
        deq_state=base_deq_state,
        proxy_ctx=proxy_ctx,
        baseline_bytes=baseline_bytes,
        bytes_weight_per_kib=args.bytes_weight_per_kib,
    )

    sensitivity_ranking = parse_sensitivity_ranking(args.sensitivity_result)
    all_units = infer_k_units(flat_state)
    active_units = rank_units(
        all_units,
        sensitivity_ranking,
        layers=parse_int_list(args.layers),
        top_k=int(args.top_k_units),
    )
    units_by_name: dict[str, list[dict[str, int | str]]] = {}
    for unit in all_units:
        units_by_name.setdefault(str(unit["name"]), []).append(unit)
    slice_library = build_slice_library(
        flat_state,
        all_units,
        prod_bits=prod_bits,
        block_size=int(scheme["block_size"]),
    )

    init_bitalloc_meta = None
    allocation: dict[tuple[int, int], int] = {}
    if args.init_bitalloc_result is not None:
        init_data = json.loads(args.init_bitalloc_result.read_text())
        init_bitalloc_meta = {
            "source": str(args.init_bitalloc_result),
            "allocation": init_data.get("final", {}).get("allocation", {}),
            "eval": init_data.get("final", {}).get("eval"),
        }
        allocation = parse_allocation_map(init_data.get("final", {}).get("allocation"))
    cache: dict[tuple[tuple[int, int, int], ...], dict[str, object]] = {}
    history: list[dict[str, object]] = []

    def evaluate_current(current_allocation: dict[tuple[int, int], int]) -> dict[str, object]:
        key = tuple(sorted((layer, kv_head, bits) for (layer, kv_head), bits in current_allocation.items()))
        cached = cache.get(key)
        if cached is not None:
            return cached
        quant_obj, stats, deq_state = build_candidate(
            flat_state=flat_state,
            base_quant_obj=base_quant_obj,
            base_stats=base_stats,
            base_deq_state=base_deq_state,
            global_codebook_overrides=global_codebook_overrides,
            tensor_codebook_overrides=tensor_codebook_overrides,
            units_by_name=units_by_name,
            slice_library=slice_library,
            allocation=current_allocation,
            default_prod_bits=default_prod_bits,
        )
        summary = summarize_candidate(
            quant_obj=quant_obj,
            stats=stats,
            deq_state=deq_state,
            proxy_ctx=proxy_ctx,
            baseline_bytes=baseline_bytes,
            bytes_weight_per_kib=args.bytes_weight_per_kib,
        )
        summary["allocation"] = {
            f"{layer}:{kv_head}": int(bits)
            for (layer, kv_head), bits in sorted(current_allocation.items())
        }
        cache[key] = summary
        return summary

    incumbent = evaluate_current(allocation)
    for round_idx in range(max(int(args.max_rounds), 0)):
        improved = False
        for unit in active_units:
            key = unit_key(unit)
            current_bits = int(allocation.get(key, default_prod_bits))
            best_local = incumbent
            best_bits = current_bits
            for bits in prod_bits:
                if bits == current_bits:
                    continue
                candidate_alloc = dict(allocation)
                if bits == default_prod_bits:
                    candidate_alloc.pop(key, None)
                else:
                    candidate_alloc[key] = int(bits)
                candidate = evaluate_current(candidate_alloc)
                if args.max_extra_zlib_bytes > 0:
                    extra_bytes = int(candidate["bytes"]["zlib_bytes"]) - int(baseline_bytes["zlib_bytes"])
                    if extra_bytes > int(args.max_extra_zlib_bytes):
                        continue
                if selection_key(candidate) < selection_key(best_local):
                    best_local = candidate
                    best_bits = bits
            accepted = best_bits != current_bits
            if accepted:
                if best_bits == default_prod_bits:
                    allocation.pop(key, None)
                else:
                    allocation[key] = int(best_bits)
                incumbent = best_local
                improved = True
            history.append(
                {
                    "round": round_idx,
                    "layer": int(unit["layer"]),
                    "kv_head": int(unit["kv_head"]),
                    "accepted": bool(accepted),
                    "bits": int(best_bits),
                    "score": float(incumbent["score"]),
                    "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                    "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                    "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
                }
            )
        if not improved:
            break

    for pair_round_idx in range(max(int(args.max_pair_rounds), 0)):
        best_pair = incumbent
        best_pair_alloc: dict[tuple[int, int], int] | None = None
        best_pair_meta: tuple[dict[str, int | str], int, dict[str, int | str], int] | None = None
        for idx_a, unit_a in enumerate(active_units):
            key_a = unit_key(unit_a)
            current_a = int(allocation.get(key_a, default_prod_bits))
            for bits_a in prod_bits:
                if bits_a == current_a:
                    continue
                delta_a = bits_a - current_a
                for idx_b in range(idx_a + 1, len(active_units)):
                    unit_b = active_units[idx_b]
                    key_b = unit_key(unit_b)
                    current_b = int(allocation.get(key_b, default_prod_bits))
                    for bits_b in prod_bits:
                        if bits_b == current_b:
                            continue
                        delta_b = bits_b - current_b
                        # Pair swaps are for genuine reallocations, not double upgrades/downgrades.
                        if delta_a * delta_b >= 0:
                            continue
                        candidate_alloc = dict(allocation)
                        if bits_a == default_prod_bits:
                            candidate_alloc.pop(key_a, None)
                        else:
                            candidate_alloc[key_a] = int(bits_a)
                        if bits_b == default_prod_bits:
                            candidate_alloc.pop(key_b, None)
                        else:
                            candidate_alloc[key_b] = int(bits_b)
                        candidate = evaluate_current(candidate_alloc)
                        if args.max_extra_zlib_bytes > 0:
                            extra_bytes = int(candidate["bytes"]["zlib_bytes"]) - int(baseline_bytes["zlib_bytes"])
                            if extra_bytes > int(args.max_extra_zlib_bytes):
                                continue
                        if selection_key(candidate) < selection_key(best_pair):
                            best_pair = candidate
                            best_pair_alloc = candidate_alloc
                            best_pair_meta = (unit_a, int(bits_a), unit_b, int(bits_b))
        if best_pair_alloc is None or best_pair_meta is None:
            break
        allocation = best_pair_alloc
        incumbent = best_pair
        unit_a, bits_a, unit_b, bits_b = best_pair_meta
        history.append(
            {
                "round": f"pair_{pair_round_idx}",
                "layer": int(unit_a["layer"]),
                "kv_head": int(unit_a["kv_head"]),
                "bits": int(bits_a),
                "other_layer": int(unit_b["layer"]),
                "other_kv_head": int(unit_b["kv_head"]),
                "other_bits": int(bits_b),
                "accepted": True,
                "score": float(incumbent["score"]),
                "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "gauge_result": None if args.gauge_result is None else str(args.gauge_result),
        "codebook_result": None if args.codebook_result is None else str(args.codebook_result),
        "gauge_meta": gauge_meta,
        "codebook_meta": codebook_meta,
        "search": {
            "candidate_prod_bits": list(prod_bits),
            "default_prod_bits": default_prod_bits,
            "top_k_units": int(args.top_k_units),
            "bytes_weight_per_kib": float(args.bytes_weight_per_kib),
            "max_rounds": int(args.max_rounds),
            "max_pair_rounds": int(args.max_pair_rounds),
            "max_extra_zlib_bytes": int(args.max_extra_zlib_bytes),
        },
        "init_bitalloc": init_bitalloc_meta,
        "baseline": baseline,
        "active_units": [
            {
                "name": str(unit["name"]),
                "layer": int(unit["layer"]),
                "kv_head": int(unit["kv_head"]),
                "row_start": int(unit["row_start"]),
                "row_end": int(unit["row_end"]),
                "sensitivity": float(sensitivity_ranking.get((int(unit["layer"]), int(unit["kv_head"])), 0.0)),
            }
            for unit in active_units
        ],
        "final": incumbent,
        "history": history,
    }
    if eval_ctx is not None:
        _final_quant_obj, _final_stats, final_deq_state = build_candidate(
            flat_state=flat_state,
            base_quant_obj=base_quant_obj,
            base_stats=base_stats,
            base_deq_state=base_deq_state,
            global_codebook_overrides=global_codebook_overrides,
            tensor_codebook_overrides=tensor_codebook_overrides,
            units_by_name=units_by_name,
            slice_library=slice_library,
            allocation=allocation,
            default_prod_bits=default_prod_bits,
        )
        base_val_loss, base_val_bpb = aqe.eval_state(eval_ctx, base_deq_state)
        final_val_loss, final_val_bpb = aqe.eval_state(eval_ctx, final_deq_state)
        result["baseline"]["eval"] = {"val_loss": float(base_val_loss), "val_bpb": float(base_val_bpb)}
        result["final"]["eval"] = {"val_loss": float(final_val_loss), "val_bpb": float(final_val_bpb)}
        result["final"]["delta_bpb_vs_baseline_quant"] = float(final_val_bpb - base_val_bpb)
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
