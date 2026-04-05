#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
import export_mlx_taskaware_turbo as emt
import optimize_mlx_rope_gauge_continuous as ogc
import refine_mlx_turbo_codebook_ce as rcc
import turbo_quant_mlx as tq


def codebook_override_key(scheme: dict[str, object], refine_mode: str) -> tuple[str, int, int]:
    if refine_mode != "prod":
        raise ValueError(f"Unsupported refine_mode={refine_mode}")
    return ("prod", int(scheme["prod_bits"]) - 1, int(scheme["block_size"]))


def codebook_override_string(override_key: tuple[str, int, int]) -> str:
    mode, mse_bits, block_size = override_key
    return f"{mode}:{mse_bits}:{block_size}"


def default_codebook(scheme: dict[str, object], refine_mode: str) -> np.ndarray:
    mode, mse_bits, block_size = codebook_override_key(scheme, refine_mode)
    if mode != "prod":
        raise ValueError(f"Unsupported mode={mode}")
    return np.asarray(tq._codebook_np(mse_bits, block_size), dtype=np.float32)


def maybe_load_codebook(
    scheme: dict[str, object],
    refine_mode: str,
    result_path: Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object] | None]:
    if result_path is None:
        return default_codebook(scheme, refine_mode), {}, None
    overrides, tensor_overrides, meta = emt.load_codebook_overrides(result_path, scheme=scheme)
    tensor_codebooks: dict[str, np.ndarray] = {}
    if tensor_overrides:
        key = codebook_override_string(codebook_override_key(scheme, refine_mode))
        for name, per_tensor in tensor_overrides.items():
            arr = per_tensor.get(key)
            if arr is not None:
                tensor_codebooks[str(name)] = np.asarray(arr, dtype=np.float32)
    if not overrides:
        return default_codebook(scheme, refine_mode), tensor_codebooks, meta
    key = codebook_override_string(codebook_override_key(scheme, refine_mode))
    arr = overrides.get(key)
    if arr is None:
        raise ValueError(f"Codebook result {result_path} does not contain override {key}")
    return np.asarray(arr, dtype=np.float32), tensor_codebooks, meta


def build_candidate_quant_obj(
    candidate_state: dict[str, object],
    *,
    scheme: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    override_key: tuple[str, int, int],
    codebook: np.ndarray,
    tensor_codebooks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    quant_obj, stats, _ = aqe.realize_scheme(
        candidate_state,
        scheme,
        turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    quant_obj = dict(quant_obj)
    quant_obj["turbo_codebook_overrides"] = {
        codebook_override_string(override_key): np.asarray(codebook, dtype=np.float32)
    }
    tensor_override_tuples = None
    if tensor_codebooks:
        quant_obj["turbo_codebook_overrides_by_tensor"] = {
            name: {
                codebook_override_string(override_key): np.asarray(value, dtype=np.float32)
            }
            for name, value in tensor_codebooks.items()
        }
        tensor_override_tuples = {
            name: {override_key: np.asarray(value, dtype=np.float32)}
            for name, value in tensor_codebooks.items()
        }
    deq_state = aqe.dequantize_quant_obj(
        quant_obj,
        {override_key: np.asarray(codebook, dtype=np.float32)},
        tensor_override_tuples,
    )
    return quant_obj, stats, deq_state


def summarize_joint_candidate(
    flat_state: dict[str, object],
    *,
    band_angles: np.ndarray,
    codebook: np.ndarray,
    transform: str,
    layers: tuple[int, ...],
    scheme: dict[str, object],
    output_proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    baseline_quant: dict[str, object],
    bytes_weight_per_kib: float,
    override_key: tuple[str, int, int],
    tensor_codebooks: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    candidate_state, gauge_meta = ogc.srg.apply_rope_gauge_band_angles(
        flat_state,
        band_angles=band_angles,
        transform=transform,
        layers=layers,
    )
    quant_obj, stats, deq_state = build_candidate_quant_obj(
        candidate_state,
        scheme=scheme,
        turbo_embed_export=turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
        override_key=override_key,
        codebook=codebook,
        tensor_codebooks=tensor_codebooks,
    )
    bytes_info = aqe.summarize_quant_bytes(quant_obj, stats)
    proxy_summary = aop.summarize_search_proxy(
        aop.analyze_compare_state_proxies(
            output_proxy_ctx["ref_model"],
            deq_state,
            output_proxy_ctx["val_tokens"],
            int(output_proxy_ctx["seq_len"]),
            int(output_proxy_ctx["max_seqs"]),
            int(output_proxy_ctx["batch_seqs"]),
        )
    )
    delta_kib = (int(bytes_info["zlib_bytes"]) - int(baseline_quant["bytes"]["zlib_bytes"])) / 1024.0
    score = float(proxy_summary["mean_ce_delta"]) + float(bytes_weight_per_kib) * float(delta_kib)
    return {
        "gauge": gauge_meta,
        "bytes": bytes_info,
        "output_proxy": proxy_summary,
        "score": float(score),
        "delta_ce_vs_baseline_quant": float(
            proxy_summary["mean_ce_delta"] - float(output_proxy_ctx["baseline_output_proxy"]["mean_ce_delta"])
        ),
        "delta_margin_vs_baseline_quant": float(
            proxy_summary["mean_margin_delta"] - float(output_proxy_ctx["baseline_output_proxy"]["mean_margin_delta"])
        ),
        "delta_zlib_bytes_vs_baseline_quant": int(bytes_info["zlib_bytes"]) - int(baseline_quant["bytes"]["zlib_bytes"]),
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


def optimize_gauge_phase(
    *,
    cache: dict[tuple[bytes, bytes], dict[str, object]],
    flat_state: dict[str, object],
    base_band_angles: np.ndarray,
    current_codebook: np.ndarray,
    transform: str,
    active_layers: tuple[int, ...],
    active_kv_heads: tuple[int, ...],
    scheme: dict[str, object],
    output_proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    baseline_quant: dict[str, object],
    bytes_weight_per_kib: float,
    override_key: tuple[str, int, int],
    tensor_codebooks: dict[str, np.ndarray] | None,
    initial_step: float,
    min_step: float,
    step_decay: float,
    max_rounds_per_step: int,
    include_dense: bool,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]], bool]:
    global_adjust = 0.0
    layer_adjust = np.zeros((len(active_layers),), dtype=np.float32)
    kv_adjust = np.zeros((len(active_kv_heads),), dtype=np.float32)
    band_adjust = np.zeros((base_band_angles.shape[-1],), dtype=np.float32)
    dense_adjust = (
        np.zeros((len(active_layers), len(active_kv_heads), base_band_angles.shape[-1]), dtype=np.float32)
        if include_dense
        else None
    )
    history: list[dict[str, object]] = []

    def evaluate(band_angles: np.ndarray) -> dict[str, object]:
        key = (band_angles.astype(np.float32, copy=False).tobytes(), current_codebook.astype(np.float32, copy=False).tobytes())
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = summarize_joint_candidate(
            flat_state,
            band_angles=band_angles,
            codebook=current_codebook,
            transform=transform,
            layers=active_layers,
            scheme=scheme,
            output_proxy_ctx=output_proxy_ctx,
            turbo_embed_export=turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
            baseline_quant=baseline_quant,
            bytes_weight_per_kib=bytes_weight_per_kib,
            override_key=override_key,
            tensor_codebooks=tensor_codebooks,
        )
        cache[key] = result
        return result

    current_band_angles = np.array(base_band_angles, copy=True, dtype=np.float32)
    incumbent = evaluate(current_band_angles)
    params = ogc.param_records(active_layers, active_kv_heads, base_band_angles.shape[-1], include_dense)
    step = float(initial_step)
    any_improvement = False
    while step >= float(min_step):
        round_improved = False
        for round_idx in range(max(int(max_rounds_per_step), 1)):
            improved_this_round = False
            for record in params:
                current_value = ogc.get_param_value(
                    global_adjust,
                    layer_adjust,
                    kv_adjust,
                    band_adjust,
                    dense_adjust,
                    record,
                )
                best_local = incumbent
                best_value = current_value
                best_band_angles = current_band_angles
                for delta in (step, -step):
                    g2, l2, k2, b2, d2 = ogc.set_param_value(
                        global_adjust,
                        layer_adjust,
                        kv_adjust,
                        band_adjust,
                        dense_adjust,
                        record,
                        current_value + delta,
                    )
                    trial_band_angles = ogc.render_band_angles(
                        base_band_angles,
                        active_layers=active_layers,
                        active_kv_heads=active_kv_heads,
                        global_adjust=g2,
                        layer_adjust=l2,
                        kv_adjust=k2,
                        band_adjust=b2,
                        dense_adjust=d2,
                    )
                    candidate = evaluate(trial_band_angles)
                    if selection_key(candidate) < selection_key(best_local):
                        best_local = candidate
                        best_value = current_value + delta
                        best_band_angles = trial_band_angles
                accepted = not math.isclose(best_value, current_value, rel_tol=0.0, abs_tol=1e-12)
                if accepted:
                    global_adjust, layer_adjust, kv_adjust, band_adjust, dense_adjust = ogc.set_param_value(
                        global_adjust,
                        layer_adjust,
                        kv_adjust,
                        band_adjust,
                        dense_adjust,
                        record,
                        best_value,
                    )
                    incumbent = best_local
                    current_band_angles = best_band_angles
                    any_improvement = True
                    round_improved = True
                    improved_this_round = True
                history.append(
                    {
                        "phase": "gauge",
                        "step_size": float(step),
                        "round": int(round_idx),
                        "group": record["group"],
                        "index": list(record["index"]),
                        "accepted": bool(accepted),
                        "new_value": float(best_value),
                        "score": float(incumbent["score"]),
                        "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                        "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                        "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
                    }
                )
            if not improved_this_round:
                break
        if not round_improved:
            step *= float(step_decay)
        else:
            base_band_angles = np.array(current_band_angles, copy=True)
            global_adjust = 0.0
            layer_adjust.fill(0.0)
            kv_adjust.fill(0.0)
            band_adjust.fill(0.0)
            if dense_adjust is not None:
                dense_adjust.fill(0.0)
    return current_band_angles, incumbent, history, any_improvement


def optimize_codebook_phase(
    *,
    cache: dict[tuple[bytes, bytes], dict[str, object]],
    flat_state: dict[str, object],
    current_band_angles: np.ndarray,
    current_codebook: np.ndarray,
    transform: str,
    active_layers: tuple[int, ...],
    scheme: dict[str, object],
    output_proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    baseline_quant: dict[str, object],
    bytes_weight_per_kib: float,
    override_key: tuple[str, int, int],
    tensor_codebooks: dict[str, np.ndarray] | None,
    initial_step: float,
    min_step: float,
    step_decay: float,
    max_rounds_per_step: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]], bool]:
    history: list[dict[str, object]] = []

    def evaluate(codebook: np.ndarray) -> dict[str, object]:
        key = (current_band_angles.astype(np.float32, copy=False).tobytes(), codebook.astype(np.float32, copy=False).tobytes())
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = summarize_joint_candidate(
            flat_state,
            band_angles=current_band_angles,
            codebook=codebook,
            transform=transform,
            layers=active_layers,
            scheme=scheme,
            output_proxy_ctx=output_proxy_ctx,
            turbo_embed_export=turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
            baseline_quant=baseline_quant,
            bytes_weight_per_kib=bytes_weight_per_kib,
            override_key=override_key,
            tensor_codebooks=tensor_codebooks,
        )
        cache[key] = result
        return result

    incumbent = evaluate(current_codebook)
    pos_values = np.asarray(current_codebook[len(current_codebook) // 2 :], dtype=np.float32).copy()
    step = float(initial_step)
    any_improvement = False
    while step >= float(min_step):
        round_improved = False
        for round_idx in range(max(int(max_rounds_per_step), 1)):
            improved_this_round = False
            for idx in range(pos_values.size):
                current_value = float(pos_values[idx])
                best_local = incumbent
                best_value = current_value
                best_codebook = current_codebook
                for delta in (step, -step):
                    trial = np.array(pos_values, copy=True)
                    trial[idx] = current_value + delta
                    codebook = rcc.build_symmetric_codebook(trial)
                    candidate = evaluate(codebook)
                    if selection_key(candidate) < selection_key(best_local):
                        best_local = candidate
                        best_value = float(codebook[len(codebook) // 2 + idx])
                        best_codebook = codebook
                accepted = not math.isclose(best_value, current_value, rel_tol=0.0, abs_tol=1e-12)
                if accepted:
                    pos_values[idx] = best_value
                    current_codebook = np.array(best_codebook, copy=True)
                    pos_values = np.asarray(current_codebook[len(current_codebook) // 2 :], dtype=np.float32).copy()
                    incumbent = best_local
                    any_improvement = True
                    round_improved = True
                    improved_this_round = True
                history.append(
                    {
                        "phase": "codebook",
                        "step_size": float(step),
                        "round": int(round_idx),
                        "index": int(idx),
                        "accepted": bool(accepted),
                        "new_value": float(best_value),
                        "score": float(incumbent["score"]),
                        "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                        "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                        "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
                    }
                )
            if not improved_this_round:
                break
        if not round_improved:
            step *= float(step_decay)
    return current_codebook, incumbent, history, any_improvement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint task-aware optimization of exact gauge angles and Turbo prod codebook.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--gauge-layers")
    parser.add_argument("--gauge-kv-heads")
    parser.add_argument("--gauge-num-bands", type=int, default=4)
    parser.add_argument("--gauge-init-angle-scale", type=float, default=math.pi)
    parser.add_argument(
        "--gauge-init-parameterization",
        choices=("global_head_phase", "banded_phase", "full_pair_phase"),
        default="banded_phase",
    )
    parser.add_argument("--gauge-dense-refine", action="store_true")
    parser.add_argument("--codebook-result", type=Path)
    parser.add_argument("--refine-mode", choices=("prod",), default="prod")
    parser.add_argument("--outer-loops", type=int, default=2)
    parser.add_argument("--bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--gauge-initial-step", type=float, default=0.0375)
    parser.add_argument("--gauge-min-step", type=float, default=0.01)
    parser.add_argument("--gauge-step-decay", type=float, default=0.5)
    parser.add_argument("--gauge-max-rounds-per-step", type=int, default=1)
    parser.add_argument("--codebook-initial-step", type=float, default=0.0025)
    parser.add_argument("--codebook-min-step", type=float, default=0.0005)
    parser.add_argument("--codebook-step-decay", type=float, default=0.5)
    parser.add_argument("--codebook-max-rounds-per-step", type=int, default=2)
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
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_flat_state = aqe.load_flat_state(args.checkpoint)
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1 or schemes[0]["kind"] != "turbo":
        raise SystemExit("Expected exactly one Turbo scheme")
    scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)

    eval_ctx = None
    if args.confirm_val_max_seqs > 0:
        config = aqe.infer_model_config(raw_flat_state)
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
    baseline_quant = aqe.analyze_scheme(
        raw_flat_state,
        scheme,
        eval_ctx,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    output_proxy_ctx = aop.build_proxy_context(
        raw_flat_state,
        args.data_path,
        args.tokenizer_path,
        args.proxy_seq_len,
        args.proxy_max_seqs,
        args.proxy_batch_seqs,
    )
    baseline_compare_state, _ = aop.make_compare_state(
        raw_flat_state,
        scheme=scheme,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
    )
    baseline_output_proxy = aop.summarize_search_proxy(
        aop.analyze_compare_state_proxies(
            output_proxy_ctx["ref_model"],
            baseline_compare_state,
            output_proxy_ctx["val_tokens"],
            int(output_proxy_ctx["seq_len"]),
            int(output_proxy_ctx["max_seqs"]),
            int(output_proxy_ctx["batch_seqs"]),
        )
    )
    output_proxy_ctx = dict(output_proxy_ctx)
    output_proxy_ctx["baseline_output_proxy"] = baseline_output_proxy

    init_band_angles, init_gauge_meta = ogc.parse_init_result(
        args.gauge_result,
        raw_flat_state,
        angle_scale=args.gauge_init_angle_scale,
        parameterization=args.gauge_init_parameterization,
        num_bands=args.gauge_num_bands,
    )
    active_layers, active_kv_heads = ogc.build_active_sets(
        raw_flat_state,
        ogc.parse_int_list(args.gauge_layers),
        ogc.parse_int_list(args.gauge_kv_heads),
        init_band_angles,
    )
    current_band_angles = np.array(init_band_angles, copy=True, dtype=np.float32)
    current_codebook, current_tensor_codebooks, init_codebook_meta = maybe_load_codebook(scheme, args.refine_mode, args.codebook_result)
    override_key = codebook_override_key(scheme, args.refine_mode)
    cache: dict[tuple[bytes, bytes], dict[str, object]] = {}

    incumbent = summarize_joint_candidate(
        raw_flat_state,
        band_angles=current_band_angles,
        codebook=current_codebook,
        transform=args.gauge_transform,
        layers=active_layers,
        scheme=scheme,
        output_proxy_ctx=output_proxy_ctx,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
        baseline_quant=baseline_quant,
        bytes_weight_per_kib=args.bytes_weight_per_kib,
        override_key=override_key,
        tensor_codebooks=current_tensor_codebooks,
    )
    cache[(current_band_angles.astype(np.float32, copy=False).tobytes(), current_codebook.astype(np.float32, copy=False).tobytes())] = incumbent

    history: list[dict[str, object]] = []
    outer_history: list[dict[str, object]] = []
    for outer_idx in range(max(int(args.outer_loops), 1)):
        current_band_angles, incumbent, gauge_history, gauge_improved = optimize_gauge_phase(
            cache=cache,
            flat_state=raw_flat_state,
            base_band_angles=current_band_angles,
            current_codebook=current_codebook,
            transform=args.gauge_transform,
            active_layers=active_layers,
            active_kv_heads=active_kv_heads,
            scheme=scheme,
            output_proxy_ctx=output_proxy_ctx,
            turbo_embed_export=args.turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
            baseline_quant=baseline_quant,
            bytes_weight_per_kib=args.bytes_weight_per_kib,
            override_key=override_key,
            tensor_codebooks=current_tensor_codebooks,
            initial_step=args.gauge_initial_step,
            min_step=args.gauge_min_step,
            step_decay=args.gauge_step_decay,
            max_rounds_per_step=args.gauge_max_rounds_per_step,
            include_dense=args.gauge_dense_refine,
        )
        current_codebook, incumbent, codebook_history, codebook_improved = optimize_codebook_phase(
            cache=cache,
            flat_state=raw_flat_state,
            current_band_angles=current_band_angles,
            current_codebook=current_codebook,
            transform=args.gauge_transform,
            active_layers=active_layers,
            scheme=scheme,
            output_proxy_ctx=output_proxy_ctx,
            turbo_embed_export=args.turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
            baseline_quant=baseline_quant,
            bytes_weight_per_kib=args.bytes_weight_per_kib,
            override_key=override_key,
            tensor_codebooks=current_tensor_codebooks,
            initial_step=args.codebook_initial_step,
            min_step=args.codebook_min_step,
            step_decay=args.codebook_step_decay,
            max_rounds_per_step=args.codebook_max_rounds_per_step,
        )
        history.extend(gauge_history)
        history.extend(codebook_history)
        outer_history.append(
            {
                "outer_loop": int(outer_idx),
                "gauge_improved": bool(gauge_improved),
                "codebook_improved": bool(codebook_improved),
                "score": float(incumbent["score"]),
                "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
            }
        )
        if not gauge_improved and not codebook_improved:
            break

    final_candidate_state, final_gauge_meta = ogc.srg.apply_rope_gauge_band_angles(
        raw_flat_state,
        band_angles=current_band_angles,
        transform=args.gauge_transform,
        layers=active_layers,
    )
    final_quant_obj, final_stats, final_deq_state = build_candidate_quant_obj(
        final_candidate_state,
        scheme=scheme,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
        override_key=override_key,
        codebook=current_codebook,
        tensor_codebooks=current_tensor_codebooks,
    )

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "search": {
            "transform": args.gauge_transform,
            "active_layers": list(active_layers),
            "active_kv_heads": list(active_kv_heads),
            "gauge_num_bands": int(args.gauge_num_bands),
            "bytes_weight_per_kib": float(args.bytes_weight_per_kib),
            "outer_loops": int(args.outer_loops),
        },
        "init": {
            "gauge": init_gauge_meta,
            "codebook": init_codebook_meta,
        },
        "baseline_quant": baseline_quant,
        "baseline_output_proxy": baseline_output_proxy,
        "final": {
            "band_angles": current_band_angles.tolist(),
            "codebook": current_codebook.tolist(),
            "codebooks_by_tensor": {
                name: np.asarray(value, dtype=np.float32).tolist()
                for name, value in sorted(current_tensor_codebooks.items())
            },
            "gauge": final_gauge_meta,
            "bytes": incumbent["bytes"],
            "output_proxy": incumbent["output_proxy"],
            "score": float(incumbent["score"]),
            "delta_ce_vs_baseline_quant": float(incumbent["delta_ce_vs_baseline_quant"]),
            "delta_margin_vs_baseline_quant": float(incumbent["delta_margin_vs_baseline_quant"]),
            "delta_zlib_bytes_vs_baseline_quant": int(incumbent["delta_zlib_bytes_vs_baseline_quant"]),
        },
        "outer_history": outer_history,
        "history": history,
    }

    if eval_ctx is not None:
        final_loss, final_bpb = aqe.eval_state(eval_ctx, final_deq_state)
        result["final"]["eval"] = {"val_loss": float(final_loss), "val_bpb": float(final_bpb)}
        result["final"]["delta_bpb_vs_baseline_quant"] = float(final_bpb - baseline_quant["eval"]["val_bpb"])

    if args.artifact_out is not None:
        quant_raw = pickle.dumps(final_quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        quant_blob = zlib.compress(quant_raw, level=9)
        args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_out.write_bytes(quant_blob)
        result["artifact"] = {
            "path": str(args.artifact_out),
            "artifact_bytes": int(len(quant_blob)),
            "raw_pickle_bytes": int(len(quant_raw)),
        }

    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
