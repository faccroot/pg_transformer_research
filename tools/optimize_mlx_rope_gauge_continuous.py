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

import analyze_mlx_output_proxies as aop
import analyze_mlx_quant_export as aqe
import search_mlx_rope_gauge as srg


def parse_int_list(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(values) if values else None


def wrap_angles(arr: np.ndarray) -> np.ndarray:
    return ((arr + math.pi) % (2.0 * math.pi)) - math.pi


def parse_init_result(
    path: Path | None,
    flat_state,
    *,
    angle_scale: float,
    parameterization: str,
    num_bands: int,
) -> tuple[np.ndarray, dict[str, object] | None]:
    if path is None:
        return srg.zero_band_angle_tensor(flat_state, num_bands), None
    data = json.loads(path.read_text())
    final = data.get("final", {})
    if "band_angles" in final:
        band_angles = np.asarray(final["band_angles"], dtype=np.float32)
        return band_angles, {"source": str(path), "kind": "continuous"}
    seed_by_layer_raw = final.get("seed_by_layer", {}) or {}
    seed_by_block_kv_raw = final.get("seed_by_block_kv", {}) or {}
    layers_raw = final.get("layers", [])
    layers = tuple(int(x) for x in layers_raw) if layers_raw else None
    seed_by_layer = {int(k): int(v) for k, v in seed_by_layer_raw.items()}
    seed_by_block_kv: dict[tuple[int, int], int] = {}
    for key, value in seed_by_block_kv_raw.items():
        layer_idx, kv_head_idx = (int(part) for part in str(key).split(":", 1))
        seed_by_block_kv[(layer_idx, kv_head_idx)] = int(value)
    band_angles = srg.band_angle_tensor_from_seed_assignments(
        flat_state,
        base_seed=0,
        angle_scale=angle_scale,
        parameterization=parameterization,
        num_bands=num_bands,
        layers=layers,
        seed_by_layer=seed_by_layer,
        seed_by_block_kv=seed_by_block_kv,
    )
    return band_angles, {
        "source": str(path),
        "kind": "discrete_seed_map",
        "layers": list(layers) if layers is not None else None,
        "seed_by_layer": {str(k): int(v) for k, v in sorted(seed_by_layer.items())},
        "seed_by_block_kv": {
            f"{layer_idx}:{kv_head_idx}": int(v)
            for (layer_idx, kv_head_idx), v in sorted(seed_by_block_kv.items())
        },
    }


def build_active_sets(
    flat_state,
    layers_arg: tuple[int, ...] | None,
    kv_heads_arg: tuple[int, ...] | None,
    init_band_angles: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cfg = srg.gauge_config(flat_state)
    if layers_arg is not None:
        layers = tuple(sorted(set(int(x) for x in layers_arg)))
    else:
        nonzero_layers = [idx for idx in range(cfg["num_blocks"]) if np.any(np.abs(init_band_angles[idx]) > 1e-9)]
        layers = tuple(nonzero_layers if nonzero_layers else range(cfg["num_blocks"]))
    if kv_heads_arg is not None:
        kv_heads = tuple(sorted(set(int(x) for x in kv_heads_arg)))
    else:
        kv_heads = tuple(range(cfg["num_kv_heads"]))
    return layers, kv_heads


def render_band_angles(
    base_band_angles: np.ndarray,
    *,
    active_layers: tuple[int, ...],
    active_kv_heads: tuple[int, ...],
    global_adjust: float,
    layer_adjust: np.ndarray,
    kv_adjust: np.ndarray,
    band_adjust: np.ndarray,
    dense_adjust: np.ndarray | None,
) -> np.ndarray:
    out = np.array(base_band_angles, copy=True, dtype=np.float32)
    for li, layer_idx in enumerate(active_layers):
        for ki, kv_head_idx in enumerate(active_kv_heads):
            out[layer_idx, kv_head_idx] += global_adjust
            out[layer_idx, kv_head_idx] += float(layer_adjust[li])
            out[layer_idx, kv_head_idx] += float(kv_adjust[ki])
            out[layer_idx, kv_head_idx] += band_adjust
            if dense_adjust is not None:
                out[layer_idx, kv_head_idx] += dense_adjust[li, ki]
    return wrap_angles(out.astype(np.float32, copy=False))


def summarize_candidate(
    flat_state,
    *,
    band_angles: np.ndarray,
    transform: str,
    layers: tuple[int, ...],
    scheme: dict[str, object],
    output_proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    baseline_quant: dict[str, object],
    bytes_weight_per_kib: float,
    eval_ctx: dict[str, object] | None,
) -> dict[str, object]:
    candidate_state, gauge_meta = srg.apply_rope_gauge_band_angles(
        flat_state,
        band_angles=band_angles,
        transform=transform,
        layers=layers,
    )
    quant_obj, stats, deq_state = aqe.realize_scheme(
        candidate_state,
        scheme,
        turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    bytes_info = aqe.summarize_quant_bytes(quant_obj, stats)
    proxy_metrics = aop.analyze_compare_state_proxies(
        output_proxy_ctx["ref_model"],
        deq_state,
        output_proxy_ctx["val_tokens"],
        int(output_proxy_ctx["seq_len"]),
        int(output_proxy_ctx["max_seqs"]),
        int(output_proxy_ctx["batch_seqs"]),
    )
    proxy_summary = aop.summarize_search_proxy(proxy_metrics)
    delta_kib = (int(bytes_info["zlib_bytes"]) - int(baseline_quant["bytes"]["zlib_bytes"])) / 1024.0
    score = float(proxy_summary["mean_ce_delta"]) + float(bytes_weight_per_kib) * float(delta_kib)
    result: dict[str, object] = {
        "gauge": gauge_meta,
        "bytes": bytes_info,
        "output_proxy": proxy_summary,
        "score": score,
        "delta_ce_vs_baseline_quant": float(
            proxy_summary["mean_ce_delta"] - float(output_proxy_ctx["baseline_output_proxy"]["mean_ce_delta"])
        ),
        "delta_margin_vs_baseline_quant": float(
            proxy_summary["mean_margin_delta"] - float(output_proxy_ctx["baseline_output_proxy"]["mean_margin_delta"])
        ),
        "delta_zlib_bytes_vs_baseline_quant": int(bytes_info["zlib_bytes"]) - int(baseline_quant["bytes"]["zlib_bytes"]),
    }
    if eval_ctx is not None:
        val_loss, val_bpb = aqe.eval_state(eval_ctx, deq_state)
        result["eval"] = {"val_loss": float(val_loss), "val_bpb": float(val_bpb)}
        result["delta_bpb_vs_baseline_quant"] = float(val_bpb - baseline_quant["eval"]["val_bpb"])
    return result


def selection_key(candidate: dict[str, object]) -> tuple[float, ...]:
    proxy = candidate["output_proxy"]
    return (
        float(candidate["score"]),
        float(proxy["mean_ce_delta"]),
        -float(proxy["mean_margin_delta"]),
        -float(proxy["top1_true_rate"]),
        int(candidate["bytes"]["zlib_bytes"]),
    )


def param_records(
    active_layers: tuple[int, ...],
    active_kv_heads: tuple[int, ...],
    num_bands: int,
    include_dense: bool,
) -> list[dict[str, object]]:
    params: list[dict[str, object]] = [{"group": "global", "index": ()}]
    params.extend({"group": "layer", "index": (li,)} for li in range(len(active_layers)))
    params.extend({"group": "kv", "index": (ki,)} for ki in range(len(active_kv_heads)))
    params.extend({"group": "band", "index": (bi,)} for bi in range(num_bands))
    if include_dense:
        for li in range(len(active_layers)):
            for ki in range(len(active_kv_heads)):
                for bi in range(num_bands):
                    params.append({"group": "dense", "index": (li, ki, bi)})
    return params


def get_param_value(
    global_adjust: float,
    layer_adjust: np.ndarray,
    kv_adjust: np.ndarray,
    band_adjust: np.ndarray,
    dense_adjust: np.ndarray | None,
    record: dict[str, object],
) -> float:
    group = record["group"]
    index = record["index"]
    if group == "global":
        return float(global_adjust)
    if group == "layer":
        return float(layer_adjust[index[0]])
    if group == "kv":
        return float(kv_adjust[index[0]])
    if group == "band":
        return float(band_adjust[index[0]])
    if group == "dense" and dense_adjust is not None:
        return float(dense_adjust[index[0], index[1], index[2]])
    raise ValueError(f"Bad parameter record: {record}")


def set_param_value(
    global_adjust: float,
    layer_adjust: np.ndarray,
    kv_adjust: np.ndarray,
    band_adjust: np.ndarray,
    dense_adjust: np.ndarray | None,
    record: dict[str, object],
    value: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    group = record["group"]
    index = record["index"]
    global_next = float(global_adjust)
    layer_next = np.array(layer_adjust, copy=True)
    kv_next = np.array(kv_adjust, copy=True)
    band_next = np.array(band_adjust, copy=True)
    dense_next = None if dense_adjust is None else np.array(dense_adjust, copy=True)
    if group == "global":
        global_next = float(value)
    elif group == "layer":
        layer_next[index[0]] = value
    elif group == "kv":
        kv_next[index[0]] = value
    elif group == "band":
        band_next[index[0]] = value
    elif group == "dense" and dense_next is not None:
        dense_next[index[0], index[1], index[2]] = value
    else:
        raise ValueError(f"Bad parameter record: {record}")
    return global_next, layer_next, kv_next, band_next, dense_next


def evaluate_params(
    *,
    cache: dict[bytes, dict[str, object]],
    flat_state,
    base_band_angles: np.ndarray,
    transform: str,
    active_layers: tuple[int, ...],
    active_kv_heads: tuple[int, ...],
    global_adjust: float,
    layer_adjust: np.ndarray,
    kv_adjust: np.ndarray,
    band_adjust: np.ndarray,
    dense_adjust: np.ndarray | None,
    scheme: dict[str, object],
    output_proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
    baseline_quant: dict[str, object],
    bytes_weight_per_kib: float,
    eval_ctx: dict[str, object] | None,
) -> dict[str, object]:
    band_angles = render_band_angles(
        base_band_angles,
        active_layers=active_layers,
        active_kv_heads=active_kv_heads,
        global_adjust=global_adjust,
        layer_adjust=layer_adjust,
        kv_adjust=kv_adjust,
        band_adjust=band_adjust,
        dense_adjust=dense_adjust,
    )
    key = band_angles.astype(np.float32, copy=False).tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = summarize_candidate(
        flat_state,
        band_angles=band_angles,
        transform=transform,
        layers=active_layers,
        scheme=scheme,
        output_proxy_ctx=output_proxy_ctx,
        turbo_embed_export=turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
        baseline_quant=baseline_quant,
        bytes_weight_per_kib=bytes_weight_per_kib,
        eval_ctx=eval_ctx,
    )
    result["band_angles"] = band_angles.tolist()
    result["selection_key"] = list(selection_key(result))
    cache[key] = result
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous exact-angle optimization for RoPE-compatible Q/K gauge transforms using true-token CE plus bytes."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--layers")
    parser.add_argument("--kv-heads")
    parser.add_argument("--init-result", type=Path, help="Optional prior discrete or continuous gauge-search JSON")
    parser.add_argument("--init-angle-scale", type=float, default=math.pi)
    parser.add_argument(
        "--init-parameterization",
        choices=("global_head_phase", "banded_phase", "full_pair_phase"),
        default="banded_phase",
    )
    parser.add_argument("--proxy-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-max-seqs", type=int, default=8)
    parser.add_argument("--proxy-batch-seqs", type=int, default=1)
    parser.add_argument("--bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--initial-step", type=float, default=0.25)
    parser.add_argument("--min-step", type=float, default=0.01)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--max-rounds-per-step", type=int, default=2)
    parser.add_argument("--dense-refine", action="store_true")
    parser.add_argument("--confirm-val-max-seqs", type=int, default=0)
    parser.add_argument("--confirm-val-batch-size", type=int, default=262144)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-stride", type=int, default=0)
    parser.add_argument("--confirm-eval-batch-seqs", type=int, default=0)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    config = aqe.infer_model_config(flat_state)
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1:
        raise SystemExit("Expected exactly one scheme")
    scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)

    base_band_angles, init_meta = parse_init_result(
        args.init_result,
        flat_state,
        angle_scale=args.init_angle_scale,
        parameterization=args.init_parameterization,
        num_bands=args.num_bands,
    )
    active_layers, active_kv_heads = build_active_sets(
        flat_state,
        parse_int_list(args.layers),
        parse_int_list(args.kv_heads),
        base_band_angles,
    )

    baseline_quant = aqe.analyze_scheme(
        flat_state,
        scheme,
        None,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    output_proxy_ctx = aop.build_proxy_context(
        flat_state,
        args.data_path,
        args.tokenizer_path,
        args.proxy_seq_len,
        args.proxy_max_seqs,
        args.proxy_batch_seqs,
    )
    baseline_compare_state, _baseline_quant_meta = aop.make_compare_state(
        flat_state,
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

    eval_ctx = None
    if args.confirm_val_max_seqs > 0:
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
            flat_state,
            scheme,
            eval_ctx,
            args.turbo_embed_export,
            turbo_mse_patterns,
            turbo_prod_patterns,
        )

    global_adjust = 0.0
    layer_adjust = np.zeros((len(active_layers),), dtype=np.float32)
    kv_adjust = np.zeros((len(active_kv_heads),), dtype=np.float32)
    band_adjust = np.zeros((args.num_bands,), dtype=np.float32)
    dense_adjust = (
        np.zeros((len(active_layers), len(active_kv_heads), args.num_bands), dtype=np.float32)
        if args.dense_refine
        else None
    )
    cache: dict[bytes, dict[str, object]] = {}
    history: list[dict[str, object]] = []

    incumbent = evaluate_params(
        cache=cache,
        flat_state=flat_state,
        base_band_angles=base_band_angles,
        transform=args.transform,
        active_layers=active_layers,
        active_kv_heads=active_kv_heads,
        global_adjust=global_adjust,
        layer_adjust=layer_adjust,
        kv_adjust=kv_adjust,
        band_adjust=band_adjust,
        dense_adjust=dense_adjust,
        scheme=scheme,
        output_proxy_ctx=output_proxy_ctx,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
        baseline_quant=baseline_quant,
        bytes_weight_per_kib=args.bytes_weight_per_kib,
        eval_ctx=eval_ctx,
    )

    params = param_records(active_layers, active_kv_heads, args.num_bands, args.dense_refine)
    step = float(args.initial_step)
    while step >= float(args.min_step):
        round_improved = False
        for round_idx in range(max(int(args.max_rounds_per_step), 1)):
            improved_this_round = False
            for record in params:
                current_value = get_param_value(
                    global_adjust,
                    layer_adjust,
                    kv_adjust,
                    band_adjust,
                    dense_adjust,
                    record,
                )
                best_local = incumbent
                best_value = current_value
                for delta in (step, -step):
                    candidate_value = current_value + delta
                    g2, l2, k2, b2, d2 = set_param_value(
                        global_adjust,
                        layer_adjust,
                        kv_adjust,
                        band_adjust,
                        dense_adjust,
                        record,
                        candidate_value,
                    )
                    candidate = evaluate_params(
                        cache=cache,
                        flat_state=flat_state,
                        base_band_angles=base_band_angles,
                        transform=args.transform,
                        active_layers=active_layers,
                        active_kv_heads=active_kv_heads,
                        global_adjust=g2,
                        layer_adjust=l2,
                        kv_adjust=k2,
                        band_adjust=b2,
                        dense_adjust=d2,
                        scheme=scheme,
                        output_proxy_ctx=output_proxy_ctx,
                        turbo_embed_export=args.turbo_embed_export,
                        turbo_mse_patterns=turbo_mse_patterns,
                        turbo_prod_patterns=turbo_prod_patterns,
                        baseline_quant=baseline_quant,
                        bytes_weight_per_kib=args.bytes_weight_per_kib,
                        eval_ctx=eval_ctx,
                    )
                    if selection_key(candidate) < selection_key(best_local):
                        best_local = candidate
                        best_value = candidate_value
                accepted = best_value != current_value
                if accepted:
                    global_adjust, layer_adjust, kv_adjust, band_adjust, dense_adjust = set_param_value(
                        global_adjust,
                        layer_adjust,
                        kv_adjust,
                        band_adjust,
                        dense_adjust,
                        record,
                        best_value,
                    )
                    incumbent = best_local
                    round_improved = True
                    improved_this_round = True
                history.append(
                    {
                        "step_size": step,
                        "round": round_idx,
                        "group": record["group"],
                        "index": list(record["index"]),
                        "accepted": accepted,
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
            step *= float(args.step_decay)
        else:
            print(
                f"[step={step:.6f}] score={incumbent['score']:.8f} ce={incumbent['output_proxy']['mean_ce_delta']:.8f} zlib={incumbent['bytes']['zlib_bytes']}",
                flush=True,
            )

    result = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "search": {
            "transform": args.transform,
            "num_bands": args.num_bands,
            "active_layers": list(active_layers),
            "active_kv_heads": list(active_kv_heads),
            "bytes_weight_per_kib": args.bytes_weight_per_kib,
            "initial_step": args.initial_step,
            "min_step": args.min_step,
            "step_decay": args.step_decay,
            "max_rounds_per_step": args.max_rounds_per_step,
            "dense_refine": args.dense_refine,
        },
        "init": init_meta,
        "baseline_quant": baseline_quant,
        "baseline_output_proxy": baseline_output_proxy,
        "final": {
            "global_adjust": float(global_adjust),
            "layer_adjust": layer_adjust.tolist(),
            "kv_adjust": kv_adjust.tolist(),
            "band_adjust": band_adjust.tolist(),
            "dense_adjust": None if dense_adjust is None else dense_adjust.tolist(),
            "band_angles": incumbent["band_angles"],
            "gauge": incumbent["gauge"],
            "bytes": incumbent["bytes"],
            "output_proxy": incumbent["output_proxy"],
            "score": float(incumbent["score"]),
            "selection_key": incumbent["selection_key"],
            "delta_ce_vs_baseline_quant": float(incumbent["delta_ce_vs_baseline_quant"]),
            "delta_margin_vs_baseline_quant": float(incumbent["delta_margin_vs_baseline_quant"]),
            "delta_zlib_bytes_vs_baseline_quant": int(incumbent["delta_zlib_bytes_vs_baseline_quant"]),
        },
        "history": history,
    }
    if "eval" in incumbent:
        result["final"]["eval"] = incumbent["eval"]
        result["final"]["delta_bpb_vs_baseline_quant"] = float(incumbent["delta_bpb_vs_baseline_quant"])

    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
