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
import turbo_quant_mlx as tq


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


def build_symmetric_codebook(pos_values: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_values, dtype=np.float32).copy()
    pos = np.maximum(pos, 1.0e-6)
    for i in range(1, pos.size):
        pos[i] = max(pos[i], pos[i - 1] + 1.0e-6)
    neg = -pos[::-1]
    return np.concatenate([neg, pos]).astype(np.float32, copy=False)


def override_key_string(override_key: tuple[str, int, int]) -> str:
    mode, mse_bits, block_size = override_key
    return f"{mode}:{mse_bits}:{block_size}"


def load_initial_codebooks(
    result_path: Path | None,
    *,
    scheme: dict[str, object],
) -> tuple[np.ndarray | None, dict[str, np.ndarray], dict[str, object] | None]:
    if result_path is None:
        return None, {}, None
    data = json.loads(result_path.read_text())
    final = data.get("final", {})
    global_codebook = final.get("codebook")
    tensor_codebooks_raw = final.get("codebooks_by_tensor")
    tensor_codebooks: dict[str, np.ndarray] = {}
    if isinstance(tensor_codebooks_raw, dict):
        tensor_codebooks = {
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in tensor_codebooks_raw.items()
        }
    key = override_key_string(("prod", int(scheme["prod_bits"]) - 1, int(scheme["block_size"])))
    kind = "none"
    if global_codebook is not None and tensor_codebooks:
        kind = "global_plus_tensor_codebook_override"
    elif global_codebook is not None:
        kind = "global_codebook_override"
    elif tensor_codebooks:
        kind = "per_tensor_codebook_override"
    return (
        None if global_codebook is None else np.asarray(global_codebook, dtype=np.float32),
        tensor_codebooks,
        {
            "source": str(result_path),
            "kind": kind,
            "override_keys": [] if kind == "none" else [key],
            "tensor_names": sorted(tensor_codebooks.keys()),
        },
    )


def candidate_quant_obj(
    base_quant_obj: dict[str, object],
    override_key: tuple[str, int, int],
    codebook: np.ndarray | None,
    *,
    codebooks_by_tensor: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    quant_obj = dict(base_quant_obj)
    key = override_key_string(override_key)
    if codebook is not None:
        overrides = dict(base_quant_obj.get("turbo_codebook_overrides", {}))
        overrides[key] = np.asarray(codebook, dtype=np.float32)
        quant_obj["turbo_codebook_overrides"] = overrides
    if codebooks_by_tensor:
        quant_obj["turbo_codebook_overrides_by_tensor"] = {
            name: {key: np.asarray(value, dtype=np.float32)}
            for name, value in codebooks_by_tensor.items()
        }
    return quant_obj


def summarize_candidate(
    *,
    base_quant_obj: dict[str, object],
    base_stats: dict[str, object],
    override_key: tuple[str, int, int],
    codebook: np.ndarray | None,
    codebooks_by_tensor: dict[str, np.ndarray] | None,
    proxy_ctx: dict[str, object],
    baseline_bytes: dict[str, object],
    bytes_weight_per_kib: float,
) -> dict[str, object]:
    overrides = None if codebook is None else {override_key: np.asarray(codebook, dtype=np.float32)}
    tensor_overrides = None
    if codebooks_by_tensor:
        tensor_overrides = {
            name: {override_key: np.asarray(value, dtype=np.float32)}
            for name, value in codebooks_by_tensor.items()
        }
    deq_state = aqe.dequantize_quant_obj(base_quant_obj, overrides, tensor_overrides)
    quant_obj = candidate_quant_obj(
        base_quant_obj,
        override_key,
        codebook,
        codebooks_by_tensor=codebooks_by_tensor,
    )
    bytes_info = aqe.summarize_quant_bytes(quant_obj, base_stats)
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
        "codebook": None if codebook is None else np.asarray(codebook, dtype=np.float32).tolist(),
        "codebooks_by_tensor": None
        if not codebooks_by_tensor
        else {
            name: np.asarray(value, dtype=np.float32).tolist()
            for name, value in sorted(codebooks_by_tensor.items())
        },
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
    parser = argparse.ArgumentParser(description="Task-aware Turbo codebook refinement using true-token CE.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-result", type=Path)
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--codebook-result", type=Path)
    parser.add_argument("--refine-mode", choices=("prod",), default="prod")
    parser.add_argument("--per-tensor", action="store_true")
    parser.add_argument("--target-tensors")
    parser.add_argument("--bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--initial-step", type=float, default=0.01)
    parser.add_argument("--min-step", type=float, default=0.001)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--max-rounds-per-step", type=int, default=2)
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
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    base_quant_obj, base_stats, base_deq_state = aqe.realize_scheme(
        flat_state,
        scheme,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
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
    baseline_quant_proxy = aop.summarize_search_proxy(
        aop.analyze_compare_state_proxies(
            proxy_ctx["ref_model"],
            base_deq_state,
            proxy_ctx["val_tokens"],
            int(proxy_ctx["seq_len"]),
            int(proxy_ctx["max_seqs"]),
            int(proxy_ctx["batch_seqs"]),
        )
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

    if args.refine_mode != "prod":
        raise SystemExit("Only prod refinement is supported")
    mse_bits = int(scheme["prod_bits"]) - 1
    block_size = int(scheme["block_size"])
    override_key = ("prod", mse_bits, block_size)
    baseline_codebook = np.asarray(tq._codebook_np(mse_bits, block_size), dtype=np.float32)
    global_codebook, tensor_codebooks, init_codebook_meta = load_initial_codebooks(
        args.codebook_result,
        scheme=scheme,
    )
    if global_codebook is None:
        global_codebook = np.array(baseline_codebook, copy=True)
    pos_values = np.asarray(global_codebook[len(global_codebook) // 2 :], dtype=np.float32).copy()
    prod_tensor_names = sorted(
        name
        for name, meta in base_quant_obj.get("turbo", {}).items()
        if str(meta.get("mode", "")) == "prod"
    )
    if args.target_tensors:
        target_tensors = tuple(part.strip() for part in args.target_tensors.split(",") if part.strip())
    else:
        target_tensors = tuple(prod_tensor_names)
    if args.per_tensor:
        unknown = [name for name in target_tensors if name not in prod_tensor_names]
        if unknown:
            raise SystemExit(f"Unknown prod tensors in --target-tensors: {unknown}")

    baseline = summarize_candidate(
        base_quant_obj=base_quant_obj,
        base_stats=base_stats,
        override_key=override_key,
        codebook=global_codebook,
        codebooks_by_tensor=tensor_codebooks,
        proxy_ctx=proxy_ctx,
        baseline_bytes=baseline_bytes,
        bytes_weight_per_kib=args.bytes_weight_per_kib,
    )
    incumbent = baseline
    history: list[dict[str, object]] = []
    step = float(args.initial_step)
    while step >= float(args.min_step):
        round_improved = False
        for round_idx in range(max(int(args.max_rounds_per_step), 1)):
            improved_this_round = False
            if not args.per_tensor:
                for idx in range(pos_values.size):
                    current_value = float(pos_values[idx])
                    best_local = incumbent
                    best_value = current_value
                    for delta in (step, -step):
                        trial = np.array(pos_values, copy=True)
                        trial[idx] = current_value + delta
                        codebook = build_symmetric_codebook(trial)
                        candidate = summarize_candidate(
                            base_quant_obj=base_quant_obj,
                            base_stats=base_stats,
                            override_key=override_key,
                            codebook=codebook,
                            codebooks_by_tensor=tensor_codebooks,
                            proxy_ctx=proxy_ctx,
                            baseline_bytes=baseline_bytes,
                            bytes_weight_per_kib=args.bytes_weight_per_kib,
                        )
                        if selection_key(candidate) < selection_key(best_local):
                            best_local = candidate
                            best_value = float(build_symmetric_codebook(trial)[len(codebook) // 2 + idx])
                    accepted = not math.isclose(best_value, current_value, rel_tol=0.0, abs_tol=1.0e-12)
                    if accepted:
                        pos_values[idx] = best_value
                        pos_values = build_symmetric_codebook(pos_values)[len(global_codebook) // 2 :]
                        global_codebook = build_symmetric_codebook(pos_values)
                        incumbent = best_local
                        improved_this_round = True
                        round_improved = True
                    history.append(
                        {
                            "step_size": float(step),
                            "round": int(round_idx),
                            "index": int(idx),
                            "accepted": bool(accepted),
                            "new_value": float(best_value),
                            "scope": "global",
                            "score": float(incumbent["score"]),
                            "mean_ce_delta": float(incumbent["output_proxy"]["mean_ce_delta"]),
                            "mean_margin_delta": float(incumbent["output_proxy"]["mean_margin_delta"]),
                            "zlib_bytes": int(incumbent["bytes"]["zlib_bytes"]),
                        }
                    )
            else:
                for tensor_name in target_tensors:
                    current_codebook = np.array(tensor_codebooks.get(tensor_name, global_codebook), copy=True)
                    current_pos = np.asarray(current_codebook[len(current_codebook) // 2 :], dtype=np.float32).copy()
                    for idx in range(current_pos.size):
                        current_value = float(current_pos[idx])
                        best_local = incumbent
                        best_value = current_value
                        best_codebooks = dict(tensor_codebooks)
                        for delta in (step, -step):
                            trial_pos = np.array(current_pos, copy=True)
                            trial_pos[idx] = current_value + delta
                            trial_codebook = build_symmetric_codebook(trial_pos)
                            trial_codebooks = dict(tensor_codebooks)
                            if np.allclose(trial_codebook, global_codebook, atol=1.0e-12, rtol=0.0):
                                trial_codebooks.pop(tensor_name, None)
                            else:
                                trial_codebooks[tensor_name] = trial_codebook
                            candidate = summarize_candidate(
                                base_quant_obj=base_quant_obj,
                                base_stats=base_stats,
                                override_key=override_key,
                                codebook=global_codebook,
                                codebooks_by_tensor=trial_codebooks,
                                proxy_ctx=proxy_ctx,
                                baseline_bytes=baseline_bytes,
                                bytes_weight_per_kib=args.bytes_weight_per_kib,
                            )
                            if selection_key(candidate) < selection_key(best_local):
                                best_local = candidate
                                best_value = float(trial_codebook[len(trial_codebook) // 2 + idx])
                                best_codebooks = trial_codebooks
                        accepted = not math.isclose(best_value, current_value, rel_tol=0.0, abs_tol=1.0e-12)
                        if accepted:
                            tensor_codebooks = best_codebooks
                            incumbent = best_local
                            improved_this_round = True
                            round_improved = True
                            current_codebook = np.array(tensor_codebooks.get(tensor_name, global_codebook), copy=True)
                            current_pos = np.asarray(current_codebook[len(current_codebook) // 2 :], dtype=np.float32).copy()
                        history.append(
                            {
                                "step_size": float(step),
                                "round": int(round_idx),
                                "tensor": str(tensor_name),
                                "index": int(idx),
                                "accepted": bool(accepted),
                                "new_value": float(best_value),
                                "scope": "per_tensor",
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
        "gauge_result": None if args.gauge_result is None else str(args.gauge_result),
        "codebook_result": None if args.codebook_result is None else str(args.codebook_result),
        "gauge_meta": gauge_meta,
        "init_codebook_meta": init_codebook_meta,
        "search": {
            "refine_mode": args.refine_mode,
            "per_tensor": bool(args.per_tensor),
            "target_tensors": list(target_tensors) if args.per_tensor else [],
            "bytes_weight_per_kib": float(args.bytes_weight_per_kib),
            "initial_step": float(args.initial_step),
            "min_step": float(args.min_step),
            "step_decay": float(args.step_decay),
            "max_rounds_per_step": int(args.max_rounds_per_step),
        },
        "baseline_quant": {
            "bytes": baseline_bytes,
            "output_proxy": baseline_quant_proxy,
        },
        "baseline": baseline,
        "final": incumbent,
        "history": history,
    }
    if eval_ctx is not None:
        baseline_eval_loss, baseline_eval_bpb = aqe.eval_state(eval_ctx, base_deq_state)
        final_deq_state = aqe.dequantize_quant_obj(
            base_quant_obj,
            None if incumbent["codebook"] is None else {override_key: np.asarray(incumbent["codebook"], dtype=np.float32)},
            None
            if not incumbent.get("codebooks_by_tensor")
            else {
                name: {override_key: np.asarray(value, dtype=np.float32)}
                for name, value in incumbent["codebooks_by_tensor"].items()
            },
        )
        final_eval_loss, final_eval_bpb = aqe.eval_state(eval_ctx, final_deq_state)
        result["baseline_quant"]["eval"] = {"val_loss": float(baseline_eval_loss), "val_bpb": float(baseline_eval_bpb)}
        result["baseline"]["eval"] = {"val_loss": float(baseline_eval_loss), "val_bpb": float(baseline_eval_bpb)}
        result["final"]["eval"] = {"val_loss": float(final_eval_loss), "val_bpb": float(final_eval_bpb)}
        result["final"]["delta_bpb_vs_baseline_quant"] = float(final_eval_bpb - baseline_eval_bpb)

    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
