#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

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


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds parsed")
    return seeds


def parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    layers: list[int] = []
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        layers.append(int(part))
    return tuple(layers) if layers else None


def assign_band_indices(pair_count: int, num_bands: int) -> np.ndarray:
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}")
    edges = np.linspace(0, pair_count, num_bands + 1, dtype=np.int64)
    band_idx = np.empty((pair_count,), dtype=np.int64)
    for band in range(num_bands):
        band_idx[edges[band] : edges[band + 1]] = band
    return band_idx


def expand_band_angles(head_dim: int, band_angles: np.ndarray) -> np.ndarray:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE-compatible gauge, got {head_dim}")
    band_angles = np.asarray(band_angles, dtype=np.float32)
    pair_count = head_dim // 2
    band_idx = assign_band_indices(pair_count, int(band_angles.shape[0]))
    return band_angles[band_idx]


def compress_pair_angles_to_bands(pair_angles: np.ndarray, num_bands: int) -> np.ndarray:
    pair_angles = np.asarray(pair_angles, dtype=np.float32)
    pair_count = int(pair_angles.shape[0])
    band_idx = assign_band_indices(pair_count, num_bands)
    out = np.zeros((num_bands,), dtype=np.float32)
    for band in range(num_bands):
        mask = band_idx == band
        out[band] = float(np.mean(pair_angles[mask], dtype=np.float64)) if np.any(mask) else 0.0
    return out


def build_pair_rotation_matrix(head_dim: int, angles: np.ndarray) -> np.ndarray:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE-compatible gauge, got {head_dim}")
    pair_count = head_dim // 2
    if angles.shape != (pair_count,):
        raise ValueError(f"Expected {pair_count} angles, got {angles.shape}")
    out = np.zeros((head_dim, head_dim), dtype=np.float32)
    half = head_dim // 2
    for pair_idx, theta in enumerate(angles):
        # MLX RoPE with traditional=False uses split-half pairings.
        i = pair_idx
        j = pair_idx + half
        c = float(math.cos(float(theta)))
        s = float(math.sin(float(theta)))
        out[i, i] = c
        out[i, j] = -s
        out[j, i] = s
        out[j, j] = c
    return out


def sample_angles(
    block_idx: int,
    kv_head_idx: int,
    head_dim: int,
    seed: int,
    angle_scale: float,
    parameterization: str,
    num_bands: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 104729 * (block_idx + 1) + 130363 * (kv_head_idx + 1))
    pair_count = head_dim // 2
    if parameterization == "full_pair_phase":
        return rng.uniform(-angle_scale, angle_scale, size=(pair_count,)).astype(np.float32)
    if parameterization == "global_head_phase":
        theta = float(rng.uniform(-angle_scale, angle_scale))
        return np.full((pair_count,), theta, dtype=np.float32)
    if parameterization == "banded_phase":
        bands = rng.uniform(-angle_scale, angle_scale, size=(num_bands,)).astype(np.float32)
        band_idx = assign_band_indices(pair_count, num_bands)
        return bands[band_idx]
    raise ValueError(f"Unsupported parameterization: {parameterization}")


def candidate_rotation(
    block_idx: int,
    kv_head_idx: int,
    head_dim: int,
    seed: int,
    angle_scale: float,
    parameterization: str,
    num_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    angles = sample_angles(block_idx, kv_head_idx, head_dim, seed, angle_scale, parameterization, num_bands)
    return build_pair_rotation_matrix(head_dim, angles), angles


def gauge_config(flat_state: dict[str, mx.array]) -> dict[str, int]:
    config = aqe.infer_model_config(flat_state)
    model_dim = int(config["model_dim"])
    num_heads = int(config["num_heads"])
    num_kv_heads = int(config["num_kv_heads"])
    num_blocks = int(config["num_layer_templates"])
    attention_blocks = config["attention_blocks"]
    head_dim = model_dim // num_heads
    q_per_kv = num_heads // num_kv_heads
    return {
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_blocks": num_blocks,
        "attention_blocks": attention_blocks,
        "head_dim": head_dim,
        "q_per_kv": q_per_kv,
    }


def zero_band_angle_tensor(flat_state: dict[str, mx.array], num_bands: int) -> np.ndarray:
    cfg = gauge_config(flat_state)
    return np.zeros((cfg["num_blocks"], cfg["num_kv_heads"], num_bands), dtype=np.float32)


def band_angle_tensor_from_seed_assignments(
    flat_state: dict[str, mx.array],
    *,
    base_seed: int,
    angle_scale: float,
    parameterization: str,
    num_bands: int,
    layers: tuple[int, ...] | None = None,
    seed_by_layer: dict[int, int] | None = None,
    seed_by_block_kv: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    cfg = gauge_config(flat_state)
    band_tensor = np.zeros((cfg["num_blocks"], cfg["num_kv_heads"], num_bands), dtype=np.float32)
    layer_set = set(layers) if layers is not None else None
    for block_idx in range(cfg["num_blocks"]):
        if layer_set is not None and block_idx not in layer_set:
            continue
        for kv_head_idx in range(cfg["num_kv_heads"]):
            chosen_seed: int | None
            if seed_by_block_kv is not None and (block_idx, kv_head_idx) in seed_by_block_kv:
                chosen_seed = int(seed_by_block_kv[(block_idx, kv_head_idx)])
            elif seed_by_layer is not None and block_idx in seed_by_layer:
                chosen_seed = int(seed_by_layer[block_idx])
            elif seed_by_block_kv is not None:
                chosen_seed = None
            else:
                chosen_seed = int(base_seed)
            if chosen_seed is None:
                continue
            _rotation, pair_angles = candidate_rotation(
                block_idx,
                kv_head_idx,
                cfg["head_dim"],
                chosen_seed,
                angle_scale,
                parameterization,
                num_bands,
            )
            band_tensor[block_idx, kv_head_idx] = compress_pair_angles_to_bands(pair_angles, num_bands)
    return band_tensor


def apply_rope_gauge_band_angles(
    flat_state: dict[str, mx.array],
    *,
    band_angles: np.ndarray,
    transform: str,
    layers: tuple[int, ...] | None = None,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    cfg = gauge_config(flat_state)
    band_angles = np.asarray(band_angles, dtype=np.float32)
    expected_shape = (cfg["num_blocks"], cfg["num_kv_heads"], band_angles.shape[-1])
    if band_angles.ndim != 3 or band_angles.shape[0] != cfg["num_blocks"] or band_angles.shape[1] != cfg["num_kv_heads"]:
        raise ValueError(
            f"Expected band_angles shape (num_blocks, num_kv_heads, num_bands), got {band_angles.shape}"
        )
    transformed = dict(flat_state)
    layer_set = set(layers) if layers is not None else None
    angle_abs: list[float] = []
    per_block: dict[str, object] = {}

    for block_idx in range(cfg["num_blocks"]):
        if layer_set is not None and block_idx not in layer_set:
            continue
        block_meta = cfg["attention_blocks"][block_idx]
        rotations: list[np.ndarray] = []
        pair_angle_lists: list[np.ndarray] = []
        for kv_head_idx in range(cfg["num_kv_heads"]):
            pair_angles = expand_band_angles(cfg["head_dim"], band_angles[block_idx, kv_head_idx])
            rotations.append(build_pair_rotation_matrix(cfg["head_dim"], pair_angles))
            pair_angle_lists.append(pair_angles)
            angle_abs.append(float(np.mean(np.abs(pair_angles), dtype=np.float64)))

        q_name = str(block_meta["q_name"])
        if q_name in transformed:
            q_np = np.asarray(transformed[q_name].astype(mx.float32), dtype=np.float32).reshape(
                cfg["num_heads"], cfg["head_dim"], -1
            )
            q_out = np.empty_like(q_np)
            for q_head_idx in range(cfg["num_heads"]):
                kv_head_idx = q_head_idx // cfg["q_per_kv"]
                q_out[q_head_idx] = rotations[kv_head_idx] @ q_np[q_head_idx]
            transformed[q_name] = mx.array(q_out.reshape(cfg["model_dim"], -1), dtype=flat_state[q_name].dtype)

        target_names = [str(block_meta["k_name"])] if transform == "qk_only" else [
            str(block_meta["k_name"]),
            str(block_meta["v_name"]),
        ]
        for name in target_names:
            if name not in transformed:
                continue
            arr_np = np.asarray(transformed[name].astype(mx.float32), dtype=np.float32).reshape(
                cfg["num_kv_heads"], cfg["head_dim"], -1
            )
            arr_out = np.empty_like(arr_np)
            for kv_head_idx in range(cfg["num_kv_heads"]):
                arr_out[kv_head_idx] = rotations[kv_head_idx] @ arr_np[kv_head_idx]
            transformed[name] = mx.array(
                arr_out.reshape(cfg["num_kv_heads"] * cfg["head_dim"], -1),
                dtype=flat_state[name].dtype,
            )

        proj_name = str(block_meta["proj_name"])
        if transform == "qkvo_full" and proj_name in transformed:
            proj_np = np.asarray(transformed[proj_name].astype(mx.float32), dtype=np.float32).reshape(
                -1, cfg["num_heads"], cfg["head_dim"]
            )
            proj_out = np.empty_like(proj_np)
            for q_head_idx in range(cfg["num_heads"]):
                kv_head_idx = q_head_idx // cfg["q_per_kv"]
                proj_out[:, q_head_idx, :] = proj_np[:, q_head_idx, :] @ rotations[kv_head_idx].T
            transformed[proj_name] = mx.array(
                proj_out.reshape(-1, cfg["model_dim"]),
                dtype=flat_state[proj_name].dtype,
            )

        per_block[str(block_idx)] = {
            "kv_head_trace_mean": float(np.mean([np.trace(rot, dtype=np.float64) / cfg["head_dim"] for rot in rotations])),
            "kv_head_mean_abs_angle": float(np.mean([np.mean(np.abs(angles), dtype=np.float64) for angles in pair_angle_lists])),
        }

    metadata = {
        "transform": transform,
        "layers": list(layers) if layers is not None else None,
        "num_bands": int(band_angles.shape[-1]),
        "mean_abs_angle": float(np.mean(angle_abs)) if angle_abs else 0.0,
        "num_blocks": cfg["num_blocks"],
        "num_heads": cfg["num_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
        "head_dim": cfg["head_dim"],
        "per_block": per_block,
        "band_angles": band_angles.tolist(),
    }
    return transformed, metadata


def summarize_proxy_metrics(metrics: dict[str, object]) -> dict[str, float]:
    qk_gram = metrics.get("qk_gram", {})
    k_rmse: list[float] = []
    q_rmse: list[float] = []
    k_abs_bias: list[float] = []
    q_abs_bias: list[float] = []
    for name, entry in qk_gram.items():
        rmse = float(entry["rmse"])
        abs_bias = abs(float(entry["bias"]))
        if ".attn.c_k.weight" in name:
            k_rmse.append(rmse)
            k_abs_bias.append(abs_bias)
        elif ".attn.c_q.weight" in name:
            q_rmse.append(rmse)
            q_abs_bias.append(abs_bias)
    out = {
        "k_rmse_mean": float(np.mean(k_rmse)) if k_rmse else 0.0,
        "q_rmse_mean": float(np.mean(q_rmse)) if q_rmse else 0.0,
        "k_abs_bias_mean": float(np.mean(k_abs_bias)) if k_abs_bias else 0.0,
        "q_abs_bias_mean": float(np.mean(q_abs_bias)) if q_abs_bias else 0.0,
    }
    out["score"] = out["k_rmse_mean"] + 0.5 * out["k_abs_bias_mean"] + 0.25 * out["q_rmse_mean"] + 0.1 * out["q_abs_bias_mean"]
    return out


def apply_rope_gauge_transform(
    flat_state: dict[str, mx.array],
    *,
    seed: int,
    angle_scale: float,
    transform: str,
    parameterization: str,
    num_bands: int,
    layers: tuple[int, ...] | None = None,
    seed_by_layer: dict[int, int] | None = None,
    seed_by_block_kv: dict[tuple[int, int], int] | None = None,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    config = aqe.infer_model_config(flat_state)
    model_dim = int(config["model_dim"])
    num_heads = int(config["num_heads"])
    num_kv_heads = int(config["num_kv_heads"])
    num_blocks = int(config["num_layer_templates"])
    attention_blocks = config["attention_blocks"]
    head_dim = model_dim // num_heads
    q_per_kv = num_heads // num_kv_heads
    transformed = dict(flat_state)
    angle_abs: list[float] = []
    per_block: dict[str, object] = {}

    layer_set = set(layers) if layers is not None else None
    for block_idx in range(num_blocks):
        if layer_set is not None and block_idx not in layer_set:
            continue
        rotation_pack = []
        for kv_head_idx in range(num_kv_heads):
            if seed_by_block_kv is not None and (block_idx, kv_head_idx) in seed_by_block_kv:
                rotation_pack.append(
                    candidate_rotation(
                        block_idx,
                        kv_head_idx,
                        head_dim,
                        int(seed_by_block_kv[(block_idx, kv_head_idx)]),
                        angle_scale,
                        parameterization,
                        num_bands,
                    )
                )
            elif seed_by_layer is not None and block_idx in seed_by_layer:
                rotation_pack.append(
                    candidate_rotation(
                        block_idx,
                        kv_head_idx,
                        head_dim,
                        int(seed_by_layer[block_idx]),
                        angle_scale,
                        parameterization,
                        num_bands,
                    )
                )
            elif seed_by_block_kv is not None:
                identity = np.eye(head_dim, dtype=np.float32)
                rotation_pack.append((identity, np.zeros((head_dim // 2,), dtype=np.float32)))
            else:
                rotation_pack.append(
                    candidate_rotation(
                        block_idx,
                        kv_head_idx,
                        head_dim,
                        seed,
                        angle_scale,
                        parameterization,
                        num_bands,
                    )
                )
        rotations = [rot for rot, _angles in rotation_pack]
        angle_lists = [angles for _rot, angles in rotation_pack]
        angle_abs.extend(float(np.mean(np.abs(angles), dtype=np.float64)) for angles in angle_lists)
        block_meta = attention_blocks[block_idx]

        q_name = str(block_meta["q_name"])
        if q_name in transformed:
            q_np = np.asarray(transformed[q_name].astype(mx.float32), dtype=np.float32).reshape(num_heads, head_dim, -1)
            q_out = np.empty_like(q_np)
            for q_head_idx in range(num_heads):
                kv_head_idx = q_head_idx // q_per_kv
                q_out[q_head_idx] = rotations[kv_head_idx] @ q_np[q_head_idx]
            transformed[q_name] = mx.array(q_out.reshape(model_dim, -1), dtype=flat_state[q_name].dtype)

        target_names = [str(block_meta["k_name"])] if transform == "qk_only" else [
            str(block_meta["k_name"]),
            str(block_meta["v_name"]),
        ]
        for name in target_names:
            if name not in transformed:
                continue
            arr_np = np.asarray(transformed[name].astype(mx.float32), dtype=np.float32).reshape(num_kv_heads, head_dim, -1)
            arr_out = np.empty_like(arr_np)
            for kv_head_idx in range(num_kv_heads):
                arr_out[kv_head_idx] = rotations[kv_head_idx] @ arr_np[kv_head_idx]
            transformed[name] = mx.array(arr_out.reshape(num_kv_heads * head_dim, -1), dtype=flat_state[name].dtype)

        proj_name = str(block_meta["proj_name"])
        if transform == "qkvo_full" and proj_name in transformed:
            proj_np = np.asarray(transformed[proj_name].astype(mx.float32), dtype=np.float32).reshape(-1, num_heads, head_dim)
            proj_out = np.empty_like(proj_np)
            for q_head_idx in range(num_heads):
                kv_head_idx = q_head_idx // q_per_kv
                proj_out[:, q_head_idx, :] = proj_np[:, q_head_idx, :] @ rotations[kv_head_idx].T
            transformed[proj_name] = mx.array(proj_out.reshape(-1, model_dim), dtype=flat_state[proj_name].dtype)

        per_block[str(block_idx)] = {
            "kv_head_trace_mean": float(np.mean([np.trace(rot, dtype=np.float64) / head_dim for rot in rotations])),
            "kv_head_mean_abs_angle": float(np.mean([np.mean(np.abs(angles), dtype=np.float64) for angles in angle_lists])),
        }

    metadata = {
        "seed": seed,
        "angle_scale": angle_scale,
        "transform": transform,
        "parameterization": parameterization,
        "num_bands": num_bands,
        "layers": list(layers) if layers is not None else None,
        "seed_by_layer": {str(k): int(v) for k, v in sorted((seed_by_layer or {}).items())} if seed_by_layer else None,
        "seed_by_block_kv": (
            {f"{block_idx}:{kv_head_idx}": int(v) for (block_idx, kv_head_idx), v in sorted(seed_by_block_kv.items())}
            if seed_by_block_kv
            else None
        ),
        "mean_abs_angle": float(np.mean(angle_abs)) if angle_abs else 0.0,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "per_block": per_block,
    }
    return transformed, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search RoPE-compatible attention gauge rotations that preserve the float model and may improve quantized export.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--angle-scale", type=float, default=math.pi, help="Uniform angle range is [-scale, scale] per RoPE pair")
    parser.add_argument("--transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--parameterization", choices=("global_head_phase", "banded_phase", "full_pair_phase"), default="global_head_phase")
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--layers", help="Optional comma-separated block indices to transform")
    parser.add_argument("--topk-eval", type=int, default=0, help="If >0, evaluate only the best proxy-ranked candidates")
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--val-max-seqs", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=262144)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--raw-sanity-seed", type=int, default=-1, help="Optional seed to sanity-check raw invariance with a float eval")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    config = aqe.infer_model_config(flat_state)
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1:
        raise SystemExit("Expected exactly one scheme for gauge search")
    scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    layer_subset = parse_layers(args.layers)
    eval_ctx = None
    if args.data_path and args.tokenizer_path:
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

    baseline_raw = None
    baseline_quant = None
    if eval_ctx is not None:
        base_loss, base_bpb = aqe.eval_state(eval_ctx, flat_state)
        baseline_raw = {"val_loss": float(base_loss), "val_bpb": float(base_bpb)}
    baseline_quant = aqe.analyze_scheme(
        flat_state,
        scheme,
        eval_ctx,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )

    raw_sanity = None
    if eval_ctx is not None and args.raw_sanity_seed >= 0:
        sanity_state, sanity_meta = apply_rope_gauge_transform(
            flat_state,
            seed=args.raw_sanity_seed,
            angle_scale=args.angle_scale,
            transform=args.transform,
            parameterization=args.parameterization,
            num_bands=args.num_bands,
            layers=layer_subset,
        )
        sanity_loss, sanity_bpb = aqe.eval_state(eval_ctx, sanity_state)
        raw_sanity = {
            "seed": args.raw_sanity_seed,
            "meta": sanity_meta,
            "eval": {"val_loss": float(sanity_loss), "val_bpb": float(sanity_bpb)},
        }
        if baseline_raw is not None:
            raw_sanity["delta_bpb"] = float(sanity_bpb - baseline_raw["val_bpb"])
            raw_sanity["delta_loss"] = float(sanity_loss - baseline_raw["val_loss"])

    candidates: list[dict[str, object]] = []
    for seed in parse_seeds(args.seeds):
        candidate_state, meta = apply_rope_gauge_transform(
            flat_state,
            seed=seed,
            angle_scale=args.angle_scale,
            transform=args.transform,
            parameterization=args.parameterization,
            num_bands=args.num_bands,
            layers=layer_subset,
        )
        analysis = aqe.analyze_scheme(
            candidate_state,
            scheme,
            None if args.topk_eval > 0 else eval_ctx,
            args.turbo_embed_export,
            turbo_mse_patterns,
            turbo_prod_patterns,
        )
        proxy = summarize_proxy_metrics(analysis["metrics"])
        candidate: dict[str, object] = {
            "seed": seed,
            "meta": meta,
            "proxy": proxy,
            "analysis": analysis,
        }
        if "eval" in analysis and "eval" in baseline_quant:
            candidate["delta_bpb_vs_baseline_quant"] = float(analysis["eval"]["val_bpb"] - baseline_quant["eval"]["val_bpb"])
            candidate["delta_loss_vs_baseline_quant"] = float(analysis["eval"]["val_loss"] - baseline_quant["eval"]["val_loss"])
        candidate["delta_zlib_bytes_vs_baseline_quant"] = int(analysis["bytes"]["zlib_bytes"]) - int(baseline_quant["bytes"]["zlib_bytes"])
        candidates.append(candidate)

    candidates.sort(key=lambda item: (float(item["proxy"]["score"]), int(item["analysis"]["bytes"]["zlib_bytes"])))
    if eval_ctx is not None and args.topk_eval > 0:
        for candidate in candidates[: min(args.topk_eval, len(candidates))]:
            candidate_state, _meta = apply_rope_gauge_transform(
                flat_state,
                seed=int(candidate["seed"]),
                angle_scale=args.angle_scale,
                transform=args.transform,
                parameterization=args.parameterization,
                num_bands=args.num_bands,
                layers=layer_subset,
            )
            eval_analysis = aqe.analyze_scheme(
                candidate_state,
                scheme,
                eval_ctx,
                args.turbo_embed_export,
                turbo_mse_patterns,
                turbo_prod_patterns,
            )
            candidate["analysis"]["eval"] = eval_analysis["eval"]
            candidate["delta_bpb_vs_baseline_quant"] = float(eval_analysis["eval"]["val_bpb"] - baseline_quant["eval"]["val_bpb"])
            candidate["delta_loss_vs_baseline_quant"] = float(eval_analysis["eval"]["val_loss"] - baseline_quant["eval"]["val_loss"])
        candidates.sort(
            key=lambda item: (
                0 if "eval" in item["analysis"] else 1,
                float(item["analysis"]["eval"]["val_bpb"]) if "eval" in item["analysis"] else float(item["proxy"]["score"]),
                int(item["analysis"]["bytes"]["zlib_bytes"]),
            )
        )
    elif eval_ctx is not None:
        candidates.sort(key=lambda item: (float(item["analysis"]["eval"]["val_bpb"]), int(item["analysis"]["bytes"]["zlib_bytes"])))
    else:
        candidates.sort(key=lambda item: int(item["analysis"]["bytes"]["zlib_bytes"]))

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "search": {
            "transform": args.transform,
            "parameterization": args.parameterization,
            "num_bands": args.num_bands,
            "layers": list(layer_subset) if layer_subset is not None else None,
            "angle_scale": args.angle_scale,
            "topk_eval": args.topk_eval,
        },
        "baseline_raw": baseline_raw,
        "baseline_quant": baseline_quant,
        "raw_sanity": raw_sanity,
        "candidates": candidates,
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
