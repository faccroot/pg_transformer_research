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
import pandas as pd
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import analyze_logic_sidecar as als
import analyze_mlx_quant_export as aq
import eval_saved_sidecar as ess
import train_gpt_mlx as tg
import train_gpt_mlx_jepa_sidecar as tgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a sampled curriculum corpus with baseline and sidecar MLX checkpoints.")
    p.add_argument("--sample-csv", required=True, help="Path to curriculum_data_map_sample.csv.gz")
    p.add_argument("--data-path", required=True, help="Token shard directory")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--baseline-artifact", required=True, help="Plain model .npz or .ptz")
    p.add_argument("--sidecar-artifact", required=True, help="Sidecar .pklz or .npz")
    p.add_argument("--operator-token-json", default="", help="Optional operator token JSON for focused loss around NOT")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--operator-window", type=int, default=5)
    p.add_argument("--sidecar-polarity-write", type=int, default=1)
    p.add_argument("--sidecar-polarity-pool", default="max")
    p.add_argument("--logic-dim", type=int, default=8)
    p.add_argument("--logic-layer-index", type=int, default=4)
    p.add_argument("--logic-route-to-next-token", type=int, default=1)
    p.add_argument("--logic-operator-mode", default="not_only")
    p.add_argument("--polarity-detector-enabled", type=int, default=1)
    p.add_argument("--polarity-detector-layer-index", type=int, default=3)
    p.add_argument("--polarity-detector-hidden-dim", type=int, default=64)
    p.add_argument("--polarity-seed-blend", type=float, default=0.5)
    p.add_argument("--polarity-seed-weight", type=float, default=0.05)
    p.add_argument("--polarity-sparse-weight", type=float, default=0.001)
    p.add_argument("--polarity-smooth-weight", type=float, default=0.001)
    return p.parse_args()


def load_sample_frame(path: Path, max_samples: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if max_samples > 0 and len(frame) > max_samples:
        frame = frame.iloc[:max_samples].copy()
    return frame.reset_index(drop=True)


def build_baseline_model(tokenizer_path: Path, artifact_path: Path):
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if artifact_path.suffix == ".npz":
        flat_state = aq.load_flat_state(artifact_path)
    elif artifact_path.suffix == ".ptz":
        flat_state = tg.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    else:
        raise ValueError(f"Unsupported baseline artifact type: {artifact_path}")
    config = als.infer_logic_config(flat_state, logic_layer_index=None, logic_route_to_next_token=True)
    model = als.build_model(config, sp)
    model.update(tree_unflatten(list(flat_state.items())))
    if hasattr(model, "set_turbo_qat"):
        model.set_turbo_qat(False, 0.0)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()
    return sp, model


def build_sidecar_model(args: argparse.Namespace, tokenizer_path: Path, artifact_path: Path):
    ess.set_env(
        argparse.Namespace(
            tokenizer_path=str(tokenizer_path),
            data_path=args.data_path,
            val_max_seqs=0,
            train_seq_len=args.chunk_size,
            eval_seq_len=0,
            persistent=0,
            mode="reset",
            persist_group_seqs=1,
            cache_variant="sp1024",
            train_shards=1,
            tie_embeddings=0,
            logic_dim=args.logic_dim,
            logic_layer_index=args.logic_layer_index,
            logic_route_to_next_token=args.logic_route_to_next_token,
            logic_operator_mode=args.logic_operator_mode,
            polarity_detector_enabled=args.polarity_detector_enabled,
            polarity_detector_layer_index=args.polarity_detector_layer_index,
            polarity_detector_hidden_dim=args.polarity_detector_hidden_dim,
            polarity_seed_blend=args.polarity_seed_blend,
            polarity_seed_weight=args.polarity_seed_weight,
            polarity_sparse_weight=args.polarity_sparse_weight,
            polarity_smooth_weight=args.polarity_smooth_weight,
            sidecar_polarity_write=args.sidecar_polarity_write,
            sidecar_polarity_pool=args.sidecar_polarity_pool,
            sidecar_tap_layer=3,
            sidecar_pred_weight=0.05,
            sidecar_pred_offset=1,
            sidecar_sigreg_weight=0.01,
            sidecar_spherical_weight=0.01,
            sidecar_sigreg_mode="full",
            sidecar_weak_sigreg_dim=32,
            sidecar_read_rmsnorm=1,
            sidecar_summary_mode="query",
            sidecar_pred_target_mode="delta",
            adapt_sidecar=0,
            adapt_lr=1e-4,
            adapt_beta1=0.9,
            adapt_beta2=0.999,
            adapt_recompute_carry=1,
        )
    )
    base_mod, sidecar_mod = ess.load_modules()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    flat_state = ess.load_flat_state(artifact_path, base_mod)
    hps = sidecar_mod.Hyperparameters()
    model = sidecar_mod.make_sidecar_gpt(hps, sp)
    model.update(tree_unflatten(list(flat_state.items())))
    if hasattr(model, "set_turbo_qat"):
        model.set_turbo_qat(False, 0.0)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()
    routing = base_mod.build_operator_routing_spec(sp, hps.vocab_size)
    return base_mod, sp, model, routing


def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_z


def token_nll_from_logits(logits: mx.array, targets: np.ndarray) -> np.ndarray:
    logits_np = np.asarray(logits.astype(mx.float32), dtype=np.float32)[0]
    log_probs = stable_log_softmax(logits_np)
    return -log_probs[np.arange(targets.size), targets]


def load_chunk(data_path: Path, shard_name: str, chunk_index: int, chunk_size: int) -> np.ndarray:
    shard_path = data_path / shard_name
    tokens = np.memmap(shard_path, dtype="<u2", mode="r", offset=256 * np.dtype("<i4").itemsize)
    start = int(chunk_index) * int(chunk_size)
    chunk = np.asarray(tokens[start : start + chunk_size + 1], dtype=np.int32)
    if chunk.shape[0] != chunk_size + 1:
        raise ValueError(f"Short chunk read for {shard_name} chunk {chunk_index}")
    return chunk


def operator_windows(codes: np.ndarray, code_id: int, max_distance: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    positions = np.where(codes == code_id)[0]
    for d in range(max_distance + 1):
        idx = positions + d
        idx = idx[(idx >= 0) & (idx < codes.size)]
        out[d] = idx
    return out


def load_operator_token_ids(path: str) -> set[int]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = payload.get("token_ids") or payload.get("operator_token_ids") or []
    return {int(x) for x in ids}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()

    frame = load_sample_frame(sample_csv, args.max_samples)
    _sp_base, baseline_model = build_baseline_model(tokenizer_path, Path(args.baseline_artifact).expanduser().resolve())
    base_mod, _sp_sidecar, sidecar_model, routing = build_sidecar_model(args, tokenizer_path, Path(args.sidecar_artifact).expanduser().resolve())
    operator_token_ids = load_operator_token_ids(args.operator_token_json)

    rows: list[dict[str, object]] = []
    not_profile: dict[str, dict[str, float]] = {
        str(d): {"count": 0.0, "baseline_sum": 0.0, "sidecar_sum": 0.0}
        for d in range(args.operator_window + 1)
    }

    for row in frame.itertuples(index=False):
        chunk = load_chunk(data_path, row.shard_name, int(row.chunk_index), args.chunk_size)
        x_np = chunk[:-1].reshape(1, args.chunk_size).astype(np.int32, copy=False)
        y_np = chunk[1:].astype(np.int32, copy=False)
        x = mx.array(x_np, dtype=mx.int32)

        base_logits = baseline_model.forward_logits(x)
        base_nll = token_nll_from_logits(base_logits, y_np)

        raw_codes = base_mod.detect_operator_codes_np(x_np, routing)
        op_codes = mx.array(raw_codes.astype(np.int32), dtype=mx.int32)
        side_logits = sidecar_model.forward_logits(x, operator_codes=op_codes)
        side_nll = token_nll_from_logits(side_logits, y_np)

        not_codes = raw_codes[0]
        windows = operator_windows(not_codes, 1, args.operator_window)
        for d, idx in windows.items():
            if idx.size == 0:
                continue
            bucket = not_profile[str(d)]
            bucket["count"] += float(idx.size)
            bucket["baseline_sum"] += float(base_nll[idx].sum())
            bucket["sidecar_sum"] += float(side_nll[idx].sum())

        token_ids = x_np.reshape(-1)
        not_piece_hits = int(np.sum(np.isin(token_ids, list(operator_token_ids)))) if operator_token_ids else 0
        rows.append(
            {
                "sample_index": int(row.sample_index),
                "shard_id": int(row.shard_id),
                "shard_name": row.shard_name,
                "chunk_index": int(row.chunk_index),
                "cluster_id": int(row.cluster_id),
                "compressibility": float(row.compressibility),
                "operator_density": float(row.operator_density),
                "baseline_loss": float(base_nll.mean()),
                "sidecar_loss": float(side_nll.mean()),
                "sidecar_delta": float(side_nll.mean() - base_nll.mean()),
                "baseline_loss_std": float(base_nll.std()),
                "sidecar_loss_std": float(side_nll.std()),
                "not_operator_count": int(np.sum(not_codes == 1)),
                "and_operator_count": int(np.sum(not_codes == 2)),
                "or_operator_count": int(np.sum(not_codes == 3)),
                "operator_window_tokens": int(np.sum(not_codes > 0)),
                "not_piece_hits": not_piece_hits,
            }
        )

    scored = pd.DataFrame.from_records(rows)
    merged = frame.merge(scored, on=["sample_index", "shard_id", "shard_name", "chunk_index", "cluster_id", "compressibility", "operator_density"], how="left")
    scored_path = out_dir / "curriculum_scored_sample.csv.gz"
    merged.to_csv(scored_path, index=False, compression="gzip")

    profile = {}
    for key, bucket in not_profile.items():
        count = max(bucket["count"], 1.0)
        baseline_mean = bucket["baseline_sum"] / count
        sidecar_mean = bucket["sidecar_sum"] / count
        profile[key] = {
            "count": int(bucket["count"]),
            "baseline_mean_nll": float(baseline_mean),
            "sidecar_mean_nll": float(sidecar_mean),
            "delta": float(sidecar_mean - baseline_mean),
        }

    summary = {
        "sample_csv": str(sample_csv),
        "baseline_artifact": str(Path(args.baseline_artifact).expanduser().resolve()),
        "sidecar_artifact": str(Path(args.sidecar_artifact).expanduser().resolve()),
        "num_samples": int(len(merged)),
        "baseline_loss_mean": float(merged["baseline_loss"].mean()),
        "sidecar_loss_mean": float(merged["sidecar_loss"].mean()),
        "sidecar_delta_mean": float(merged["sidecar_delta"].mean()),
        "sidecar_delta_p10": float(np.quantile(merged["sidecar_delta"], 0.10)),
        "sidecar_delta_p50": float(np.quantile(merged["sidecar_delta"], 0.50)),
        "sidecar_delta_p90": float(np.quantile(merged["sidecar_delta"], 0.90)),
        "compressibility_corr_sidecar_delta": float(np.corrcoef(merged["compressibility"], merged["sidecar_delta"])[0, 1]),
        "operator_density_corr_sidecar_delta": float(np.corrcoef(merged["operator_density"], merged["sidecar_delta"])[0, 1]),
        "not_loss_profile": profile,
        "output_csv": str(scored_path),
    }
    (out_dir / "curriculum_scored_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
