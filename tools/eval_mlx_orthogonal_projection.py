#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as tg


def load_flat_state(path: Path) -> dict[str, mx.array]:
    return {name: value for name, value in mx.load(str(path)).items()}


def infer_model_config(flat_state: dict[str, mx.array]) -> dict[str, object]:
    tok_emb = flat_state["tok_emb.weight"]
    vocab_size, model_dim = map(int, tok_emb.shape)
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in flat_state
            if name.startswith("blocks.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
        }
    )
    if not block_ids:
        raise ValueError("Could not infer block structure from checkpoint")
    num_layer_templates = max(block_ids) + 1
    num_heads = int(flat_state["blocks.0.attn.q_gain"].size)
    head_dim = model_dim // num_heads
    num_kv_heads = int(flat_state["blocks.0.attn.c_k.weight"].shape[0]) // head_dim
    mlp_mult = int(flat_state["blocks.0.mlp.fc.weight"].shape[0]) // model_dim
    tie_embeddings = "lm_head.weight" not in flat_state
    skip_weights = flat_state.get("skip_weights")
    num_layers = num_layer_templates
    if skip_weights is not None and skip_weights.ndim >= 1:
        num_layers = max(num_layers, int(skip_weights.shape[0]) * 2)
    return {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_mult": mlp_mult,
        "num_layer_templates": num_layer_templates,
        "num_layers": num_layers,
        "tie_embeddings": tie_embeddings,
        "tied_embed_init_std": 0.005,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.5,
    }


def build_eval_context(
    config: dict[str, object],
    data_path: Path,
    tokenizer_path: Path,
    train_seq_len: int,
    val_max_seqs: int,
    val_batch_size: int,
    eval_seq_len: int,
    eval_stride: int,
    eval_batch_seqs: int,
) -> dict[str, object]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    effective_eval_seq_len = eval_seq_len if eval_seq_len > 0 else train_seq_len
    val_tokens = tg.limit_validation_tokens(
        tg.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), max(train_seq_len, effective_eval_seq_len)),
        effective_eval_seq_len,
        val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(sp, int(config["vocab_size"]))
    model = tg.GPT(
        vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_layers"]),
        num_layer_templates=int(config["num_layer_templates"]),
        dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        mlp_mult=int(config["mlp_mult"]),
        mlp_leaky_slope=0.5,
        tie_embeddings=bool(config["tie_embeddings"]),
        logit_chunk_tokens=0,
        logit_softcap=float(config["logit_softcap"]),
        rope_base=float(config["rope_base"]),
        tied_embed_init_std=float(config["tied_embed_init_std"]),
        qk_gain_init=float(config["qk_gain_init"]),
    )
    effective_eval_batch_seqs = (
        eval_batch_seqs
        if eval_batch_seqs > 0
        else max(max(val_batch_size // 8, effective_eval_seq_len) // effective_eval_seq_len, 1)
    )
    args = SimpleNamespace(
        train_seq_len=train_seq_len,
        eval_seq_len=eval_seq_len,
        eval_stride=eval_stride,
        eval_batch_seqs=eval_batch_seqs,
        effective_eval_seq_len=effective_eval_seq_len,
        effective_eval_batch_seqs=effective_eval_batch_seqs,
        grad_accum_steps=8,
        val_batch_size=val_batch_size,
    )
    return {
        "args": args,
        "model": model,
        "val_tokens": val_tokens,
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space_lut": has_leading_space_lut,
        "is_boundary_token_lut": is_boundary_token_lut,
    }


def eval_state(eval_ctx: dict[str, object], flat_state: dict[str, mx.array]) -> tuple[float, float]:
    model = eval_ctx["model"]
    model.clear_turbo_cache()
    model.set_turbo_qat(False, 0.0)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()
    return tg.eval_val(
        eval_ctx["args"],
        model,
        lambda x, y, operator_codes=None: model.loss(x, y, operator_codes),
        lambda x, operator_codes=None: model.forward_logits(x, operator_codes),
        eval_ctx["val_tokens"],
        eval_ctx["base_bytes_lut"],
        eval_ctx["has_leading_space_lut"],
        eval_ctx["is_boundary_token_lut"],
    )


def project_diag_orthogonal(weight: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {weight.shape}")
    rows, cols = weight.shape
    if rows > cols:
        raise ValueError(f"Expected rows <= cols for row-orthogonal projection, got {weight.shape}")
    row_scales = np.sqrt(np.sum(weight.astype(np.float64) ** 2, axis=1) + 1e-12, dtype=np.float64).astype(np.float32)
    normalized = weight / row_scales[:, None]
    u, _, vt = np.linalg.svd(normalized.astype(np.float64), full_matrices=False)
    ortho = (u @ vt).astype(np.float32)
    projected = row_scales[:, None] * ortho
    diff = projected.astype(np.float64) - weight.astype(np.float64)
    denom = max(float(np.square(weight.astype(np.float64)).mean()), 1e-12)
    cosine_gram = ortho @ ortho.T
    offdiag = cosine_gram - np.eye(rows, dtype=np.float32)
    return projected, {
        "rmse": float(np.sqrt(np.square(diff).mean())),
        "rel_rmse": float(np.sqrt(np.square(diff).mean() / denom)),
        "mae": float(np.abs(diff).mean()),
        "row_scale_mean": float(row_scales.mean(dtype=np.float64)),
        "row_scale_std": float(row_scales.std(dtype=np.float64)),
        "ortho_offdiag_fro_per_dim": float(np.linalg.norm(offdiag, ord="fro") / offdiag.shape[0]),
        "rows": rows,
        "cols": cols,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate square-matrix diag×orthogonal projection on an MLX checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Exact flat-state tensor name or comma-separated list to project")
    parser.add_argument("--data-path", type=Path, default=Path("~/data/datasets/fineweb10B_sp1024").expanduser())
    parser.add_argument("--tokenizer-path", type=Path, default=Path("~/data/tokenizers/fineweb_1024_bpe.model").expanduser())
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--val-max-seqs", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flat_state = load_flat_state(args.checkpoint)
    projected_state = dict(flat_state)
    targets = [part.strip() for part in args.target.split(",") if part.strip()]
    if not targets:
        raise SystemExit("No targets parsed")
    projection_metrics: dict[str, dict[str, float]] = {}
    for target_name in targets:
        if target_name not in flat_state:
            raise SystemExit(f"Target not found: {target_name}")
        target_np = np.asarray(flat_state[target_name].astype(mx.float32), dtype=np.float32)
        projected_np, metrics = project_diag_orthogonal(target_np)
        projected_state[target_name] = mx.array(projected_np, dtype=flat_state[target_name].dtype)
        projection_metrics[target_name] = metrics

    config = infer_model_config(flat_state)
    eval_ctx = build_eval_context(
        config,
        args.data_path.expanduser(),
        args.tokenizer_path.expanduser(),
        args.train_seq_len,
        args.val_max_seqs,
        args.val_batch_size,
        args.eval_seq_len,
        args.eval_stride,
        args.eval_batch_seqs,
    )

    baseline_metrics: dict[str, float] | None = None
    if not args.skip_baseline:
        base_loss, base_bpb = eval_state(eval_ctx, flat_state)
        baseline_metrics = {"val_loss": base_loss, "val_bpb": base_bpb}
    proj_loss, proj_bpb = eval_state(eval_ctx, projected_state)

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "target": args.target,
        "targets": targets,
        "projection": projection_metrics,
        "projected_eval": {"val_loss": proj_loss, "val_bpb": proj_bpb},
    }
    if baseline_metrics is not None:
        result["baseline_eval"] = baseline_metrics
        result["delta_bpb"] = proj_bpb - float(baseline_metrics["val_bpb"])
        result["delta_loss"] = proj_loss - float(baseline_metrics["val_loss"])

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
