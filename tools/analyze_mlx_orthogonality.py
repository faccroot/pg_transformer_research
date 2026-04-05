#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gram_offdiag_metrics(w: np.ndarray) -> dict[str, float]:
    if w.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    m, n = w.shape
    if m >= n:
        gram = w.T @ w
    else:
        gram = w @ w.T
    d = gram.shape[0]
    diag = np.diag(np.diag(gram))
    offdiag = gram - diag
    eye = np.eye(d, dtype=np.float64)
    return {
        "gram_dim": float(d),
        "offdiag_fro_per_dim": float(np.linalg.norm(offdiag, ord="fro") / max(d, 1)),
        "offdiag_rel_fro": float(np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(gram, ord="fro"), 1e-12)),
        "identity_fro_per_dim": float(np.linalg.norm(gram - eye, ord="fro") / max(d, 1)),
        "diag_mean": float(np.mean(np.diag(gram), dtype=np.float64)),
        "diag_std": float(np.std(np.diag(gram), dtype=np.float64)),
    }


def cosine_gram_metrics(w: np.ndarray) -> dict[str, float]:
    if w.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    m, n = w.shape
    if m >= n:
        cols = w
    else:
        cols = w.T
    norms = np.linalg.norm(cols, axis=0, ord=2, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = cols / norms
    gram = normed.T @ normed
    d = gram.shape[0]
    diag = np.diag(np.diag(gram))
    offdiag = gram - diag
    eye = np.eye(d, dtype=np.float64)
    return {
        "cosine_offdiag_fro_per_dim": float(np.linalg.norm(offdiag, ord="fro") / max(d, 1)),
        "cosine_offdiag_rel_fro": float(np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(gram, ord="fro"), 1e-12)),
        "cosine_identity_fro_per_dim": float(np.linalg.norm(gram - eye, ord="fro") / max(d, 1)),
    }


def svd_metrics(w: np.ndarray) -> dict[str, float]:
    singular_values = np.linalg.svd(w.astype(np.float64), compute_uv=False)
    min_sv = float(singular_values[-1])
    max_sv = float(singular_values[0])
    cond = float(max_sv / max(min_sv, 1e-12))
    stable_rank = float(np.square(singular_values).sum() / max(np.square(singular_values[0]), 1e-12))
    return {
        "sv_max": max_sv,
        "sv_min": min_sv,
        "condition_number": cond,
        "stable_rank": stable_rank,
    }


def analyze_matrix(name: str, w: np.ndarray) -> dict[str, float | str | list[int]]:
    metrics = {
        "name": name,
        "shape": list(w.shape),
        "dtype": str(w.dtype),
        "fro_norm": float(np.linalg.norm(w.astype(np.float64), ord="fro")),
    }
    metrics.update(gram_offdiag_metrics(w.astype(np.float64)))
    metrics.update(cosine_gram_metrics(w.astype(np.float64)))
    metrics.update(svd_metrics(w.astype(np.float64)))
    return metrics


def summarize(matrices: list[dict[str, float | str | list[int]]]) -> dict[str, dict[str, float]]:
    groups = {
        "all": matrices,
        "attn_q": [m for m in matrices if ".attn.c_q.weight" in str(m["name"])],
        "attn_k": [m for m in matrices if ".attn.c_k.weight" in str(m["name"])],
        "attn_proj": [m for m in matrices if ".attn.proj.weight" in str(m["name"])],
        "lm_head": [m for m in matrices if str(m["name"]) == "lm_head.weight"],
    }
    out: dict[str, dict[str, float]] = {}
    numeric_keys = [
        "offdiag_fro_per_dim",
        "offdiag_rel_fro",
        "identity_fro_per_dim",
        "cosine_offdiag_fro_per_dim",
        "cosine_offdiag_rel_fro",
        "cosine_identity_fro_per_dim",
        "condition_number",
        "stable_rank",
        "diag_mean",
        "diag_std",
    ]
    for group_name, group in groups.items():
        if not group:
            continue
        out[group_name] = {
            key: float(np.mean([float(m[key]) for m in group], dtype=np.float64))
            for key in numeric_keys
        }
        out[group_name]["count"] = float(len(group))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze orthogonality / rotation-plus-scaling structure of MLX checkpoints.")
    parser.add_argument("checkpoint", type=Path, help="Path to an .npz checkpoint")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--include-rectangular", action="store_true", help="Include rectangular 2D tensors via smaller Gram matrix")
    args = parser.parse_args()

    ckpt = np.load(args.checkpoint, allow_pickle=False)
    matrices: list[dict[str, float | str | list[int]]] = []
    for name in ckpt.files:
        w = ckpt[name]
        if w.ndim != 2:
            continue
        if not np.issubdtype(w.dtype, np.number):
            continue
        if not args.include_rectangular and w.shape[0] != w.shape[1]:
            continue
        matrices.append(analyze_matrix(name, np.asarray(w)))

    matrices.sort(key=lambda item: str(item["name"]))
    payload = {
        "checkpoint": str(args.checkpoint),
        "include_rectangular": args.include_rectangular,
        "num_matrices": len(matrices),
        "summary": summarize(matrices),
        "matrices": matrices,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
