#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_matrix(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path) as payload:
        if key not in payload:
            raise KeyError(f"{key!r} not found in {npz_path}")
        matrix = np.asarray(payload[key], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for {key!r}, got shape {matrix.shape}")
    return matrix


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    min_dist_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for idx in range(1, k):
        probs = min_dist_sq / np.maximum(np.sum(min_dist_sq), 1e-12)
        choice = int(rng.choice(n, p=probs))
        centers[idx] = x[choice]
        dist_sq = np.sum((x - centers[idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return centers


def assign_clusters(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist_sq = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dist_sq, axis=1).astype(np.int32)
    min_dist = dist_sq[np.arange(dist_sq.shape[0]), assign]
    return assign, min_dist


def run_kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = kmeans_pp_init(x, k, rng)
    assign = np.zeros((x.shape[0],), dtype=np.int32)
    for _ in range(iters):
        assign, min_dist = assign_clusters(x, centers)
        for cluster_id in range(k):
            mask = assign == cluster_id
            if np.any(mask):
                centers[cluster_id] = np.mean(x[mask], axis=0, dtype=np.float32)
            else:
                farthest = int(np.argmax(min_dist))
                centers[cluster_id] = x[farthest]
        centers = l2_normalize_rows(centers)
    assign, _ = assign_clusters(x, centers)
    return assign, centers


def maybe_load_pieces(tokenizer_path: Path | None, vocab_size: int) -> list[str] | None:
    if tokenizer_path is None:
        return None
    try:
        import sentencepiece as spm
    except Exception:
        return None
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if int(sp.vocab_size()) != vocab_size:
        return None
    return [str(sp.id_to_piece(i)) for i in range(vocab_size)]


def write_summary(
    output_path: Path,
    *,
    assign: np.ndarray,
    pieces: list[str] | None,
    counts: list[int],
    checkpoint: Path,
    matrix_key: str,
) -> None:
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "matrix_key": matrix_key,
        "num_tokens": int(assign.shape[0]),
        "num_groups": int(len(counts)),
        "group_sizes": counts,
    }
    if pieces is not None:
        groups: list[dict[str, object]] = []
        for group_id in range(len(counts)):
            ids = np.where(assign == group_id)[0].tolist()
            groups.append(
                {
                    "group_id": group_id,
                    "size": len(ids),
                    "sample_ids": ids[:16],
                    "sample_pieces": [pieces[idx] for idx in ids[:16]],
                }
            )
        payload["groups"] = groups
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed token->group map by k-means clustering a trained token matrix.")
    parser.add_argument("checkpoint", type=Path, help="Path to a local .npz checkpoint")
    parser.add_argument("output", type=Path, help="Path to output .npy token->group map")
    parser.add_argument("--matrix-key", default="lm_head.weight", help="Checkpoint key to cluster")
    parser.add_argument("--groups", type=int, default=32, help="Number of token groups")
    parser.add_argument("--iters", type=int, default=20, help="Number of k-means iterations")
    parser.add_argument("--seed", type=int, default=17, help="K-means RNG seed")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional SentencePiece model for human-readable cluster summaries")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary output")
    args = parser.parse_args()

    matrix = load_matrix(args.checkpoint, args.matrix_key)
    matrix = l2_normalize_rows(matrix.astype(np.float32))
    assign, _ = run_kmeans(matrix, args.groups, args.iters, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, assign.astype(np.int32))

    if args.summary_json is not None:
        pieces = maybe_load_pieces(args.tokenizer_path, matrix.shape[0])
        counts = [int(np.sum(assign == group_id)) for group_id in range(args.groups)]
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        write_summary(
            args.summary_json,
            assign=assign,
            pieces=pieces,
            counts=counts,
            checkpoint=args.checkpoint,
            matrix_key=args.matrix_key,
        )

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "matrix_key": args.matrix_key,
                "groups": args.groups,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
