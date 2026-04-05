#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curriculum import (
    ChunkFeatures,
    chunk_token_matrix,
    classify_replay_buckets,
    cosine_kmeans,
    hashed_token_histograms,
    operator_density,
    zlib_compressibility_ratio,
)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


def parse_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return []
    return [int(part) for part in parts]


def load_operator_ids(raw_ids: str, json_path: str) -> list[int]:
    ids = set(parse_int_list(raw_ids))
    if json_path:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Operator token JSON must be an object: {json_path}")
        json_ids = payload.get("token_ids") or payload.get("operator_token_ids") or []
        if not isinstance(json_ids, list):
            raise ValueError(f"Operator token JSON must contain token_ids or operator_token_ids array: {json_path}")
        ids.update(int(value) for value in json_ids)
    return sorted(ids)


def load_optional_vector(spec: str, expected_len: int) -> np.ndarray | None:
    if not spec:
        return None
    path_str = spec
    key = ""
    if ":" in spec and spec.rsplit(":", 1)[0].endswith(".npz"):
        path_str, key = spec.rsplit(":", 1)
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Optional curriculum score file not found: {spec}")
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        if key:
            if key not in payload:
                raise KeyError(f"Key {key!r} not found in {path}")
            array = payload[key]
        else:
            keys = list(payload.keys())
            if len(keys) != 1:
                raise ValueError(f"NPZ spec without key must contain exactly one array: {path}")
            array = payload[keys[0]]
    else:
        raise ValueError(f"Unsupported optional vector format: {spec}")
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.shape[0] < expected_len:
        raise ValueError(f"Optional curriculum score {spec} has {flat.shape[0]} items, expected at least {expected_len}")
    return flat[:expected_len]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chunk-level curriculum features from cached FineWeb-style token shards."
    )
    parser.add_argument("input_glob", help="Shard glob, for example data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("output", help="Output .npz path")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Tokens per chunk")
    parser.add_argument("--histogram-bins", type=int, default=256, help="Hashed histogram bins per chunk")
    parser.add_argument("--num-clusters", type=int, default=256, help="Number of structural clusters")
    parser.add_argument("--kmeans-iterations", type=int, default=8, help="K-means refinement steps")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for clustering")
    parser.add_argument("--max-shards", type=int, default=0, help="Optional cap on shards to process")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap on chunks to keep")
    parser.add_argument("--zlib-level", type=int, default=6, help="zlib level for chunk compressibility scoring")
    parser.add_argument("--learnability-spec", default="", help="Optional .npy or .npz:key vector aligned to chunk order")
    parser.add_argument("--quality-spec", default="", help="Optional .npy or .npz:key vector aligned to chunk order")
    parser.add_argument("--operator-token-json", default="", help="Optional JSON artifact containing token_ids or operator_token_ids")
    parser.add_argument(
        "--operator-token-ids",
        default="",
        help="Comma-separated token ids used as operator markers for density scoring",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")
    if args.max_shards > 0:
        files = files[: args.max_shards]

    operator_ids = load_operator_ids(args.operator_token_ids, args.operator_token_json)
    histograms: list[np.ndarray] = []
    operator_scores: list[np.ndarray] = []
    compressibility_scores: list[np.ndarray] = []
    shard_index: list[np.ndarray] = []
    chunk_index: list[np.ndarray] = []
    chunks_remaining = args.max_chunks if args.max_chunks > 0 else None

    for shard_id, path in enumerate(files):
        tokens = load_data_shard(path)
        chunks = chunk_token_matrix(tokens, args.chunk_size)
        if chunks_remaining is not None:
            if chunks_remaining <= 0:
                break
            chunks = chunks[:chunks_remaining]
        if chunks.size == 0:
            continue
        histograms.append(hashed_token_histograms(chunks, args.chunk_size, args.histogram_bins))
        operator_scores.append(operator_density(chunks, args.chunk_size, operator_ids))
        compressibility_scores.append(zlib_compressibility_ratio(chunks, args.chunk_size, level=args.zlib_level))
        shard_index.append(np.full((chunks.shape[0],), shard_id, dtype=np.int32))
        chunk_index.append(np.arange(chunks.shape[0], dtype=np.int32))
        if chunks_remaining is not None:
            chunks_remaining -= int(chunks.shape[0])

    if not histograms:
        raise SystemExit("No chunks available after applying the current limits")

    hist = np.concatenate(histograms, axis=0)
    operator_score = np.concatenate(operator_scores, axis=0)
    compressibility_ratio = np.concatenate(compressibility_scores, axis=0)
    shard_ids = np.concatenate(shard_index, axis=0)
    chunk_ids = np.concatenate(chunk_index, axis=0)

    if args.max_chunks > 0 and hist.shape[0] > args.max_chunks:
        hist = hist[: args.max_chunks]
        operator_score = operator_score[: args.max_chunks]
        compressibility_ratio = compressibility_ratio[: args.max_chunks]
        shard_ids = shard_ids[: args.max_chunks]
        chunk_ids = chunk_ids[: args.max_chunks]

    cluster_ids = np.zeros((hist.shape[0],), dtype=np.int32)
    centers = np.zeros((0, hist.shape[1]), dtype=np.float32)
    if args.num_clusters > 0:
        cluster_ids, centers = cosine_kmeans(
            hist,
            num_clusters=args.num_clusters,
            iterations=args.kmeans_iterations,
            seed=args.seed,
        )

    entropy = -np.sum(hist * np.log(np.clip(hist, 1e-12, None)), axis=1, dtype=np.float32)
    entropy = entropy / max(np.log(max(args.histogram_bins, 2)), 1e-6)
    duplicate_score = np.max(hist, axis=1).astype(np.float32)
    learnability_score = load_optional_vector(args.learnability_spec, hist.shape[0])
    quality_score = load_optional_vector(args.quality_spec, hist.shape[0])
    replay_bucket = classify_replay_buckets(
        ChunkFeatures(
            cluster_ids=cluster_ids,
            operator_density=operator_score,
            difficulty=entropy.astype(np.float32),
            duplicate_score=duplicate_score,
            compressibility_ratio=compressibility_ratio.astype(np.float32),
            learnability_score=learnability_score,
            quality_score=quality_score,
        )
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_size": np.array([args.chunk_size], dtype=np.int32),
        "histograms": hist.astype(np.float32),
        "operator_density": operator_score.astype(np.float32),
        "difficulty": entropy.astype(np.float32),
        "duplicate_score": duplicate_score.astype(np.float32),
        "compressibility_ratio": compressibility_ratio.astype(np.float32),
        "replay_bucket": replay_bucket.astype(np.int8),
        "cluster_ids": cluster_ids.astype(np.int32),
        "cluster_centers": centers.astype(np.float32),
        "shard_index": shard_ids.astype(np.int32),
        "chunk_index": chunk_ids.astype(np.int32),
    }
    if learnability_score is not None:
        payload["learnability_score"] = learnability_score.astype(np.float32)
    if quality_score is not None:
        payload["quality_score"] = quality_score.astype(np.float32)
    np.savez_compressed(output_path, **payload)

    summary = {
        "chunk_size": args.chunk_size,
        "histogram_bins": args.histogram_bins,
        "input_glob": args.input_glob,
        "kmeans_iterations": args.kmeans_iterations,
        "max_chunks": args.max_chunks,
        "max_shards": args.max_shards,
        "num_chunks": int(hist.shape[0]),
        "num_clusters": int(centers.shape[0]),
        "operator_token_ids": operator_ids,
        "operator_token_json": args.operator_token_json,
        "output": output_path.as_posix(),
        "seed": args.seed,
        "learnability_spec": args.learnability_spec,
        "quality_spec": args.quality_spec,
        "zlib_level": args.zlib_level,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
