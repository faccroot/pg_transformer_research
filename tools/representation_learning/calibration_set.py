#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curriculum import cosine_kmeans, hashed_token_histograms, operator_density, zlib_compressibility_ratio


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


def chunk_token_matrix(tokens: np.ndarray, chunk_size: int) -> np.ndarray:
    usable = (tokens.shape[0] // chunk_size) * chunk_size
    if usable <= 0:
        return np.zeros((0, chunk_size), dtype=np.int32)
    return tokens[:usable].reshape(-1, chunk_size)


def parse_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in parts]


def load_operator_ids(raw_ids: str, json_path: str) -> list[int]:
    ids = set(parse_int_list(raw_ids))
    if json_path:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        json_ids = payload.get("token_ids") or payload.get("operator_token_ids") or []
        if not isinstance(json_ids, list):
            raise ValueError(f"Operator token JSON must contain token_ids or operator_token_ids array: {json_path}")
        ids.update(int(value) for value in json_ids)
    return sorted(ids)


def load_sentencepiece_processor(tokenizer_path: str):
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentencepiece is required to build decoded calibration examples. "
            "Install requirements-representation-learning.txt on the extraction host."
        ) from exc
    processor = spm.SentencePieceProcessor()
    if not processor.load(tokenizer_path):
        raise RuntimeError(f"Failed to load SentencePiece tokenizer: {tokenizer_path}")
    return processor


@dataclass(frozen=True)
class CalibrationRecord:
    chunk_id: str
    shard_index: int
    chunk_index: int
    text: str
    cluster_id: int
    difficulty: float
    operator_density: float
    compressibility_ratio: float
    duplicate_score: float


def stratified_round_robin_indices(cluster_ids: np.ndarray, sample_size: int) -> np.ndarray:
    groups = {
        int(cluster_id): np.where(cluster_ids == cluster_id)[0].astype(np.int32)
        for cluster_id in np.unique(cluster_ids)
    }
    cursors = {cluster_id: 0 for cluster_id in groups}
    selected: list[int] = []
    ordered_clusters = sorted(groups)
    while len(selected) < sample_size:
        made_progress = False
        for cluster_id in ordered_clusters:
            indices = groups[cluster_id]
            cursor = cursors[cluster_id]
            if cursor >= indices.shape[0]:
                continue
            selected.append(int(indices[cursor]))
            cursors[cluster_id] = cursor + 1
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return np.asarray(selected, dtype=np.int32)


def build_records(
    *,
    input_glob: str,
    tokenizer_path: str,
    chunk_size: int,
    histogram_bins: int,
    num_clusters: int,
    kmeans_iterations: int,
    sample_size: int,
    max_shards: int,
    max_chunks: int,
    zlib_level: int,
    operator_ids: list[int],
) -> tuple[list[CalibrationRecord], dict[str, object]]:
    expanded_input_glob = str(Path(input_glob).expanduser())
    tokenizer_path = str(Path(tokenizer_path).expanduser())
    files = [Path(path) for path in sorted(glob.glob(expanded_input_glob))]
    if not files:
        raise FileNotFoundError(f"No files matched: {expanded_input_glob}")
    if max_shards > 0:
        files = files[:max_shards]

    chunks_all: list[np.ndarray] = []
    shard_indices_all: list[np.ndarray] = []
    chunk_indices_all: list[np.ndarray] = []
    chunks_remaining = max_chunks if max_chunks > 0 else None
    for shard_idx, path in enumerate(files):
        shard_tokens = load_data_shard(path)
        chunks = chunk_token_matrix(shard_tokens, chunk_size)
        if chunks_remaining is not None:
            if chunks_remaining <= 0:
                break
            chunks = chunks[:chunks_remaining]
        if chunks.size == 0:
            continue
        chunks_all.append(chunks)
        shard_indices_all.append(np.full((chunks.shape[0],), shard_idx, dtype=np.int32))
        chunk_indices_all.append(np.arange(chunks.shape[0], dtype=np.int32))
        if chunks_remaining is not None:
            chunks_remaining -= int(chunks.shape[0])

    if not chunks_all:
        raise RuntimeError("No chunks available for calibration-set construction")

    chunk_matrix = np.concatenate(chunks_all, axis=0)
    shard_indices = np.concatenate(shard_indices_all, axis=0)
    chunk_indices = np.concatenate(chunk_indices_all, axis=0)
    hist = hashed_token_histograms(chunk_matrix, chunk_size, histogram_bins)
    operator_score = operator_density(chunk_matrix, chunk_size, operator_ids)
    compressibility = zlib_compressibility_ratio(chunk_matrix, chunk_size, level=zlib_level)
    duplicate_score = np.max(hist, axis=1).astype(np.float32)
    entropy = -np.sum(hist * np.log(np.clip(hist, 1e-12, None)), axis=1, dtype=np.float32)
    entropy = entropy / max(np.log(max(histogram_bins, 2)), 1e-6)
    cluster_ids = np.zeros((hist.shape[0],), dtype=np.int32)
    if num_clusters > 0:
        cluster_ids, _centers = cosine_kmeans(hist, num_clusters=num_clusters, iterations=kmeans_iterations, seed=17)

    sample_size = min(max(sample_size, 1), int(chunk_matrix.shape[0]))
    selected = stratified_round_robin_indices(cluster_ids, sample_size=sample_size)
    processor = load_sentencepiece_processor(tokenizer_path)
    records: list[CalibrationRecord] = []
    for selected_idx in selected.tolist():
        chunk = chunk_matrix[selected_idx]
        chunk_id = f"shard{int(shard_indices[selected_idx]):03d}_chunk{int(chunk_indices[selected_idx]):07d}"
        text = processor.decode(chunk.astype(int).tolist())
        records.append(
            CalibrationRecord(
                chunk_id=chunk_id,
                shard_index=int(shard_indices[selected_idx]),
                chunk_index=int(chunk_indices[selected_idx]),
                text=text,
                cluster_id=int(cluster_ids[selected_idx]),
                difficulty=float(entropy[selected_idx]),
                operator_density=float(operator_score[selected_idx]),
                compressibility_ratio=float(compressibility[selected_idx]),
                duplicate_score=float(duplicate_score[selected_idx]),
            )
        )
    summary = {
        "chunk_size": chunk_size,
        "histogram_bins": histogram_bins,
        "input_glob": expanded_input_glob,
        "kmeans_iterations": kmeans_iterations,
        "max_chunks": max_chunks,
        "max_shards": max_shards,
        "num_clusters": num_clusters,
        "num_records": len(records),
        "operator_token_ids": operator_ids,
        "sample_size": sample_size,
        "tokenizer_path": tokenizer_path,
        "zlib_level": zlib_level,
    }
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a fixed decoded calibration set from cached FineWeb token shards.")
    parser.add_argument("input_glob", help="Shard glob, for example data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("output", help="Output .jsonl path")
    parser.add_argument("--tokenizer-path", required=True, help="SentencePiece tokenizer used by the cached shards")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Tokens per chunk")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of calibration chunks to emit")
    parser.add_argument("--histogram-bins", type=int, default=256, help="Hashed histogram bins per chunk")
    parser.add_argument("--num-clusters", type=int, default=256, help="Number of structural clusters")
    parser.add_argument("--kmeans-iterations", type=int, default=8, help="K-means refinement steps")
    parser.add_argument("--max-shards", type=int, default=4, help="Optional shard cap for the first extraction pass")
    parser.add_argument("--max-chunks", type=int, default=50000, help="Optional chunk cap before stratified sampling")
    parser.add_argument("--zlib-level", type=int, default=6, help="zlib level for compressibility scoring")
    parser.add_argument("--operator-token-json", default="", help="Optional JSON artifact containing token_ids or operator_token_ids")
    parser.add_argument("--operator-token-ids", default="", help="Comma-separated token ids used as operator markers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    operator_ids = load_operator_ids(args.operator_token_ids, args.operator_token_json)
    records, summary = build_records(
        input_glob=args.input_glob,
        tokenizer_path=args.tokenizer_path,
        chunk_size=args.chunk_size,
        histogram_bins=args.histogram_bins,
        num_clusters=args.num_clusters,
        kmeans_iterations=args.kmeans_iterations,
        sample_size=args.sample_size,
        max_shards=args.max_shards,
        max_chunks=args.max_chunks,
        zlib_level=args.zlib_level,
        operator_ids=operator_ids,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
