#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _collect_layer_records(
    representations: list[ModelRepresentation],
    *,
    relative_depth: float,
    canonical_dim: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rep in representations:
        source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
        mapped = _map_directions(
            source_layer.directions,
            canonical_dim=canonical_dim,
            seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:weight_merge"),
        )
        mapped = _normalize_rows(mapped)
        if source_layer.scales is None or source_layer.scales.size == 0:
            scales = np.ones((mapped.shape[0],), dtype=np.float32)
        else:
            scales = np.asarray(source_layer.scales, dtype=np.float32).reshape(-1)
            scales = np.abs(scales[: mapped.shape[0]])
        for direction_idx in range(mapped.shape[0]):
            records.append(
                {
                    "model_id": rep.model_id,
                    "source_layer_idx": int(source_layer_idx),
                    "relative_depth": relative_depth,
                    "direction_idx": int(direction_idx),
                    "scale": float(scales[direction_idx]),
                    "vector": mapped[direction_idx],
                    "source_hidden_dim": int(rep.hidden_dim),
                }
            )
    records.sort(key=lambda record: (float(record["scale"]), str(record["model_id"])), reverse=True)
    return records


def _cluster_direction_records(
    records: list[dict[str, object]],
    *,
    similarity_threshold: float,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for record in records:
        vector = np.asarray(record["vector"], dtype=np.float32)
        best_idx = -1
        best_similarity = -1.0
        for idx, cluster in enumerate(clusters):
            centroid = np.asarray(cluster["centroid"], dtype=np.float32)
            similarity = float(abs(np.dot(vector, centroid)))
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        if best_idx >= 0 and best_similarity >= similarity_threshold:
            cluster = clusters[best_idx]
            members = cluster["members"]
            members.append(record)
            weighted_vectors = np.stack(
                [np.asarray(member["vector"], dtype=np.float32) * max(float(member["scale"]), 1e-8) for member in members],
                axis=0,
            )
            centroid = weighted_vectors.sum(axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 1e-8:
                centroid = centroid / centroid_norm
            champion = max(members, key=lambda member: (float(member["scale"]), str(member["model_id"])))
            cluster["centroid"] = centroid.astype(np.float32)
            cluster["champion"] = champion
            cluster["max_similarity"] = max(float(cluster["max_similarity"]), best_similarity)
        else:
            clusters.append(
                {
                    "centroid": vector.astype(np.float32),
                    "champion": record,
                    "max_similarity": 1.0,
                    "members": [record],
                }
            )
    return clusters


def _layer_geometry_from_clusters(
    clusters: list[dict[str, object]],
    *,
    relative_depth: float,
    canonical_dim: int,
    top_k: int,
) -> LayerGeometry:
    ranked = sorted(
        clusters,
        key=lambda cluster: (
            float(cluster["champion"]["scale"]) * max(len(cluster["members"]), 1),
            float(cluster["champion"]["scale"]),
            str(cluster["champion"]["model_id"]),
        ),
        reverse=True,
    )[: max(int(top_k), 1)]
    if not ranked:
        raise RuntimeError("No ranked clusters available to build a merged layer")
    directions = _normalize_rows(np.stack([np.asarray(cluster["champion"]["vector"], dtype=np.float32) for cluster in ranked], axis=0))
    scales = np.asarray([float(cluster["champion"]["scale"]) for cluster in ranked], dtype=np.float32)
    covariance = directions.T @ np.diag(scales.astype(np.float64)) @ directions
    coactivation = directions @ covariance.astype(np.float32) @ directions.T
    importance = np.abs(directions.T) @ np.abs(scales)
    metadata_clusters: list[dict[str, object]] = []
    for cluster_idx, cluster in enumerate(ranked):
        members = cluster["members"]
        source_models = sorted({str(member["model_id"]) for member in members})
        metadata_clusters.append(
            {
                "cluster_idx": int(cluster_idx),
                "champion_model": str(cluster["champion"]["model_id"]),
                "champion_scale": float(cluster["champion"]["scale"]),
                "champion_direction_idx": int(cluster["champion"]["direction_idx"]),
                "champion_source_layer_idx": int(cluster["champion"]["source_layer_idx"]),
                "support": int(len(members)),
                "source_models": source_models,
                "max_similarity": float(cluster["max_similarity"]),
            }
        )
    return LayerGeometry(
        relative_depth=relative_depth,
        directions=directions.astype(np.float32),
        scales=scales.astype(np.float32),
        covariance=covariance.astype(np.float32),
        coactivation=coactivation.astype(np.float32),
        importance=np.asarray(importance, dtype=np.float32),
        metadata={
            "selection_method": "weight_spectral_argmax",
            "clusters": metadata_clusters,
        },
    )


def merge_spectral_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
) -> PlatonicGeometry:
    if not representations:
        raise ValueError("representations must not be empty")
    layer_geometries: dict[int, LayerGeometry] = {}
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        records = _collect_layer_records(
            representations,
            relative_depth=relative_depth,
            canonical_dim=canonical_dim,
        )
        clusters = _cluster_direction_records(records, similarity_threshold=similarity_threshold)
        layer_geometries[target_layer] = _layer_geometry_from_clusters(
            clusters,
            relative_depth=relative_depth,
            canonical_dim=canonical_dim,
            top_k=top_k,
        )
    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=[rep.model_id for rep in representations],
        metadata={
            "selection_method": "spectral_argmax",
            "source_extraction_methods": sorted(
                {str(rep.metadata.get("extraction_method", "unknown")) for rep in representations}
            ),
            "similarity_threshold": float(similarity_threshold),
            "top_k": int(top_k),
            "num_layers": int(num_layers),
        },
    )


def merge_weight_spectral_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
) -> PlatonicGeometry:
    geometry = merge_spectral_representations(
        representations,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    geometry.metadata["selection_method"] = "weight_spectral_argmax"
    return geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge spectral ModelRepresentation artifacts by selecting the strongest aligned direction per cluster.")
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input spectral ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Canonical dimension used for cross-model comparison")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins in the merged geometry")
    parser.add_argument("--top-k", type=int, default=16, help="Directions to retain per merged layer")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="Absolute cosine threshold for clustering source directions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_spectral_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "canonical_dim": args.canonical_dim,
                "num_layers": args.num_layers,
                "output": str(output_path),
                "selection_method": geometry.metadata["selection_method"],
                "similarity_threshold": args.similarity_threshold,
                "source_models": geometry.source_models,
                "top_k": args.top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
