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
    from tools.representation_learning.merge_weight_spectral_representations import _cluster_direction_records, _layer_geometry_from_clusters
    from tools.representation_learning.schemas import ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from merge_weight_spectral_representations import _cluster_direction_records, _layer_geometry_from_clusters  # type: ignore[no-redef]
    from schemas import ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _chunk_index_lookup(rep: ModelRepresentation) -> dict[str, int]:
    if not rep.chunk_ids:
        return {}
    return {str(chunk_id): idx for idx, chunk_id in enumerate(rep.chunk_ids)}


def _canonical_chunk_embedding(rep: ModelRepresentation, *, target_layer: int, canonical_dim: int, num_layers: int) -> tuple[int, np.ndarray] | None:
    relative_depth = target_layer / max(num_layers, 1)
    source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
    projections = rep.chunk_layer_projections.get(int(source_layer_idx))
    if projections is None:
        return None
    mapped = _map_directions(
        source_layer.directions,
        canonical_dim=canonical_dim,
        seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:gauge_embed"),
    )
    mapped = _normalize_rows(mapped)
    projection_array = np.asarray(projections, dtype=np.float32)
    if projection_array.ndim != 2 or projection_array.shape[1] != mapped.shape[0]:
        return None
    count = min(len(rep.chunk_ids or []), int(projection_array.shape[0]))
    return int(source_layer_idx), projection_array[:count] @ mapped


def _shared_chunk_embedding_matrices(
    anchor_rep: ModelRepresentation,
    other_rep: ModelRepresentation,
    *,
    target_layer: int,
    canonical_dim: int,
    num_layers: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    anchor_payload = _canonical_chunk_embedding(anchor_rep, target_layer=target_layer, canonical_dim=canonical_dim, num_layers=num_layers)
    other_payload = _canonical_chunk_embedding(other_rep, target_layer=target_layer, canonical_dim=canonical_dim, num_layers=num_layers)
    if anchor_payload is None or other_payload is None:
        return None
    _anchor_source_layer_idx, anchor_matrix = anchor_payload
    _other_source_layer_idx, other_matrix = other_payload
    anchor_lookup = _chunk_index_lookup(anchor_rep)
    other_lookup = _chunk_index_lookup(other_rep)
    shared_ids = sorted(set(anchor_lookup).intersection(other_lookup))
    if not shared_ids:
        return None
    anchor_rows = []
    other_rows = []
    for chunk_id in shared_ids:
        anchor_idx = anchor_lookup[chunk_id]
        other_idx = other_lookup[chunk_id]
        if anchor_idx >= anchor_matrix.shape[0] or other_idx >= other_matrix.shape[0]:
            continue
        anchor_rows.append(anchor_matrix[anchor_idx])
        other_rows.append(other_matrix[other_idx])
    if not anchor_rows or not other_rows:
        return None
    return np.asarray(anchor_rows, dtype=np.float32), np.asarray(other_rows, dtype=np.float32)


def activation_gauge_rotation(
    anchor_rep: ModelRepresentation,
    other_rep: ModelRepresentation,
    *,
    target_layer: int,
    canonical_dim: int,
    num_layers: int,
) -> np.ndarray:
    shared = _shared_chunk_embedding_matrices(
        anchor_rep,
        other_rep,
        target_layer=target_layer,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
    )
    if shared is None:
        return np.eye(canonical_dim, dtype=np.float32)
    anchor_matrix, other_matrix = shared
    cross = np.asarray(other_matrix, dtype=np.float64).T @ np.asarray(anchor_matrix, dtype=np.float64)
    u, _s, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = (u @ vt).astype(np.float32)
    return rotation


def _aligned_records_for_rep(
    rep: ModelRepresentation,
    *,
    anchor_rep: ModelRepresentation,
    target_layer: int,
    canonical_dim: int,
    num_layers: int,
) -> list[dict[str, object]]:
    relative_depth = target_layer / max(num_layers, 1)
    source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
    mapped = _map_directions(
        source_layer.directions,
        canonical_dim=canonical_dim,
        seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:gauge_records"),
    )
    mapped = _normalize_rows(mapped)
    if rep.model_id != anchor_rep.model_id:
        rotation = activation_gauge_rotation(
            anchor_rep,
            rep,
            target_layer=target_layer,
            canonical_dim=canonical_dim,
            num_layers=num_layers,
        )
        mapped = _normalize_rows(mapped @ rotation)
    if source_layer.scales is None or source_layer.scales.size == 0:
        scales = np.ones((mapped.shape[0],), dtype=np.float32)
    else:
        scales = np.abs(np.asarray(source_layer.scales, dtype=np.float32).reshape(-1)[: mapped.shape[0]])
    return [
        {
            "model_id": rep.model_id,
            "source_layer_idx": int(source_layer_idx),
            "relative_depth": relative_depth,
            "direction_idx": int(direction_idx),
            "scale": float(scales[direction_idx]),
            "vector": mapped[direction_idx],
            "source_hidden_dim": int(rep.hidden_dim),
        }
        for direction_idx in range(mapped.shape[0])
    ]


def merge_gauge_fixed_spectral_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
    incremental: bool,
) -> PlatonicGeometry:
    if not representations:
        raise ValueError("representations must not be empty")
    anchor_rep = representations[0]
    layer_geometries: dict[int, object] = {}
    for target_layer in range(1, num_layers + 1):
        if incremental:
            current_records = _aligned_records_for_rep(
                anchor_rep,
                anchor_rep=anchor_rep,
                target_layer=target_layer,
                canonical_dim=canonical_dim,
                num_layers=num_layers,
            )
            current_clusters = _cluster_direction_records(current_records, similarity_threshold=similarity_threshold)
            for rep in representations[1:]:
                new_records = _aligned_records_for_rep(
                    rep,
                    anchor_rep=anchor_rep,
                    target_layer=target_layer,
                    canonical_dim=canonical_dim,
                    num_layers=num_layers,
                )
                champion_records = [dict(cluster["champion"]) for cluster in current_clusters]
                current_clusters = _cluster_direction_records(champion_records + new_records, similarity_threshold=similarity_threshold)
            layer_geometries[target_layer] = _layer_geometry_from_clusters(
                current_clusters,
                relative_depth=target_layer / max(num_layers, 1),
                canonical_dim=canonical_dim,
                top_k=top_k,
            )
        else:
            flat_records: list[dict[str, object]] = []
            for rep in representations:
                flat_records.extend(
                    _aligned_records_for_rep(
                        rep,
                        anchor_rep=anchor_rep,
                        target_layer=target_layer,
                        canonical_dim=canonical_dim,
                        num_layers=num_layers,
                    )
                )
            clusters = _cluster_direction_records(flat_records, similarity_threshold=similarity_threshold)
            layer_geometries[target_layer] = _layer_geometry_from_clusters(
                clusters,
                relative_depth=target_layer / max(num_layers, 1),
                canonical_dim=canonical_dim,
                top_k=top_k,
            )
    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=[rep.model_id for rep in representations],
        metadata={
            "selection_method": "gauge_fixed_spectral_argmax_incremental" if incremental else "gauge_fixed_spectral_argmax",
            "anchor_model": anchor_rep.model_id,
            "incremental": bool(incremental),
            "similarity_threshold": float(similarity_threshold),
            "top_k": int(top_k),
            "num_layers": int(num_layers),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge spectral ModelRepresentation artifacts after activation-anchored gauge fixing in the shared canonical frame.")
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Canonical dimension used for cross-model comparison")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins in the merged geometry")
    parser.add_argument("--top-k", type=int, default=16, help="Directions to retain per merged layer")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="Absolute cosine threshold for clustering source directions")
    parser.add_argument("--incremental", action="store_true", help="Add models sequentially against the current merged champions instead of clustering all source records at once")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_gauge_fixed_spectral_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        incremental=args.incremental,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "anchor_model": geometry.metadata["anchor_model"],
                "canonical_dim": args.canonical_dim,
                "incremental": bool(args.incremental),
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
