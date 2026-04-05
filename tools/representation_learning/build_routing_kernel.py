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
    from tools.representation_learning.compare_model_representations import _normalize_rows
    from tools.representation_learning.schemas import ModelRepresentation, RoutingKernel, VerificationTable
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from compare_model_representations import _normalize_rows  # type: ignore[no-redef]
    from schemas import ModelRepresentation, RoutingKernel, VerificationTable  # type: ignore[no-redef]


def _chunk_index_lookup(rep: ModelRepresentation) -> dict[str, int]:
    if not rep.chunk_ids:
        return {}
    return {str(chunk_id): idx for idx, chunk_id in enumerate(rep.chunk_ids)}


def _canonical_chunk_embeddings(
    rep: ModelRepresentation,
    *,
    canonical_dim: int,
    num_layers: int,
) -> dict[int, dict[str, object]]:
    embeddings: dict[int, dict[str, object]] = {}
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
        projections = rep.chunk_layer_projections.get(int(source_layer_idx))
        if projections is None:
            continue
        mapped = _map_directions(
            source_layer.directions,
            canonical_dim=canonical_dim,
            seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:routing"),
        )
        mapped = _normalize_rows(mapped)
        projection_array = np.asarray(projections, dtype=np.float32)
        if projection_array.ndim != 2 or projection_array.shape[1] != mapped.shape[0]:
            continue
        embeddings[target_layer] = {
            "source_layer_idx": int(source_layer_idx),
            "matrix": projection_array @ mapped,
        }
    return embeddings


def _append_cluster(
    clusters: list[dict[str, object]],
    *,
    record: dict[str, object],
) -> None:
    vector = np.asarray(record["vector"], dtype=np.float32)
    weight = float(record["verification_confidence"])
    clusters.append(
        {
            "centroid": vector.copy(),
            "weight_sum": weight,
            "support": 1,
            "chunk_ids": [str(record["chunk_id"])],
            "probe_ids": [str(record["probe_id"])],
            "source_layer_idx_counts": {int(record["source_layer_idx"]): 1},
            "score_sum": weight,
        }
    )


def _update_cluster(cluster: dict[str, object], record: dict[str, object]) -> None:
    vector = np.asarray(record["vector"], dtype=np.float32)
    weight = float(record["verification_confidence"])
    centroid = np.asarray(cluster["centroid"], dtype=np.float32)
    total = float(cluster["weight_sum"]) + weight
    updated = (centroid * float(cluster["weight_sum"]) + vector * weight) / max(total, 1e-6)
    norm = float(np.linalg.norm(updated))
    cluster["centroid"] = updated / max(norm, 1e-8)
    cluster["weight_sum"] = total
    cluster["support"] = int(cluster["support"]) + 1
    cluster["chunk_ids"] = list(cluster["chunk_ids"]) + [str(record["chunk_id"])]
    cluster["probe_ids"] = list(cluster["probe_ids"]) + [str(record["probe_id"])]
    counts = dict(cluster["source_layer_idx_counts"])
    layer_idx = int(record["source_layer_idx"])
    counts[layer_idx] = counts.get(layer_idx, 0) + 1
    cluster["source_layer_idx_counts"] = counts
    cluster["score_sum"] = float(cluster["score_sum"]) + weight


def _cluster_records(
    records: list[dict[str, object]],
    *,
    similarity_threshold: float,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for record in sorted(records, key=lambda entry: float(entry["verification_confidence"]), reverse=True):
        vector = np.asarray(record["vector"], dtype=np.float32)
        best_idx = None
        best_similarity = -1.0
        for idx, cluster in enumerate(clusters):
            centroid = np.asarray(cluster["centroid"], dtype=np.float32)
            similarity = float(np.dot(vector, centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        if best_idx is not None and best_similarity >= float(similarity_threshold):
            _update_cluster(clusters[best_idx], record)
        else:
            _append_cluster(clusters, record=record)
    return clusters


def build_routing_kernel(
    representations: list[ModelRepresentation],
    verification_table: VerificationTable,
    *,
    canonical_dim: int,
    num_layers: int,
    similarity_threshold: float,
    min_support: int,
) -> RoutingKernel:
    rep_by_id = {rep.model_id: rep for rep in representations}
    chunk_indices = {rep.model_id: _chunk_index_lookup(rep) for rep in representations}
    chunk_embeddings = {
        rep.model_id: _canonical_chunk_embeddings(rep, canonical_dim=canonical_dim, num_layers=num_layers)
        for rep in representations
    }
    grouped_records: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    fallback_records: dict[tuple[str, str], list[dict[str, object]]] = {}
    for entry in verification_table.entries:
        if not bool(entry.get("verified", True)):
            continue
        winner_model = str(entry["verified_winner_model"])
        chunk_id = str(entry["chunk_id"])
        rep = rep_by_id.get(winner_model)
        if rep is None:
            continue
        chunk_idx = chunk_indices[winner_model].get(chunk_id)
        if chunk_idx is None:
            continue
        embedded = False
        for target_layer in range(1, num_layers + 1):
            layer_payload = chunk_embeddings[winner_model].get(target_layer)
            if layer_payload is None:
                continue
            matrix = np.asarray(layer_payload["matrix"], dtype=np.float32)
            if chunk_idx >= matrix.shape[0]:
                continue
            vector = np.asarray(matrix[chunk_idx], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            grouped_records.setdefault(
                (str(entry["probe_type"]), winner_model, target_layer),
                [],
            ).append(
                {
                    "probe_id": str(entry["probe_id"]),
                    "chunk_id": chunk_id,
                    "probe_type": str(entry["probe_type"]),
                    "winner_model": winner_model,
                    "target_layer": int(target_layer),
                    "source_layer_idx": int(layer_payload["source_layer_idx"]),
                    "verification_confidence": float(max(float(entry.get("verification_confidence", 0.0)), 1e-6)),
                    "vector": vector / norm,
                }
            )
            embedded = True
        if not embedded:
            fallback_records.setdefault((str(entry["probe_type"]), winner_model), []).append(
                {
                    "probe_id": str(entry["probe_id"]),
                    "chunk_id": chunk_id,
                    "probe_type": str(entry["probe_type"]),
                    "winner_model": winner_model,
                    "verification_confidence": float(max(float(entry.get("verification_confidence", 0.0)), 1e-6)),
                }
            )

    rules: list[dict[str, object]] = []
    for (probe_type, winner_model, target_layer), records in grouped_records.items():
        clusters = _cluster_records(records, similarity_threshold=similarity_threshold)
        for cluster_idx, cluster in enumerate(clusters):
            support = int(cluster["support"])
            if support < int(min_support):
                continue
            layer_counts = dict(cluster["source_layer_idx_counts"])
            source_layer_idx = max(layer_counts.items(), key=lambda item: (item[1], -item[0]))[0]
            chunk_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in cluster["chunk_ids"]))
            rules.append(
                {
                    "rule_id": f"{probe_type}:{winner_model}:L{target_layer}:C{cluster_idx}",
                    "routing_mode": "centroid_cluster",
                    "probe_type": probe_type,
                    "winner_model": winner_model,
                    "target_layer": int(target_layer),
                    "source_layer_idx": int(source_layer_idx),
                    "support": support,
                    "mean_verification_confidence": float(cluster["score_sum"]) / max(support, 1),
                    "cluster_score": float(cluster["score_sum"]),
                    "covered_chunk_ids": chunk_ids[: min(16, len(chunk_ids))],
                    "probe_ids": list(dict.fromkeys(str(probe_id) for probe_id in cluster["probe_ids"]))[:16],
                    "centroid": np.asarray(cluster["centroid"], dtype=np.float32).tolist(),
                }
            )
    for (probe_type, winner_model), records in fallback_records.items():
        if len(records) < int(min_support):
            continue
        unique_chunk_ids = list(dict.fromkeys(str(record["chunk_id"]) for record in records))
        unique_probe_ids = list(dict.fromkeys(str(record["probe_id"]) for record in records))
        mean_confidence = float(np.mean([float(record["verification_confidence"]) for record in records])) if records else 0.0
        rules.append(
            {
                "rule_id": f"{probe_type}:{winner_model}:lookup",
                "routing_mode": "chunk_lookup_fallback",
                "probe_type": probe_type,
                "winner_model": winner_model,
                "target_layer": None,
                "source_layer_idx": None,
                "support": len(records),
                "mean_verification_confidence": mean_confidence,
                "cluster_score": float(sum(float(record["verification_confidence"]) for record in records)),
                "covered_chunk_ids": unique_chunk_ids[: min(32, len(unique_chunk_ids))],
                "probe_ids": unique_probe_ids[: min(32, len(unique_probe_ids))],
                "centroid": None,
            }
        )
    rules.sort(
        key=lambda rule: (
            float(rule["cluster_score"]),
            int(rule["support"]),
            str(rule["winner_model"]),
            -1 if rule["target_layer"] is None else int(rule["target_layer"]),
        ),
        reverse=True,
    )
    return RoutingKernel(
        source_models=[rep.model_id for rep in representations],
        rules=rules,
        metadata={
            "kernel_build_method": "verified_winner_chunk_projection_clusters",
            "canonical_dim": int(canonical_dim),
            "num_layers": int(num_layers),
            "similarity_threshold": float(similarity_threshold),
            "min_support": int(min_support),
            "rule_count": len(rules),
            "verified_entry_count": sum(1 for entry in verification_table.entries if bool(entry.get("verified", True))),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first routing kernel from verified disagreement probes.")
    parser.add_argument("output", help="Output .npz path for the routing kernel")
    parser.add_argument("verification_table", help="Input verification table artifact")
    parser.add_argument("representations", nargs="+", help="Input model representation artifacts")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()

    verification_table = VerificationTable.load(args.verification_table)
    reps = [ModelRepresentation.load(path) for path in args.representations]
    kernel = build_routing_kernel(
        reps,
        verification_table,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        similarity_threshold=args.similarity_threshold,
        min_support=args.min_support,
    )
    kernel.save(args.output)
    summary = {
        "output": str(Path(args.output).resolve()),
        "rule_count": len(kernel.rules),
        "source_models": kernel.source_models,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
