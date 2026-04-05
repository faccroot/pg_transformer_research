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
    from tools.representation_learning.compare_model_representations import (
        _collect_named_anchor_records,
        _load_calibration_lookup,
        _map_directions,
        _nearest_layer,
        _normalize_rows,
        _stable_seed,
    )
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from compare_model_representations import (  # type: ignore[no-redef]
        _collect_named_anchor_records,
        _load_calibration_lookup,
        _map_directions,
        _nearest_layer,
        _normalize_rows,
        _stable_seed,
    )
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def _chunk_loss_lookup(rep: ModelRepresentation) -> dict[str, float]:
    if rep.chunk_losses is None or rep.chunk_ids is None:
        return {}
    losses = np.asarray(rep.chunk_losses, dtype=np.float32).reshape(-1)
    count = min(len(rep.chunk_ids), int(losses.shape[0]))
    return {str(rep.chunk_ids[idx]): float(losses[idx]) for idx in range(count)}


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
    chunk_indices = _chunk_index_lookup(rep)
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
            seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}"),
        )
        mapped = _normalize_rows(mapped)
        projection_array = np.asarray(projections, dtype=np.float32)
        if projection_array.ndim != 2 or projection_array.shape[1] != mapped.shape[0]:
            continue
        count = min(len(chunk_indices), int(projection_array.shape[0]))
        embeddings[target_layer] = {
            "source_layer_idx": int(source_layer_idx),
            "matrix": projection_array[:count] @ mapped,
        }
    return embeddings


def _make_difference_records(
    winner: ModelRepresentation,
    loser: ModelRepresentation,
    *,
    canonical_dim: int,
    num_layers: int,
    min_advantage: float,
    calibration_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    winner_losses = _chunk_loss_lookup(winner)
    loser_losses = _chunk_loss_lookup(loser)
    shared_ids = sorted(set(winner_losses).intersection(loser_losses))
    if not shared_ids:
        return []

    winner_indices = _chunk_index_lookup(winner)
    loser_indices = _chunk_index_lookup(loser)
    winner_embeddings = _canonical_chunk_embeddings(winner, canonical_dim=canonical_dim, num_layers=num_layers)
    loser_embeddings = _canonical_chunk_embeddings(loser, canonical_dim=canonical_dim, num_layers=num_layers)

    records: list[dict[str, object]] = []
    for chunk_id in shared_ids:
        advantage = float(loser_losses[chunk_id] - winner_losses[chunk_id])
        if advantage < min_advantage:
            continue
        winner_idx = winner_indices.get(chunk_id)
        loser_idx = loser_indices.get(chunk_id)
        if winner_idx is None or loser_idx is None:
            continue
        record = calibration_lookup.get(chunk_id, {})
        metadata = {key: value for key, value in record.items() if key != "text"}
        text_preview = " ".join(str(record.get("text", "")).split())[:240] if record else None
        for target_layer in range(1, num_layers + 1):
            winner_layer = winner_embeddings.get(target_layer)
            loser_layer = loser_embeddings.get(target_layer)
            if winner_layer is None or loser_layer is None:
                continue
            winner_matrix = np.asarray(winner_layer["matrix"], dtype=np.float32)
            loser_matrix = np.asarray(loser_layer["matrix"], dtype=np.float32)
            if winner_idx >= winner_matrix.shape[0] or loser_idx >= loser_matrix.shape[0]:
                continue
            diff = winner_matrix[winner_idx] - loser_matrix[loser_idx]
            norm = float(np.linalg.norm(diff))
            if norm <= 1e-8:
                continue
            records.append(
                {
                    "winner_model": winner.model_id,
                    "loser_model": loser.model_id,
                    "chunk_id": chunk_id,
                    "target_layer": target_layer,
                    "advantage": advantage,
                    "vector": (diff / norm).astype(np.float32),
                    "norm": norm,
                    "winner_loss": float(winner_losses[chunk_id]),
                    "loser_loss": float(loser_losses[chunk_id]),
                    "metadata": metadata,
                    "text_preview": text_preview,
                }
            )
    records.sort(key=lambda entry: (float(entry["advantage"]), float(entry["norm"]), str(entry["chunk_id"])), reverse=True)
    return records


def _top_named_anchors(
    centroid: np.ndarray,
    *,
    anchor_vectors: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    entries = []
    for anchor in anchor_vectors:
        alignment = abs(float(np.dot(centroid, np.asarray(anchor["vector"], dtype=np.float32))))
        entries.append(
            {
                "concept": str(anchor["concept"]),
                "model_id": str(anchor["model_id"]),
                "target_layer": int(anchor["target_layer"]),
                "source_layer_idx": int(anchor["source_layer_idx"]),
                "alignment": alignment,
            }
        )
    entries.sort(key=lambda entry: (float(entry["alignment"]), entry["concept"], entry["model_id"]), reverse=True)
    return entries[: min(limit, len(entries))]


def _build_chunk_anchor_profiles(
    anchor_vectors: list[dict[str, object]],
    *,
    canonical_dim: int,
    num_layers: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    total_dim = canonical_dim * num_layers
    for anchor in anchor_vectors:
        concept = str(anchor["concept"])
        model_id = str(anchor["model_id"])
        key = (concept, model_id)
        entry = grouped.setdefault(
            key,
            {
                "concept": concept,
                "model_id": model_id,
                "vector": np.zeros((total_dim,), dtype=np.float32),
                "target_layers": set(),
            },
        )
        target_layer = int(anchor["target_layer"])
        if target_layer < 1 or target_layer > num_layers:
            continue
        offset = (target_layer - 1) * canonical_dim
        entry["vector"][offset: offset + canonical_dim] = np.asarray(anchor["vector"], dtype=np.float32)
        entry["target_layers"].add(target_layer)
    profiles: list[dict[str, object]] = []
    for entry in grouped.values():
        vector = np.asarray(entry["vector"], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        profiles.append(
            {
                "concept": str(entry["concept"]),
                "model_id": str(entry["model_id"]),
                "vector": vector / norm,
                "target_layers": sorted(int(layer) for layer in entry["target_layers"]),
            }
        )
    return profiles


def _top_chunk_named_anchors(
    centroid: np.ndarray,
    *,
    anchor_profiles: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    entries = []
    for anchor in anchor_profiles:
        alignment = abs(float(np.dot(centroid, np.asarray(anchor["vector"], dtype=np.float32))))
        entries.append(
            {
                "concept": str(anchor["concept"]),
                "model_id": str(anchor["model_id"]),
                "target_layers": list(anchor["target_layers"]),
                "alignment": alignment,
            }
        )
    entries.sort(key=lambda entry: (float(entry["alignment"]), entry["concept"], entry["model_id"]), reverse=True)
    return entries[: min(limit, len(entries))]


def _chunk_overlap_score(chunk_ids_a: set[str], chunk_ids_b: set[str]) -> float:
    if not chunk_ids_a or not chunk_ids_b:
        return 0.0
    overlap = len(chunk_ids_a.intersection(chunk_ids_b))
    return float(overlap / max(min(len(chunk_ids_a), len(chunk_ids_b)), 1))


def _summarize_record_group(
    records: list[dict[str, object]],
    *,
    group_idx_key: str,
    group_idx_value: int,
    anchor_vectors: list[dict[str, object]],
    top_chunks: int,
    top_named_anchors: int,
    total_winning_chunks: int,
    shared_chunk_count: int,
    member_cluster_ids: list[int] | None = None,
) -> dict[str, object]:
    weighted_sum = None
    for record in records:
        vector = np.asarray(record["vector"], dtype=np.float32)
        weight = float(record["advantage"])
        if weighted_sum is None:
            weighted_sum = np.zeros_like(vector, dtype=np.float32)
        weighted_sum += vector * weight
    centroid = np.zeros((0,), dtype=np.float32)
    if weighted_sum is not None:
        centroid_norm = float(np.linalg.norm(weighted_sum))
        if centroid_norm > 1e-8:
            centroid = weighted_sum / centroid_norm

    total_advantage = float(sum(float(record["advantage"]) for record in records))
    mean_advantage = total_advantage / max(len(records), 1)
    layer_counts: dict[int, int] = {}
    chunk_best: dict[str, dict[str, object]] = {}
    for record in records:
        layer = int(record["target_layer"])
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        current = chunk_best.get(str(record["chunk_id"]))
        if current is None or float(record["advantage"]) > float(current["advantage"]):
            chunk_best[str(record["chunk_id"])] = record
    dominant_layers = [
        {"target_layer": layer, "count": count}
        for layer, count in sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    top_chunks_payload = []
    for record in sorted(chunk_best.values(), key=lambda entry: (float(entry["advantage"]), str(entry["chunk_id"])), reverse=True)[:top_chunks]:
        top_chunks_payload.append(
            {
                "chunk_id": str(record["chunk_id"]),
                "advantage": float(record["advantage"]),
                "target_layer": int(record["target_layer"]),
                "winner_loss": float(record["winner_loss"]),
                "loser_loss": float(record["loser_loss"]),
                "text_preview": record["text_preview"],
                "metadata": dict(record["metadata"]),
            }
        )
    unique_chunk_advantage = float(sum(float(record["advantage"]) for record in chunk_best.values()))
    return {
        group_idx_key: group_idx_value,
        "member_cluster_ids": sorted(member_cluster_ids or [group_idx_value]),
        "member_cluster_count": len(member_cluster_ids or [group_idx_value]),
        "count": len(records),
        "unique_chunk_count": len(chunk_best),
        "total_advantage": total_advantage,
        "unique_chunk_advantage": unique_chunk_advantage,
        "mean_advantage": mean_advantage,
        "mean_advantage_unique": unique_chunk_advantage / max(len(chunk_best), 1),
        "coverage_of_winner_chunks": float(len(chunk_best) / max(total_winning_chunks, 1)),
        "coverage_of_shared_chunks": float(len(chunk_best) / max(shared_chunk_count, 1)),
        "dominant_layers": dominant_layers,
        "top_named_anchors": _top_named_anchors(centroid, anchor_vectors=anchor_vectors, limit=top_named_anchors)
        if centroid.size > 0 else [],
        "top_chunks": top_chunks_payload,
        "chunk_ids": sorted(chunk_best),
    }


def _build_chunk_family_reports(
    records: list[dict[str, object]],
    *,
    canonical_dim: int,
    num_layers: int,
    chunk_similarity_threshold: float,
    top_chunks: int,
    top_named_anchors: int,
    shared_chunk_count: int,
    anchor_profiles: list[dict[str, object]],
) -> list[dict[str, object]]:
    total_dim = canonical_dim * num_layers
    chunk_map: dict[str, dict[str, object]] = {}
    for record in records:
        chunk_id = str(record["chunk_id"])
        entry = chunk_map.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "advantage": float(record["advantage"]),
                "winner_loss": float(record["winner_loss"]),
                "loser_loss": float(record["loser_loss"]),
                "metadata": dict(record["metadata"]),
                "text_preview": record["text_preview"],
                "layer_vectors": {},
                "layer_energy": {},
            },
        )
        layer = int(record["target_layer"])
        vector = np.asarray(record["vector"], dtype=np.float32) * float(record["norm"])
        entry["layer_vectors"][layer] = vector
        entry["layer_energy"][layer] = float(np.linalg.norm(vector))

    chunk_items: list[dict[str, object]] = []
    for entry in chunk_map.values():
        profile = np.zeros((total_dim,), dtype=np.float32)
        for layer, vector in dict(entry["layer_vectors"]).items():
            if layer < 1 or layer > num_layers:
                continue
            offset = (layer - 1) * canonical_dim
            profile[offset: offset + canonical_dim] = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(profile))
        if norm <= 1e-8:
            continue
        chunk_items.append(
            {
                **entry,
                "profile": profile / norm,
            }
        )

    raw_families: list[dict[str, object]] = []
    for item in sorted(chunk_items, key=lambda entry: (float(entry["advantage"]), str(entry["chunk_id"])), reverse=True):
        vector = np.asarray(item["profile"], dtype=np.float32)
        best_idx = None
        best_sim = -1.0
        for idx, family in enumerate(raw_families):
            centroid = np.asarray(family["centroid"], dtype=np.float32)
            similarity = float(np.dot(vector, centroid))
            if similarity > best_sim:
                best_sim = similarity
                best_idx = idx
        if best_idx is not None and best_sim >= chunk_similarity_threshold:
            family = raw_families[best_idx]
            family["weighted_sum"] += vector * float(item["advantage"])
            family["weight"] += float(item["advantage"])
            family["chunks"].append(item)
            centroid = family["weighted_sum"] / max(float(family["weight"]), 1e-8)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 1e-8:
                family["centroid"] = centroid / centroid_norm
        else:
            raw_families.append(
                {
                    "weighted_sum": vector * float(item["advantage"]),
                    "weight": float(item["advantage"]),
                    "centroid": vector.copy(),
                    "chunks": [item],
                }
            )

    reports: list[dict[str, object]] = []
    total_winning_chunks = len(chunk_items)
    for family_idx, family in enumerate(raw_families):
        chunks = list(family["chunks"])
        centroid = np.asarray(family["centroid"], dtype=np.float32)
        layer_energy: dict[int, float] = {}
        for chunk in chunks:
            for layer, energy in dict(chunk["layer_energy"]).items():
                layer_energy[int(layer)] = layer_energy.get(int(layer), 0.0) + float(energy)
        dominant_layers = [
            {"target_layer": layer, "energy": energy}
            for layer, energy in sorted(layer_energy.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
        top_chunks_payload = []
        for chunk in sorted(chunks, key=lambda entry: (float(entry["advantage"]), str(entry["chunk_id"])), reverse=True)[:top_chunks]:
            top_chunks_payload.append(
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "advantage": float(chunk["advantage"]),
                    "winner_loss": float(chunk["winner_loss"]),
                    "loser_loss": float(chunk["loser_loss"]),
                    "text_preview": chunk["text_preview"],
                    "metadata": dict(chunk["metadata"]),
                }
            )
        total_advantage = float(sum(float(chunk["advantage"]) for chunk in chunks))
        reports.append(
            {
                "chunk_family_idx": family_idx,
                "chunk_count": len(chunks),
                "total_advantage": total_advantage,
                "mean_advantage": total_advantage / max(len(chunks), 1),
                "coverage_of_winner_chunks": float(len(chunks) / max(total_winning_chunks, 1)),
                "coverage_of_shared_chunks": float(len(chunks) / max(shared_chunk_count, 1)),
                "dominant_layers": dominant_layers,
                "top_named_anchors": _top_chunk_named_anchors(centroid, anchor_profiles=anchor_profiles, limit=top_named_anchors),
                "top_chunks": top_chunks_payload,
                "chunk_ids": sorted(str(chunk["chunk_id"]) for chunk in chunks),
            }
        )
    reports.sort(
        key=lambda entry: (
            float(entry["total_advantage"]),
            float(entry["coverage_of_winner_chunks"]),
            int(entry["chunk_count"]),
        ),
        reverse=True,
    )
    return reports


def _build_raw_clusters(
    records: list[dict[str, object]],
    *,
    similarity_threshold: float,
    max_clusters: int,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for record in records:
        vector = np.asarray(record["vector"], dtype=np.float32)
        best_idx = None
        best_sim = -1.0
        for idx, cluster in enumerate(clusters):
            centroid = np.asarray(cluster["centroid"], dtype=np.float32)
            similarity = float(np.dot(vector, centroid))
            if similarity > best_sim:
                best_sim = similarity
                best_idx = idx
        if best_idx is not None and best_sim >= similarity_threshold:
            cluster = clusters[best_idx]
            cluster["weighted_sum"] += vector * float(record["advantage"])
            cluster["weight"] += float(record["advantage"])
            cluster["records"].append(record)
            centroid = cluster["weighted_sum"] / max(float(cluster["weight"]), 1e-8)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 1e-8:
                cluster["centroid"] = centroid / centroid_norm
        elif len(clusters) < max_clusters:
            clusters.append(
                {
                    "weighted_sum": vector * float(record["advantage"]),
                    "weight": float(record["advantage"]),
                    "centroid": vector.copy(),
                    "records": [record],
                }
            )
        else:
            cluster = min(clusters, key=lambda entry: float(entry["weight"]))
            cluster["weighted_sum"] += vector * float(record["advantage"])
            cluster["weight"] += float(record["advantage"])
            cluster["records"].append(record)
            centroid = cluster["weighted_sum"] / max(float(cluster["weight"]), 1e-8)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 1e-8:
                cluster["centroid"] = centroid / centroid_norm
    for idx, cluster in enumerate(clusters):
        cluster["cluster_id"] = idx
    return clusters


def _collapse_cluster_families(
    raw_clusters: list[dict[str, object]],
    *,
    overlap_threshold: float,
) -> list[dict[str, object]]:
    families: list[dict[str, object]] = []
    ordered = sorted(
        raw_clusters,
        key=lambda cluster: (
            sum(float(record["advantage"]) for record in cluster["records"]),
            len({str(record["chunk_id"]) for record in cluster["records"]}),
            -int(cluster.get("cluster_id", 0)),
        ),
        reverse=True,
    )
    for cluster in ordered:
        chunk_ids = {str(record["chunk_id"]) for record in cluster["records"]}
        best_family = None
        best_score = -1.0
        for family in families:
            score = _chunk_overlap_score(chunk_ids, family["chunk_ids"])
            if score > best_score:
                best_score = score
                best_family = family
        if best_family is not None and best_score >= overlap_threshold:
            best_family["records"].extend(cluster["records"])
            best_family["chunk_ids"].update(chunk_ids)
            best_family["member_cluster_ids"].append(int(cluster["cluster_id"]))
        else:
            families.append(
                {
                    "records": list(cluster["records"]),
                    "chunk_ids": set(chunk_ids),
                    "member_cluster_ids": [int(cluster["cluster_id"])],
                }
            )
    return families


def _cluster_difference_records(
    records: list[dict[str, object]],
    *,
    anchor_vectors: list[dict[str, object]],
    similarity_threshold: float,
    top_chunks: int,
    top_named_anchors: int,
    max_clusters: int,
    family_overlap_threshold: float,
    shared_chunk_count: int,
) -> dict[str, object]:
    total_winning_chunks = len({str(record["chunk_id"]) for record in records})
    raw_clusters = _build_raw_clusters(
        records,
        similarity_threshold=similarity_threshold,
        max_clusters=max_clusters,
    )
    cluster_reports = [
        _summarize_record_group(
            list(cluster["records"]),
            group_idx_key="cluster_idx",
            group_idx_value=int(cluster["cluster_id"]),
            anchor_vectors=anchor_vectors,
            top_chunks=top_chunks,
            top_named_anchors=top_named_anchors,
            total_winning_chunks=total_winning_chunks,
            shared_chunk_count=shared_chunk_count,
        )
        for cluster in raw_clusters
    ]
    cluster_reports.sort(
        key=lambda entry: (
            float(entry["unique_chunk_advantage"]),
            float(entry["total_advantage"]),
            int(entry["unique_chunk_count"]),
        ),
        reverse=True,
    )

    family_reports = [
        _summarize_record_group(
            list(family["records"]),
            group_idx_key="family_idx",
            group_idx_value=family_idx,
            anchor_vectors=anchor_vectors,
            top_chunks=top_chunks,
            top_named_anchors=top_named_anchors,
            total_winning_chunks=total_winning_chunks,
            shared_chunk_count=shared_chunk_count,
            member_cluster_ids=list(family["member_cluster_ids"]),
        )
        for family_idx, family in enumerate(
            _collapse_cluster_families(raw_clusters, overlap_threshold=family_overlap_threshold)
        )
    ]
    family_reports.sort(
        key=lambda entry: (
            float(entry["unique_chunk_advantage"]),
            float(entry["coverage_of_winner_chunks"]),
            int(entry["member_cluster_count"]),
        ),
        reverse=True,
    )
    return {
        "difference_record_count": len(records),
        "winning_chunk_count": total_winning_chunks,
        "cluster_count": len(cluster_reports),
        "family_count": len(family_reports),
        "clusters": cluster_reports,
        "families": family_reports,
    }


def build_advantage_cluster_report(
    representations: list[ModelRepresentation],
    *,
    calibration_lookup: dict[str, dict[str, object]],
    canonical_dim: int,
    num_layers: int,
    similarity_threshold: float,
    family_overlap_threshold: float,
    chunk_family_similarity_threshold: float,
    min_advantage: float,
    top_chunks: int,
    top_named_anchors: int,
    max_clusters: int,
) -> dict[str, object]:
    anchor_vectors = _collect_named_anchor_records(representations, canonical_dim=canonical_dim, num_layers=num_layers)
    anchor_vectors = [
        {**record, "vector": np.asarray(record["vector"], dtype=np.float32)}
        for record in anchor_vectors
    ]
    chunk_anchor_profiles = _build_chunk_anchor_profiles(anchor_vectors, canonical_dim=canonical_dim, num_layers=num_layers)
    pair_reports: list[dict[str, object]] = []
    for left_idx, left_rep in enumerate(representations):
        for right_rep in representations[left_idx + 1:]:
            sides = []
            shared_chunk_count = len(set(_chunk_loss_lookup(left_rep)).intersection(_chunk_loss_lookup(right_rep)))
            for winner, loser in ((left_rep, right_rep), (right_rep, left_rep)):
                side_report = _cluster_difference_records(
                    _make_difference_records(
                        winner,
                        loser,
                        canonical_dim=canonical_dim,
                        num_layers=num_layers,
                        min_advantage=min_advantage,
                        calibration_lookup=calibration_lookup,
                    ),
                    anchor_vectors=anchor_vectors,
                    similarity_threshold=similarity_threshold,
                    top_chunks=top_chunks,
                    top_named_anchors=top_named_anchors,
                    max_clusters=max_clusters,
                    family_overlap_threshold=family_overlap_threshold,
                    shared_chunk_count=shared_chunk_count,
                )
                chunk_families = _build_chunk_family_reports(
                    _make_difference_records(
                        winner,
                        loser,
                        canonical_dim=canonical_dim,
                        num_layers=num_layers,
                        min_advantage=min_advantage,
                        calibration_lookup=calibration_lookup,
                    ),
                    canonical_dim=canonical_dim,
                    num_layers=num_layers,
                    chunk_similarity_threshold=chunk_family_similarity_threshold,
                    top_chunks=top_chunks,
                    top_named_anchors=top_named_anchors,
                    shared_chunk_count=shared_chunk_count,
                    anchor_profiles=chunk_anchor_profiles,
                )
                sides.append(
                    {
                        "winner_model": winner.model_id,
                        "loser_model": loser.model_id,
                        **side_report,
                        "chunk_family_count": len(chunk_families),
                        "chunk_families": chunk_families,
                    }
                )
            pair_reports.append(
                {
                    "model_a": left_rep.model_id,
                    "model_b": right_rep.model_id,
                    "shared_chunk_count": shared_chunk_count,
                    "sides": sides,
                }
            )
    return {
        "canonical_dim": canonical_dim,
        "num_layers": num_layers,
        "similarity_threshold": similarity_threshold,
        "family_overlap_threshold": family_overlap_threshold,
        "chunk_family_similarity_threshold": chunk_family_similarity_threshold,
        "min_advantage": min_advantage,
        "representations": [rep.model_id for rep in representations],
        "pairwise_advantage_clusters": pair_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster per-chunk winner-vs-loser difference directions into advantage-conditioned factors."
    )
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--calibration-jsonl", default="", help="Local calibration JSONL for chunk text and metadata")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Shared canonical dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Relative-depth bins used for chunk-difference reconstruction")
    parser.add_argument("--similarity-threshold", type=float, default=0.88, help="Cosine threshold for greedy cluster assignment")
    parser.add_argument("--family-overlap-threshold", type=float, default=0.8, help="Chunk-overlap threshold for collapsing layer-duplicated clusters into factor families")
    parser.add_argument("--chunk-family-similarity-threshold", type=float, default=0.8, help="Cosine threshold for clustering unique winning chunks into chunk families")
    parser.add_argument("--min-advantage", type=float, default=0.05, help="Minimum per-chunk loss advantage required to include a difference vector")
    parser.add_argument("--top-chunks", type=int, default=8, help="Top chunks to keep per cluster")
    parser.add_argument("--top-named-anchors", type=int, default=5, help="Named anchors to retain per cluster")
    parser.add_argument("--max-clusters", type=int, default=256, help="Hard cap on cluster count per winner/loser side")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    report = build_advantage_cluster_report(
        representations,
        calibration_lookup=_load_calibration_lookup(args.calibration_jsonl or None),
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        similarity_threshold=args.similarity_threshold,
        family_overlap_threshold=args.family_overlap_threshold,
        chunk_family_similarity_threshold=args.chunk_family_similarity_threshold,
        min_advantage=args.min_advantage,
        top_chunks=args.top_chunks,
        top_named_anchors=args.top_named_anchors,
        max_clusters=args.max_clusters,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "canonical_dim": args.canonical_dim,
                "num_layers": args.num_layers,
                "model_ids": [rep.model_id for rep in representations],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
