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
    from tools.representation_learning.build_platonic_geometry import (
        _concept_layers,
        _concept_sharpness,
        _map_directions,
        _nearest_concept_layer,
        _nearest_layer,
        _stable_seed,
    )
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import (  # type: ignore[no-redef]
        _concept_layers,
        _concept_sharpness,
        _map_directions,
        _nearest_concept_layer,
        _nearest_layer,
        _stable_seed,
    )
    from schemas import LayerGeometry, ModelRepresentation  # type: ignore[no-redef]


def _subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"subspace inputs must be 2D, got {a.shape} and {b.shape}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    prod = a @ b.T
    denom = max(min(int(a.shape[0]), int(b.shape[0])), 1)
    return float(np.linalg.norm(prod, ord="fro") ** 2 / denom)


def _scale_energy(layer: LayerGeometry) -> float:
    if layer.scales is None:
        return float(layer.directions.shape[0])
    return float(np.sum(np.abs(np.nan_to_num(np.asarray(layer.scales, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))))


def _scale_mean_abs(layer: LayerGeometry) -> float:
    if layer.scales is None or layer.scales.size == 0:
        return 1.0 if layer.directions.shape[0] > 0 else 0.0
    return float(np.mean(np.abs(np.nan_to_num(np.asarray(layer.scales, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))))


def _covariance_trace(layer: LayerGeometry) -> float | None:
    if layer.covariance is None:
        return None
    return float(np.trace(np.nan_to_num(np.asarray(layer.covariance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)))


def _importance_mean(layer: LayerGeometry) -> float | None:
    if layer.importance is None or layer.importance.size == 0:
        return None
    return float(np.mean(np.nan_to_num(np.asarray(layer.importance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(scalar):
        return default
    return scalar


def _finite_summary(values: list[float]) -> tuple[float | None, float | None, float | None]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return None, None, None
    return float(np.mean(finite)), float(np.min(finite)), float(np.max(finite))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _layer_scale_array(layer: LayerGeometry, count: int) -> np.ndarray:
    if layer.scales is None or layer.scales.size == 0:
        return np.ones((count,), dtype=np.float32)
    scales = np.nan_to_num(np.asarray(layer.scales, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if scales.shape[0] < count:
        scales = np.pad(scales, (0, count - scales.shape[0]), constant_values=1.0)
    return np.abs(scales[:count])


def _collect_direction_records(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        for rep in representations:
            source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
            mapped = _map_directions(
                source_layer.directions,
                canonical_dim=canonical_dim,
                seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}"),
            )
            mapped = _normalize_rows(mapped)
            scales = _layer_scale_array(source_layer, mapped.shape[0])
            for direction_idx in range(mapped.shape[0]):
                records.append(
                    {
                        "model_id": rep.model_id,
                        "target_layer": target_layer,
                        "relative_depth": relative_depth,
                        "source_layer_idx": int(source_layer_idx),
                        "direction_idx": direction_idx,
                        "weight": float(scales[direction_idx]),
                        "vector": mapped[direction_idx],
                    }
                )
    group_means: dict[tuple[str, int], float] = {}
    for record in records:
        key = (str(record["model_id"]), int(record["target_layer"]))
        group_means.setdefault(key, 0.0)
        group_means[key] += float(record["weight"])
    group_counts: dict[tuple[str, int], int] = {}
    for record in records:
        key = (str(record["model_id"]), int(record["target_layer"]))
        group_counts[key] = group_counts.get(key, 0) + 1
    for record in records:
        key = (str(record["model_id"]), int(record["target_layer"]))
        mean_weight = group_means[key] / max(group_counts[key], 1)
        record["normalized_weight"] = float(max(float(record["weight"]) / max(mean_weight, 1e-8), 1e-8))
    return records


def _collect_named_anchor_records(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    concept_names = sorted({str(name) for rep in representations for name in rep.concept_profiles})
    for concept_name in concept_names:
        for target_layer in range(1, num_layers + 1):
            relative_depth = target_layer / max(num_layers, 1)
            for rep in representations:
                profile = rep.concept_profiles.get(concept_name)
                if not isinstance(profile, dict):
                    continue
                nearest = _nearest_concept_layer(profile, relative_depth)
                if nearest is None:
                    continue
                source_layer_idx, source_payload = nearest
                direction = np.asarray(source_payload.get("direction", []), dtype=np.float32).reshape(1, -1)
                if direction.size == 0:
                    continue
                mapped = _map_directions(
                    direction,
                    canonical_dim=canonical_dim,
                    seed=_stable_seed(f"{rep.model_id}:{concept_name}:{source_layer_idx}:{canonical_dim}"),
                )
                mapped = _normalize_rows(mapped)
                records.append(
                    {
                        "concept": concept_name,
                        "model_id": rep.model_id,
                        "target_layer": target_layer,
                        "source_layer_idx": int(source_layer_idx),
                        "vector": mapped[0],
                    }
                )
    return records


def _orthonormal_row_basis(matrix: np.ndarray, *, tol: float = 1e-6, max_rank: int | None = None) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, matrix.shape[-1] if matrix.ndim == 2 else 0), dtype=np.float32)
    _u, singular_values, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    keep = singular_values > tol
    if not np.any(keep):
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    basis = np.asarray(vh[keep], dtype=np.float32)
    if max_rank is not None:
        basis = basis[: max(int(max_rank), 0)]
    return basis


def _project_residual(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if basis.ndim != 2 or basis.shape[0] == 0:
        return vector
    coeffs = vector @ basis.T
    return vector - coeffs @ basis


def _build_factor_report(
    direction_records: list[dict[str, object]],
    *,
    anchor_vectors: list[dict[str, object]],
    num_factors: int,
    top_contributors: int,
    weight_mode: str,
    residual_basis: np.ndarray | None = None,
) -> list[dict[str, object]]:
    if not direction_records:
        return []

    processed_records: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    weights_list: list[float] = []
    for record in direction_records:
        direction = np.asarray(record["vector"], dtype=np.float32)
        if residual_basis is not None:
            direction = _project_residual(direction, residual_basis)
        residual_norm = float(np.linalg.norm(direction))
        if residual_norm <= 1e-8:
            continue
        normalized = direction / residual_norm
        base_weight = max(_safe_float(record.get("normalized_weight"), 1.0), 1e-8)
        if residual_basis is None:
            effective_weight = base_weight
            residual_energy_fraction = 1.0
        else:
            residual_energy_fraction = float(min(max(residual_norm ** 2, 0.0), 1.0))
            effective_weight = max(base_weight * residual_energy_fraction, 1e-8)
        vectors.append(normalized)
        weights_list.append(effective_weight)
        processed_records.append(
            {
                **record,
                "vector": normalized,
                "effective_weight": float(effective_weight),
                "residual_energy_fraction": float(residual_energy_fraction),
            }
        )

    if not processed_records:
        return []

    bank = _normalize_rows(np.stack(vectors, axis=0))
    weights = np.asarray(weights_list, dtype=np.float64)
    weighted_bank = bank * np.sqrt(weights)[:, None]
    covariance = weighted_bank.T @ weighted_bank
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1]

    factor_reports: list[dict[str, object]] = []
    max_factors = min(max(int(num_factors), 0), int(bank.shape[1]), len(order))
    for factor_rank in range(max_factors):
        eigenvalue = float(max(eigenvalues[order[factor_rank]], 0.0))
        vector = np.asarray(eigenvectors[:, order[factor_rank]], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        factor = vector / norm
        contributions = []
        model_scores: dict[str, list[float]] = {}
        layer_scores: dict[str, dict[int, list[float]]] = {}
        residual_energy_values: list[float] = []
        for record, weight in zip(processed_records, weights.tolist(), strict=False):
            direction = np.asarray(record["vector"], dtype=np.float32)
            score = float(weight * float(np.dot(factor, direction)) ** 2)
            model_id = str(record["model_id"])
            target_layer = int(record["target_layer"])
            model_scores.setdefault(model_id, []).append(score)
            layer_scores.setdefault(model_id, {}).setdefault(target_layer, []).append(score)
            residual_energy_values.append(float(record.get("residual_energy_fraction", 1.0)))
            contributions.append(
                {
                    "model_id": model_id,
                    "target_layer": target_layer,
                    "source_layer_idx": int(record["source_layer_idx"]),
                    "direction_idx": int(record["direction_idx"]),
                    "score": score,
                    "residual_energy_fraction": float(record.get("residual_energy_fraction", 1.0)),
                }
            )

        model_entries = []
        for model_id, scores in model_scores.items():
            layer_entry_map = layer_scores.get(model_id, {})
            model_entries.append(
                {
                    "model_id": model_id,
                    "score": float(np.mean(scores)) if scores else 0.0,
                    "best_layer": max(
                        (
                            (layer_idx, float(np.mean(layer_vals)))
                            for layer_idx, layer_vals in layer_entry_map.items()
                            if layer_vals
                        ),
                        key=lambda item: item[1],
                        default=(None, 0.0),
                    )[0],
                }
            )
        best_model = None
        if model_entries:
            best_model = max(model_entries, key=lambda entry: (float(entry["score"]), str(entry["model_id"])))["model_id"]

        anchor_entries = []
        for anchor in anchor_vectors:
            alignment = abs(float(np.dot(factor, anchor["vector"])))
            anchor_entries.append(
                {
                    "concept": str(anchor["concept"]),
                    "model_id": str(anchor["model_id"]),
                    "target_layer": int(anchor["target_layer"]),
                    "source_layer_idx": int(anchor["source_layer_idx"]),
                    "alignment": alignment,
                }
            )
        anchor_entries.sort(key=lambda entry: (float(entry["alignment"]), entry["concept"], entry["model_id"]), reverse=True)
        contributions.sort(key=lambda entry: (float(entry["score"]), entry["model_id"]), reverse=True)
        factor_reports.append(
            {
                "factor_idx": factor_rank,
                "eigenvalue": eigenvalue,
                "best_model": best_model,
                "weight_mode": weight_mode,
                "projection_basis_rank": int(residual_basis.shape[0]) if residual_basis is not None else 0,
                "mean_residual_energy_fraction": float(np.mean(residual_energy_values)) if residual_energy_values else None,
                "model_scores": model_entries,
                "top_named_anchors": anchor_entries[: min(5, len(anchor_entries))],
                "top_contributors": contributions[: min(top_contributors, len(contributions))],
            }
        )
    return factor_reports


def _build_latent_factor_report(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    num_factors: int,
    top_contributors: int,
) -> list[dict[str, object]]:
    direction_records = _collect_direction_records(representations, canonical_dim=canonical_dim, num_layers=num_layers)
    anchor_records = _collect_named_anchor_records(representations, canonical_dim=canonical_dim, num_layers=num_layers)
    anchor_vectors = [
        {
            **record,
            "vector": np.asarray(record["vector"], dtype=np.float32),
        }
        for record in anchor_records
    ]
    return _build_factor_report(
        direction_records,
        anchor_vectors=anchor_vectors,
        num_factors=num_factors,
        top_contributors=top_contributors,
        weight_mode="normalized_by_model_layer_mean",
    )


def _build_residual_latent_factor_report(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    num_factors: int,
    top_contributors: int,
) -> list[dict[str, object]]:
    direction_records = _collect_direction_records(representations, canonical_dim=canonical_dim, num_layers=num_layers)
    anchor_records = _collect_named_anchor_records(representations, canonical_dim=canonical_dim, num_layers=num_layers)
    anchor_vectors = [
        {
            **record,
            "vector": np.asarray(record["vector"], dtype=np.float32),
        }
        for record in anchor_records
    ]
    if not anchor_vectors:
        return []
    anchor_matrix = _normalize_rows(np.stack([np.asarray(record["vector"], dtype=np.float32) for record in anchor_vectors], axis=0))
    concept_count = len({str(record["concept"]) for record in anchor_vectors})
    anchor_basis_rank = min(
        canonical_dim,
        max(num_factors, concept_count),
    )
    residual_basis = _orthonormal_row_basis(anchor_matrix, max_rank=anchor_basis_rank)
    return _build_factor_report(
        direction_records,
        anchor_vectors=anchor_vectors,
        num_factors=num_factors,
        top_contributors=top_contributors,
        weight_mode="residual_to_named_anchor_span",
        residual_basis=residual_basis,
    )


def _chunk_loss_lookup(rep: ModelRepresentation) -> dict[str, float]:
    if rep.chunk_losses is None or rep.chunk_ids is None:
        return {}
    losses = np.asarray(rep.chunk_losses, dtype=np.float32).reshape(-1)
    count = min(len(rep.chunk_ids), int(losses.shape[0]))
    return {str(rep.chunk_ids[idx]): float(losses[idx]) for idx in range(count)}


def _pairwise_chunk_loss_metrics(rep_a: ModelRepresentation, rep_b: ModelRepresentation) -> dict[str, object]:
    losses_a = _chunk_loss_lookup(rep_a)
    losses_b = _chunk_loss_lookup(rep_b)
    if not losses_a or not losses_b:
        return {
            "shared_chunk_count": 0,
            "shared_chunk_loss_corr": None,
            "shared_chunk_loss_mae": None,
        }
    shared_ids = sorted(set(losses_a).intersection(losses_b))
    if not shared_ids:
        return {
            "shared_chunk_count": 0,
            "shared_chunk_loss_corr": None,
            "shared_chunk_loss_mae": None,
        }
    a = np.asarray([losses_a[chunk_id] for chunk_id in shared_ids], dtype=np.float32)
    b = np.asarray([losses_b[chunk_id] for chunk_id in shared_ids], dtype=np.float32)
    corr = None
    if a.shape[0] >= 2:
        corr_matrix = np.corrcoef(a, b)
        corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else None
    return {
        "shared_chunk_count": int(a.shape[0]),
        "shared_chunk_loss_corr": corr,
        "shared_chunk_loss_mae": float(np.mean(np.abs(a - b))),
    }


def _load_calibration_lookup(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    calibration_path = Path(path)
    if not calibration_path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with calibration_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            chunk_id = payload.get("chunk_id")
            if chunk_id is None:
                continue
            records[str(chunk_id)] = payload
    return records


def _summarize_advantage_side(
    *,
    winner_model: str,
    loser_model: str,
    winner_losses: dict[str, float],
    loser_losses: dict[str, float],
    chunk_ids: list[str],
    calibration_lookup: dict[str, dict[str, object]],
    top_chunks: int,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for chunk_id in chunk_ids:
        winner_loss = float(winner_losses[chunk_id])
        loser_loss = float(loser_losses[chunk_id])
        advantage = float(loser_loss - winner_loss)
        record = calibration_lookup.get(chunk_id, {})
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"text"}
        }
        text_preview = None
        if "text" in record:
            text_preview = " ".join(str(record["text"]).split())[:240]
        entries.append(
            {
                "chunk_id": chunk_id,
                "winner_model": winner_model,
                "loser_model": loser_model,
                "advantage": advantage,
                "winner_loss": winner_loss,
                "loser_loss": loser_loss,
                "text_preview": text_preview,
                "metadata": metadata,
            }
        )
    entries.sort(key=lambda entry: (float(entry["advantage"]), str(entry["chunk_id"])), reverse=True)

    numeric_keys = sorted(
        {
            key
            for entry in entries
            for key, value in dict(entry["metadata"]).items()
            if isinstance(value, (int, float))
        }
    )
    metadata_summary: dict[str, object] = {}
    for key in numeric_keys:
        values = [
            float(entry["metadata"][key])
            for entry in entries
            if key in entry["metadata"] and isinstance(entry["metadata"][key], (int, float))
        ]
        if values:
            metadata_summary[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    cluster_counts: dict[str, int] = {}
    for entry in entries:
        cluster_id = entry["metadata"].get("cluster_id")
        if cluster_id is None:
            continue
        key = str(cluster_id)
        cluster_counts[key] = cluster_counts.get(key, 0) + 1
    top_clusters = [
        {"cluster_id": cluster_id, "count": count}
        for cluster_id, count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]

    return {
        "winner_model": winner_model,
        "loser_model": loser_model,
        "win_count": len(entries),
        "mean_advantage": float(np.mean([float(entry["advantage"]) for entry in entries])) if entries else 0.0,
        "max_advantage": float(entries[0]["advantage"]) if entries else 0.0,
        "metadata_summary": metadata_summary,
        "top_clusters": top_clusters,
        "top_chunks": entries[: min(top_chunks, len(entries))],
    }


def _pairwise_advantage_report(
    rep_a: ModelRepresentation,
    rep_b: ModelRepresentation,
    *,
    calibration_lookup: dict[str, dict[str, object]],
    top_chunks: int,
) -> dict[str, object]:
    losses_a = _chunk_loss_lookup(rep_a)
    losses_b = _chunk_loss_lookup(rep_b)
    shared_ids = sorted(set(losses_a).intersection(losses_b))
    a_better = [chunk_id for chunk_id in shared_ids if float(losses_a[chunk_id]) < float(losses_b[chunk_id])]
    b_better = [chunk_id for chunk_id in shared_ids if float(losses_b[chunk_id]) < float(losses_a[chunk_id])]
    ties = len(shared_ids) - len(a_better) - len(b_better)
    return {
        "model_a": rep_a.model_id,
        "model_b": rep_b.model_id,
        "shared_chunk_count": len(shared_ids),
        "tie_count": int(ties),
        "model_a_better": _summarize_advantage_side(
            winner_model=rep_a.model_id,
            loser_model=rep_b.model_id,
            winner_losses=losses_a,
            loser_losses=losses_b,
            chunk_ids=a_better,
            calibration_lookup=calibration_lookup,
            top_chunks=top_chunks,
        ),
        "model_b_better": _summarize_advantage_side(
            winner_model=rep_b.model_id,
            loser_model=rep_a.model_id,
            winner_losses=losses_b,
            loser_losses=losses_a,
            chunk_ids=b_better,
            calibration_lookup=calibration_lookup,
            top_chunks=top_chunks,
        ),
    }


def build_comparison_report(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    latent_factors: int = 16,
    latent_top_contributors: int = 8,
    calibration_lookup: dict[str, dict[str, object]] | None = None,
    advantage_top_chunks: int = 8,
) -> dict[str, object]:
    if len(representations) < 2:
        raise ValueError("comparison requires at least two representations")
    if calibration_lookup is None:
        calibration_lookup = {}

    representation_summaries = []
    for rep in representations:
        chunk_loss_mean = None
        if rep.chunk_losses is not None and np.asarray(rep.chunk_losses).size > 0:
            chunk_loss_mean = float(np.mean(np.asarray(rep.chunk_losses, dtype=np.float32)))
        representation_summaries.append(
            {
                "model_id": rep.model_id,
                "architecture_family": rep.architecture_family,
                "num_parameters": rep.num_parameters,
                "hidden_dim": rep.hidden_dim,
                "num_layers": rep.num_layers,
                "mean_chunk_loss": chunk_loss_mean,
            }
        )

    pairwise_overlap_acc: dict[tuple[str, str], list[float]] = {}
    layer_reports: list[dict[str, object]] = []
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        mapped_by_model: dict[str, np.ndarray] = {}
        model_entries: list[dict[str, object]] = []
        for rep in representations:
            source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
            mapped = _map_directions(
                source_layer.directions,
                canonical_dim=canonical_dim,
                seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}"),
            )
            mapped_by_model[rep.model_id] = mapped
            model_entries.append(
                {
                    "model_id": rep.model_id,
                    "source_layer_idx": int(source_layer_idx),
                    "source_relative_depth": float(source_layer.relative_depth),
                    "num_directions": int(source_layer.directions.shape[0]),
                    "mapped_num_directions": int(mapped.shape[0]),
                    "scale_energy": _scale_energy(source_layer),
                    "scale_mean_abs": _scale_mean_abs(source_layer),
                    "covariance_trace": _covariance_trace(source_layer),
                    "importance_mean": _importance_mean(source_layer),
                }
            )
        pairwise_entries: list[dict[str, object]] = []
        for left_idx, left_rep in enumerate(representations):
            for right_rep in representations[left_idx + 1:]:
                key = (left_rep.model_id, right_rep.model_id)
                overlap = _subspace_overlap(mapped_by_model[left_rep.model_id], mapped_by_model[right_rep.model_id])
                pairwise_overlap_acc.setdefault(key, []).append(overlap)
                pairwise_entries.append(
                    {
                        "model_a": left_rep.model_id,
                        "model_b": right_rep.model_id,
                        "subspace_overlap": overlap,
                    }
                )
        best_entry = max(model_entries, key=lambda entry: (float(entry["scale_energy"]), str(entry["model_id"])))
        layer_reports.append(
            {
                "target_layer": target_layer,
                "relative_depth": relative_depth,
                "best_scale_energy_model": best_entry["model_id"],
                "mean_pairwise_subspace_overlap": float(np.mean([entry["subspace_overlap"] for entry in pairwise_entries]))
                if pairwise_entries else None,
                "models": model_entries,
                "pairwise_subspace_overlap": pairwise_entries,
            }
        )

    pairwise_summary: list[dict[str, object]] = []
    pairwise_advantages: list[dict[str, object]] = []
    for left_idx, left_rep in enumerate(representations):
        for right_rep in representations[left_idx + 1:]:
            key = (left_rep.model_id, right_rep.model_id)
            chunk_metrics = _pairwise_chunk_loss_metrics(left_rep, right_rep)
            overlaps = pairwise_overlap_acc.get(key, [])
            pairwise_summary.append(
                {
                    "model_a": left_rep.model_id,
                    "model_b": right_rep.model_id,
                    "mean_subspace_overlap": _finite_summary(overlaps)[0],
                    "min_subspace_overlap": _finite_summary(overlaps)[1],
                    "max_subspace_overlap": _finite_summary(overlaps)[2],
                    **chunk_metrics,
                }
            )
            pairwise_advantages.append(
                _pairwise_advantage_report(
                    left_rep,
                    right_rep,
                    calibration_lookup=calibration_lookup,
                    top_chunks=advantage_top_chunks,
                )
            )

    best_loss_model = None
    comparable_losses = [entry for entry in representation_summaries if entry["mean_chunk_loss"] is not None]
    if comparable_losses:
        best_loss_model = min(comparable_losses, key=lambda entry: float(entry["mean_chunk_loss"]))["model_id"]

    concept_reports: dict[str, object] = {}
    concept_names = sorted({str(name) for rep in representations for name in rep.concept_profiles})
    for concept_name in concept_names:
        model_entries: list[dict[str, object]] = []
        layer_reports_for_concept: list[dict[str, object]] = []
        description = ""
        for rep in representations:
            profile = rep.concept_profiles.get(concept_name)
            if not isinstance(profile, dict):
                continue
            if not description:
                description = str(profile.get("description", ""))
            layer_scores = [float(layer_payload.get("layer_score", 0.0)) for _layer_idx, layer_payload in _concept_layers(profile)]
            model_entries.append(
                {
                    "model_id": rep.model_id,
                    "sharpness": _concept_sharpness(profile),
                    "num_pairs": int(profile.get("num_pairs", 0)),
                    "best_layer_score": max(layer_scores) if layer_scores else 0.0,
                }
            )
        for target_layer in range(1, num_layers + 1):
            relative_depth = target_layer / max(num_layers, 1)
            mapped_by_model: dict[str, np.ndarray] = {}
            model_layer_entries: list[dict[str, object]] = []
            for rep in representations:
                profile = rep.concept_profiles.get(concept_name)
                if not isinstance(profile, dict):
                    continue
                nearest = _nearest_concept_layer(profile, relative_depth)
                if nearest is None:
                    continue
                source_layer_idx, source_payload = nearest
                direction = np.asarray(source_payload.get("direction", []), dtype=np.float32).reshape(1, -1)
                if direction.size == 0:
                    continue
                mapped = _map_directions(
                    direction,
                    canonical_dim=canonical_dim,
                    seed=_stable_seed(f"{rep.model_id}:{concept_name}:{source_layer_idx}:{canonical_dim}"),
                )
                mapped_by_model[rep.model_id] = mapped
                model_layer_entries.append(
                    {
                        "model_id": rep.model_id,
                        "source_layer_idx": int(source_layer_idx),
                        "source_relative_depth": float(source_payload.get("relative_depth", 0.0)),
                        "sharpness": _concept_sharpness(profile),
                        "layer_score": float(source_payload.get("layer_score", 0.0)),
                    }
                )
            pairwise_entries: list[dict[str, object]] = []
            for left_idx, left_rep in enumerate(representations):
                if left_rep.model_id not in mapped_by_model:
                    continue
                for right_rep in representations[left_idx + 1:]:
                    if right_rep.model_id not in mapped_by_model:
                        continue
                    pairwise_entries.append(
                        {
                            "model_a": left_rep.model_id,
                            "model_b": right_rep.model_id,
                            "subspace_overlap": _subspace_overlap(
                                mapped_by_model[left_rep.model_id],
                                mapped_by_model[right_rep.model_id],
                            ),
                        }
                    )
            best_model = None
            if model_layer_entries:
                best_model = max(
                    model_layer_entries,
                    key=lambda entry: (float(entry["layer_score"]), float(entry["sharpness"]), str(entry["model_id"])),
                )["model_id"]
            layer_reports_for_concept.append(
                {
                    "target_layer": target_layer,
                    "relative_depth": relative_depth,
                    "best_model": best_model,
                    "models": model_layer_entries,
                    "pairwise_subspace_overlap": pairwise_entries,
                }
            )
        best_model = None
        if model_entries:
            best_model = max(
                model_entries,
                key=lambda entry: (float(entry["sharpness"]), float(entry["best_layer_score"]), str(entry["model_id"])),
            )["model_id"]
        concept_reports[concept_name] = {
            "description": description,
            "best_model": best_model,
            "models": model_entries,
            "layers": layer_reports_for_concept,
        }

    latent_factor_reports = _build_latent_factor_report(
        representations,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
        num_factors=latent_factors,
        top_contributors=latent_top_contributors,
    )
    residual_latent_factor_reports = _build_residual_latent_factor_report(
        representations,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
        num_factors=latent_factors,
        top_contributors=latent_top_contributors,
    )

    return {
        "canonical_dim": canonical_dim,
        "num_layers": num_layers,
        "representations": representation_summaries,
        "best_mean_chunk_loss_model": best_loss_model,
        "layers": layer_reports,
        "pairwise_summary": pairwise_summary,
        "pairwise_advantages": pairwise_advantages,
        "concepts": concept_reports,
        "latent_factors": latent_factor_reports,
        "residual_latent_factors": residual_latent_factor_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple ModelRepresentation artifacts in a shared canonical frame and emit a JSON report."
    )
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Shared latent dimension used for comparison")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins used for comparison")
    parser.add_argument("--latent-factors", type=int, default=16, help="Number of latent factors to discover from the shared direction bank")
    parser.add_argument("--latent-top-contributors", type=int, default=8, help="Number of top contributing source directions to retain per latent factor")
    parser.add_argument("--calibration-jsonl", default="", help="Optional local calibration JSONL for joining chunk ids back to text and metadata")
    parser.add_argument("--advantage-top-chunks", type=int, default=8, help="Top winning chunks to retain per side in pairwise advantage reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    report = build_comparison_report(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        latent_factors=args.latent_factors,
        latent_top_contributors=args.latent_top_contributors,
        calibration_lookup=_load_calibration_lookup(args.calibration_jsonl or None),
        advantage_top_chunks=args.advantage_top_chunks,
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
