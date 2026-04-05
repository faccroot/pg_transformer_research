#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed
    from tools.representation_learning.compare_model_representations import _normalize_rows
    from tools.representation_learning.schemas import (
        ActivationEventDataset,
        EcologyTrainingSet,
        ForwardSignatureDataset,
        ModelRepresentation,
        VerificationTable,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from compare_model_representations import _normalize_rows  # type: ignore[no-redef]
    from schemas import ActivationEventDataset, EcologyTrainingSet, ForwardSignatureDataset, ModelRepresentation, VerificationTable  # type: ignore[no-redef]


DEFAULT_NUM_DISAGREEMENT_FACTORS = 8

BASE_FEATURE_NAMES: list[str] = [
    "loss",
    "loss_gap_to_best",
    "loss_gap_to_mean",
    "loss_rank_frac",
    "mean_loss",
    "disagreement_score",
    "num_models_present",
    "relative_depth",
    "has_projection",
    "projection_norm",
    "projection_abs_mean",
    "projection_peak_frac",
    "log10_num_parameters",
    "log2_hidden_dim",
    "concept_sharpness_mean",
    "concept_sharpness_max",
    "scale_energy",
    "scale_mean_abs",
    "covariance_trace",
    "importance_mean",
    "coactivation_trace",
    "coactivation_offdiag_mean",
    "centroid_cosine",
    "nearest_neighbor_cosine",
    "knn_mean_cosine",
    "prev_layer_cosine",
    "next_layer_cosine",
    "prev_layer_norm_ratio",
    "next_layer_norm_ratio",
    "cross_model_mean_cosine",
    "cross_model_max_cosine",
    "cross_model_min_cosine",
    "cross_model_low_agreement_frac",
    "has_forward_signature",
    "last_token_entropy",
    "sequence_mean_entropy",
    "last_token_top1_prob",
    "last_token_margin",
    "last_token_topk_mass",
    "attention_entropy",
    "attention_peak_frac",
    "cross_model_topk_jaccard_mean",
    "cross_model_topk_jaccard_max",
    "cross_model_topk_prob_l1_mean",
]


def _factor_feature_names(num_factors: int) -> list[str]:
    names: list[str] = []
    for factor_idx in range(max(int(num_factors), 0)):
        prefix = f"factor_{factor_idx + 1}"
        names.extend(
            [
                f"{prefix}_proj",
                f"{prefix}_abs",
                f"{prefix}_share",
                f"{prefix}_prev_align",
                f"{prefix}_next_align",
            ]
        )
    return names


def build_feature_names(num_disagreement_factors: int = DEFAULT_NUM_DISAGREEMENT_FACTORS) -> list[str]:
    return list(BASE_FEATURE_NAMES) + _factor_feature_names(num_disagreement_factors)


FEATURE_NAMES: list[str] = build_feature_names()


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


def _row_normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None), norms.reshape(-1)


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
            seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:ecology"),
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


def _concept_sharpness_stats(rep: ModelRepresentation) -> tuple[float, float]:
    sharpness = []
    for payload in rep.concept_profiles.values():
        if isinstance(payload, dict) and "sharpness" in payload:
            try:
                sharpness.append(float(payload["sharpness"]))
            except (TypeError, ValueError):
                continue
    if not sharpness:
        return 0.0, 0.0
    return float(np.mean(sharpness)), float(np.max(sharpness))


def _projection_summary(vector: np.ndarray | None) -> tuple[int, float, float, float, list[float]]:
    if vector is None:
        return 0, 0.0, 0.0, 0.0, []
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        return 1, 0.0, 0.0, 0.0, embedding.tolist()
    abs_embedding = np.abs(embedding)
    abs_mean = float(np.mean(abs_embedding))
    peak_frac = float(np.max(abs_embedding) / max(float(abs_embedding.sum()), 1e-8))
    return 1, norm, abs_mean, peak_frac, embedding.tolist()


def _scale_energy(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None:
        return 0.0
    if layer.scales is None:
        return float(layer.directions.shape[0])
    return float(np.sum(np.abs(np.asarray(layer.scales, dtype=np.float32))))


def _scale_mean_abs(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None:
        return 0.0
    if layer.scales is None or layer.scales.size == 0:
        return 1.0 if layer.directions.shape[0] > 0 else 0.0
    return float(np.mean(np.abs(np.asarray(layer.scales, dtype=np.float32))))


def _covariance_trace(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None or layer.covariance is None:
        return 0.0
    return float(np.trace(np.asarray(layer.covariance, dtype=np.float32)))


def _importance_mean(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None or layer.importance is None or layer.importance.size == 0:
        return 0.0
    return float(np.mean(np.asarray(layer.importance, dtype=np.float32)))


def _coactivation_trace(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None or layer.coactivation is None:
        return 0.0
    return float(np.trace(np.asarray(layer.coactivation, dtype=np.float32)))


def _coactivation_offdiag_mean(rep: ModelRepresentation, source_layer_idx: int | None) -> float:
    if source_layer_idx is None:
        return 0.0
    layer = rep.layer_geometries.get(int(source_layer_idx))
    if layer is None or layer.coactivation is None:
        return 0.0
    matrix = np.asarray(layer.coactivation, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask]))) if np.any(mask) else 0.0


def _layer_geometry_stats(
    chunk_embeddings_by_layer: dict[int, dict[str, object]],
) -> dict[int, dict[str, np.ndarray]]:
    stats: dict[int, dict[str, np.ndarray]] = {}
    for target_layer, payload in chunk_embeddings_by_layer.items():
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue
        normed, norms = _row_normalize(matrix)
        centroid = normed.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        centroid_cos = np.asarray(normed @ centroid, dtype=np.float32)
        if normed.shape[0] > 1:
            pairwise = np.asarray(normed @ normed.T, dtype=np.float32)
            np.fill_diagonal(pairwise, -np.inf)
            nearest_neighbor = np.max(pairwise, axis=1)
            nearest_neighbor = np.where(np.isfinite(nearest_neighbor), nearest_neighbor, 0.0).astype(np.float32)
            k = min(8, pairwise.shape[0] - 1)
            topk = np.partition(pairwise, kth=pairwise.shape[1] - k, axis=1)[:, -k:]
            topk = np.where(np.isfinite(topk), topk, 0.0)
            knn_mean = np.asarray(np.mean(topk, axis=1), dtype=np.float32)
        else:
            nearest_neighbor = np.zeros((normed.shape[0],), dtype=np.float32)
            knn_mean = np.zeros((normed.shape[0],), dtype=np.float32)
        stats[int(target_layer)] = {
            "normed": normed.astype(np.float32),
            "norms": norms.astype(np.float32),
            "centroid_cos": centroid_cos,
            "nearest_neighbor_cos": nearest_neighbor,
            "knn_mean_cos": knn_mean,
        }
    return stats


def _cross_layer_stats(
    layer_stats: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    results: dict[int, dict[str, np.ndarray]] = {}
    ordered_layers = sorted(layer_stats)
    for idx, target_layer in enumerate(ordered_layers):
        current = layer_stats[target_layer]
        current_normed = np.asarray(current["normed"], dtype=np.float32)
        current_norms = np.asarray(current["norms"], dtype=np.float32)
        row_count = current_normed.shape[0]
        prev_cos = np.zeros((row_count,), dtype=np.float32)
        next_cos = np.zeros((row_count,), dtype=np.float32)
        prev_ratio = np.ones((row_count,), dtype=np.float32)
        next_ratio = np.ones((row_count,), dtype=np.float32)

        if idx > 0:
            prev = layer_stats[ordered_layers[idx - 1]]
            prev_normed = np.asarray(prev["normed"], dtype=np.float32)
            prev_norms = np.asarray(prev["norms"], dtype=np.float32)
            count = min(row_count, prev_normed.shape[0])
            prev_cos[:count] = np.sum(current_normed[:count] * prev_normed[:count], axis=1)
            prev_ratio[:count] = current_norms[:count] / np.clip(prev_norms[:count], 1e-8, None)
        if idx + 1 < len(ordered_layers):
            nxt = layer_stats[ordered_layers[idx + 1]]
            next_normed = np.asarray(nxt["normed"], dtype=np.float32)
            next_norms = np.asarray(nxt["norms"], dtype=np.float32)
            count = min(row_count, next_normed.shape[0])
            next_cos[:count] = np.sum(current_normed[:count] * next_normed[:count], axis=1)
            next_ratio[:count] = current_norms[:count] / np.clip(next_norms[:count], 1e-8, None)
        results[target_layer] = {
            "prev_layer_cosine": prev_cos,
            "next_layer_cosine": next_cos,
            "prev_layer_norm_ratio": prev_ratio,
            "next_layer_norm_ratio": next_ratio,
        }
    return results


def _safe_row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _build_disagreement_factors(
    source_models: list[str],
    *,
    verification_entries: list[dict[str, object]],
    losses_by_model: dict[str, dict[str, float]],
    chunk_indices: dict[str, dict[str, int]],
    chunk_embeddings: dict[str, dict[int, dict[str, object]]],
    num_layers: int,
    canonical_dim: int,
    num_disagreement_factors: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]]]:
    factors_by_layer: dict[int, np.ndarray] = {}
    metadata_by_layer: dict[int, dict[str, object]] = {}
    factor_count = max(int(num_disagreement_factors), 0)
    for target_layer in range(1, num_layers + 1):
        basis = np.zeros((factor_count, canonical_dim), dtype=np.float32)
        diff_vectors: list[np.ndarray] = []
        for entry in verification_entries:
            chunk_id = str(entry["chunk_id"])
            winner_model = str(entry["verified_winner_model"])
            if winner_model not in source_models:
                continue
            if chunk_id not in losses_by_model.get(winner_model, {}):
                continue
            winner_chunk_idx = chunk_indices.get(winner_model, {}).get(chunk_id)
            winner_payload = chunk_embeddings.get(winner_model, {}).get(target_layer)
            if winner_chunk_idx is None or winner_payload is None:
                continue
            winner_matrix = np.asarray(winner_payload["matrix"], dtype=np.float32)
            if winner_chunk_idx >= winner_matrix.shape[0]:
                continue
            winner_embedding = np.asarray(winner_matrix[winner_chunk_idx], dtype=np.float32)
            for other_model in source_models:
                if other_model == winner_model:
                    continue
                if chunk_id not in losses_by_model.get(other_model, {}):
                    continue
                other_chunk_idx = chunk_indices.get(other_model, {}).get(chunk_id)
                other_payload = chunk_embeddings.get(other_model, {}).get(target_layer)
                if other_chunk_idx is None or other_payload is None:
                    continue
                other_matrix = np.asarray(other_payload["matrix"], dtype=np.float32)
                if other_chunk_idx >= other_matrix.shape[0]:
                    continue
                other_embedding = np.asarray(other_matrix[other_chunk_idx], dtype=np.float32)
                diff = winner_embedding - other_embedding
                if float(np.linalg.norm(diff)) > 1e-8:
                    diff_vectors.append(diff.astype(np.float32))
        singular_values: list[float] = []
        explained_energy: list[float] = []
        component_count = 0
        if diff_vectors and factor_count > 0:
            diff_matrix = np.stack(diff_vectors, axis=0).astype(np.float32)
            centered = diff_matrix - diff_matrix.mean(axis=0, keepdims=True)
            if float(np.linalg.norm(centered)) <= 1e-8:
                centered = diff_matrix
            try:
                _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
                component_count = min(factor_count, vt.shape[0], vt.shape[1])
                if component_count > 0:
                    basis[:component_count] = _safe_row_normalize(vt[:component_count]).astype(np.float32)
                singular_values = [float(value) for value in singular[:component_count].tolist()]
                energy = singular[:component_count] ** 2
                total_energy = float(np.sum(singular**2))
                if total_energy > 1e-8:
                    explained_energy = [float(value) for value in (energy / total_energy).tolist()]
                else:
                    explained_energy = [0.0 for _ in range(component_count)]
            except np.linalg.LinAlgError:
                component_count = 0
        factors_by_layer[int(target_layer)] = basis
        metadata_by_layer[str(int(target_layer))] = {
            "sample_count": int(len(diff_vectors)),
            "component_count": int(component_count),
            "singular_values": singular_values,
            "explained_energy": explained_energy,
        }
    return factors_by_layer, metadata_by_layer


def _factor_projection_tables(
    chunk_embeddings: dict[str, dict[int, dict[str, object]]],
    factors_by_layer: dict[int, np.ndarray],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    output: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for model_id, layers in chunk_embeddings.items():
        model_tables: dict[int, dict[str, np.ndarray]] = {}
        for target_layer, payload in layers.items():
            matrix = np.asarray(payload["matrix"], dtype=np.float32)
            basis = np.asarray(factors_by_layer.get(int(target_layer), np.zeros((0, matrix.shape[1]), dtype=np.float32)), dtype=np.float32)
            if matrix.ndim != 2 or basis.ndim != 2:
                continue
            coords = np.asarray(matrix @ basis.T, dtype=np.float32)
            abs_coords = np.abs(coords)
            shares = abs_coords / np.clip(abs_coords.sum(axis=1, keepdims=True), 1e-8, None)
            model_tables[int(target_layer)] = {
                "coords": coords,
                "shares": shares.astype(np.float32),
            }
        output[model_id] = model_tables
    return output


def _factor_alignment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.asarray((2.0 * a * b) / np.clip((a * a) + (b * b), 1e-8, None), dtype=np.float32)


def _factor_transition_stats(
    factor_tables: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    ordered_layers = sorted(factor_tables)
    for idx, target_layer in enumerate(ordered_layers):
        current = np.asarray(factor_tables[target_layer]["coords"], dtype=np.float32)
        if current.ndim != 2:
            continue
        row_count, factor_count = current.shape
        prev_align = np.zeros((row_count, factor_count), dtype=np.float32)
        next_align = np.zeros((row_count, factor_count), dtype=np.float32)
        if idx > 0:
            prev = np.asarray(factor_tables[ordered_layers[idx - 1]]["coords"], dtype=np.float32)
            count = min(row_count, prev.shape[0])
            prev_align[:count] = _factor_alignment(current[:count], prev[:count])
        if idx + 1 < len(ordered_layers):
            nxt = np.asarray(factor_tables[ordered_layers[idx + 1]]["coords"], dtype=np.float32)
            count = min(row_count, nxt.shape[0])
            next_align[:count] = _factor_alignment(current[:count], nxt[:count])
        output[int(target_layer)] = {
            "prev_align": prev_align,
            "next_align": next_align,
        }
    return output


def _digit_frac(text: str) -> float:
    return float(sum(ch.isdigit() for ch in text)) / max(len(text), 1)


def _upper_frac(text: str) -> float:
    return float(sum(ch.isupper() for ch in text)) / max(len(text), 1)


def _punct_frac(text: str) -> float:
    punctuation = set(".,;:!?-()[]{}\"'`")
    return float(sum(ch in punctuation for ch in text)) / max(len(text), 1)


def _newline_frac(text: str) -> float:
    return float(text.count("\n")) / max(len(text), 1)


def _urlish_score(text: str) -> float:
    lowered = text.lower()
    markers = ("http", "www.", ".com", ".org", ".net", "@", "://")
    count = sum(lowered.count(marker) for marker in markers)
    return float(count) / max(len(text.split()), 1)


def _code_symbol_frac(text: str) -> float:
    symbols = set("{}[]();=<>_/\\")
    return float(sum(ch in symbols for ch in text)) / max(len(text), 1)


def _alpha_frac(text: str) -> float:
    return float(sum(ch.isalpha() for ch in text)) / max(len(text), 1)


def _operator_token_frac(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens:
        return 0.0
    operator_words = {"if", "then", "else", "because", "therefore", "not", "and", "or", "but", "when", "while", "unless"}
    return float(sum(token in operator_words for token in tokens)) / max(len(tokens), 1)


def _text_feature_payload(text: str) -> dict[str, float]:
    tokens = re.findall(r"\S+", text)
    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    avg_word_len = float(np.mean([len(token) for token in alpha_tokens], dtype=np.float32)) if alpha_tokens else 0.0
    return {
        "text_char_length": float(len(text)),
        "text_word_count": float(len(tokens)),
        "text_avg_word_length": avg_word_len,
        "text_digit_frac": _digit_frac(text),
        "text_upper_frac": _upper_frac(text),
        "text_punct_frac": _punct_frac(text),
        "text_newline_frac": _newline_frac(text),
        "text_urlish_score": _urlish_score(text),
        "text_code_symbol_frac": _code_symbol_frac(text),
        "text_alpha_frac": _alpha_frac(text),
        "text_operator_token_frac": _operator_token_frac(text),
    }


def _load_calibration_context(calibration_jsonl: str | Path | None) -> dict[str, dict[str, float]]:
    if calibration_jsonl is None:
        return {}
    path = Path(calibration_jsonl)
    if not path.exists():
        return {}
    payloads: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                continue
            payloads[chunk_id] = _text_feature_payload(str(row.get("text", "")))
    return payloads


def _entry_metadata_context(entry: dict[str, object]) -> dict[str, float]:
    payload = entry.get("metadata", {})
    if not isinstance(payload, dict):
        return {}
    output: dict[str, float] = {}
    for key in ("compressibility_ratio", "duplicate_score", "operator_density"):
        try:
            output[f"meta_{key}"] = float(payload[key])
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _forward_chunk_index_lookup(dataset: ForwardSignatureDataset) -> dict[str, int]:
    return {str(chunk_id): idx for idx, chunk_id in enumerate(dataset.chunk_ids)}


def _nearest_forward_layer(
    dataset: ForwardSignatureDataset,
    *,
    rep: ModelRepresentation,
    target_layer: int,
    num_layers: int,
) -> int | None:
    if not dataset.layer_features:
        return None
    target_depth = float(target_layer / max(num_layers, 1))
    return min(
        dataset.layer_features,
        key=lambda layer_idx: abs(float(layer_idx / max(rep.num_layers, 1)) - target_depth),
    )


def _topk_signature(
    dataset: ForwardSignatureDataset | None,
    chunk_idx: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset is None or chunk_idx is None:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    if dataset.topk_token_ids is None or dataset.topk_token_probs is None:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    if chunk_idx < 0 or chunk_idx >= dataset.topk_token_ids.shape[0]:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return (
        np.asarray(dataset.topk_token_ids[chunk_idx], dtype=np.int32).reshape(-1),
        np.asarray(dataset.topk_token_probs[chunk_idx], dtype=np.float32).reshape(-1),
    )


def _topk_jaccard(ids_a: np.ndarray, ids_b: np.ndarray) -> float:
    if ids_a.size == 0 or ids_b.size == 0:
        return 0.0
    set_a = {int(value) for value in ids_a.tolist()}
    set_b = {int(value) for value in ids_b.tolist()}
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b)) / float(len(union))


def _topk_prob_l1(ids_a: np.ndarray, probs_a: np.ndarray, ids_b: np.ndarray, probs_b: np.ndarray) -> float:
    if ids_a.size == 0 or ids_b.size == 0:
        return 0.0
    weight_a = {int(token_id): float(prob) for token_id, prob in zip(ids_a.tolist(), probs_a.tolist(), strict=True)}
    weight_b = {int(token_id): float(prob) for token_id, prob in zip(ids_b.tolist(), probs_b.tolist(), strict=True)}
    tokens = set(weight_a) | set(weight_b)
    if not tokens:
        return 0.0
    return float(sum(abs(weight_a.get(token, 0.0) - weight_b.get(token, 0.0)) for token in tokens))


def build_ecology_training_data(
    representations: list[ModelRepresentation],
    verification_table: VerificationTable,
    *,
    canonical_dim: int,
    num_layers: int,
    calibration_jsonl: str | Path | None = None,
    forward_signatures: list[ForwardSignatureDataset] | None = None,
    num_disagreement_factors: int = DEFAULT_NUM_DISAGREEMENT_FACTORS,
) -> tuple[ActivationEventDataset, EcologyTrainingSet]:
    if not representations:
        raise ValueError("representations must not be empty")
    source_models = [rep.model_id for rep in representations]
    rep_by_id = {rep.model_id: rep for rep in representations}
    losses_by_model = {rep.model_id: _chunk_loss_lookup(rep) for rep in representations}
    chunk_indices = {rep.model_id: _chunk_index_lookup(rep) for rep in representations}
    chunk_embeddings = {
        rep.model_id: _canonical_chunk_embeddings(rep, canonical_dim=canonical_dim, num_layers=num_layers)
        for rep in representations
    }
    verified_entries = [entry for entry in verification_table.entries if bool(entry.get("verified", True))]
    disagreement_factors, disagreement_factor_metadata = _build_disagreement_factors(
        source_models,
        verification_entries=verified_entries,
        losses_by_model=losses_by_model,
        chunk_indices=chunk_indices,
        chunk_embeddings=chunk_embeddings,
        num_layers=num_layers,
        canonical_dim=canonical_dim,
        num_disagreement_factors=num_disagreement_factors,
    )
    factor_projection_tables = _factor_projection_tables(chunk_embeddings, disagreement_factors)
    layer_geometry_stats = {
        rep.model_id: _layer_geometry_stats(chunk_embeddings[rep.model_id])
        for rep in representations
    }
    layer_transition_stats = {
        rep.model_id: _cross_layer_stats(layer_geometry_stats[rep.model_id])
        for rep in representations
    }
    factor_transition_stats = {
        model_id: _factor_transition_stats(factor_projection_tables.get(model_id, {}))
        for model_id in source_models
    }
    model_static = {
        rep.model_id: {
            "log10_num_parameters": float(math.log10(max(int(rep.num_parameters), 1))),
            "log2_hidden_dim": float(math.log2(max(int(rep.hidden_dim), 1))),
            "concept_sharpness_mean": _concept_sharpness_stats(rep)[0],
            "concept_sharpness_max": _concept_sharpness_stats(rep)[1],
        }
        for rep in representations
    }
    calibration_context = _load_calibration_context(calibration_jsonl)
    forward_by_model = {dataset.model_id: dataset for dataset in (forward_signatures or [])}
    forward_chunk_indices = {
        model_id: _forward_chunk_index_lookup(dataset)
        for model_id, dataset in forward_by_model.items()
    }

    events: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    max_embedding_dim = 0

    feature_names = build_feature_names(num_disagreement_factors)
    for entry in verified_entries:
        chunk_id = str(entry["chunk_id"])
        present_models = [model_id for model_id in source_models if chunk_id in losses_by_model[model_id]]
        if len(present_models) < 2:
            continue
        present_losses = [float(losses_by_model[model_id][chunk_id]) for model_id in present_models]
        best_loss = min(present_losses)
        mean_loss = float(np.mean(np.asarray(present_losses, dtype=np.float32)))
        disagreement_score = float((max(present_losses) - min(present_losses)) / max(mean_loss, 1e-6))
        ranked_models = sorted(
            ((model_id, float(losses_by_model[model_id][chunk_id])) for model_id in present_models),
            key=lambda item: (item[1], item[0]),
        )
        rank_map = {model_id: idx for idx, (model_id, _loss) in enumerate(ranked_models)}

        for target_layer in range(1, num_layers + 1):
            relative_depth = float(target_layer / max(num_layers, 1))
            candidate_event_ids: list[str] = []
            candidate_payloads: list[dict[str, object]] = []
            normed_by_model: dict[str, np.ndarray] = {}
            topk_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            for model_id in present_models:
                chunk_idx = chunk_indices[model_id].get(chunk_id)
                layer_stats = layer_geometry_stats[model_id].get(target_layer)
                if chunk_idx is not None and layer_stats is not None and chunk_idx < int(layer_stats["normed"].shape[0]):
                    normed_by_model[model_id] = np.asarray(layer_stats["normed"][chunk_idx], dtype=np.float32)
                forward_dataset = forward_by_model.get(model_id)
                forward_chunk_idx = forward_chunk_indices.get(model_id, {}).get(chunk_id)
                topk_by_model[model_id] = _topk_signature(forward_dataset, forward_chunk_idx)

            for model_id in present_models:
                loss = float(losses_by_model[model_id][chunk_id])
                chunk_idx = chunk_indices[model_id].get(chunk_id)
                embedding_vector = None
                source_layer_idx = None
                layer_payload = chunk_embeddings[model_id].get(target_layer)
                if layer_payload is not None and chunk_idx is not None:
                    matrix = np.asarray(layer_payload["matrix"], dtype=np.float32)
                    if chunk_idx < matrix.shape[0]:
                        embedding_vector = np.asarray(matrix[chunk_idx], dtype=np.float32)
                        source_layer_idx = int(layer_payload["source_layer_idx"])
                has_projection, projection_norm, projection_abs_mean, projection_peak_frac, embedding = _projection_summary(embedding_vector)
                static = model_static[model_id]

                centroid_cosine = 0.0
                nearest_neighbor_cosine = 0.0
                knn_mean_cosine = 0.0
                prev_layer_cosine = 0.0
                next_layer_cosine = 0.0
                prev_layer_norm_ratio = 1.0
                next_layer_norm_ratio = 1.0
                if chunk_idx is not None:
                    layer_stats = layer_geometry_stats[model_id].get(target_layer)
                    if layer_stats is not None and chunk_idx < int(layer_stats["centroid_cos"].shape[0]):
                        centroid_cosine = float(layer_stats["centroid_cos"][chunk_idx])
                        nearest_neighbor_cosine = float(layer_stats["nearest_neighbor_cos"][chunk_idx])
                        knn_mean_cosine = float(layer_stats["knn_mean_cos"][chunk_idx])
                    transition = layer_transition_stats[model_id].get(target_layer)
                    if transition is not None and chunk_idx < int(transition["prev_layer_cosine"].shape[0]):
                        prev_layer_cosine = float(transition["prev_layer_cosine"][chunk_idx])
                        next_layer_cosine = float(transition["next_layer_cosine"][chunk_idx])
                        prev_layer_norm_ratio = float(transition["prev_layer_norm_ratio"][chunk_idx])
                        next_layer_norm_ratio = float(transition["next_layer_norm_ratio"][chunk_idx])

                cross_model_cosines: list[float] = []
                current_normed = normed_by_model.get(model_id)
                if current_normed is not None:
                    for other_model, other_normed in normed_by_model.items():
                        if other_model == model_id:
                            continue
                        cross_model_cosines.append(float(np.dot(current_normed, other_normed)))
                cross_model_mean_cosine = float(np.mean(cross_model_cosines)) if cross_model_cosines else 0.0
                cross_model_max_cosine = float(np.max(cross_model_cosines)) if cross_model_cosines else 0.0
                cross_model_min_cosine = float(np.min(cross_model_cosines)) if cross_model_cosines else 0.0
                cross_model_low_agreement_frac = (
                    float(sum(value < 0.25 for value in cross_model_cosines)) / max(len(cross_model_cosines), 1)
                    if cross_model_cosines else 0.0
                )
                forward_dataset = forward_by_model.get(model_id)
                forward_chunk_idx = forward_chunk_indices.get(model_id, {}).get(chunk_id)
                has_forward_signature = 1.0 if forward_dataset is not None and forward_chunk_idx is not None else 0.0
                last_token_entropy = 0.0
                sequence_mean_entropy = 0.0
                last_token_top1_prob = 0.0
                last_token_margin = 0.0
                last_token_topk_mass = 0.0
                attention_entropy = 0.0
                attention_peak_frac = 0.0
                if forward_dataset is not None and forward_chunk_idx is not None:
                    for name, default in (
                        ("last_token_entropy", 0.0),
                        ("sequence_mean_entropy", 0.0),
                        ("last_token_top1_prob", 0.0),
                        ("last_token_margin", 0.0),
                        ("last_token_topk_mass", 0.0),
                    ):
                        values = forward_dataset.global_features.get(name)
                        if values is not None and forward_chunk_idx < values.shape[0]:
                            if name == "last_token_entropy":
                                last_token_entropy = float(values[forward_chunk_idx])
                            elif name == "sequence_mean_entropy":
                                sequence_mean_entropy = float(values[forward_chunk_idx])
                            elif name == "last_token_top1_prob":
                                last_token_top1_prob = float(values[forward_chunk_idx])
                            elif name == "last_token_margin":
                                last_token_margin = float(values[forward_chunk_idx])
                            elif name == "last_token_topk_mass":
                                last_token_topk_mass = float(values[forward_chunk_idx])
                    forward_layer_idx = _nearest_forward_layer(
                        forward_dataset,
                        rep=rep_by_id[model_id],
                        target_layer=target_layer,
                        num_layers=num_layers,
                    )
                    if forward_layer_idx is not None:
                        attention_values = forward_dataset.layer_features.get(int(forward_layer_idx), {})
                        values = attention_values.get("attention_entropy")
                        if values is not None and forward_chunk_idx < values.shape[0]:
                            attention_entropy = float(values[forward_chunk_idx])
                        values = attention_values.get("attention_peak_frac")
                        if values is not None and forward_chunk_idx < values.shape[0]:
                            attention_peak_frac = float(values[forward_chunk_idx])
                current_ids, current_probs = topk_by_model.get(model_id, (np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)))
                topk_jaccards: list[float] = []
                topk_l1s: list[float] = []
                for other_model, (other_ids, other_probs) in topk_by_model.items():
                    if other_model == model_id:
                        continue
                    topk_jaccards.append(_topk_jaccard(current_ids, other_ids))
                    topk_l1s.append(_topk_prob_l1(current_ids, current_probs, other_ids, other_probs))
                cross_model_topk_jaccard_mean = float(np.mean(topk_jaccards)) if topk_jaccards else 0.0
                cross_model_topk_jaccard_max = float(np.max(topk_jaccards)) if topk_jaccards else 0.0
                cross_model_topk_prob_l1_mean = float(np.mean(topk_l1s)) if topk_l1s else 0.0
                factor_features: list[float] = []
                factor_table = factor_projection_tables.get(model_id, {}).get(target_layer)
                factor_transition = factor_transition_stats.get(model_id, {}).get(target_layer)
                if (
                    chunk_idx is not None
                    and factor_table is not None
                    and chunk_idx < int(np.asarray(factor_table["coords"]).shape[0])
                ):
                    coords = np.asarray(factor_table["coords"][chunk_idx], dtype=np.float32).reshape(-1)
                    shares = np.asarray(factor_table["shares"][chunk_idx], dtype=np.float32).reshape(-1)
                    if factor_transition is not None and chunk_idx < int(np.asarray(factor_transition["prev_align"]).shape[0]):
                        prev_align = np.asarray(factor_transition["prev_align"][chunk_idx], dtype=np.float32).reshape(-1)
                        next_align = np.asarray(factor_transition["next_align"][chunk_idx], dtype=np.float32).reshape(-1)
                    else:
                        prev_align = np.zeros_like(coords, dtype=np.float32)
                        next_align = np.zeros_like(coords, dtype=np.float32)
                else:
                    coords = np.zeros((max(int(num_disagreement_factors), 0),), dtype=np.float32)
                    shares = np.zeros_like(coords, dtype=np.float32)
                    prev_align = np.zeros_like(coords, dtype=np.float32)
                    next_align = np.zeros_like(coords, dtype=np.float32)
                for factor_idx in range(max(int(num_disagreement_factors), 0)):
                    factor_features.extend(
                        [
                            float(coords[factor_idx]) if factor_idx < coords.shape[0] else 0.0,
                            float(abs(coords[factor_idx])) if factor_idx < coords.shape[0] else 0.0,
                            float(shares[factor_idx]) if factor_idx < shares.shape[0] else 0.0,
                            float(prev_align[factor_idx]) if factor_idx < prev_align.shape[0] else 0.0,
                            float(next_align[factor_idx]) if factor_idx < next_align.shape[0] else 0.0,
                        ]
                    )

                feature_values = [
                    loss,
                    loss - best_loss,
                    loss - mean_loss,
                    float(rank_map[model_id]) / max(len(present_models) - 1, 1),
                    mean_loss,
                    disagreement_score,
                    float(len(present_models)),
                    relative_depth,
                    float(has_projection),
                    projection_norm,
                    projection_abs_mean,
                    projection_peak_frac,
                    float(static["log10_num_parameters"]),
                    float(static["log2_hidden_dim"]),
                    float(static["concept_sharpness_mean"]),
                    float(static["concept_sharpness_max"]),
                    _scale_energy(rep_by_id[model_id], source_layer_idx),
                    _scale_mean_abs(rep_by_id[model_id], source_layer_idx),
                    _covariance_trace(rep_by_id[model_id], source_layer_idx),
                    _importance_mean(rep_by_id[model_id], source_layer_idx),
                    _coactivation_trace(rep_by_id[model_id], source_layer_idx),
                    _coactivation_offdiag_mean(rep_by_id[model_id], source_layer_idx),
                    centroid_cosine,
                    nearest_neighbor_cosine,
                    knn_mean_cosine,
                    prev_layer_cosine,
                    next_layer_cosine,
                    prev_layer_norm_ratio,
                    next_layer_norm_ratio,
                    cross_model_mean_cosine,
                    cross_model_max_cosine,
                    cross_model_min_cosine,
                    cross_model_low_agreement_frac,
                    has_forward_signature,
                    last_token_entropy,
                    sequence_mean_entropy,
                    last_token_top1_prob,
                    last_token_margin,
                    last_token_topk_mass,
                    attention_entropy,
                    attention_peak_frac,
                    cross_model_topk_jaccard_mean,
                    cross_model_topk_jaccard_max,
                    cross_model_topk_prob_l1_mean,
                    *factor_features,
                ]
                event_id = f"{entry['probe_id']}::{model_id}::L{target_layer}"
                event = {
                    "event_id": event_id,
                    "probe_id": str(entry["probe_id"]),
                    "chunk_id": chunk_id,
                    "model_id": model_id,
                    "target_layer": int(target_layer),
                    "source_layer_idx": source_layer_idx,
                    "probe_type": str(entry.get("probe_type", "unknown")),
                    "features": {name: float(value) for name, value in zip(feature_names, feature_values, strict=True)},
                    "embedding": embedding,
                    "metadata": {
                        "winner_model": str(entry["verified_winner_model"]),
                        "relative_depth": relative_depth,
                    },
                }
                events.append(event)
                candidate_event_ids.append(event_id)
                candidate_payloads.append(
                    {
                        "event_id": event_id,
                        "model_id": model_id,
                        "features": event["features"],
                        "embedding": embedding,
                        "source_layer_idx": source_layer_idx,
                    }
                )
                max_embedding_dim = max(max_embedding_dim, len(embedding))

            examples.append(
                {
                    "example_id": f"{entry['probe_id']}::L{target_layer}",
                    "probe_id": str(entry["probe_id"]),
                    "chunk_id": chunk_id,
                    "probe_type": str(entry.get("probe_type", "unknown")),
                    "target_layer": int(target_layer),
                    "winner_model": str(entry["verified_winner_model"]),
                    "verification_confidence": float(entry.get("verification_confidence", 0.0)),
                    "candidate_model_ids": present_models,
                    "candidate_event_ids": candidate_event_ids,
                    "candidates": candidate_payloads,
                    "context_features": {
                        "mean_loss": mean_loss,
                        "disagreement_score": disagreement_score,
                        "num_models_present": float(len(present_models)),
                        **_entry_metadata_context(entry),
                        **calibration_context.get(chunk_id, {}),
                    },
                }
            )

    event_dataset = ActivationEventDataset(
        source_models=source_models,
        feature_names=list(feature_names),
        embedding_dim=int(max_embedding_dim),
        events=events,
        metadata={
            "builder": "build_ecology_training_data",
            "canonical_dim": int(canonical_dim),
            "num_layers": int(num_layers),
            "event_count": len(events),
            "calibration_jsonl": None if calibration_jsonl is None else str(Path(calibration_jsonl)),
            "forward_signature_models": sorted(forward_by_model),
            "num_disagreement_factors": int(num_disagreement_factors),
            "disagreement_factor_metadata": disagreement_factor_metadata,
        },
    )
    training_set = EcologyTrainingSet(
        source_models=source_models,
        feature_names=list(feature_names),
        embedding_dim=int(max_embedding_dim),
        examples=examples,
        metadata={
            "builder": "build_ecology_training_data",
            "canonical_dim": int(canonical_dim),
            "num_layers": int(num_layers),
            "example_count": len(examples),
            "verified_entry_count": len(verified_entries),
            "calibration_jsonl": None if calibration_jsonl is None else str(Path(calibration_jsonl)),
            "forward_signature_models": sorted(forward_by_model),
            "num_disagreement_factors": int(num_disagreement_factors),
            "disagreement_factor_metadata": disagreement_factor_metadata,
        },
    )
    return event_dataset, training_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed activation events and ecology training examples from verified cross-model disagreements.")
    parser.add_argument("output_dir", help="Directory for the generated activation-event and ecology-training artifacts")
    parser.add_argument("verification_table", help="Input verification table artifact")
    parser.add_argument("representations", nargs="+", help="Input model representation artifacts")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-disagreement-factors", type=int, default=DEFAULT_NUM_DISAGREEMENT_FACTORS)
    parser.add_argument("--calibration-jsonl", default=None, help="Optional calibration JSONL to derive chunk text features")
    parser.add_argument("--forward-signatures", nargs="*", default=None, help="Optional forward signature sidecars aligned to the representations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verification_table = VerificationTable.load(args.verification_table)
    reps = [ModelRepresentation.load(path) for path in args.representations]
    forward_signatures = [ForwardSignatureDataset.load(path) for path in args.forward_signatures or []]
    event_dataset, training_set = build_ecology_training_data(
        reps,
        verification_table,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        calibration_jsonl=args.calibration_jsonl,
        forward_signatures=forward_signatures,
        num_disagreement_factors=args.num_disagreement_factors,
    )
    events_path = output_dir / "activation_event_dataset.npz"
    training_path = output_dir / "ecology_training_set.npz"
    event_dataset.save(events_path)
    training_set.save(training_path)
    summary = {
        "output_dir": str(output_dir),
        "activation_event_dataset": str(events_path),
        "ecology_training_set": str(training_path),
        "event_count": len(event_dataset.events),
        "example_count": len(training_set.examples),
        "embedding_dim": int(event_dataset.embedding_dim),
        "feature_names": event_dataset.feature_names,
        "source_models": event_dataset.source_models,
        "num_disagreement_factors": int(args.num_disagreement_factors),
        "calibration_jsonl": None if args.calibration_jsonl is None else str(Path(args.calibration_jsonl).resolve()),
        "forward_signatures": [dataset.model_id for dataset in forward_signatures],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
