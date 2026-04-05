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
        _dominant_directions_from_covariance,
        _map_directions,
        _nearest_layer,
        _stable_seed,
    )
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _dominant_directions_from_covariance, _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _project_residual(vector: np.ndarray, basis: np.ndarray | None) -> tuple[np.ndarray, float]:
    direction = np.asarray(vector, dtype=np.float32).reshape(-1)
    if basis is None or basis.size == 0:
        return direction, 1.0
    orthobasis = _normalize_rows(np.asarray(basis, dtype=np.float32))
    projected = (direction @ orthobasis.T) @ orthobasis
    residual = direction - projected
    base_norm = float(np.linalg.norm(direction))
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-8 or base_norm <= 1e-8:
        return np.zeros_like(direction), 0.0
    return residual / residual_norm, residual_norm / base_norm


def _append_basis_vector(basis: np.ndarray | None, vector: np.ndarray) -> np.ndarray:
    candidate = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if basis is None or basis.size == 0:
        return _normalize_rows(candidate)
    stacked = np.concatenate([np.asarray(basis, dtype=np.float32), candidate], axis=0)
    q, _r = np.linalg.qr(stacked.T, mode="reduced")
    return np.asarray(q.T, dtype=np.float32)


def _normalized_scales(layer: LayerGeometry, *, count: int) -> np.ndarray:
    if layer.scales is None or layer.scales.size == 0:
        return np.ones((count,), dtype=np.float32)
    scales = np.asarray(layer.scales, dtype=np.float32).reshape(-1)[:count]
    return np.abs(scales) / max(float(np.mean(np.abs(scales))), 1e-6)


def _novelty_weight_map(
    novelty_scores: list[dict[str, float | str]],
    *,
    temperature: float = 0.75,
) -> dict[str, float]:
    if not novelty_scores:
        return {}
    raw = np.asarray(
        [float(score["leave_one_out_novelty"]) for score in novelty_scores],
        dtype=np.float32,
    )
    if raw.size <= 1 or float(raw.std()) <= 1e-6:
        weights = np.ones_like(raw)
    else:
        z = (raw - float(raw.mean())) / max(float(raw.std()), 1e-6)
        weights = np.exp(np.clip(z * float(temperature), -1.0, 1.0))
        weights = np.clip(weights, 0.5, 2.0)
    return {
        str(score["model_id"]): float(weight)
        for score, weight in zip(novelty_scores, weights.tolist())
    }


def _select_model_contributions(
    *,
    rep: ModelRepresentation,
    source_layer_idx: int,
    mapped: np.ndarray,
    scales: np.ndarray,
    basis: np.ndarray | None,
    novelty_weight: float,
    max_accept: int,
    min_residual_ratio: float,
    min_model_score_ratio: float,
    residual_power: float,
    novelty_power: float,
) -> tuple[list[dict[str, object]], np.ndarray | None]:
    if mapped.size == 0 or max_accept <= 0:
        return [], basis
    remaining = set(range(mapped.shape[0]))
    selected: list[dict[str, object]] = []
    temp_basis = basis
    best_selected_score: float | None = None
    while remaining and len(selected) < max_accept:
        best_idx: int | None = None
        best_record: dict[str, object] | None = None
        best_score = -1.0
        for direction_idx in sorted(remaining):
            direction = mapped[direction_idx]
            residual_dir, residual_ratio = _project_residual(direction, temp_basis)
            if temp_basis is not None and temp_basis.size > 0 and residual_ratio < float(min_residual_ratio):
                continue
            kept = direction if temp_basis is None or temp_basis.size == 0 else residual_dir
            raw_scale = float(scales[direction_idx])
            candidate_score = raw_scale
            candidate_score *= max(float(residual_ratio), 1e-6) ** max(float(residual_power), 0.0)
            candidate_score *= max(float(novelty_weight), 1e-6) ** max(float(novelty_power), 0.0)
            if candidate_score > best_score:
                best_idx = direction_idx
                best_score = candidate_score
                best_record = {
                    "model_id": rep.model_id,
                    "source_layer_idx": int(source_layer_idx),
                    "direction_idx": int(direction_idx),
                    "raw_scale": raw_scale,
                    "residual_ratio": float(residual_ratio),
                    "novelty_weight": float(novelty_weight),
                    "candidate_score": float(candidate_score),
                    "effective_scale": float(candidate_score),
                    "vector": kept.astype(np.float32),
                }
        if best_idx is None or best_record is None:
            break
        if best_selected_score is None:
            best_selected_score = float(best_record["candidate_score"])
        elif best_score < best_selected_score * max(float(min_model_score_ratio), 0.0):
            break
        selected.append(best_record)
        temp_basis = _append_basis_vector(temp_basis, np.asarray(best_record["vector"], dtype=np.float32))
        remaining.remove(best_idx)
    return selected, temp_basis


def encode_model_representation(
    rep: ModelRepresentation,
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
        mapped = _map_directions(
            source_layer.directions,
            canonical_dim=canonical_dim,
            seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:cascade_embed"),
        )
        mapped = _normalize_rows(mapped)
        active = min(mapped.shape[0], max(int(top_k), 1))
        flat_dirs = np.zeros((top_k, canonical_dim), dtype=np.float32)
        flat_scales = np.zeros((top_k,), dtype=np.float32)
        if active > 0:
            flat_dirs[:active] = mapped[:active]
            if source_layer.scales is None or source_layer.scales.size == 0:
                scales = np.ones((active,), dtype=np.float32)
            else:
                scales = np.asarray(source_layer.scales, dtype=np.float32).reshape(-1)[:active]
                scales = np.abs(scales) / max(float(np.mean(np.abs(scales))), 1e-6)
            flat_scales[:active] = scales
        chunks.append(flat_dirs.reshape(-1))
        chunks.append(flat_scales)
    if rep.chunk_losses is not None and rep.chunk_losses.size > 0:
        losses = np.asarray(rep.chunk_losses, dtype=np.float32).reshape(-1)
        chunks.append(np.array([float(losses.mean()), float(losses.std())], dtype=np.float32))
    else:
        chunks.append(np.zeros((2,), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_model_zoo_matrix(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
) -> np.ndarray:
    if not representations:
        raise ValueError("representations must not be empty")
    rows = [
        encode_model_representation(
            rep,
            canonical_dim=canonical_dim,
            num_layers=num_layers,
            top_k=top_k,
        )
        for rep in representations
    ]
    matrix = np.stack(rows, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def pairwise_cosine_matrix(model_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(model_matrix, dtype=np.float32)
    return np.asarray(matrix @ matrix.T, dtype=np.float32)


def leave_one_out_novelty(model_matrix: np.ndarray, model_ids: list[str]) -> list[dict[str, float | str]]:
    matrix = np.asarray(model_matrix, dtype=np.float32)
    novelty: list[dict[str, float | str]] = []
    for idx, model_id in enumerate(model_ids):
        if matrix.shape[0] <= 1:
            residual_ratio = 1.0
        else:
            other_rows = np.delete(matrix, idx, axis=0)
            residual, residual_ratio = _project_residual(matrix[idx], other_rows)
            _ = residual
        novelty.append({"model_id": model_id, "leave_one_out_novelty": float(residual_ratio)})
    return novelty


def greedy_model_order(
    model_matrix: np.ndarray,
    model_ids: list[str],
) -> list[int]:
    matrix = np.asarray(model_matrix, dtype=np.float32)
    if matrix.shape[0] != len(model_ids):
        raise ValueError("model_matrix row count must match model_ids")
    if matrix.shape[0] <= 1:
        return list(range(matrix.shape[0]))
    cosine = pairwise_cosine_matrix(matrix)
    mean_cos = (cosine.sum(axis=1) - np.diag(cosine)) / max(matrix.shape[0] - 1, 1)
    selected = [int(np.argmax(mean_cos))]
    remaining = set(range(matrix.shape[0])) - set(selected)
    basis = matrix[selected].copy()
    while remaining:
        best_idx = None
        best_score = -1.0
        for idx in sorted(remaining):
            _residual, ratio = _project_residual(matrix[idx], basis)
            if ratio > best_score:
                best_idx = idx
                best_score = ratio
        assert best_idx is not None
        selected.append(best_idx)
        basis = _append_basis_vector(basis, matrix[best_idx])
        remaining.remove(best_idx)
    return selected


def merge_cascade_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    min_residual_ratio: float = 0.15,
    max_base_directions: int | None = None,
    max_residual_directions: int | None = None,
    min_model_score_ratio: float = 0.35,
    residual_power: float = 1.5,
    novelty_power: float = 1.0,
    model_order: list[str] | None = None,
) -> PlatonicGeometry:
    if not representations:
        raise ValueError("representations must not be empty")
    model_ids = [rep.model_id for rep in representations]
    model_matrix = build_model_zoo_matrix(
        representations,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
        top_k=max(1, min(top_k, 8)),
    )
    if model_order is None:
        order_indices = greedy_model_order(model_matrix, model_ids)
    else:
        index_by_id = {rep.model_id: idx for idx, rep in enumerate(representations)}
        missing = [model_id for model_id in model_order if model_id not in index_by_id]
        if missing:
            raise ValueError(f"Unknown model ids in model_order: {missing}")
        order_indices = [index_by_id[model_id] for model_id in model_order]
        order_indices.extend(idx for idx in range(len(representations)) if idx not in order_indices)
    ordered_reps = [representations[idx] for idx in order_indices]
    ordered_ids = [model_ids[idx] for idx in order_indices]
    novelty_scores = leave_one_out_novelty(model_matrix, model_ids)
    novelty_weight_by_model = _novelty_weight_map(novelty_scores)
    pairwise = pairwise_cosine_matrix(model_matrix)
    auto_base_directions = max(2, int(np.ceil(top_k * 0.75)))
    auto_residual_directions = max(1, int(np.ceil(top_k / max(len(representations), 1))))
    base_limit = max(1, int(max_base_directions if max_base_directions is not None else auto_base_directions))
    residual_limit = max(
        1,
        int(max_residual_directions if max_residual_directions is not None else auto_residual_directions),
    )

    layer_geometries: dict[int, LayerGeometry] = {}
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        accepted: list[dict[str, object]] = []
        basis = None
        for step_idx, rep in enumerate(ordered_reps):
            source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
            mapped = _map_directions(
                source_layer.directions,
                canonical_dim=canonical_dim,
                seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}:cascade_merge"),
            )
            mapped = _normalize_rows(mapped)
            scales = _normalized_scales(source_layer, count=mapped.shape[0])
            selected_records, basis = _select_model_contributions(
                rep=rep,
                source_layer_idx=source_layer_idx,
                mapped=mapped,
                scales=scales,
                basis=basis,
                novelty_weight=novelty_weight_by_model.get(rep.model_id, 1.0),
                max_accept=base_limit if step_idx == 0 else residual_limit,
                min_residual_ratio=min_residual_ratio,
                min_model_score_ratio=min_model_score_ratio,
                residual_power=residual_power,
                novelty_power=novelty_power,
            )
            accepted.extend(selected_records)
        if not accepted:
            raise RuntimeError(f"no accepted cascade directions for target layer {target_layer}")
        covariance = np.zeros((canonical_dim, canonical_dim), dtype=np.float64)
        contribution_by_model: dict[str, dict[str, float | int]] = {}
        for record in accepted:
            vector = np.asarray(record["vector"], dtype=np.float32)
            scale = float(record["effective_scale"])
            covariance += np.outer(vector, vector).astype(np.float64) * scale
            summary = contribution_by_model.setdefault(
                str(record["model_id"]),
                {
                    "accepted_direction_count": 0,
                    "total_effective_scale": 0.0,
                    "total_candidate_score": 0.0,
                    "mean_residual_ratio": 0.0,
                    "novelty_weight": float(record["novelty_weight"]),
                },
            )
            summary["accepted_direction_count"] = int(summary["accepted_direction_count"]) + 1
            summary["total_effective_scale"] = float(summary["total_effective_scale"]) + scale
            summary["total_candidate_score"] = float(summary["total_candidate_score"]) + float(record["candidate_score"])
            summary["mean_residual_ratio"] = float(summary["mean_residual_ratio"]) + float(record["residual_ratio"])
        for summary in contribution_by_model.values():
            count = max(int(summary["accepted_direction_count"]), 1)
            summary["mean_residual_ratio"] = float(summary["mean_residual_ratio"]) / count
        directions, scales = _dominant_directions_from_covariance(covariance, top_k=top_k)
        coactivation = directions @ covariance.astype(np.float32) @ directions.T
        importance = np.abs(directions.T) @ np.abs(scales)
        layer_geometries[target_layer] = LayerGeometry(
            relative_depth=relative_depth,
            directions=directions.astype(np.float32),
            scales=scales.astype(np.float32),
            covariance=covariance.astype(np.float32),
            coactivation=coactivation.astype(np.float32),
            importance=np.asarray(importance, dtype=np.float32),
            metadata={
                "selection_method": "cascade_sparse_residual_merge",
                "model_order": ordered_ids,
                "accepted_contribution_count": len(accepted),
                "max_base_directions": int(base_limit),
                "max_residual_directions": int(residual_limit),
                "contribution_by_model": contribution_by_model,
                "accepted_contributions": [
                    {
                        "model_id": str(record["model_id"]),
                        "source_layer_idx": int(record["source_layer_idx"]),
                        "direction_idx": int(record["direction_idx"]),
                        "raw_scale": float(record["raw_scale"]),
                        "residual_ratio": float(record["residual_ratio"]),
                        "novelty_weight": float(record["novelty_weight"]),
                        "candidate_score": float(record["candidate_score"]),
                        "effective_scale": float(record["effective_scale"]),
                    }
                    for record in sorted(accepted, key=lambda item: float(item["candidate_score"]), reverse=True)[:32]
                ],
            },
        )

    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=ordered_ids,
        metadata={
            "selection_method": "cascade_sparse_residual_merge",
            "model_order": ordered_ids,
            "min_residual_ratio": float(min_residual_ratio),
            "min_model_score_ratio": float(min_model_score_ratio),
            "residual_power": float(residual_power),
            "novelty_power": float(novelty_power),
            "max_base_directions": int(base_limit),
            "max_residual_directions": int(residual_limit),
            "model_novelty": novelty_scores,
            "novelty_weight_by_model": novelty_weight_by_model,
            "pairwise_cosine_matrix": pairwise.astype(np.float32).tolist(),
            "num_layers": int(num_layers),
            "top_k": int(top_k),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ModelRepresentation artifacts with a sequential residual cascade over the compressed representation outputs.",
    )
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-residual-ratio", type=float, default=0.15)
    parser.add_argument("--max-base-directions", type=int, default=0)
    parser.add_argument("--max-residual-directions", type=int, default=0)
    parser.add_argument("--min-model-score-ratio", type=float, default=0.35)
    parser.add_argument("--residual-power", type=float, default=1.5)
    parser.add_argument("--novelty-power", type=float, default=1.0)
    parser.add_argument(
        "--model-order",
        default="",
        help="Optional comma-separated model_id order. Unspecified models are appended after the provided order.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    model_order = [token.strip() for token in args.model_order.split(",") if token.strip()] or None
    geometry = merge_cascade_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        min_residual_ratio=args.min_residual_ratio,
        max_base_directions=args.max_base_directions if args.max_base_directions > 0 else None,
        max_residual_directions=args.max_residual_directions if args.max_residual_directions > 0 else None,
        min_model_score_ratio=args.min_model_score_ratio,
        residual_power=args.residual_power,
        novelty_power=args.novelty_power,
        model_order=model_order,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "selection_method": geometry.metadata["selection_method"],
                "model_order": geometry.metadata["model_order"],
                "min_residual_ratio": geometry.metadata["min_residual_ratio"],
                "max_base_directions": geometry.metadata["max_base_directions"],
                "max_residual_directions": geometry.metadata["max_residual_directions"],
                "source_models": geometry.source_models,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
