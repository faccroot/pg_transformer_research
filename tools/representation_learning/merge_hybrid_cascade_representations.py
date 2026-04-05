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
    from tools.representation_learning.merge_cascade_representations import (
        build_model_zoo_matrix,
        greedy_model_order,
        leave_one_out_novelty,
        _novelty_weight_map,
    )
    from tools.representation_learning.merge_weight_spectral_representations import (
        _cluster_direction_records,
        _collect_layer_records,
    )
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from merge_cascade_representations import build_model_zoo_matrix, greedy_model_order, leave_one_out_novelty, _novelty_weight_map  # type: ignore[no-redef]
    from merge_weight_spectral_representations import _cluster_direction_records, _collect_layer_records  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _order_rank_map(order_indices: list[int], model_ids: list[str]) -> dict[str, int]:
    return {model_ids[idx]: rank for rank, idx in enumerate(order_indices)}


def _champion_and_score(
    cluster: dict[str, object],
    *,
    novelty_weight_by_model: dict[str, float],
    order_rank_by_model: dict[str, int],
    num_models: int,
    novelty_power: float,
    unique_bonus: float,
    shared_support_bonus: float,
    margin_power: float,
) -> tuple[dict[str, object], float, float, int, float, float]:
    members = list(cluster["members"])
    source_models = sorted({str(member["model_id"]) for member in members})
    support = len(source_models)
    unique_fraction = 1.0 if num_models <= 1 else 1.0 - ((support - 1) / max(num_models - 1, 1))
    best_member: dict[str, object] | None = None
    best_member_score = -1.0
    second_best_score = 0.0
    for member in members:
        model_id = str(member["model_id"])
        novelty_weight = float(novelty_weight_by_model.get(model_id, 1.0))
        order_rank = int(order_rank_by_model.get(model_id, num_models - 1))
        residual_bonus = 1.0 + (float(unique_bonus) * unique_fraction)
        later_model_bonus = 1.0 + (0.15 * (order_rank / max(num_models - 1, 1)))
        member_score = float(member["scale"])
        member_score *= max(novelty_weight, 1e-6) ** max(float(novelty_power), 0.0)
        member_score *= residual_bonus
        member_score *= later_model_bonus
        if member_score > best_member_score:
            second_best_score = best_member_score if best_member_score > 0.0 else second_best_score
            best_member = member
            best_member_score = member_score
        elif member_score > second_best_score:
            second_best_score = member_score
    assert best_member is not None
    winner_margin = best_member_score / max(second_best_score, 1e-6) if len(members) > 1 else float("inf")
    margin_bonus = max(min(winner_margin, 8.0), 1.0) ** max(float(margin_power), 0.0)
    cluster_score = best_member_score * (
        1.0 + (float(shared_support_bonus) * ((support - 1) / max(num_models - 1, 1)))
    )
    cluster_score *= margin_bonus
    return (
        best_member,
        float(best_member_score),
        float(cluster_score),
        int(support),
        float(second_best_score),
        float(winner_margin),
    )


def merge_hybrid_cascade_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
    novelty_power: float = 1.25,
    unique_bonus: float = 0.75,
    shared_support_bonus: float = 0.25,
    margin_power: float = 0.0,
    selection_method: str = "hybrid_cascade_guided_spectral_merge",
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
    order_indices = greedy_model_order(model_matrix, model_ids)
    novelty_scores = leave_one_out_novelty(model_matrix, model_ids)
    novelty_weight_by_model = _novelty_weight_map(novelty_scores)
    order_rank_by_model = _order_rank_map(order_indices, model_ids)
    pairwise = np.asarray(model_matrix @ model_matrix.T, dtype=np.float32)

    layer_geometries: dict[int, LayerGeometry] = {}
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        records = _collect_layer_records(
            representations,
            relative_depth=relative_depth,
            canonical_dim=canonical_dim,
        )
        clusters = _cluster_direction_records(records, similarity_threshold=similarity_threshold)
        ranked_clusters: list[dict[str, object]] = []
        for cluster in clusters:
            champion, champion_score, cluster_score, support, second_best_score, winner_margin = _champion_and_score(
                cluster,
                novelty_weight_by_model=novelty_weight_by_model,
                order_rank_by_model=order_rank_by_model,
                num_models=len(representations),
                novelty_power=novelty_power,
                unique_bonus=unique_bonus,
                shared_support_bonus=shared_support_bonus,
                margin_power=margin_power,
            )
            ranked_clusters.append(
                {
                    "cluster": cluster,
                    "champion": champion,
                    "champion_score": champion_score,
                    "cluster_score": cluster_score,
                    "support": support,
                    "second_best_score": second_best_score,
                    "winner_margin": winner_margin,
                    "source_models": sorted({str(member["model_id"]) for member in cluster["members"]}),
                }
            )
        ranked_clusters.sort(
            key=lambda item: (
                float(item["cluster_score"]),
                float(item["champion_score"]),
                str(item["champion"]["model_id"]),
            ),
            reverse=True,
        )
        chosen = ranked_clusters[: max(int(top_k), 1)]
        if not chosen:
            raise RuntimeError(f"No ranked hybrid clusters available for target layer {target_layer}")
        directions = _normalize_rows(
            np.stack([np.asarray(item["champion"]["vector"], dtype=np.float32) for item in chosen], axis=0)
        )
        scales = np.asarray([float(item["champion_score"]) for item in chosen], dtype=np.float32)
        covariance = directions.T @ np.diag(scales.astype(np.float64)) @ directions
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
                "selection_method": selection_method,
                "model_order": [model_ids[idx] for idx in order_indices],
                "clusters": [
                    {
                        "cluster_idx": int(cluster_idx),
                        "champion_model": str(item["champion"]["model_id"]),
                        "champion_scale": float(item["champion"]["scale"]),
                        "champion_score": float(item["champion_score"]),
                        "runner_up_score": float(item["second_best_score"]),
                        "winner_margin": float(item["winner_margin"]),
                        "champion_direction_idx": int(item["champion"]["direction_idx"]),
                        "champion_source_layer_idx": int(item["champion"]["source_layer_idx"]),
                        "support": int(item["support"]),
                        "source_models": item["source_models"],
                        "max_similarity": float(item["cluster"]["max_similarity"]),
                    }
                    for cluster_idx, item in enumerate(chosen)
                ],
            },
        )

    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=[model_ids[idx] for idx in order_indices],
        metadata={
            "selection_method": selection_method,
            "model_order": [model_ids[idx] for idx in order_indices],
            "model_novelty": novelty_scores,
            "novelty_weight_by_model": novelty_weight_by_model,
            "pairwise_cosine_matrix": pairwise.tolist(),
            "similarity_threshold": float(similarity_threshold),
            "novelty_power": float(novelty_power),
            "unique_bonus": float(unique_bonus),
            "shared_support_bonus": float(shared_support_bonus),
            "margin_power": float(margin_power),
            "num_layers": int(num_layers),
            "top_k": int(top_k),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ModelRepresentation artifacts by combining spectral clustering with zoo-level novelty and ordering priors.",
    )
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--novelty-power", type=float, default=1.25)
    parser.add_argument("--unique-bonus", type=float, default=0.75)
    parser.add_argument("--shared-support-bonus", type=float, default=0.25)
    parser.add_argument("--margin-power", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_hybrid_cascade_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        novelty_power=args.novelty_power,
        unique_bonus=args.unique_bonus,
        shared_support_bonus=args.shared_support_bonus,
        margin_power=args.margin_power,
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
                "source_models": geometry.source_models,
                "top_k": geometry.metadata["top_k"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
