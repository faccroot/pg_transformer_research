#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.merge_hybrid_cascade_representations import (
        _champion_and_score,
        _normalize_rows,
        _order_rank_map,
    )
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
    from merge_hybrid_cascade_representations import _champion_and_score, _normalize_rows, _order_rank_map  # type: ignore[no-redef]
    from merge_cascade_representations import build_model_zoo_matrix, greedy_model_order, leave_one_out_novelty, _novelty_weight_map  # type: ignore[no-redef]
    from merge_weight_spectral_representations import _cluster_direction_records, _collect_layer_records  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _build_ranked_clusters(
    clusters: list[dict[str, object]],
    *,
    novelty_weight_by_model: dict[str, float],
    order_rank_by_model: dict[str, int],
    num_models: int,
    novelty_power: float,
    unique_bonus: float,
    shared_support_bonus: float,
    margin_power: float,
) -> list[dict[str, object]]:
    ranked_clusters: list[dict[str, object]] = []
    for cluster in clusters:
        champion, champion_score, cluster_score, support, second_best_score, winner_margin = _champion_and_score(
            cluster,
            novelty_weight_by_model=novelty_weight_by_model,
            order_rank_by_model=order_rank_by_model,
            num_models=num_models,
            novelty_power=novelty_power,
            unique_bonus=unique_bonus,
            shared_support_bonus=shared_support_bonus,
            margin_power=margin_power,
        )
        source_models = sorted({str(member["model_id"]) for member in cluster["members"]})
        ranked_clusters.append(
            {
                "cluster": cluster,
                "champion": champion,
                "champion_score": champion_score,
                "cluster_score": cluster_score,
                "support": support,
                "second_best_score": second_best_score,
                "winner_margin": winner_margin,
                "source_models": source_models,
                "is_shared": support > 1,
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
    return ranked_clusters


def _largest_remainder_quotas(
    capacity_by_model: dict[str, int],
    weight_by_model: dict[str, float],
    total_budget: int,
) -> dict[str, int]:
    if total_budget <= 0 or not capacity_by_model:
        return {model_id: 0 for model_id in capacity_by_model}
    eligible = {model_id: cap for model_id, cap in capacity_by_model.items() if cap > 0}
    if not eligible:
        return {model_id: 0 for model_id in capacity_by_model}
    raw_weights = {model_id: max(float(weight_by_model.get(model_id, 1.0)), 1e-6) for model_id in eligible}
    weight_sum = sum(raw_weights.values())
    quotas = {model_id: 0 for model_id in capacity_by_model}
    floor_parts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0
    for model_id, cap in eligible.items():
        exact = total_budget * (raw_weights[model_id] / max(weight_sum, 1e-6))
        floor_value = min(int(math.floor(exact)), cap)
        quotas[model_id] = floor_value
        floor_parts[model_id] = floor_value
        used += floor_value
        remainders.append((exact - floor_value, model_id))
    remaining = min(total_budget - used, sum(eligible.values()) - used)
    for _frac, model_id in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[model_id] >= eligible[model_id]:
            continue
        quotas[model_id] += 1
        remaining -= 1
    if remaining > 0:
        leftovers = sorted(
            eligible,
            key=lambda model_id: (raw_weights[model_id], -quotas[model_id], model_id),
            reverse=True,
        )
        for model_id in leftovers:
            if remaining <= 0:
                break
            room = eligible[model_id] - quotas[model_id]
            if room <= 0:
                continue
            take = min(room, remaining)
            quotas[model_id] += take
            remaining -= take
    return quotas


def _take_unique_clusters_with_quotas(
    unique_clusters: list[dict[str, object]],
    *,
    unique_budget: int,
    novelty_weight_by_model: dict[str, float],
) -> list[dict[str, object]]:
    if unique_budget <= 0 or not unique_clusters:
        return []
    by_model: dict[str, list[dict[str, object]]] = {}
    for item in unique_clusters:
        by_model.setdefault(str(item["champion"]["model_id"]), []).append(item)
    for items in by_model.values():
        items.sort(key=lambda item: (float(item["cluster_score"]), float(item["champion_score"])), reverse=True)
    quotas = _largest_remainder_quotas(
        {model_id: len(items) for model_id, items in by_model.items()},
        novelty_weight_by_model,
        unique_budget,
    )
    chosen: list[dict[str, object]] = []
    chosen_ids: set[int] = set()
    for model_id, quota in quotas.items():
        for item in by_model.get(model_id, [])[:quota]:
            chosen.append(item)
            chosen_ids.add(id(item))
    if len(chosen) < unique_budget:
        leftovers = [
            item
            for item in unique_clusters
            if id(item) not in chosen_ids
        ]
        leftovers.sort(key=lambda item: (float(item["cluster_score"]), float(item["champion_score"])), reverse=True)
        chosen.extend(leftovers[: max(unique_budget - len(chosen), 0)])
    chosen.sort(key=lambda item: (float(item["cluster_score"]), float(item["champion_score"])), reverse=True)
    return chosen[:unique_budget]


def _cluster_metadata(cluster_idx: int, item: dict[str, object]) -> dict[str, object]:
    return {
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
        "cluster_type": "shared" if bool(item["is_shared"]) else "unique",
    }


def merge_moonshot_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
    novelty_power: float = 1.5,
    unique_bonus: float = 1.25,
    shared_support_bonus: float = 0.75,
    margin_power: float = 1.25,
    shared_fraction: float = 0.5,
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
    novelty_weight_by_model = _novelty_weight_map(novelty_scores, temperature=1.0)
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
        ranked = _build_ranked_clusters(
            clusters,
            novelty_weight_by_model=novelty_weight_by_model,
            order_rank_by_model=order_rank_by_model,
            num_models=len(representations),
            novelty_power=novelty_power,
            unique_bonus=unique_bonus,
            shared_support_bonus=shared_support_bonus,
            margin_power=margin_power,
        )
        shared_clusters = [item for item in ranked if bool(item["is_shared"])]
        unique_clusters = [item for item in ranked if not bool(item["is_shared"])]

        desired_shared = min(len(shared_clusters), max(0, int(round(top_k * float(shared_fraction)))))
        desired_unique = min(len(unique_clusters), max(int(top_k) - desired_shared, 0))

        chosen_shared = shared_clusters[:desired_shared]
        chosen_unique = _take_unique_clusters_with_quotas(
            unique_clusters,
            unique_budget=desired_unique,
            novelty_weight_by_model=novelty_weight_by_model,
        )
        chosen_ids = {id(item) for item in chosen_shared + chosen_unique}
        chosen = chosen_shared + chosen_unique
        if len(chosen) < int(top_k):
            leftovers = [item for item in ranked if id(item) not in chosen_ids]
            chosen.extend(leftovers[: max(int(top_k) - len(chosen), 0)])
        chosen = chosen[: max(int(top_k), 1)]
        if not chosen:
            raise RuntimeError(f"No ranked moonshot clusters available for target layer {target_layer}")

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
                "selection_method": "moonshot_stratified_hardmax_merge",
                "model_order": [model_ids[idx] for idx in order_indices],
                "shared_budget": int(desired_shared),
                "unique_budget": int(desired_unique),
                "clusters": [_cluster_metadata(cluster_idx, item) for cluster_idx, item in enumerate(chosen)],
            },
        )

    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=[model_ids[idx] for idx in order_indices],
        metadata={
            "selection_method": "moonshot_stratified_hardmax_merge",
            "model_order": [model_ids[idx] for idx in order_indices],
            "model_novelty": novelty_scores,
            "novelty_weight_by_model": novelty_weight_by_model,
            "pairwise_cosine_matrix": pairwise.tolist(),
            "similarity_threshold": float(similarity_threshold),
            "novelty_power": float(novelty_power),
            "unique_bonus": float(unique_bonus),
            "shared_support_bonus": float(shared_support_bonus),
            "margin_power": float(margin_power),
            "shared_fraction": float(shared_fraction),
            "num_layers": int(num_layers),
            "top_k": int(top_k),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ModelRepresentation artifacts with a stratified moonshot merge that reserves capacity for robust shared clusters and novelty-weighted unique clusters.",
    )
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--novelty-power", type=float, default=1.5)
    parser.add_argument("--unique-bonus", type=float, default=1.25)
    parser.add_argument("--shared-support-bonus", type=float, default=0.75)
    parser.add_argument("--margin-power", type=float, default=1.25)
    parser.add_argument("--shared-fraction", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_moonshot_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        novelty_power=args.novelty_power,
        unique_bonus=args.unique_bonus,
        shared_support_bonus=args.shared_support_bonus,
        margin_power=args.margin_power,
        shared_fraction=args.shared_fraction,
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
