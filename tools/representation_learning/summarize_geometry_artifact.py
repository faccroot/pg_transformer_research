#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.representation_learning.schemas import PlatonicGeometry


def _update_model_summary(
    bucket: dict[str, dict[str, float | int]],
    *,
    model_id: str,
    count: int = 0,
    scale: float = 0.0,
) -> None:
    summary = bucket.setdefault(model_id, {"count": 0, "total_scale": 0.0})
    summary["count"] = int(summary["count"]) + int(count)
    summary["total_scale"] = float(summary["total_scale"]) + float(scale)


def summarize_geometry(geometry: PlatonicGeometry) -> dict[str, object]:
    ownership_by_model: dict[str, dict[str, float | int]] = {}
    per_layer: dict[str, object] = {}
    total_cluster_like = 0
    total_shared = 0
    total_unique = 0
    total_contribution_like = 0

    for layer_idx, layer in sorted(geometry.layer_geometries.items()):
        layer_summary: dict[str, object] = {
            "relative_depth": float(layer.relative_depth),
            "direction_count": int(layer.directions.shape[0]),
            "selection_method": str(layer.metadata.get("selection_method", geometry.metadata.get("selection_method", "unknown"))),
        }
        if layer.scales is not None:
            layer_summary["total_direction_scale"] = float(layer.scales.sum())

        if isinstance(layer.metadata.get("clusters"), list):
            clusters = layer.metadata["clusters"]
            assert isinstance(clusters, list)
            total_cluster_like += len(clusters)
            layer_model_totals: dict[str, dict[str, float | int]] = {}
            shared = 0
            unique = 0
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue
                model_id = str(cluster.get("champion_model", "unknown"))
                scale = float(cluster.get("champion_score", cluster.get("champion_scale", 0.0)))
                support = int(cluster.get("support", 1))
                if support > 1:
                    shared += 1
                else:
                    unique += 1
                _update_model_summary(ownership_by_model, model_id=model_id, count=1, scale=scale)
                _update_model_summary(layer_model_totals, model_id=model_id, count=1, scale=scale)
            total_shared += shared
            total_unique += unique
            layer_summary["shared_cluster_count"] = shared
            layer_summary["unique_cluster_count"] = unique
            layer_summary["ownership_by_model"] = layer_model_totals

        if isinstance(layer.metadata.get("contribution_by_model"), dict):
            contribution_by_model = layer.metadata["contribution_by_model"]
            assert isinstance(contribution_by_model, dict)
            for model_id, model_summary in contribution_by_model.items():
                if not isinstance(model_summary, dict):
                    continue
                count = int(model_summary.get("accepted_direction_count", 0))
                scale = float(
                    model_summary.get(
                        "total_candidate_score",
                        model_summary.get("total_effective_scale", 0.0),
                    )
                )
                total_contribution_like += count
                _update_model_summary(
                    ownership_by_model,
                    model_id=str(model_id),
                    count=count,
                    scale=scale,
                )
            layer_summary["contribution_by_model"] = contribution_by_model

        per_layer[str(layer_idx)] = layer_summary

    return {
        "selection_method": str(geometry.metadata.get("selection_method", "unknown")),
        "source_models": geometry.source_models,
        "canonical_dim": int(geometry.canonical_dim),
        "num_layers": int(len(geometry.layer_geometries)),
        "ownership_by_model": ownership_by_model,
        "total_cluster_like_entries": total_cluster_like,
        "total_shared_cluster_count": total_shared,
        "total_unique_cluster_count": total_unique,
        "total_contribution_like_entries": total_contribution_like,
        "per_layer": per_layer,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a PlatonicGeometry artifact by source ownership and layer structure.")
    parser.add_argument("geometry", help="Path to a PlatonicGeometry .npz artifact")
    parser.add_argument("--output", default="", help="Optional path to write summary JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geometry = PlatonicGeometry.load(Path(args.geometry).resolve())
    summary = summarize_geometry(geometry)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
