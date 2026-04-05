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
    from tools.representation_learning.build_platonic_geometry import _concept_layers, _map_directions, _nearest_layer, _stable_seed
    from tools.representation_learning.compare_model_representations import _subspace_overlap
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_platonic_geometry import _concept_layers, _map_directions, _nearest_layer, _stable_seed  # type: ignore[no-redef]
    from compare_model_representations import _subspace_overlap  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _prepare_directions(rep: ModelRepresentation, layer_idx: int, canonical_dim: int | None) -> np.ndarray:
    directions = np.asarray(rep.layer_geometries[int(layer_idx)].directions, dtype=np.float32)
    if canonical_dim is None or directions.shape[1] == canonical_dim:
        return _normalize_rows(directions)
    return _normalize_rows(
        _map_directions(
            directions,
            canonical_dim=canonical_dim,
            seed=_stable_seed(f"{rep.model_id}:{rep.metadata.get('extraction_method', 'unknown')}:{layer_idx}:{canonical_dim}"),
        )
    )


def _prepare_single_direction(
    vector: np.ndarray,
    *,
    model_id: str,
    label: str,
    canonical_dim: int | None,
) -> np.ndarray:
    direction = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if canonical_dim is None or direction.shape[1] == canonical_dim:
        return _normalize_rows(direction)[0]
    mapped = _map_directions(direction, canonical_dim=canonical_dim, seed=_stable_seed(f"{model_id}:{label}:{canonical_dim}"))
    return _normalize_rows(mapped)[0]


def build_view_comparison_report(
    primary: ModelRepresentation,
    secondary: ModelRepresentation,
    *,
    canonical_dim: int | None = None,
    num_layers: int | None = None,
) -> dict[str, object]:
    if canonical_dim is None and primary.hidden_dim != secondary.hidden_dim:
        canonical_dim = min(max(primary.hidden_dim, secondary.hidden_dim, 1), 64)
    if num_layers is None:
        num_layers = max(primary.num_layers, secondary.num_layers, 1)

    layers_payload: list[dict[str, object]] = []
    overlaps: list[float] = []
    mean_max_cosines: list[float] = []
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        primary_layer_idx, _primary_layer = _nearest_layer(primary, relative_depth)
        secondary_layer_idx, _secondary_layer = _nearest_layer(secondary, relative_depth)
        primary_dirs = _prepare_directions(primary, primary_layer_idx, canonical_dim)
        secondary_dirs = _prepare_directions(secondary, secondary_layer_idx, canonical_dim)
        cosine_matrix = np.abs(primary_dirs @ secondary_dirs.T)
        mean_max_cosine = float(cosine_matrix.max(axis=1).mean()) if cosine_matrix.size else 0.0
        overlap = _subspace_overlap(primary_dirs, secondary_dirs)
        overlaps.append(overlap)
        mean_max_cosines.append(mean_max_cosine)
        layers_payload.append(
            {
                "target_layer": target_layer,
                "relative_depth": relative_depth,
                "primary_layer_idx": int(primary_layer_idx),
                "secondary_layer_idx": int(secondary_layer_idx),
                "subspace_overlap": overlap,
                "mean_max_abs_cosine": mean_max_cosine,
                "best_match_pairs": [
                    {
                        "primary_direction_idx": int(direction_idx),
                        "secondary_direction_idx": int(np.argmax(cosine_matrix[direction_idx])) if cosine_matrix.size else 0,
                        "abs_cosine": float(cosine_matrix[direction_idx].max()) if cosine_matrix.size else 0.0,
                    }
                    for direction_idx in range(min(primary_dirs.shape[0], 4))
                ],
            }
        )

    concepts_payload: dict[str, object] = {}
    concept_alignments: list[float] = []
    for concept_name, profile in sorted(primary.concept_profiles.items()):
        if not isinstance(profile, dict):
            continue
        layer_entries = _concept_layers(profile)
        if not layer_entries:
            continue
        concept_layers_payload: list[dict[str, object]] = []
        local_alignments: list[float] = []
        for source_layer_idx, payload in layer_entries:
            relative_depth = float(payload.get("relative_depth", source_layer_idx / max(primary.num_layers, 1)))
            secondary_layer_idx, _secondary_layer = _nearest_layer(secondary, relative_depth)
            concept_direction = _prepare_single_direction(
                np.asarray(payload.get("direction", []), dtype=np.float32),
                model_id=primary.model_id,
                label=f"{concept_name}:{source_layer_idx}",
                canonical_dim=canonical_dim,
            )
            secondary_dirs = _prepare_directions(secondary, secondary_layer_idx, canonical_dim)
            cosines = np.abs(secondary_dirs @ concept_direction)
            best_idx = int(np.argmax(cosines)) if cosines.size else 0
            best_alignment = float(cosines[best_idx]) if cosines.size else 0.0
            local_alignments.append(best_alignment)
            concept_layers_payload.append(
                {
                    "primary_layer_idx": int(source_layer_idx),
                    "secondary_layer_idx": int(secondary_layer_idx),
                    "relative_depth": relative_depth,
                    "best_direction_idx": best_idx,
                    "max_abs_cosine": best_alignment,
                    "best_direction_scale": float(
                        secondary.layer_geometries[int(secondary_layer_idx)].scales[best_idx]
                    ) if secondary.layer_geometries[int(secondary_layer_idx)].scales is not None else None,
                }
            )
        mean_alignment = float(np.mean(local_alignments)) if local_alignments else 0.0
        concept_alignments.append(mean_alignment)
        concepts_payload[concept_name] = {
            "description": str(profile.get("description", "")),
            "mean_alignment": mean_alignment,
            "layers": concept_layers_payload,
        }

    return {
        "primary_model_id": primary.model_id,
        "secondary_model_id": secondary.model_id,
        "primary_extraction_method": str(primary.metadata.get("extraction_method", "unknown")),
        "secondary_extraction_method": str(secondary.metadata.get("extraction_method", "unknown")),
        "canonical_dim": canonical_dim,
        "num_layers": num_layers,
        "layers": layers_payload,
        "concepts": concepts_payload,
        "summary": {
            "mean_subspace_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
            "mean_max_abs_cosine": float(np.mean(mean_max_cosines)) if mean_max_cosines else 0.0,
            "mean_concept_alignment": float(np.mean(concept_alignments)) if concept_alignments else 0.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two representation views, for example activation-space vs weight-space, and emit a JSON report.")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("primary", help="Primary ModelRepresentation .npz path")
    parser.add_argument("secondary", help="Secondary ModelRepresentation .npz path")
    parser.add_argument("--canonical-dim", type=int, default=0, help="Shared dimension when hidden sizes differ; 0 means direct comparison when possible")
    parser.add_argument("--num-layers", type=int, default=0, help="Number of relative-depth bins; 0 uses the max source depth")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    primary = ModelRepresentation.load(args.primary)
    secondary = ModelRepresentation.load(args.secondary)
    report = build_view_comparison_report(
        primary,
        secondary,
        canonical_dim=None if args.canonical_dim <= 0 else args.canonical_dim,
        num_layers=None if args.num_layers <= 0 else args.num_layers,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "primary_extraction_method": report["primary_extraction_method"],
                "secondary_extraction_method": report["secondary_extraction_method"],
                "mean_subspace_overlap": report["summary"]["mean_subspace_overlap"],
                "mean_concept_alignment": report["summary"]["mean_concept_alignment"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
