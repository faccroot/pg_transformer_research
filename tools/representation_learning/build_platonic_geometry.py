#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def _stable_seed(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)


def _orthonormal_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    q, _r = np.linalg.qr(np.asarray(matrix, dtype=np.float32).T, mode="reduced")
    return np.asarray(q.T, dtype=np.float32)


def _map_directions(directions: np.ndarray, canonical_dim: int, seed: int) -> np.ndarray:
    source = np.asarray(directions, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"directions must have shape [k, dim], got {source.shape}")
    if source.shape[0] == 0:
        return np.zeros((0, canonical_dim), dtype=np.float32)
    if source.shape[1] == canonical_dim:
        return _orthonormal_rows(source)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((source.shape[1], canonical_dim), dtype=np.float32) / max(math.sqrt(source.shape[1]), 1.0)
    return _orthonormal_rows(source @ projection)


def _dominant_directions_from_covariance(covariance: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    top_k = min(max(int(top_k), 1), int(covariance.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:top_k]
    values = np.clip(eigenvalues[order], 0.0, None).astype(np.float32)
    vectors = eigenvectors[:, order].T.astype(np.float32)
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
    return vectors, values


def _nearest_layer(rep: ModelRepresentation, relative_depth: float) -> tuple[int, LayerGeometry]:
    best_idx = min(rep.layer_geometries, key=lambda idx: abs(rep.layer_geometries[idx].relative_depth - relative_depth))
    return best_idx, rep.layer_geometries[best_idx]


def _concept_layers(profile: dict[str, object]) -> list[tuple[int, dict[str, object]]]:
    raw_layers = profile.get("layers", {})
    if not isinstance(raw_layers, dict):
        return []
    parsed: list[tuple[int, dict[str, object]]] = []
    for layer_idx, payload in raw_layers.items():
        if isinstance(payload, dict):
            parsed.append((int(layer_idx), payload))
    return sorted(parsed, key=lambda item: item[0])


def _nearest_concept_layer(profile: dict[str, object], relative_depth: float) -> tuple[int, dict[str, object]] | None:
    candidates = _concept_layers(profile)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: abs(float(item[1].get("relative_depth", 0.0)) - relative_depth),
    )


def _concept_sharpness(profile: dict[str, object]) -> float:
    return float(profile.get("sharpness", 0.0))


def build_platonic_geometry(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
) -> PlatonicGeometry:
    if not representations:
        raise ValueError("representations must not be empty")
    layer_geometries: dict[int, LayerGeometry] = {}
    for target_layer in range(1, num_layers + 1):
        relative_depth = target_layer / max(num_layers, 1)
        covariance_acc = np.zeros((canonical_dim, canonical_dim), dtype=np.float64)
        importance_acc = np.zeros((canonical_dim,), dtype=np.float64)
        coactivation_terms: list[np.ndarray] = []
        sources: list[dict[str, object]] = []
        for rep in representations:
            source_layer_idx, source_layer = _nearest_layer(rep, relative_depth)
            mapped = _map_directions(
                source_layer.directions,
                canonical_dim=canonical_dim,
                seed=_stable_seed(f"{rep.model_id}:{source_layer_idx}:{canonical_dim}"),
            )
            active = mapped.shape[0]
            if active <= 0:
                continue
            if source_layer.scales is None:
                scales = np.ones((active,), dtype=np.float32)
            else:
                scales = np.asarray(source_layer.scales, dtype=np.float32)[:active]
                scales = scales / max(float(np.mean(np.abs(scales))), 1e-6)
            covariance_acc += mapped.T @ np.diag(scales.astype(np.float64)) @ mapped
            coactivation_terms.append(mapped @ mapped.T)
            if source_layer.importance is not None:
                mapped_importance = np.abs(mapped.T) @ np.ones((active,), dtype=np.float32)
                importance_acc += mapped_importance.astype(np.float64)
            sources.append(
                {
                    "model_id": rep.model_id,
                    "source_layer_idx": source_layer_idx,
                    "source_relative_depth": source_layer.relative_depth,
                }
            )
        directions, scales = _dominant_directions_from_covariance(covariance_acc, top_k=top_k)
        coactivation = directions @ covariance_acc.astype(np.float32) @ directions.T
        layer_geometries[target_layer] = LayerGeometry(
            relative_depth=relative_depth,
            directions=directions,
            scales=scales,
            covariance=covariance_acc.astype(np.float32),
            coactivation=coactivation.astype(np.float32),
            importance=importance_acc.astype(np.float32),
            metadata={"sources": sources},
        )
    frontier_floor = None
    chunk_ids = None
    if all(rep.chunk_losses is not None for rep in representations):
        base_ids = representations[0].chunk_ids
        if base_ids and all(rep.chunk_ids == base_ids for rep in representations):
            stacked = np.stack([np.asarray(rep.chunk_losses, dtype=np.float32) for rep in representations], axis=0)
            frontier_floor = stacked.mean(axis=0).astype(np.float32)
            chunk_ids = base_ids

    concept_profiles: dict[str, object] = {}
    concept_names = sorted({str(name) for rep in representations for name in rep.concept_profiles})
    for concept_name in concept_names:
        model_entries: list[dict[str, object]] = []
        layer_entries: dict[str, object] = {}
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
            candidates: list[dict[str, object]] = []
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
                )[0]
                candidates.append(
                    {
                        "model_id": rep.model_id,
                        "source_layer_idx": int(source_layer_idx),
                        "source_relative_depth": float(source_payload.get("relative_depth", 0.0)),
                        "sharpness": _concept_sharpness(profile),
                        "layer_score": float(source_payload.get("layer_score", 0.0)),
                        "direction": mapped.tolist(),
                    }
                )
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda item: (float(item["layer_score"]), float(item["sharpness"]), str(item["model_id"])),
            )
            layer_entries[str(target_layer)] = best

        if model_entries or layer_entries:
            best_model = max(
                model_entries,
                key=lambda item: (float(item["sharpness"]), float(item["best_layer_score"]), str(item["model_id"])),
                default=None,
            )
            concept_profiles[concept_name] = {
                "best_model": None if best_model is None else best_model["model_id"],
                "description": description,
                "models": model_entries,
                "layers": layer_entries,
            }
    return PlatonicGeometry(
        canonical_dim=canonical_dim,
        layer_geometries=layer_geometries,
        source_models=[rep.model_id for rep in representations],
        frontier_floor=frontier_floor,
        concept_profiles=concept_profiles,
        metadata={
            "canonical_dim": canonical_dim,
            "num_layers": num_layers,
            "top_k": top_k,
            "source_chunk_ids": chunk_ids,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine one or more ModelRepresentation artifacts into a PlatonicGeometry artifact.")
    parser.add_argument("output", help="Output .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Shared latent dimension for the combined geometry")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins in the combined geometry")
    parser.add_argument("--top-k", type=int, default=16, help="Directions to retain per combined layer")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = build_platonic_geometry(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
    )
    output_path = Path(args.output).resolve()
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "canonical_dim": args.canonical_dim,
                "num_layers": args.num_layers,
                "output": str(output_path),
                "source_models": geometry.source_models,
                "top_k": args.top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
