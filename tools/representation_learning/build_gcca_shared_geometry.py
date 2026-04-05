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
    from tools.representation_learning.build_ecology_training_data import _canonical_chunk_embeddings
    from tools.representation_learning.schemas import ModelRepresentation, SharedLatentGeometry, SharedLatentLayer
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_ecology_training_data import _canonical_chunk_embeddings  # type: ignore[no-redef]
    from schemas import ModelRepresentation, SharedLatentGeometry, SharedLatentLayer  # type: ignore[no-redef]


def _chunk_index_lookup(rep: ModelRepresentation) -> dict[str, int]:
    if not rep.chunk_ids:
        return {}
    return {str(chunk_id): idx for idx, chunk_id in enumerate(rep.chunk_ids)}


def _stabilize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    stabilized = matrix.copy()
    for col_idx in range(stabilized.shape[1]):
        column = stabilized[:, col_idx]
        if column.size == 0:
            continue
        pivot = int(np.argmax(np.abs(column)))
        if float(column[pivot]) < 0.0:
            stabilized[:, col_idx] *= -1.0
    return stabilized.astype(np.float32)


def _shared_chunk_ids(
    source_models: list[str],
    *,
    chunk_embeddings: dict[str, dict[int, dict[str, object]]],
    chunk_indices: dict[str, dict[str, int]],
    target_layer: int,
) -> list[str]:
    shared: set[str] | None = None
    for model_id in source_models:
        payload = chunk_embeddings.get(model_id, {}).get(int(target_layer))
        if payload is None:
            return []
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        available = {
            chunk_id
            for chunk_id, idx in chunk_indices.get(model_id, {}).items()
            if idx < matrix.shape[0]
        }
        shared = available if shared is None else shared & available
        if not shared:
            return []
    return sorted(shared) if shared is not None else []


def _view_matrix(
    rep: ModelRepresentation,
    *,
    chunk_embeddings: dict[int, dict[str, object]],
    chunk_index: dict[str, int],
    target_layer: int,
    shared_chunk_ids: list[str],
) -> np.ndarray:
    payload = chunk_embeddings[int(target_layer)]
    matrix = np.asarray(payload["matrix"], dtype=np.float32)
    rows = [matrix[int(chunk_index[chunk_id])] for chunk_id in shared_chunk_ids]
    return np.asarray(rows, dtype=np.float32)


def _standardize_view(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=0).astype(np.float32) if matrix.size else np.zeros((matrix.shape[1],), dtype=np.float32)
    centered = matrix - mean
    return centered.astype(np.float32), mean.astype(np.float32)


def _gcca_view_basis(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must have shape [num_samples, dim], got {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    u, singular_values, _vt = np.linalg.svd(matrix, full_matrices=False)
    keep = singular_values > float(rcond)
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(u[:, keep], dtype=np.float32), np.asarray(singular_values[keep], dtype=np.float32)


def _gcca_maxvar(
    views: dict[str, np.ndarray],
    *,
    latent_dim: int,
    rcond: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    if not views:
        raise ValueError("views must not be empty")
    sample_count = next(iter(views.values())).shape[0]
    if sample_count <= 0:
        raise ValueError("views must contain at least one sample")
    bases: dict[str, np.ndarray] = {}
    kept_singular_values: dict[str, list[float]] = {}
    accumulator = np.zeros((sample_count, sample_count), dtype=np.float64)
    for model_id, matrix in views.items():
        basis, singular_values = _gcca_view_basis(matrix, rcond=rcond)
        bases[model_id] = basis
        kept_singular_values[model_id] = [float(value) for value in singular_values.tolist()]
        if basis.size > 0:
            accumulator += basis.astype(np.float64) @ basis.T.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    order = np.argsort(eigenvalues)[::-1]
    component_count = min(int(latent_dim), int(np.count_nonzero(eigenvalues > 1e-8)), sample_count)
    if component_count <= 0:
        raise ValueError("No valid GCCA components found")
    shared = np.asarray(eigenvectors[:, order[:component_count]], dtype=np.float32)
    shared = _stabilize_columns(shared)
    projections: dict[str, np.ndarray] = {}
    aligned: dict[str, np.ndarray] = {}
    residuals: dict[str, float] = {}
    for model_id, matrix in views.items():
        projection, *_ = np.linalg.lstsq(matrix, shared, rcond=rcond)
        projection = np.asarray(projection, dtype=np.float32)
        aligned_latents = np.asarray(matrix @ projection, dtype=np.float32)
        projections[model_id] = projection
        aligned[model_id] = aligned_latents
        denom = max(float(np.linalg.norm(shared)), 1e-8)
        residuals[model_id] = float(np.linalg.norm(aligned_latents - shared) / denom)
    metadata = {
        "method": "gcca_maxvar_linear",
        "component_count": int(component_count),
        "eigenvalues": [float(value) for value in eigenvalues[order[:component_count]].tolist()],
        "view_kept_singular_values": kept_singular_values,
        "view_residuals": residuals,
    }
    return shared, projections, aligned, metadata


def build_gcca_shared_geometry(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    latent_dim: int,
    num_layers: int,
    min_shared_chunks: int = 16,
    rcond: float = 1e-5,
) -> SharedLatentGeometry:
    if not representations:
        raise ValueError("representations must not be empty")
    source_models = [rep.model_id for rep in representations]
    chunk_indices = {rep.model_id: _chunk_index_lookup(rep) for rep in representations}
    chunk_embeddings = {
        rep.model_id: _canonical_chunk_embeddings(rep, canonical_dim=canonical_dim, num_layers=num_layers)
        for rep in representations
    }
    layers: dict[int, SharedLatentLayer] = {}
    for target_layer in range(1, num_layers + 1):
        shared_chunk_ids = _shared_chunk_ids(
            source_models,
            chunk_embeddings=chunk_embeddings,
            chunk_indices=chunk_indices,
            target_layer=target_layer,
        )
        if len(shared_chunk_ids) < int(min_shared_chunks):
            continue
        views_centered: dict[str, np.ndarray] = {}
        model_means: dict[str, np.ndarray] = {}
        relative_depth = float(target_layer / max(num_layers, 1))
        for rep in representations:
            raw_matrix = _view_matrix(
                rep,
                chunk_embeddings=chunk_embeddings[rep.model_id],
                chunk_index=chunk_indices[rep.model_id],
                target_layer=target_layer,
                shared_chunk_ids=shared_chunk_ids,
            )
            centered, mean = _standardize_view(raw_matrix)
            views_centered[rep.model_id] = centered
            model_means[rep.model_id] = mean
        shared_latents, projections, aligned_latents, layer_metadata = _gcca_maxvar(
            views_centered,
            latent_dim=latent_dim,
            rcond=rcond,
        )
        layers[int(target_layer)] = SharedLatentLayer(
            relative_depth=relative_depth,
            chunk_ids=shared_chunk_ids,
            shared_latents=shared_latents,
            model_projections=projections,
            model_means=model_means,
            aligned_latents=aligned_latents,
            metadata=layer_metadata,
        )
    if not layers:
        raise ValueError("No layers satisfied the shared-chunk requirement")
    return SharedLatentGeometry(
        latent_dim=int(latent_dim),
        input_dim=int(canonical_dim),
        source_models=source_models,
        layers=layers,
        metadata={
            "method": "gcca_maxvar_linear",
            "canonical_dim": int(canonical_dim),
            "latent_dim": int(latent_dim),
            "num_layers": int(num_layers),
            "min_shared_chunks": int(min_shared_chunks),
            "rcond": float(rcond),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a multi-view GCCA shared-latent geometry from ModelRepresentation artifacts.")
    parser.add_argument("output", help="Output .npz path for the SharedLatentGeometry artifact")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation artifacts")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Canonical input dimension used to map each source view")
    parser.add_argument("--latent-dim", type=int, default=16, help="Shared latent dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins to evaluate")
    parser.add_argument("--min-shared-chunks", type=int, default=16, help="Minimum chunk intersection required for a layer")
    parser.add_argument("--rcond", type=float, default=1e-5, help="SVD / least-squares cutoff")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    artifact = build_gcca_shared_geometry(
        representations,
        canonical_dim=args.canonical_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        min_shared_chunks=args.min_shared_chunks,
        rcond=args.rcond,
    )
    output_path = Path(args.output).resolve()
    artifact.save(output_path)
    summary = {
        "output": str(output_path),
        "source_models": artifact.source_models,
        "input_dim": int(artifact.input_dim),
        "latent_dim": int(artifact.latent_dim),
        "layer_indices": sorted(artifact.layers),
        "layers": {
            str(layer_idx): {
                "relative_depth": float(layer.relative_depth),
                "chunk_count": len(layer.chunk_ids),
                "component_count": int(layer.shared_latents.shape[1]),
                "view_residuals": layer.metadata.get("view_residuals", {}),
            }
            for layer_idx, layer in artifact.layers.items()
        },
        "metadata": artifact.metadata,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
