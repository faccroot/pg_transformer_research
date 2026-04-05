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
    from tools.representation_learning.schemas import SharedLatentGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import SharedLatentGeometry  # type: ignore[no-redef]


def summarize_shared_latent_geometry(artifact: SharedLatentGeometry) -> dict[str, object]:
    by_model: dict[str, list[float]] = {model_id: [] for model_id in artifact.source_models}
    best_by_layer: dict[str, str] = {}
    layer_summaries: dict[str, dict[str, object]] = {}
    for layer_idx, layer in sorted(artifact.layers.items()):
        residuals = {
            model_id: float(layer.metadata.get("view_residuals", {}).get(model_id, float("inf")))
            for model_id in artifact.source_models
        }
        for model_id, value in residuals.items():
            if np.isfinite(value):
                by_model.setdefault(model_id, []).append(value)
        best_model = min(residuals.items(), key=lambda item: (item[1], item[0]))[0]
        best_by_layer[str(layer_idx)] = best_model
        pairwise_cosines: dict[str, float] = {}
        model_ids = sorted(layer.aligned_latents)
        for idx, model_a in enumerate(model_ids):
            a = np.asarray(layer.aligned_latents[model_a], dtype=np.float32)
            for model_b in model_ids[idx + 1:]:
                b = np.asarray(layer.aligned_latents[model_b], dtype=np.float32)
                count = min(a.shape[0], b.shape[0])
                if count <= 0:
                    cosine = 0.0
                else:
                    dot = np.sum(a[:count] * b[:count], axis=1)
                    denom = np.linalg.norm(a[:count], axis=1) * np.linalg.norm(b[:count], axis=1)
                    cosine = float(np.mean(dot / np.clip(denom, 1e-8, None)))
                pairwise_cosines[f"{model_a}__{model_b}"] = cosine
        layer_summaries[str(layer_idx)] = {
            "relative_depth": float(layer.relative_depth),
            "chunk_count": len(layer.chunk_ids),
            "component_count": int(layer.shared_latents.shape[1]),
            "view_residuals": residuals,
            "pairwise_aligned_cosine_mean": pairwise_cosines,
        }
    model_summary = {
        model_id: {
            "mean_residual": float(np.mean(np.asarray(values, dtype=np.float32))) if values else None,
            "min_residual": float(np.min(np.asarray(values, dtype=np.float32))) if values else None,
            "max_residual": float(np.max(np.asarray(values, dtype=np.float32))) if values else None,
            "best_layer_count": int(sum(best_model == model_id for best_model in best_by_layer.values())),
        }
        for model_id, values in by_model.items()
    }
    return {
        "source_models": artifact.source_models,
        "latent_dim": int(artifact.latent_dim),
        "input_dim": int(artifact.input_dim),
        "layer_count": len(artifact.layers),
        "model_summary": model_summary,
        "best_model_by_layer": best_by_layer,
        "layers": layer_summaries,
        "metadata": artifact.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a SharedLatentGeometry artifact.")
    parser.add_argument("artifact", help="Input SharedLatentGeometry .npz artifact")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    artifact = SharedLatentGeometry.load(args.artifact)
    summary = summarize_shared_latent_geometry(artifact)
    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
