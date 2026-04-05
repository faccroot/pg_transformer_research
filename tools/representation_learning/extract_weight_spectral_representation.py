#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation  # type: ignore[no-redef]


@dataclass(frozen=True)
class WeightMatrixSpec:
    module_name: str
    weight: np.ndarray
    side: str

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.shape}")
        if self.side not in {"input", "output"}:
            raise ValueError(f"side must be 'input' or 'output', got {self.side!r}")
        object.__setattr__(self, "weight", weight)


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if not spec:
        return list(range(1, num_layers + 1))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value > num_layers:
            raise ValueError(f"Layer {value} is outside [1, {num_layers}]")
        layers.append(value)
    if not layers:
        raise ValueError("No valid layers parsed from --layers")
    return sorted(set(layers))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _iter_transformer_blocks(model: Any) -> list[Any]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "transformer", None), "layers", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            blocks = list(candidate)
        except TypeError:
            continue
        if blocks:
            return blocks
    raise RuntimeError("Unable to discover transformer blocks on the loaded model")


def discover_layer_weight_specs(model: Any, *, hidden_dim: int, layers: list[int]) -> dict[int, list[WeightMatrixSpec]]:
    try:
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to inspect raw model weights") from exc

    blocks = _iter_transformer_blocks(model)
    layer_specs: dict[int, list[WeightMatrixSpec]] = {}
    for layer_idx in layers:
        block = blocks[layer_idx - 1]
        specs: list[WeightMatrixSpec] = []
        for module_name, module in block.named_modules():
            if not module_name:
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not hasattr(weight, "detach"):
                continue
            if not isinstance(module, nn.Linear) and getattr(weight, "ndim", None) != 2:
                continue
            weight_np = weight.detach().to(dtype=getattr(weight, "dtype", None)).cpu().float().numpy()
            if weight_np.ndim != 2:
                continue
            out_features, in_features = int(weight_np.shape[0]), int(weight_np.shape[1])
            if in_features == hidden_dim:
                specs.append(WeightMatrixSpec(module_name=module_name, weight=weight_np, side="input"))
            elif out_features == hidden_dim:
                specs.append(WeightMatrixSpec(module_name=module_name, weight=weight_np, side="output"))
        if not specs:
            raise RuntimeError(f"Layer {layer_idx} did not expose any hidden-space linear weights")
        layer_specs[layer_idx] = specs
    return layer_specs


def spectral_directions_from_weight(
    spec: WeightMatrixSpec,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weight = np.nan_to_num(np.asarray(spec.weight, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    u, singular_values, vh = np.linalg.svd(weight, full_matrices=False)
    keep = min(max(int(top_k), 1), int(singular_values.shape[0]))
    scales = singular_values[:keep].astype(np.float32)
    if spec.side == "input":
        directions = vh[:keep].astype(np.float32)
    else:
        directions = u[:, :keep].T.astype(np.float32)
    directions = _normalize_rows(directions)
    return directions, scales, {
        "module_name": spec.module_name,
        "side": spec.side,
        "input_dim": int(spec.weight.shape[1]),
        "output_dim": int(spec.weight.shape[0]),
        "max_singular_value": float(scales[0]) if scales.size else 0.0,
        "retained_rank": int(keep),
    }


def build_weight_spectral_representation_from_layer_weights(
    *,
    model_id: str,
    architecture_family: str,
    num_parameters: int,
    hidden_dim: int,
    num_layers: int,
    layer_weights: dict[int, list[WeightMatrixSpec]],
    top_k: int,
    top_k_per_module: int,
    metadata: dict[str, Any] | None = None,
) -> ModelRepresentation:
    layer_geometries: dict[int, LayerGeometry] = {}
    for layer_idx in sorted(layer_weights):
        candidates: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        direction_sources: list[dict[str, Any]] = []
        module_summaries: list[dict[str, Any]] = []
        for spec in layer_weights[layer_idx]:
            directions, singular_values, module_summary = spectral_directions_from_weight(spec, top_k=top_k_per_module)
            module_summaries.append(module_summary)
            for local_rank, singular_value in enumerate(singular_values.tolist()):
                direction_sources.append(
                    {
                        "module_name": spec.module_name,
                        "side": spec.side,
                        "source_rank": int(local_rank),
                        "singular_value": float(singular_value),
                    }
                )
            candidates.append(directions)
            scales.append(singular_values)
        if not candidates:
            raise RuntimeError(f"Layer {layer_idx} has no spectral candidates")
        stacked_directions = np.concatenate(candidates, axis=0)
        stacked_scales = np.concatenate(scales, axis=0)
        order = np.argsort(stacked_scales)[::-1][: min(max(int(top_k), 1), int(stacked_scales.shape[0]))]
        selected_directions = _normalize_rows(stacked_directions[order].astype(np.float32))
        selected_scales = np.asarray(stacked_scales[order], dtype=np.float32)
        covariance = selected_directions.T @ np.diag(selected_scales.astype(np.float64)) @ selected_directions
        importance = np.abs(selected_directions.T) @ np.abs(selected_scales)
        coactivation = selected_directions @ covariance.astype(np.float32) @ selected_directions.T
        layer_geometries[int(layer_idx)] = LayerGeometry(
            relative_depth=float(layer_idx / max(num_layers, 1)),
            directions=selected_directions,
            scales=selected_scales,
            covariance=covariance.astype(np.float32),
            coactivation=coactivation.astype(np.float32),
            importance=np.asarray(importance, dtype=np.float32),
            metadata={
                "direction_sources": [direction_sources[int(index)] for index in order.tolist()],
                "extraction_method": "weight_svd",
                "module_summaries": module_summaries,
            },
        )
    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "extraction_method": "weight_svd",
            "top_k": int(top_k),
            "top_k_per_module": int(top_k_per_module),
            "layer_indices": sorted(int(layer_idx) for layer_idx in layer_weights),
        }
    )
    return ModelRepresentation(
        model_id=model_id,
        architecture_family=architecture_family,
        num_parameters=num_parameters,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        layer_geometries=layer_geometries,
        metadata=merged_metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a raw-weight spectral representation from a Hugging Face causal LM.")
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen3-4B")
    parser.add_argument("output", help="Output .npz path for ModelRepresentation")
    parser.add_argument("--top-k", type=int, default=16, help="Directions to retain per layer after module aggregation")
    parser.add_argument("--top-k-per-module", type=int, default=4, help="Singular directions retained per linear module")
    parser.add_argument("--layers", default="", help="Comma-separated 1-based transformer block indices; default is all")
    parser.add_argument("--device", default="auto", help="Device passed to the Hugging Face adapter")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype passed to from_pretrained, for example float16")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to transformers.from_pretrained")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = HFCausalLMAdapter(
        args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
    layers = parse_layers(args.layers, adapter.num_layers)
    layer_weights = discover_layer_weight_specs(adapter.model, hidden_dim=adapter.hidden_dim, layers=layers)
    representation = build_weight_spectral_representation_from_layer_weights(
        model_id=adapter.model_id,
        architecture_family=adapter.architecture_family,
        num_parameters=adapter.num_parameters,
        hidden_dim=adapter.hidden_dim,
        num_layers=adapter.num_layers,
        layer_weights=layer_weights,
        top_k=args.top_k,
        top_k_per_module=args.top_k_per_module,
        metadata={
            "device": adapter.device,
            "torch_dtype": args.torch_dtype,
        },
    )
    output_path = Path(args.output).resolve()
    representation.save(output_path)
    print(
        json.dumps(
            {
                "architecture_family": adapter.architecture_family,
                "extraction_method": "weight_svd",
                "hidden_dim": adapter.hidden_dim,
                "layers": layers,
                "model_id": adapter.model_id,
                "num_layers": adapter.num_layers,
                "num_parameters": adapter.num_parameters,
                "output": str(output_path),
                "top_k": args.top_k,
                "top_k_per_module": args.top_k_per_module,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
