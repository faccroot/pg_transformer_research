#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.concept_probes import ConceptProbeSpec, load_concept_probe_specs
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from concept_probes import ConceptProbeSpec, load_concept_probe_specs  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation  # type: ignore[no-redef]


def read_calibration_records(path: str | Path, max_examples: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or "text" not in payload or "chunk_id" not in payload:
                raise ValueError(f"Malformed calibration record: {line[:120]!r}")
            records.append(payload)
            if max_examples > 0 and len(records) >= max_examples:
                break
    if not records:
        raise ValueError(f"No calibration records found in {path}")
    return records


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if not spec:
        return list(range(1, num_layers + 1))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or value > num_layers:
            raise ValueError(f"Layer {value} is outside [0, {num_layers}]")
        layers.append(value)
    if not layers:
        raise ValueError("No valid layers parsed from --layers")
    return sorted(set(layers))


def batched(items: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [items[offset: offset + batch_size] for offset in range(0, len(items), batch_size)]


def batched_pairs(items: tuple[tuple[str, str], ...] | list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    return [list(items[offset: offset + batch_size]) for offset in range(0, len(items), batch_size)]


def dominant_directions_from_covariance(covariance: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=1e9, neginf=-1e9)
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}")
    top_k = min(max(int(top_k), 1), int(covariance.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:top_k]
    selected_values = np.clip(eigenvalues[order], 0.0, None).astype(np.float32)
    selected_vectors = eigenvectors[:, order].T.astype(np.float32)
    norms = np.linalg.norm(selected_vectors, axis=1, keepdims=True)
    selected_vectors = selected_vectors / np.clip(norms, 1e-8, None)
    return selected_vectors, selected_values


def chunk_projection_summary(
    *,
    chunk_ids: list[str] | None,
    chunk_layer_projections: dict[int, np.ndarray],
    requested_layers: list[int],
) -> dict[str, object]:
    expected_chunk_count = len(chunk_ids or [])
    present_layers = sorted(int(layer_idx) for layer_idx in chunk_layer_projections)
    missing_layers = [int(layer_idx) for layer_idx in requested_layers if int(layer_idx) not in chunk_layer_projections]
    row_mismatches: dict[str, int] = {}
    for layer_idx in present_layers:
        projections = np.asarray(chunk_layer_projections[int(layer_idx)], dtype=np.float32)
        if projections.shape[0] != expected_chunk_count:
            row_mismatches[str(int(layer_idx))] = int(projections.shape[0])
    return {
        "expected_chunk_count": expected_chunk_count,
        "projection_layer_count": len(present_layers),
        "projection_layers": present_layers,
        "missing_projection_layers": missing_layers,
        "row_mismatches": row_mismatches,
    }


def require_chunk_projection_coverage(
    *,
    chunk_ids: list[str] | None,
    chunk_layer_projections: dict[int, np.ndarray],
    requested_layers: list[int],
) -> None:
    summary = chunk_projection_summary(
        chunk_ids=chunk_ids,
        chunk_layer_projections=chunk_layer_projections,
        requested_layers=requested_layers,
    )
    if summary["missing_projection_layers"]:
        raise RuntimeError(
            "Missing chunk projections for layers "
            f"{summary['missing_projection_layers']} with expected_chunk_count={summary['expected_chunk_count']}"
        )
    if summary["row_mismatches"]:
        raise RuntimeError(
            "Chunk projection row mismatches detected: "
            + ", ".join(f"layer {layer}: {count}" for layer, count in dict(summary["row_mismatches"]).items())
        )


def extract_concept_profiles(
    adapter: HFCausalLMAdapter,
    *,
    layers: list[int],
    max_length: int,
    batch_size: int,
    probe_specs: list[ConceptProbeSpec],
) -> dict[str, object]:
    if not probe_specs:
        return {}

    profiles: dict[str, object] = {}
    for probe in probe_specs:
        layer_direction_sum: dict[int, np.ndarray] = {
            int(layer_idx): np.zeros((adapter.hidden_dim,), dtype=np.float64) for layer_idx in layers
        }
        layer_norm_weighted_sum: dict[int, float] = {int(layer_idx): 0.0 for layer_idx in layers}
        layer_weight_sum: dict[int, float] = {int(layer_idx): 0.0 for layer_idx in layers}
        js_values: list[float] = []

        for pair_batch in batched_pairs(probe.pairs, batch_size=max(batch_size, 1)):
            left_texts = [pair[0] for pair in pair_batch]
            right_texts = [pair[1] for pair in pair_batch]
            stats = adapter.get_contrastive_statistics(
                left_texts,
                right_texts,
                layers=layers,
                max_length=max_length,
            )
            js = np.asarray(stats.js_divergence, dtype=np.float64).reshape(-1)
            js = np.nan_to_num(js, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.clip(js, 1e-8, None)
            js_values.extend(js.tolist())
            for layer_idx, deltas in stats.layer_deltas.items():
                deltas_np = np.asarray(deltas, dtype=np.float64)
                deltas_np = np.nan_to_num(deltas_np, nan=0.0, posinf=0.0, neginf=0.0)
                if deltas_np.ndim != 2 or deltas_np.shape[0] != weights.shape[0]:
                    raise ValueError(
                        f"Contrastive deltas for layer {layer_idx} must have shape [batch, dim], got {deltas_np.shape}"
                    )
                layer_direction_sum[int(layer_idx)] += (deltas_np * weights[:, None]).sum(axis=0)
                layer_norm_weighted_sum[int(layer_idx)] += float(np.linalg.norm(deltas_np, axis=1).dot(weights))
                layer_weight_sum[int(layer_idx)] += float(weights.sum())

        if not js_values:
            continue

        sharpness = float(np.mean(js_values))
        layers_payload: dict[str, object] = {}
        for layer_idx in layers:
            direction_sum = layer_direction_sum[int(layer_idx)]
            direction_norm = float(np.linalg.norm(direction_sum))
            if direction_norm <= 1e-8:
                direction = np.zeros((adapter.hidden_dim,), dtype=np.float32)
            else:
                direction = (direction_sum / direction_norm).astype(np.float32)
            contrast_norm_mean = float(layer_norm_weighted_sum[int(layer_idx)] / max(layer_weight_sum[int(layer_idx)], 1e-8))
            layer_score = float(sharpness * contrast_norm_mean)
            layers_payload[str(int(layer_idx))] = {
                "relative_depth": float(layer_idx / max(adapter.num_layers, 1)),
                "direction": direction.tolist(),
                "direction_norm": direction_norm,
                "contrast_norm_mean": contrast_norm_mean,
                "layer_score": layer_score,
                "weight_sum": float(layer_weight_sum[int(layer_idx)]),
            }
        profiles[probe.name] = {
            "description": probe.description,
            "num_pairs": len(probe.pairs),
            "sharpness": sharpness,
            "sharpness_std": float(np.std(np.asarray(js_values, dtype=np.float64))),
            "layers": layers_payload,
        }
    return profiles


def build_model_representation(
    records: list[dict[str, object]],
    *,
    adapter: HFCausalLMAdapter,
    calibration_jsonl: str | Path,
    layers: list[int],
    top_k: int,
    batch_size: int,
    max_length: int,
    torch_dtype: str,
    concept_probe_specs: list[ConceptProbeSpec] | None = None,
    concept_probe_batch_size: int | None = None,
) -> ModelRepresentation:
    chunk_ids = [str(record["chunk_id"]) for record in records]
    chunk_losses: np.ndarray | None = None
    layer_geometries: dict[int, LayerGeometry] = {}
    chunk_layer_projections: dict[int, np.ndarray] = {}

    for layer_idx in layers:
        importance_sum: np.ndarray | None = None
        covariance_numerator: np.ndarray | None = None
        weight_denom = 0.0
        loss_sum = 0.0
        token_count = 0
        sequence_losses_parts: list[np.ndarray] = []
        for batch in batched(records, batch_size):
            texts = [str(record["text"]) for record in batch]
            stats = adapter.get_gradient_statistics(texts, layer_idx=layer_idx, max_length=max_length)
            if importance_sum is None:
                importance_sum = np.zeros_like(stats.importance_sum, dtype=np.float32)
                covariance_numerator = np.zeros_like(stats.covariance_numerator, dtype=np.float32)
            importance_sum += stats.importance_sum
            covariance_numerator += stats.covariance_numerator
            weight_denom += stats.weight_denom
            loss_sum += stats.loss_sum
            token_count += stats.token_count
            if chunk_losses is None:
                sequence_losses_parts.append(stats.sequence_losses)
        if importance_sum is None or covariance_numerator is None:
            raise RuntimeError(f"Layer {layer_idx} produced no statistics")
        covariance = covariance_numerator / max(weight_denom, 1e-8)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=1e9, neginf=-1e9)
        directions, scales = dominant_directions_from_covariance(covariance, top_k=top_k)
        coactivation = directions @ covariance.astype(np.float32) @ directions.T
        layer_geometries[layer_idx] = LayerGeometry(
            relative_depth=layer_idx / max(adapter.num_layers, 1),
            directions=directions,
            scales=scales,
            covariance=covariance.astype(np.float32),
            coactivation=coactivation.astype(np.float32),
            importance=(importance_sum / max(token_count, 1)).astype(np.float32),
            metadata={
                "layer_idx": layer_idx,
                "mean_loss": float(loss_sum / max(token_count, 1)),
                "token_count": token_count,
                "weight_denom": weight_denom,
            },
        )
        if chunk_losses is None:
            chunk_losses = np.concatenate(sequence_losses_parts, axis=0).astype(np.float32)

    projection_parts: dict[int, list[np.ndarray]] = {int(layer_idx): [] for layer_idx in layers}
    for batch in batched(records, batch_size):
        texts = [str(record["text"]) for record in batch]
        pooled_states = adapter.get_mean_pooled_hidden_states(texts, layers=layers, max_length=max_length)
        for layer_idx in layers:
            pooled = np.asarray(pooled_states[int(layer_idx)], dtype=np.float32)
            directions = np.asarray(layer_geometries[int(layer_idx)].directions, dtype=np.float32)
            projections = pooled @ directions.T
            projection_parts[int(layer_idx)].append(projections.astype(np.float32))
    for layer_idx in layers:
        if projection_parts[int(layer_idx)]:
            chunk_layer_projections[int(layer_idx)] = np.concatenate(projection_parts[int(layer_idx)], axis=0).astype(np.float32)

    concept_profiles = extract_concept_profiles(
        adapter,
        layers=layers,
        max_length=max_length,
        batch_size=concept_probe_batch_size if concept_probe_batch_size is not None else batch_size,
        probe_specs=concept_probe_specs or [],
    )

    return ModelRepresentation(
        model_id=adapter.model_id,
        architecture_family=adapter.architecture_family,
        num_parameters=adapter.num_parameters,
        hidden_dim=adapter.hidden_dim,
        num_layers=adapter.num_layers,
        layer_geometries=layer_geometries,
        chunk_losses=chunk_losses,
        chunk_ids=chunk_ids[: 0 if chunk_losses is None else int(chunk_losses.shape[0])],
        chunk_layer_projections=chunk_layer_projections,
        concept_profiles=concept_profiles,
        metadata={
            "batch_size": batch_size,
            "calibration_jsonl": str(Path(calibration_jsonl).resolve()),
            "chunk_projection_layers": sorted(chunk_layer_projections),
            "chunk_projection_missing_layers": [int(layer_idx) for layer_idx in layers if int(layer_idx) not in chunk_layer_projections],
            "device": adapter.device,
            "layers": layers,
            "max_examples": len(records),
            "max_length": max_length,
            "top_k": top_k,
            "torch_dtype": torch_dtype,
            "concept_probe_count": len(concept_probe_specs or []),
            "concept_probe_batch_size": concept_probe_batch_size if concept_probe_batch_size is not None else batch_size,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a first-pass functional geometry from a Hugging Face causal LM.")
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen3-4B")
    parser.add_argument("calibration_jsonl", help="Decoded calibration-set JSONL built by calibration_set.py")
    parser.add_argument("output", help="Output .npz path for ModelRepresentation")
    parser.add_argument("--top-k", type=int, default=16, help="Top functional directions to keep per layer")
    parser.add_argument("--batch-size", type=int, default=2, help="Texts per extraction batch")
    parser.add_argument("--max-examples", type=int, default=128, help="Optional cap on calibration examples")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length")
    parser.add_argument("--layers", default="", help="Comma-separated layers; default is all hidden layers")
    parser.add_argument("--device", default="auto", help="Device passed to the Hugging Face adapter")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype passed to from_pretrained, for example float16")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to transformers.from_pretrained")
    parser.add_argument("--concept-probes", default="default", help="Concept probe set: default, none, or a JSON file path")
    parser.add_argument("--probe-batch-size", type=int, default=4, help="Pairs per concept-probe batch")
    parser.add_argument("--require-chunk-projections", action="store_true", help="Fail if any requested layer is missing chunk projections")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = read_calibration_records(args.calibration_jsonl, max_examples=args.max_examples)
    adapter = HFCausalLMAdapter(
        args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
    layers = parse_layers(args.layers, adapter.num_layers)
    concept_probe_specs = load_concept_probe_specs(args.concept_probes)
    representation = build_model_representation(
        records,
        adapter=adapter,
        calibration_jsonl=args.calibration_jsonl,
        layers=layers,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        concept_probe_specs=concept_probe_specs,
        concept_probe_batch_size=args.probe_batch_size,
    )
    if args.require_chunk_projections:
        require_chunk_projection_coverage(
            chunk_ids=representation.chunk_ids,
            chunk_layer_projections=representation.chunk_layer_projections,
            requested_layers=layers,
        )
    output_path = Path(args.output).resolve()
    representation.save(output_path)
    projection_summary = chunk_projection_summary(
        chunk_ids=representation.chunk_ids,
        chunk_layer_projections=representation.chunk_layer_projections,
        requested_layers=layers,
    )
    summary = {
        "architecture_family": adapter.architecture_family,
        "chunk_projection_summary": projection_summary,
        "hidden_dim": adapter.hidden_dim,
        "layers": layers,
        "model_id": args.model_id,
        "num_layers": adapter.num_layers,
        "num_parameters": adapter.num_parameters,
        "output": str(output_path),
        "top_k": args.top_k,
        "concepts": sorted(representation.concept_profiles),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
