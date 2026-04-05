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
    from tools.representation_learning.extract_model_representation import batched, read_calibration_records
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from extract_model_representation import batched, read_calibration_records  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation  # type: ignore[no-redef]


@dataclass
class BlockSampleContext:
    block: Any
    hidden_states: Any
    args_tail: tuple[Any, ...]
    kwargs: dict[str, Any]
    token_index: int


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


def _detach_tree(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _slice_batch_tree(value: Any, index: int, batch_size: int) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index: index + 1].detach()
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_slice_batch_tree(item, index, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_tree(item, index, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _slice_batch_tree(item, index, batch_size) for key, item in value.items()}
    return value


def capture_block_sample_contexts(
    adapter: HFCausalLMAdapter,
    texts: list[str],
    *,
    layers: list[int],
    max_length: int,
) -> dict[int, list[BlockSampleContext]]:
    torch = adapter._torch
    if not texts:
        return {int(layer_idx): [] for layer_idx in layers}
    encoded = adapter.tokenize(texts, max_length=max_length)
    blocks = _iter_transformer_blocks(adapter.model)
    requested = {int(layer_idx): int(layer_idx) - 1 for layer_idx in layers}
    captures: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    handles = []

    for layer_idx, zero_idx in requested.items():
        block = blocks[zero_idx]

        def _make_pre_hook(current_layer_idx: int):
            def _pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                if current_layer_idx not in captures:
                    captures[current_layer_idx] = (
                        tuple(_detach_tree(arg) for arg in args),
                        {key: _detach_tree(value) for key, value in kwargs.items()},
                    )

            return _pre_hook

        handles.append(block.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True))

    try:
        with torch.no_grad():
            adapter.model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        seq_len = int(encoded["input_ids"].shape[1])
        token_positions = [seq_len - 1 for _ in texts]
    else:
        token_positions = (attention_mask.sum(dim=1).to(dtype=torch.long).clamp_min(1) - 1).detach().cpu().tolist()

    contexts: dict[int, list[BlockSampleContext]] = {int(layer_idx): [] for layer_idx in layers}
    batch_size = len(texts)
    for layer_idx in layers:
        if layer_idx not in captures:
            raise RuntimeError(f"Did not capture layer {layer_idx} inputs during model forward")
        args, kwargs = captures[layer_idx]
        if not args:
            raise RuntimeError(f"Layer {layer_idx} was called without positional hidden states")
        hidden_states = args[0]
        args_tail = tuple(args[1:])
        for batch_index in range(batch_size):
            contexts[int(layer_idx)].append(
                BlockSampleContext(
                    block=blocks[int(layer_idx) - 1],
                    hidden_states=_slice_batch_tree(hidden_states, batch_index, batch_size),
                    args_tail=_slice_batch_tree(args_tail, batch_index, batch_size),
                    kwargs=_slice_batch_tree(kwargs, batch_index, batch_size),
                    token_index=int(token_positions[batch_index]),
                )
            )
    return contexts


def _block_hidden_output(block: Any, hidden_states: Any, args_tail: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    output = block(hidden_states, *args_tail, **kwargs)
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output


def token_local_map(context: BlockSampleContext, x_token_flat: Any) -> Any:
    torch = __import__("torch")
    base_hidden = context.hidden_states.detach().clone()
    token = x_token_flat.reshape(1, -1).to(device=base_hidden.device, dtype=base_hidden.dtype)
    base_hidden[:, context.token_index, :] = token
    output = _block_hidden_output(context.block, base_hidden, context.args_tail, context.kwargs)
    token_output = output[:, context.token_index, :].reshape(-1)
    return token_output.to(dtype=torch.float32)


def jtj_product_for_context(context: BlockSampleContext, vector: np.ndarray) -> np.ndarray:
    torch = __import__("torch")
    x0 = context.hidden_states[:, context.token_index, :].reshape(-1).detach().clone()
    x0 = x0.to(dtype=torch.float32)
    vector_tensor = torch.tensor(np.asarray(vector, dtype=np.float32), device=x0.device, dtype=torch.float32)

    def _fn(x_flat: Any) -> Any:
        return token_local_map(context, x_flat)

    x = x0.detach().clone().requires_grad_(True)
    y = _fn(x)
    _base, jv = torch.autograd.functional.jvp(_fn, (x,), (vector_tensor,), create_graph=False, strict=False)
    scalar = (y * jv.detach()).sum()
    grad = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
    return grad.detach().cpu().numpy().astype(np.float32)


def leading_spectral_directions_from_jtj(
    apply_jtj: Any,
    *,
    dim: int,
    top_k: int,
    power_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    singular_values: list[float] = []

    def _orthogonalize(vector: np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32)
        for basis in vectors:
            result = result - float(result @ basis) * basis
        norm = float(np.linalg.norm(result))
        if norm <= 1e-8:
            return np.zeros_like(result)
        return result / norm

    for _ in range(min(max(int(top_k), 1), dim)):
        vector = rng.standard_normal((dim,), dtype=np.float32)
        vector = _orthogonalize(vector)
        if not np.any(vector):
            break
        for _step in range(max(int(power_steps), 1)):
            vector = _orthogonalize(apply_jtj(vector))
            if not np.any(vector):
                break
        if not np.any(vector):
            break
        jtj_vector = np.asarray(apply_jtj(vector), dtype=np.float32)
        eigenvalue = float(max(vector @ jtj_vector, 0.0))
        singular_value = float(np.sqrt(eigenvalue))
        vectors.append(vector.astype(np.float32))
        singular_values.append(singular_value)
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32), np.asarray(singular_values, dtype=np.float32)


def extract_layer_jacobian_geometry(
    contexts: list[BlockSampleContext],
    *,
    hidden_dim: int,
    top_k: int,
    power_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not contexts:
        return np.zeros((0, hidden_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    def _apply_average_jtj(vector: np.ndarray) -> np.ndarray:
        accum = np.zeros((hidden_dim,), dtype=np.float32)
        for context in contexts:
            accum += jtj_product_for_context(context, vector)
        return accum / max(len(contexts), 1)

    return leading_spectral_directions_from_jtj(
        _apply_average_jtj,
        dim=hidden_dim,
        top_k=top_k,
        power_steps=power_steps,
        seed=seed,
    )


def build_jacobian_spectral_representation(
    records: list[dict[str, object]],
    *,
    adapter: HFCausalLMAdapter,
    layers: list[int],
    top_k: int,
    power_steps: int,
    batch_size: int,
    max_length: int,
) -> ModelRepresentation:
    layer_contexts: dict[int, list[BlockSampleContext]] = {int(layer_idx): [] for layer_idx in layers}
    chunk_losses_parts: list[np.ndarray] = []
    chunk_ids = [str(record["chunk_id"]) for record in records]

    for batch in batched(records, batch_size):
        texts = [str(record["text"]) for record in batch]
        batch_contexts = capture_block_sample_contexts(adapter, texts, layers=layers, max_length=max_length)
        for layer_idx in layers:
            layer_contexts[int(layer_idx)].extend(batch_contexts[int(layer_idx)])
        chunk_losses_parts.append(adapter.compute_sequence_losses(texts, max_length=max_length))

    layer_geometries: dict[int, LayerGeometry] = {}
    for layer_idx in layers:
        directions, scales = extract_layer_jacobian_geometry(
            layer_contexts[int(layer_idx)],
            hidden_dim=adapter.hidden_dim,
            top_k=top_k,
            power_steps=power_steps,
            seed=17 + int(layer_idx) * 7919,
        )
        covariance = directions.T @ np.diag((scales ** 2).astype(np.float64)) @ directions if directions.size else np.zeros((adapter.hidden_dim, adapter.hidden_dim), dtype=np.float64)
        coactivation = directions @ covariance.astype(np.float32) @ directions.T if directions.size else np.zeros((0, 0), dtype=np.float32)
        importance = np.abs(directions.T) @ np.abs(scales) if directions.size else np.zeros((adapter.hidden_dim,), dtype=np.float32)
        layer_geometries[int(layer_idx)] = LayerGeometry(
            relative_depth=float(layer_idx / max(adapter.num_layers, 1)),
            directions=directions.astype(np.float32),
            scales=scales.astype(np.float32),
            covariance=covariance.astype(np.float32),
            coactivation=coactivation.astype(np.float32),
            importance=np.asarray(importance, dtype=np.float32),
            metadata={
                "extraction_method": "jacobian_svd",
                "num_contexts": len(layer_contexts[int(layer_idx)]),
                "power_steps": int(power_steps),
            },
        )

    chunk_layer_projections: dict[int, np.ndarray] = {}
    for layer_idx in layers:
        if layer_geometries[int(layer_idx)].directions.shape[0] == 0:
            continue
        projection_parts: list[np.ndarray] = []
        for batch in batched(records, batch_size):
            texts = [str(record["text"]) for record in batch]
            pooled = adapter.get_mean_pooled_hidden_states(texts, layers=[int(layer_idx)], max_length=max_length)[int(layer_idx)]
            projection_parts.append((np.asarray(pooled, dtype=np.float32) @ layer_geometries[int(layer_idx)].directions.T).astype(np.float32))
        chunk_layer_projections[int(layer_idx)] = np.concatenate(projection_parts, axis=0).astype(np.float32)

    return ModelRepresentation(
        model_id=adapter.model_id,
        architecture_family=adapter.architecture_family,
        num_parameters=adapter.num_parameters,
        hidden_dim=adapter.hidden_dim,
        num_layers=adapter.num_layers,
        layer_geometries=layer_geometries,
        chunk_losses=np.concatenate(chunk_losses_parts, axis=0).astype(np.float32) if chunk_losses_parts else None,
        chunk_ids=chunk_ids,
        chunk_layer_projections=chunk_layer_projections,
        metadata={
            "batch_size": int(batch_size),
            "device": adapter.device,
            "extraction_method": "jacobian_svd",
            "layers": list(layers),
            "max_examples": len(records),
            "max_length": int(max_length),
            "power_steps": int(power_steps),
            "top_k": int(top_k),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a local-Jacobian spectral representation from a Hugging Face causal LM using calibration-set hidden states.")
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen3-4B")
    parser.add_argument("calibration_jsonl", help="Decoded calibration-set JSONL built by calibration_set.py")
    parser.add_argument("output", help="Output .npz path for ModelRepresentation")
    parser.add_argument("--top-k", type=int, default=8, help="Directions to retain per layer")
    parser.add_argument("--power-steps", type=int, default=6, help="Power-iteration steps per direction")
    parser.add_argument("--batch-size", type=int, default=2, help="Texts per context-capture batch")
    parser.add_argument("--max-examples", type=int, default=32, help="Optional cap on calibration examples")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer truncation length")
    parser.add_argument("--layers", default="", help="Comma-separated 1-based layers; default is all hidden layers")
    parser.add_argument("--device", default="auto", help="Device passed to the Hugging Face adapter")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype passed to from_pretrained, for example float32")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to transformers.from_pretrained")
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
    representation = build_jacobian_spectral_representation(
        records,
        adapter=adapter,
        layers=layers,
        top_k=args.top_k,
        power_steps=args.power_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    output_path = Path(args.output).resolve()
    representation.save(output_path)
    print(
        json.dumps(
            {
                "architecture_family": adapter.architecture_family,
                "extraction_method": "jacobian_svd",
                "hidden_dim": adapter.hidden_dim,
                "layers": layers,
                "max_examples": args.max_examples,
                "model_id": adapter.model_id,
                "num_layers": adapter.num_layers,
                "num_parameters": adapter.num_parameters,
                "output": str(output_path),
                "power_steps": args.power_steps,
                "top_k": args.top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
