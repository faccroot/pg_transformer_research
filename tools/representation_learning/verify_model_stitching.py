#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.extract_model_representation import read_calibration_records
    from tools.representation_learning.model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch
    from tools.representation_learning.schemas import SharedLatentGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from extract_model_representation import read_calibration_records  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter, SequenceRepresentationBatch  # type: ignore[no-redef]
    from schemas import SharedLatentGeometry  # type: ignore[no-redef]


@dataclass
class StitchFit:
    weight: np.ndarray
    bias: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float


def _chunk_text_lookup(records: list[dict[str, object]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for record in records:
        chunk_id = str(record["chunk_id"])
        text = str(record["text"])
        lookup[chunk_id] = text
    return lookup


def _pad_sequence_rows(parts: list[np.ndarray], *, fill_value: float = 0.0, dtype: Any = np.float32) -> np.ndarray:
    if not parts:
        return np.zeros((0, 0), dtype=dtype)
    max_len = max(int(part.shape[1]) for part in parts)
    padded_parts: list[np.ndarray] = []
    for part in parts:
        array = np.asarray(part, dtype=dtype)
        if array.shape[1] == max_len:
            padded_parts.append(array)
            continue
        if array.ndim == 2:
            padded = np.full((array.shape[0], max_len), fill_value, dtype=dtype)
            padded[:, : array.shape[1]] = array
        elif array.ndim == 3:
            padded = np.full((array.shape[0], max_len, array.shape[2]), fill_value, dtype=dtype)
            padded[:, : array.shape[1], :] = array
        else:
            raise ValueError(f"Expected 2D or 3D sequence rows for padding, got {array.shape}")
        padded_parts.append(padded)
    return np.concatenate(padded_parts, axis=0)


def _select_chunk_ids(
    records: list[dict[str, object]],
    *,
    shared_geometry: SharedLatentGeometry | None,
    shared_layer: int | None,
    max_examples: int,
) -> list[str]:
    available = {str(record["chunk_id"]) for record in records}
    if shared_geometry is None:
        chunk_ids = [str(record["chunk_id"]) for record in records]
    else:
        if shared_layer is None:
            shared_layer = max(shared_geometry.layers)
        if int(shared_layer) not in shared_geometry.layers:
            raise ValueError(f"shared_layer={shared_layer} is not present in the shared geometry artifact")
        layer = shared_geometry.layers[int(shared_layer)]
        chunk_ids = [chunk_id for chunk_id in layer.chunk_ids if chunk_id in available]
    if max_examples > 0:
        chunk_ids = chunk_ids[:max_examples]
    if not chunk_ids:
        raise ValueError("No chunk ids selected for stitching evaluation")
    return chunk_ids


def _texts_for_chunk_ids(records: list[dict[str, object]], chunk_ids: list[str]) -> list[str]:
    lookup = _chunk_text_lookup(records)
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} selected chunk ids from calibration records")
    return [lookup[chunk_id] for chunk_id in chunk_ids]


def _collect_sequence_representations(
    adapter: Any,
    texts: list[str],
    *,
    batch_size: int,
    layers: list[int] | None = None,
    capture_full_sequences: bool = False,
    max_length: int | None,
) -> SequenceRepresentationBatch:
    mean_parts: list[np.ndarray] = []
    last_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    attention_mask_parts: list[np.ndarray] = []
    layer_last_parts: dict[int, list[np.ndarray]] = {int(layer_idx): [] for layer_idx in (layers or [])}
    layer_sequence_parts: dict[int, list[np.ndarray]] = {int(layer_idx): [] for layer_idx in (layers or [])}
    for offset in range(0, len(texts), max(int(batch_size), 1)):
        batch = texts[offset: offset + max(int(batch_size), 1)]
        try:
            representations = adapter.get_sequence_representations(
                batch,
                layers=layers,
                capture_full_sequences=capture_full_sequences,
                max_length=max_length,
            )
        except TypeError:
            representations = adapter.get_sequence_representations(batch, max_length=max_length)
        mean_parts.append(np.asarray(representations.mean_hidden, dtype=np.float32))
        last_parts.append(np.asarray(representations.last_hidden, dtype=np.float32))
        logit_parts.append(np.asarray(representations.last_logits, dtype=np.float32))
        if representations.attention_mask is not None:
            attention_mask_parts.append(np.asarray(representations.attention_mask, dtype=np.int32))
        for layer_idx in layer_last_parts:
            layer_last = representations.layer_last_hidden.get(int(layer_idx))
            if layer_last is not None:
                layer_last_parts[int(layer_idx)].append(np.asarray(layer_last, dtype=np.float32))
            layer_sequence = representations.layer_hidden_sequences.get(int(layer_idx))
            if layer_sequence is not None:
                layer_sequence_parts[int(layer_idx)].append(np.asarray(layer_sequence, dtype=np.float32))
    return SequenceRepresentationBatch(
        mean_hidden=np.concatenate(mean_parts, axis=0) if mean_parts else np.zeros((0, 0), dtype=np.float32),
        last_hidden=np.concatenate(last_parts, axis=0) if last_parts else np.zeros((0, 0), dtype=np.float32),
        last_logits=np.concatenate(logit_parts, axis=0) if logit_parts else np.zeros((0, 0), dtype=np.float32),
        attention_mask=_pad_sequence_rows(attention_mask_parts, fill_value=0, dtype=np.int32) if attention_mask_parts else None,
        layer_last_hidden={
            int(layer_idx): np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)
            for layer_idx, parts in layer_last_parts.items()
        },
        layer_hidden_sequences={
            int(layer_idx): _pad_sequence_rows(parts, fill_value=0.0, dtype=np.float32) if parts else np.zeros((0, 0, 0), dtype=np.float32)
            for layer_idx, parts in layer_sequence_parts.items()
        },
    )


def _train_eval_split(num_examples: int, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_examples < 2:
        raise ValueError("Need at least two examples for a train/eval split")
    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(num_examples)
    train_count = int(round(num_examples * train_fraction))
    train_count = min(max(train_count, 1), num_examples - 1)
    train_idx = np.sort(order[:train_count]).astype(np.int32)
    eval_idx = np.sort(order[train_count:]).astype(np.int32)
    return train_idx, eval_idx


def resolve_model_layer_index(
    *,
    shared_geometry: SharedLatentGeometry | None,
    shared_layer: int | None,
    model_num_layers: int,
) -> int:
    if model_num_layers <= 0:
        raise ValueError(f"model_num_layers must be positive, got {model_num_layers}")
    if shared_layer is None:
        return int(model_num_layers)
    if shared_geometry is not None and int(shared_layer) in shared_geometry.layers:
        relative_depth = float(shared_geometry.layers[int(shared_layer)].relative_depth)
        resolved = int(round(relative_depth * float(model_num_layers)))
    else:
        resolved = int(shared_layer)
    return min(max(resolved, 1), int(model_num_layers))


def fit_affine_stitch(
    source_hidden: np.ndarray,
    target_hidden: np.ndarray,
    *,
    ridge_lambda: float = 1e-4,
) -> StitchFit:
    source_hidden = np.asarray(source_hidden, dtype=np.float64)
    target_hidden = np.asarray(target_hidden, dtype=np.float64)
    if source_hidden.ndim != 2 or target_hidden.ndim != 2:
        raise ValueError("source_hidden and target_hidden must both be 2D")
    if source_hidden.shape[0] != target_hidden.shape[0]:
        raise ValueError("source_hidden and target_hidden must have the same number of rows")
    if source_hidden.shape[0] == 0:
        raise ValueError("Cannot fit stitching map with zero examples")
    source_aug = np.concatenate(
        [source_hidden, np.ones((source_hidden.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    ridge = max(float(ridge_lambda), 0.0)
    if ridge > 0.0:
        penalty = np.sqrt(ridge) * np.eye(source_aug.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        design = np.concatenate([source_aug, penalty], axis=0)
        target = np.concatenate([target_hidden, np.zeros((source_aug.shape[1], target_hidden.shape[1]), dtype=np.float64)], axis=0)
    else:
        design = source_aug
        target = target_hidden
    solution, _residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    weight = np.asarray(solution[:-1, :], dtype=np.float32)
    bias = np.asarray(solution[-1, :], dtype=np.float32)
    singular_values = np.asarray(singular_values, dtype=np.float32)
    finite_singular = singular_values[singular_values > 1e-8]
    condition_number = float(finite_singular.max() / finite_singular.min()) if finite_singular.size > 0 else float("inf")
    return StitchFit(
        weight=weight,
        bias=bias,
        rank=int(rank),
        singular_values=singular_values,
        condition_number=condition_number,
    )


def apply_affine_stitch(hidden: np.ndarray, fit: StitchFit) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float32)
    return (hidden @ fit.weight + fit.bias).astype(np.float32)


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.clip(left_norm * right_norm, 1e-8, None)
    return np.sum(left * right, axis=1) / denom


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_sum


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> np.ndarray:
    ref_log = _log_softmax(reference_logits)
    cand_log = _log_softmax(candidate_logits)
    ref_prob = np.exp(ref_log)
    return np.sum(ref_prob * (ref_log - cand_log), axis=1)


def _js_divergence(left_logits: np.ndarray, right_logits: np.ndarray) -> np.ndarray:
    left_log = _log_softmax(left_logits)
    right_log = _log_softmax(right_logits)
    left_prob = np.exp(left_log)
    right_prob = np.exp(right_log)
    mixture = np.clip(0.5 * (left_prob + right_prob), 1e-8, None)
    mixture_log = np.log(mixture)
    left_kl = np.sum(left_prob * (left_log - mixture_log), axis=1)
    right_kl = np.sum(right_prob * (right_log - mixture_log), axis=1)
    return 0.5 * (left_kl + right_kl)


def _as_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def evaluate_stitch_split(
    *,
    source_hidden: np.ndarray,
    target_hidden: np.ndarray,
    target_logits: np.ndarray,
    fit: StitchFit,
    target_adapter: Any,
    logit_batch_size: int = 128,
) -> dict[str, float]:
    stitched_hidden = apply_affine_stitch(source_hidden, fit)
    hidden_error = stitched_hidden - target_hidden
    mse = float(np.mean(hidden_error ** 2))
    cosine = _row_cosine(stitched_hidden, target_hidden)
    var = float(np.var(target_hidden))
    explained_variance = float(1.0 - mse / max(var, 1e-8))

    stitched_logits_parts: list[np.ndarray] = []
    for offset in range(0, stitched_hidden.shape[0], max(int(logit_batch_size), 1)):
        batch_hidden = stitched_hidden[offset: offset + max(int(logit_batch_size), 1)]
        batch_logits = target_adapter.project_hidden_to_logits(batch_hidden)
        stitched_logits_parts.append(_as_numpy_float32(batch_logits))
    stitched_logits = np.concatenate(stitched_logits_parts, axis=0) if stitched_logits_parts else np.zeros_like(target_logits)

    kl = _kl_divergence(target_logits, stitched_logits)
    js = _js_divergence(target_logits, stitched_logits)
    top1_agreement = np.argmax(target_logits, axis=1) == np.argmax(stitched_logits, axis=1)

    return {
        "hidden_mse": mse,
        "hidden_cosine_mean": float(np.mean(cosine)),
        "hidden_cosine_min": float(np.min(cosine)),
        "hidden_cosine_max": float(np.max(cosine)),
        "hidden_explained_variance": explained_variance,
        "target_logit_kl_mean": float(np.mean(kl)),
        "target_logit_js_mean": float(np.mean(js)),
        "target_top1_agreement": float(np.mean(top1_agreement.astype(np.float32))),
    }


def evaluate_continuation_stitch_split(
    *,
    source_last_hidden: np.ndarray,
    target_layer_hidden_sequence: np.ndarray,
    target_layer_last_hidden: np.ndarray,
    target_final_last_hidden: np.ndarray,
    target_final_logits: np.ndarray,
    attention_mask: np.ndarray,
    fit: StitchFit,
    target_adapter: Any,
    target_layer_idx: int,
) -> dict[str, float]:
    stitched_last_hidden = apply_affine_stitch(source_last_hidden, fit)
    stitched_sequence = np.asarray(target_layer_hidden_sequence, dtype=np.float32).copy()
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    last_indices = np.clip(attention_mask.sum(axis=1).astype(np.int64) - 1, 0, stitched_sequence.shape[1] - 1)
    stitched_sequence[np.arange(stitched_sequence.shape[0]), last_indices, :] = stitched_last_hidden

    continued = target_adapter.continue_from_layer_hidden_sequence(
        stitched_sequence,
        attention_mask,
        layer_idx=target_layer_idx,
    )
    layer_hidden_error = stitched_last_hidden - np.asarray(target_layer_last_hidden, dtype=np.float32)
    layer_mse = float(np.mean(layer_hidden_error ** 2))
    layer_cosine = _row_cosine(stitched_last_hidden, np.asarray(target_layer_last_hidden, dtype=np.float32))
    layer_var = float(np.var(target_layer_last_hidden))
    hidden_error = np.asarray(continued.last_hidden, dtype=np.float32) - np.asarray(target_final_last_hidden, dtype=np.float32)
    mse = float(np.mean(hidden_error ** 2))
    cosine = _row_cosine(np.asarray(continued.last_hidden, dtype=np.float32), np.asarray(target_final_last_hidden, dtype=np.float32))
    var = float(np.var(target_final_last_hidden))
    explained_variance = float(1.0 - mse / max(var, 1e-8))
    kl = _kl_divergence(np.asarray(target_final_logits, dtype=np.float32), np.asarray(continued.last_logits, dtype=np.float32))
    js = _js_divergence(np.asarray(target_final_logits, dtype=np.float32), np.asarray(continued.last_logits, dtype=np.float32))
    top1_agreement = np.argmax(np.asarray(target_final_logits, dtype=np.float32), axis=1) == np.argmax(np.asarray(continued.last_logits, dtype=np.float32), axis=1)
    return {
        "layer_hidden_mse": layer_mse,
        "layer_hidden_cosine_mean": float(np.mean(layer_cosine)),
        "hidden_mse": mse,
        "hidden_cosine_mean": float(np.mean(cosine)),
        "hidden_cosine_min": float(np.min(cosine)),
        "hidden_cosine_max": float(np.max(cosine)),
        "hidden_explained_variance": explained_variance,
        "layer_hidden_explained_variance": float(1.0 - layer_mse / max(layer_var, 1e-8)),
        "target_logit_kl_mean": float(np.mean(kl)),
        "target_logit_js_mean": float(np.mean(js)),
        "target_top1_agreement": float(np.mean(top1_agreement.astype(np.float32))),
    }


def build_stitching_report_from_batches(
    *,
    source_model_id: str,
    target_model_id: str,
    calibration_jsonl: str | Path,
    selected_chunk_ids: list[str],
    source_batch: SequenceRepresentationBatch,
    target_batch: SequenceRepresentationBatch,
    target_adapter: Any,
    shared_geometry_path: str | Path | None = None,
    shared_layer: int | None = None,
    source_layer_idx: int | None = None,
    target_layer_idx: int | None = None,
    train_fraction: float = 0.75,
    seed: int = 17,
    ridge_lambda: float = 1e-4,
    batch_size: int = 8,
    max_length: int | None = 1024,
    representation_mode: str = "final_last_hidden",
) -> dict[str, object]:
    train_idx, eval_idx = _train_eval_split(len(selected_chunk_ids), train_fraction=train_fraction, seed=seed)
    if representation_mode == "final_last_hidden":
        fit = fit_affine_stitch(
            source_batch.last_hidden[train_idx],
            target_batch.last_hidden[train_idx],
            ridge_lambda=ridge_lambda,
        )
        train_metrics = evaluate_stitch_split(
            source_hidden=source_batch.last_hidden[train_idx],
            target_hidden=target_batch.last_hidden[train_idx],
            target_logits=target_batch.last_logits[train_idx],
            fit=fit,
            target_adapter=target_adapter,
        )
        eval_metrics = evaluate_stitch_split(
            source_hidden=source_batch.last_hidden[eval_idx],
            target_hidden=target_batch.last_hidden[eval_idx],
            target_logits=target_batch.last_logits[eval_idx],
            fit=fit,
            target_adapter=target_adapter,
        )
    elif representation_mode == "layer_last_hidden_continuation":
        if source_layer_idx is None or target_layer_idx is None:
            raise ValueError("source_layer_idx and target_layer_idx are required for layer_last_hidden_continuation mode")
        source_layer_hidden = source_batch.layer_last_hidden.get(int(source_layer_idx))
        target_layer_last_hidden = target_batch.layer_last_hidden.get(int(target_layer_idx))
        target_layer_sequence = target_batch.layer_hidden_sequences.get(int(target_layer_idx))
        if source_layer_hidden is None or target_layer_last_hidden is None or target_layer_sequence is None:
            raise ValueError("Missing per-layer hidden representations required for continuation stitching")
        if target_batch.attention_mask is None:
            raise ValueError("Target batch is missing attention_mask required for continuation stitching")
        fit = fit_affine_stitch(
            np.asarray(source_layer_hidden, dtype=np.float32)[train_idx],
            np.asarray(target_layer_last_hidden, dtype=np.float32)[train_idx],
            ridge_lambda=ridge_lambda,
        )
        train_metrics = evaluate_continuation_stitch_split(
            source_last_hidden=np.asarray(source_layer_hidden, dtype=np.float32)[train_idx],
            target_layer_hidden_sequence=np.asarray(target_layer_sequence, dtype=np.float32)[train_idx],
            target_layer_last_hidden=np.asarray(target_layer_last_hidden, dtype=np.float32)[train_idx],
            target_final_last_hidden=np.asarray(target_batch.last_hidden, dtype=np.float32)[train_idx],
            target_final_logits=np.asarray(target_batch.last_logits, dtype=np.float32)[train_idx],
            attention_mask=np.asarray(target_batch.attention_mask, dtype=np.int32)[train_idx],
            fit=fit,
            target_adapter=target_adapter,
            target_layer_idx=int(target_layer_idx),
        )
        eval_metrics = evaluate_continuation_stitch_split(
            source_last_hidden=np.asarray(source_layer_hidden, dtype=np.float32)[eval_idx],
            target_layer_hidden_sequence=np.asarray(target_layer_sequence, dtype=np.float32)[eval_idx],
            target_layer_last_hidden=np.asarray(target_layer_last_hidden, dtype=np.float32)[eval_idx],
            target_final_last_hidden=np.asarray(target_batch.last_hidden, dtype=np.float32)[eval_idx],
            target_final_logits=np.asarray(target_batch.last_logits, dtype=np.float32)[eval_idx],
            attention_mask=np.asarray(target_batch.attention_mask, dtype=np.int32)[eval_idx],
            fit=fit,
            target_adapter=target_adapter,
            target_layer_idx=int(target_layer_idx),
        )
    else:
        raise ValueError(f"Unsupported representation_mode: {representation_mode}")
    return {
        "source_model_id": source_model_id,
        "target_model_id": target_model_id,
        "calibration_jsonl": str(Path(calibration_jsonl).resolve()),
        "shared_geometry_path": str(Path(shared_geometry_path).resolve()) if shared_geometry_path else None,
        "shared_layer": int(shared_layer) if shared_layer is not None else None,
        "source_layer_idx": int(source_layer_idx) if source_layer_idx is not None else None,
        "target_layer_idx": int(target_layer_idx) if target_layer_idx is not None else None,
        "num_examples": len(selected_chunk_ids),
        "train_examples": int(train_idx.shape[0]),
        "eval_examples": int(eval_idx.shape[0]),
        "source_hidden_dim": int(
            source_batch.layer_last_hidden[int(source_layer_idx)].shape[1]
            if representation_mode == "layer_last_hidden_continuation" and source_layer_idx is not None
            else source_batch.last_hidden.shape[1]
        ),
        "target_hidden_dim": int(
            target_batch.layer_last_hidden[int(target_layer_idx)].shape[1]
            if representation_mode == "layer_last_hidden_continuation" and target_layer_idx is not None
            else target_batch.last_hidden.shape[1]
        ),
        "target_vocab_dim": int(target_batch.last_logits.shape[1]),
        "fit": {
            "rank": int(fit.rank),
            "condition_number": float(fit.condition_number),
            "singular_values": [float(value) for value in fit.singular_values.tolist()],
            "ridge_lambda": float(ridge_lambda),
        },
        "metrics": {
            "train": train_metrics,
            "eval": eval_metrics,
        },
        "metadata": {
            "method": "last_hidden_affine_stitching" if representation_mode == "final_last_hidden" else "layer_last_hidden_continuation_stitching",
            "representation_source": "final_last_hidden_only" if representation_mode == "final_last_hidden" else "source_last_position_hidden_at_selected_layer",
            "target_evaluation": "target_lm_head_projection_only" if representation_mode == "final_last_hidden" else "target_remainder_execution_to_logits",
            "shared_layer_role": "chunk_selection_and_metadata_only" if representation_mode == "final_last_hidden" else "relative_depth_and_layer_mapping",
            "representation_mode": representation_mode,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "max_length": int(max_length) if max_length is not None else None,
            "chunk_ids": selected_chunk_ids,
        },
    }


def verify_model_stitching(
    *,
    source_model_id: str,
    target_model_id: str,
    calibration_jsonl: str | Path,
    output_path: str | Path,
    shared_geometry_path: str | Path | None = None,
    shared_layer: int | None = None,
    max_examples: int = 128,
    train_fraction: float = 0.75,
    seed: int = 17,
    batch_size: int = 8,
    max_length: int | None = 1024,
    ridge_lambda: float = 1e-4,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    representation_mode: str = "final_last_hidden",
) -> dict[str, object]:
    records = read_calibration_records(calibration_jsonl, max_examples=0)
    shared_geometry = SharedLatentGeometry.load(shared_geometry_path) if shared_geometry_path else None
    selected_chunk_ids = _select_chunk_ids(
        records,
        shared_geometry=shared_geometry,
        shared_layer=shared_layer,
        max_examples=max_examples,
    )
    texts = _texts_for_chunk_ids(records, selected_chunk_ids)

    source_adapter = HFCausalLMAdapter(
        source_model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    target_adapter = HFCausalLMAdapter(
        target_model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    source_layer_idx = resolve_model_layer_index(
        shared_geometry=shared_geometry,
        shared_layer=shared_layer,
        model_num_layers=int(getattr(source_adapter, "num_layers", 0)),
    ) if representation_mode == "layer_last_hidden_continuation" else None
    target_layer_idx = resolve_model_layer_index(
        shared_geometry=shared_geometry,
        shared_layer=shared_layer,
        model_num_layers=int(getattr(target_adapter, "num_layers", 0)),
    ) if representation_mode == "layer_last_hidden_continuation" else None

    source_batch = _collect_sequence_representations(
        source_adapter,
        texts,
        batch_size=batch_size,
        layers=[int(source_layer_idx)] if source_layer_idx is not None else None,
        max_length=max_length,
    )
    target_batch = _collect_sequence_representations(
        target_adapter,
        texts,
        batch_size=batch_size,
        layers=[int(target_layer_idx)] if target_layer_idx is not None else None,
        capture_full_sequences=representation_mode == "layer_last_hidden_continuation",
        max_length=max_length,
    )
    effective_shared_layer = int(shared_layer) if shared_layer is not None else (max(shared_geometry.layers) if shared_geometry else None)
    report = build_stitching_report_from_batches(
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        calibration_jsonl=calibration_jsonl,
        selected_chunk_ids=selected_chunk_ids,
        source_batch=source_batch,
        target_batch=target_batch,
        target_adapter=target_adapter,
        shared_geometry_path=shared_geometry_path,
        shared_layer=effective_shared_layer,
        source_layer_idx=source_layer_idx,
        target_layer_idx=target_layer_idx,
        train_fraction=train_fraction,
        seed=seed,
        ridge_lambda=ridge_lambda,
        batch_size=batch_size,
        max_length=max_length,
        representation_mode=representation_mode,
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify cross-model transfer by fitting an affine stitch from one model's final hidden states to another's."
    )
    parser.add_argument("output", help="Output JSON report path")
    parser.add_argument("--source-model", required=True, help="Hugging Face source model id")
    parser.add_argument("--target-model", required=True, help="Hugging Face target model id")
    parser.add_argument("--calibration-jsonl", required=True, help="Calibration JSONL with chunk_id/text fields")
    parser.add_argument("--shared-geometry", default="", help="Optional SharedLatentGeometry artifact used to choose shared chunk ids")
    parser.add_argument("--shared-layer", type=int, default=None, help="Optional layer index inside the SharedLatentGeometry artifact")
    parser.add_argument("--max-examples", type=int, default=128, help="Maximum number of chunk examples to evaluate")
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Train split fraction for fitting the stitch map")
    parser.add_argument("--seed", type=int, default=17, help="Random split seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Forward-pass batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Optional tokenizer truncation length")
    parser.add_argument("--ridge-lambda", type=float, default=1e-4, help="Ridge penalty for the affine stitch fit")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading Hugging Face models")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype name passed to the adapters")
    parser.add_argument(
        "--representation-mode",
        choices=["final_last_hidden", "layer_last_hidden_continuation"],
        default="final_last_hidden",
        help="Whether to stitch final hidden states directly or continue the target from a selected layer after replacing its last-position hidden state",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_model_stitching(
        source_model_id=args.source_model,
        target_model_id=args.target_model,
        calibration_jsonl=args.calibration_jsonl,
        output_path=args.output,
        shared_geometry_path=args.shared_geometry or None,
        shared_layer=args.shared_layer,
        max_examples=args.max_examples,
        train_fraction=args.train_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        ridge_lambda=args.ridge_lambda,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=args.torch_dtype,
        representation_mode=args.representation_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
