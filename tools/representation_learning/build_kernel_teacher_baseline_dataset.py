#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import KernelTeacherDataset
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import KernelTeacherDataset  # type: ignore[no-redef]


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 0.0)
    total = float(clipped.sum())
    if total <= 1.0e-8:
        return np.zeros_like(clipped, dtype=np.float32)
    return (clipped / total).astype(np.float32)


def _candidate_embeddings(example: dict[str, Any], embedding_dim: int) -> np.ndarray:
    raw = np.asarray(example.get("candidate_embeddings", []), dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, int(embedding_dim)), dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < int(embedding_dim):
        raw = np.pad(raw, ((0, 0), (0, int(embedding_dim) - raw.shape[1])))
    elif raw.shape[1] > int(embedding_dim):
        raw = raw[:, : int(embedding_dim)]
    return raw.astype(np.float32, copy=False)


def _mean_source_weights(teacher: KernelTeacherDataset) -> dict[str, float]:
    totals = {model_id: 0.0 for model_id in teacher.source_models}
    count = 0
    for example in teacher.examples:
        source_weights = example.get("source_model_weights", {})
        if not isinstance(source_weights, dict):
            continue
        for model_id in teacher.source_models:
            totals[model_id] += float(source_weights.get(model_id, 0.0))
        count += 1
    if count <= 0:
        uniform = 1.0 / max(len(teacher.source_models), 1)
        return {model_id: uniform for model_id in teacher.source_models}
    return {
        model_id: float(total / count)
        for model_id, total in totals.items()
    }


def _build_candidate_weights(
    *,
    mode: str,
    example: dict[str, Any],
    source_models: list[str],
    fixed_source_model: str | None,
    static_mix_strategy: str,
    static_mix_source_weights: dict[str, float] | None,
) -> np.ndarray:
    candidate_model_ids = [str(value) for value in example.get("candidate_model_ids", [])]
    if not candidate_model_ids:
        raise ValueError(f"Example {example.get('example_id', '')!r} has no candidate_model_ids")
    weights = np.zeros((len(candidate_model_ids),), dtype=np.float32)
    if mode == "fixed_source":
        if not fixed_source_model:
            raise ValueError("fixed_source mode requires fixed_source_model")
        if fixed_source_model not in candidate_model_ids:
            raise ValueError(
                f"Fixed source model {fixed_source_model!r} is not present in candidates for "
                f"example {example.get('example_id', '')!r}"
            )
        weights[candidate_model_ids.index(fixed_source_model)] = 1.0
        return weights
    if mode == "winner_only":
        winner_model = str(example.get("winner_model", ""))
        if winner_model not in candidate_model_ids:
            raise ValueError(
                f"Winner model {winner_model!r} is not present in candidates for "
                f"example {example.get('example_id', '')!r}"
            )
        weights[candidate_model_ids.index(winner_model)] = 1.0
        return weights
    if mode != "static_mix":
        raise ValueError(f"Unsupported mode: {mode}")
    if static_mix_strategy == "equal":
        return np.full((len(candidate_model_ids),), 1.0 / len(candidate_model_ids), dtype=np.float32)
    if static_mix_strategy != "teacher_mean":
        raise ValueError(f"Unsupported static_mix_strategy: {static_mix_strategy}")
    if static_mix_source_weights is None:
        raise ValueError("teacher_mean static mix requires static_mix_source_weights")
    for idx, model_id in enumerate(candidate_model_ids):
        weights[idx] = float(static_mix_source_weights.get(model_id, 0.0))
    normalized = _normalize_weights(weights)
    if float(normalized.sum()) <= 1.0e-8:
        return np.full((len(candidate_model_ids),), 1.0 / len(candidate_model_ids), dtype=np.float32)
    return normalized


def build_kernel_teacher_baseline_dataset(
    *,
    output_path: str | Path,
    kernel_teacher_dataset: str | Path,
    mode: str,
    fixed_source_model: str | None = None,
    static_mix_strategy: str = "equal",
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    input_path = Path(kernel_teacher_dataset).resolve()
    teacher = KernelTeacherDataset.load(input_path)
    mode = str(mode).strip().lower()
    static_mix_strategy = str(static_mix_strategy).strip().lower()

    if mode == "fixed_source":
        if not fixed_source_model:
            raise ValueError("fixed_source mode requires fixed_source_model")
        if fixed_source_model not in teacher.source_models:
            raise ValueError(
                f"Fixed source model {fixed_source_model!r} is not one of {teacher.source_models}"
            )

    static_mix_source_weights = _mean_source_weights(teacher) if mode == "static_mix" and static_mix_strategy == "teacher_mean" else None

    baseline_examples: list[dict[str, Any]] = []
    mean_winner_probability = 0.0
    teacher_matches = 0
    predicted_counts = {model_id: 0 for model_id in teacher.source_models}
    for example in teacher.examples:
        candidate_model_ids = [str(value) for value in example.get("candidate_model_ids", [])]
        candidate_weights = _build_candidate_weights(
            mode=mode,
            example=example,
            source_models=list(teacher.source_models),
            fixed_source_model=fixed_source_model,
            static_mix_strategy=static_mix_strategy,
            static_mix_source_weights=static_mix_source_weights,
        )
        candidate_weights = _normalize_weights(candidate_weights)
        candidate_embeddings = _candidate_embeddings(example, teacher.embedding_dim)
        if candidate_embeddings.shape[0] == candidate_weights.shape[0] and candidate_embeddings.shape[0] > 0:
            cleared_embedding = np.sum(candidate_embeddings * candidate_weights[:, None], axis=0).astype(np.float32)
        else:
            cleared_embedding = np.asarray(example.get("cleared_embedding", []), dtype=np.float32).reshape(-1)
            if cleared_embedding.shape[0] < int(teacher.embedding_dim):
                cleared_embedding = np.pad(cleared_embedding, (0, int(teacher.embedding_dim) - cleared_embedding.shape[0]))
            elif cleared_embedding.shape[0] > int(teacher.embedding_dim):
                cleared_embedding = cleared_embedding[: int(teacher.embedding_dim)]
        predicted_idx = int(np.argmax(candidate_weights))
        predicted_model = str(candidate_model_ids[predicted_idx])
        predicted_counts[predicted_model] = predicted_counts.get(predicted_model, 0) + 1
        winner_model = str(example.get("winner_model", ""))
        winner_probability = 0.0
        if winner_model in candidate_model_ids:
            winner_probability = float(candidate_weights[candidate_model_ids.index(winner_model)])
        teacher_matches_winner = bool(predicted_model == winner_model)
        teacher_matches += int(teacher_matches_winner)
        mean_winner_probability += winner_probability

        source_weight_map = {model_id: 0.0 for model_id in teacher.source_models}
        for model_id, weight in zip(candidate_model_ids, candidate_weights.tolist(), strict=True):
            source_weight_map[str(model_id)] = float(weight)

        payload = dict(example)
        payload["predicted_model"] = predicted_model
        payload["teacher_matches_winner"] = teacher_matches_winner
        payload["winner_probability"] = float(winner_probability)
        payload["candidate_weights"] = [float(value) for value in candidate_weights.tolist()]
        payload["source_model_weights"] = source_weight_map
        payload["cleared_embedding"] = cleared_embedding.tolist()
        payload["teacher_mode"] = mode
        if fixed_source_model:
            payload["teacher_fixed_source_model"] = str(fixed_source_model)
        if mode == "static_mix":
            payload["teacher_static_mix_strategy"] = static_mix_strategy
        baseline_examples.append(payload)

    example_count = max(len(baseline_examples), 1)
    summary = {
        "kernel_teacher_dataset": str(input_path),
        "example_count": int(len(baseline_examples)),
        "embedding_dim": int(teacher.embedding_dim),
        "source_models": list(teacher.source_models),
        "mode": mode,
        "fixed_source_model": fixed_source_model,
        "static_mix_strategy": static_mix_strategy if mode == "static_mix" else "",
        "static_mix_source_weights": static_mix_source_weights or {},
        "teacher_accuracy": float(teacher_matches / example_count),
        "mean_winner_probability": float(mean_winner_probability / example_count),
        "predicted_model_counts": predicted_counts,
    }

    artifact = KernelTeacherDataset(
        source_models=list(teacher.source_models),
        embedding_dim=int(teacher.embedding_dim),
        examples=baseline_examples,
        metadata={
            **teacher.metadata,
            "builder": "build_kernel_teacher_baseline_dataset",
            "kernel_teacher_dataset": str(input_path),
            "mode": mode,
            "fixed_source_model": fixed_source_model,
            "static_mix_strategy": static_mix_strategy if mode == "static_mix" else "",
            "static_mix_source_weights": static_mix_source_weights or {},
            "summary": summary,
        },
    )
    artifact.save(output_path)
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline kernel-teacher datasets from an existing ecology-teacher artifact.")
    parser.add_argument("output", help="Output .npz path for the baseline KernelTeacherDataset artifact")
    parser.add_argument("--kernel-teacher-dataset", required=True, help="Input KernelTeacherDataset artifact")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fixed_source", "winner_only", "static_mix"],
        help="Baseline teacher mode",
    )
    parser.add_argument(
        "--fixed-source-model",
        default="",
        help="Source model id for fixed_source mode",
    )
    parser.add_argument(
        "--static-mix-strategy",
        default="equal",
        choices=["equal", "teacher_mean"],
        help="Weight strategy for static_mix mode",
    )
    args = parser.parse_args()

    summary = build_kernel_teacher_baseline_dataset(
        output_path=args.output,
        kernel_teacher_dataset=args.kernel_teacher_dataset,
        mode=args.mode,
        fixed_source_model=args.fixed_source_model or None,
        static_mix_strategy=args.static_mix_strategy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
