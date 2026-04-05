#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import EcologyTrainingSet, KernelTeacherDataset
    from tools.representation_learning.train_ecology_model import (
        EcologyTransformer,
        ExampleRecord,
        _collate_batch,
        _context_feature_names,
        _evaluate_predictions,
        _example_to_record,
        _probe_types,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import EcologyTrainingSet, KernelTeacherDataset  # type: ignore[no-redef]
    from train_ecology_model import (  # type: ignore[no-redef]
        EcologyTransformer,
        ExampleRecord,
        _collate_batch,
        _context_feature_names,
        _evaluate_predictions,
        _example_to_record,
        _probe_types,
    )


def _apply_normalizer(record: ExampleRecord, normalizer: dict[str, Any]) -> ExampleRecord:
    candidate_mean = np.asarray(normalizer["candidate_mean"], dtype=np.float32)
    candidate_std = np.asarray(normalizer["candidate_std"], dtype=np.float32)
    context_mean = np.asarray(normalizer["context_mean"], dtype=np.float32)
    context_std = np.asarray(normalizer["context_std"], dtype=np.float32)
    candidate_std = np.where(candidate_std < 1e-6, 1.0, candidate_std)
    context_std = np.where(context_std < 1e-6, 1.0, context_std)
    return ExampleRecord(
        example_id=record.example_id,
        chunk_id=record.chunk_id,
        probe_id=record.probe_id,
        probe_type=record.probe_type,
        winner_model=record.winner_model,
        candidate_model_ids=record.candidate_model_ids,
        label_idx=record.label_idx,
        candidate_continuous=((record.candidate_continuous - candidate_mean) / candidate_std).astype(np.float32),
        candidate_onehot=record.candidate_onehot.astype(np.float32),
        context_continuous=((record.context_continuous - context_mean) / context_std).astype(np.float32),
        context_onehot=record.context_onehot.astype(np.float32),
        verification_confidence=record.verification_confidence,
        target_layer=record.target_layer,
        raw_candidates=record.raw_candidates,
    )


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict at {path}, got {type(checkpoint).__name__}")
    required = {
        "model_state_dict",
        "normalizer",
        "selected_feature_names",
        "context_feature_names",
        "source_models",
        "probe_types",
        "candidate_dim",
        "context_dim",
        "hidden_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "use_embeddings",
        "embedding_dim",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    return checkpoint


def _build_records(
    training_set: EcologyTrainingSet,
    *,
    checkpoint: dict[str, Any],
) -> list[ExampleRecord]:
    selected_feature_names = [str(value) for value in checkpoint["selected_feature_names"]]
    context_feature_names = [str(value) for value in checkpoint["context_feature_names"]]
    source_models = [str(value) for value in checkpoint["source_models"]]
    probe_types = [str(value) for value in checkpoint["probe_types"]]
    use_embeddings = bool(checkpoint["use_embeddings"])
    embedding_dim = int(checkpoint["embedding_dim"])
    raw_records = [
        _example_to_record(
            example,
            selected_feature_names=selected_feature_names,
            use_embeddings=use_embeddings,
            embedding_dim=embedding_dim,
            context_feature_names=context_feature_names,
            source_models=source_models,
            probe_types=probe_types,
        )
        for example in training_set.examples
    ]
    records = [record for record in raw_records if record is not None]
    normalizer = checkpoint["normalizer"]
    return [_apply_normalizer(record, normalizer) for record in records]


def build_kernel_teacher_dataset(
    *,
    output_path: str | Path,
    ecology_training_set: str | Path,
    ecology_checkpoint: str | Path,
    temperature: float = 1.0,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    checkpoint_path = Path(ecology_checkpoint).resolve()
    training_set_path = Path(ecology_training_set).resolve()
    training_set = EcologyTrainingSet.load(training_set_path)
    torch_device = torch.device(device)
    checkpoint = _load_checkpoint(checkpoint_path, torch_device)
    records = _build_records(training_set, checkpoint=checkpoint)
    if not records:
        raise ValueError("No valid ecology examples available for teacher export")

    model = EcologyTransformer(
        candidate_dim=int(checkpoint["candidate_dim"]),
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_heads=int(checkpoint["num_heads"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        max_tokens=max(len(record.candidate_model_ids) for record in records) + 1,
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    teacher_examples: list[dict[str, Any]] = []
    predicted_indices: list[int] = []
    mean_entropy = 0.0
    mean_winner_prob = 0.0
    batches_seen = 0

    with torch.no_grad():
        for start in range(0, len(records), max(int(batch_size), 1)):
            batch_records = records[start: start + max(int(batch_size), 1)]
            tensors = _collate_batch(batch_records, device=torch_device)
            logits = model(tensors["candidate_inputs"], tensors["candidate_mask"], tensors["context_inputs"])
            scaled_logits = logits / max(float(temperature), 1e-6)
            probs = F.softmax(scaled_logits, dim=-1)
            probs_np = probs.detach().cpu().numpy().astype(np.float32)
            predicted_indices.extend(probs.argmax(dim=1).detach().cpu().tolist())
            row_entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1)
            mean_entropy += float(row_entropy.mean().detach().cpu().item())
            batches_seen += 1
            for row_idx, record in enumerate(batch_records):
                count = len(record.candidate_model_ids)
                weights = probs_np[row_idx, :count]
                weights = weights / max(float(weights.sum()), 1e-8)
                winner_prob = float(weights[int(record.label_idx)])
                mean_winner_prob += winner_prob
                predicted_idx = int(np.argmax(weights))
                predicted_model = str(record.candidate_model_ids[predicted_idx])
                candidate_embeddings = []
                for candidate in record.raw_candidates:
                    embedding = np.asarray(candidate.get("embedding", []), dtype=np.float32).reshape(-1)
                    if embedding.shape[0] < int(training_set.embedding_dim):
                        embedding = np.pad(embedding, (0, int(training_set.embedding_dim) - embedding.shape[0]))
                    elif embedding.shape[0] > int(training_set.embedding_dim):
                        embedding = embedding[: int(training_set.embedding_dim)]
                    candidate_embeddings.append(embedding.astype(np.float32))
                if candidate_embeddings:
                    candidate_embeddings_np = np.stack(candidate_embeddings, axis=0)
                    cleared_embedding = np.sum(candidate_embeddings_np * weights[:, None], axis=0).astype(np.float32)
                    candidate_embeddings_payload = [embedding.tolist() for embedding in candidate_embeddings_np]
                else:
                    cleared_embedding = np.zeros((int(training_set.embedding_dim),), dtype=np.float32)
                    candidate_embeddings_payload = []
                source_weight_map = {model_id: 0.0 for model_id in training_set.source_models}
                for model_id, weight in zip(record.candidate_model_ids, weights.tolist(), strict=True):
                    source_weight_map[str(model_id)] = float(weight)
                teacher_examples.append(
                    {
                        "example_id": record.example_id,
                        "chunk_id": record.chunk_id,
                        "probe_id": record.probe_id,
                        "probe_type": record.probe_type,
                        "target_layer": int(record.target_layer),
                        "verification_confidence": float(record.verification_confidence),
                        "winner_model": record.winner_model,
                        "predicted_model": predicted_model,
                        "teacher_matches_winner": bool(predicted_model == record.winner_model),
                        "winner_probability": winner_prob,
                        "candidate_model_ids": list(record.candidate_model_ids),
                        "candidate_weights": [float(value) for value in weights.tolist()],
                        "source_model_weights": source_weight_map,
                        "candidate_embeddings": candidate_embeddings_payload,
                        "cleared_embedding": cleared_embedding.tolist(),
                    }
                )

    metrics = _evaluate_predictions(records, predicted_indices)
    example_count = max(len(records), 1)
    summary = {
        "ecology_training_set": str(training_set_path),
        "ecology_checkpoint": str(checkpoint_path),
        "example_count": int(len(records)),
        "source_models": list(training_set.source_models),
        "embedding_dim": int(training_set.embedding_dim),
        "temperature": float(temperature),
        "teacher_metrics": metrics,
        "mean_teacher_entropy": float(mean_entropy / max(batches_seen, 1)),
        "mean_winner_probability": float(mean_winner_prob / example_count),
        "feature_mode": str(checkpoint.get("feature_mode", "")),
    }
    artifact = KernelTeacherDataset(
        source_models=list(training_set.source_models),
        embedding_dim=int(training_set.embedding_dim),
        examples=teacher_examples,
        metadata={
            **summary,
            "selected_feature_names": list(checkpoint["selected_feature_names"]),
            "context_feature_names": list(checkpoint["context_feature_names"]),
            "use_embeddings": bool(checkpoint["use_embeddings"]),
        },
    )
    artifact.save(output_path)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable kernel-teacher dataset from a trained ecology checkpoint and EcologyTrainingSet artifact.")
    parser.add_argument("output", help="Output .npz path for the KernelTeacherDataset artifact")
    parser.add_argument("--ecology-training-set", required=True, help="Input EcologyTrainingSet artifact")
    parser.add_argument("--ecology-checkpoint", required=True, help="Trained ecology model checkpoint (.pt)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature applied to teacher routing logits")
    parser.add_argument("--batch-size", type=int, default=128, help="Teacher export batch size")
    parser.add_argument("--device", default="cpu", help="Torch device for teacher export")
    args = parser.parse_args()

    summary = build_kernel_teacher_dataset(
        output_path=args.output,
        ecology_training_set=args.ecology_training_set,
        ecology_checkpoint=args.ecology_checkpoint,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
