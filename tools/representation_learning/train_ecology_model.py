#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import EcologyTrainingSet
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import EcologyTrainingSet  # type: ignore[no-redef]


LOSS_FEATURES = {
    "loss",
    "loss_gap_to_best",
    "loss_gap_to_mean",
    "loss_rank_frac",
    "mean_loss",
    "disagreement_score",
}

FORWARD_SIGNATURE_FEATURES = {
    "has_forward_signature",
    "last_token_entropy",
    "sequence_mean_entropy",
    "last_token_top1_prob",
    "last_token_margin",
    "last_token_topk_mass",
    "attention_entropy",
    "attention_peak_frac",
    "cross_model_topk_jaccard_mean",
    "cross_model_topk_jaccard_max",
    "cross_model_topk_prob_l1_mean",
}

STRUCTURE_FEATURES = {
    "relative_depth",
    "has_projection",
    "projection_norm",
    "projection_abs_mean",
    "projection_peak_frac",
    "concept_sharpness_mean",
    "concept_sharpness_max",
    *FORWARD_SIGNATURE_FEATURES,
}

MODEL_SIZE_FEATURES = {
    "log10_num_parameters",
    "log2_hidden_dim",
}

GEOMETRY_FEATURES = STRUCTURE_FEATURES - FORWARD_SIGNATURE_FEATURES
DISAGREEMENT_FACTOR_PREFIX = "factor_"

FEATURE_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {
        "exclude_features": set(),
        "use_embeddings": True,
    },
    "no_loss": {
        "exclude_features": set(LOSS_FEATURES),
        "use_embeddings": True,
    },
    "structure_only": {
        "exclude_features": set(LOSS_FEATURES | MODEL_SIZE_FEATURES) | {"num_models_present"},
        "use_embeddings": True,
    },
    "structure_no_embeddings": {
        "exclude_features": set(LOSS_FEATURES | MODEL_SIZE_FEATURES) | {"num_models_present"},
        "use_embeddings": False,
    },
    "geometry_only": {
        "exclude_features": set(LOSS_FEATURES | MODEL_SIZE_FEATURES | FORWARD_SIGNATURE_FEATURES) | {"num_models_present"},
        "use_embeddings": True,
    },
    "forward_only": {
        "exclude_features": set(),
        "use_embeddings": False,
        "feature_allowlist": set(FORWARD_SIGNATURE_FEATURES),
    },
    "factor_only": {
        "exclude_features": set(),
        "use_embeddings": False,
        "feature_prefix_allowlist": [DISAGREEMENT_FACTOR_PREFIX],
    },
    "embedding_only": {
        "exclude_features": set(),
        "use_embeddings": True,
        "feature_allowlist": set(),
    },
}


@dataclass
class ExampleRecord:
    example_id: str
    chunk_id: str
    probe_id: str
    probe_type: str
    winner_model: str
    candidate_model_ids: list[str]
    label_idx: int
    candidate_continuous: np.ndarray
    candidate_onehot: np.ndarray
    context_continuous: np.ndarray
    context_onehot: np.ndarray
    verification_confidence: float
    target_layer: int
    raw_candidates: list[dict[str, Any]]


class EcologyTransformer(nn.Module):
    def __init__(
        self,
        *,
        candidate_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_proj = nn.Linear(candidate_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.position = nn.Embedding(max_tokens, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 4, 64),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.candidate_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        candidate_inputs: torch.Tensor,
        candidate_mask: torch.Tensor,
        context_inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_candidates, _ = candidate_inputs.shape
        device = candidate_inputs.device
        candidate_tokens = self.candidate_proj(candidate_inputs)
        context_token = self.context_proj(context_inputs).unsqueeze(1)
        tokens = torch.cat([context_token, candidate_tokens], dim=1)
        positions = torch.arange(tokens.shape[1], device=device, dtype=torch.long)
        tokens = tokens + self.position(positions).unsqueeze(0)
        src_key_padding_mask = torch.cat(
            [torch.zeros((batch_size, 1), dtype=torch.bool, device=device), ~candidate_mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        context_encoded = encoded[:, :1, :]
        candidate_encoded = encoded[:, 1:, :]
        gated_candidates = candidate_encoded + torch.tanh(self.context_gate(context_encoded)).expand_as(candidate_encoded)
        scores = self.candidate_score(gated_candidates).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask, -1e9)
        return scores


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _feature_selection(feature_names: list[str], mode: str) -> tuple[list[str], bool]:
    config = FEATURE_MODE_CONFIGS[mode]
    allowlist = config.get("feature_allowlist")
    prefix_allowlist = tuple(str(prefix) for prefix in config.get("feature_prefix_allowlist", []))
    excluded = set(config.get("exclude_features", set()))
    selected = []
    for name in feature_names:
        if prefix_allowlist and not any(str(name).startswith(prefix) for prefix in prefix_allowlist):
            if allowlist is None or name not in allowlist:
                continue
        if allowlist is not None and name not in allowlist:
            continue
        if name in excluded:
            continue
        selected.append(name)
    return selected, bool(config.get("use_embeddings", True))


def _context_feature_names(examples: list[dict[str, Any]], excluded: set[str]) -> list[str]:
    names: set[str] = set()
    for example in examples:
        payload = example.get("context_features", {})
        if isinstance(payload, dict):
            for key in payload:
                if key not in excluded:
                    names.add(str(key))
    return sorted(names)


def _probe_types(examples: list[dict[str, Any]]) -> list[str]:
    return sorted({str(example.get("probe_type", "unknown")) for example in examples})


def _example_to_record(
    example: dict[str, Any],
    *,
    selected_feature_names: list[str],
    use_embeddings: bool,
    embedding_dim: int,
    context_feature_names: list[str],
    source_models: list[str],
    probe_types: list[str],
) -> ExampleRecord | None:
    candidates = list(example.get("candidates", []))
    if len(candidates) < 2:
        return None
    winner_model = str(example["winner_model"])
    candidate_model_ids = [str(candidate["model_id"]) for candidate in candidates]
    if winner_model not in candidate_model_ids:
        return None
    candidate_continuous_rows: list[np.ndarray] = []
    candidate_onehot_rows: list[np.ndarray] = []
    model_to_index = {model_id: idx for idx, model_id in enumerate(source_models)}
    for candidate in candidates:
        features = candidate.get("features", {})
        scalar_values = [float(features.get(name, 0.0)) for name in selected_feature_names]
        if use_embeddings:
            embedding = [float(value) for value in candidate.get("embedding", [])]
            if len(embedding) < embedding_dim:
                embedding = embedding + [0.0] * (embedding_dim - len(embedding))
            else:
                embedding = embedding[:embedding_dim]
            scalar_values.extend(embedding)
        candidate_continuous_rows.append(np.asarray(scalar_values, dtype=np.float32))
        onehot = np.zeros(len(source_models), dtype=np.float32)
        model_idx = model_to_index.get(str(candidate["model_id"]))
        if model_idx is not None:
            onehot[model_idx] = 1.0
        candidate_onehot_rows.append(onehot)
    context_payload = example.get("context_features", {})
    context_values = [float(context_payload.get(name, 0.0)) for name in context_feature_names]
    context_values.extend(
        [
            float(example.get("target_layer", 0)),
            float(example.get("verification_confidence", 0.0)),
        ]
    )
    context_continuous = np.asarray(context_values, dtype=np.float32)
    probe_type_to_index = {probe_type: idx for idx, probe_type in enumerate(probe_types)}
    context_onehot = np.zeros(len(probe_types), dtype=np.float32)
    probe_idx = probe_type_to_index.get(str(example.get("probe_type", "unknown")))
    if probe_idx is not None:
        context_onehot[probe_idx] = 1.0
    return ExampleRecord(
        example_id=str(example["example_id"]),
        chunk_id=str(example.get("chunk_id", "")),
        probe_id=str(example.get("probe_id", "")),
        probe_type=str(example.get("probe_type", "unknown")),
        winner_model=winner_model,
        candidate_model_ids=candidate_model_ids,
        label_idx=candidate_model_ids.index(winner_model),
        candidate_continuous=np.stack(candidate_continuous_rows, axis=0),
        candidate_onehot=np.stack(candidate_onehot_rows, axis=0),
        context_continuous=context_continuous,
        context_onehot=context_onehot,
        verification_confidence=float(example.get("verification_confidence", 0.0)),
        target_layer=int(example.get("target_layer", 0)),
        raw_candidates=candidates,
    )


def _normalize_records(train_records: list[ExampleRecord], val_records: list[ExampleRecord]) -> tuple[list[ExampleRecord], list[ExampleRecord], dict[str, Any]]:
    if not train_records:
        raise ValueError("train_records must not be empty")
    candidate_continuous = np.concatenate([record.candidate_continuous for record in train_records], axis=0)
    context_continuous = np.stack([record.context_continuous for record in train_records], axis=0)
    candidate_mean = candidate_continuous.mean(axis=0)
    candidate_std = candidate_continuous.std(axis=0)
    context_mean = context_continuous.mean(axis=0)
    context_std = context_continuous.std(axis=0)
    candidate_std = np.where(candidate_std < 1e-6, 1.0, candidate_std)
    context_std = np.where(context_std < 1e-6, 1.0, context_std)

    def _apply(records: list[ExampleRecord]) -> list[ExampleRecord]:
        output: list[ExampleRecord] = []
        for record in records:
            output.append(
                ExampleRecord(
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
            )
        return output

    normalizer = {
        "candidate_mean": candidate_mean.tolist(),
        "candidate_std": candidate_std.tolist(),
        "context_mean": context_mean.tolist(),
        "context_std": context_std.tolist(),
    }
    return _apply(train_records), _apply(val_records), normalizer


def _split_records(records: list[ExampleRecord], val_fraction: float) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    train_records: list[ExampleRecord] = []
    val_records: list[ExampleRecord] = []
    for record in records:
        bucket = _stable_unit_interval(record.chunk_id or record.probe_id or record.example_id)
        if bucket < val_fraction:
            val_records.append(record)
        else:
            train_records.append(record)
    if not train_records or not val_records:
        raise ValueError("Need non-empty train and validation splits")
    return train_records, val_records


def _collate_batch(records: list[ExampleRecord], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = len(records)
    max_candidates = max(len(record.candidate_model_ids) for record in records)
    candidate_dim = records[0].candidate_continuous.shape[1] + records[0].candidate_onehot.shape[1]
    context_dim = records[0].context_continuous.shape[0] + records[0].context_onehot.shape[0]
    candidate_inputs = torch.zeros((batch_size, max_candidates, candidate_dim), dtype=torch.float32, device=device)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)
    context_inputs = torch.zeros((batch_size, context_dim), dtype=torch.float32, device=device)
    labels = torch.zeros((batch_size,), dtype=torch.long, device=device)
    sample_weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
    for idx, record in enumerate(records):
        count = len(record.candidate_model_ids)
        candidate_payload = np.concatenate([record.candidate_continuous, record.candidate_onehot], axis=1)
        context_payload = np.concatenate([record.context_continuous, record.context_onehot], axis=0)
        candidate_inputs[idx, :count] = torch.from_numpy(candidate_payload)
        candidate_mask[idx, :count] = True
        context_inputs[idx] = torch.from_numpy(context_payload)
        labels[idx] = int(record.label_idx)
        sample_weights[idx] = float(record.verification_confidence) + 0.25
    return {
        "candidate_inputs": candidate_inputs,
        "candidate_mask": candidate_mask,
        "context_inputs": context_inputs,
        "labels": labels,
        "sample_weights": sample_weights,
    }


def _evaluate_predictions(records: list[ExampleRecord], predicted_indices: list[int]) -> dict[str, Any]:
    if not records:
        return {"accuracy": 0.0}
    correct = 0
    by_model: dict[str, dict[str, int]] = {}
    for record, pred_idx in zip(records, predicted_indices, strict=True):
        winner = record.winner_model
        predicted_model = record.candidate_model_ids[pred_idx]
        if predicted_model == winner:
            correct += 1
        payload = by_model.setdefault(winner, {"total": 0, "correct": 0})
        payload["total"] += 1
        if predicted_model == winner:
            payload["correct"] += 1
    by_model_accuracy = {
        model_id: (payload["correct"] / max(payload["total"], 1))
        for model_id, payload in by_model.items()
    }
    return {
        "accuracy": correct / max(len(records), 1),
        "by_winner_model": by_model_accuracy,
    }


def _baseline_predictions(records: list[ExampleRecord], baseline: str, majority_model: str | None = None) -> list[int]:
    predicted: list[int] = []
    for record in records:
        if baseline == "argmin_loss":
            scores = [float(candidate["features"].get("loss", math.inf)) for candidate in record.raw_candidates]
            predicted.append(int(np.argmin(np.asarray(scores, dtype=np.float32))))
            continue
        if baseline == "argmax_projection_norm":
            scores = [float(candidate["features"].get("projection_norm", 0.0)) for candidate in record.raw_candidates]
            predicted.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
            continue
        if baseline == "argmax_concept_sharpness":
            scores = [float(candidate["features"].get("concept_sharpness_max", 0.0)) for candidate in record.raw_candidates]
            predicted.append(int(np.argmax(np.asarray(scores, dtype=np.float32))))
            continue
        if baseline == "majority_model" and majority_model is not None:
            predicted.append(record.candidate_model_ids.index(majority_model))
            continue
        raise ValueError(f"Unsupported baseline: {baseline}")
    return predicted


def _majority_model(records: list[ExampleRecord]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.winner_model] = counts.get(record.winner_model, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _iterate_minibatches(records: list[ExampleRecord], batch_size: int, seed: int) -> list[list[ExampleRecord]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return [shuffled[idx: idx + batch_size] for idx in range(0, len(shuffled), batch_size)]


def _run_epoch(
    model: EcologyTransformer,
    optimizer: torch.optim.Optimizer,
    records: list[ExampleRecord],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    class_weight_by_model: dict[str, float] | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in _iterate_minibatches(records, batch_size=batch_size, seed=seed):
        tensors = _collate_batch(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors["candidate_inputs"], tensors["candidate_mask"], tensors["context_inputs"])
        per_example = F.cross_entropy(logits, tensors["labels"], reduction="none")
        weights = tensors["sample_weights"]
        if class_weight_by_model:
            class_weights = torch.tensor(
                [float(class_weight_by_model.get(record.winner_model, 1.0)) for record in batch],
                dtype=torch.float32,
                device=device,
            )
            weights = weights * class_weights
        loss = (per_example * weights).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(np.asarray(losses, dtype=np.float32))) if losses else 0.0


def _predict(
    model: EcologyTransformer,
    records: list[ExampleRecord],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], float]:
    model.eval()
    predictions: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start: start + batch_size]
            tensors = _collate_batch(batch, device=device)
            logits = model(tensors["candidate_inputs"], tensors["candidate_mask"], tensors["context_inputs"])
            per_example = F.cross_entropy(logits, tensors["labels"], reduction="none")
            loss = (per_example * tensors["sample_weights"]).mean()
            losses.append(float(loss.detach().cpu().item()))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    mean_loss = float(np.mean(np.asarray(losses, dtype=np.float32))) if losses else 0.0
    return predictions, mean_loss


def _count_winners(records: list[ExampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.winner_model] = counts.get(record.winner_model, 0) + 1
    return counts


def _class_weight_map(records: list[ExampleRecord], power: float) -> dict[str, float]:
    counts = _count_winners(records)
    if not counts or power <= 0.0:
        return {model_id: 1.0 for model_id in counts}
    total = float(sum(counts.values()))
    num_classes = float(len(counts))
    weights: dict[str, float] = {}
    for model_id, count in counts.items():
        balanced = total / max(num_classes * float(count), 1.0)
        weights[model_id] = float(balanced ** power)
    return weights


def train_one_mode(
    training_set: EcologyTrainingSet,
    *,
    mode: str,
    output_dir: Path,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    val_fraction: float,
    seed: int,
    class_balance_power: float = 0.0,
    torch_num_threads: int | None = None,
) -> dict[str, Any]:
    if mode not in FEATURE_MODE_CONFIGS:
        raise ValueError(f"Unsupported feature mode: {mode}")
    selected_feature_names, use_embeddings = _feature_selection(training_set.feature_names, mode)
    excluded = set(training_set.feature_names) - set(selected_feature_names)
    context_feature_names = _context_feature_names(training_set.examples, excluded=excluded)
    probe_types = _probe_types(training_set.examples)
    source_models = list(training_set.source_models)
    raw_records = [
        _example_to_record(
            example,
            selected_feature_names=selected_feature_names,
            use_embeddings=use_embeddings,
            embedding_dim=int(training_set.embedding_dim),
            context_feature_names=context_feature_names,
            source_models=source_models,
            probe_types=probe_types,
        )
        for example in training_set.examples
    ]
    records = [record for record in raw_records if record is not None]
    train_records_raw, val_records_raw = _split_records(records, val_fraction=val_fraction)
    train_records, val_records, normalizer = _normalize_records(train_records_raw, val_records_raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch_num_threads is not None and int(torch_num_threads) > 0:
        torch.set_num_threads(int(torch_num_threads))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    candidate_dim = train_records[0].candidate_continuous.shape[1] + train_records[0].candidate_onehot.shape[1]
    context_dim = train_records[0].context_continuous.shape[0] + train_records[0].context_onehot.shape[0]
    max_candidates = max(len(record.candidate_model_ids) for record in train_records + val_records)
    model = EcologyTransformer(
        candidate_dim=candidate_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_tokens=max_candidates + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    class_weight_by_model = _class_weight_map(train_records, power=class_balance_power)

    best_state = None
    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []
    for epoch_idx in range(epochs):
        train_loss = _run_epoch(
            model,
            optimizer,
            train_records,
            batch_size=batch_size,
            device=device,
            seed=seed + epoch_idx,
            class_weight_by_model=class_weight_by_model,
        )
        train_pred, train_eval_loss = _predict(model, train_records, batch_size=batch_size, device=device)
        val_pred, val_eval_loss = _predict(model, val_records, batch_size=batch_size, device=device)
        train_metrics = _evaluate_predictions(train_records, train_pred)
        val_metrics = _evaluate_predictions(val_records, val_pred)
        history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": val_eval_loss,
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )
        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_state = {
                "model_state_dict": model.state_dict(),
                "history": history,
            }

    if best_state is None:
        raise RuntimeError("No model state captured during training")
    model.load_state_dict(best_state["model_state_dict"])
    train_pred, train_eval_loss = _predict(model, train_records, batch_size=batch_size, device=device)
    val_pred, val_eval_loss = _predict(model, val_records, batch_size=batch_size, device=device)
    train_metrics = _evaluate_predictions(train_records, train_pred)
    val_metrics = _evaluate_predictions(val_records, val_pred)

    majority_model = _majority_model(train_records)
    baselines = {}
    for baseline_name in ("argmin_loss", "argmax_projection_norm", "argmax_concept_sharpness", "majority_model"):
        baseline_pred = _baseline_predictions(val_records, baseline_name, majority_model=majority_model)
        baselines[baseline_name] = _evaluate_predictions(val_records, baseline_pred)

    mistakes = []
    for record, pred_idx in zip(val_records, val_pred, strict=True):
        predicted_model = record.candidate_model_ids[pred_idx]
        if predicted_model == record.winner_model:
            continue
        mistakes.append(
            {
                "example_id": record.example_id,
                "chunk_id": record.chunk_id,
                "probe_type": record.probe_type,
                "winner_model": record.winner_model,
                "predicted_model": predicted_model,
                "candidate_model_ids": record.candidate_model_ids,
                "target_layer": record.target_layer,
            }
        )
        if len(mistakes) >= 10:
            break

    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = mode_dir / "ecology_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer,
            "selected_feature_names": selected_feature_names,
            "context_feature_names": context_feature_names,
            "source_models": source_models,
            "probe_types": probe_types,
            "candidate_dim": candidate_dim,
            "context_dim": context_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_mode": mode,
            "use_embeddings": use_embeddings,
            "embedding_dim": int(training_set.embedding_dim if use_embeddings else 0),
        },
        checkpoint_path,
    )

    report = {
        "feature_mode": mode,
        "checkpoint_path": str(checkpoint_path),
        "selected_feature_names": selected_feature_names,
        "excluded_feature_names": sorted(excluded),
        "use_embeddings": use_embeddings,
        "context_feature_names": context_feature_names,
        "source_models": source_models,
        "probe_types": probe_types,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "candidate_dim": candidate_dim,
        "context_dim": context_dim,
        "train_winner_counts": _count_winners(train_records),
        "val_winner_counts": _count_winners(val_records),
        "train_metrics": {**train_metrics, "loss": train_eval_loss},
        "val_metrics": {**val_metrics, "loss": val_eval_loss},
        "baseline_metrics": baselines,
        "majority_model": majority_model,
        "class_balance_power": class_balance_power,
        "class_weight_by_model": class_weight_by_model,
        "history": history,
        "sample_mistakes": mistakes,
    }
    (mode_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small ecology model to predict verified cross-model winners from typed activation events.")
    parser.add_argument("output_dir", help="Directory for ecology-model checkpoints and reports")
    parser.add_argument("ecology_training_set", help="Input ecology training-set artifact")
    parser.add_argument("--feature-modes", default="full,no_loss,structure_only,embedding_only")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_set = EcologyTrainingSet.load(args.ecology_training_set)
    modes = [mode.strip() for mode in str(args.feature_modes).split(",") if mode.strip()]
    reports = []
    for offset, mode in enumerate(modes):
        report = train_one_mode(
            training_set,
            mode=mode,
            output_dir=output_dir,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed + offset,
            class_balance_power=args.class_balance_power,
            torch_num_threads=args.torch_num_threads,
        )
        reports.append(report)
        partial_summary = {
            "output_dir": str(output_dir),
            "ecology_training_set": str(Path(args.ecology_training_set).resolve()),
            "feature_modes": modes,
            "completed_feature_modes": [item["feature_mode"] for item in reports],
            "mode_reports": reports,
            "torch_num_threads": args.torch_num_threads,
        }
        (output_dir / "summary.partial.json").write_text(json.dumps(partial_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "ecology_training_set": str(Path(args.ecology_training_set).resolve()),
        "feature_modes": modes,
        "mode_reports": reports,
        "torch_num_threads": args.torch_num_threads,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
