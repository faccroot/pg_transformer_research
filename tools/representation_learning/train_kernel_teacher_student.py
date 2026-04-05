#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import sentencepiece as spm  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from logic_register_mlx import CastedLinear  # noqa: E402
import train_gpt_mlx as base  # noqa: E402
try:
    from tools.representation_learning.schemas import KernelTeacherTextDataset  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import KernelTeacherTextDataset  # type: ignore[no-redef]


def apply_overrides_from_config(argv: list[str]) -> None:
    config_path: str | None = None
    passthrough = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        passthrough.append(arg)
        i += 1
    if config_path is None:
        return
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", {})
    if env_payload:
        if not isinstance(env_payload, dict):
            raise SystemExit(f"Config env payload must be a JSON object: {config_path}")
        for key, value in env_payload.items():
            if value is None:
                os.environ.pop(str(key), None)
            else:
                os.environ[str(key)] = str(value)
    args_payload = payload.get("args", {})
    if args_payload and not isinstance(args_payload, dict):
        raise SystemExit(f"Config args payload must be a JSON object: {config_path}")
    expanded = list(passthrough)
    for key, value in args_payload.items():
        if value is None:
            continue
        option = f"--{str(key).strip()}"
        if isinstance(value, bool):
            if value:
                expanded.append(option)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                expanded.extend((option, str(item)))
            continue
        expanded.extend((option, str(value)))
    sys.argv[:] = expanded


apply_overrides_from_config(sys.argv)


@dataclass
class TeacherTextRecord:
    example_id: str
    chunk_id: str
    text: str
    token_ids: list[int]
    teacher_embedding: np.ndarray
    winner_model: str
    predicted_model: str
    winner_probability: float
    verification_confidence: float
    candidate_model_ids: list[str]
    candidate_weights: np.ndarray
    candidate_embeddings: np.ndarray


def masked_sequence_mean(hidden: mx.array, token_weights: mx.array) -> mx.array:
    weights = token_weights.astype(mx.float32)
    denom = mx.maximum(mx.sum(weights, axis=1, keepdims=True), mx.array(1.0e-6, dtype=mx.float32))
    return mx.sum(hidden.astype(mx.float32) * weights[..., None], axis=1) / denom


def masked_sequence_last(hidden: mx.array, token_weights: mx.array) -> mx.array:
    weights = token_weights.astype(mx.float32)
    lengths = mx.maximum(mx.sum(weights, axis=1).astype(mx.int32), mx.array(1, dtype=mx.int32))
    last_idx = lengths - mx.array(1, dtype=mx.int32)
    batch_idx = mx.arange(hidden.shape[0], dtype=mx.int32)
    return hidden[batch_idx, last_idx].astype(mx.float32)


class KernelTeacherStudent(nn.Module):
    def __init__(
        self,
        backbone: base.GPT,
        *,
        teacher_dim: int,
        readout_mode: str = "mean",
        projection_mode: str = "direct",
        factor_basis: np.ndarray | None = None,
        projection_init_std: float = 0.005,
    ):
        super().__init__()
        self.backbone = backbone
        self.teacher_dim = int(teacher_dim)
        self.readout_mode = str(readout_mode).strip().lower()
        self.projection_mode = str(projection_mode).strip().lower()
        hidden_dim = int(backbone.tok_emb.weight.shape[1])
        if self.readout_mode not in {"mean", "last", "attn", "mean_last"}:
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")
        if self.projection_mode not in {"direct", "factorized"}:
            raise ValueError(f"Unsupported projection_mode: {projection_mode}")
        if self.readout_mode == "attn":
            self.readout_query = CastedLinear(hidden_dim, 1)
        elif self.readout_mode == "mean_last":
            self.readout_fuse = CastedLinear(hidden_dim * 2, hidden_dim)
        self.factor_count = 0
        self._factor_basis_np: np.ndarray | None = None
        if self.projection_mode == "direct":
            self.kernel_projection = CastedLinear(hidden_dim, int(teacher_dim))
            self.kernel_projection.weight = (
                mx.random.normal(self.kernel_projection.weight.shape, dtype=mx.float32) * float(projection_init_std)
            ).astype(mx.float32)
        else:
            factor_basis_np = np.asarray(factor_basis, dtype=np.float32) if factor_basis is not None else np.zeros((teacher_dim, 0), dtype=np.float32)
            if factor_basis_np.ndim != 2 or factor_basis_np.shape[0] != int(teacher_dim) or factor_basis_np.shape[1] <= 0:
                raise ValueError("factorized projection_mode requires a non-empty factor_basis with shape [teacher_dim, factor_count]")
            self._factor_basis_np = factor_basis_np.astype(np.float32, copy=False)
            self.factor_count = int(factor_basis_np.shape[1])
            self.factor_projection = CastedLinear(hidden_dim, int(self.factor_count))
            self.factor_projection.weight = (
                mx.random.normal(self.factor_projection.weight.shape, dtype=mx.float32) * float(projection_init_std)
            ).astype(mx.float32)
        if self.readout_mode == "attn":
            self.readout_query.weight = (
                mx.random.normal(self.readout_query.weight.shape, dtype=mx.float32) * float(projection_init_std)
            ).astype(mx.float32)
        elif self.readout_mode == "mean_last":
            self.readout_fuse.weight = (
                mx.random.normal(self.readout_fuse.weight.shape, dtype=mx.float32) * float(projection_init_std)
            ).astype(mx.float32)

    def pooled_readout(
        self,
        hidden: mx.array,
        *,
        token_weights: mx.array,
    ) -> mx.array:
        mode = self.readout_mode
        if mode == "mean":
            return masked_sequence_mean(hidden, token_weights)
        if mode == "last":
            return masked_sequence_last(hidden, token_weights)
        if mode == "mean_last":
            mean_pool = masked_sequence_mean(hidden, token_weights)
            last_pool = masked_sequence_last(hidden, token_weights)
            fused = mx.concatenate([mean_pool, last_pool], axis=-1)
            return self.readout_fuse(fused.astype(COMPUTE_DTYPE_ALIAS())).astype(mx.float32)
        weights = token_weights.astype(mx.float32)
        scores = self.readout_query(hidden.astype(COMPUTE_DTYPE_ALIAS())).astype(mx.float32).squeeze(-1)
        masked_scores = mx.where(
            weights > 0.0,
            scores,
            mx.full(scores.shape, -1.0e9, dtype=mx.float32),
        )
        attn = mx.softmax(masked_scores, axis=1).astype(mx.float32)
        return mx.sum(hidden.astype(mx.float32) * attn[..., None], axis=1)

    def encode_project(
        self,
        input_ids: mx.array,
        *,
        token_weights: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        hidden, _captured, _aux = self.backbone.forward_hidden_with_aux(input_ids)
        pooled = self.pooled_readout(hidden, token_weights=token_weights)
        if self.projection_mode == "direct":
            projected = self.kernel_projection(pooled.astype(COMPUTE_DTYPE_ALIAS()))
            return hidden, projected.astype(mx.float32), None
        factor_scores = self.factor_projection(pooled.astype(COMPUTE_DTYPE_ALIAS())).astype(mx.float32)
        basis = mx.array(self._factor_basis_np, dtype=mx.float32)
        projected = mx.matmul(factor_scores, basis.T)
        return hidden, projected.astype(mx.float32), factor_scores.astype(mx.float32)

    def loss_terms(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        teacher_embedding: mx.array,
        *,
        token_weights: mx.array,
        ce_weight: float,
        distill_weight: float,
        factor_loss_weight: float,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        hidden, projected, factor_scores = self.encode_project(input_ids, token_weights=token_weights)
        ce_probe = self.backbone.token_ce_from_hidden(hidden, target_ids, token_weights=token_weights)
        ce_loss = (
            ce_probe
            if ce_weight > 0.0
            else mx.array(0.0, dtype=mx.float32)
        )
        student = projected.astype(mx.float32)
        teacher = mx.stop_gradient(teacher_embedding.astype(mx.float32))
        student_norm = student / mx.maximum(mx.linalg.norm(student, axis=-1, keepdims=True), mx.array(1.0e-6, dtype=mx.float32))
        teacher_norm = teacher / mx.maximum(mx.linalg.norm(teacher, axis=-1, keepdims=True), mx.array(1.0e-6, dtype=mx.float32))
        cosine = mx.sum(student_norm * teacher_norm, axis=-1).astype(mx.float32)
        distill_loss = mx.mean(1.0 - cosine)
        factor_loss = mx.array(0.0, dtype=mx.float32)
        if self.projection_mode == "factorized" and factor_scores is not None and self._factor_basis_np is not None:
            basis = mx.array(self._factor_basis_np, dtype=mx.float32)
            teacher_factors = mx.matmul(teacher, basis).astype(mx.float32)
            factor_loss = mx.mean(mx.square(factor_scores.astype(mx.float32) - teacher_factors))
        total = (
            mx.array(float(ce_weight), dtype=mx.float32) * ce_loss
            + mx.array(float(distill_weight), dtype=mx.float32) * distill_loss
            + mx.array(float(factor_loss_weight), dtype=mx.float32) * factor_loss
        )
        bpb_probe = ce_probe.astype(mx.float32) / mx.array(np.log(2.0), dtype=mx.float32)
        return total.astype(mx.float32), {
            "loss_total": total.astype(mx.float32),
            "loss_ce": ce_loss.astype(mx.float32),
            "loss_ce_probe": ce_probe.astype(mx.float32),
            "bpb_probe": bpb_probe.astype(mx.float32),
            "loss_distill": distill_loss.astype(mx.float32),
            "loss_factor": factor_loss.astype(mx.float32),
            "mean_teacher_cosine": mx.mean(cosine).astype(mx.float32),
        }


def COMPUTE_DTYPE_ALIAS():
    return getattr(base, "COMPUTE_DTYPE", mx.float16)


def _rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(int(seed))
    return rng


def load_teacher_text_records(
    *,
    teacher_dataset_path: str | Path,
    tokenizer_path: str | Path,
    max_seq_len: int,
) -> tuple[list[TeacherTextRecord], spm.SentencePieceProcessor]:
    teacher_dataset = KernelTeacherTextDataset.load(teacher_dataset_path)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    records: list[TeacherTextRecord] = []
    for example in teacher_dataset.examples:
        text = str(example.get("text", ""))
        if not text.strip():
            continue
        token_ids = list(sp.encode(text, out_type=int))
        if len(token_ids) < 2:
            continue
        token_ids = token_ids[: max(int(max_seq_len), 1) + 1]
        if len(token_ids) < 2:
            continue
        teacher_embedding = np.asarray(example.get("cleared_embedding", []), dtype=np.float32).reshape(-1)
        if teacher_embedding.shape[0] != int(teacher_dataset.embedding_dim):
            if teacher_embedding.shape[0] < int(teacher_dataset.embedding_dim):
                teacher_embedding = np.pad(teacher_embedding, (0, int(teacher_dataset.embedding_dim) - teacher_embedding.shape[0]))
            else:
                teacher_embedding = teacher_embedding[: int(teacher_dataset.embedding_dim)]
        records.append(
            TeacherTextRecord(
                example_id=str(example.get("example_id", "")),
                chunk_id=str(example.get("chunk_id", "")),
                text=text,
                token_ids=token_ids,
                teacher_embedding=teacher_embedding.astype(np.float32),
                winner_model=str(example.get("winner_model", "")),
                predicted_model=str(example.get("predicted_model", "")),
                winner_probability=float(example.get("winner_probability", 0.0)),
                verification_confidence=float(example.get("verification_confidence", 0.0)),
                candidate_model_ids=[str(value) for value in example.get("candidate_model_ids", [])],
                candidate_weights=np.asarray(example.get("candidate_weights", []), dtype=np.float32).reshape(-1),
                candidate_embeddings=np.asarray(example.get("candidate_embeddings", []), dtype=np.float32),
            )
        )
    if not records:
        raise ValueError(f"No usable teacher-text records found in {teacher_dataset_path}")
    return records, sp


def split_records(
    records: list[TeacherTextRecord],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[TeacherTextRecord], list[TeacherTextRecord]]:
    ordered = list(records)
    _rng(seed).shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, ordered
    val_count = int(round(len(ordered) * float(val_fraction)))
    val_count = max(1, min(val_count, len(ordered) - 1))
    return ordered[val_count:], ordered[:val_count]


def batch_records(
    records: list[TeacherTextRecord],
    *,
    start: int,
    batch_size: int,
    pad_id: int,
    max_seq_len: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch = records[start: start + max(int(batch_size), 1)]
    if not batch:
        raise ValueError("Empty batch")
    seq_lens = [max(1, min(len(record.token_ids) - 1, int(max_seq_len))) for record in batch]
    width = max(seq_lens)
    x_np = np.full((len(batch), width), int(pad_id), dtype=np.int32)
    y_np = np.full((len(batch), width), int(pad_id), dtype=np.int32)
    weights_np = np.zeros((len(batch), width), dtype=np.float32)
    teacher_np = np.zeros((len(batch), int(batch[0].teacher_embedding.shape[0])), dtype=np.float32)
    for row_idx, (record, seq_len) in enumerate(zip(batch, seq_lens, strict=True)):
        tokens = np.asarray(record.token_ids[: seq_len + 1], dtype=np.int32)
        x_np[row_idx, :seq_len] = tokens[:-1]
        y_np[row_idx, :seq_len] = tokens[1:]
        weights_np[row_idx, :seq_len] = 1.0
        teacher_np[row_idx] = record.teacher_embedding.astype(np.float32)
    return (
        mx.array(x_np, dtype=mx.int32),
        mx.array(y_np, dtype=mx.int32),
        mx.array(weights_np, dtype=mx.float32),
        mx.array(teacher_np, dtype=mx.float32),
    )


def _optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    grads_tree,
    *,
    freeze_backbone: bool,
) -> None:
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads_tree))
    if freeze_backbone:
        for name, value in list(grads.items()):
            if name.startswith("backbone."):
                grads[name] = mx.zeros_like(value)
    updated = optimizer.apply_gradients(grads, params)
    model.update(tree_unflatten(list(updated.items())))


def compute_disagreement_factor_basis(
    records: list[TeacherTextRecord],
    *,
    teacher_dim: int,
    factor_count: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    factor_count = int(factor_count)
    if factor_count <= 0:
        return None, {"factor_count": 0, "basis_source": "disabled", "samples_used": 0}
    diffs: list[np.ndarray] = []
    for record in records:
        candidate_embeddings = np.asarray(record.candidate_embeddings, dtype=np.float32)
        if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[1] != int(teacher_dim):
            continue
        candidate_count = candidate_embeddings.shape[0]
        if candidate_count < 2:
            continue
        winner_idx = None
        if record.winner_model and record.candidate_model_ids:
            try:
                winner_idx = record.candidate_model_ids.index(record.winner_model)
            except ValueError:
                winner_idx = None
        if winner_idx is None:
            candidate_weights = np.asarray(record.candidate_weights, dtype=np.float32).reshape(-1)
            if candidate_weights.shape[0] == candidate_count:
                winner_idx = int(np.argmax(candidate_weights))
            else:
                winner_idx = 0
        winner_embedding = candidate_embeddings[int(winner_idx)]
        for idx in range(candidate_count):
            if idx == int(winner_idx):
                continue
            diffs.append((winner_embedding - candidate_embeddings[idx]).astype(np.float32))
    basis_source = "winner_differences"
    if not diffs:
        diffs = [np.asarray(record.teacher_embedding, dtype=np.float32) for record in records if int(np.asarray(record.teacher_embedding).shape[0]) == int(teacher_dim)]
        basis_source = "teacher_embeddings"
    if not diffs:
        return None, {"factor_count": 0, "basis_source": "empty", "samples_used": 0}
    matrix = np.stack(diffs, axis=0).astype(np.float32, copy=False)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, {"factor_count": 0, "basis_source": "svd_failed", "samples_used": int(matrix.shape[0])}
    k = min(int(factor_count), int(vh.shape[0]), int(vh.shape[1]))
    if k <= 0:
        return None, {"factor_count": 0, "basis_source": "degenerate", "samples_used": int(matrix.shape[0])}
    basis = vh[:k].T.astype(np.float32, copy=False)
    return basis, {"factor_count": int(k), "basis_source": basis_source, "samples_used": int(matrix.shape[0])}


def evaluate_student(
    model: KernelTeacherStudent,
    records: list[TeacherTextRecord],
    *,
    batch_size: int,
    pad_id: int,
    max_seq_len: int,
    ce_weight: float,
    distill_weight: float,
    factor_loss_weight: float,
) -> dict[str, float]:
    totals = {
        "loss_total": 0.0,
        "loss_ce": 0.0,
        "loss_ce_probe": 0.0,
        "bpb_probe": 0.0,
        "loss_distill": 0.0,
        "loss_factor": 0.0,
        "mean_teacher_cosine": 0.0,
    }
    batches = 0
    for start in range(0, len(records), max(int(batch_size), 1)):
        x, y, token_weights, teacher_embedding = batch_records(
            records,
            start=start,
            batch_size=batch_size,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
        )
        _loss, metrics = model.loss_terms(
            x,
            y,
            teacher_embedding,
            token_weights=token_weights,
            ce_weight=ce_weight,
            distill_weight=distill_weight,
            factor_loss_weight=factor_loss_weight,
        )
        mx.eval(*metrics.values())
        for key in totals:
            totals[key] += float(metrics[key].item())
        batches += 1
    denom = max(batches, 1)
    return {key: float(value / denom) for key, value in totals.items()}


def build_backbone_args(args: argparse.Namespace, sp: spm.SentencePieceProcessor) -> base.Hyperparameters:
    hp = base.Hyperparameters()
    hp.tokenizer_path = str(Path(args.tokenizer_path).expanduser().resolve())
    hp.vocab_size = int(sp.vocab_size())
    hp.train_seq_len = int(args.max_seq_len)
    hp.num_layers = int(args.num_layers)
    hp.num_layer_templates = int(args.num_layer_templates)
    hp.model_dim = int(args.model_dim)
    hp.num_heads = int(args.num_heads)
    hp.num_kv_heads = int(args.num_kv_heads)
    hp.mlp_mult = int(args.mlp_mult)
    hp.mlp_leaky_slope = float(args.mlp_leaky_slope)
    hp.tie_embeddings = bool(args.tie_embeddings)
    hp.logit_softcap = float(args.logit_softcap)
    hp.tied_embed_init_std = float(args.tied_embed_init_std)
    hp.qk_gain_init = float(args.qk_gain_init)
    hp.seed = int(args.seed)
    return hp


def save_student_state(path: str | Path, model: KernelTeacherStudent) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exportable_state = base.exportable_flat_state(model)
    mx.savez(str(path), **exportable_state)


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_kernel_teacher_student(
    *,
    teacher_dataset_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    seed: int = 17,
    batch_size: int = 8,
    epochs: int = 4,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 0.0,
    max_seq_len: int = 256,
    val_fraction: float = 0.2,
    ce_weight: float = 1.0,
    distill_weight: float = 1.0,
    projection_mode: str = "direct",
    factor_count: int = 0,
    factor_loss_weight: float = 1.0,
    readout_mode: str = "mean",
    projection_init_std: float = 0.005,
    freeze_backbone: bool = False,
    model_dim: int = 256,
    num_layers: int = 4,
    num_layer_templates: int = 4,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 4,
    mlp_leaky_slope: float = 0.0,
    tie_embeddings: bool = True,
    logit_softcap: float = 30.0,
    tied_embed_init_std: float = 0.02,
    qk_gain_init: float = 1.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(seed))
    np.random.seed(int(seed))
    mx.random.seed(int(seed))

    records, sp = load_teacher_text_records(
        teacher_dataset_path=teacher_dataset_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )
    train_records, val_records = split_records(records, val_fraction=val_fraction, seed=seed)
    pad_id = int(sp.pad_id()) if int(sp.pad_id()) >= 0 else 0
    factor_basis, factor_basis_summary = compute_disagreement_factor_basis(
        train_records,
        teacher_dim=int(records[0].teacher_embedding.shape[0]),
        factor_count=factor_count,
    )

    backbone_args = argparse.Namespace(
        tokenizer_path=str(tokenizer_path),
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_layer_templates=num_layer_templates,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        mlp_leaky_slope=mlp_leaky_slope,
        tie_embeddings=tie_embeddings,
        logit_softcap=logit_softcap,
        tied_embed_init_std=tied_embed_init_std,
        qk_gain_init=qk_gain_init,
        seed=seed,
    )
    hp = build_backbone_args(backbone_args, sp)
    backbone = base.make_gpt(hp, sp)
    backbone.set_turbo_qat(False, 0.0)
    student = KernelTeacherStudent(
        backbone,
        teacher_dim=int(records[0].teacher_embedding.shape[0]),
        readout_mode=readout_mode,
        projection_mode=projection_mode,
        factor_basis=factor_basis,
        projection_init_std=projection_init_std,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def loss_and_grad(x, y, teacher_embedding, token_weights):
        return nn.value_and_grad(
            student,
            lambda x_inner, y_inner, teacher_inner, weights_inner: student.loss_terms(
                x_inner,
                y_inner,
                teacher_inner,
                token_weights=weights_inner,
                ce_weight=ce_weight,
                distill_weight=distill_weight,
                factor_loss_weight=factor_loss_weight,
            )[0],
        )(x, y, teacher_embedding, token_weights)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = output_dir / "best_kernel_teacher_student.npz"
    partial_path = output_dir / "summary.partial.json"
    for epoch in range(max(int(epochs), 1)):
        shuffled = list(train_records)
        _rng(seed + epoch).shuffle(shuffled)
        train_metrics_totals = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_ce_probe": 0.0,
            "bpb_probe": 0.0,
            "loss_distill": 0.0,
            "loss_factor": 0.0,
            "mean_teacher_cosine": 0.0,
        }
        train_batches = 0
        for start in range(0, len(shuffled), max(int(batch_size), 1)):
            x, y, token_weights, teacher_embedding = batch_records(
                shuffled,
                start=start,
                batch_size=batch_size,
                pad_id=pad_id,
                max_seq_len=max_seq_len,
            )
            loss, grads = loss_and_grad(x, y, teacher_embedding, token_weights)
            _loss_for_metrics, metrics = student.loss_terms(
                x,
                y,
                teacher_embedding,
                token_weights=token_weights,
                ce_weight=ce_weight,
                distill_weight=distill_weight,
                factor_loss_weight=factor_loss_weight,
            )
            _optimizer_step(student, optimizer, grads, freeze_backbone=freeze_backbone)
            mx.eval(loss, *metrics.values())
            for key in train_metrics_totals:
                train_metrics_totals[key] += float(metrics[key].item())
            train_batches += 1
        train_metrics = {key: float(value / max(train_batches, 1)) for key, value in train_metrics_totals.items()}
        val_metrics = evaluate_student(
            student,
            val_records,
            batch_size=batch_size,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
            ce_weight=ce_weight,
            distill_weight=distill_weight,
            factor_loss_weight=factor_loss_weight,
        )
        epoch_summary = {
            "epoch": float(epoch + 1),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_summary)
        candidate_best = min(best_val, float(val_metrics["loss_total"]))
        partial_summary = {
            "teacher_dataset_path": str(Path(teacher_dataset_path).resolve()),
            "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
            "train_examples": int(len(train_records)),
            "val_examples": int(len(val_records)),
            "teacher_dim": int(records[0].teacher_embedding.shape[0]),
            "max_seq_len": int(max_seq_len),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "ce_weight": float(ce_weight),
            "distill_weight": float(distill_weight),
            "projection_mode": str(projection_mode),
            "factor_count": int(factor_basis_summary.get("factor_count", 0)),
            "factor_loss_weight": float(factor_loss_weight),
            "factor_basis_source": str(factor_basis_summary.get("basis_source", "")),
            "factor_basis_samples": int(factor_basis_summary.get("samples_used", 0)),
            "readout_mode": str(readout_mode),
            "freeze_backbone": bool(freeze_backbone),
            "model_dim": int(model_dim),
            "num_layers": int(num_layers),
            "num_layer_templates": int(num_layer_templates),
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "history": history,
            "best_checkpoint_path": str(best_path),
            "best_val_loss_total": float(candidate_best),
            "latest_val_metrics": val_metrics,
            "status": "running",
        }
        write_summary_json(partial_path, partial_summary)
        if float(val_metrics["loss_total"]) < best_val:
            best_val = float(val_metrics["loss_total"])
            save_student_state(best_path, student)

    final_eval = evaluate_student(
        student,
        val_records,
        batch_size=batch_size,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        ce_weight=ce_weight,
        distill_weight=distill_weight,
        factor_loss_weight=factor_loss_weight,
    )
    final_path = output_dir / "kernel_teacher_student_final.npz"
    save_student_state(final_path, student)
    summary = {
        "teacher_dataset_path": str(Path(teacher_dataset_path).resolve()),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "train_examples": int(len(train_records)),
        "val_examples": int(len(val_records)),
        "teacher_dim": int(records[0].teacher_embedding.shape[0]),
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "ce_weight": float(ce_weight),
        "distill_weight": float(distill_weight),
        "projection_mode": str(projection_mode),
        "factor_count": int(factor_basis_summary.get("factor_count", 0)),
        "factor_loss_weight": float(factor_loss_weight),
        "factor_basis_source": str(factor_basis_summary.get("basis_source", "")),
        "factor_basis_samples": int(factor_basis_summary.get("samples_used", 0)),
        "readout_mode": str(readout_mode),
        "freeze_backbone": bool(freeze_backbone),
        "model_dim": int(model_dim),
        "num_layers": int(num_layers),
        "num_layer_templates": int(num_layer_templates),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "history": history,
        "best_checkpoint_path": str(best_path),
        "final_checkpoint_path": str(final_path),
        "final_val_metrics": final_eval,
        "status": "completed",
    }
    write_summary_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    defaults = base.Hyperparameters()
    parser = argparse.ArgumentParser(description="Train a first-pass kernel student against text-conditioned cleared embeddings from the ecology teacher.")
    parser.add_argument("--config", default="", help="Optional JSON config with env/args overrides")
    parser.add_argument("--teacher-dataset", required=True, help="Input KernelTeacherTextDataset artifact")
    parser.add_argument("--tokenizer-path", default=defaults.tokenizer_path, help="SentencePiece tokenizer .model")
    parser.add_argument("--output-dir", required=True, help="Directory for summary/checkpoints")
    parser.add_argument("--seed", type=int, default=int(defaults.seed))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--readout-mode", choices=("mean", "last", "attn", "mean_last"), default="mean")
    parser.add_argument("--projection-mode", choices=("direct", "factorized"), default="direct")
    parser.add_argument("--factor-count", type=int, default=0)
    parser.add_argument("--factor-loss-weight", type=float, default=1.0)
    parser.add_argument("--projection-init-std", type=float, default=0.005)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-layer-templates", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=int(defaults.mlp_mult))
    parser.add_argument("--mlp-leaky-slope", type=float, default=float(defaults.mlp_leaky_slope))
    parser.add_argument("--tie-embeddings", action="store_true", default=bool(defaults.tie_embeddings))
    parser.add_argument("--no-tie-embeddings", action="store_false", dest="tie_embeddings")
    parser.add_argument("--logit-softcap", type=float, default=float(defaults.logit_softcap))
    parser.add_argument("--tied-embed-init-std", type=float, default=float(defaults.tied_embed_init_std))
    parser.add_argument("--qk-gain-init", type=float, default=float(defaults.qk_gain_init))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = train_kernel_teacher_student(
        teacher_dataset_path=args.teacher_dataset,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        val_fraction=args.val_fraction,
        ce_weight=args.ce_weight,
        distill_weight=args.distill_weight,
        projection_mode=args.projection_mode,
        factor_count=args.factor_count,
        factor_loss_weight=args.factor_loss_weight,
        readout_mode=args.readout_mode,
        projection_init_std=args.projection_init_std,
        freeze_backbone=args.freeze_backbone,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_layer_templates=args.num_layer_templates,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_leaky_slope=args.mlp_leaky_slope,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
