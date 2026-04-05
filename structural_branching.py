from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuralBranchingConfig:
    enabled: bool = False
    start_frac: float = 0.6
    weight: float = 0.1
    branch_length: int = 6
    max_branches: int = 1
    min_structural_miss: float = 0.5
    max_top1_gap: float = 0.75
    max_top12_cosine: float = 1.0
    min_branch_score: float = 0.0
    min_top1_prob: float = 0.0
    min_position_gap: int = 8
    margin: float = 0.1
    state_divergence_weight: float = 0.0
    state_target_max_cosine: float = 0.25
    adaptive_depth_enabled: bool = True
    adaptive_min_depth: int = 2
    adaptive_plateau_tol: float = 0.02
    adaptive_converged_divergence: float = 0.05


@dataclass(frozen=True)
class StructuralBranchBudgetController:
    enabled: bool = False
    operator_density_high: float = 0.02
    operator_density_low: float = 0.005
    high_human_compressibility: float = 0.50
    low_human_compressibility: float = 0.33


@dataclass(frozen=True)
class StructuralBranchBudgetSignals:
    phase_focus: str | None = None
    mean_operator_density: float | None = None
    mean_human_compressibility: float | None = None
    mean_compressibility: float | None = None


@dataclass(frozen=True)
class StructuralBranchPoint:
    pos: int
    predicted_token: int
    target_token: int
    alternate_token: int
    top1_prob: float
    top1_gap: float
    top12_cosine: float
    structural_miss: float
    score: float


def derive_structural_branching_config(
    base: StructuralBranchingConfig,
    signals: StructuralBranchBudgetSignals,
    controller: StructuralBranchBudgetController,
) -> StructuralBranchingConfig:
    if not base.enabled or not controller.enabled:
        return base

    max_branches = int(base.max_branches)
    min_structural_miss = float(base.min_structural_miss)
    max_top1_gap = float(base.max_top1_gap)

    focus = (signals.phase_focus or "").strip().lower()
    if focus in {"easy", "diverse"}:
        max_branches = 0
        min_structural_miss *= 1.15
        max_top1_gap *= 0.90
    elif focus == "operator_dense":
        max_branches = max(max_branches, int(base.max_branches) + 1)
        min_structural_miss *= 0.90
        max_top1_gap *= 1.10
    elif focus == "hard":
        max_branches = min(max_branches, 1)
        min_structural_miss *= 1.10
        max_top1_gap *= 0.90

    operator_density = signals.mean_operator_density
    if operator_density is not None:
        if operator_density >= controller.operator_density_high:
            max_branches = max(max_branches, int(base.max_branches) + 1)
            min_structural_miss *= 0.90
            max_top1_gap *= 1.10
        elif operator_density <= controller.operator_density_low:
            max_branches = min(max_branches, 1)
            min_structural_miss *= 1.05

    human_compressibility = signals.mean_human_compressibility
    if human_compressibility is None and signals.mean_compressibility is not None:
        human_compressibility = 1.0 - float(signals.mean_compressibility)
    if human_compressibility is not None:
        if human_compressibility >= controller.high_human_compressibility:
            max_branches = min(max_branches, 1)
            min_structural_miss *= 1.10
        elif human_compressibility <= controller.low_human_compressibility:
            max_branches = min(max_branches, 1)
            min_structural_miss *= 1.15
            max_top1_gap *= 0.90

    max_branches = max(int(max_branches), 0)
    min_structural_miss = float(np.clip(min_structural_miss, 0.0, 2.0))
    max_top1_gap = float(max(max_top1_gap, 0.05))
    return StructuralBranchingConfig(
        enabled=base.enabled and base.weight > 0.0 and max_branches > 0,
        start_frac=base.start_frac,
        weight=base.weight,
        branch_length=base.branch_length,
        max_branches=max_branches,
        min_structural_miss=min_structural_miss,
        max_top1_gap=max_top1_gap,
        max_top12_cosine=base.max_top12_cosine,
        min_branch_score=base.min_branch_score,
        min_top1_prob=base.min_top1_prob,
        min_position_gap=base.min_position_gap,
        margin=base.margin,
        state_divergence_weight=base.state_divergence_weight,
        state_target_max_cosine=base.state_target_max_cosine,
        adaptive_depth_enabled=base.adaptive_depth_enabled,
        adaptive_min_depth=base.adaptive_min_depth,
        adaptive_plateau_tol=base.adaptive_plateau_tol,
        adaptive_converged_divergence=base.adaptive_converged_divergence,
    )


def adaptive_branch_length_from_divergence(
    divergence: np.ndarray,
    *,
    min_depth: int,
    plateau_tol: float,
    converged_divergence: float,
) -> int:
    values = np.asarray(divergence, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return 0
    min_depth = max(1, min(int(min_depth), int(values.size)))
    if values.size <= min_depth:
        return int(values.size)
    if float(values[min_depth - 1]) <= float(converged_divergence):
        return min_depth
    for idx in range(min_depth, int(values.size)):
        current = float(values[idx])
        if current <= float(converged_divergence):
            return idx + 1
        if idx >= 2:
            growth_now = current - float(values[idx - 1])
            growth_prev = float(values[idx - 1]) - float(values[idx - 2])
            if abs(growth_now) <= float(plateau_tol) and abs(growth_prev) <= float(plateau_tol):
                return idx + 1
    return int(values.size)


def branch_state_divergence_penalty_np(
    real_hidden: np.ndarray,
    wrong_hidden: np.ndarray,
    *,
    effective_len: int,
    target_max_cosine: float,
) -> float:
    real = np.asarray(real_hidden, dtype=np.float32)
    wrong = np.asarray(wrong_hidden, dtype=np.float32)
    if real.ndim != 2 or wrong.ndim != 2:
        raise ValueError(
            f"real_hidden and wrong_hidden must have shape [seq, dim], got {real.shape!r} and {wrong.shape!r}"
        )
    usable = max(0, min(int(effective_len), int(real.shape[0]), int(wrong.shape[0])))
    if usable <= 0:
        return 0.0
    real = _normalize_rows(real[:usable])
    wrong = _normalize_rows(wrong[:usable])
    cosines = np.sum(real * wrong, axis=-1, dtype=np.float32)
    penalties = np.maximum(cosines - float(target_max_cosine), 0.0).astype(np.float32, copy=False)
    return float(np.mean(penalties, dtype=np.float32))


def _rowwise_top2(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {logits.shape!r}")
    if logits.shape[-1] < 2:
        raise ValueError(f"logits vocab axis must be at least 2, got {logits.shape[-1]!r}")
    top2_idx = np.argpartition(logits, kth=logits.shape[-1] - 2, axis=-1)[..., -2:]
    top2_vals = np.take_along_axis(logits, top2_idx, axis=-1)
    swap = top2_vals[..., 0] >= top2_vals[..., 1]
    pred_idx = np.where(swap, top2_idx[..., 0], top2_idx[..., 1]).astype(np.int32, copy=False)
    alt_idx = np.where(swap, top2_idx[..., 1], top2_idx[..., 0]).astype(np.int32, copy=False)
    top1 = np.where(swap, top2_vals[..., 0], top2_vals[..., 1]).astype(np.float32, copy=False)
    top2 = np.where(swap, top2_vals[..., 1], top2_vals[..., 0]).astype(np.float32, copy=False)
    return pred_idx, alt_idx, top1, top2


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return (x_max + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))).squeeze(axis)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return x / denom


def select_structural_branch_points_np(
    logits: np.ndarray,
    target_ids: np.ndarray,
    embedding_table: np.ndarray,
    config: StructuralBranchingConfig,
) -> list[list[StructuralBranchPoint]]:
    logits_arr = np.asarray(logits, dtype=np.float32)
    target_arr = np.asarray(target_ids, dtype=np.int32)
    embed_arr = np.asarray(embedding_table, dtype=np.float32)
    if logits_arr.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {logits_arr.shape!r}")
    if target_arr.shape != logits_arr.shape[:2]:
        raise ValueError(
            f"target_ids must match logits batch/seq dims, got {target_arr.shape!r} and {logits_arr.shape[:2]!r}"
        )
    if embed_arr.ndim != 2 or embed_arr.shape[0] != logits_arr.shape[-1]:
        raise ValueError(
            f"embedding_table must have shape [vocab, dim], got {embed_arr.shape!r} for vocab {logits_arr.shape[-1]!r}"
        )
    if not config.enabled:
        return [[] for _ in range(logits_arr.shape[0])]

    pred_idx, alt_idx, top1, top2 = _rowwise_top2(logits_arr)
    log_norm = _logsumexp(logits_arr, axis=-1).astype(np.float32, copy=False)
    top1_prob = np.exp(top1 - log_norm).astype(np.float32, copy=False)
    top1_gap = (top1 - top2).astype(np.float32, copy=False)

    norm_embed = _normalize_rows(embed_arr)
    pred_embed = norm_embed[pred_idx]
    alt_embed = norm_embed[alt_idx]
    target_embed = norm_embed[target_arr]
    pred_target_cos = np.sum(pred_embed * target_embed, axis=-1, dtype=np.float32)
    top12_cosine = np.sum(pred_embed * alt_embed, axis=-1, dtype=np.float32)
    structural_miss = (1.0 - pred_target_cos).astype(np.float32, copy=False)
    semantic_distance = np.maximum(1.0 - top12_cosine, 0.0).astype(np.float32, copy=False)
    combined_score = (
        structural_miss * semantic_distance / (1.0 + np.maximum(top1_gap, 0.0))
    ).astype(np.float32, copy=False)

    plans: list[list[StructuralBranchPoint]] = []
    for row_idx in range(logits_arr.shape[0]):
        row_candidates: list[StructuralBranchPoint] = []
        for pos in range(logits_arr.shape[1]):
            pred_token = int(pred_idx[row_idx, pos])
            target_token = int(target_arr[row_idx, pos])
            if pred_token == target_token:
                continue
            miss = float(structural_miss[row_idx, pos])
            gap = float(top1_gap[row_idx, pos])
            prob = float(top1_prob[row_idx, pos])
            if miss < config.min_structural_miss:
                continue
            if gap > config.max_top1_gap:
                continue
            competing_cosine = float(top12_cosine[row_idx, pos])
            if competing_cosine > config.max_top12_cosine:
                continue
            if prob < config.min_top1_prob:
                continue
            score = float(combined_score[row_idx, pos])
            if score < config.min_branch_score:
                continue
            row_candidates.append(
                StructuralBranchPoint(
                    pos=pos,
                    predicted_token=pred_token,
                    target_token=target_token,
                    alternate_token=int(alt_idx[row_idx, pos]),
                    top1_prob=prob,
                    top1_gap=gap,
                    top12_cosine=competing_cosine,
                    structural_miss=miss,
                    score=score,
                )
            )
        row_candidates.sort(key=lambda item: (-item.score, item.pos))
        selected: list[StructuralBranchPoint] = []
        for candidate in row_candidates:
            if any(abs(candidate.pos - prior.pos) < config.min_position_gap for prior in selected):
                continue
            selected.append(candidate)
            if len(selected) >= config.max_branches:
                break
        plans.append(selected)
    return plans
