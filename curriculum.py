from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import zlib


FocusMode = Literal["diverse", "operator_dense", "hard", "easy", "mixed", "sequential"]

SHOW_NEVER = 0
SHOW_ONCE = 1
SHOW_REPEAT = 2


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    start_frac: float
    end_frac: float
    focus: FocusMode
    enable_jepa: bool = False
    enable_logic_sidecar: bool = False
    enable_qat: bool = False
    enable_ema: bool = False
    focal_loss_weight: float = 0.0
    skip_mastered: bool = True


@dataclass
class ChunkFeatures:
    cluster_ids: np.ndarray | None = None
    operator_density: np.ndarray | None = None
    difficulty: np.ndarray | None = None
    frontier_floor: np.ndarray | None = None
    structural_gap: np.ndarray | None = None
    duplicate_score: np.ndarray | None = None
    compressibility_ratio: np.ndarray | None = None
    learnability_score: np.ndarray | None = None
    quality_score: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        sizes = {
            int(array.shape[0])
            for array in (
                self.cluster_ids,
                self.operator_density,
                self.difficulty,
                self.frontier_floor,
                self.structural_gap,
                self.duplicate_score,
                self.compressibility_ratio,
                self.learnability_score,
                self.quality_score,
                self.confidence,
            )
            if array is not None
        }
        if not sizes:
            raise ValueError("ChunkFeatures requires at least one feature array")
        if len(sizes) != 1:
            raise ValueError(f"Chunk feature arrays must share the same length, got {sorted(sizes)}")

    def __len__(self) -> int:
        for array in (
                self.cluster_ids,
                self.operator_density,
                self.difficulty,
                self.frontier_floor,
                self.structural_gap,
                self.duplicate_score,
                self.compressibility_ratio,
                self.learnability_score,
                self.quality_score,
                self.confidence,
        ):
            if array is not None:
                return int(array.shape[0])
        raise RuntimeError("ChunkFeatures has no arrays")


def _structural_gap_vector(features: ChunkFeatures) -> np.ndarray | None:
    if features.structural_gap is not None:
        return np.clip(np.asarray(features.structural_gap, dtype=np.float32), 0.0, 1.0)
    if features.frontier_floor is not None and features.difficulty is not None:
        difficulty = np.clip(np.asarray(features.difficulty, dtype=np.float32), 0.0, 1.0)
        frontier = np.clip(np.asarray(features.frontier_floor, dtype=np.float32), 0.0, 1.0)
        return np.clip(difficulty - frontier, 0.0, 1.0)
    return None


def default_phase_plan() -> list[CurriculumPhase]:
    return [
        CurriculumPhase(
            name="structural-foundation",
            start_frac=0.00,
            end_frac=0.20,
            focus="diverse",
        ),
        CurriculumPhase(
            name="logic-polarity",
            start_frac=0.20,
            end_frac=0.40,
            focus="operator_dense",
            enable_jepa=True,
            enable_logic_sidecar=True,
        ),
        CurriculumPhase(
            name="hard-content",
            start_frac=0.40,
            end_frac=0.60,
            focus="hard",
            enable_jepa=True,
            enable_logic_sidecar=True,
            focal_loss_weight=0.25,
        ),
        CurriculumPhase(
            name="qat-geometry",
            start_frac=0.60,
            end_frac=0.80,
            focus="easy",
            enable_qat=True,
        ),
        CurriculumPhase(
            name="consolidation",
            start_frac=0.80,
            end_frac=1.00,
            focus="mixed",
            enable_qat=True,
            enable_ema=True,
        ),
    ]


def validate_phase_plan(phases: list[CurriculumPhase], atol: float = 1e-6) -> None:
    if not phases:
        raise ValueError("phase plan must not be empty")
    prev_end = 0.0
    for phase in phases:
        if phase.start_frac < -atol or phase.end_frac > 1.0 + atol:
            raise ValueError(f"phase {phase.name!r} is outside [0, 1]")
        if phase.end_frac <= phase.start_frac:
            raise ValueError(f"phase {phase.name!r} has a non-positive duration")
        if abs(phase.start_frac - prev_end) > atol:
            raise ValueError(
                f"phase plan must be contiguous: expected {prev_end:.6f}, got {phase.start_frac:.6f} for {phase.name!r}"
            )
        prev_end = phase.end_frac
    if abs(prev_end - 1.0) > atol:
        raise ValueError(f"phase plan must end at 1.0, got {prev_end:.6f}")


def phase_for_progress(progress_frac: float, phases: list[CurriculumPhase]) -> CurriculumPhase:
    validate_phase_plan(phases)
    progress = min(max(float(progress_frac), 0.0), 1.0)
    for phase in phases:
        if progress < phase.end_frac:
            return phase
    return phases[-1]


def chunk_token_matrix(tokens: np.ndarray, chunk_size: int) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    flat = np.asarray(tokens, dtype=np.int32).reshape(-1)
    usable = ((max(flat.size - 1, 0)) // chunk_size) * chunk_size
    if usable <= 0:
        return np.zeros((0, chunk_size), dtype=np.int32)
    return flat[:usable].reshape(-1, chunk_size)


def hashed_token_histograms(tokens_or_chunks: np.ndarray, chunk_size: int, num_bins: int) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    chunks = (
        np.asarray(tokens_or_chunks, dtype=np.int32)
        if np.asarray(tokens_or_chunks).ndim == 2
        else chunk_token_matrix(np.asarray(tokens_or_chunks), chunk_size)
    )
    if chunks.size == 0:
        return np.zeros((0, num_bins), dtype=np.float32)
    row_ids = np.repeat(np.arange(chunks.shape[0], dtype=np.int64), chunks.shape[1])
    bin_ids = np.mod(chunks.reshape(-1).astype(np.int64), num_bins)
    hist = np.zeros((chunks.shape[0], num_bins), dtype=np.float32)
    np.add.at(hist, (row_ids, bin_ids), 1.0)
    totals = np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    return hist / totals


def operator_density(tokens_or_chunks: np.ndarray, chunk_size: int, operator_token_ids: np.ndarray | list[int]) -> np.ndarray:
    chunks = (
        np.asarray(tokens_or_chunks, dtype=np.int32)
        if np.asarray(tokens_or_chunks).ndim == 2
        else chunk_token_matrix(np.asarray(tokens_or_chunks), chunk_size)
    )
    if chunks.size == 0:
        return np.zeros((0,), dtype=np.float32)
    operator_ids = np.asarray(operator_token_ids, dtype=np.int32).reshape(-1)
    if operator_ids.size == 0:
        return np.zeros((chunks.shape[0],), dtype=np.float32)
    mask = np.isin(chunks, operator_ids)
    return mask.mean(axis=1, dtype=np.float32)


def zlib_compressibility_ratio(tokens_or_chunks: np.ndarray, chunk_size: int, level: int = 6) -> np.ndarray:
    chunks = (
        np.asarray(tokens_or_chunks, dtype=np.int32)
        if np.asarray(tokens_or_chunks).ndim == 2
        else chunk_token_matrix(np.asarray(tokens_or_chunks), chunk_size)
    )
    if chunks.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ratios = np.zeros((chunks.shape[0],), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        raw = np.asarray(chunk, dtype="<u2").tobytes()
        ratios[idx] = len(zlib.compress(raw, level=level)) / max(len(raw), 1)
    return ratios


def cosine_kmeans(features: np.ndarray, num_clusters: int, iterations: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must have shape [num_items, dim] with num_items > 0")
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    k = min(int(num_clusters), int(x.shape[0]))
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)

    init_idx = rng.choice(x.shape[0], size=k, replace=False)
    centers = x[init_idx].copy()
    assignments = np.zeros((x.shape[0],), dtype=np.int32)
    for _ in range(max(int(iterations), 1)):
        sims = x @ centers.T
        assignments = np.argmax(sims, axis=1).astype(np.int32)
        for cluster_idx in range(k):
            members = x[assignments == cluster_idx]
            if members.size == 0:
                centers[cluster_idx] = x[rng.integers(0, x.shape[0])]
            else:
                center = members.mean(axis=0)
                centers[cluster_idx] = center / max(np.linalg.norm(center), 1e-12)
    return assignments, centers.astype(np.float32)


def _difficulty_vector(features: ChunkFeatures) -> np.ndarray:
    structural_gap = _structural_gap_vector(features)
    if structural_gap is not None:
        if features.difficulty is not None:
            base = np.clip(np.asarray(features.difficulty, dtype=np.float32), 0.0, 1.0)
            return np.clip(0.7 * structural_gap + 0.3 * base, 0.0, 1.0)
        return structural_gap
    if features.difficulty is not None:
        base = np.clip(np.asarray(features.difficulty, dtype=np.float32), 0.0, 1.0)
        if features.compressibility_ratio is None:
            return base
        comp = np.clip(np.asarray(features.compressibility_ratio, dtype=np.float32), 0.0, 1.0)
        return np.clip(0.5 * base + 0.5 * comp, 0.0, 1.0)
    if features.compressibility_ratio is not None:
        return np.clip(np.asarray(features.compressibility_ratio, dtype=np.float32), 0.0, 1.0)
    if features.confidence is not None:
        confidence = np.clip(np.asarray(features.confidence, dtype=np.float32), 0.0, 1.0)
        return 1.0 - confidence
    return np.full((len(features),), 0.5, dtype=np.float32)


def _learnability_vector(features: ChunkFeatures) -> np.ndarray:
    structural_gap = _structural_gap_vector(features)
    if features.learnability_score is not None:
        return np.clip(np.asarray(features.learnability_score, dtype=np.float32), 0.0, 1.0)
    if structural_gap is not None:
        return structural_gap
    if features.confidence is not None and features.difficulty is not None:
        confidence = np.clip(np.asarray(features.confidence, dtype=np.float32), 0.0, 1.0)
        difficulty = np.clip(np.asarray(features.difficulty, dtype=np.float32), 0.0, 1.0)
        return np.clip(difficulty * (1.0 - confidence), 0.0, 1.0)
    if features.duplicate_score is not None:
        duplicate = np.clip(np.asarray(features.duplicate_score, dtype=np.float32), 0.0, 1.0)
        return np.clip(1.0 - duplicate, 0.0, 1.0)
    return np.ones((len(features),), dtype=np.float32)


def _quality_vector(features: ChunkFeatures) -> np.ndarray:
    if features.quality_score is not None:
        return np.clip(np.asarray(features.quality_score, dtype=np.float32), 0.0, 1.0)
    if features.duplicate_score is not None:
        duplicate = np.clip(np.asarray(features.duplicate_score, dtype=np.float32), 0.0, 1.0)
        return np.clip(1.0 - duplicate, 0.0, 1.0)
    return np.ones((len(features),), dtype=np.float32)


def _bell_score(values: np.ndarray, center: float, width: float) -> np.ndarray:
    scaled = (values - center) / max(width, 1e-6)
    return np.exp(-0.5 * scaled * scaled).astype(np.float32)


def score_chunk_priority(
    features: ChunkFeatures,
    phase: CurriculumPhase,
    seen_clusters: np.ndarray | None = None,
    mastered_mask: np.ndarray | None = None,
) -> np.ndarray:
    size = len(features)
    difficulty = _difficulty_vector(features)
    learnability = _learnability_vector(features)
    quality = _quality_vector(features)
    scores = np.ones((size,), dtype=np.float32)
    scores *= 0.25 + 0.75 * quality
    scores *= 0.10 + 0.90 * learnability

    if features.duplicate_score is not None:
        duplicate_score = np.clip(np.asarray(features.duplicate_score, dtype=np.float32), 0.0, 1.0)
        scores *= np.clip(1.0 - duplicate_score, 0.05, 1.0)
    comp = (
        np.clip(np.asarray(features.compressibility_ratio, dtype=np.float32), 0.0, 1.0)
        if features.compressibility_ratio is not None
        else None
    )

    if phase.skip_mastered and mastered_mask is not None:
        mask = np.asarray(mastered_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != size:
            raise ValueError("mastered_mask must match the number of chunks")
        scores = np.where(mask, 0.0, scores)

    if phase.focus == "diverse":
        scores *= 0.5 + _bell_score(difficulty, center=0.45, width=0.35)
        if features.cluster_ids is not None:
            cluster_ids = np.asarray(features.cluster_ids, dtype=np.int32)
            counts = np.bincount(cluster_ids, minlength=int(cluster_ids.max()) + 1)
            rarity = 1.0 / np.sqrt(np.maximum(counts[cluster_ids], 1))
            scores *= rarity / max(float(rarity.mean()), 1e-6)
            if seen_clusters is not None:
                seen = np.asarray(seen_clusters, dtype=bool)
                if cluster_ids.max(initial=0) >= seen.shape[0]:
                    raise ValueError("seen_clusters must cover all cluster ids")
                scores *= np.where(seen[cluster_ids], 1.0, 2.5)
    elif phase.focus == "operator_dense":
        operator = np.asarray(features.operator_density if features.operator_density is not None else 0.0, dtype=np.float32)
        scores *= 1.0 + 4.0 * operator
        scores *= 0.5 + _bell_score(difficulty, center=0.65, width=0.30)
        if comp is not None:
            scores *= 0.75 + 0.5 * (1.0 - comp)
    elif phase.focus == "hard":
        scores *= 0.1 + np.square(difficulty)
    elif phase.focus == "easy":
        scores *= 0.1 + np.square(1.0 - difficulty)
        if comp is not None:
            scores *= 0.75 + 0.5 * (1.0 - comp)
    elif phase.focus == "mixed":
        scores *= 0.5 + _bell_score(difficulty, center=0.55, width=0.25)
        if features.operator_density is not None:
            operator = np.asarray(features.operator_density, dtype=np.float32)
            scores *= 1.0 + 0.5 * operator
    else:
        raise ValueError(f"unknown focus mode: {phase.focus!r}")

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.astype(np.float32)


def order_chunk_indices(
    features: ChunkFeatures,
    phase: CurriculumPhase,
    seen_clusters: np.ndarray | None = None,
    mastered_mask: np.ndarray | None = None,
    limit: int | None = None,
) -> np.ndarray:
    if phase.focus == "sequential":
        order = np.arange(len(features), dtype=np.int32)
        if limit is None:
            return order
        return order[: max(int(limit), 0)]
    scores = score_chunk_priority(features, phase, seen_clusters=seen_clusters, mastered_mask=mastered_mask)
    order = np.argsort(-scores, kind="stable")
    if limit is None:
        return order
    return order[: max(int(limit), 0)]


def classify_replay_buckets(
    features: ChunkFeatures,
    mastered_threshold: float = 0.85,
    duplicate_threshold: float = 0.90,
    hard_threshold: float = 0.60,
    operator_repeat_threshold: float = 0.15,
    learnability_drop_threshold: float = 0.10,
    quality_drop_threshold: float = 0.10,
) -> np.ndarray:
    difficulty = _difficulty_vector(features)
    learnability = _learnability_vector(features)
    quality = _quality_vector(features)
    confidence = (
        np.clip(np.asarray(features.confidence, dtype=np.float32), 0.0, 1.0)
        if features.confidence is not None
        else 1.0 - difficulty
    )
    duplicate_score = (
        np.clip(np.asarray(features.duplicate_score, dtype=np.float32), 0.0, 1.0)
        if features.duplicate_score is not None
        else np.zeros((len(features),), dtype=np.float32)
    )
    compressibility_ratio = (
        np.clip(np.asarray(features.compressibility_ratio, dtype=np.float32), 0.0, 1.0)
        if features.compressibility_ratio is not None
        else np.zeros((len(features),), dtype=np.float32)
    )
    operator = (
        np.clip(np.asarray(features.operator_density, dtype=np.float32), 0.0, 1.0)
        if features.operator_density is not None
        else np.zeros((len(features),), dtype=np.float32)
    )

    buckets = np.full((len(features),), SHOW_ONCE, dtype=np.int8)
    never_mask = duplicate_score >= duplicate_threshold
    never_mask |= (confidence >= mastered_threshold) & (difficulty <= 0.20)
    never_mask |= (confidence >= mastered_threshold) & (compressibility_ratio <= 0.35)
    if (
        features.learnability_score is not None
        or features.structural_gap is not None
        or features.frontier_floor is not None
    ):
        never_mask |= learnability <= learnability_drop_threshold
    if features.quality_score is not None:
        never_mask |= quality <= quality_drop_threshold
    repeat_mask = (difficulty >= hard_threshold) | (operator >= operator_repeat_threshold)
    repeat_mask &= ~never_mask

    buckets[never_mask] = SHOW_NEVER
    buckets[repeat_mask] = SHOW_REPEAT
    return buckets
