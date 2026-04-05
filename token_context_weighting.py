from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContextDeltaWeightingConfig:
    enabled: bool = False
    short_context_len: int = 128
    max_multiplier: float = 4.0
    topk_fraction: float = 0.0
    score_power: float = 1.0
    use_absolute_delta: bool = True


def normalize_context_delta_scores_np(
    long_nll: np.ndarray,
    short_nll: np.ndarray,
    *,
    max_multiplier: float = 4.0,
    topk_fraction: float = 0.0,
    score_power: float = 1.0,
    use_absolute_delta: bool = True,
) -> np.ndarray:
    long_arr = np.asarray(long_nll, dtype=np.float32)
    short_arr = np.asarray(short_nll, dtype=np.float32)
    if long_arr.shape != short_arr.shape:
        raise ValueError(
            f"long_nll and short_nll must share a shape, got {long_arr.shape!r} and {short_arr.shape!r}"
        )

    delta = short_arr - long_arr
    scores = np.abs(delta) if use_absolute_delta else np.maximum(delta, 0.0)
    scores = np.maximum(scores, 0.0).astype(np.float32, copy=False)
    if score_power != 1.0:
        scores = np.power(scores, score_power, dtype=np.float32)

    weights = 1.0 + scores
    flat_scores = scores.reshape(-1)
    if 0.0 < topk_fraction < 1.0 and flat_scores.size > 0:
        keep = max(1, int(np.ceil(flat_scores.size * topk_fraction)))
        kth = flat_scores.size - keep
        threshold = np.partition(flat_scores, kth)[kth]
        weights = np.where(scores >= threshold, weights, 0.0)

    if max_multiplier > 0.0:
        weights = np.minimum(weights, max_multiplier)
    return weights.astype(np.float32, copy=False)
