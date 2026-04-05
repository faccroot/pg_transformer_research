from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SidecarTransitionResetController:
    enabled: bool = False
    cosine_threshold: float = 0.80
    cosine_sharpness: float = 12.0
    prior_weight: float = 1.0
    learned_weight: float = 0.0
    max_gate: float = 1.0


def transition_reset_prior_from_cosine(
    cosine: float | np.ndarray,
    controller: SidecarTransitionResetController,
) -> np.ndarray:
    cos = np.clip(np.asarray(cosine, dtype=np.float32), -1.0, 1.0)
    if not controller.enabled:
        return np.zeros_like(cos, dtype=np.float32)
    logits = (float(controller.cosine_threshold) - cos) * float(controller.cosine_sharpness)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)


def blend_transition_reset_signals(
    prior_signal: float | np.ndarray,
    learned_signal: float | np.ndarray,
    controller: SidecarTransitionResetController,
) -> np.ndarray:
    prior = np.asarray(prior_signal, dtype=np.float32)
    learned = np.asarray(learned_signal, dtype=np.float32)
    if not controller.enabled:
        return np.zeros_like(prior, dtype=np.float32)
    mixed = float(controller.prior_weight) * prior + float(controller.learned_weight) * learned
    return (np.clip(mixed, 0.0, 1.0) * float(controller.max_gate)).astype(np.float32, copy=False)
