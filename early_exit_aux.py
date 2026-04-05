from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EarlyExitBudgetController:
    enabled: bool = False
    min_scale: float = 0.50
    max_scale: float = 1.50
    operator_density_high: float = 0.02
    operator_density_low: float = 0.005
    high_human_compressibility: float = 0.50
    low_human_compressibility: float = 0.33


def select_contiguous_draft_horizons(
    horizons: Iterable[int],
    confidences: Iterable[float],
    *,
    threshold: float,
    max_tokens: int,
) -> tuple[int, ...]:
    horizon_conf = sorted(
        ((int(h), float(c)) for h, c in zip(horizons, confidences, strict=False) if int(h) > 0),
        key=lambda item: item[0],
    )
    accepted: list[int] = []
    expected = 1
    limit = max(int(max_tokens), 0)
    for horizon, confidence in horizon_conf:
        if len(accepted) >= limit:
            break
        if horizon != expected:
            break
        if confidence < float(threshold):
            break
        accepted.append(horizon)
        expected += 1
    return tuple(accepted)


def parse_horizons(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        values = [int(part) for part in parts]
    else:
        values = [int(value) for value in raw]
    if not values:
        raise ValueError(f"Expected at least one positive horizon, got {raw!r}")
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value <= 0:
            raise ValueError(f"Horizons must be positive integers, got {value!r}")
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def horizon_shift(horizon: int) -> int:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    return horizon - 1


def aligned_horizon_views_np(
    target_ids: np.ndarray,
    horizon: int,
    token_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    targets = np.asarray(target_ids)
    if targets.ndim != 2:
        raise ValueError(f"target_ids must have shape [batch, seq], got {targets.shape!r}")
    shift = horizon_shift(horizon)
    if shift >= targets.shape[1]:
        return targets[:, :0], None if token_weights is None else np.asarray(token_weights)[:, :0]
    weights_arr = None if token_weights is None else np.asarray(token_weights)
    if weights_arr is not None and weights_arr.shape != targets.shape:
        raise ValueError(
            f"token_weights must match target_ids shape, got {weights_arr.shape!r} vs {targets.shape!r}"
        )
    return targets[:, shift:], None if weights_arr is None else weights_arr[:, shift:]


def derive_early_exit_aux_weight(
    base_weight: float,
    *,
    phase_focus: str | None,
    controller: EarlyExitBudgetController,
    mean_operator_density: float | None = None,
    mean_human_compressibility: float | None = None,
    mean_compressibility: float | None = None,
) -> float:
    base = float(base_weight)
    if base <= 0.0 or not controller.enabled:
        return base

    scale = 1.0
    focus = (phase_focus or "").strip().lower()
    if focus == "operator_dense":
        scale *= 1.35
    elif focus == "hard":
        scale *= 1.10
    elif focus == "diverse":
        scale *= 0.75
    elif focus == "easy":
        scale *= 0.50

    if mean_operator_density is not None:
        op_density = float(mean_operator_density)
        if op_density >= controller.operator_density_high:
            scale *= 1.15
        elif op_density <= controller.operator_density_low:
            scale *= 0.90

    human_compressibility = mean_human_compressibility
    if human_compressibility is None and mean_compressibility is not None:
        human_compressibility = 1.0 - float(mean_compressibility)
    if human_compressibility is not None:
        human = float(human_compressibility)
        if human >= controller.high_human_compressibility:
            scale *= 0.85
        elif human <= controller.low_human_compressibility:
            scale *= 1.10

    scale = float(np.clip(scale, controller.min_scale, controller.max_scale))
    return base * scale
