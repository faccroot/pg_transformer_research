from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SidecarAuxBudgetController:
    enabled: bool = False
    min_scale: float = 0.50
    max_scale: float = 1.50
    operator_density_high: float = 0.02
    operator_density_low: float = 0.005
    high_human_compressibility: float = 0.50
    low_human_compressibility: float = 0.33


def derive_sidecar_aux_scale(
    *,
    base_scale: float,
    phase_focus: str | None,
    controller: SidecarAuxBudgetController,
    mean_operator_density: float | None = None,
    mean_human_compressibility: float | None = None,
    mean_compressibility: float | None = None,
) -> float:
    scale = float(base_scale)
    if scale <= 0.0 or not controller.enabled:
        return scale

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

    return float(np.clip(scale, controller.min_scale, controller.max_scale))
