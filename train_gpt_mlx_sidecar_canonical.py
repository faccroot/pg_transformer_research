#!/usr/bin/env python3
"""
Canonical causal sidecar entrypoint.

This aliases the chunk-causal JEPA sidecar runner so future experiments use a
single stable trainer path instead of picking among older sidecar families.
"""
from __future__ import annotations

from train_gpt_mlx_jepa_sidecar_chunkcausal import (  # noqa: F401
    GPTJEPASidecar,
    GRUSidecarCell,
    Hyperparameters,
    make_sidecar_gpt,
    main,
)


if __name__ == "__main__":
    main()
