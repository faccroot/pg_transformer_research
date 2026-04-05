#!/usr/bin/env python3
from __future__ import annotations

"""
Opt-in representation-learning runner built on top of the plain MLX GPT baseline.

This keeps `train_gpt_mlx.py` as the default Parameter Golf path and applies
representation-learning priors only through this separate entrypoint.
"""

import os

import train_gpt_mlx as base

try:
    from tools.representation_learning.runtime_mlx import apply_priors
except ModuleNotFoundError as exc:
    # `dispatch.sh` stages support files flat by basename, so queue-mode cluster runs
    # need a local-import fallback rather than a package import.
    if exc.name != "tools":
        raise
    from runtime_mlx import apply_priors  # type: ignore[no-redef]


os.environ.setdefault("REP_LEARN_QK_INIT", "1")


class Hyperparameters(base.Hyperparameters):
    rep_learn_priors_path: str = os.path.expanduser(os.environ.get("REP_LEARN_PRIORS_PATH", ""))
    rep_learn_qk_init: bool = bool(int(os.environ.get("REP_LEARN_QK_INIT", "1")))
    rep_learn_init_strength: float = float(os.environ.get("REP_LEARN_INIT_STRENGTH", 0.5))
    rep_learn_init_targets: str = os.environ.get("REP_LEARN_INIT_TARGETS", "qk")
    rep_learn_adapter_mode: str = os.environ.get("REP_LEARN_ADAPTER_MODE", "random")


_BASE_MAKE_GPT = base.make_gpt


def make_gpt(args: Hyperparameters, sp=None):
    model = _BASE_MAKE_GPT(args, sp)
    if args.rep_learn_qk_init:
        apply_priors(
            model,
            priors_path=args.rep_learn_priors_path,
            strength=args.rep_learn_init_strength,
            targets=args.rep_learn_init_targets,
            adapter_mode=args.rep_learn_adapter_mode,
        )
    return model


base.Hyperparameters = Hyperparameters
base.make_gpt = make_gpt


def main() -> None:
    args = Hyperparameters()
    if args.rep_learn_qk_init and not args.rep_learn_priors_path:
        raise SystemExit("REP_LEARN_QK_INIT=1 requires REP_LEARN_PRIORS_PATH")
    print(
        "representation_learning:"
        f"qk_init={int(args.rep_learn_qk_init)} "
        f"priors={args.rep_learn_priors_path or 'off'} "
        f"strength={args.rep_learn_init_strength:.3f} "
        f"targets={args.rep_learn_init_targets} "
        f"adapter={args.rep_learn_adapter_mode}",
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
