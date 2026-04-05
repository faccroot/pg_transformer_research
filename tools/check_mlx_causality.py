#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_config_env(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text())
    env = payload.get("env")
    if not isinstance(env, dict):
        raise ValueError(f"{config_path} does not contain a top-level 'env' object")
    return {str(k): str(v) for k, v in env.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prefix-logit causality for an MLX parameter-golf config/checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Config JSON with an 'env' object.")
    parser.add_argument("--checkpoint", type=Path, help="Optional raw MLX .npz checkpoint to load.")
    parser.add_argument(
        "--trainer-module",
        default="train_gpt_mlx",
        help="Trainer module to instantiate the model from, e.g. train_gpt_mlx or train_gpt_mlx_jepa_sidecar.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    env = load_config_env(args.config)
    for key, value in env.items():
        os.environ.setdefault(key, value)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import mlx.core as mx
    import sentencepiece as spm
    import train_gpt_mlx as base

    trainer = importlib.import_module(args.trainer_module)
    hp = trainer.Hyperparameters()
    if args.prefix_len <= 0 or args.prefix_len > args.seq_len:
        raise ValueError(f"--prefix-len must be in [1, --seq-len], got {args.prefix_len} > {args.seq_len}")

    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(hp.tokenizer_path))
    if hasattr(trainer, "make_sidecar_gpt"):
        model = trainer.make_sidecar_gpt(hp, sp)
    elif hasattr(trainer, "make_export_eval_model"):
        model = trainer.make_export_eval_model(hp, sp)
    elif hasattr(trainer, "make_gpt"):
        model = trainer.make_gpt(hp, sp)
    else:
        model = base.make_gpt(hp, sp)
    if args.checkpoint is not None:
        flat_state = {name: value for name, value in mx.load(str(args.checkpoint)).items()}
        base.apply_flat_arrays(model, flat_state)

    rng = np.random.default_rng(args.seed)
    max_abs_diff = 0.0
    trials: list[dict[str, float | int]] = []
    for trial_idx in range(args.trials):
        seq_a = rng.integers(0, hp.vocab_size, size=(1, args.seq_len), dtype=np.int32)
        seq_b = np.array(seq_a, copy=True)
        seq_b[:, args.prefix_len :] = rng.integers(
            0,
            hp.vocab_size,
            size=(1, args.seq_len - args.prefix_len),
            dtype=np.int32,
        )
        x_a = mx.array(seq_a, dtype=mx.int32)
        x_b = mx.array(seq_b, dtype=mx.int32)
        uses_operator_codes = (
            getattr(model, "logic_sidecar", None) is not None
            or getattr(model, "polarity_detector", None) is not None
            or bool(getattr(model, "sidecar_polarity_write", False))
        )
        op_a = model.operator_codes_for_input(x_a) if uses_operator_codes and hasattr(model, "operator_codes_for_input") else None
        op_b = model.operator_codes_for_input(x_b) if uses_operator_codes and hasattr(model, "operator_codes_for_input") else None
        logits_a = model.forward_logits(x_a, op_a).astype(mx.float32)
        logits_b = model.forward_logits(x_b, op_b).astype(mx.float32)
        diff = mx.max(mx.abs(logits_a[:, : args.prefix_len, :] - logits_b[:, : args.prefix_len, :]))
        mx.eval(diff)
        diff_value = float(diff.item())
        max_abs_diff = max(max_abs_diff, diff_value)
        trials.append({"trial": trial_idx, "max_abs_prefix_logit_diff": diff_value})

    report = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "trainer_module": args.trainer_module,
        "register_layout": hp.register_layout,
        "register_mask_mode": hp.register_mask_mode,
        "num_registers": hp.num_registers,
        "register_stride": hp.register_stride,
        "seq_len": args.seq_len,
        "prefix_len": args.prefix_len,
        "trials": trials,
        "max_abs_prefix_logit_diff": max_abs_diff,
        "passes_tolerance": bool(max_abs_diff <= args.tolerance),
        "tolerance": args.tolerance,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
