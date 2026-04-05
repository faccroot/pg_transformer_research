#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import train_gpt_mlx as base
import train_gpt_mlx_jepa_aux as aux


def max_abs_diff(lhs: mx.array, rhs: mx.array) -> float:
    return float(mx.max(mx.abs(lhs.astype(mx.float32) - rhs.astype(mx.float32))).item())


def common_param_keys(lhs: dict[str, mx.array], rhs: dict[str, mx.array]) -> list[str]:
    return sorted(set(lhs) & set(rhs))


def build_aux_model(args: aux.Hyperparameters) -> aux.GPTJEPAAux:
    return aux.GPTJEPAAux(
        **base.gpt_kwargs_from_args(args, None),
        jepa_chunk_size=args.jepa_chunk_size,
        jepa_latent_dim=args.jepa_latent_dim,
        jepa_pred_hidden=args.jepa_pred_hidden,
        jepa_pred_weight=args.jepa_pred_weight,
        jepa_sigreg_weight=args.jepa_sigreg_weight,
        jepa_summary_mode=args.jepa_summary_mode,
        jepa_pred_mode=args.jepa_pred_mode,
        jepa_pred_init_std=args.jepa_pred_init_std,
        jepa_sigreg_knots=args.jepa_sigreg_knots,
        jepa_sigreg_num_proj=args.jepa_sigreg_num_proj,
        jepa_sigreg_seed=args.jepa_sigreg_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that JEPA aux reduces cleanly to baseline GPT when disabled")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--tol-loss", type=float, default=1e-6)
    parser.add_argument("--tol-grad", type=float, default=1e-5)
    parser.add_argument("--tol-step", type=float, default=1e-5)
    parser.add_argument("support_files", nargs="*")
    args_cli = parser.parse_args()

    base_args = base.Hyperparameters()
    base_args.train_seq_len = args_cli.seq_len
    base_args.tie_embeddings = True

    aux_args = aux.Hyperparameters()
    aux_args.train_seq_len = args_cli.seq_len
    aux_args.tie_embeddings = True
    aux_args.jepa_pred_weight = 0.0
    aux_args.jepa_sigreg_weight = 0.0
    aux_args.jepa_aux_start_frac = 0.0
    aux_args.jepa_aux_ramp_frac = 0.0
    aux_args.jepa_grad_scrub_nonfinite = False
    aux_args.jepa_log_nonfinite = False

    mx.random.seed(args_cli.seed)
    base_model = base.make_gpt(base_args, None)
    mx.random.seed(args_cli.seed)
    aux_model = build_aux_model(aux_args)

    base_params = base.flat_parameter_state(base_model)
    aux_params = base.flat_parameter_state(aux_model)
    for key, value in base_params.items():
        if key in aux_params:
            aux_params[key] = value
    base.apply_flat_arrays(aux_model, aux_params)

    rng = np.random.default_rng(args_cli.seed + 1)
    x = mx.array(rng.integers(0, base_args.vocab_size, size=(args_cli.batch_size, args_cli.seq_len), dtype=np.int32))
    y = mx.array(rng.integers(0, base_args.vocab_size, size=(args_cli.batch_size, args_cli.seq_len), dtype=np.int32))

    base_loss = base_model.loss(x, y)
    aux_ce = aux_model.ce_loss(x, y)
    aux_total, aux_ce_terms, aux_pred, aux_sig = aux_model.loss_terms(x, y, aux_scale=0.0)
    mx.eval(base_loss, aux_ce, aux_total, aux_ce_terms, aux_pred, aux_sig)

    base_loss_value = float(base_loss.item())
    aux_ce_value = float(aux_ce.item())
    aux_total_value = float(aux_total.item())

    base_loss_and_grad = nn.value_and_grad(base_model, lambda xb, yb: base_model.loss(xb, yb))
    aux_loss_and_grad = nn.value_and_grad(aux_model, lambda xb, yb: aux_model.ce_loss(xb, yb))
    base_loss2, base_grads = base_loss_and_grad(x, y)
    aux_loss2, aux_grads = aux_loss_and_grad(x, y)
    mx.eval(base_loss2, aux_loss2, base_grads, aux_grads)

    base_grad_flat = dict(tree_flatten(base_grads))
    aux_grad_flat = dict(tree_flatten(aux_grads))
    common_keys = common_param_keys(base_grad_flat, aux_grad_flat)
    grad_diff = max(max_abs_diff(base_grad_flat[k], aux_grad_flat[k]) for k in common_keys)

    base_opt = base.SplitOptimizers(base_model, base_args)
    aux_opt = base.SplitOptimizers(aux_model, aux_args)
    base_opt.step(base_model, base_grads, step=0, lr_mul=1.0)
    aux_opt.step(aux_model, aux_grads, step=0, lr_mul=1.0)
    mx.synchronize()

    base_after = base.flat_parameter_state(base_model)
    aux_after = base.flat_parameter_state(aux_model)
    step_diff = max(max_abs_diff(base_after[k], aux_after[k]) for k in common_keys)

    report = {
        "seed": args_cli.seed,
        "batch_size": args_cli.batch_size,
        "seq_len": args_cli.seq_len,
        "base_loss": base_loss_value,
        "aux_ce_loss": aux_ce_value,
        "aux_total_loss_scale0": aux_total_value,
        "loss_diff_base_vs_aux_ce": abs(base_loss_value - aux_ce_value),
        "loss_diff_base_vs_aux_total_scale0": abs(base_loss_value - aux_total_value),
        "aux_pred_loss_scale0": float(aux_pred.item()),
        "aux_sigreg_loss_scale0": float(aux_sig.item()),
        "common_param_count": len(common_keys),
        "max_abs_grad_diff": grad_diff,
        "max_abs_post_step_param_diff": step_diff,
        "pass": (
            abs(base_loss_value - aux_ce_value) <= args_cli.tol_loss
            and abs(base_loss_value - aux_total_value) <= args_cli.tol_loss
            and grad_diff <= args_cli.tol_grad
            and step_diff <= args_cli.tol_step
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
