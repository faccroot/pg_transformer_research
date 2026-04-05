from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from apollo_mlx import ApolloMatrixOptimizer


def _stable_randn_np(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(shape, dtype=np.float32)


def _next_seed_np(seed: int, adv: int = 15) -> int:
    rng = np.random.default_rng(int(seed))
    values = rng.integers(0, np.iinfo(np.int64).max, size=max(int(adv), 1), dtype=np.int64)
    return int(values[-1])


def _l2_norm_np(x: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.sum(x.astype(np.float32) * x.astype(np.float32), axis=axis, keepdims=keepdims) + 1.0e-12)


def _reference_apollo_step(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    step: int,
    seed: int,
    exp_avg: np.ndarray | None,
    exp_avg_sq: np.ndarray | None,
    prev_scaled_norm: np.ndarray | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    rank: int,
    scale: float,
    scale_type: str,
    scale_front: bool,
    disable_nl: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    grad = grad.astype(np.float32)
    rows, cols = grad.shape
    if rows >= cols:
        proj = _stable_randn_np((rank, cols), seed) / math.sqrt(max(rank, 1))
        grad_low = grad @ proj.T
        norm_dim = 1
    else:
        proj = _stable_randn_np((rows, rank), seed) / math.sqrt(max(rank, 1))
        grad_low = proj.T @ grad
        norm_dim = 0
    if exp_avg is None:
        exp_avg = np.zeros_like(grad_low)
        exp_avg_sq = np.zeros_like(grad_low)
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad_low
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad_low * grad_low)
    denom = np.sqrt(exp_avg_sq.astype(np.float32)) + eps
    norm_grad_low = exp_avg.astype(np.float32) / denom
    if scale_type == "tensor":
        grad_scaling = _l2_norm_np(norm_grad_low) / (_l2_norm_np(grad_low) + 1.0e-8)
    else:
        grad_scaling = _l2_norm_np(norm_grad_low, axis=norm_dim) / (_l2_norm_np(grad_low, axis=norm_dim) + 1.0e-8)
        if norm_dim == 1:
            grad_scaling = np.expand_dims(grad_scaling, axis=1)
    scaled_grad = grad * grad_scaling
    if scale_front:
        scaled_grad = scaled_grad * math.sqrt(scale)
    if not disable_nl:
        cur_norm = _l2_norm_np(scaled_grad)
        if prev_scaled_norm is not None:
            limiter = np.maximum(cur_norm / (prev_scaled_norm + 1.0e-8), 1.01) / 1.01
            scaled_grad = scaled_grad / limiter
            cur_norm = cur_norm / limiter
        prev_scaled_norm = cur_norm
    if not scale_front:
        scaled_grad = scaled_grad * math.sqrt(scale)
    step_num = step + 1
    bias_correction1 = 1.0 - beta1**step_num
    bias_correction2 = 1.0 - beta2**step_num
    corrected_step = lr * math.sqrt(max(bias_correction2, 1.0e-12)) / max(bias_correction1, 1.0e-12)
    updated = param.astype(np.float32) - corrected_step * scaled_grad.astype(np.float32)
    return updated, exp_avg, exp_avg_sq, prev_scaled_norm


def test_apollo_seed_base_offsets_projector_seeds() -> None:
    params = {
        "a": mx.zeros((4, 4), dtype=mx.float32),
        "b": mx.zeros((4, 4), dtype=mx.float32),
    }
    opt0 = ApolloMatrixOptimizer(
        ["a"],
        params,
        lr=0.005,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        rank=1,
        scale=64.0,
        seed_base=0,
    )
    opt1 = ApolloMatrixOptimizer(
        ["b"],
        params,
        lr=0.005,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        rank=1,
        scale=64.0,
        seed_base=1,
    )
    assert opt0.projectors["a"].seed == 1
    assert opt1.projectors["b"].seed == 2


def test_apollo_two_step_update_matches_reference() -> None:
    shape = (8, 6)
    rng = np.random.default_rng(123)
    param0 = rng.standard_normal(shape, dtype=np.float32)
    grad1 = rng.standard_normal(shape, dtype=np.float32)
    grad2 = rng.standard_normal(shape, dtype=np.float32)
    params = {"w": mx.array(param0)}
    opt = ApolloMatrixOptimizer(
        ["w"],
        params,
        lr=0.005,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        rank=1,
        scale=64.0,
        scale_type="tensor",
        update_proj_gap=200,
        proj_type="std",
        seed_base=0,
        scale_front=True,
        disable_nl=False,
    )

    ref_exp_avg = None
    ref_exp_avg_sq = None
    ref_prev_norm = None
    ref_param = param0.copy()
    ref_seed = 1

    upd1, *_ = opt.step({"w": mx.array(ref_param)}, {"w": mx.array(grad1)}, lr_mul=1.0)
    ref_param, ref_exp_avg, ref_exp_avg_sq, ref_prev_norm = _reference_apollo_step(
        ref_param,
        grad1,
        step=0,
        seed=ref_seed,
        exp_avg=ref_exp_avg,
        exp_avg_sq=ref_exp_avg_sq,
        prev_scaled_norm=ref_prev_norm,
        lr=0.005,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        rank=1,
        scale=64.0,
        scale_type="tensor",
        scale_front=True,
        disable_nl=False,
    )
    np.testing.assert_allclose(np.array(upd1["w"]), ref_param, rtol=1e-5, atol=1e-6)

    upd2, *_ = opt.step({"w": mx.array(np.array(upd1["w"]))}, {"w": mx.array(grad2)}, lr_mul=1.0)
    ref_param, ref_exp_avg, ref_exp_avg_sq, ref_prev_norm = _reference_apollo_step(
        ref_param,
        grad2,
        step=1,
        seed=ref_seed,
        exp_avg=ref_exp_avg,
        exp_avg_sq=ref_exp_avg_sq,
        prev_scaled_norm=ref_prev_norm,
        lr=0.005,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        rank=1,
        scale=64.0,
        scale_type="tensor",
        scale_front=True,
        disable_nl=False,
    )
    np.testing.assert_allclose(np.array(upd2["w"]), ref_param, rtol=1e-5, atol=1e-6)

