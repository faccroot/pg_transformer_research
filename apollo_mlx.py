from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


def _stable_randn(shape: tuple[int, ...], seed: int) -> mx.array:
    # Distribution-compatible random Gaussian for APOLLO projections.
    # This is intentionally dependency-free for MLX hosts, but it is not
    # seed-equivalent to torch.Generator used by the official reference.
    rng = np.random.default_rng(int(seed))
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _next_seed(seed: int, adv: int = 15) -> int:
    rng = np.random.default_rng(int(seed))
    values = rng.integers(0, np.iinfo(np.int64).max, size=max(int(adv), 1), dtype=np.int64)
    return int(values[-1])


def _l2_norm(x: mx.array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> mx.array:
    return mx.sqrt(mx.sum(x.astype(mx.float32) * x.astype(mx.float32), axis=axis, keepdims=keepdims) + 1.0e-12)


@dataclass
class _ProjectorState:
    kind: str
    matrix: mx.array
    seed: int


class RandomProjector:
    def __init__(
        self,
        rank: int,
        *,
        update_proj_gap: int,
        proj_type: str = "std",
        seed: int = 0,
    ):
        self.rank = int(rank)
        self.update_proj_gap = int(update_proj_gap)
        self.proj_type = str(proj_type)
        self.seed = int(seed)
        self.state: _ProjectorState | None = None

    def _resolve_kind(self, grad: mx.array) -> str:
        rows, cols = map(int, grad.shape)
        if self.proj_type == "std":
            return "right" if rows >= cols else "left"
        if self.proj_type == "reverse_std":
            return "left" if rows >= cols else "right"
        if self.proj_type in {"left", "right"}:
            return self.proj_type
        raise ValueError(f"Unsupported APOLLO proj_type: {self.proj_type}")

    def _build_state(self, grad: mx.array) -> _ProjectorState:
        kind = self._resolve_kind(grad)
        rows, cols = map(int, grad.shape)
        if kind == "left":
            shape = (rows, self.rank)
        else:
            shape = (self.rank, cols)
        matrix = _stable_randn(shape, self.seed) / math.sqrt(max(self.rank, 1))
        next_seed = _next_seed(self.seed)
        return _ProjectorState(kind=kind, matrix=matrix, seed=next_seed)

    def project(self, grad: mx.array, step: int) -> mx.array:
        if self.state is None or (self.update_proj_gap > 0 and step % self.update_proj_gap == 0):
            self.state = self._build_state(grad)
            self.seed = self.state.seed
        assert self.state is not None
        if self.state.kind == "left":
            return self.state.matrix.T @ grad
        return grad @ self.state.matrix.T


class ApolloMatrixOptimizer:
    def __init__(
        self,
        keys: list[str],
        params: dict[str, mx.array],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        rank: int = 1,
        scale: float = 128.0,
        scale_type: str = "tensor",
        update_proj_gap: int = 200,
        proj_type: str = "std",
        seed_base: int = 0,
        scale_front: bool = False,
        disable_nl: bool = False,
    ):
        self.keys = list(keys)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.rank = int(rank)
        self.scale = float(scale)
        self.scale_type = str(scale_type)
        self.update_proj_gap = int(update_proj_gap)
        self.proj_type = str(proj_type)
        self.seed_base = int(seed_base)
        self.scale_front = bool(scale_front)
        self.disable_nl = bool(disable_nl)

        self.steps = {k: 0 for k in self.keys}
        self.exp_avg: dict[str, mx.array] = {}
        self.exp_avg_sq: dict[str, mx.array] = {}
        self.projectors = {
            k: RandomProjector(
                self.rank,
                update_proj_gap=self.update_proj_gap,
                proj_type=self.proj_type,
                seed=(self.seed_base + idx + 1),
            )
            for idx, k in enumerate(self.keys)
        }
        self.prev_scaled_norm: dict[str, mx.array] = {}

        for k in self.keys:
            if params[k].ndim != 2:
                raise ValueError(f"APOLLO matrix optimizer only supports 2D params, got {k}: {params[k].shape}")

    def step(
        self,
        params: dict[str, mx.array],
        grads: dict[str, mx.array],
        *,
        lr_mul: float,
    ) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array]:
        step_size = self.lr * float(lr_mul)
        out: dict[str, mx.array] = {}
        alignment = mx.array(0.0, dtype=mx.float32)
        grad_sq = mx.array(0.0, dtype=mx.float32)
        buf_sq = mx.array(0.0, dtype=mx.float32)

        for k in self.keys:
            p = params[k]
            g_full = grads[k].astype(mx.float32)
            step = self.steps[k]

            g_low = self.projectors[k].project(g_full, step)
            if k not in self.exp_avg:
                self.exp_avg[k] = mx.zeros_like(g_low)
                self.exp_avg_sq[k] = mx.zeros_like(g_low)

            prev_avg = self.exp_avg[k].astype(mx.float32)
            g_low32 = g_low.astype(mx.float32)
            alignment = alignment + mx.sum(g_low32 * prev_avg)
            grad_sq = grad_sq + mx.sum(g_low32 * g_low32)
            buf_sq = buf_sq + mx.sum(prev_avg * prev_avg)

            exp_avg = self.beta1 * self.exp_avg[k] + (1.0 - self.beta1) * g_low
            exp_avg_sq = self.beta2 * self.exp_avg_sq[k] + (1.0 - self.beta2) * (g_low * g_low)
            self.exp_avg[k] = exp_avg
            self.exp_avg_sq[k] = exp_avg_sq
            self.steps[k] = step + 1

            denom = mx.sqrt(exp_avg_sq.astype(mx.float32)) + self.eps
            norm_grad_low = exp_avg.astype(mx.float32) / denom

            if self.scale_type == "tensor":
                grad_scaling = _l2_norm(norm_grad_low) / (_l2_norm(g_low32) + 1.0e-8)
            elif self.scale_type == "channel":
                norm_dim = 0 if int(g_full.shape[0]) < int(g_full.shape[1]) else 1
                grad_scaling = _l2_norm(norm_grad_low, axis=norm_dim) / (_l2_norm(g_low32, axis=norm_dim) + 1.0e-8)
                if norm_dim == 1:
                    grad_scaling = mx.expand_dims(grad_scaling, axis=1)
            else:
                raise ValueError(f"Unsupported APOLLO scale_type: {self.scale_type}")

            scaled_grad = g_full * grad_scaling
            if self.scale_front:
                scaled_grad = scaled_grad * math.sqrt(self.scale)

            if not self.disable_nl:
                cur_norm = _l2_norm(scaled_grad)
                prev_norm = self.prev_scaled_norm.get(k)
                if prev_norm is not None:
                    limiter = mx.maximum(cur_norm / (prev_norm + 1.0e-8), 1.01) / 1.01
                    scaled_grad = scaled_grad / limiter
                    cur_norm = cur_norm / limiter
                self.prev_scaled_norm[k] = cur_norm

            if not self.scale_front:
                scaled_grad = scaled_grad * math.sqrt(self.scale)

            bias_correction1 = 1.0 - self.beta1 ** self.steps[k]
            bias_correction2 = 1.0 - self.beta2 ** self.steps[k]
            corrected_step = step_size * math.sqrt(max(bias_correction2, 1.0e-12)) / max(bias_correction1, 1.0e-12)
            out[k] = p - corrected_step * scaled_grad.astype(p.dtype)

        return out, alignment, grad_sq, buf_sq
