#!/usr/bin/env python3
"""
Baseline GPT with a training-only JEPA auxiliary loss on chunk-boundary hidden states.

The generative path is unchanged:
tokens -> transformer -> logits -> cross-entropy

The JEPA path is auxiliary only:
hidden(chunk_t) -> projection -> latent_t
predict(latent_t) -> latent_{t+1}

This keeps exact token likelihood as the primary objective while shaping hidden-state geometry
with a lightweight latent prediction loss plus SIGReg.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_unflatten

import train_gpt_mlx as base


_JEPA_GEOM_DEFAULT = int(os.environ.get("JEPA_GEOM_DIM", os.environ.get("JEPA_LATENT_DIM", 96)))
_JEPA_DYN_DEFAULT = int(os.environ.get("JEPA_DYN_DIM", 32))


def l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    denom = mx.sqrt(mx.sum(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (x32 / denom).astype(x.dtype)


class Hyperparameters(base.Hyperparameters):
    jepa_chunk_size: int = int(os.environ.get("JEPA_CHUNK_SIZE", 8))
    jepa_geom_dim: int = _JEPA_GEOM_DEFAULT
    jepa_dyn_dim: int = _JEPA_DYN_DEFAULT
    jepa_pred_hidden: int = int(os.environ.get("JEPA_PRED_HIDDEN", max(_JEPA_DYN_DEFAULT * 2, 64)))
    jepa_tap_layer: int = int(os.environ.get("JEPA_TAP_LAYER", -1))
    jepa_pred_offset: int = int(os.environ.get("JEPA_PRED_OFFSET", 1))
    jepa_pred_weight: float = float(os.environ.get("JEPA_PRED_WEIGHT", 0.10))
    jepa_pred_end_weight: float = float(
        os.environ.get("JEPA_PRED_END_WEIGHT", os.environ.get("JEPA_PRED_WEIGHT", 0.10))
    )
    jepa_pred_decay_start_frac: float = float(os.environ.get("JEPA_PRED_DECAY_START_FRAC", 0.0))
    jepa_pred_decay_end_frac: float = float(os.environ.get("JEPA_PRED_DECAY_END_FRAC", 1.0))
    jepa_sigreg_weight: float = float(os.environ.get("JEPA_SIGREG_WEIGHT", 0.01))
    jepa_spherical_weight: float = float(os.environ.get("JEPA_SPHERICAL_WEIGHT", 0.0))
    jepa_dyn_spherical_weight: float = float(os.environ.get("JEPA_DYN_SPHERICAL_WEIGHT", 0.0))
    jepa_dyn_cov_weight: float = float(os.environ.get("JEPA_DYN_COV_WEIGHT", 0.0))
    jepa_cross_weight: float = float(os.environ.get("JEPA_CROSS_WEIGHT", 0.0))
    jepa_aux_start_frac: float = float(os.environ.get("JEPA_AUX_START_FRAC", 0.0))
    jepa_aux_ramp_frac: float = float(os.environ.get("JEPA_AUX_RAMP_FRAC", 0.0))
    jepa_summary_mode: str = os.environ.get("JEPA_SUMMARY_MODE", "query")
    jepa_pred_mode: str = os.environ.get("JEPA_PRED_MODE", "linear")
    jepa_pred_target_mode: str = os.environ.get("JEPA_PRED_TARGET_MODE", "next")
    jepa_pred_init_std: float = float(os.environ.get("JEPA_PRED_INIT_STD", "1e-4"))
    jepa_grad_scrub_nonfinite: bool = bool(int(os.environ.get("JEPA_GRAD_SCRUB_NONFINITE", "1")))
    jepa_log_nonfinite: bool = bool(int(os.environ.get("JEPA_LOG_NONFINITE", "1")))
    jepa_sigreg_knots: int = int(os.environ.get("JEPA_SIGREG_KNOTS", 17))
    jepa_sigreg_num_proj: int = int(os.environ.get("JEPA_SIGREG_NUM_PROJ", 256))
    jepa_sigreg_seed: int = int(os.environ.get("JEPA_SIGREG_SEED", 17))
    jepa_sigreg_sample_mode: str = os.environ.get("JEPA_SIGREG_SAMPLE_MODE", "flatten")
    jepa_sigreg_resample_proj: bool = bool(int(os.environ.get("JEPA_SIGREG_RESAMPLE_PROJ", "1")))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class SIGReg:
    def __init__(
        self,
        input_dim: int,
        knots: int = 17,
        num_proj: int = 256,
        seed: int = 17,
        *,
        resample_proj: bool = True,
    ):
        self.input_dim = input_dim
        self.seed = seed
        self.resample_proj = resample_proj
        t = np.linspace(0.0, 3.0, knots, dtype=np.float32)
        dt = 3.0 / max(knots - 1, 1)
        weights = np.full((knots,), 2.0 * dt, dtype=np.float32)
        if knots > 1:
            weights[0] = dt
            weights[-1] = dt
        window = np.exp(-(t * t) / 2.0, dtype=np.float32)
        rng = np.random.default_rng(seed)
        proj_matrix = rng.standard_normal((input_dim, num_proj), dtype=np.float32)
        proj_matrix = proj_matrix / np.sqrt(np.sum(proj_matrix * proj_matrix, axis=0, keepdims=True) + 1e-6)
        self.t = mx.array(t, dtype=mx.float32)
        self.phi = mx.array(window, dtype=mx.float32)
        self.weights = mx.array(weights * window, dtype=mx.float32)
        self.proj_matrix = mx.array(proj_matrix, dtype=mx.float32)
        self._rng = np.random.default_rng(seed)

    def _sample_proj_matrix(self) -> mx.array:
        proj_matrix = self._rng.standard_normal((self.input_dim, self.proj_matrix.shape[1]), dtype=np.float32)
        proj_matrix = proj_matrix / np.sqrt(np.sum(proj_matrix * proj_matrix, axis=0, keepdims=True) + 1e-6)
        return mx.array(proj_matrix, dtype=mx.float32)

    def __call__(self, proj: mx.array) -> mx.array:
        proj32 = proj.astype(mx.float32)
        proj_matrix = self._sample_proj_matrix() if self.resample_proj else self.proj_matrix
        x_t = mx.expand_dims(proj32 @ proj_matrix, axis=-1) * self.t
        cos_mean = mx.mean(mx.cos(x_t), axis=1)
        sin_mean = mx.mean(mx.sin(x_t), axis=1)
        err = mx.square(cos_mean - self.phi) + mx.square(sin_mean)
        statistic = mx.sum(err * self.weights, axis=-1) * float(proj.shape[1])
        return mx.mean(statistic)


class WeakSIGReg:
    """Sketched covariance isotropy regularizer.

    This is a lighter alternative to full Epps-Pulley SIGReg. It only matches
    second-order structure in a lower-dimensional random sketch, which is often
    enough to prevent collapse without over-constraining the latent dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        sketch_dim: int = 64,
        seed: int = 17,
        *,
        resample_proj: bool = True,
    ):
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.seed = seed
        self.resample_proj = resample_proj
        rng = np.random.default_rng(seed)
        sketch = rng.standard_normal((input_dim, sketch_dim), dtype=np.float32) / math.sqrt(float(input_dim))
        self.sketch_matrix = mx.array(sketch, dtype=mx.float32)
        self._rng = np.random.default_rng(seed)

    def _sample_sketch_matrix(self) -> mx.array:
        sketch = self._rng.standard_normal((self.input_dim, self.sketch_dim), dtype=np.float32)
        sketch = sketch / math.sqrt(float(self.input_dim))
        return mx.array(sketch, dtype=mx.float32)

    def __call__(self, proj: mx.array) -> mx.array:
        proj32 = proj.astype(mx.float32)
        sketch_matrix = self._sample_sketch_matrix() if self.resample_proj else self.sketch_matrix
        sketched = proj32 @ sketch_matrix
        centered = sketched - mx.mean(sketched, axis=1, keepdims=True)
        sample_count = int(centered.shape[1])
        if sample_count <= 1:
            return mx.array(0.0, dtype=mx.float32)
        cov = mx.matmul(mx.swapaxes(centered, -1, -2), centered) / float(max(sample_count - 1, 1))
        diag = mx.diagonal(cov, axis1=-2, axis2=-1)
        diag_loss = mx.mean(mx.square(diag - 1.0))
        offdiag_mask = mx.ones(cov.shape[-2:], dtype=mx.float32) - mx.eye(cov.shape[-1], dtype=mx.float32)
        offdiag = cov * offdiag_mask[None, :, :]
        offdiag_loss = mx.sum(mx.square(offdiag)) / float(max(cov.shape[0] * cov.shape[-1] * max(cov.shape[-1] - 1, 1), 1))
        return diag_loss + offdiag_loss


def spherical_uniformity(u: mx.array) -> mx.array:
    flat_u = u.reshape(-1, u.shape[-1]).astype(mx.float32)
    n = int(flat_u.shape[0])
    if n <= 1:
        return mx.array(0.0, dtype=mx.float32)
    gram = mx.abs(flat_u @ flat_u.T)
    mask = mx.ones((n, n), dtype=mx.float32) - mx.eye(n, dtype=mx.float32)
    return mx.sum(gram * mask) / float(n * (n - 1))


def cross_covariance_penalty(a: mx.array, b: mx.array) -> mx.array:
    flat_a = a.reshape(-1, a.shape[-1]).astype(mx.float32)
    flat_b = b.reshape(-1, b.shape[-1]).astype(mx.float32)
    n = int(flat_a.shape[0])
    if n <= 1:
        return mx.array(0.0, dtype=mx.float32)
    centered_a = flat_a - mx.mean(flat_a, axis=0, keepdims=True)
    centered_b = flat_b - mx.mean(flat_b, axis=0, keepdims=True)
    cov = (centered_a.T @ centered_b) / float(max(n - 1, 1))
    return mx.mean(mx.square(cov))


def spherical_second_moment_penalty(u: mx.array) -> mx.array:
    flat_u = u.reshape(-1, u.shape[-1]).astype(mx.float32)
    n = int(flat_u.shape[0])
    d = int(flat_u.shape[1])
    if n <= 1 or d <= 0:
        return mx.array(0.0, dtype=mx.float32)
    centered = flat_u - mx.mean(flat_u, axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(max(n - 1, 1))
    target_var = 1.0 / float(d)
    diag = mx.diag(cov)
    diag_loss = mx.mean(mx.square(diag - target_var))
    offdiag_mask = mx.ones((d, d), dtype=mx.float32) - mx.eye(d, dtype=mx.float32)
    offdiag_loss = mx.sum(mx.square(cov * offdiag_mask)) / float(max(d * (d - 1), 1))
    return diag_loss + offdiag_loss


class GPTJEPAAux(base.GPT):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        num_layer_templates: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_leaky_slope: float,
        tie_embeddings: bool,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
        num_registers: int = 0,
        register_layout: str = "prefix",
        register_stride: int = 256,
        logic_dim: int = 0,
        logic_layer_index: int | None = None,
        logic_route_to_next_token: bool = True,
        operator_routing: base.OperatorRoutingSpec | None = None,
        register_mask_mode: str = "bidirectional",
        logic_operator_mode: str = "all",
        polarity_detector_enabled: bool = False,
        polarity_detector_layer_index: int | None = None,
        polarity_detector_hidden_dim: int = 0,
        polarity_seed_blend: float = 1.0,
        polarity_seed_weight: float = 0.0,
        polarity_sparse_weight: float = 0.0,
        polarity_smooth_weight: float = 0.0,
        jepa_chunk_size: int,
        jepa_geom_dim: int,
        jepa_dyn_dim: int,
        jepa_pred_hidden: int,
        jepa_tap_layer: int,
        jepa_pred_offset: int,
        jepa_pred_weight: float,
        jepa_sigreg_weight: float,
        jepa_spherical_weight: float,
        jepa_dyn_spherical_weight: float,
        jepa_dyn_cov_weight: float,
        jepa_cross_weight: float,
        jepa_summary_mode: str,
        jepa_pred_mode: str,
        jepa_pred_target_mode: str,
        jepa_pred_init_std: float,
        jepa_sigreg_knots: int,
        jepa_sigreg_num_proj: int,
        jepa_sigreg_seed: int,
        jepa_sigreg_sample_mode: str,
        jepa_sigreg_resample_proj: bool,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_layer_templates=num_layer_templates,
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            mlp_leaky_slope=mlp_leaky_slope,
            tie_embeddings=tie_embeddings,
            logit_chunk_tokens=logit_chunk_tokens,
            logit_softcap=logit_softcap,
            rope_base=rope_base,
            tied_embed_init_std=tied_embed_init_std,
            qk_gain_init=qk_gain_init,
            num_registers=num_registers,
            register_layout=register_layout,
            register_stride=register_stride,
            logic_dim=logic_dim,
            logic_layer_index=logic_layer_index,
            logic_route_to_next_token=logic_route_to_next_token,
            operator_routing=operator_routing,
            register_mask_mode=register_mask_mode,
            logic_operator_mode=logic_operator_mode,
            polarity_detector_enabled=polarity_detector_enabled,
            polarity_detector_layer_index=polarity_detector_layer_index,
            polarity_detector_hidden_dim=polarity_detector_hidden_dim,
            polarity_seed_blend=polarity_seed_blend,
            polarity_seed_weight=polarity_seed_weight,
            polarity_sparse_weight=polarity_sparse_weight,
            polarity_smooth_weight=polarity_smooth_weight,
        )
        if jepa_chunk_size <= 1:
            raise ValueError(f"JEPA_CHUNK_SIZE must be > 1, got {jepa_chunk_size}")
        if jepa_pred_offset <= 0:
            raise ValueError(f"JEPA_PRED_OFFSET must be > 0, got {jepa_pred_offset}")
        if jepa_summary_mode not in {"last", "mean", "mean_last", "query"}:
            raise ValueError(f"Unsupported JEPA_SUMMARY_MODE={jepa_summary_mode!r}")
        if jepa_pred_mode not in {"linear", "mlp", "residual_linear", "residual_mlp"}:
            raise ValueError(f"Unsupported JEPA_PRED_MODE={jepa_pred_mode!r}")
        if jepa_pred_target_mode not in {"next", "delta"}:
            raise ValueError(f"Unsupported JEPA_PRED_TARGET_MODE={jepa_pred_target_mode!r}")
        if jepa_sigreg_sample_mode not in {"flatten", "by_chunk"}:
            raise ValueError(f"Unsupported JEPA_SIGREG_SAMPLE_MODE={jepa_sigreg_sample_mode!r}")
        if jepa_geom_dim <= 0 or jepa_dyn_dim <= 0:
            raise ValueError(f"JEPA dims must be positive, got geom={jepa_geom_dim} dyn={jepa_dyn_dim}")
        self.jepa_chunk_size = jepa_chunk_size
        self.jepa_geom_dim = jepa_geom_dim
        self.jepa_dyn_dim = jepa_dyn_dim
        self.jepa_pred_offset = jepa_pred_offset
        self.jepa_pred_weight = jepa_pred_weight
        self.jepa_sigreg_weight = jepa_sigreg_weight
        self.jepa_spherical_weight = jepa_spherical_weight
        self.jepa_dyn_spherical_weight = jepa_dyn_spherical_weight
        self.jepa_dyn_cov_weight = jepa_dyn_cov_weight
        self.jepa_cross_weight = jepa_cross_weight
        self.jepa_summary_mode = jepa_summary_mode
        self.jepa_pred_mode = jepa_pred_mode
        self.jepa_pred_target_mode = jepa_pred_target_mode
        self.jepa_pred_init_std = jepa_pred_init_std
        self.jepa_sigreg_sample_mode = jepa_sigreg_sample_mode
        self.jepa_tap_layer_indices = self._default_tap_layer_indices(jepa_tap_layer)
        self.jepa_geom_summary_query = (
            mx.random.normal((dim,), dtype=mx.float32) * (dim ** -0.5)
        ).astype(mx.float32)
        self.jepa_dyn_summary_query = (
            mx.random.normal((dim,), dtype=mx.float32) * (dim ** -0.5)
        ).astype(mx.float32)
        self.jepa_geom_proj = base.CastedLinear(dim, jepa_geom_dim)
        self.jepa_dyn_proj = base.CastedLinear(dim, jepa_dyn_dim)
        self.jepa_pred_linear = None
        self.jepa_pred_fc = None
        self.jepa_pred_out = None
        if jepa_pred_mode in {"linear", "residual_linear"}:
            self.jepa_pred_linear = base.CastedLinear(jepa_dyn_dim, jepa_dyn_dim)
            self.jepa_pred_linear.weight = (
                mx.random.normal(self.jepa_pred_linear.weight.shape, dtype=mx.float32) * jepa_pred_init_std
            )
        else:
            self.jepa_pred_fc = base.CastedLinear(jepa_dyn_dim, jepa_pred_hidden)
            self.jepa_pred_out = base.CastedLinear(jepa_pred_hidden, jepa_dyn_dim)
            self.jepa_pred_out.weight = (
                mx.random.normal(self.jepa_pred_out.weight.shape, dtype=mx.float32) * jepa_pred_init_std
            )
        self.jepa_sigreg = SIGReg(
            input_dim=jepa_geom_dim,
            knots=jepa_sigreg_knots,
            num_proj=jepa_sigreg_num_proj,
            seed=jepa_sigreg_seed,
            resample_proj=jepa_sigreg_resample_proj,
        )

    def _default_tap_layer_indices(self, requested_layer: int) -> tuple[int, ...]:
        if requested_layer >= 0:
            if requested_layer >= self.num_layers:
                raise ValueError(f"JEPA_TAP_LAYER={requested_layer} out of range for num_layers={self.num_layers}")
            return (requested_layer,)
        return (max(self.num_encoder_layers - 1, 0),)

    def _query_pool_chunks(self, chunks: mx.array, query: mx.array) -> mx.array:
        query32 = query.astype(mx.float32)
        scores = mx.sum(chunks.astype(mx.float32) * query32[None, None, None, :], axis=-1) / math.sqrt(
            float(chunks.shape[-1])
        )
        weights = mx.softmax(scores, axis=2)
        return mx.sum(chunks.astype(mx.float32) * weights[..., None], axis=2).astype(chunks.dtype)

    def chunk_summary_from_hidden(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        if hidden.shape[1] % self.jepa_chunk_size != 0:
            raise ValueError(
                f"TRAIN_SEQ_LEN={hidden.shape[1]} must be divisible by JEPA_CHUNK_SIZE={self.jepa_chunk_size}"
            )
        chunks = hidden.reshape(
            hidden.shape[0],
            hidden.shape[1] // self.jepa_chunk_size,
            self.jepa_chunk_size,
            hidden.shape[2],
        )
        last = chunks[:, :, -1, :]
        if self.jepa_summary_mode == "last":
            return last, last
        mean = mx.mean(chunks.astype(mx.float32), axis=2).astype(hidden.dtype)
        if self.jepa_summary_mode == "mean":
            return mean, mean
        if self.jepa_summary_mode == "mean_last":
            summary = (0.5 * (mean.astype(mx.float32) + last.astype(mx.float32))).astype(hidden.dtype)
            return summary, summary
        return (
            self._query_pool_chunks(chunks, self.jepa_geom_summary_query),
            self._query_pool_chunks(chunks, self.jepa_dyn_summary_query),
        )

    def forward_with_jepa_hidden(
        self,
        input_ids: mx.array,
        operator_codes: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        final_hidden, semantic_hidden, _aux = self.forward_with_jepa_hidden_aux(
            input_ids,
            operator_codes=operator_codes,
        )
        return final_hidden, semantic_hidden

    def forward_with_jepa_hidden_aux(
        self,
        input_ids: mx.array,
        operator_codes: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array | None]]:
        final_hidden, captured, aux = self.forward_hidden_with_aux(
            input_ids,
            capture_layers=self.jepa_tap_layer_indices,
            operator_codes=operator_codes,
        )
        tap_states = [captured[i] for i in self.jepa_tap_layer_indices]
        semantic_hidden = tap_states[0] if len(tap_states) == 1 else mx.stack(tap_states, axis=0).mean(axis=0)
        return final_hidden, semantic_hidden, aux

    def token_ce_from_hidden(self, hidden: mx.array, target_ids: mx.array) -> mx.array:
        x = hidden.reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x)
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x[s:e])
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

    def _tangent_displacement(self, source: mx.array, other: mx.array) -> mx.array:
        source32 = source.astype(mx.float32)
        other32 = other.astype(mx.float32)
        radial = mx.sum(source32 * other32, axis=-1, keepdims=True) * source32
        tangent = other32 - radial
        return l2_normalize(tangent.astype(base.COMPUTE_DTYPE))

    def _jepa_latent_views(
        self,
        hidden: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        geom_summary_h, dyn_summary_h = self.chunk_summary_from_hidden(hidden)
        z_geom = self.jepa_geom_proj(geom_summary_h).astype(base.COMPUTE_DTYPE)
        u_geom = l2_normalize(z_geom)
        z_dyn = self.jepa_dyn_proj(dyn_summary_h).astype(base.COMPUTE_DTYPE)
        u_dyn = l2_normalize(z_dyn)
        if z_dyn.shape[1] <= self.jepa_pred_offset:
            empty = u_dyn[:, :0, :]
            return z_geom, u_geom, z_dyn, u_dyn, empty, empty, empty, empty
        source = u_dyn[:, :-self.jepa_pred_offset, :]
        target_state = mx.stop_gradient(u_dyn[:, self.jepa_pred_offset :, :])
        if self.jepa_pred_mode in {"linear", "residual_linear"}:
            assert self.jepa_pred_linear is not None
            pred_state = self.jepa_pred_linear(source).astype(base.COMPUTE_DTYPE)
        else:
            assert self.jepa_pred_fc is not None and self.jepa_pred_out is not None
            pred_state = self.jepa_pred_out(nn.silu(self.jepa_pred_fc(source))).astype(base.COMPUTE_DTYPE)
        if self.jepa_pred_mode.startswith("residual_") and self.jepa_pred_target_mode == "next":
            pred_state = source + pred_state
        if self.jepa_pred_target_mode == "delta":
            pred = self._tangent_displacement(source, pred_state)
            pred_target = self._tangent_displacement(source, target_state)
        else:
            pred = l2_normalize(pred_state)
            pred_target = target_state
        return z_geom, u_geom, z_dyn, u_dyn, source, target_state, pred, pred_target

    def _sigreg_input(self, z_geom: mx.array) -> mx.array:
        if self.jepa_sigreg_sample_mode == "flatten":
            return z_geom.reshape(1, -1, z_geom.shape[-1])
        return mx.transpose(z_geom, (1, 0, 2))

    def jepa_terms_from_hidden(
        self,
        hidden: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        z_geom, u_geom, _z_dyn, u_dyn, _source, _target_state, pred, pred_target = self._jepa_latent_views(hidden)
        if pred_target.shape[1] <= 0:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, zero, zero, zero, zero
        pred_loss = mx.mean(
            (1.0 - mx.sum(pred.astype(mx.float32) * pred_target.astype(mx.float32), axis=-1)).astype(mx.float32)
        )
        sigreg_loss = self.jepa_sigreg(self._sigreg_input(z_geom))
        spherical_loss = spherical_uniformity(u_geom)
        dyn_spherical_loss = spherical_uniformity(u_dyn)
        dyn_cov_loss = spherical_second_moment_penalty(u_dyn)
        cross_loss = cross_covariance_penalty(u_geom, u_dyn)
        return pred_loss, sigreg_loss, spherical_loss, dyn_spherical_loss, dyn_cov_loss, cross_loss

    def jepa_debug_metrics_from_hidden(self, hidden: mx.array) -> tuple[mx.array, ...]:
        z_geom, u_geom, z_dyn, u_dyn, source, target_state, pred, pred_target = self._jepa_latent_views(hidden)
        if target_state.shape[1] <= 0:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero
        flat_z_geom = z_geom.reshape(-1, z_geom.shape[-1]).astype(mx.float32)
        flat_u_geom = u_geom.reshape(-1, u_geom.shape[-1]).astype(mx.float32)
        flat_u_dyn = u_dyn.reshape(-1, u_dyn.shape[-1]).astype(mx.float32)
        z_norms = mx.sqrt(mx.sum(mx.square(flat_z_geom), axis=-1) + 1e-6)
        z_centered = flat_z_geom - mx.mean(flat_z_geom, axis=0, keepdims=True)
        dyn_centered = flat_u_dyn - mx.mean(flat_u_dyn, axis=0, keepdims=True)
        z_dim_std = mx.sqrt(mx.mean(mx.square(z_centered), axis=0) + 1e-6)
        dyn_dim_std = mx.sqrt(mx.mean(mx.square(dyn_centered), axis=0) + 1e-6)
        src_tgt_cos = mx.mean(mx.sum(source.astype(mx.float32) * target_state.astype(mx.float32), axis=-1))
        pred_tgt_cos = mx.mean(mx.sum(pred.astype(mx.float32) * pred_target.astype(mx.float32), axis=-1))
        return (
            src_tgt_cos,
            pred_tgt_cos,
            spherical_uniformity(u_geom),
            spherical_uniformity(u_dyn),
            mx.mean(z_norms),
            mx.sqrt(mx.mean(mx.square(z_norms - mx.mean(z_norms)))),
            mx.mean(z_dim_std),
            mx.mean(dyn_dim_std),
            spherical_second_moment_penalty(u_dyn),
            mx.sqrt(cross_covariance_penalty(u_geom, u_dyn) + 1e-12),
        )

    def ce_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
    ) -> mx.array:
        final_hidden = super().__call__(input_ids, operator_codes=operator_codes)
        return self.token_ce_from_hidden(final_hidden, target_ids)

    def loss_terms(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
        aux_scale: float = 1.0,
        pred_weight: float | None = None,
        sigreg_weight: float | None = None,
        spherical_weight: float | None = None,
        dyn_spherical_weight: float | None = None,
        dyn_cov_weight: float | None = None,
        cross_weight: float | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        pred_weight = self.jepa_pred_weight if pred_weight is None else pred_weight
        sigreg_weight = self.jepa_sigreg_weight if sigreg_weight is None else sigreg_weight
        spherical_weight = self.jepa_spherical_weight if spherical_weight is None else spherical_weight
        dyn_spherical_weight = (
            self.jepa_dyn_spherical_weight if dyn_spherical_weight is None else dyn_spherical_weight
        )
        dyn_cov_weight = self.jepa_dyn_cov_weight if dyn_cov_weight is None else dyn_cov_weight
        cross_weight = self.jepa_cross_weight if cross_weight is None else cross_weight
        final_hidden, semantic_hidden, aux = self.forward_with_jepa_hidden_aux(input_ids, operator_codes=operator_codes)
        ce_loss = self.token_ce_from_hidden(final_hidden, target_ids)
        zero = mx.array(0.0, dtype=mx.float32)
        if aux_scale <= 0.0:
            pred_loss, sigreg_loss, spherical_loss, dyn_spherical_loss, dyn_cov_loss, cross_loss = (
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
            )
        else:
            pred_loss, sigreg_loss, spherical_loss, dyn_spherical_loss, dyn_cov_loss, cross_loss = (
                self.jepa_terms_from_hidden(semantic_hidden)
            )
        polarity_seed_loss, polarity_sparse_loss, polarity_smooth_loss = self.polarity_loss_terms_from_aux(aux)
        aux_scale_arr = mx.array(aux_scale, dtype=mx.float32)
        total = ce_loss + aux_scale_arr * (
            pred_weight * pred_loss
            + sigreg_weight * sigreg_loss
            + spherical_weight * spherical_loss
            + dyn_spherical_weight * dyn_spherical_loss
            + dyn_cov_weight * dyn_cov_loss
            + cross_weight * cross_loss
        ) + (
            self.polarity_seed_weight * polarity_seed_loss
            + self.polarity_sparse_weight * polarity_sparse_loss
            + self.polarity_smooth_weight * polarity_smooth_loss
        )
        return total, ce_loss, pred_loss, sigreg_loss, spherical_loss, dyn_spherical_loss, dyn_cov_loss, cross_loss


def exportable_flat_state(model: GPTJEPAAux) -> dict[str, mx.array]:
    return {
        k: v
        for k, v in tree_flatten(model.state)
        if not k.startswith("jepa_")
        and not any(part.startswith("_") for part in k.split("."))
        and not any(pattern in k for pattern in base.SERIALIZATION_SKIP_NAME_PATTERNS)
    }


def make_export_eval_model(
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None = None,
) -> base.GPT:
    return base.make_gpt(args, sp)


def assert_export_schema_matches(
    model: GPTJEPAAux,
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None = None,
) -> tuple[int, int]:
    reference_model = make_export_eval_model(args, sp)
    actual = exportable_flat_state(model)
    expected = base.exportable_flat_state(reference_model)
    actual_keys = tuple(sorted(actual))
    expected_keys = tuple(sorted(expected))
    if actual_keys != expected_keys:
        only_actual = [k for k in actual_keys if k not in expected][:10]
        only_expected = [k for k in expected_keys if k not in actual][:10]
        raise ValueError(
            "JEPA export schema mismatch against baseline GPT. "
            f"only_actual={only_actual} only_expected={only_expected}"
        )
    shape_mismatches = [
        k for k in actual_keys
        if tuple(actual[k].shape) != tuple(expected[k].shape) or str(actual[k].dtype) != str(expected[k].dtype)
    ]
    if shape_mismatches:
        sample = shape_mismatches[:10]
        raise ValueError(f"JEPA export tensor mismatch against baseline GPT for keys={sample}")
    return len(actual_keys), sum(int(np.prod(v.shape)) for v in actual.values())


def loss_and_grad_chunked(
    args: Hyperparameters,
    model: GPTJEPAAux,
    train_loader: base.TokenLoader,
    compiled_loss_and_grad,
    *,
    logic_phase_enabled: bool = True,
) -> tuple[mx.array, dict, tuple[mx.array, mx.array] | None]:
    chunk_sizes = base.token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    last_batch: tuple[mx.array, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x_np, y_np = train_loader.next_batch_np(chunk_tokens, args.train_seq_len)
        operator_codes = base.operator_codes_mx_for_numpy_batch(model, x_np, enabled=logic_phase_enabled)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        last_batch = (x, y)
        loss, grads = compiled_loss_and_grad(x, y, operator_codes)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items())), last_batch


def jepa_aux_scale_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    frac = base.training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    if frac < args.jepa_aux_start_frac:
        return 0.0
    if args.jepa_aux_ramp_frac <= 0.0:
        return 1.0
    return min((frac - args.jepa_aux_start_frac) / args.jepa_aux_ramp_frac, 1.0)


def jepa_pred_weight_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    frac = base.training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    start = args.jepa_pred_decay_start_frac
    end = args.jepa_pred_decay_end_frac
    if end <= start:
        return args.jepa_pred_end_weight if frac >= end else args.jepa_pred_weight
    if frac <= start:
        return args.jepa_pred_weight
    if frac >= end:
        return args.jepa_pred_end_weight
    mix = (frac - start) / (end - start)
    return args.jepa_pred_weight + mix * (args.jepa_pred_end_weight - args.jepa_pred_weight)


def first_nonfinite_keys(tree: dict, limit: int = 8) -> list[str]:
    bad: list[str] = []
    for key, value in tree_flatten(tree):
        arr = base._np_float32(value)
        if not np.isfinite(arr).all():
            bad.append(key)
            if len(bad) >= limit:
                break
    return bad


def zero_nonfinite_tree(tree: dict):
    cleaned = []
    for key, value in tree_flatten(tree):
        cleaned.append((key, mx.where(mx.isfinite(value), value, mx.zeros_like(value))))
    return tree_unflatten(cleaned)


def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    if args.train_seq_len % args.jepa_chunk_size != 0:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by JEPA_CHUNK_SIZE={args.jepa_chunk_size}"
        )

    dataset_name, actual_train_files, expected_train_files = base.validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = base.limit_validation_tokens(
        base.load_validation_tokens(args.val_files, args.train_seq_len),
        args.train_seq_len,
        args.val_max_seqs,
    )
    quant_eval_tokens = base.limit_validation_tokens(val_tokens, args.train_seq_len, args.quant_eval_max_seqs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = base.TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = GPTJEPAAux(
        **base.gpt_kwargs_from_args(args, sp),
        jepa_chunk_size=args.jepa_chunk_size,
        jepa_geom_dim=args.jepa_geom_dim,
        jepa_dyn_dim=args.jepa_dyn_dim,
        jepa_pred_hidden=args.jepa_pred_hidden,
        jepa_tap_layer=args.jepa_tap_layer,
        jepa_pred_offset=args.jepa_pred_offset,
        jepa_pred_weight=args.jepa_pred_weight,
        jepa_sigreg_weight=args.jepa_sigreg_weight,
        jepa_spherical_weight=args.jepa_spherical_weight,
        jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
        jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
        jepa_cross_weight=args.jepa_cross_weight,
        jepa_summary_mode=args.jepa_summary_mode,
        jepa_pred_mode=args.jepa_pred_mode,
        jepa_pred_target_mode=args.jepa_pred_target_mode,
        jepa_pred_init_std=args.jepa_pred_init_std,
        jepa_sigreg_knots=args.jepa_sigreg_knots,
        jepa_sigreg_num_proj=args.jepa_sigreg_num_proj,
        jepa_sigreg_seed=args.jepa_sigreg_seed,
        jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
        jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
    )
    export_key_count, export_param_count = assert_export_schema_matches(model, args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: base.GPT | None = None
    compiled_quant_ce = None
    compiled_quant_forward_logits = None

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    uses_logic = model.logic_sidecar is not None
    if compiled:
        if uses_logic:
            compiled_ce_loss_impl = mx.compile(
                lambda x, y, operator_codes: model.ce_loss(x, y, operator_codes),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_forward_logits_impl = mx.compile(
                lambda x, operator_codes: model.forward_logits(x, operator_codes),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_ce_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y, operator_codes: model.ce_loss(x, y, operator_codes)),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss_components_impl = mx.compile(
                lambda x, y, operator_codes: model.loss_terms(x, y, operator_codes),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y, operator_codes: model.loss_terms(x, y, operator_codes)[0]),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_ce_loss = lambda x, y, operator_codes=None: compiled_ce_loss_impl(x, y, operator_codes)
            compiled_forward_logits = lambda x, operator_codes=None: compiled_forward_logits_impl(x, operator_codes)
            compiled_ce_loss_and_grad = lambda x, y, operator_codes=None: compiled_ce_loss_and_grad_impl(x, y, operator_codes)
            compiled_loss_components = lambda x, y, operator_codes=None: compiled_loss_components_impl(x, y, operator_codes)
            compiled_loss_and_grad = lambda x, y, operator_codes=None: compiled_loss_and_grad_impl(x, y, operator_codes)
        else:
            compiled_ce_loss_impl = mx.compile(lambda x, y: model.ce_loss(x, y), inputs=model.state, outputs=model.state)
            compiled_forward_logits_impl = mx.compile(
                lambda x: model.forward_logits(x),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_ce_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y: model.ce_loss(x, y)),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss_components_impl = mx.compile(lambda x, y: model.loss_terms(x, y), inputs=model.state, outputs=model.state)
            compiled_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y: model.loss_terms(x, y)[0]),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_ce_loss = lambda x, y, operator_codes=None: compiled_ce_loss_impl(x, y)
            compiled_forward_logits = lambda x, operator_codes=None: compiled_forward_logits_impl(x)
            compiled_ce_loss_and_grad = lambda x, y, operator_codes=None: compiled_ce_loss_and_grad_impl(x, y)
            compiled_loss_components = lambda x, y, operator_codes=None: compiled_loss_components_impl(x, y)
            compiled_loss_and_grad = lambda x, y, operator_codes=None: compiled_loss_and_grad_impl(x, y)
    else:
        if uses_logic:
            compiled_ce_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y, operator_codes)
            compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits(x, operator_codes)
            compiled_ce_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y, operator_codes: model.ce_loss(x, y, operator_codes))
            compiled_loss_components = lambda x, y, operator_codes=None: model.loss_terms(x, y, operator_codes)
            compiled_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y, operator_codes: model.loss_terms(x, y, operator_codes)[0])
            compiled_ce_loss_and_grad = lambda x, y, operator_codes=None: compiled_ce_loss_and_grad_impl(x, y, operator_codes)
            compiled_loss_and_grad = lambda x, y, operator_codes=None: compiled_loss_and_grad_impl(x, y, operator_codes)
        else:
            compiled_ce_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y)
            compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits(x)
            compiled_ce_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y: model.ce_loss(x, y))
            compiled_loss_components = lambda x, y, operator_codes=None: model.loss_terms(x, y)
            compiled_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y: model.loss_terms(x, y)[0])
            compiled_ce_loss_and_grad = lambda x, y, operator_codes=None: compiled_ce_loss_and_grad_impl(x, y)
            compiled_loss_and_grad = lambda x, y, operator_codes=None: compiled_loss_and_grad_impl(x, y)

    trainable_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    export_params = export_param_count
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params_trainable:{trainable_params} model_params_export:{export_params} "
        f"vocab_size:{args.vocab_size} layers:{args.num_layers} layer_templates:{args.num_layer_templates} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(f"export_schema:matched keys:{export_key_count} params:{export_param_count}")
    log(
        f"logic_registers:registers:{args.num_registers} layout:{args.register_layout} stride:{args.register_stride} "
        f"logic_dim:{args.logic_dim} "
        f"logic_layer:{model.logic_layer_index if model.logic_sidecar is not None else 'off'} "
        f"logic_route_to_next:{args.logic_route_to_next_token} "
        f"register_mask_mode:{args.register_mask_mode} logic_operator_mode:{args.logic_operator_mode}"
    )
    log(
        f"polarity_detector:enabled:{int(args.polarity_detector_enabled)} "
        f"layer:{model.polarity_detector_layer_index if model.polarity_detector is not None else 'off'} "
        f"hidden_dim:{args.polarity_detector_hidden_dim} seed_blend:{args.polarity_seed_blend:.3f} "
        f"seed_weight:{args.polarity_seed_weight:.6f} sparse_weight:{args.polarity_sparse_weight:.6f} "
        f"smooth_weight:{args.polarity_smooth_weight:.6f}"
    )
    log(
        f"jepa_aux:chunk_size:{args.jepa_chunk_size} geom_dim:{args.jepa_geom_dim} dyn_dim:{args.jepa_dyn_dim} "
        f"tap_layer:{args.jepa_tap_layer} "
        f"pred_offset:{args.jepa_pred_offset} "
        f"pred_hidden:{args.jepa_pred_hidden} pred_weight:{args.jepa_pred_weight} "
        f"pred_end_weight:{args.jepa_pred_end_weight} "
        f"pred_decay_start_frac:{args.jepa_pred_decay_start_frac} "
        f"pred_decay_end_frac:{args.jepa_pred_decay_end_frac} "
        f"sigreg_weight:{args.jepa_sigreg_weight} sigreg_proj:{args.jepa_sigreg_num_proj} "
        f"spherical_weight:{args.jepa_spherical_weight} "
        f"dyn_spherical_weight:{args.jepa_dyn_spherical_weight} "
        f"dyn_cov_weight:{args.jepa_dyn_cov_weight} "
        f"cross_weight:{args.jepa_cross_weight} "
        f"sigreg_sample_mode:{args.jepa_sigreg_sample_mode} "
        f"sigreg_resample_proj:{int(args.jepa_sigreg_resample_proj)} "
        f"aux_start_frac:{args.jepa_aux_start_frac} aux_ramp_frac:{args.jepa_aux_ramp_frac} "
        f"summary_mode:{args.jepa_summary_mode} summary_queries:{'split' if args.jepa_summary_mode == 'query' else 'shared'} "
        f"pred_mode:{args.jepa_pred_mode} "
        f"pred_target_mode:{args.jepa_pred_target_mode} pred_init_std:{args.jepa_pred_init_std} "
        f"grad_scrub_nonfinite:{int(args.jepa_grad_scrub_nonfinite)} "
        f"tap_layers:{list(model.jepa_tap_layer_indices)} pred_metric:cosine"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} val_seqs:{(val_tokens.size - 1) // args.train_seq_len} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(
        f"compute_dtype:{base.COMPUTE_DTYPE} compile:{compiled} compile_mode:{args.mlx_compile} "
        f"quant_format:{args.quant_format} turbo_qat:{args.turbo_qat} "
        f"turbo_block:{args.turbo_block_size} turbo_bits_mse:{args.turbo_mse_bits} turbo_bits_prod:{args.turbo_prod_bits}"
    )
    if args.quant_eval_every > 0:
        log(
            f"quant_eval:enabled every:{args.quant_eval_every} seqs:{(quant_eval_tokens.size - 1) // args.train_seq_len}"
        )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, _ = loss_and_grad_chunked(args, model, train_loader, compiled_loss_and_grad)
                accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val_np = warm_chunk[:-1].reshape(-1, args.train_seq_len)
        x_val = mx.array(x_val_np, dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_operator_codes = base.operator_codes_mx_for_numpy_batch(model, x_val_np)
        warm_val_loss = compiled_ce_loss(x_val, y_val, warm_operator_codes)
        mx.eval(warm_val_loss)
        mx.synchronize()
        train_loader = base.TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_quant_eval = args.quant_eval_every > 0 and (last_step or step % args.quant_eval_every == 0)
        if should_validate or should_quant_eval:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if should_validate:
                val_loss, val_bpb = base.eval_val(
                    args,
                    model,
                    compiled_ce_loss,
                    compiled_forward_logits,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                )
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = make_export_eval_model(args, sp)
                    if uses_logic:
                        compiled_quant_ce_impl = mx.compile(
                            lambda x, y, operator_codes: quant_eval_model.loss(x, y, operator_codes),
                            inputs=quant_eval_model.state,
                            outputs=quant_eval_model.state,
                        )
                        compiled_quant_forward_logits_impl = mx.compile(
                            lambda x, operator_codes: quant_eval_model.forward_logits(x, operator_codes),
                            inputs=quant_eval_model.state,
                            outputs=quant_eval_model.state,
                        )
                        compiled_quant_ce = lambda x, y, operator_codes=None: compiled_quant_ce_impl(x, y, operator_codes)
                        compiled_quant_forward_logits = lambda x, operator_codes=None: compiled_quant_forward_logits_impl(x, operator_codes)
                    else:
                        compiled_quant_ce_impl = mx.compile(
                            lambda x, y: quant_eval_model.loss(x, y),
                            inputs=quant_eval_model.state,
                            outputs=quant_eval_model.state,
                        )
                        compiled_quant_forward_logits_impl = mx.compile(
                            lambda x: quant_eval_model.forward_logits(x),
                            inputs=quant_eval_model.state,
                            outputs=quant_eval_model.state,
                        )
                        compiled_quant_ce = lambda x, y, operator_codes=None: compiled_quant_ce_impl(x, y)
                        compiled_quant_forward_logits = lambda x, operator_codes=None: compiled_quant_forward_logits_impl(x)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = base.eval_val(
                    args,
                    model,
                    compiled_ce_loss,
                    compiled_forward_logits,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = base.eval_val(
                    args,
                    quant_eval_model,
                    compiled_quant_ce,
                    compiled_quant_forward_logits,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                log(
                    f"step:{step}/{args.iterations} quant_diag_seqs:{(quant_eval_tokens.size - 1) // args.train_seq_len} "
                    f"raw_val_loss:{raw_q_val_loss:.4f} raw_val_bpb:{raw_q_val_bpb:.4f} "
                    f"quant_val_loss:{q_val_loss:.4f} quant_val_bpb:{q_val_bpb:.4f} "
                    f"quant_gap_bpb:{q_val_bpb - raw_q_val_bpb:+.4f} int8_zlib_bytes:{len(quant_blob)} "
                    f"payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)} "
                    f"{base.format_quant_stats(quant_stats)} "
                    f"eval_time:{1000.0 * (time.perf_counter() - q_t0):.0f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        progress_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        lr_mul = args.lr_mul(step, progress_ms)
        qat_scale = base.turbo_qat_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        aux_scale = jepa_aux_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        pred_weight = jepa_pred_weight_for_progress(args, step, progress_ms, max_wallclock_ms)
        sigreg_weight = args.jepa_sigreg_weight
        spherical_weight = args.jepa_spherical_weight
        dyn_spherical_weight = args.jepa_dyn_spherical_weight
        dyn_cov_weight = args.jepa_dyn_cov_weight
        cross_weight = args.jepa_cross_weight
        model.set_turbo_qat(qat_active, qat_scale)
        dynamic_aux_weights = (
            pred_weight != args.jepa_pred_weight
            or sigreg_weight != args.jepa_sigreg_weight
            or spherical_weight != args.jepa_spherical_weight
            or dyn_spherical_weight != args.jepa_dyn_spherical_weight
            or dyn_cov_weight != args.jepa_dyn_cov_weight
            or cross_weight != args.jepa_cross_weight
            or aux_scale != 1.0
        )
        step_loss_and_grad = (
            (
                (lambda x, y, operator_codes=None: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner, operator_codes_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        operator_codes_inner,
                                aux_scale,
                                pred_weight,
                                sigreg_weight,
                                spherical_weight,
                                dyn_spherical_weight,
                                dyn_cov_weight,
                                cross_weight,
                            )[0] + qat_lambda * model.turbo_regularizer(),
                )(x, y, operator_codes))
                if uses_logic
                else (lambda x, y, operator_codes=None: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        aux_scale=aux_scale,
                        pred_weight=pred_weight,
                        sigreg_weight=sigreg_weight,
                        spherical_weight=spherical_weight,
                        dyn_spherical_weight=dyn_spherical_weight,
                        dyn_cov_weight=dyn_cov_weight,
                        cross_weight=cross_weight,
                    )[0] + qat_lambda * model.turbo_regularizer(),
                )(x, y))
            )
            if qat_active
            else (
                compiled_ce_loss_and_grad
                if aux_scale <= 0.0
                else (
                    compiled_loss_and_grad
                    if not dynamic_aux_weights
                    else (
                        (lambda x, y, operator_codes=None: nn.value_and_grad(
                            model,
                            lambda x_inner, y_inner, operator_codes_inner: model.loss_terms(
                                x_inner,
                                y_inner,
                                operator_codes_inner,
                                aux_scale,
                                pred_weight,
                                sigreg_weight,
                                spherical_weight,
                                dyn_spherical_weight,
                                dyn_cov_weight,
                                cross_weight,
                            )[0],
                        )(x, y, operator_codes))
                        if uses_logic
                        else (lambda x, y, operator_codes=None: nn.value_and_grad(
                            model,
                            lambda x_inner, y_inner: model.loss_terms(
                                x_inner,
                                y_inner,
                                aux_scale=aux_scale,
                                pred_weight=pred_weight,
                                sigreg_weight=sigreg_weight,
                                spherical_weight=spherical_weight,
                                dyn_spherical_weight=dyn_spherical_weight,
                                dyn_cov_weight=dyn_cov_weight,
                                cross_weight=cross_weight,
                            )[0],
                        )(x, y))
                    )
                )
            )
        )
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        last_batch: tuple[mx.array, mx.array] | None = None
        for _ in range(args.grad_accum_steps):
            loss, grads, last_batch = loss_and_grad_chunked(args, model, train_loader, step_loss_and_grad)
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        bad_grad_keys = first_nonfinite_keys(grads) if (args.jepa_grad_scrub_nonfinite or args.jepa_log_nonfinite) else []
        if bad_grad_keys and args.jepa_log_nonfinite:
            log(f"nonfinite_grads step:{step + 1} keys:{bad_grad_keys}")
        if bad_grad_keys and args.jepa_grad_scrub_nonfinite:
            grads = zero_nonfinite_tree(grads)
        grads = base.clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        model.clear_turbo_cache()
        mx.synchronize()
        bad_param_keys = first_nonfinite_keys(model.parameters()) if args.jepa_log_nonfinite else []
        if bad_param_keys:
            log(f"nonfinite_params step:{step + 1} keys:{bad_param_keys}")

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            extra = ""
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                if aux_scale <= 0.0:
                    operator_codes = base.operator_codes_mx_for_numpy_batch(model, np.asarray(last_batch[0], dtype=np.int32))
                    ce_metric = compiled_ce_loss(*last_batch, operator_codes)
                    mx.eval(ce_metric)
                    extra = (
                        f" ce:{float(ce_metric.item()):.4f}"
                        f" jepa_pred:0.0000 jepa_sigreg:0.0000"
                        f" jepa_spherical:0.0000 jepa_dyn_spherical:0.0000"
                        f" jepa_dyn_cov:0.0000"
                        f" jepa_cross:0.0000"
                    )
                else:
                    operator_codes = base.operator_codes_mx_for_numpy_batch(model, np.asarray(last_batch[0], dtype=np.int32))
                    final_hidden_metric, semantic_hidden_metric = model.forward_with_jepa_hidden(
                        last_batch[0],
                        operator_codes=operator_codes,
                    )
                    ce_metric = model.token_ce_from_hidden(final_hidden_metric, last_batch[1])
                    (
                        pred_metric,
                        sig_metric,
                        spherical_metric,
                        dyn_spherical_metric,
                        dyn_cov_metric,
                        cross_metric,
                    ) = model.jepa_terms_from_hidden(semantic_hidden_metric)
                    debug_metrics = model.jepa_debug_metrics_from_hidden(semantic_hidden_metric)
                    mx.eval(
                        ce_metric,
                        pred_metric,
                        sig_metric,
                        spherical_metric,
                        dyn_spherical_metric,
                        dyn_cov_metric,
                        cross_metric,
                        *debug_metrics,
                    )
                    (
                        src_tgt_cos,
                        pred_tgt_cos,
                        geom_pairwise_abs_cos,
                        dyn_pairwise_abs_cos,
                        z_norm_mean,
                        z_norm_std,
                        geom_dim_std_mean,
                        dyn_dim_std_mean,
                        dyn_cov_debug,
                        geom_dyn_xcov_rms,
                    ) = debug_metrics
                    extra = (
                        f" ce:{float(ce_metric.item()):.4f}"
                        f" jepa_pred:{float(pred_metric.item()):.4f}"
                        f" jepa_sigreg:{float(sig_metric.item()):.4f}"
                        f" jepa_spherical:{float(spherical_metric.item()):.4f}"
                        f" jepa_dyn_spherical:{float(dyn_spherical_metric.item()):.4f}"
                        f" jepa_dyn_cov:{float(dyn_cov_metric.item()):.4f}"
                        f" jepa_cross:{float(cross_metric.item()):.4f}"
                        f" src_tgt_cos:{float(src_tgt_cos.item()):.4f}"
                        f" pred_tgt_cos:{float(pred_tgt_cos.item()):.4f}"
                        f" geom_pairwise_abs_cos:{float(geom_pairwise_abs_cos.item()):.4f}"
                        f" dyn_pairwise_abs_cos:{float(dyn_pairwise_abs_cos.item()):.4f}"
                        f" dyn_cov_debug:{float(dyn_cov_debug.item()):.4f}"
                        f" geom_dyn_xcov_rms:{float(geom_dyn_xcov_rms.item()):.4f}"
                        f" geom_norm_mean:{float(z_norm_mean.item()):.4f}"
                        f" geom_norm_std:{float(z_norm_std.item()):.4f}"
                        f" geom_dim_std_mean:{float(geom_dim_std_mean.item()):.4f}"
                        f" dyn_dim_std_mean:{float(dyn_dim_std_mean.item()):.4f}"
                    )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"jepa_aux_scale:{aux_scale:.3f} jepa_pred_weight:{pred_weight:.4f} "
                f"jepa_sigreg_weight:{sigreg_weight:.4f} jepa_spherical_weight:{spherical_weight:.4f} "
                f"jepa_dyn_spherical_weight:{dyn_spherical_weight:.4f} "
                f"jepa_dyn_cov_weight:{dyn_cov_weight:.4f} "
                f"jepa_cross_weight:{cross_weight:.4f} "
                f"turbo_qat_scale:{qat_scale:.3f} "
                f"turbo_qat_lambda:{qat_lambda:.6f}{extra}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    export_ready_val_loss, export_ready_val_bpb = base.eval_val(
        args,
        model,
        compiled_ce_loss,
        compiled_forward_logits,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    log(f"final_raw_export_ready val_loss:{export_ready_val_loss:.4f} val_bpb:{export_ready_val_bpb:.4f}")
    log(f"final_raw_export_ready_exact val_loss:{export_ready_val_loss:.8f} val_bpb:{export_ready_val_bpb:.8f}")
    flat_state = exportable_flat_state(model)
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x "
        f"{base.format_quant_stats(quant_stats)})"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = base.dequantize_state_dict(pickle.loads(zlib.decompress(quant_blob_disk)))
    final_eval_model = make_export_eval_model(args, sp)
    final_eval_model.clear_turbo_cache()
    final_eval_model.update(tree_unflatten(list(quant_flat.items())))
    final_eval_model.set_turbo_qat(False, 0.0)
    final_eval_model.clear_turbo_cache()
    if uses_logic:
        final_ce = lambda x, y, operator_codes=None: final_eval_model.loss(x, y, operator_codes)
        final_logits = lambda x, operator_codes=None: final_eval_model.forward_logits(x, operator_codes)
    else:
        final_ce = lambda x, y, operator_codes=None: final_eval_model.loss(x, y)
        final_logits = lambda x, operator_codes=None: final_eval_model.forward_logits(x)
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        final_eval_model,
        final_ce,
        final_logits,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
