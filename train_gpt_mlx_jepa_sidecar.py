#!/usr/bin/env python3
"""
GPT with a causal structural sidecar.

The backbone is trained by cross-entropy only. A small recurrent sidecar reads a
stop-gradient view of a middle-layer hidden state, learns a predictable
structural state with JEPA-style losses, and conditions the decoder through a
separate CE-trained readout path. This keeps the prediction objective off the
shared backbone weights.
"""
from __future__ import annotations

import inspect
import math
import os
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_unflatten

import train_gpt_mlx as base
import train_gpt_mlx_jepa_aux as auxlib


class Hyperparameters(base.Hyperparameters):
    sidecar_chunk_size: int = int(os.environ.get("SIDECAR_CHUNK_SIZE", 8))
    sidecar_tap_layer: int = int(os.environ.get("SIDECAR_TAP_LAYER", -1))
    sidecar_state_dim: int = int(os.environ.get("SIDECAR_STATE_DIM", 64))
    sidecar_pred_hidden: int = int(os.environ.get("SIDECAR_PRED_HIDDEN", 64))
    sidecar_pred_offset: int = int(os.environ.get("SIDECAR_PRED_OFFSET", 1))
    sidecar_pred_weight: float = float(os.environ.get("SIDECAR_PRED_WEIGHT", 0.05))
    sidecar_sigreg_weight: float = float(os.environ.get("SIDECAR_SIGREG_WEIGHT", 0.01))
    sidecar_spherical_weight: float = float(os.environ.get("SIDECAR_SPHERICAL_WEIGHT", 0.01))
    sidecar_aux_start_frac: float = float(os.environ.get("SIDECAR_AUX_START_FRAC", 0.0))
    sidecar_aux_ramp_frac: float = float(os.environ.get("SIDECAR_AUX_RAMP_FRAC", 0.0))
    sidecar_summary_mode: str = os.environ.get("SIDECAR_SUMMARY_MODE", "query")
    sidecar_pred_target_mode: str = os.environ.get("SIDECAR_PRED_TARGET_MODE", "delta")
    sidecar_read_init_std: float = float(os.environ.get("SIDECAR_READ_INIT_STD", "1e-3"))
    sidecar_pred_init_std: float = float(os.environ.get("SIDECAR_PRED_INIT_STD", "1e-4"))
    sidecar_grad_scrub_nonfinite: bool = bool(int(os.environ.get("SIDECAR_GRAD_SCRUB_NONFINITE", "1")))
    sidecar_log_nonfinite: bool = bool(int(os.environ.get("SIDECAR_LOG_NONFINITE", "1")))
    sidecar_sigreg_knots: int = int(os.environ.get("SIDECAR_SIGREG_KNOTS", 17))
    sidecar_sigreg_num_proj: int = int(os.environ.get("SIDECAR_SIGREG_NUM_PROJ", 256))
    sidecar_sigreg_seed: int = int(os.environ.get("SIDECAR_SIGREG_SEED", 17))
    sidecar_sigreg_resample_proj: bool = bool(int(os.environ.get("SIDECAR_SIGREG_RESAMPLE_PROJ", "1")))
    sidecar_sigreg_mode: str = os.environ.get("SIDECAR_SIGREG_MODE", "full")
    sidecar_weak_sigreg_dim: int = int(os.environ.get("SIDECAR_WEAK_SIGREG_DIM", 32))
    sidecar_read_rmsnorm: bool = bool(int(os.environ.get("SIDECAR_READ_RMSNORM", "1")))
    sidecar_polarity_write: bool = bool(int(os.environ.get("SIDECAR_POLARITY_WRITE", "0")))
    sidecar_polarity_pool: str = os.environ.get("SIDECAR_POLARITY_POOL", "max")
    sidecar_reset_on_bos: bool = bool(int(os.environ.get("SIDECAR_RESET_ON_BOS", "0")))
    sidecar_reset_token_id: int = int(os.environ.get("SIDECAR_RESET_TOKEN_ID", "1"))
    sidecar_eval_persistent: bool = bool(int(os.environ.get("SIDECAR_EVAL_PERSISTENT", "0")))
    sidecar_eval_persist_group_seqs: int = int(os.environ.get("SIDECAR_EVAL_PERSIST_GROUP_SEQS", "1"))
    curriculum_apply_jepa_phase_gating: bool = bool(int(os.environ.get("CURRICULUM_APPLY_JEPA_PHASE_GATING", "1")))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class GatedLinearSidecarCell(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.input_proj = base.CastedLinear(state_dim, state_dim)
        self.gate_proj = base.CastedLinear(state_dim, state_dim)
        self.input_proj.weight = (
            mx.random.normal(self.input_proj.weight.shape, dtype=mx.float32) * (state_dim ** -0.5)
        ).astype(mx.float32)
        self.gate_proj.weight = (
            mx.random.normal(self.gate_proj.weight.shape, dtype=mx.float32) * (state_dim ** -0.5)
        ).astype(mx.float32)

    def __call__(self, x_t: mx.array, h_t: mx.array) -> mx.array:
        new_input = self.input_proj(x_t).astype(mx.float32)
        gate = mx.sigmoid(self.gate_proj(x_t).astype(mx.float32))
        prev = h_t.astype(mx.float32)
        out = (1.0 - gate) * prev + gate * new_input
        return out.astype(base.COMPUTE_DTYPE)


class GPTJEPASidecar(base.GPT):
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
        early_exit_layer_index: int = -1,
        early_exit_horizons: tuple[int, ...] = (),
        early_exit_aux_weight: float = 0.0,
        early_exit_head_init_std: float = 0.005,
        early_exit_cascaded_enabled: bool = False,
        early_exit_condition_init_std: float = 0.001,
        early_exit_branch_draft_enabled: bool = False,
        early_exit_branch_conf_threshold: float = 0.70,
        early_exit_branch_max_draft_tokens: int = 1,
        sidecar_chunk_size: int,
        sidecar_tap_layer: int,
        sidecar_state_dim: int,
        sidecar_pred_hidden: int,
        sidecar_pred_offset: int,
        sidecar_pred_weight: float,
        sidecar_sigreg_weight: float,
        sidecar_spherical_weight: float,
        sidecar_summary_mode: str,
        sidecar_pred_target_mode: str,
        sidecar_read_init_std: float,
        sidecar_pred_init_std: float,
        sidecar_sigreg_knots: int,
        sidecar_sigreg_num_proj: int,
        sidecar_sigreg_seed: int,
        sidecar_sigreg_resample_proj: bool,
        sidecar_sigreg_mode: str,
        sidecar_weak_sigreg_dim: int,
        sidecar_read_rmsnorm: bool,
        sidecar_polarity_write: bool,
        sidecar_polarity_pool: str,
        sidecar_reset_on_bos: bool,
        sidecar_reset_token_id: int,
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
            early_exit_layer_index=early_exit_layer_index,
            early_exit_horizons=early_exit_horizons,
            early_exit_aux_weight=early_exit_aux_weight,
            early_exit_head_init_std=early_exit_head_init_std,
            early_exit_cascaded_enabled=early_exit_cascaded_enabled,
            early_exit_condition_init_std=early_exit_condition_init_std,
            early_exit_branch_draft_enabled=early_exit_branch_draft_enabled,
            early_exit_branch_conf_threshold=early_exit_branch_conf_threshold,
            early_exit_branch_max_draft_tokens=early_exit_branch_max_draft_tokens,
        )
        if self.num_registers > 0:
            raise ValueError("JEPA sidecar prototype currently requires NUM_REGISTERS=0")
        if sidecar_chunk_size <= 0:
            raise ValueError(f"SIDECAR_CHUNK_SIZE must be > 0, got {sidecar_chunk_size}")
        if sidecar_pred_offset <= 0:
            raise ValueError(f"SIDECAR_PRED_OFFSET must be > 0, got {sidecar_pred_offset}")
        if sidecar_summary_mode not in {"last", "mean", "mean_last", "query"}:
            raise ValueError(f"Unsupported SIDECAR_SUMMARY_MODE={sidecar_summary_mode!r}")
        if sidecar_pred_target_mode not in {"next", "delta"}:
            raise ValueError(f"Unsupported SIDECAR_PRED_TARGET_MODE={sidecar_pred_target_mode!r}")
        if sidecar_sigreg_mode not in {"full", "weak", "none"}:
            raise ValueError(f"Unsupported SIDECAR_SIGREG_MODE={sidecar_sigreg_mode!r}")
        if sidecar_polarity_pool not in {"max", "mean", "last"}:
            raise ValueError(f"Unsupported SIDECAR_POLARITY_POOL={sidecar_polarity_pool!r}")
        self.sidecar_chunk_size = sidecar_chunk_size
        self.sidecar_pred_offset = sidecar_pred_offset
        self.sidecar_pred_weight = sidecar_pred_weight
        self.sidecar_sigreg_weight = sidecar_sigreg_weight
        self.sidecar_spherical_weight = sidecar_spherical_weight
        self.sidecar_summary_mode = sidecar_summary_mode
        self.sidecar_pred_target_mode = sidecar_pred_target_mode
        self.sidecar_tap_layer_index = self._resolve_tap_layer(sidecar_tap_layer)
        self.sidecar_state_dim = sidecar_state_dim
        self.sidecar_loss_stride = sidecar_chunk_size
        self.sidecar_sigreg_mode = sidecar_sigreg_mode
        self.sidecar_read_rmsnorm = sidecar_read_rmsnorm
        self.sidecar_polarity_write = sidecar_polarity_write
        self.sidecar_polarity_pool = sidecar_polarity_pool
        self.sidecar_reset_on_bos = sidecar_reset_on_bos
        self.sidecar_reset_token_id = sidecar_reset_token_id
        # JEPA-trained sidecar path.
        self.sidecar_in_proj = base.CastedLinear(dim, sidecar_state_dim)
        self.sidecar_cell = GatedLinearSidecarCell(sidecar_state_dim)
        self.sidecar_pred = base.CastedLinear(sidecar_state_dim, sidecar_state_dim)
        self.sidecar_pred.weight = (
            mx.random.normal(self.sidecar_pred.weight.shape, dtype=mx.float32) * sidecar_pred_init_std
        ).astype(mx.float32)
        if sidecar_sigreg_mode == "full":
            self.sidecar_sigreg = auxlib.SIGReg(
                input_dim=sidecar_state_dim,
                knots=sidecar_sigreg_knots,
                num_proj=sidecar_sigreg_num_proj,
                seed=sidecar_sigreg_seed,
                resample_proj=sidecar_sigreg_resample_proj,
            )
        elif sidecar_sigreg_mode == "weak":
            self.sidecar_sigreg = auxlib.WeakSIGReg(
                input_dim=sidecar_state_dim,
                sketch_dim=sidecar_weak_sigreg_dim,
                seed=sidecar_sigreg_seed,
                resample_proj=sidecar_sigreg_resample_proj,
            )
        else:
            self.sidecar_sigreg = None
        # CE-trained readout from a stop-grad sidecar state.
        self.sidecar_read_proj = base.CastedLinear(sidecar_state_dim, dim)
        self.sidecar_read_proj.weight = (
            mx.random.normal(self.sidecar_read_proj.weight.shape, dtype=mx.float32) * sidecar_read_init_std
        ).astype(mx.float32)
        self.sidecar_read_scale = mx.full((self.num_decoder_layers, dim), 0.05, dtype=mx.float32)
        self.sidecar_polarity_write_scale = mx.zeros((sidecar_state_dim,), dtype=mx.float32)

    def _resolve_tap_layer(self, requested_layer: int) -> int:
        if requested_layer >= 0:
            if requested_layer >= self.num_layers:
                raise ValueError(f"SIDECAR_TAP_LAYER={requested_layer} out of range for num_layers={self.num_layers}")
            return requested_layer
        return max(self.num_encoder_layers - 1, 0)

    def token_sidecar_inputs(
        self,
        hidden: mx.array,
        polarity_scores: mx.array | None = None,
    ) -> mx.array:
        side_in = self.sidecar_in_proj(mx.stop_gradient(hidden)).astype(base.COMPUTE_DTYPE)
        if self.sidecar_polarity_write and polarity_scores is not None:
            side_in = side_in + (
                mx.expand_dims(self.strip_registers(polarity_scores).astype(base.COMPUTE_DTYPE), axis=-1)
                * mx.tanh(self.sidecar_polarity_write_scale).astype(base.COMPUTE_DTYPE)[None, None, :]
            )
        return side_in

    def _normalize_initial_state(
        self,
        initial_state: mx.array | None,
        batch: int,
    ) -> mx.array:
        if initial_state is None:
            return mx.zeros((batch, self.sidecar_state_dim), dtype=base.COMPUTE_DTYPE)
        state = initial_state.astype(base.COMPUTE_DTYPE)
        if state.ndim == 1:
            state = mx.broadcast_to(state[None, :], (batch, state.shape[0]))
        if int(state.shape[0]) != batch or int(state.shape[1]) != self.sidecar_state_dim:
            raise ValueError(
                f"initial sidecar state shape {tuple(state.shape)} does not match "
                f"(batch={batch}, state_dim={self.sidecar_state_dim})"
            )
        return state

    def _sample_prediction_states(self, side_states: mx.array) -> mx.array:
        if side_states.shape[1] <= 0:
            return side_states
        stride = max(int(self.sidecar_loss_stride), 1)
        start = stride - 1
        return side_states[:, start::stride, :]

    def _sidecar_reset_mask(self, input_ids: mx.array) -> mx.array | None:
        if not self.sidecar_reset_on_bos:
            return None
        return input_ids == self.sidecar_reset_token_id

    def _tangent_displacement(self, source: mx.array, other: mx.array) -> mx.array:
        source32 = source.astype(mx.float32)
        other32 = other.astype(mx.float32)
        radial = mx.sum(source32 * other32, axis=-1, keepdims=True) * source32
        tangent = other32 - radial
        return auxlib.l2_normalize(tangent.astype(base.COMPUTE_DTYPE))

    def sidecar_states_from_hidden(
        self,
        hidden: mx.array,
        polarity_scores: mx.array | None = None,
        initial_state: mx.array | None = None,
        reset_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        side_in = self.token_sidecar_inputs(hidden, polarity_scores=polarity_scores)
        batch = int(side_in.shape[0])
        tokens = int(side_in.shape[1])
        state = self._normalize_initial_state(initial_state, batch)
        zero_state = mx.zeros_like(state)
        states: list[mx.array] = []
        for idx in range(tokens):
            if reset_mask is not None:
                token_reset = mx.expand_dims(reset_mask[:, idx].astype(base.COMPUTE_DTYPE), axis=-1)
                state = token_reset * zero_state + (1.0 - token_reset) * state
            state = self.sidecar_cell(side_in[:, idx, :], state)
            states.append(state)
        side_states = mx.stack(states, axis=1) if states else side_in[:, :0, :]
        return side_in, side_states, state

    def sidecar_token_condition(
        self,
        side_states: mx.array,
        token_len: int,
        initial_state: mx.array | None = None,
        reset_mask: mx.array | None = None,
    ) -> mx.array:
        if side_states.shape[1] <= 0:
            return mx.zeros((side_states.shape[0], token_len, self.tok_emb.weight.shape[1]), dtype=base.COMPUTE_DTYPE)
        init = self._normalize_initial_state(initial_state, int(side_states.shape[0]))
        prev_state = mx.expand_dims(init, axis=1)
        # True causal token-level readout: token t is conditioned by state after token t-1 only.
        side_read_input = mx.concatenate([prev_state, side_states[:, :-1, :]], axis=1)
        if reset_mask is not None:
            token_reset = mx.expand_dims(reset_mask[:, :token_len].astype(base.COMPUTE_DTYPE), axis=-1)
            side_read_input = (1.0 - token_reset) * side_read_input
        side_read_input = mx.stop_gradient(side_read_input)
        if self.sidecar_read_rmsnorm:
            side_read_input = base.rms_norm(side_read_input).astype(base.COMPUTE_DTYPE)
        side_read = self.sidecar_read_proj(side_read_input).astype(base.COMPUTE_DTYPE)
        return side_read[:, :token_len, :]

    def forward_hidden_with_aux(
        self,
        input_ids: mx.array,
        capture_layers: tuple[int, ...] = (),
        operator_codes: mx.array | None = None,
        initial_sidecar_state: mx.array | None = None,
    ) -> tuple[mx.array, dict[int, mx.array], dict[str, mx.array | None]]:
        x = self.embed_inputs(input_ids)
        x0 = x
        attn_mask = self.attention_mask(x.shape[1])
        if operator_codes is None and (self.logic_sidecar is not None or self.polarity_detector is not None or self.sidecar_polarity_write):
            operator_codes = self.operator_codes_for_input(input_ids)
        skips: list[mx.array] = []
        captured: dict[int, mx.array] = {}
        layer_idx = 0
        seed_polarity_scores = self.seed_polarity_scores(operator_codes)
        polarity_scores = seed_polarity_scores if (self.logic_operator_mode == "not_only" and self.polarity_detector is None) else None
        detector_logits: mx.array | None = None
        detector_scores: mx.array | None = None
        tap_hidden: mx.array | None = None
        sidecar_reset_mask = self._sidecar_reset_mask(input_ids)

        for i in range(self.num_encoder_layers):
            x = self.block_for_step(i)(x, x0, attn_mask=attn_mask)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            if not self.sidecar_polarity_write:
                x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            if layer_idx == self.sidecar_tap_layer_index:
                tap_hidden = x
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            skips.append(x)
            layer_idx += 1

        if tap_hidden is None:
            tap_hidden = x
        side_source_scores = polarity_scores if polarity_scores is not None else seed_polarity_scores
        side_inputs, side_states, final_sidecar_state = self.sidecar_states_from_hidden(
            tap_hidden,
            polarity_scores=side_source_scores,
            initial_state=initial_sidecar_state,
            reset_mask=sidecar_reset_mask,
        )
        side_tokens = self.sidecar_token_condition(
            side_states,
            int(input_ids.shape[1]),
            initial_state=initial_sidecar_state,
            reset_mask=sidecar_reset_mask,
        )

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = x + self.sidecar_read_scale[i].astype(x.dtype)[None, None, :] * side_tokens.astype(x.dtype)
            x = self.block_for_step(self.num_encoder_layers + i)(x, x0, attn_mask=attn_mask)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            if not self.sidecar_polarity_write:
                x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            layer_idx += 1

        return self.final_norm(self.strip_registers(x)), captured, {
            "tap_hidden": tap_hidden,
            "sidecar_summary": side_inputs,
            "sidecar_states": side_states,
            "final_sidecar_state": final_sidecar_state,
            "sidecar_tokens": side_tokens,
            "sidecar_chunk_polarity": side_source_scores,
            "operator_codes": operator_codes,
            "seed_polarity_scores": seed_polarity_scores,
            "sidecar_reset_mask": sidecar_reset_mask,
            "polarity_detector_logits": detector_logits,
            "polarity_detector_scores": detector_scores,
            "polarity_scores": polarity_scores,
        }

    def forward_with_sidecar_hidden_aux(
        self,
        input_ids: mx.array,
        operator_codes: mx.array | None = None,
        initial_sidecar_state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array | None]]:
        final_hidden, _captured, aux = self.forward_hidden_with_aux(
            input_ids,
            capture_layers=(self.sidecar_tap_layer_index,),
            operator_codes=operator_codes,
            initial_sidecar_state=initial_sidecar_state,
        )
        assert aux["tap_hidden"] is not None
        return final_hidden, aux["tap_hidden"], aux

    def forward_logits_with_sidecar_state(
        self,
        input_ids: mx.array,
        operator_codes: mx.array | None = None,
        initial_sidecar_state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array | None]]:
        final_hidden, _tap_hidden, aux = self.forward_with_sidecar_hidden_aux(
            input_ids,
            operator_codes=operator_codes,
            initial_sidecar_state=initial_sidecar_state,
        )
        logits_proj = (
            final_hidden @ self.tok_emb.weight.astype(final_hidden.dtype).T
            if self.tie_embeddings
            else self.lm_head(final_hidden)
        )
        logits = self.softcap(logits_proj)
        assert aux["final_sidecar_state"] is not None
        return logits, aux["final_sidecar_state"], aux

    def sidecar_terms_from_states(
        self,
        side_states: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        pred_states = self._sample_prediction_states(side_states)
        if pred_states.shape[1] <= self.sidecar_pred_offset:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, zero
        u = auxlib.l2_normalize(pred_states)
        source = u[:, :-self.sidecar_pred_offset, :]
        target_state = mx.stop_gradient(u[:, self.sidecar_pred_offset :, :])
        pred_raw = self.sidecar_pred(source).astype(base.COMPUTE_DTYPE)
        if self.sidecar_pred_target_mode == "delta":
            pred = self._tangent_displacement(source, pred_raw)
            pred_target = self._tangent_displacement(source, target_state)
        else:
            pred = auxlib.l2_normalize(pred_raw)
            pred_target = target_state
        pred_loss = mx.mean(
            (1.0 - mx.sum(pred.astype(mx.float32) * pred_target.astype(mx.float32), axis=-1)).astype(mx.float32)
        )
        if self.sidecar_sigreg is None:
            sigreg_loss = mx.array(0.0, dtype=mx.float32)
        else:
            sigreg_loss = self.sidecar_sigreg(side_states.reshape(1, -1, side_states.shape[-1]))
        spherical_loss = auxlib.spherical_uniformity(auxlib.l2_normalize(side_states))
        return pred_loss, sigreg_loss, spherical_loss

    def sidecar_debug_metrics_from_states(self, side_states: mx.array) -> tuple[mx.array, ...]:
        pred_states = self._sample_prediction_states(side_states)
        if pred_states.shape[1] <= self.sidecar_pred_offset:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, zero, zero, zero
        u = auxlib.l2_normalize(pred_states)
        source = u[:, :-self.sidecar_pred_offset, :]
        target_state = mx.stop_gradient(u[:, self.sidecar_pred_offset :, :])
        pred_raw = self.sidecar_pred(source).astype(base.COMPUTE_DTYPE)
        if self.sidecar_pred_target_mode == "delta":
            pred = self._tangent_displacement(source, pred_raw)
            pred_target = self._tangent_displacement(source, target_state)
        else:
            pred = auxlib.l2_normalize(pred_raw)
            pred_target = target_state
        flat = side_states.reshape(-1, side_states.shape[-1]).astype(mx.float32)
        norms = mx.sqrt(mx.sum(mx.square(flat), axis=-1) + 1e-6)
        centered = flat - mx.mean(flat, axis=0, keepdims=True)
        dim_std = mx.sqrt(mx.mean(mx.square(centered), axis=0) + 1e-6)
        return (
            mx.mean(mx.sum(source.astype(mx.float32) * target_state.astype(mx.float32), axis=-1)),
            mx.mean(mx.sum(pred.astype(mx.float32) * pred_target.astype(mx.float32), axis=-1)),
            auxlib.spherical_uniformity(auxlib.l2_normalize(side_states)),
            mx.mean(norms),
            mx.mean(dim_std),
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
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        pred_weight = self.sidecar_pred_weight if pred_weight is None else pred_weight
        sigreg_weight = self.sidecar_sigreg_weight if sigreg_weight is None else sigreg_weight
        spherical_weight = self.sidecar_spherical_weight if spherical_weight is None else spherical_weight
        effective_early_exit_weight = float(self._early_exit_aux_weight)
        capture_layers = (
            (self._early_exit_layer_index,)
            if bool(self.early_exit_heads) and effective_early_exit_weight > 0.0
            else ()
        )
        if capture_layers:
            final_hidden, captured, aux = self.forward_hidden_with_aux(
                input_ids,
                capture_layers=capture_layers,
                operator_codes=operator_codes,
            )
        else:
            final_hidden, _tap_hidden, aux = self.forward_with_sidecar_hidden_aux(
                input_ids,
                operator_codes=operator_codes,
            )
            captured = {}
        ce_loss = self.token_ce_from_hidden(final_hidden, target_ids)
        zero = mx.array(0.0, dtype=mx.float32)
        if aux_scale <= 0.0:
            pred_loss, sigreg_loss, spherical_loss = zero, zero, zero
        else:
            side_states = aux["sidecar_states"]
            assert side_states is not None
            pred_loss, sigreg_loss, spherical_loss = self.sidecar_terms_from_states(side_states)
        polarity_seed_loss, polarity_sparse_loss, polarity_smooth_loss = self.polarity_loss_terms_from_aux(aux)
        early_exit_loss = zero
        if bool(self.early_exit_heads) and effective_early_exit_weight > 0.0:
            early_exit_hidden = captured.get(self._early_exit_layer_index)
            if early_exit_hidden is not None:
                early_exit_loss = self.early_exit_aux_loss(early_exit_hidden, target_ids)
        total = ce_loss + mx.array(aux_scale, dtype=mx.float32) * (
            pred_weight * pred_loss + sigreg_weight * sigreg_loss + spherical_weight * spherical_loss
        ) + effective_early_exit_weight * early_exit_loss + (
            self.polarity_seed_weight * polarity_seed_loss
            + self.polarity_sparse_weight * polarity_sparse_loss
            + self.polarity_smooth_weight * polarity_smooth_loss
        )
        return total, ce_loss, pred_loss, sigreg_loss, spherical_loss, early_exit_loss


def exportable_flat_state(model: GPTJEPASidecar) -> dict[str, mx.array]:
    return base.exportable_flat_state(model)


def eval_val_sidecar_persistent(
    args: Hyperparameters,
    model: GPTJEPASidecar,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    if args.eval_stride > 0:
        raise ValueError("SIDECAR_EVAL_PERSISTENT currently supports only non-overlapping eval (EVAL_STRIDE=0)")
    eval_seq_len = args.effective_eval_seq_len
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    group_seqs = max(int(args.sidecar_eval_persist_group_seqs), 1)
    total_groups = max((total_seqs + group_seqs - 1) // group_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    carry_state: mx.array | None = None
    for group_idx, seq_start in enumerate(range(0, total_seqs, group_seqs), start=1):
        seq_end = min(seq_start + group_seqs, total_seqs)
        raw_start = seq_start * eval_seq_len
        raw_end = seq_end * eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1][None, :]
        y_np = chunk[1:][None, :]
        operator_codes = base.operator_codes_mx_for_numpy_batch(model, x_np)
        logits, carry_state, _aux = model.forward_logits_with_sidecar_state(
            mx.array(x_np, dtype=mx.int32),
            operator_codes=operator_codes,
            initial_sidecar_state=carry_state,
        )
        nll = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            mx.array(y_np, dtype=mx.int32),
            reduction="none",
        ).astype(mx.float32)
        nll_sum = mx.sum(nll)
        mx.eval(nll_sum, carry_state)
        total_loss_sum += float(nll_sum.item())
        chunk_token_count = float(y_np.size)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_groups > 1 and (
            group_idx == 1 or group_idx == total_groups or group_idx % 25 == 0
        ):
            log_fn(f"persistent_val_progress:{group_idx}/{total_groups}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def make_sidecar_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTJEPASidecar:
    compatible_kwargs = _compatible_sidecar_base_kwargs(args, sp)
    return GPTJEPASidecar(
        **compatible_kwargs,
        sidecar_chunk_size=args.sidecar_chunk_size,
        sidecar_tap_layer=args.sidecar_tap_layer,
        sidecar_state_dim=args.sidecar_state_dim,
        sidecar_pred_hidden=args.sidecar_pred_hidden,
        sidecar_pred_offset=args.sidecar_pred_offset,
        sidecar_pred_weight=args.sidecar_pred_weight,
        sidecar_sigreg_weight=args.sidecar_sigreg_weight,
        sidecar_spherical_weight=args.sidecar_spherical_weight,
        sidecar_summary_mode=args.sidecar_summary_mode,
        sidecar_pred_target_mode=args.sidecar_pred_target_mode,
        sidecar_read_init_std=args.sidecar_read_init_std,
        sidecar_pred_init_std=args.sidecar_pred_init_std,
        sidecar_sigreg_knots=args.sidecar_sigreg_knots,
        sidecar_sigreg_num_proj=args.sidecar_sigreg_num_proj,
        sidecar_sigreg_seed=args.sidecar_sigreg_seed,
        sidecar_sigreg_resample_proj=args.sidecar_sigreg_resample_proj,
        sidecar_sigreg_mode=args.sidecar_sigreg_mode,
        sidecar_weak_sigreg_dim=args.sidecar_weak_sigreg_dim,
        sidecar_read_rmsnorm=args.sidecar_read_rmsnorm,
        sidecar_polarity_write=args.sidecar_polarity_write,
        sidecar_polarity_pool=args.sidecar_polarity_pool,
        sidecar_reset_on_bos=args.sidecar_reset_on_bos,
        sidecar_reset_token_id=args.sidecar_reset_token_id,
    )


def _compatible_sidecar_base_kwargs(
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None = None,
) -> dict[str, object]:
    """Filter base GPT kwargs to the subset accepted by the older sidecar ctor."""
    kwargs = base.gpt_kwargs_from_args(args, sp)
    kwargs.update(
        {
            "early_exit_layer_index": args.early_exit_layer_index,
            "early_exit_horizons": base.parse_horizons(args.early_exit_horizons),
            "early_exit_aux_weight": args.early_exit_aux_weight,
            "early_exit_head_init_std": args.early_exit_head_init_std,
            "early_exit_cascaded_enabled": args.early_exit_cascaded_enabled,
            "early_exit_condition_init_std": args.early_exit_condition_init_std,
            "early_exit_branch_draft_enabled": args.early_exit_branch_draft_enabled,
            "early_exit_branch_conf_threshold": args.early_exit_branch_conf_threshold,
            "early_exit_branch_max_draft_tokens": args.early_exit_branch_max_draft_tokens,
        }
    )
    accepted = set(inspect.signature(GPTJEPASidecar.__init__).parameters)
    accepted.discard("self")
    return {key: value for key, value in kwargs.items() if key in accepted}


def sidecar_aux_scale_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    frac = base.training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    if frac < args.sidecar_aux_start_frac:
        return 0.0
    if args.sidecar_aux_ramp_frac <= 0.0:
        return 1.0
    return min((frac - args.sidecar_aux_start_frac) / args.sidecar_aux_ramp_frac, 1.0)


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
    train_loader = base.build_train_loader(args, log_fn=log, dataset_name=dataset_name)
    model = make_sidecar_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: GPTJEPASidecar | None = None
    compiled_quant_ce = None
    compiled_quant_forward_logits = None

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    uses_logic = (
        model.logic_sidecar is not None
        or model.polarity_detector is not None
        or model.sidecar_polarity_write
    )
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
    export_params = sum(int(np.prod(v.shape)) for _, v in tree_flatten(exportable_flat_state(model)))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files}"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(
        f"model_params_trainable:{trainable_params} model_params_export:{export_params} "
        f"vocab_size:{args.vocab_size} layers:{args.num_layers} layer_templates:{args.num_layer_templates} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"sidecar:recurrence:token loss_stride:{model.sidecar_loss_stride} tap_layer:{model.sidecar_tap_layer_index} "
        f"state_dim:{args.sidecar_state_dim} pred_offset:{args.sidecar_pred_offset} "
        f"pred_weight:{args.sidecar_pred_weight} sigreg_weight:{args.sidecar_sigreg_weight} "
        f"spherical_weight:{args.sidecar_spherical_weight} sigreg_mode:{args.sidecar_sigreg_mode} "
        f"weak_sigreg_dim:{args.sidecar_weak_sigreg_dim} read_rmsnorm:{int(args.sidecar_read_rmsnorm)} "
        f"summary_mode:{args.sidecar_summary_mode}(legacy) pred_target_mode:{args.sidecar_pred_target_mode} "
        f"polarity_write:{int(args.sidecar_polarity_write)} polarity_pool:{args.sidecar_polarity_pool} "
        f"reset_on_bos:{int(args.sidecar_reset_on_bos)} reset_token_id:{args.sidecar_reset_token_id} "
        f"aux_start_frac:{args.sidecar_aux_start_frac} "
        f"aux_ramp_frac:{args.sidecar_aux_ramp_frac}"
    )
    log(
        f"logic_layer:{model.logic_layer_index if model.logic_sidecar is not None else 'off'} "
        f"logic_route_to_next:{args.logic_route_to_next_token} logic_mode:{args.logic_operator_mode}"
    )
    log(
        f"polarity_detector:enabled:{int(args.polarity_detector_enabled)} "
        f"layer:{model.polarity_detector_layer_index if model.polarity_detector is not None else 'off'} "
        f"hidden_dim:{args.polarity_detector_hidden_dim} seed_blend:{args.polarity_seed_blend:.3f} "
        f"seed_weight:{args.polarity_seed_weight:.6f} sparse_weight:{args.polarity_sparse_weight:.6f} "
        f"smooth_weight:{args.polarity_smooth_weight:.6f}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(
        f"compute_dtype:{base.COMPUTE_DTYPE} compile:{compiled} compile_mode:{args.mlx_compile} "
        f"quant_format:{args.quant_format} turbo_qat:{args.turbo_qat} "
        f"turbo_block:{args.turbo_block_size} turbo_bits_mse:{args.turbo_mse_bits} turbo_bits_prod:{args.turbo_prod_bits}"
    )
    if args.curriculum_enabled:
        summary = train_loader.summary() if hasattr(train_loader, "summary") else {}
        summary_bits = " ".join(f"{key}:{value}" for key, value in summary.items())
        log(
            f"curriculum:enabled features:{args.curriculum_features_path} "
            f"phase_plan:{args.curriculum_phase_plan_path or 'default'} "
            f"apply_logic_gate:{int(args.curriculum_apply_logic_phase_gating)} "
            f"apply_jepa_gate:{int(args.curriculum_apply_jepa_phase_gating)} "
            f"apply_qat_gate:{int(args.curriculum_apply_qat_phase_gating)}"
            + (f" {summary_bits}" if summary_bits else "")
        )
    if args.sidecar_eval_persistent:
        log(
            f"sidecar_eval:persistent group_seqs:{args.sidecar_eval_persist_group_seqs} "
            f"note:non_overlapping_only"
        )

    def eval_val_for_model(
        eval_model: GPTJEPASidecar,
        eval_ce_loss,
        eval_forward_logits,
        eval_tokens: np.ndarray,
        *,
        log_eval_progress: bool = False,
    ) -> tuple[float, float]:
        if args.sidecar_eval_persistent:
            return eval_val_sidecar_persistent(
                args,
                eval_model,
                eval_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log if log_eval_progress else None,
            )
        return base.eval_val(
            args,
            eval_model,
            eval_ce_loss,
            eval_forward_logits,
            eval_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log if log_eval_progress else None,
        )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, _ = auxlib.loss_and_grad_chunked(args, model, train_loader, compiled_loss_and_grad)
                accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        train_loader = base.build_train_loader(args, log_fn=log, dataset_name=dataset_name)

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
                val_loss, val_bpb = eval_val_for_model(
                    model,
                    compiled_ce_loss,
                    compiled_forward_logits,
                    val_tokens,
                    log_eval_progress=True,
                )
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = make_sidecar_gpt(args, sp)
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
                        compiled_quant_forward_logits = (
                            lambda x, operator_codes=None: compiled_quant_forward_logits_impl(x, operator_codes)
                        )
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
                raw_q_val_loss, raw_q_val_bpb = eval_val_for_model(
                    model,
                    compiled_ce_loss,
                    compiled_forward_logits,
                    quant_eval_tokens,
                )
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = eval_val_for_model(
                    quant_eval_model,
                    compiled_quant_ce,
                    compiled_quant_forward_logits,
                    quant_eval_tokens,
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
        if args.curriculum_enabled and hasattr(train_loader, "begin_step"):
            curriculum_phase = train_loader.begin_step()
        else:
            curriculum_phase = train_loader.current_phase() if args.curriculum_enabled and hasattr(train_loader, "current_phase") else None
        curriculum_logic_enabled = (
            curriculum_phase.enable_logic_sidecar
            if curriculum_phase is not None and args.curriculum_apply_logic_phase_gating
            else True
        )
        curriculum_jepa_enabled = (
            curriculum_phase.enable_jepa
            if curriculum_phase is not None and args.curriculum_apply_jepa_phase_gating
            else True
        )
        qat_scale = base.turbo_qat_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        if args.curriculum_apply_qat_phase_gating and curriculum_phase is not None and not curriculum_phase.enable_qat:
            qat_scale = 0.0
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        aux_scale = sidecar_aux_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        if args.curriculum_apply_jepa_phase_gating and curriculum_phase is not None and not curriculum_phase.enable_jepa:
            aux_scale = 0.0
        pred_weight = args.sidecar_pred_weight
        sigreg_weight = args.sidecar_sigreg_weight
        spherical_weight = args.sidecar_spherical_weight
        model.set_turbo_qat(qat_active, qat_scale)
        dynamic_aux_weights = (
            pred_weight != args.sidecar_pred_weight
            or sigreg_weight != args.sidecar_sigreg_weight
            or spherical_weight != args.sidecar_spherical_weight
            or aux_scale != 1.0
        )
        step_loss_and_grad = (
            (
                (lambda x, y, operator_codes=None: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        aux_scale=aux_scale,
                        pred_weight=pred_weight,
                        sigreg_weight=sigreg_weight,
                        spherical_weight=spherical_weight,
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
                    else (lambda x, y, operator_codes=None: nn.value_and_grad(
                        model,
                        lambda x_inner, y_inner: model.loss_terms(
                            x_inner,
                            y_inner,
                            aux_scale=aux_scale,
                            pred_weight=pred_weight,
                            sigreg_weight=sigreg_weight,
                            spherical_weight=spherical_weight,
                        )[0],
                    )(x, y))
                )
            )
        )
        step_t0 = time.perf_counter()
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        last_batch: tuple[mx.array, mx.array] | None = None
        for _ in range(args.grad_accum_steps):
            loss, grads, last_batch = auxlib.loss_and_grad_chunked(
                args,
                model,
                train_loader,
                step_loss_and_grad,
                logic_phase_enabled=curriculum_logic_enabled,
            )
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
        if args.curriculum_enabled and hasattr(train_loader, "end_step"):
            train_loader.end_step()

        grads = tree_unflatten(list(accum.items()))
        bad_grad_keys = (
            auxlib.first_nonfinite_keys(grads)
            if (args.sidecar_grad_scrub_nonfinite or args.sidecar_log_nonfinite)
            else []
        )
        if bad_grad_keys and args.sidecar_log_nonfinite:
            log(f"nonfinite_grads step:{step + 1} keys:{bad_grad_keys}")
        if bad_grad_keys and args.sidecar_grad_scrub_nonfinite:
            grads = auxlib.zero_nonfinite_tree(grads)
        grads = base.clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        model.clear_turbo_cache()
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            extra = ""
            curriculum_step_metrics = (
                train_loader.last_step_metrics()
                if args.curriculum_enabled and hasattr(train_loader, "last_step_metrics")
                else {}
            )
            curriculum_metrics_str = ""
            if curriculum_step_metrics:
                metric_order = (
                    "chunks",
                    "unique_chunk_frac",
                    "repeat_bucket_frac",
                    "once_bucket_frac",
                    "repeat_reuse_frac",
                    "mean_difficulty",
                    "mean_operator_density",
                    "mean_compressibility",
                    "mean_learnability",
                    "mean_quality",
                )
                parts: list[str] = []
                for key in metric_order:
                    value = curriculum_step_metrics.get(key)
                    if value is None:
                        continue
                    if isinstance(value, int):
                        parts.append(f"{key}:{value}")
                    else:
                        parts.append(f"{key}:{float(value):.3f}")
                if parts:
                    curriculum_metrics_str = " " + " ".join(parts)
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                if aux_scale <= 0.0:
                    operator_codes = base.operator_codes_mx_for_numpy_batch(
                        model,
                        np.asarray(last_batch[0], dtype=np.int32),
                        enabled=curriculum_logic_enabled,
                    )
                    ce_metric = compiled_ce_loss(*last_batch, operator_codes)
                    mx.eval(ce_metric)
                    extra = " ce:{:.4f} sidecar_pred:0.0000 sidecar_sigreg:0.0000 sidecar_spherical:0.0000".format(
                        float(ce_metric.item())
                    )
                else:
                    operator_codes = base.operator_codes_mx_for_numpy_batch(
                        model,
                        np.asarray(last_batch[0], dtype=np.int32),
                        enabled=curriculum_logic_enabled,
                    )
                    final_hidden_metric, _tap_hidden_metric, aux_metric = model.forward_with_sidecar_hidden_aux(
                        last_batch[0],
                        operator_codes=operator_codes,
                    )
                    ce_metric = model.token_ce_from_hidden(final_hidden_metric, last_batch[1])
                    side_states_metric = aux_metric["sidecar_states"]
                    assert side_states_metric is not None
                    pred_metric, sig_metric, spherical_metric = model.sidecar_terms_from_states(side_states_metric)
                    pol_seed_metric, pol_sparse_metric, pol_smooth_metric = model.polarity_loss_terms_from_aux(aux_metric)
                    debug_metrics = model.sidecar_debug_metrics_from_states(side_states_metric)
                    mx.eval(
                        ce_metric,
                        pred_metric,
                        sig_metric,
                        spherical_metric,
                        pol_seed_metric,
                        pol_sparse_metric,
                        pol_smooth_metric,
                        *debug_metrics,
                    )
                    src_tgt_cos, pred_tgt_cos, pairwise_abs_cos, state_norm_mean, state_dim_std_mean = debug_metrics
                    extra = (
                        f" ce:{float(ce_metric.item()):.4f}"
                        f" sidecar_pred:{float(pred_metric.item()):.4f}"
                        f" sidecar_sigreg:{float(sig_metric.item()):.4f}"
                        f" sidecar_spherical:{float(spherical_metric.item()):.4f}"
                        f" pol_seed:{float(pol_seed_metric.item()):.4f}"
                        f" pol_sparse:{float(pol_sparse_metric.item()):.4f}"
                        f" pol_smooth:{float(pol_smooth_metric.item()):.4f}"
                        f" src_tgt_cos:{float(src_tgt_cos.item()):.4f}"
                        f" pred_tgt_cos:{float(pred_tgt_cos.item()):.4f}"
                        f" sidecar_pairwise_abs_cos:{float(pairwise_abs_cos.item()):.4f}"
                        f" sidecar_state_norm_mean:{float(state_norm_mean.item()):.4f}"
                        f" sidecar_dim_std_mean:{float(state_dim_std_mean.item()):.4f}"
                    )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"curriculum_phase:{curriculum_phase.name if curriculum_phase is not None else 'off'} "
                f"curriculum_logic:{int(curriculum_logic_enabled)} "
                f"curriculum_jepa:{int(curriculum_jepa_enabled)} "
                f"sidecar_aux_scale:{aux_scale:.3f} sidecar_pred_weight:{pred_weight:.4f} "
                f"sidecar_sigreg_weight:{sigreg_weight:.4f} sidecar_spherical_weight:{spherical_weight:.4f} "
                f"turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f}{extra}"
                f"{curriculum_metrics_str}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    export_ready_val_loss, export_ready_val_bpb = eval_val_for_model(
        model,
        compiled_ce_loss,
        compiled_forward_logits,
        val_tokens,
        log_eval_progress=True,
    )
    log(f"final_raw_export_ready val_loss:{export_ready_val_loss:.4f} val_bpb:{export_ready_val_bpb:.4f}")
    log(f"final_raw_export_ready_exact val_loss:{export_ready_val_loss:.8f} val_bpb:{export_ready_val_bpb:.8f}")
    flat_state = exportable_flat_state(model)
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
    quant_path = out_dir / f"{args.run_id}_int8zlib.pklz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    log(
        f"serialized_model_int8_zlib:{quant_path.stat().st_size} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)} "
        f"payload_ratio:{(sum(v.nbytes for v in flat_state.values()) / max(len(quant_blob), 1)):.2f}x "
        f"{base.format_quant_stats(quant_stats)})"
    )

    eval_model = make_sidecar_gpt(args, sp)
    eval_model.clear_turbo_cache()
    quant_flat = base.dequantize_state_dict(pickle.loads(zlib.decompress(quant_path.read_bytes())))
    eval_model.update(tree_unflatten(list(quant_flat.items())))
    if uses_logic:
        compiled_quant_ce = lambda x, y, operator_codes=None: eval_model.loss(x, y, operator_codes)
        compiled_quant_forward_logits = lambda x, operator_codes=None: eval_model.forward_logits(x, operator_codes)
    else:
        compiled_quant_ce = lambda x, y, operator_codes=None: eval_model.loss(x, y)
        compiled_quant_forward_logits = lambda x, operator_codes=None: eval_model.forward_logits(x)
    q_val_loss, q_val_bpb = eval_val_for_model(
        eval_model,
        compiled_quant_ce,
        compiled_quant_forward_logits,
        val_tokens,
        log_eval_progress=True,
    )
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
