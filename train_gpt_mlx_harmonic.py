#!/usr/bin/env python3
"""
Harmonic patch-to-chord prototype for Parameter Golf.

This trainer keeps exact local token order inside the transformer window, but
adds a detached long-range summary path:

- fixed micro-patches of HARM_PATCH_LEN tokens
- causal chord boundaries from semantic flux over patch summaries
- a bank of previous completed chord summaries
- patch-level reads from that bank back into the decoder
- optional 1-step JEPA over chord states

The design goal is to test whether compressed previous-chord context is useful
without relying on a drifting recurrent carry state.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_unflatten

import train_gpt_mlx as base
import train_gpt_mlx_jepa_aux as auxlib


class Hyperparameters(base.Hyperparameters):
    longctx_num_streams: int = int(os.environ.get("LONGCTX_NUM_STREAMS", "0"))
    harmonic_patch_len: int = int(os.environ.get("HARM_PATCH_LEN", "8"))
    harmonic_tap_layer: int = int(os.environ.get("HARM_TAP_LAYER", "3"))
    harmonic_chord_dim: int = int(os.environ.get("HARM_CHORD_DIM", "64"))
    harmonic_min_patches_per_chord: int = int(os.environ.get("HARM_MIN_PATCHES_PER_CHORD", "1"))
    harmonic_max_patches_per_chord: int = int(os.environ.get("HARM_MAX_PATCHES_PER_CHORD", "8"))
    harmonic_boundary_threshold: float = float(os.environ.get("HARM_BOUNDARY_THRESHOLD", "0.12"))
    harmonic_bank_size: int = int(os.environ.get("HARM_BANK_SIZE", "32"))
    harmonic_read_topk: int = int(os.environ.get("HARM_READ_TOPK", "0"))
    harmonic_enable_read: bool = bool(int(os.environ.get("HARM_ENABLE_READ", "1")))
    harmonic_jepa_weight: float = float(os.environ.get("HARM_JEPA_WEIGHT", "0.0"))
    harmonic_pred_offset: int = int(os.environ.get("HARM_PRED_OFFSET", "1"))
    harmonic_read_init_std: float = float(os.environ.get("HARM_READ_INIT_STD", "1e-3"))
    harmonic_pred_init_std: float = float(os.environ.get("HARM_PRED_INIT_STD", "1e-4"))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class GroupedStreamingTokenLoader:
    """Persistent stream lanes for contiguous local-window training."""

    def __init__(
        self,
        pattern: str,
        *,
        num_streams: int,
        seq_len: int,
        log_fn=None,
        dataset_name: str = "",
    ):
        if num_streams <= 0:
            raise ValueError(f"LONGCTX_NUM_STREAMS must be > 0, got {num_streams}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.stream = base.TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)
        self.num_streams = int(num_streams)
        self.seq_len = int(seq_len)
        self.windows = [self.stream.take(self.seq_len + 1) for _ in range(self.num_streams)]

    def summary(self) -> dict[str, object]:
        return {
            "mode": "grouped_streaming",
            "num_streams": int(self.num_streams),
            "seq_len": int(self.seq_len),
        }

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        if seq_len != self.seq_len:
            raise ValueError(f"loader seq_len={self.seq_len} does not match request seq_len={seq_len}")
        if batch_seqs != self.num_streams:
            raise ValueError(
                f"grouped streaming expects batch_seqs == num_streams, got batch_seqs={batch_seqs} "
                f"num_streams={self.num_streams}"
            )
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for lane_idx in range(self.num_streams):
            window = self.windows[lane_idx]
            x_rows.append(window[:-1])
            y_rows.append(window[1:])
            next_tokens = self.stream.take(self.seq_len)
            self.windows[lane_idx] = np.concatenate((window[-1:], next_tokens), axis=0)
        x = np.stack(x_rows, axis=0).astype(np.int32, copy=False)
        y = np.stack(y_rows, axis=0).astype(np.int32, copy=False)
        return x, y


class GPTHarmonic(base.GPT):
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
        prosody_type_embeddings_enabled: bool = False,
        prosody_type_embedding_init_std: float = 0.002,
        prosody_feature_embeddings_enabled: bool = False,
        prosody_feature_embedding_init_std: float = 0.002,
        prosody_state_adapter_enabled: bool = False,
        prosody_state_dim: int = 64,
        prosody_state_init_std: float = 0.005,
        prosody_state_scale: float = 0.50,
        prosody_state_reset_prior_weight: float = 1.0,
        prosody_aux_layer_index: int = -1,
        prosody_aux_weight: float = 0.0,
        prosody_aux_head_init_std: float = 0.005,
        prosody_aux_token_class_weight: float = 1.0,
        prosody_aux_boundary_weight: float = 1.0,
        prosody_aux_quote_weight: float = 0.25,
        token_prosody_luts: base.TokenProsodyLuts | None = None,
        harmonic_patch_len: int,
        harmonic_tap_layer: int,
        harmonic_chord_dim: int,
        harmonic_min_patches_per_chord: int,
        harmonic_max_patches_per_chord: int,
        harmonic_boundary_threshold: float,
        harmonic_bank_size: int,
        harmonic_read_topk: int,
        harmonic_enable_read: bool,
        harmonic_pred_offset: int,
        harmonic_jepa_weight: float,
        harmonic_read_init_std: float,
        harmonic_pred_init_std: float,
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
            prosody_type_embeddings_enabled=prosody_type_embeddings_enabled,
            prosody_type_embedding_init_std=prosody_type_embedding_init_std,
            prosody_feature_embeddings_enabled=prosody_feature_embeddings_enabled,
            prosody_feature_embedding_init_std=prosody_feature_embedding_init_std,
            prosody_state_adapter_enabled=prosody_state_adapter_enabled,
            prosody_state_dim=prosody_state_dim,
            prosody_state_init_std=prosody_state_init_std,
            prosody_state_scale=prosody_state_scale,
            prosody_state_reset_prior_weight=prosody_state_reset_prior_weight,
            prosody_aux_layer_index=prosody_aux_layer_index,
            prosody_aux_weight=prosody_aux_weight,
            prosody_aux_head_init_std=prosody_aux_head_init_std,
            prosody_aux_token_class_weight=prosody_aux_token_class_weight,
            prosody_aux_boundary_weight=prosody_aux_boundary_weight,
            prosody_aux_quote_weight=prosody_aux_quote_weight,
            token_prosody_luts=token_prosody_luts,
        )
        if self.num_registers > 0:
            raise ValueError("Harmonic prototype currently requires NUM_REGISTERS=0")
        if harmonic_patch_len <= 0:
            raise ValueError(f"HARM_PATCH_LEN must be > 0, got {harmonic_patch_len}")
        if harmonic_chord_dim <= 0:
            raise ValueError(f"HARM_CHORD_DIM must be > 0, got {harmonic_chord_dim}")
        if harmonic_min_patches_per_chord <= 0:
            raise ValueError(
                f"HARM_MIN_PATCHES_PER_CHORD must be > 0, got {harmonic_min_patches_per_chord}"
            )
        if harmonic_max_patches_per_chord < harmonic_min_patches_per_chord:
            raise ValueError(
                "HARM_MAX_PATCHES_PER_CHORD must be >= HARM_MIN_PATCHES_PER_CHORD, got "
                f"{harmonic_max_patches_per_chord} < {harmonic_min_patches_per_chord}"
            )
        if harmonic_pred_offset <= 0:
            raise ValueError(f"HARM_PRED_OFFSET must be > 0, got {harmonic_pred_offset}")
        if harmonic_bank_size < 0:
            raise ValueError(f"HARM_BANK_SIZE must be >= 0, got {harmonic_bank_size}")
        if harmonic_read_topk < 0:
            raise ValueError(f"HARM_READ_TOPK must be >= 0, got {harmonic_read_topk}")
        self.harmonic_patch_len = int(harmonic_patch_len)
        self.harmonic_tap_layer_index = self._resolve_tap_layer(harmonic_tap_layer)
        self.harmonic_chord_dim = int(harmonic_chord_dim)
        self.harmonic_min_patches_per_chord = int(harmonic_min_patches_per_chord)
        self.harmonic_max_patches_per_chord = int(harmonic_max_patches_per_chord)
        self.harmonic_boundary_threshold = float(harmonic_boundary_threshold)
        self.harmonic_bank_size = int(harmonic_bank_size)
        self.harmonic_read_topk = int(harmonic_read_topk)
        self.harmonic_enable_read = bool(harmonic_enable_read)
        self.harmonic_pred_offset = int(harmonic_pred_offset)
        self.harmonic_jepa_weight = float(harmonic_jepa_weight)

        self.harmonic_write_proj = base.CastedLinear(dim, harmonic_chord_dim)
        self.harmonic_write_proj.weight = (
            mx.random.normal(self.harmonic_write_proj.weight.shape, dtype=mx.float32) * (dim ** -0.5)
        ).astype(mx.float32)
        self.harmonic_query_proj = base.CastedLinear(dim, harmonic_chord_dim)
        self.harmonic_query_proj.weight = (
            mx.random.normal(self.harmonic_query_proj.weight.shape, dtype=mx.float32) * (dim ** -0.5)
        ).astype(mx.float32)
        self.harmonic_read_proj = base.CastedLinear(harmonic_chord_dim, dim)
        self.harmonic_read_proj.weight = (
            mx.random.normal(self.harmonic_read_proj.weight.shape, dtype=mx.float32) * harmonic_read_init_std
        ).astype(mx.float32)
        self.harmonic_pred = base.CastedLinear(harmonic_chord_dim, harmonic_chord_dim)
        self.harmonic_pred.weight = (
            mx.random.normal(self.harmonic_pred.weight.shape, dtype=mx.float32) * harmonic_pred_init_std
        ).astype(mx.float32)
        self.harmonic_read_scale = mx.full((self.num_decoder_layers, dim), 0.05, dtype=mx.float32)

    def _resolve_tap_layer(self, requested_layer: int) -> int:
        if requested_layer >= 0:
            if requested_layer >= self.num_layers:
                raise ValueError(f"HARM_TAP_LAYER={requested_layer} out of range for num_layers={self.num_layers}")
            return requested_layer
        return max(self.num_encoder_layers - 1, 0)

    def _patch_bounds(self, token_len: int) -> list[tuple[int, int]]:
        bounds: list[tuple[int, int]] = []
        start = 0
        while start < token_len:
            end = min(start + self.harmonic_patch_len, token_len)
            bounds.append((start, end))
            start = end
        return bounds

    def _patch_pool_hidden(self, hidden: mx.array) -> mx.array:
        batch, token_len, dim = hidden.shape
        if token_len <= 0:
            return hidden[:, :0, :]
        patch_count = (int(token_len) + self.harmonic_patch_len - 1) // self.harmonic_patch_len
        pad = patch_count * self.harmonic_patch_len - int(token_len)
        if pad > 0:
            hidden = mx.concatenate(
                [
                    hidden,
                    mx.zeros((batch, pad, dim), dtype=hidden.dtype),
                ],
                axis=1,
            )
        valid = mx.concatenate(
            [
                mx.ones((int(token_len),), dtype=hidden.dtype),
                mx.zeros((pad,), dtype=hidden.dtype),
            ],
            axis=0,
        )
        reshaped = hidden.reshape(batch, patch_count, self.harmonic_patch_len, dim)
        weights = valid.reshape(1, patch_count, self.harmonic_patch_len, 1)
        sums = mx.sum(reshaped * weights, axis=2)
        counts = mx.maximum(mx.sum(weights, axis=2), 1.0)
        return sums / counts

    def harmonic_patch_write_embeddings(self, hidden: mx.array) -> mx.array:
        patch_hidden = self._patch_pool_hidden(mx.stop_gradient(self.strip_registers(hidden)))
        if patch_hidden.shape[1] <= 0:
            return patch_hidden[:, :, : self.harmonic_chord_dim]
        return self.harmonic_write_proj(base.rms_norm(patch_hidden)).astype(base.COMPUTE_DTYPE)

    def harmonic_patch_query_embeddings(self, hidden: mx.array) -> mx.array:
        patch_hidden = self._patch_pool_hidden(self.strip_registers(hidden))
        if patch_hidden.shape[1] <= 0:
            return patch_hidden[:, :, : self.harmonic_chord_dim]
        return self.harmonic_query_proj(base.rms_norm(patch_hidden)).astype(base.COMPUTE_DTYPE)

    def _flux_from_patch_embeddings_np(self, patches_np: np.ndarray) -> np.ndarray:
        num_patches = int(patches_np.shape[0])
        flux = np.zeros((num_patches,), dtype=np.float32)
        if num_patches <= 1:
            return flux
        cur = patches_np[1:]
        prev = patches_np[:-1]
        denom = np.linalg.norm(cur, axis=-1) * np.linalg.norm(prev, axis=-1)
        denom = np.maximum(denom, 1e-6)
        cosine = np.sum(cur * prev, axis=-1) / denom
        flux[1:] = 1.0 - np.clip(cosine, -1.0, 1.0)
        return flux

    def harmonic_structure_from_hidden(
        self,
        tap_hidden: mx.array,
        query_hidden: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        write_patch_hidden = self._patch_pool_hidden(mx.stop_gradient(self.strip_registers(tap_hidden)))
        query_patch_hidden = self._patch_pool_hidden(self.strip_registers(query_hidden))
        if write_patch_hidden.shape[1] <= 0:
            write_patches = write_patch_hidden[:, :, : self.harmonic_chord_dim]
        else:
            write_patches = self.harmonic_write_proj(base.rms_norm(write_patch_hidden)).astype(base.COMPUTE_DTYPE)
        if query_patch_hidden.shape[1] <= 0:
            query_patches = query_patch_hidden[:, :, : self.harmonic_chord_dim]
        else:
            query_patches = self.harmonic_query_proj(base.rms_norm(query_patch_hidden)).astype(base.COMPUTE_DTYPE)
        batch = int(write_patches.shape[0])
        num_patches = int(write_patches.shape[1])
        chord_dim = int(write_patches.shape[2]) if write_patches.ndim == 3 else self.harmonic_chord_dim
        if num_patches <= 0:
            empty_patch_reads = mx.zeros((batch, 0, chord_dim), dtype=base.COMPUTE_DTYPE)
            empty_chords = mx.zeros((batch, 0, chord_dim), dtype=base.COMPUTE_DTYPE)
            empty_mask = mx.zeros((batch, 0), dtype=mx.float32)
            zeros = mx.zeros((batch,), dtype=mx.float32)
            return empty_patch_reads, empty_chords, empty_mask, zeros, zeros

        flux_source = base.rms_norm(write_patch_hidden).astype(base.COMPUTE_DTYPE)
        flux_unit = auxlib.l2_normalize(flux_source)
        flux = mx.zeros((batch, num_patches), dtype=mx.float32)
        if num_patches > 1:
            flux_delta = 1.0 - mx.sum(
                flux_unit[:, 1:, :].astype(mx.float32) * flux_unit[:, :-1, :].astype(mx.float32),
                axis=-1,
            )
            flux = mx.concatenate(
                [
                    mx.zeros((batch, 1), dtype=mx.float32),
                    flux_delta.astype(mx.float32),
                ],
                axis=1,
            )

        patch_positions = mx.arange(num_patches, dtype=mx.int32)
        periodic_boundary = ((patch_positions % self.harmonic_max_patches_per_chord) == 0)[None, :]
        threshold_boundary = flux >= mx.array(self.harmonic_boundary_threshold, dtype=mx.float32)
        if num_patches > 2:
            prev_flux = mx.concatenate(
                [mx.full((batch, 1), -1e9, dtype=mx.float32), flux[:, :-1]],
                axis=1,
            )
            next_flux = mx.concatenate(
                [flux[:, 1:], mx.full((batch, 1), -1e9, dtype=mx.float32)],
                axis=1,
            )
            local_peak = mx.logical_and(flux >= prev_flux, flux > next_flux)
            threshold_boundary = mx.logical_and(threshold_boundary, local_peak)
        boundary_flags = mx.logical_or(periodic_boundary, threshold_boundary)
        first_patch = mx.concatenate(
            [
                mx.ones((batch, 1), dtype=mx.bool_),
                mx.zeros((batch, max(num_patches - 1, 0)), dtype=mx.bool_),
            ],
            axis=1,
        )
        boundary_flags = mx.logical_or(boundary_flags, first_patch)

        segment_ids = mx.cumsum(boundary_flags.astype(mx.int32), axis=1) - 1
        chord_slots = mx.arange(num_patches, dtype=mx.int32)
        segment_onehot = (segment_ids[:, :, None] == chord_slots[None, None, :]).astype(mx.float32)
        chord_counts = mx.sum(segment_onehot, axis=1)
        chord_mask = chord_counts > 0.0
        chord_sums = mx.matmul(mx.swapaxes(segment_onehot, 1, 2), write_patches.astype(base.COMPUTE_DTYPE))
        chord_states = chord_sums / mx.maximum(chord_counts[:, :, None].astype(base.COMPUTE_DTYPE), 1.0)

        if self.harmonic_bank_size > 0 and num_patches > self.harmonic_bank_size:
            chord_count_per_batch = mx.sum(chord_mask.astype(mx.int32), axis=1)
            keep_start = mx.maximum(chord_count_per_batch - self.harmonic_bank_size, 0)
            recent_keep = chord_slots[None, :] >= keep_start[:, None]
            chord_mask = mx.logical_and(chord_mask, recent_keep)

        if self.harmonic_enable_read:
            query_unit = auxlib.l2_normalize(query_patches.astype(base.COMPUTE_DTYPE))
            key_unit = auxlib.l2_normalize(mx.stop_gradient(chord_states).astype(base.COMPUTE_DTYPE))
            scores = (mx.matmul(query_unit, mx.swapaxes(key_unit, 1, 2)) * 8.0).astype(mx.float32)
            prev_mask = mx.logical_and(
                chord_slots[None, None, :] < segment_ids[:, :, None],
                chord_mask[:, None, :],
            )
            prev_mask_f = prev_mask.astype(mx.float32)
            masked_scores = scores + (prev_mask_f - 1.0) * 1e9
            weights = mx.softmax(masked_scores, axis=-1).astype(base.COMPUTE_DTYPE) * prev_mask_f.astype(base.COMPUTE_DTYPE)
            weights_sum = mx.sum(weights, axis=-1, keepdims=True)
            weights = weights / mx.maximum(weights_sum, 1e-6)
            patch_reads_out = mx.matmul(weights, mx.stop_gradient(chord_states).astype(base.COMPUTE_DTYPE))
        else:
            patch_reads_out = mx.zeros((batch, num_patches, chord_dim), dtype=base.COMPUTE_DTYPE)

        valid_flux = flux[:, 1:] if num_patches > 1 else flux[:, :0]
        if num_patches > 1:
            flux_mean_arr = mx.mean(valid_flux.astype(mx.float32), axis=1)
        else:
            flux_mean_arr = mx.zeros((batch,), dtype=mx.float32)
        chord_count_arr = mx.sum(chord_mask.astype(mx.float32), axis=1)
        return patch_reads_out, chord_states.astype(base.COMPUTE_DTYPE), chord_mask.astype(mx.float32), chord_count_arr, flux_mean_arr

    def harmonic_token_condition(
        self,
        patch_reads: mx.array,
        token_len: int,
    ) -> mx.array:
        batch = int(patch_reads.shape[0]) if patch_reads.ndim > 0 else 0
        model_dim = self.tok_emb.weight.shape[1]
        if int(patch_reads.shape[1]) <= 0 or not self.harmonic_enable_read:
            return mx.zeros((batch, token_len, model_dim), dtype=base.COMPUTE_DTYPE)
        patch_reads_proj = self.harmonic_read_proj(base.rms_norm(patch_reads)).astype(base.COMPUTE_DTYPE)
        patch_count = int(patch_reads_proj.shape[1])
        token_positions = mx.arange(token_len, dtype=mx.int32)
        patch_ids = mx.minimum(
            token_positions // self.harmonic_patch_len,
            mx.array(patch_count - 1, dtype=mx.int32),
        )
        return mx.take(patch_reads_proj, patch_ids, axis=1)

    def harmonic_terms_from_chords(
        self,
        chord_states: mx.array,
        chord_mask: mx.array,
    ) -> mx.array:
        if chord_states.shape[1] <= self.harmonic_pred_offset:
            return mx.array(0.0, dtype=mx.float32)
        source = chord_states[:, :-self.harmonic_pred_offset, :]
        target = chord_states[:, self.harmonic_pred_offset :, :]
        pair_mask = chord_mask[:, :-self.harmonic_pred_offset] * chord_mask[:, self.harmonic_pred_offset :]
        source_u = auxlib.l2_normalize(source)
        target_u = mx.stop_gradient(auxlib.l2_normalize(target))
        pred_u = auxlib.l2_normalize(self.harmonic_pred(source_u).astype(base.COMPUTE_DTYPE))
        cosine_err = (1.0 - mx.sum(pred_u.astype(mx.float32) * target_u.astype(mx.float32), axis=-1)).astype(mx.float32)
        denom = mx.maximum(mx.sum(pair_mask.astype(mx.float32)), mx.array(1.0, dtype=mx.float32))
        return mx.sum(cosine_err * pair_mask.astype(mx.float32)) / denom

    def forward_hidden_with_aux(
        self,
        input_ids: mx.array,
        capture_layers: tuple[int, ...] = (),
        operator_codes: mx.array | None = None,
    ) -> tuple[mx.array, dict[int, mx.array], dict[str, mx.array | None]]:
        del operator_codes
        x = self.embed_inputs(input_ids)
        x0 = x
        attn_mask = self.attention_mask(x.shape[1])
        skips: list[mx.array] = []
        captured: dict[int, mx.array] = {}
        layer_idx = 0
        tap_hidden: mx.array | None = None

        for i in range(self.num_encoder_layers):
            x = self.block_for_step(i)(x, x0, attn_mask=attn_mask)
            if layer_idx == self.harmonic_tap_layer_index:
                tap_hidden = x
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            skips.append(x)
            layer_idx += 1

        if tap_hidden is None:
            tap_hidden = x
        patch_reads, chord_states, chord_mask, chord_counts, flux_means = self.harmonic_structure_from_hidden(
            tap_hidden,
            x,
        )
        harmonic_tokens = self.harmonic_token_condition(patch_reads, int(input_ids.shape[1]))

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = x + self.harmonic_read_scale[i].astype(x.dtype)[None, None, :] * harmonic_tokens.astype(x.dtype)
            x = self.block_for_step(self.num_encoder_layers + i)(x, x0, attn_mask=attn_mask)
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            layer_idx += 1

        return self.final_norm(self.strip_registers(x)), captured, {
            "tap_hidden": tap_hidden,
            "harmonic_patch_reads": patch_reads,
            "harmonic_chords": chord_states,
            "harmonic_chord_mask": chord_mask,
            "harmonic_token_reads": harmonic_tokens,
            "harmonic_chord_counts": chord_counts,
            "harmonic_flux_means": flux_means,
            "harmonic_patch_len": mx.array(float(self.harmonic_patch_len), dtype=mx.float32),
        }

    def __call__(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        return self.forward_hidden_with_aux(input_ids, operator_codes=operator_codes)[0]

    def forward_logits(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        final_hidden = self(input_ids, operator_codes=operator_codes)
        logits_proj = (
            final_hidden @ self.tok_emb.weight.astype(final_hidden.dtype).T
            if self.tie_embeddings
            else self.lm_head(final_hidden)
        )
        return self.softcap(logits_proj)

    def ce_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
    ) -> mx.array:
        final_hidden = self(input_ids, operator_codes=operator_codes)
        return self.token_ce_from_hidden(final_hidden, target_ids)

    def loss_terms(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        *,
        jepa_weight: float | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        jepa_weight = self.harmonic_jepa_weight if jepa_weight is None else jepa_weight
        final_hidden, _captured, aux = self.forward_hidden_with_aux(input_ids)
        ce_loss = self.token_ce_from_hidden(final_hidden, target_ids)
        chord_states = aux["harmonic_chords"]
        chord_mask = aux["harmonic_chord_mask"]
        assert chord_states is not None
        assert chord_mask is not None
        jepa_loss = self.harmonic_terms_from_chords(chord_states, chord_mask)
        total = ce_loss + mx.array(jepa_weight, dtype=mx.float32) * jepa_loss
        chord_counts = aux["harmonic_chord_counts"]
        flux_means = aux["harmonic_flux_means"]
        assert chord_counts is not None
        assert flux_means is not None
        return total, ce_loss, jepa_loss, mx.mean(chord_counts.astype(mx.float32)), mx.mean(flux_means.astype(mx.float32))


def make_harmonic_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTHarmonic:
    return GPTHarmonic(
        **base.gpt_kwargs_from_args(args, sp),
        harmonic_patch_len=args.harmonic_patch_len,
        harmonic_tap_layer=args.harmonic_tap_layer,
        harmonic_chord_dim=args.harmonic_chord_dim,
        harmonic_min_patches_per_chord=args.harmonic_min_patches_per_chord,
        harmonic_max_patches_per_chord=args.harmonic_max_patches_per_chord,
        harmonic_boundary_threshold=args.harmonic_boundary_threshold,
        harmonic_bank_size=args.harmonic_bank_size,
        harmonic_read_topk=args.harmonic_read_topk,
        harmonic_enable_read=args.harmonic_enable_read,
        harmonic_pred_offset=args.harmonic_pred_offset,
        harmonic_jepa_weight=args.harmonic_jepa_weight,
        harmonic_read_init_std=args.harmonic_read_init_std,
        harmonic_pred_init_std=args.harmonic_pred_init_std,
    )


def exportable_flat_state(model: GPTHarmonic) -> dict[str, mx.array]:
    flat = dict(tree_flatten(model.state))
    return {k: v for k, v in flat.items() if "mlx.core.array" in str(type(v))}


def loss_and_grad_batch(
    args: Hyperparameters,
    model: GPTHarmonic,
    train_loader,
    compiled_loss_and_grad,
    *,
    jepa_weight: float,
) -> tuple[mx.array, dict, tuple[mx.array, mx.array] | None]:
    x_np, y_np = train_loader.next_batch_np(args.microbatch_tokens, args.train_seq_len)
    chunk_sizes = base.token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    rows_per_chunk = [chunk_tokens // args.train_seq_len for chunk_tokens in chunk_sizes]
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    last_batch: tuple[mx.array, mx.array] | None = None
    row_start = 0
    for rows in rows_per_chunk:
        row_end = row_start + rows
        x = mx.array(x_np[row_start:row_end], dtype=mx.int32)
        y = mx.array(y_np[row_start:row_end], dtype=mx.int32)
        last_batch = (x, y)
        loss, grads = compiled_loss_and_grad(x, y, jepa_weight)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
        row_start = row_end
    return loss_value, tree_unflatten(list(grad_accum.items())), last_batch


def main() -> None:
    args = Hyperparameters()
    if args.num_registers != 0 or args.logic_dim != 0 or args.polarity_detector_enabled:
        raise ValueError("train_gpt_mlx_harmonic.py is a clean harmonic prototype; disable registers/logic/polarity")
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
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
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

    batch_seqs = args.microbatch_tokens // args.train_seq_len
    num_streams = args.longctx_num_streams if args.longctx_num_streams > 0 else batch_seqs
    if num_streams != batch_seqs and not args.curriculum_enabled:
        raise ValueError(
            f"LONGCTX_NUM_STREAMS must match microbatch batch size for harmonic, got {num_streams} vs {batch_seqs}"
        )

    mx.random.seed(args.seed)
    if args.curriculum_enabled:
        train_loader = base.build_train_loader(args, log_fn=log, dataset_name=dataset_name)
    else:
        train_loader = GroupedStreamingTokenLoader(
            args.train_files,
            num_streams=num_streams,
            seq_len=args.train_seq_len,
            log_fn=log,
            dataset_name=dataset_name,
        )
    model = make_harmonic_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: GPTHarmonic | None = None

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    if compiled:
        compiled_loss_components = mx.compile(
            lambda x, y, jepa_w: model.loss_terms(x, y, jepa_weight=jepa_w),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_and_grad_impl = mx.compile(
            nn.value_and_grad(model, lambda x, y, jepa_w: model.loss_terms(x, y, jepa_weight=jepa_w)[0]),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_and_grad = lambda x, y, jepa_w: compiled_loss_and_grad_impl(x, y, jepa_w)
    else:
        compiled_loss_components = lambda x, y, jepa_w: model.loss_terms(x, y, jepa_weight=jepa_w)
        compiled_loss_and_grad = nn.value_and_grad(
            model,
            lambda x, y, jepa_w: model.loss_terms(x, y, jepa_weight=jepa_w)[0],
        )

    n_params = sum(int(np.prod(param.shape)) for _, param in tree_flatten(model.trainable_parameters()))
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
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"layer_templates:{args.num_layer_templates} dim:{args.model_dim} heads:{args.num_heads} "
        f"kv_heads:{args.num_kv_heads} seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"harmonic:streams:{num_streams} patch_len:{args.harmonic_patch_len} tap_layer:{args.harmonic_tap_layer} "
        f"chord_dim:{args.harmonic_chord_dim} min_patches:{args.harmonic_min_patches_per_chord} "
        f"max_patches:{args.harmonic_max_patches_per_chord} boundary_threshold:{args.harmonic_boundary_threshold:.4f} "
        f"bank_size:{args.harmonic_bank_size} read_topk:{args.harmonic_read_topk} "
        f"enable_read:{int(args.harmonic_enable_read)} pred_offset:{args.harmonic_pred_offset} "
        f"jepa_weight:{args.harmonic_jepa_weight:.4f}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{batch_seqs} "
        f"val_batch_size:{args.val_batch_size} val_seqs:{(val_tokens.size - 1) // args.train_seq_len} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:{opt.matrix_optimizer_name}+adam matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log(
        f"compute_dtype:{base.COMPUTE_DTYPE} compile:{compiled} compile_mode:{args.mlx_compile} "
        f"quant_format:{args.quant_format}"
    )
    if args.curriculum_enabled and hasattr(train_loader, "summary"):
        summary = train_loader.summary()
        summary_parts = [f"{key}:{value}" for key, value in summary.items()]
        if summary_parts:
            log("curriculum_loader:" + " ".join(summary_parts))

    def eval_val_for_model(
        eval_model: GPTHarmonic,
        eval_tokens: np.ndarray,
        *,
        log_eval_progress: bool = False,
    ) -> tuple[float, float]:
        return base.eval_val(
            args,
            eval_model,
            lambda x, y, operator_codes=None: eval_model.ce_loss(x, y),
            lambda x, operator_codes=None: eval_model.forward_logits(x),
            eval_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log if log_eval_progress else None,
        )

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
                val_loss, val_bpb = eval_val_for_model(model, val_tokens, log_eval_progress=True)
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = make_harmonic_gpt(args, sp)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val_for_model(model, quant_eval_tokens)
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = eval_val_for_model(quant_eval_model, quant_eval_tokens)
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
        jepa_weight = args.harmonic_jepa_weight
        model.set_turbo_qat(qat_active, qat_scale)
        step_loss_and_grad = (
            (
                lambda x, y, jepa_w: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner, jepa_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        jepa_weight=jepa_inner,
                    )[0] + qat_lambda * model.turbo_regularizer(),
                )(x, y, jepa_w)
            )
            if qat_active
            else compiled_loss_and_grad
        )
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        last_batch: tuple[mx.array, mx.array] | None = None
        grad_scale = 1.0 / args.grad_accum_steps
        step_t0 = time.perf_counter()
        for _ in range(args.grad_accum_steps):
            train_loss, grads, last_batch = loss_and_grad_batch(
                args,
                model,
                train_loader,
                step_loss_and_grad,
                jepa_weight=jepa_weight,
            )
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
        grads_tree = tree_unflatten(list(accum.items()))
        train_loss_value = float(train_loss.item())
        sanitize_this_step = base.should_sanitize_nonfinite_grads(args, step, train_loss_value)
        if sanitize_this_step:
            grads_tree, grad_nonfinite = base.sanitize_grad_tree(grads_tree, topk=args.nonfinite_grad_topk)
        else:
            grad_nonfinite = base.empty_nonfinite_grad_summary()
        grads_tree = base.clip_grad_tree(grads_tree, args.grad_clip_norm)
        flat_clipped_grads = dict(tree_flatten(grads_tree))
        opt.step(model, grads_tree, step=step, lr_mul=lr_mul)
        model.clear_turbo_cache()
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            extra = ""
            tensor_activity_str = ""
            should_log_tensor_activity = (
                args.tensor_activity_log_every > 0
                and (step <= 10 or step % args.tensor_activity_log_every == 0 or stop_after_step is not None)
            )
            if should_log_tensor_activity:
                activity = base.tensor_activity_snapshot(
                    flat_clipped_grads,
                    hot_threshold=args.tensor_activity_hot_threshold,
                    warm_threshold=args.tensor_activity_warm_threshold,
                    nonzero_threshold=args.tensor_activity_nonzero_threshold,
                    topk=args.tensor_activity_topk,
                )
                total_params = max(int(activity["param_count"]), 1)
                top_items = activity["top"]
                top_str = ",".join(
                    f"{name}:{bucket}:{mean_abs:.2e}:{nonzero_frac:.2f}"
                    for mean_abs, nonzero_frac, _numel, name, bucket in top_items
                )
                tensor_activity_str = (
                    f" grad_hot_tensors:{activity['hot_tensors']}/{activity['tensor_count']}"
                    f" grad_warm_tensors:{activity['warm_tensors']}/{activity['tensor_count']}"
                    f" grad_cold_tensors:{activity['cold_tensors']}/{activity['tensor_count']}"
                    f" grad_hot_param_frac:{int(activity['hot_params']) / total_params:.3f}"
                    f" grad_warm_param_frac:{int(activity['warm_params']) / total_params:.3f}"
                    f" grad_cold_param_frac:{int(activity['cold_params']) / total_params:.3f}"
                )
                if top_str:
                    tensor_activity_str += f" grad_top:{top_str}"
            grad_nonfinite_str = ""
            if int(grad_nonfinite["nonfinite_tensors"]) > 0:
                total_params = max(int(grad_nonfinite["param_count"]), 1)
                top_nonfinite = ",".join(f"{name}:{count}" for count, name in grad_nonfinite["top"])
                grad_nonfinite_str = (
                    f" grad_nonfinite_tensors:{grad_nonfinite['nonfinite_tensors']}/{grad_nonfinite['tensor_count']}"
                    f" grad_nonfinite_param_frac:{int(grad_nonfinite['nonfinite_params']) / total_params:.3f}"
                )
                if top_nonfinite:
                    grad_nonfinite_str += f" grad_nonfinite_top:{top_nonfinite}"
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                metrics = compiled_loss_components(last_batch[0], last_batch[1], jepa_weight)
                mx.eval(*metrics)
                _, ce_metric, jepa_metric, chord_count_metric, flux_metric = metrics
                extra = (
                    f" ce:{float(ce_metric.item()):.4f}"
                    f" harmonic_jepa:{float(jepa_metric.item()):.4f}"
                    f" harmonic_chords:{float(chord_count_metric.item()):.2f}"
                    f" harmonic_flux:{float(flux_metric.item()):.4f}"
                )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.0f} harmonic_jepa_weight:{jepa_weight:.4f} "
                f"turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f}"
                f"{grad_nonfinite_str}{tensor_activity_str}{extra}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    flat_state = exportable_flat_state(model)
    mx.savez(str(out_path), **flat_state)
    model_bytes = out_path.stat().st_size
    code_bytes = len(code.encode("utf-8"))
    log(f"Serialized model: {model_bytes} bytes")
    log(f"Code size: {code_bytes} bytes")
    log(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.pkl.z"
    quant_path.write_bytes(quant_blob)
    log(
        f"Serialized model int8+zlib: {len(quant_blob)} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)})"
    )
    log(f"Total submission size int8+zlib: {len(quant_blob) + code_bytes} bytes")

    quant_roundtrip = make_harmonic_gpt(args, sp)
    quant_roundtrip.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
    quant_roundtrip.clear_turbo_cache()
    raw_val_loss, raw_val_bpb = eval_val_for_model(model, val_tokens)
    q_val_loss, q_val_bpb = eval_val_for_model(quant_roundtrip, val_tokens)
    log(f"final_raw_exact_val_loss:{raw_val_loss:.8f} final_raw_exact_val_bpb:{raw_val_bpb:.8f}")
    log(f"final_int8_zlib_roundtrip_exact_val_loss:{q_val_loss:.8f} final_int8_zlib_roundtrip_exact_val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
