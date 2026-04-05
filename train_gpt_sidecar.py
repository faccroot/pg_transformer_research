"""
Dedicated CUDA trainer for a chunk-causal structural sidecar.

This keeps the base transformer and export path close to train_gpt.py while
adding the chunk-level GRU sidecar, polarity detector, and polarity-write path.
The scored path is causal at chunk granularity: tokens in chunk c only read
sidecar state from chunk c-1.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

import train_gpt as base
from train_jepa import SIGReg


EXTRA_CONTROL_PATTERNS = (
    "sidecar_summary_query",
    "sidecar_read_scale",
    "sidecar_polarity_write_scale",
    "polarity_detector.bias",
)
base.CONTROL_TENSOR_NAME_PATTERNS = tuple(dict.fromkeys(base.CONTROL_TENSOR_NAME_PATTERNS + EXTRA_CONTROL_PATTERNS))
base.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    dict.fromkeys(base.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS + EXTRA_CONTROL_PATTERNS)
)


NOT_OPERATOR_SURFACES = frozenset(
    {
        "not",
        "n't",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "except",
        "unless",
        "lack",
        "absence",
        "fail",
        "cannot",
        "can't",
        "won't",
        "don't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
    }
)
AND_OPERATOR_SURFACES = frozenset(
    {"and", "also", "moreover", "additionally", "furthermore", "both", "plus", "together"}
)
OR_OPERATOR_SURFACES = frozenset({"or", "either", "alternatively", "otherwise", "else"})


@dataclass(frozen=True)
class OperatorRoutingSpec:
    lookup: np.ndarray
    piece_starts_word: np.ndarray
    piece_has_alnum: np.ndarray
    patterns_by_first_token: dict[int, tuple[tuple[np.ndarray, int], ...]]


def _normalize_piece(piece: str) -> str:
    return piece.lower().replace("’", "'").lstrip("▁").strip()


def operator_code_for_piece(piece: str) -> int:
    normalized = _normalize_piece(piece)
    if not normalized:
        return 0
    if normalized in NOT_OPERATOR_SURFACES or normalized.endswith("n't"):
        return 1
    if normalized in AND_OPERATOR_SURFACES:
        return 2
    if normalized in OR_OPERATOR_SURFACES:
        return 3
    return 0


def build_operator_lookup(sp: spm.SentencePieceProcessor, vocab_size: int) -> np.ndarray:
    table = np.zeros((vocab_size,), dtype=np.int32)
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        if not piece.startswith("▁"):
            continue
        table[token_id] = operator_code_for_piece(piece)
    return table


def _extract_pattern_ids(
    sp: spm.SentencePieceProcessor,
    text: str,
    target_begin: int,
    target_end: int,
) -> tuple[int, ...]:
    proto = sp.encode_as_immutable_proto(text)
    return tuple(
        int(piece.id)
        for piece in proto.pieces
        if not (int(piece.end) <= target_begin or int(piece.begin) >= target_end)
    )


def _surface_pattern_sequences(sp: spm.SentencePieceProcessor, surface: str) -> set[tuple[int, ...]]:
    patterns: set[tuple[int, ...]] = set()
    raw_ids = _extract_pattern_ids(sp, surface, 0, len(surface))
    if raw_ids:
        patterns.add(raw_ids)
    sentence_text = f"a {surface} b"
    sentence_ids = _extract_pattern_ids(sp, sentence_text, 1, 2 + len(surface))
    if sentence_ids:
        patterns.add(sentence_ids)
    return patterns


def build_operator_routing_spec(sp: spm.SentencePieceProcessor, vocab_size: int) -> OperatorRoutingSpec:
    lookup = build_operator_lookup(sp, vocab_size)
    piece_starts_word = np.zeros((vocab_size,), dtype=np.int32)
    piece_has_alnum = np.zeros((vocab_size,), dtype=np.int32)
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        piece_starts_word[token_id] = int(piece.startswith("▁"))
        piece_has_alnum[token_id] = int(any(ch.isalnum() for ch in _normalize_piece(piece)))

    pattern_groups: dict[int, dict[tuple[int, ...], int]] = {}
    for code, surfaces in (
        (1, NOT_OPERATOR_SURFACES),
        (2, AND_OPERATOR_SURFACES),
        (3, OR_OPERATOR_SURFACES),
    ):
        for surface in sorted(surfaces):
            for pattern in _surface_pattern_sequences(sp, surface):
                existing = pattern_groups.setdefault(len(pattern), {}).get(pattern)
                if existing is not None and existing != code:
                    raise ValueError(f"operator pattern collision for ids={pattern}: {existing} vs {code}")
                pattern_groups.setdefault(len(pattern), {})[pattern] = code

    patterns_by_first_token: dict[int, list[tuple[np.ndarray, int]]] = {}
    for mapping in pattern_groups.values():
        for token_ids, code in mapping.items():
            arr = np.ascontiguousarray(np.array(token_ids, dtype=np.int32))
            patterns_by_first_token.setdefault(int(arr[0]), []).append((arr, int(code)))

    final_patterns = {
        token_id: tuple(sorted(candidates, key=lambda item: (int(item[0].shape[0]), tuple(int(v) for v in item[0]))))
        for token_id, candidates in patterns_by_first_token.items()
    }
    return OperatorRoutingSpec(
        lookup=lookup,
        piece_starts_word=piece_starts_word,
        piece_has_alnum=piece_has_alnum,
        patterns_by_first_token=final_patterns,
    )


def detect_operator_codes_np(input_ids: np.ndarray, routing: OperatorRoutingSpec) -> np.ndarray:
    input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
    operator_codes = np.take(routing.lookup, input_ids, axis=0)
    batch_size, seq_len = input_ids.shape
    for batch_idx in range(batch_size):
        row = input_ids[batch_idx]
        row_codes = operator_codes[batch_idx]
        for start in range(seq_len):
            token_id = int(row[start])
            candidates = routing.patterns_by_first_token.get(token_id)
            if not candidates:
                continue
            if start > 0 and routing.piece_starts_word[token_id] <= 0 and routing.piece_has_alnum[int(row[start - 1])] != 0:
                continue
            for token_ids, code in candidates:
                length = int(token_ids.shape[0])
                end = start + length
                if end > seq_len:
                    continue
                if end < seq_len:
                    next_id = int(row[end])
                    if routing.piece_starts_word[next_id] <= 0 and routing.piece_has_alnum[next_id] != 0:
                        continue
                if length > 1 and not np.array_equal(row[start:end], token_ids):
                    continue
                row_codes[end - 1] = np.int32(code)
    return operator_codes


def route_operator_codes(operator_codes: np.ndarray) -> np.ndarray:
    routed = np.zeros_like(operator_codes)
    routed[:, 1:] = operator_codes[:, :-1]
    return routed


def l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def spherical_uniformity(u: Tensor) -> Tensor:
    flat = u.reshape(-1, u.size(-1)).float()
    n = flat.size(0)
    if n <= 1:
        return flat.new_zeros(())
    gram = torch.abs(flat @ flat.T)
    mask = torch.ones((n, n), device=flat.device, dtype=flat.dtype) - torch.eye(n, device=flat.device, dtype=flat.dtype)
    return (gram * mask).sum() / float(n * (n - 1))


class Hyperparameters(base.Hyperparameters):
    sidecar_chunk_size = int(os.environ.get("SIDECAR_CHUNK_SIZE", 8))
    sidecar_tap_layer = int(os.environ.get("SIDECAR_TAP_LAYER", -1))
    sidecar_state_dim = int(os.environ.get("SIDECAR_STATE_DIM", 64))
    sidecar_pred_offset = int(os.environ.get("SIDECAR_PRED_OFFSET", 1))
    sidecar_pred_weight = float(os.environ.get("SIDECAR_PRED_WEIGHT", 0.05))
    sidecar_sigreg_weight = float(os.environ.get("SIDECAR_SIGREG_WEIGHT", 0.01))
    sidecar_spherical_weight = float(os.environ.get("SIDECAR_SPHERICAL_WEIGHT", 0.01))
    sidecar_aux_start_frac = float(os.environ.get("SIDECAR_AUX_START_FRAC", 0.0))
    sidecar_aux_ramp_frac = float(os.environ.get("SIDECAR_AUX_RAMP_FRAC", 0.0))
    sidecar_summary_mode = os.environ.get("SIDECAR_SUMMARY_MODE", "query")
    sidecar_pred_target_mode = os.environ.get("SIDECAR_PRED_TARGET_MODE", "delta")
    sidecar_read_init_std = float(os.environ.get("SIDECAR_READ_INIT_STD", 1e-3))
    sidecar_pred_init_std = float(os.environ.get("SIDECAR_PRED_INIT_STD", 1e-4))
    sidecar_sigreg_knots = int(os.environ.get("SIDECAR_SIGREG_KNOTS", 17))
    sidecar_sigreg_num_proj = int(os.environ.get("SIDECAR_SIGREG_NUM_PROJ", 256))
    sidecar_read_rmsnorm = bool(int(os.environ.get("SIDECAR_READ_RMSNORM", "1")))
    sidecar_polarity_write = bool(int(os.environ.get("SIDECAR_POLARITY_WRITE", "0")))
    sidecar_polarity_pool = os.environ.get("SIDECAR_POLARITY_POOL", "max")
    logic_route_to_next_token = bool(int(os.environ.get("LOGIC_ROUTE_TO_NEXT_TOKEN", "1")))
    logic_operator_mode = os.environ.get("LOGIC_OPERATOR_MODE", "not_only")
    polarity_detector_enabled = bool(int(os.environ.get("POLARITY_DETECTOR_ENABLED", "0")))
    polarity_detector_layer_index = int(os.environ.get("POLARITY_DETECTOR_LAYER_INDEX", 3))
    polarity_detector_hidden_dim = int(os.environ.get("POLARITY_DETECTOR_HIDDEN_DIM", 64))
    polarity_seed_blend = float(os.environ.get("POLARITY_SEED_BLEND", 1.0))
    polarity_seed_weight = float(os.environ.get("POLARITY_SEED_WEIGHT", 0.0))
    polarity_sparse_weight = float(os.environ.get("POLARITY_SPARSE_WEIGHT", 0.0))
    polarity_smooth_weight = float(os.environ.get("POLARITY_SMOOTH_WEIGHT", 0.0))
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "1")))


class PolarityDetector(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int = 0, init_bias: float = -4.0):
        super().__init__()
        self.fc = base.CastedLinear(model_dim, hidden_dim, bias=False) if hidden_dim > 0 else None
        self.out = base.CastedLinear(hidden_dim if hidden_dim > 0 else model_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.full((1,), init_bias, dtype=torch.float32))

    def forward(self, h: Tensor) -> Tensor:
        x = F.silu(self.fc(h)) if self.fc is not None else h
        return self.out(x)[..., 0].float() + self.bias.float()


class GRUSidecarCell(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.in_proj = base.CastedLinear(state_dim, state_dim * 3, bias=False)
        self.h_proj = base.CastedLinear(state_dim, state_dim * 3, bias=False)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=state_dim ** -0.5)
        nn.init.normal_(self.h_proj.weight, mean=0.0, std=state_dim ** -0.5)

    def forward(self, x_t: Tensor, h_t: Tensor) -> Tensor:
        gates_x = self.in_proj(x_t).float()
        gates_h = self.h_proj(h_t).float()
        x_r, x_z, x_n = torch.split(gates_x, self.state_dim, dim=-1)
        h_r, h_z, h_n = torch.split(gates_h, self.state_dim, dim=-1)
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
        return ((1.0 - z) * n + z * h_t.float()).to(dtype=x_t.dtype)


class GPTSidecar(base.GPT):
    def __init__(self, args: Hyperparameters, operator_routing: OperatorRoutingSpec):
        super().__init__(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_layer_templates=args.num_layer_templates,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        if args.train_seq_len % args.sidecar_chunk_size != 0:
            raise ValueError(
                f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by SIDECAR_CHUNK_SIZE={args.sidecar_chunk_size}"
            )
        if args.sidecar_chunk_size <= 1:
            raise ValueError(f"SIDECAR_CHUNK_SIZE must be > 1, got {args.sidecar_chunk_size}")
        if args.sidecar_pred_offset <= 0:
            raise ValueError(f"SIDECAR_PRED_OFFSET must be > 0, got {args.sidecar_pred_offset}")
        self.sidecar_chunk_size = args.sidecar_chunk_size
        self.sidecar_tap_layer_index = args.sidecar_tap_layer if args.sidecar_tap_layer >= 0 else max(self.num_encoder_layers - 1, 0)
        self.sidecar_state_dim = args.sidecar_state_dim
        self.sidecar_pred_offset = args.sidecar_pred_offset
        self.sidecar_pred_weight = args.sidecar_pred_weight
        self.sidecar_sigreg_weight = args.sidecar_sigreg_weight
        self.sidecar_spherical_weight = args.sidecar_spherical_weight
        self.sidecar_summary_mode = args.sidecar_summary_mode
        self.sidecar_pred_target_mode = args.sidecar_pred_target_mode
        self.sidecar_read_rmsnorm = args.sidecar_read_rmsnorm
        self.sidecar_polarity_write = args.sidecar_polarity_write
        self.sidecar_polarity_pool = args.sidecar_polarity_pool
        self.logic_route_to_next_token = args.logic_route_to_next_token
        self.logic_operator_mode = args.logic_operator_mode
        self.polarity_seed_blend = args.polarity_seed_blend
        self.polarity_seed_weight = args.polarity_seed_weight
        self.polarity_sparse_weight = args.polarity_sparse_weight
        self.polarity_smooth_weight = args.polarity_smooth_weight
        self.polarity_detector_layer_index = args.polarity_detector_layer_index
        self.operator_routing = operator_routing

        self.register_buffer("operator_lookup", torch.from_numpy(operator_routing.lookup).long(), persistent=False)
        self.sidecar_summary_query = nn.Parameter(torch.randn(args.model_dim, dtype=torch.float32) * (args.model_dim ** -0.5))
        self.sidecar_in_proj = base.CastedLinear(args.model_dim, args.sidecar_state_dim, bias=False)
        self.sidecar_cell = GRUSidecarCell(args.sidecar_state_dim)
        self.sidecar_pred = base.CastedLinear(args.sidecar_state_dim, args.sidecar_state_dim, bias=False)
        nn.init.normal_(self.sidecar_pred.weight, mean=0.0, std=args.sidecar_pred_init_std)
        self.sidecar_sigreg = SIGReg(knots=args.sidecar_sigreg_knots, num_proj=args.sidecar_sigreg_num_proj)
        self.sidecar_read_proj = base.CastedLinear(args.sidecar_state_dim, args.model_dim, bias=False)
        nn.init.normal_(self.sidecar_read_proj.weight, mean=0.0, std=args.sidecar_read_init_std)
        self.sidecar_read_scale = nn.Parameter(torch.full((self.num_decoder_layers, args.model_dim), 0.05, dtype=torch.float32))
        self.sidecar_polarity_write_scale = nn.Parameter(torch.zeros((args.sidecar_state_dim,), dtype=torch.float32))
        self.polarity_detector = (
            PolarityDetector(args.model_dim, hidden_dim=args.polarity_detector_hidden_dim)
            if args.polarity_detector_enabled else None
        )

    def _query_pool_chunks(self, chunks: Tensor) -> Tensor:
        query = self.sidecar_summary_query.float()
        scores = (chunks.float() * query[None, None, None, :]).sum(dim=-1) / math.sqrt(float(chunks.size(-1)))
        weights = scores.softmax(dim=2)
        return (chunks.float() * weights[..., None]).sum(dim=2).to(dtype=chunks.dtype)

    def chunk_summary_from_hidden(self, hidden: Tensor) -> Tensor:
        chunks = hidden.reshape(hidden.size(0), hidden.size(1) // self.sidecar_chunk_size, self.sidecar_chunk_size, hidden.size(2))
        last = chunks[:, :, -1, :]
        if self.sidecar_summary_mode == "last":
            return last
        mean = chunks.float().mean(dim=2).to(dtype=hidden.dtype)
        if self.sidecar_summary_mode == "mean":
            return mean
        if self.sidecar_summary_mode == "mean_last":
            return 0.5 * (mean.float() + last.float()).to(dtype=hidden.dtype)
        return self._query_pool_chunks(chunks)

    def chunk_reduce_scores(self, scores: Tensor) -> Tensor:
        chunks = scores.reshape(scores.size(0), scores.size(1) // self.sidecar_chunk_size, self.sidecar_chunk_size).float()
        if self.sidecar_polarity_pool == "last":
            return chunks[:, :, -1]
        if self.sidecar_polarity_pool == "mean":
            return chunks.mean(dim=2)
        return chunks.max(dim=2).values

    def _normalize_initial_state(self, initial_state: Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if initial_state is None:
            return torch.zeros((batch, self.sidecar_state_dim), device=device, dtype=dtype)
        state = initial_state.to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(batch, -1)
        if state.size(0) != batch or state.size(1) != self.sidecar_state_dim:
            raise ValueError(
                f"initial sidecar state shape {tuple(state.shape)} does not match (batch={batch}, state_dim={self.sidecar_state_dim})"
            )
        return state

    def _tangent_displacement(self, source: Tensor, other: Tensor) -> Tensor:
        source32 = source.float()
        other32 = other.float()
        radial = (source32 * other32).sum(dim=-1, keepdim=True) * source32
        return l2_normalize((other32 - radial).to(dtype=source.dtype))

    def operator_codes_for_numpy(self, input_ids: np.ndarray) -> np.ndarray:
        operator_codes = detect_operator_codes_np(input_ids, self.operator_routing)
        if self.logic_route_to_next_token:
            operator_codes = route_operator_codes(operator_codes)
        return np.ascontiguousarray(operator_codes, dtype=np.int64)

    def operator_codes_for_input(self, input_ids: Tensor) -> Tensor:
        return torch.from_numpy(self.operator_codes_for_numpy(input_ids.detach().cpu().numpy())).to(
            device=input_ids.device, dtype=torch.long
        )

    def seed_polarity_scores(self, operator_codes: Tensor | None) -> Tensor | None:
        if operator_codes is None:
            return None
        return (operator_codes == 1).float()

    def polarity_detector_logits_and_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.polarity_detector is None:
            raise ValueError("polarity_detector_logits_and_scores called without a detector")
        logits = self.polarity_detector(x).float()
        return logits, torch.sigmoid(logits)

    def blend_polarity_scores(self, seed_scores: Tensor | None, detector_scores: Tensor | None) -> Tensor | None:
        if detector_scores is None:
            return seed_scores
        if seed_scores is None:
            return detector_scores
        alpha = self.polarity_seed_blend
        if alpha <= 0.0:
            return detector_scores
        if alpha >= 1.0:
            return seed_scores
        return alpha * seed_scores + (1.0 - alpha) * detector_scores

    def token_ce_from_hidden(self, final_hidden: Tensor, target_ids: Tensor) -> Tensor:
        flat_hidden = final_hidden.reshape(-1, final_hidden.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(flat_hidden, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(flat_hidden)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def sidecar_states_from_hidden(
        self,
        hidden: Tensor,
        chunk_polarity: Tensor | None = None,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        summary = self.chunk_summary_from_hidden(hidden)
        side_in = self.sidecar_in_proj(summary.detach()).to(dtype=hidden.dtype)
        if self.sidecar_polarity_write and chunk_polarity is not None:
            side_in = side_in + chunk_polarity.to(dtype=hidden.dtype).unsqueeze(-1) * torch.tanh(
                self.sidecar_polarity_write_scale
            ).to(dtype=hidden.dtype)[None, None, :]
        batch, chunks = int(side_in.size(0)), int(side_in.size(1))
        state = self._normalize_initial_state(initial_state, batch, side_in.device, side_in.dtype)
        states: list[Tensor] = []
        for idx in range(chunks):
            state = self.sidecar_cell(side_in[:, idx, :], state)
            states.append(state)
        side_states = torch.stack(states, dim=1) if states else side_in[:, :0, :]
        return summary, side_states, state

    def sidecar_token_condition(self, side_states: Tensor, token_len: int, initial_state: Tensor | None = None) -> Tensor:
        if side_states.size(1) <= 0:
            return side_states.new_zeros((side_states.size(0), token_len, self.tok_emb.weight.size(1)))
        init = self._normalize_initial_state(initial_state, int(side_states.size(0)), side_states.device, side_states.dtype)
        side_read_input = torch.cat([init[:, None, :], side_states[:, :-1, :]], dim=1).detach()
        if self.sidecar_read_rmsnorm:
            side_read_input = F.rms_norm(side_read_input, (side_read_input.size(-1),))
        side_read = self.sidecar_read_proj(side_read_input).to(dtype=side_states.dtype)
        side_tokens = side_read[:, :, None, :].expand(-1, -1, self.sidecar_chunk_size, -1)
        return side_tokens.reshape(side_read.size(0), token_len, side_read.size(-1))

    def forward_hidden_with_aux(
        self,
        input_ids: Tensor,
        operator_codes: Tensor | None = None,
        initial_sidecar_state: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        if operator_codes is None and (self.polarity_detector is not None or self.sidecar_polarity_write):
            operator_codes = self.operator_codes_for_input(input_ids)
        skips: list[Tensor] = []
        layer_idx = 0
        seed_polarity_scores = self.seed_polarity_scores(operator_codes)
        polarity_scores = seed_polarity_scores if (self.logic_operator_mode == "not_only" and self.polarity_detector is None) else None
        detector_logits: Tensor | None = None
        detector_scores: Tensor | None = None
        tap_hidden: Tensor | None = None

        for i in range(self.num_encoder_layers):
            x = self._block_for_step(i)(x, x0)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            if layer_idx == self.sidecar_tap_layer_index:
                tap_hidden = x
            skips.append(x)
            layer_idx += 1

        if tap_hidden is None:
            tap_hidden = x
        side_source_scores = polarity_scores if polarity_scores is not None else seed_polarity_scores
        side_chunk_polarity = self.chunk_reduce_scores(side_source_scores) if (self.sidecar_polarity_write and side_source_scores is not None) else None
        side_summary, side_states, final_sidecar_state = self.sidecar_states_from_hidden(
            tap_hidden,
            chunk_polarity=side_chunk_polarity,
            initial_state=initial_sidecar_state,
        )
        side_tokens = self.sidecar_token_condition(side_states, input_ids.size(1), initial_state=initial_sidecar_state)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = x + self.sidecar_read_scale[i].to(dtype=x.dtype)[None, None, :] * side_tokens.to(dtype=x.dtype)
            x = self._block_for_step(self.num_encoder_layers + i)(x, x0)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            layer_idx += 1

        return self.final_norm(x), {
            "sidecar_summary": side_summary,
            "sidecar_states": side_states,
            "final_sidecar_state": final_sidecar_state,
            "seed_polarity_scores": seed_polarity_scores,
            "polarity_detector_logits": detector_logits,
            "polarity_detector_scores": detector_scores,
        }

    def sidecar_terms_from_states(self, side_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if side_states.size(1) <= self.sidecar_pred_offset:
            zero = side_states.new_zeros((), dtype=torch.float32)
            return zero, zero, zero
        u = l2_normalize(side_states)
        source = u[:, :-self.sidecar_pred_offset, :]
        target_state = u[:, self.sidecar_pred_offset :, :].detach()
        pred_raw = self.sidecar_pred(source)
        if self.sidecar_pred_target_mode == "delta":
            pred = self._tangent_displacement(source, pred_raw)
            pred_target = self._tangent_displacement(source, target_state)
        else:
            pred = l2_normalize(pred_raw)
            pred_target = target_state
        pred_loss = (1.0 - (pred.float() * pred_target.float()).sum(dim=-1)).mean().float()
        sigreg_loss = self.sidecar_sigreg(side_states.transpose(0, 1)).float()
        spherical_loss = spherical_uniformity(u)
        return pred_loss, sigreg_loss, spherical_loss

    def polarity_loss_terms_from_aux(self, aux: dict[str, Tensor | None]) -> tuple[Tensor, Tensor, Tensor]:
        logits = aux["polarity_detector_logits"]
        scores = aux["polarity_detector_scores"]
        seed_scores = aux["seed_polarity_scores"]
        zero = self.tok_emb.weight.new_zeros((), dtype=torch.float32)
        if logits is None or scores is None or seed_scores is None:
            return zero, zero, zero
        seed_loss = F.binary_cross_entropy_with_logits(logits.float(), seed_scores.float())
        sparse_loss = scores.float().mean()
        smooth_loss = zero if scores.size(1) <= 1 else (scores[:, 1:].float() - scores[:, :-1].float()).abs().mean()
        return seed_loss, sparse_loss, smooth_loss

    def loss_terms(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        operator_codes: Tensor | None = None,
        aux_scale: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        final_hidden, aux = self.forward_hidden_with_aux(input_ids, operator_codes=operator_codes)
        ce_loss = self.token_ce_from_hidden(final_hidden, target_ids)
        if aux_scale > 0.0:
            pred_loss, sigreg_loss, spherical_loss = self.sidecar_terms_from_states(aux["sidecar_states"])
        else:
            pred_loss = ce_loss.new_zeros(())
            sigreg_loss = ce_loss.new_zeros(())
            spherical_loss = ce_loss.new_zeros(())
        pol_seed, pol_sparse, pol_smooth = self.polarity_loss_terms_from_aux(aux)
        total = ce_loss + aux_scale * (
            self.sidecar_pred_weight * pred_loss
            + self.sidecar_sigreg_weight * sigreg_loss
            + self.sidecar_spherical_weight * spherical_loss
        ) + (
            self.polarity_seed_weight * pol_seed
            + self.polarity_sparse_weight * pol_sparse
            + self.polarity_smooth_weight * pol_smooth
        )
        return total, {
            "ce": ce_loss.detach(),
            "sidecar_pred": pred_loss.detach(),
            "sidecar_sigreg": sigreg_loss.detach(),
            "sidecar_spherical": spherical_loss.detach(),
            "pol_seed": pol_seed.detach(),
            "pol_sparse": pol_sparse.detach(),
            "pol_smooth": pol_smooth.detach(),
        }

    def forward(self, input_ids: Tensor, target_ids: Tensor, operator_codes: Tensor | None = None) -> Tensor:
        final_hidden, _aux = self.forward_hidden_with_aux(input_ids, operator_codes=operator_codes)
        return self.token_ce_from_hidden(final_hidden, target_ids)


def eval_val_sidecar(
    args: Hyperparameters,
    model: nn.Module,
    op_model: GPTSidecar,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE must provide at least one sequence per rank")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            operator_codes = op_model.operator_codes_for_input(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, operator_codes).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    model.train()
    return float(val_loss.item()), float(bits_per_token * (val_token_count.item() / val_byte_count.item()))


def eval_val_sidecar_single_process(
    args: Hyperparameters,
    model: GPTSidecar,
    device: torch.device,
    batch_tokens: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    batch_seqs = batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, batch_seqs):
            batch_seq_end = min(batch_seq_start + batch_seqs, total_seqs)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            operator_codes = model.operator_codes_for_input(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, operator_codes).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if was_training:
        model.train()
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    return float(val_loss.item()), float(bits_per_token * (val_token_count.item() / val_byte_count.item()))


def sidecar_aux_scale(args: Hyperparameters, step: int, elapsed_ms: float, max_wallclock_ms: float | None) -> float:
    progress = step / max(args.iterations, 1)
    if max_wallclock_ms is not None and max_wallclock_ms > 0:
        progress = max(progress, elapsed_ms / max_wallclock_ms)
    if progress < args.sidecar_aux_start_frac:
        return 0.0
    if args.sidecar_aux_ramp_frac <= 0.0:
        return 1.0
    return min((progress - args.sidecar_aux_start_frac) / args.sidecar_aux_ramp_frac, 1.0)


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE does not match tokenizer")
    operator_routing = build_operator_routing_spec(sp, args.vocab_size)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    quant_eval_tokens = base.limit_validation_tokens(val_tokens, args.train_seq_len, args.quant_eval_max_seqs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPTSidecar(args, operator_routing).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, (base.CastedLinear, base.TurboLinear)):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    base_model.set_turbo_qat(False, 0.0)
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=False)
        if args.compile_model and not args.turbo_qat else base_model
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    quant_eval_model: GPTSidecar | None = None
    quant_eval_batch_tokens = max(args.val_batch_size // max(world_size * grad_accum_steps, 1), args.train_seq_len)

    named_params = list(base_model.named_parameters())
    excluded_names = {"tok_emb.weight", "lm_head.weight"}
    matrix_params = [
        p for name, p in named_params
        if name not in excluded_names and p.ndim == 2 and base.infer_turbo_mode(name) == "none"
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    turbo_matrix_params = [
        p for name, p in named_params
        if name not in excluded_names and p.ndim == 2 and base.infer_turbo_mode(name) != "none"
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in named_params
        if name not in excluded_names and (
            p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
        )
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = None
    if matrix_params:
        optimizer_muon = base.Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
    optimizer_muon_turbo = None
    if turbo_matrix_params:
        optimizer_muon_turbo = base.Muon(
            turbo_matrix_params,
            lr=args.matrix_lr,
            momentum=args.turbo_qat_muon_momentum if args.turbo_qat else args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon_turbo.param_groups:
            group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_scalar]
    if optimizer_muon is not None:
        optimizers.insert(1, optimizer_muon)
    if optimizer_muon_turbo is not None:
        optimizers.insert(2, optimizer_muon_turbo)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    log0(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"sidecar:chunk_size:{args.sidecar_chunk_size} tap_layer:{base_model.sidecar_tap_layer_index} state_dim:{args.sidecar_state_dim}")
    log0(
        f"polarity_detector:enabled:{int(args.polarity_detector_enabled)} layer:{args.polarity_detector_layer_index} "
        f"hidden_dim:{args.polarity_detector_hidden_dim} seed_blend:{args.polarity_seed_blend:.3f}"
    )
    log0(f"compile:{args.compile_model and not args.turbo_qat}")

    train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def turbo_qat_scale(step: int) -> float:
        if not args.turbo_qat or args.iterations <= 0:
            return 0.0
        frac = step / max(args.iterations, 1)
        if frac < args.turbo_qat_start_frac:
            return 0.0
        if args.turbo_qat_ramp_frac <= 0:
            return 1.0
        return min((frac - args.turbo_qat_start_frac) / args.turbo_qat_ramp_frac, 1.0)

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                operator_codes = base_model.operator_codes_for_input(x)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss, _ = base_model.loss_terms(x, y, operator_codes=operator_codes, aux_scale=1.0)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_quant_eval = args.quant_eval_every > 0 and (last_step or step % args.quant_eval_every == 0)
        if should_validate or should_quant_eval:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            if should_validate:
                val_loss, val_bpb = eval_val_sidecar(
                    args, model, base_model, rank, world_size, device, grad_accum_steps,
                    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
                )
                log0(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                )
            if should_quant_eval and master_process:
                if quant_eval_model is None:
                    quant_eval_model = copy.deepcopy(base_model)
                qeval_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val_sidecar_single_process(
                    args, base_model, device, quant_eval_batch_tokens, quant_eval_tokens,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
                )
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(base_model.state_dict())
                quant_eval_model.load_state_dict(base.dequantize_state_dict(quant_obj), strict=True)
                q_val_loss, q_val_bpb = eval_val_sidecar_single_process(
                    args, quant_eval_model, device, quant_eval_batch_tokens, quant_eval_tokens,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
                )
                log0(
                    f"step:{step}/{args.iterations} quant_diag_seqs:{(quant_eval_tokens.numel() - 1) // args.train_seq_len} "
                    f"raw_val_loss:{raw_q_val_loss:.4f} raw_val_bpb:{raw_q_val_bpb:.4f} "
                    f"quant_val_loss:{q_val_loss:.4f} quant_val_bpb:{q_val_bpb:.4f} "
                    f"quant_gap_bpb:{q_val_bpb - raw_q_val_bpb:+.4f} int8_zlib_bytes:{len(quant_blob)} "
                    f"payload:{quant_stats['int8_payload_bytes']} raw_torch:{len(quant_raw)} "
                    f"eval_time:{1000.0 * (time.perf_counter() - qeval_t0):.0f}ms"
                )
            if distributed and should_quant_eval:
                dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        qat_scale = turbo_qat_scale(step)
        qat_lambda = args.turbo_qat_lambda * qat_scale
        aux_scale = sidecar_aux_scale(args, step, elapsed_ms, max_wallclock_ms)
        base_model.set_turbo_qat(qat_scale > 0.0, qat_scale)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        metric_sums = {name: torch.zeros((), device=device) for name in ("ce", "sidecar_pred", "sidecar_sigreg", "sidecar_spherical", "pol_seed", "pol_sparse", "pol_smooth")}
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            operator_codes = base_model.operator_codes_for_input(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, metrics = base_model.loss_terms(x, y, operator_codes=operator_codes, aux_scale=aux_scale)
            if qat_lambda > 0.0:
                loss = loss + qat_lambda * base_model.turbo_regularizer()
            train_loss += loss.detach()
            for name, value in metrics.items():
                metric_sums[name] += value.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        for name in metric_sums:
            metric_sums[name] /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        if optimizer_muon is not None:
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum
        if optimizer_muon_turbo is not None:
            turbo_muon_momentum = ((1 - frac) * args.turbo_qat_muon_momentum_warmup_start + frac * args.turbo_qat_muon_momentum) if args.turbo_qat else muon_momentum
            for group in optimizer_muon_turbo.param_groups:
                group["momentum"] = turbo_muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"sidecar_aux_scale:{aux_scale:.3f} turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f} "
                f"ce:{metric_sums['ce'].item():.4f} sidecar_pred:{metric_sums['sidecar_pred'].item():.4f} "
                f"sidecar_sigreg:{metric_sums['sidecar_sigreg'].item():.4f} sidecar_spherical:{metric_sums['sidecar_spherical'].item():.4f} "
                f"pol_seed:{metric_sums['pol_seed'].item():.4f} pol_sparse:{metric_sums['pol_sparse'].item():.4f} "
                f"pol_smooth:{metric_sums['pol_smooth'].item():.4f}"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(base_model.state_dict())
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(base.dequantize_state_dict(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_sidecar(
        args, model, base_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
    )
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
