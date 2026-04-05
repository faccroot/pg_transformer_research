#!/usr/bin/env python3
"""
GPT with an exact nested chain-rule token head.

Instead of predicting the 1024-token vocabulary in one flat softmax, this
trainer factors the next-token distribution through a hierarchy of nested token
groups. The default layout is:

    32 -> 128 -> 256 -> 512 -> 1024

which yields an exact chain-rule factorization of the full token probability
while keeping the head compute tiny.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm

import train_gpt_mlx as base


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError(f"Expected a non-empty comma-separated integer list, got {raw!r}")
    return parts


class Hyperparameters(base.Hyperparameters):
    chainrule_level_sizes: str = os.environ.get("CHAINRULE_LEVEL_SIZES", "32,128,256,512,1024")
    chainrule_hidden_dims: str = os.environ.get("CHAINRULE_HIDDEN_DIMS", "32,128,256,512")


class GPTChainRule(base.GPT):
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
        chainrule_level_sizes: tuple[int, ...],
        chainrule_hidden_dims: tuple[int, ...],
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
    ):
        if tie_embeddings:
            raise ValueError("Chain-rule trainer currently requires TIE_EMBEDDINGS=0")
        if len(chainrule_level_sizes) < 2:
            raise ValueError(f"Need at least two chain-rule levels, got {chainrule_level_sizes}")
        if chainrule_level_sizes[-1] != vocab_size:
            raise ValueError(
                f"CHAINRULE_LEVEL_SIZES must end at vocab_size={vocab_size}, got {chainrule_level_sizes[-1]}"
            )
        if len(chainrule_hidden_dims) != len(chainrule_level_sizes) - 1:
            raise ValueError(
                "CHAINRULE_HIDDEN_DIMS must provide one hidden width per non-final chain level, "
                f"got hidden={chainrule_hidden_dims} levels={chainrule_level_sizes}"
            )
        for prev, curr in zip(chainrule_level_sizes, chainrule_level_sizes[1:]):
            if curr <= prev or curr % prev != 0:
                raise ValueError(
                    "CHAINRULE_LEVEL_SIZES must be strictly increasing and nested, "
                    f"got {chainrule_level_sizes}"
                )

        super().__init__(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_layer_templates=num_layer_templates,
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            mlp_leaky_slope=mlp_leaky_slope,
            tie_embeddings=False,
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

        self._chainrule_level_sizes = chainrule_level_sizes
        self._chainrule_hidden_dims = chainrule_hidden_dims
        self._chainrule_branch_sizes = tuple(
            curr // prev for prev, curr in zip(chainrule_level_sizes, chainrule_level_sizes[1:])
        )

        self.chainrule_stage_projs: list[nn.Module] = []
        self.chainrule_stage_heads: list[nn.Module] = []
        stage_input_dim = dim
        for idx, hidden_dim in enumerate(chainrule_hidden_dims):
            proj = base.CastedLinear(stage_input_dim, hidden_dim)
            proj.weight = (
                mx.random.normal(proj.weight.shape, dtype=mx.float32) * (stage_input_dim ** -0.5)
            ).astype(mx.float32)
            head = base.CastedLinear(hidden_dim, chainrule_level_sizes[idx])
            head.weight = (
                mx.random.normal(head.weight.shape, dtype=mx.float32) * tied_embed_init_std
            ).astype(mx.float32)
            self.chainrule_stage_projs.append(proj)
            self.chainrule_stage_heads.append(head)
            stage_input_dim += hidden_dim

        self.lm_head = base.make_turbo_linear("lm_head.weight", chainrule_hidden_dims[-1], vocab_size)
        self.lm_head.weight = (
            mx.random.normal(self.lm_head.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(mx.float32)

    def _stage_hidden_from_flat_hidden(self, x: mx.array) -> list[mx.array]:
        features = [x.astype(base.COMPUTE_DTYPE)]
        stage_hidden: list[mx.array] = []
        for proj in self.chainrule_stage_projs:
            stage_input = features[0] if len(features) == 1 else mx.concatenate(features, axis=-1)
            z = nn.silu(proj(stage_input)).astype(base.COMPUTE_DTYPE)
            stage_hidden.append(z)
            features.append(z)
        return stage_hidden

    def _stage_logits_from_flat_hidden(self, x: mx.array) -> list[mx.array]:
        stage_hidden = self._stage_hidden_from_flat_hidden(x)
        stage_logits = [self.softcap(head(z)) for head, z in zip(self.chainrule_stage_heads, stage_hidden)]
        stage_logits.append(self.softcap(self.lm_head(stage_hidden[-1])))
        return stage_logits

    def _joint_log_probs_from_flat_hidden(self, x: mx.array) -> mx.array:
        stage_logits = self._stage_logits_from_flat_hidden(x)
        joint_log_probs: mx.array | None = None
        for idx, logits in enumerate(stage_logits):
            logits32 = logits.astype(mx.float32)
            if idx == 0:
                joint_log_probs = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
                continue
            prev_size = self._chainrule_level_sizes[idx - 1]
            curr_size = self._chainrule_level_sizes[idx]
            branch = curr_size // prev_size
            cond_logits = logits32.reshape(logits32.shape[0], prev_size, branch)
            cond_log_probs = cond_logits - mx.logsumexp(cond_logits, axis=-1, keepdims=True)
            assert joint_log_probs is not None
            joint_log_probs = (
                joint_log_probs.reshape(joint_log_probs.shape[0], prev_size, 1) + cond_log_probs
            ).reshape(joint_log_probs.shape[0], curr_size)
        assert joint_log_probs is not None
        return joint_log_probs

    def forward_logits(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        hidden = self(input_ids, operator_codes=operator_codes)
        flat = hidden.reshape(-1, hidden.shape[-1])
        log_probs = self._joint_log_probs_from_flat_hidden(flat)
        return log_probs.reshape(input_ids.shape[0], input_ids.shape[1], self._chainrule_level_sizes[-1])

    def token_nll_from_hidden(
        self,
        hidden: mx.array,
        target_ids: mx.array,
        *,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        x = hidden.reshape(-1, hidden.shape[-1])
        y = target_ids.reshape(-1)
        flat_token_weights = token_weights.reshape(-1) if token_weights is not None else None
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            log_probs = self._joint_log_probs_from_flat_hidden(x)
            rows = mx.arange(y.shape[0], dtype=mx.int32)
            nll = -log_probs[rows, y].astype(mx.float32)
            if focal_loss_weight > 0.0:
                pt = mx.exp(-nll)
                hard_weight = mx.power(mx.maximum(1.0 - pt, mx.array(1e-6, dtype=mx.float32)), focal_gamma).astype(mx.float32)
                hard_weight = hard_weight / mx.maximum(mx.mean(hard_weight), mx.array(1e-6, dtype=mx.float32))
                if focal_max_multiplier > 0.0:
                    hard_weight = mx.minimum(hard_weight, mx.array(focal_max_multiplier, dtype=mx.float32))
                blend = (1.0 - focal_loss_weight) + focal_loss_weight * hard_weight
                nll = nll * blend.astype(mx.float32)
            if flat_token_weights is not None:
                weights = flat_token_weights.astype(mx.float32)
                weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-6, dtype=mx.float32))
                nll = nll * weights
            return nll.reshape(target_ids.shape)

        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            log_probs = self._joint_log_probs_from_flat_hidden(x[s:e])
            rows = mx.arange(e - s, dtype=mx.int32)
            nll = -log_probs[rows, y[s:e]].astype(mx.float32)
            if focal_loss_weight > 0.0:
                pt = mx.exp(-nll)
                hard_weight = mx.power(mx.maximum(1.0 - pt, mx.array(1e-6, dtype=mx.float32)), focal_gamma).astype(mx.float32)
                hard_weight = hard_weight / mx.maximum(mx.mean(hard_weight), mx.array(1e-6, dtype=mx.float32))
                if focal_max_multiplier > 0.0:
                    hard_weight = mx.minimum(hard_weight, mx.array(focal_max_multiplier, dtype=mx.float32))
                blend = (1.0 - focal_loss_weight) + focal_loss_weight * hard_weight
                nll = nll * blend.astype(mx.float32)
            if flat_token_weights is not None:
                weights = flat_token_weights[s:e].astype(mx.float32)
                weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-6, dtype=mx.float32))
                nll = nll * weights
            parts.append(nll)
        return mx.concatenate(parts, axis=0).reshape(target_ids.shape)

    def distill_kl_from_hidden(
        self,
        hidden: mx.array,
        teacher_logits: mx.array,
        *,
        temperature: float = 1.0,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        x = hidden.reshape(-1, hidden.shape[-1])
        teacher_flat = mx.stop_gradient(teacher_logits.reshape(-1, teacher_logits.shape[-1]).astype(mx.float32))
        flat_token_weights = token_weights.reshape(-1).astype(mx.float32) if token_weights is not None else None
        temp = max(float(temperature), 1e-6)
        temp_arr = mx.array(temp, dtype=mx.float32)
        scale = temp * temp

        def chunk_kl(student_log_probs_flat: mx.array, teacher_logits_flat: mx.array, weights_flat: mx.array | None) -> mx.array:
            student_scaled = student_log_probs_flat.astype(mx.float32) / temp_arr
            teacher_scaled = teacher_logits_flat.astype(mx.float32) / temp_arr
            student_log_probs = student_scaled - mx.logsumexp(student_scaled, axis=-1, keepdims=True)
            teacher_log_probs = teacher_scaled - mx.logsumexp(teacher_scaled, axis=-1, keepdims=True)
            teacher_probs = mx.exp(teacher_log_probs)
            per_token_kl = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1) * scale
            if weights_flat is None:
                return mx.mean(per_token_kl.astype(mx.float32))
            weight_sum = mx.sum(weights_flat.astype(mx.float32))
            return mx.sum(per_token_kl.astype(mx.float32) * weights_flat.astype(mx.float32)) / mx.maximum(
                weight_sum,
                mx.array(1e-6, dtype=mx.float32),
            )

        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            student_log_probs = self._joint_log_probs_from_flat_hidden(x)
            return chunk_kl(student_log_probs, teacher_flat, flat_token_weights)

        weighted_sum = mx.array(0.0, dtype=mx.float32)
        total_count = 0
        total_weight = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            student_log_probs = self._joint_log_probs_from_flat_hidden(x[s:e])
            chunk_weights = flat_token_weights[s:e] if flat_token_weights is not None else None
            chunk_value = chunk_kl(student_log_probs, teacher_flat[s:e], chunk_weights)
            if chunk_weights is None:
                chunk_count = e - s
                weighted_sum = weighted_sum + chunk_value.astype(mx.float32) * chunk_count
                total_count += chunk_count
            else:
                chunk_weight_sum = mx.sum(chunk_weights.astype(mx.float32))
                weighted_sum = weighted_sum + chunk_value.astype(mx.float32) * chunk_weight_sum
                total_weight = total_weight + chunk_weight_sum
        if flat_token_weights is None:
            return weighted_sum / max(total_count, 1)
        return weighted_sum / mx.maximum(total_weight, mx.array(1e-6, dtype=mx.float32))


def make_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTChainRule:
    return GPTChainRule(
        **base.gpt_kwargs_from_args(args, sp),
        chainrule_level_sizes=_parse_int_tuple(args.chainrule_level_sizes),
        chainrule_hidden_dims=_parse_int_tuple(args.chainrule_hidden_dims),
    )


def exportable_flat_state(model: nn.Module) -> dict[str, mx.array]:
    flat: dict[str, mx.array] = {}
    for k, v in base.tree_flatten(model.state):
        if any(part.startswith("_") for part in k.split(".")):
            continue
        if any(pattern in k for pattern in base.SERIALIZATION_SKIP_NAME_PATTERNS):
            continue
        if not (hasattr(v, "shape") and hasattr(v, "dtype") and hasattr(v, "astype")):
            continue
        flat[k] = v
    return flat


def main() -> None:
    base.Hyperparameters = Hyperparameters
    base.GPT = GPTChainRule
    base.make_gpt = make_gpt
    base.exportable_flat_state = exportable_flat_state
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
