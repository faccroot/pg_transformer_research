#!/usr/bin/env python3
"""
GPT with a lightweight structured bottleneck head.

The final token is still predicted with the full 1024-way vocabulary, but the
head must also route through a small bottleneck whose auxiliary token loss
encourages it to capture coarse predictive structure.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm

import train_gpt_mlx as base


class Hyperparameters(base.Hyperparameters):
    bottleneck_dim: int = int(os.environ.get("BOTTLENECK_DIM", "32"))
    bottleneck_aux_weight: float = float(os.environ.get("BOTTLENECK_AUX_WEIGHT", "0.1"))
    bottleneck_up_init_std: float = float(os.environ.get("BOTTLENECK_UP_INIT_STD", "0.001"))


class GPTBottleneck(base.GPT):
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
        bottleneck_dim: int,
        bottleneck_aux_weight: float,
        bottleneck_up_init_std: float,
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
        self._bottleneck_aux_weight = float(bottleneck_aux_weight)
        self.bottleneck_down = base.CastedLinear(dim, bottleneck_dim)
        self.bottleneck_down.weight = (
            mx.random.normal(self.bottleneck_down.weight.shape, dtype=mx.float32) * (dim ** -0.5)
        ).astype(mx.float32)

        self.bottleneck_up = base.CastedLinear(bottleneck_dim, dim)
        self.bottleneck_up.weight = (
            mx.random.normal(self.bottleneck_up.weight.shape, dtype=mx.float32) * bottleneck_up_init_std
        ).astype(mx.float32)

        self.bottleneck_aux_head = base.CastedLinear(bottleneck_dim, vocab_size)
        self.bottleneck_aux_head.weight = (
            mx.random.normal(self.bottleneck_aux_head.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(mx.float32)

    def _condition_hidden_flat(self, x: mx.array) -> tuple[mx.array, mx.array]:
        z = nn.silu(self.bottleneck_down(x.astype(base.COMPUTE_DTYPE))).astype(base.COMPUTE_DTYPE)
        cond = self.bottleneck_up(z).astype(base.COMPUTE_DTYPE)
        return (x.astype(base.COMPUTE_DTYPE) + cond).astype(base.COMPUTE_DTYPE), z

    def forward_logits(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        x = self(input_ids, operator_codes=operator_codes)
        flat = x.reshape(-1, x.shape[-1])
        conditioned, _z = self._condition_hidden_flat(flat)
        logits_proj = conditioned @ self.tok_emb.weight.astype(conditioned.dtype).T if self.tie_embeddings else self.lm_head(conditioned)
        logits = self.softcap(logits_proj)
        return logits.reshape(input_ids.shape[0], input_ids.shape[1], self.tok_emb.weight.shape[0])

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
            conditioned, _ = self._condition_hidden_flat(x)
            logits_proj = conditioned @ self.tok_emb.weight.astype(conditioned.dtype).T if self.tie_embeddings else self.lm_head(conditioned)
            logits = self.softcap(logits_proj)
            return base.token_cross_entropy_with_focal(
                logits,
                y,
                focal_loss_weight=focal_loss_weight,
                focal_gamma=focal_gamma,
                focal_max_multiplier=focal_max_multiplier,
                token_weights=flat_token_weights,
                reduction="none",
            ).reshape(target_ids.shape)

        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            conditioned, _ = self._condition_hidden_flat(x[s:e])
            logits_proj = conditioned @ self.tok_emb.weight.astype(conditioned.dtype).T if self.tie_embeddings else self.lm_head(conditioned)
            logits = self.softcap(logits_proj)
            parts.append(
                base.token_cross_entropy_with_focal(
                    logits,
                    y[s:e],
                    focal_loss_weight=focal_loss_weight,
                    focal_gamma=focal_gamma,
                    focal_max_multiplier=focal_max_multiplier,
                    token_weights=flat_token_weights[s:e] if flat_token_weights is not None else None,
                    reduction="none",
                )
            )
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

        def chunk_kl(student_logits_flat: mx.array, teacher_logits_flat: mx.array, weights_flat: mx.array | None) -> mx.array:
            student_scaled = student_logits_flat.astype(mx.float32) / temp_arr
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
            conditioned, _ = self._condition_hidden_flat(x)
            logits_proj = conditioned @ self.tok_emb.weight.astype(conditioned.dtype).T if self.tie_embeddings else self.lm_head(conditioned)
            logits = self.softcap(logits_proj)
            return chunk_kl(logits, teacher_flat, flat_token_weights)

        weighted_sum = mx.array(0.0, dtype=mx.float32)
        total_count = 0
        total_weight = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            conditioned, _ = self._condition_hidden_flat(x[s:e])
            logits_proj = conditioned @ self.tok_emb.weight.astype(conditioned.dtype).T if self.tie_embeddings else self.lm_head(conditioned)
            logits = self.softcap(logits_proj)
            chunk_weights = flat_token_weights[s:e] if flat_token_weights is not None else None
            chunk_value = chunk_kl(logits, teacher_flat[s:e], chunk_weights)
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

    def bottleneck_aux_nll_from_hidden(self, hidden: mx.array, target_ids: mx.array, *, token_weights: mx.array | None = None) -> mx.array:
        x = hidden.reshape(-1, hidden.shape[-1])
        y = target_ids.reshape(-1)
        flat_token_weights = token_weights.reshape(-1) if token_weights is not None else None
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            _conditioned, z = self._condition_hidden_flat(x)
            logits = self.softcap(self.bottleneck_aux_head(z))
            return base.token_cross_entropy_with_focal(
                logits,
                y,
                focal_loss_weight=0.0,
                focal_gamma=2.0,
                focal_max_multiplier=0.0,
                token_weights=flat_token_weights,
                reduction="none",
            ).reshape(target_ids.shape)

        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            _conditioned, z = self._condition_hidden_flat(x[s:e])
            logits = self.softcap(self.bottleneck_aux_head(z))
            parts.append(
                base.token_cross_entropy_with_focal(
                    logits,
                    y[s:e],
                    focal_loss_weight=0.0,
                    focal_gamma=2.0,
                    focal_max_multiplier=0.0,
                    token_weights=flat_token_weights[s:e] if flat_token_weights is not None else None,
                    reduction="none",
                )
            )
        return mx.concatenate(parts, axis=0).reshape(target_ids.shape)

    def loss_terms(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
        polarity_seed_weight: float | None = None,
        polarity_sparse_weight: float | None = None,
        polarity_smooth_weight: float | None = None,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
        context_delta_config: base.ContextDeltaWeightingConfig | None = None,
        teacher_logits: mx.array | None = None,
        ema_teacher_distill_weight: float = 0.0,
        ema_teacher_temperature: float = 1.0,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        final_hidden, _captured, aux = self.forward_hidden_with_aux(input_ids, operator_codes=operator_codes)
        combined_token_weights = base.merge_token_weights(
            token_weights,
            self.context_delta_token_weights(
                input_ids,
                target_ids,
                final_hidden,
                operator_codes,
                context_delta_config,
            ),
        )
        ce_loss = self.token_ce_from_hidden(
            final_hidden,
            target_ids,
            focal_loss_weight=focal_loss_weight,
            focal_gamma=focal_gamma,
            focal_max_multiplier=focal_max_multiplier,
            token_weights=combined_token_weights,
        )
        bottleneck_loss = mx.mean(
            self.bottleneck_aux_nll_from_hidden(final_hidden, target_ids, token_weights=combined_token_weights)
        )
        seed_weight = self.polarity_seed_weight if polarity_seed_weight is None else polarity_seed_weight
        sparse_weight = self.polarity_sparse_weight if polarity_sparse_weight is None else polarity_sparse_weight
        smooth_weight = self.polarity_smooth_weight if polarity_smooth_weight is None else polarity_smooth_weight
        seed_loss, sparse_loss, smooth_loss = self.polarity_loss_terms_from_aux(aux)
        distill_loss = mx.array(0.0, dtype=mx.float32)
        if teacher_logits is not None and ema_teacher_distill_weight > 0.0:
            distill_loss = self.distill_kl_from_hidden(
                final_hidden,
                teacher_logits,
                temperature=ema_teacher_temperature,
                token_weights=combined_token_weights,
            )
        total = (
            ce_loss
            + self._bottleneck_aux_weight * bottleneck_loss
            + seed_weight * seed_loss
            + sparse_weight * sparse_loss
            + smooth_weight * smooth_loss
            + ema_teacher_distill_weight * distill_loss
        )
        return total, ce_loss, bottleneck_loss, sparse_loss, smooth_loss


def make_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTBottleneck:
    return GPTBottleneck(
        **base.gpt_kwargs_from_args(args, sp),
        bottleneck_dim=args.bottleneck_dim,
        bottleneck_aux_weight=args.bottleneck_aux_weight,
        bottleneck_up_init_std=args.bottleneck_up_init_std,
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
    base.GPT = GPTBottleneck
    base.make_gpt = make_gpt
    base.exportable_flat_state = exportable_flat_state
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
