#!/usr/bin/env python3
"""
GPT with a stripped mid-stack multi-horizon early-exit auxiliary.

The base next-token objective is unchanged. During training, a captured hidden
state from an intermediate layer is supervised with lightweight heads to
predict token+1 / token+2 / token+3 style horizons. The auxiliary heads are
training-only and excluded from export.
"""
from __future__ import annotations

import os

import mlx.core as mx
import sentencepiece as spm

import train_gpt_mlx as base
from early_exit_aux import horizon_shift, parse_horizons


class Hyperparameters(base.Hyperparameters):
    early_exit_layer_index: int = int(os.environ.get("EARLY_EXIT_LAYER_INDEX", "-1"))
    early_exit_horizons: str = os.environ.get("EARLY_EXIT_HORIZONS", "1,2,3")
    early_exit_aux_weight: float = float(os.environ.get("EARLY_EXIT_AUX_WEIGHT", "0.1"))
    early_exit_head_init_std: float = float(os.environ.get("EARLY_EXIT_HEAD_INIT_STD", "0.005"))


def resolve_early_exit_layer_index(requested_layer: int, num_layers: int) -> int:
    if requested_layer < 0:
        return max(num_layers // 2 - 1, 0)
    if requested_layer >= num_layers:
        raise ValueError(f"EARLY_EXIT_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


class GPTEarlyExit(base.GPT):
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
        early_exit_layer_index: int,
        early_exit_horizons: tuple[int, ...],
        early_exit_aux_weight: float,
        early_exit_head_init_std: float,
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
        self._early_exit_layer_index = resolve_early_exit_layer_index(early_exit_layer_index, num_layers)
        self._early_exit_horizons = tuple(int(value) for value in early_exit_horizons)
        self._early_exit_aux_weight = float(early_exit_aux_weight)
        self.early_exit_heads = [
            base.CastedLinear(dim, vocab_size)
            for _ in self._early_exit_horizons
        ]
        for head in self.early_exit_heads:
            head.weight = (
                mx.random.normal(head.weight.shape, dtype=mx.float32) * early_exit_head_init_std
            ).astype(mx.float32)

    def _early_exit_logits_from_hidden(
        self,
        hidden: mx.array,
        head_idx: int,
    ) -> mx.array:
        x = self.final_norm(hidden.astype(base.COMPUTE_DTYPE)).reshape(-1, hidden.shape[-1])
        head = self.early_exit_heads[head_idx]
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            return self.softcap(head(x))
        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            parts.append(self.softcap(head(x[s:e])))
        return mx.concatenate(parts, axis=0)

    def _early_exit_nll_for_horizon(
        self,
        captured_hidden: mx.array,
        target_ids: mx.array,
        horizon: int,
        head_idx: int,
        *,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        shift = horizon_shift(horizon)
        if shift >= int(target_ids.shape[1]):
            return mx.zeros((target_ids.shape[0], 0), dtype=mx.float32)
        hidden_view = captured_hidden[:, : int(target_ids.shape[1]) - shift, :]
        target_view = target_ids[:, shift:]
        weight_view = token_weights[:, shift:] if token_weights is not None else None
        logits = self._early_exit_logits_from_hidden(hidden_view, head_idx).reshape(
            target_view.shape[0],
            target_view.shape[1],
            -1,
        )
        return base.token_cross_entropy_with_focal(
            logits.reshape(-1, logits.shape[-1]),
            target_view.reshape(-1),
            focal_loss_weight=0.0,
            focal_gamma=2.0,
            focal_max_multiplier=0.0,
            token_weights=weight_view.reshape(-1) if weight_view is not None else None,
            reduction="none",
        ).reshape(target_view.shape)

    def early_exit_aux_loss(
        self,
        captured_hidden: mx.array,
        target_ids: mx.array,
        *,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        losses: list[mx.array] = []
        for head_idx, horizon in enumerate(self._early_exit_horizons):
            if horizon_shift(horizon) >= int(target_ids.shape[1]):
                continue
            horizon_nll = self._early_exit_nll_for_horizon(
                captured_hidden,
                target_ids,
                horizon,
                head_idx,
                token_weights=token_weights,
            )
            if int(horizon_nll.size) > 0:
                losses.append(mx.mean(horizon_nll.astype(mx.float32)))
        if not losses:
            return mx.array(0.0, dtype=mx.float32)
        return mx.mean(mx.stack(losses).astype(mx.float32))

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
        structural_branching_cfg: base.StructuralBranchingConfig | None = None,
        branch_plans: list[list[base.StructuralBranchPoint]] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        final_hidden, captured, aux = self.forward_hidden_with_aux(
            input_ids,
            capture_layers=(self._early_exit_layer_index,),
            operator_codes=operator_codes,
        )
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
        early_exit_loss = self.early_exit_aux_loss(
            captured[self._early_exit_layer_index],
            target_ids,
            token_weights=combined_token_weights,
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
            + self._early_exit_aux_weight * early_exit_loss
            + seed_weight * seed_loss
            + sparse_weight * sparse_loss
            + smooth_weight * smooth_loss
            + ema_teacher_distill_weight * distill_loss
        )
        return total, ce_loss, seed_loss, sparse_loss, smooth_loss, early_exit_loss


def make_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTEarlyExit:
    return GPTEarlyExit(
        **base.gpt_kwargs_from_args(args, sp),
        early_exit_layer_index=args.early_exit_layer_index,
        early_exit_horizons=parse_horizons(args.early_exit_horizons),
        early_exit_aux_weight=args.early_exit_aux_weight,
        early_exit_head_init_std=args.early_exit_head_init_std,
    )


def exportable_flat_state(model: base.nn.Module) -> dict[str, mx.array]:
    return {
        k: v
        for k, v in base.tree_flatten(model.state)
        if not any(part.startswith("_") for part in k.split("."))
        if not any(pattern in k for pattern in base.SERIALIZATION_SKIP_NAME_PATTERNS)
        if "early_exit_heads" not in k
    }


def main() -> None:
    base.Hyperparameters = Hyperparameters
    base.GPT = GPTEarlyExit
    base.make_gpt = make_gpt
    base.exportable_flat_state = exportable_flat_state
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
