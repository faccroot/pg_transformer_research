from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ResidualNoveltyWeightingConfig:
    enabled: bool = False
    min_scale: float = 0.75
    max_scale: float = 1.25
    norm_epsilon: float = 1.0e-6
    ema_decay: float = 0.0


@dataclass(frozen=True)
class ResidualErrorPriorConfig:
    enabled: bool = False
    weight: float = 0.0
    bottleneck_dim: int = 64
    cosine_weight: float = 0.5
    norm_epsilon: float = 1.0e-6
    target_mode: str = "expected"


def argmax_residual_novelty_weights_from_ids(
    predicted_ids: mx.array,
    target_ids: mx.array,
    embedding_table: mx.array,
    *,
    min_scale: float,
    max_scale: float,
    norm_epsilon: float = 1.0e-6,
    default_scale: float = 1.0,
    ema_decay: float = 0.0,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    if max_scale < min_scale:
        raise ValueError(f"max_scale must be >= min_scale, got {max_scale} < {min_scale}")
    if not (0.0 <= float(ema_decay) < 1.0):
        raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
    pred = predicted_ids.astype(mx.int32)
    target = target_ids.astype(mx.int32)
    if pred.shape != target.shape:
        raise ValueError(f"predicted_ids shape {pred.shape!r} does not match target_ids {target.shape!r}")
    emb = embedding_table.astype(mx.float32)
    actual_embed = emb[target]
    pred_embed = emb[pred]
    residual = actual_embed - pred_embed
    eps_arr = mx.array(float(norm_epsilon), dtype=mx.float32)
    default_arr = mx.array(float(default_scale), dtype=mx.float32)
    norms = mx.linalg.norm(residual.astype(mx.float32), axis=-1, keepdims=True)
    normed = residual.astype(mx.float32) / mx.maximum(norms, eps_arr)
    if float(ema_decay) <= 0.0:
        history_normed = mx.concatenate([mx.zeros_like(normed[:, :1, :]), normed[:, :-1, :]], axis=1)
        zero_valid = mx.zeros(norms[:, :1, 0].shape, dtype=mx.bool_)
        history_valid = mx.concatenate(
            [zero_valid, mx.squeeze(norms[:, :-1, :], axis=-1) > float(norm_epsilon)],
            axis=1,
        )
    else:
        history_states: list[mx.array] = []
        history_valid_states: list[mx.array] = []
        ema_state = mx.zeros_like(normed[:, 0, :])
        valid_state = mx.zeros((normed.shape[0],), dtype=mx.bool_)
        decay = mx.array(float(ema_decay), dtype=mx.float32)
        one_minus_decay = mx.array(1.0 - float(ema_decay), dtype=mx.float32)
        current_valid = mx.squeeze(norms, axis=-1) > float(norm_epsilon)
        for t in range(int(normed.shape[1])):
            history_states.append(ema_state)
            history_valid_states.append(valid_state)
            token_valid = current_valid[:, t]
            token_valid_f = token_valid.astype(mx.float32)[..., None]
            updated = decay * ema_state + one_minus_decay * normed[:, t, :] * token_valid_f
            ema_state = mx.where(token_valid[..., None], updated, ema_state)
            valid_state = valid_state | token_valid
        history_stack = mx.stack(history_states, axis=1).astype(mx.float32)
        history_norms = mx.linalg.norm(history_stack, axis=-1, keepdims=True)
        history_normed = history_stack / mx.maximum(history_norms, eps_arr)
        history_valid = mx.stack(history_valid_states, axis=1)
    cosine = mx.clip(mx.sum(normed * history_normed, axis=-1).astype(mx.float32), -1.0, 1.0)
    novelty = 0.5 * (1.0 - cosine)
    valid = (mx.squeeze(norms, axis=-1) > float(norm_epsilon)) & history_valid
    scale_min = mx.array(float(min_scale), dtype=mx.float32)
    scale_span = mx.array(float(max_scale - min_scale), dtype=mx.float32)
    weights = mx.where(valid, scale_min + novelty * scale_span, default_arr)
    valid_f = valid.astype(mx.float32)
    valid_count = mx.sum(valid_f)
    denom = mx.maximum(valid_count, mx.array(1.0, dtype=mx.float32))
    mean_similarity = mx.sum(cosine * valid_f) / denom
    mean_novelty = mx.sum(novelty * valid_f) / denom
    valid_fraction = mx.mean(valid_f)
    mean_weight = mx.mean(weights.astype(mx.float32))
    return (
        weights.astype(mx.float32),
        mean_similarity.astype(mx.float32),
        mean_novelty.astype(mx.float32),
        mean_weight.astype(mx.float32),
        valid_fraction.astype(mx.float32),
    )


def residual_prediction_alignment_loss(
    predicted_residual: mx.array,
    target_residual: mx.array,
    *,
    token_weights: mx.array | None = None,
    cosine_weight: float = 0.5,
    norm_epsilon: float = 1.0e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    pred = predicted_residual.astype(mx.float32)
    target = mx.stop_gradient(target_residual.astype(mx.float32))
    if pred.shape != target.shape:
        raise ValueError(f"predicted_residual shape {pred.shape!r} does not match target_residual {target.shape!r}")
    diff_sq = mx.sum(mx.square(pred - target), axis=-1)
    pred_norm = mx.linalg.norm(pred, axis=-1)
    target_norm = mx.linalg.norm(target, axis=-1)
    eps_arr = mx.array(float(norm_epsilon), dtype=mx.float32)
    cosine = mx.sum(
        pred / mx.maximum(pred_norm[..., None], eps_arr)
        * target / mx.maximum(target_norm[..., None], eps_arr),
        axis=-1,
    ).astype(mx.float32)
    cosine = mx.clip(cosine, -1.0, 1.0)
    cosine_loss = 1.0 - cosine
    if token_weights is None:
        mse = mx.mean(diff_sq.astype(mx.float32))
        mean_cosine = mx.mean(cosine.astype(mx.float32))
    else:
        weights = token_weights.astype(mx.float32)
        denom = mx.maximum(mx.sum(weights), mx.array(1.0e-6, dtype=mx.float32))
        mse = mx.sum(diff_sq.astype(mx.float32) * weights) / denom
        mean_cosine = mx.sum(cosine.astype(mx.float32) * weights) / denom
    total = (1.0 - float(cosine_weight)) * mse + float(cosine_weight) * (1.0 - mean_cosine)
    return total.astype(mx.float32), mse.astype(mx.float32), mean_cosine.astype(mx.float32)
