from __future__ import annotations

from typing import Literal

import numpy as np


def normalize_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {arr.shape!r}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def expected_embedding_residuals(
    probs: np.ndarray,
    embedding_table: np.ndarray,
    actual_ids: np.ndarray,
) -> np.ndarray:
    prob_arr = np.asarray(probs, dtype=np.float32)
    embed_arr = np.asarray(embedding_table, dtype=np.float32)
    actual = np.asarray(actual_ids, dtype=np.int32).reshape(-1)
    if prob_arr.ndim != 2:
        raise ValueError(f"expected probs shape [tokens, vocab], got {prob_arr.shape!r}")
    if embed_arr.ndim != 2:
        raise ValueError(f"expected embedding_table shape [vocab, dim], got {embed_arr.shape!r}")
    if prob_arr.shape[1] != embed_arr.shape[0]:
        raise ValueError(f"prob vocab {prob_arr.shape[1]} does not match embedding vocab {embed_arr.shape[0]}")
    if prob_arr.shape[0] != actual.shape[0]:
        raise ValueError(f"token count mismatch: probs={prob_arr.shape[0]} actual={actual.shape[0]}")
    expected = prob_arr @ embed_arr
    actual_embed = embed_arr[actual]
    return actual_embed - expected


def argmax_embedding_residuals(
    logits: np.ndarray,
    embedding_table: np.ndarray,
    actual_ids: np.ndarray,
) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float32)
    embed_arr = np.asarray(embedding_table, dtype=np.float32)
    actual = np.asarray(actual_ids, dtype=np.int32).reshape(-1)
    if logits_arr.ndim != 2:
        raise ValueError(f"expected logits shape [tokens, vocab], got {logits_arr.shape!r}")
    pred = np.argmax(logits_arr, axis=1).astype(np.int32, copy=False)
    return embed_arr[actual] - embed_arr[pred]


def consecutive_hidden_cosines(hidden_states: np.ndarray) -> np.ndarray:
    hidden = normalize_rows(np.asarray(hidden_states, dtype=np.float32))
    if hidden.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float32)
    return np.sum(hidden[:-1] * hidden[1:], axis=1, dtype=np.float32)


def detect_regime_segments(
    hidden_states: np.ndarray,
    *,
    cosine_threshold: float | None = None,
    cosine_quantile: float = 0.05,
    min_segment_length: int = 16,
) -> dict[str, object]:
    hidden = np.asarray(hidden_states, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError(f"expected hidden_states shape [tokens, dim], got {hidden.shape!r}")
    if hidden.shape[0] <= 0:
        raise ValueError("need at least one hidden state")
    if min_segment_length <= 0:
        raise ValueError(f"min_segment_length must be positive, got {min_segment_length}")
    cosines = consecutive_hidden_cosines(hidden)
    if cosines.size <= 0:
        return {
            "segment_ids": np.zeros((hidden.shape[0],), dtype=np.int32),
            "transition_positions": np.zeros((0,), dtype=np.int32),
            "consecutive_cosines": cosines,
            "threshold": float(cosine_threshold if cosine_threshold is not None else 1.0),
        }
    if cosine_threshold is None:
        q = float(np.clip(cosine_quantile, 0.0, 1.0))
        threshold = float(np.quantile(cosines, q))
    else:
        threshold = float(cosine_threshold)
    candidate_positions = np.flatnonzero(cosines < threshold).astype(np.int32, copy=False) + 1
    accepted: list[int] = []
    last = 0
    total = hidden.shape[0]
    for pos in candidate_positions.tolist():
        if pos - last < min_segment_length:
            continue
        if total - pos < min_segment_length:
            continue
        accepted.append(int(pos))
        last = int(pos)
    transition_positions = np.array(accepted, dtype=np.int32)
    segment_ids = np.zeros((hidden.shape[0],), dtype=np.int32)
    seg = 0
    cursor = 0
    for pos in transition_positions.tolist():
        segment_ids[cursor:pos] = seg
        seg += 1
        cursor = pos
    segment_ids[cursor:] = seg
    return {
        "segment_ids": segment_ids,
        "transition_positions": transition_positions,
        "consecutive_cosines": cosines,
        "threshold": threshold,
    }


def transition_window_mask(
    length: int,
    transition_positions: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    if length < 0:
        raise ValueError(f"length must be nonnegative, got {length}")
    if window < 0:
        raise ValueError(f"window must be nonnegative, got {window}")
    mask = np.zeros((int(length),), dtype=np.bool_)
    if window == 0 or length == 0:
        return mask
    positions = np.asarray(transition_positions, dtype=np.int32).reshape(-1)
    for pos in positions.tolist():
        if pos < 0 or pos >= length:
            continue
        mask[pos : min(pos + window, length)] = True
    return mask


def cosine_acf(
    vectors: np.ndarray,
    *,
    max_lag: int,
    segment_ids: np.ndarray | None = None,
    relation: Literal["all", "within", "cross"] = "all",
) -> list[dict[str, float | int]]:
    vecs = normalize_rows(np.asarray(vectors, dtype=np.float32))
    if max_lag <= 0:
        raise ValueError(f"max_lag must be positive, got {max_lag}")
    n = int(vecs.shape[0])
    if n <= 1:
        return []
    segs = None if segment_ids is None else np.asarray(segment_ids, dtype=np.int32).reshape(-1)
    if segs is not None and segs.shape[0] != n:
        raise ValueError(f"segment_ids length {segs.shape[0]} does not match vector count {n}")
    rows: list[dict[str, float | int]] = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        sims = np.sum(vecs[:-lag] * vecs[lag:], axis=1, dtype=np.float32)
        if segs is not None and relation != "all":
            mask = segs[:-lag] == segs[lag:]
            if relation == "cross":
                mask = ~mask
            sims = sims[mask]
        if sims.size <= 0:
            rows.append({"lag": lag, "count": 0, "mean_cosine": 0.0})
        else:
            rows.append(
                {
                    "lag": lag,
                    "count": int(sims.size),
                    "mean_cosine": float(sims.mean()),
                }
            )
    return rows


def scalar_acf(
    values: np.ndarray,
    *,
    max_lag: int,
    segment_ids: np.ndarray | None = None,
    relation: Literal["all", "within", "cross"] = "all",
) -> list[dict[str, float | int]]:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if max_lag <= 0:
        raise ValueError(f"max_lag must be positive, got {max_lag}")
    n = int(vals.shape[0])
    if n <= 1:
        return []
    segs = None if segment_ids is None else np.asarray(segment_ids, dtype=np.int32).reshape(-1)
    if segs is not None and segs.shape[0] != n:
        raise ValueError(f"segment_ids length {segs.shape[0]} does not match value count {n}")
    rows: list[dict[str, float | int]] = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        lhs = vals[:-lag]
        rhs = vals[lag:]
        if segs is not None and relation != "all":
            mask = segs[:-lag] == segs[lag:]
            if relation == "cross":
                mask = ~mask
            lhs = lhs[mask]
            rhs = rhs[mask]
        if lhs.size <= 1 or rhs.size <= 1:
            rows.append({"lag": lag, "count": 0, "corr": 0.0})
            continue
        lhs_c = lhs - lhs.mean()
        rhs_c = rhs - rhs.mean()
        denom = float(np.linalg.norm(lhs_c) * np.linalg.norm(rhs_c))
        corr = 0.0 if denom <= 1e-8 else float(np.dot(lhs_c, rhs_c) / denom)
        rows.append({"lag": lag, "count": int(lhs.size), "corr": corr})
    return rows


def factorize_residual_pca(
    vectors: np.ndarray,
    *,
    max_factors: int,
    center: bool = True,
) -> dict[str, np.ndarray]:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D vectors, got shape {arr.shape!r}")
    n, d = int(arr.shape[0]), int(arr.shape[1])
    if max_factors <= 0 or n <= 0 or d <= 0:
        return {
            "mean": np.zeros((d,), dtype=np.float32),
            "components": np.zeros((0, d), dtype=np.float32),
            "scores": np.zeros((n, 0), dtype=np.float32),
            "explained_variance_ratio": np.zeros((0,), dtype=np.float32),
        }
    mean = arr.mean(axis=0, dtype=np.float32) if center else np.zeros((d,), dtype=np.float32)
    centered = arr - mean[None, :]
    rank = min(int(max_factors), n, d)
    if rank <= 0:
        return {
            "mean": mean.astype(np.float32),
            "components": np.zeros((0, d), dtype=np.float32),
            "scores": np.zeros((n, 0), dtype=np.float32),
            "explained_variance_ratio": np.zeros((0,), dtype=np.float32),
        }
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "mean": mean.astype(np.float32),
            "components": np.zeros((0, d), dtype=np.float32),
            "scores": np.zeros((n, 0), dtype=np.float32),
            "explained_variance_ratio": np.zeros((0,), dtype=np.float32),
        }
    components = vt[:rank].astype(np.float32, copy=False)
    scores = centered @ components.T
    if singular_values.size <= 0:
        explained = np.zeros((0,), dtype=np.float32)
    else:
        variances = (singular_values.astype(np.float32) ** 2) / float(max(n - 1, 1))
        total_var = float(np.sum(variances, dtype=np.float32))
        explained = (
            np.zeros((rank,), dtype=np.float32)
            if total_var <= 1.0e-8
            else (variances[:rank] / total_var).astype(np.float32, copy=False)
        )
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "scores": scores.astype(np.float32, copy=False),
        "explained_variance_ratio": explained,
    }
