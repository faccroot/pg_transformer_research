from __future__ import annotations

import numpy as np


def aggregate_patch_features(
    token_ids: np.ndarray,
    token_feature_ids: np.ndarray,
    patch_len: int,
    *,
    reduction: str = "first",
) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    feature_lut = np.asarray(token_feature_ids)
    if feature_lut.ndim != 2:
        raise ValueError(f"token_feature_ids must have shape [vocab, features], got {feature_lut.shape}")
    if patch_len <= 0:
        raise ValueError(f"patch_len must be > 0, got {patch_len}")
    if ids.size <= 0:
        return np.zeros((0, feature_lut.shape[1]), dtype=feature_lut.dtype)
    patch_count = (ids.shape[0] + patch_len - 1) // patch_len
    out = np.zeros((patch_count, feature_lut.shape[1]), dtype=np.float32)
    for patch_idx in range(patch_count):
        start = patch_idx * patch_len
        end = min(start + patch_len, ids.shape[0])
        patch_features = feature_lut[ids[start:end]]
        if patch_features.shape[0] <= 0:
            continue
        if reduction == "first":
            out[patch_idx] = np.asarray(patch_features[0], dtype=np.float32)
        elif reduction == "any":
            out[patch_idx] = np.asarray(np.any(patch_features > 0, axis=0), dtype=np.float32)
        else:
            raise ValueError(f"unsupported reduction {reduction!r}")
    return out


def segment_lengths_from_ids(segment_ids: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    seg = np.asarray(segment_ids, dtype=np.int32).reshape(-1)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if valid.shape[0] != seg.shape[0]:
            raise ValueError(f"valid mask length {valid.shape[0]} does not match segment_ids length {seg.shape[0]}")
        seg = seg[valid]
    if seg.size <= 0:
        return np.zeros((0,), dtype=np.int32)
    starts = np.flatnonzero(np.concatenate([np.array([True], dtype=np.bool_), seg[1:] != seg[:-1]]))
    ends = np.concatenate([starts[1:], np.array([seg.shape[0]], dtype=np.int64)])
    return np.asarray(ends - starts, dtype=np.int32)


def summarize_segment_lengths(lengths: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(lengths, dtype=np.int32).reshape(-1)
    if arr.size <= 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": int(arr.max()),
    }


def summarize_boundary_alignment(
    boundary_flags: np.ndarray,
    valid_patch_mask: np.ndarray,
    patch_feature_matrix: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    *,
    threshold_boundary_flags: np.ndarray | None = None,
    periodic_boundary_flags: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    exclude_first_patch: bool = True,
) -> dict[str, object]:
    boundary = np.asarray(boundary_flags, dtype=np.bool_).reshape(-1)
    valid = np.asarray(valid_patch_mask, dtype=np.bool_).reshape(-1)
    features = np.asarray(patch_feature_matrix, dtype=np.float32)
    names = tuple(feature_names)
    if features.ndim != 2:
        raise ValueError(f"patch_feature_matrix must have shape [patches, features], got {features.shape}")
    if boundary.shape[0] != valid.shape[0] or features.shape[0] != valid.shape[0]:
        raise ValueError(
            f"patch shapes do not match: boundary={boundary.shape[0]} valid={valid.shape[0]} features={features.shape[0]}"
        )
    if features.shape[1] != len(names):
        raise ValueError(f"feature_names length {len(names)} does not match feature width {features.shape[1]}")

    eval_mask = valid.copy()
    if exclude_first_patch:
        valid_positions = np.flatnonzero(valid)
        if valid_positions.size > 0:
            eval_mask[valid_positions[0]] = False
    boundary_eval = boundary & eval_mask
    nonboundary_eval = (~boundary) & eval_mask

    threshold = None if threshold_boundary_flags is None else np.asarray(threshold_boundary_flags, dtype=np.bool_).reshape(-1)
    periodic = None if periodic_boundary_flags is None else np.asarray(periodic_boundary_flags, dtype=np.bool_).reshape(-1)
    flux_arr = None if flux is None else np.asarray(flux, dtype=np.float32).reshape(-1)

    def rate(mask: np.ndarray, col: np.ndarray) -> float | None:
        count = int(mask.sum())
        if count <= 0:
            return None
        return float(col[mask].mean())

    feature_rows: dict[str, object] = {}
    for idx, name in enumerate(names):
        col = features[:, idx] > 0.0
        overall_rate = rate(eval_mask, col.astype(np.float32))
        boundary_rate = rate(boundary_eval, col.astype(np.float32))
        nonboundary_rate = rate(nonboundary_eval, col.astype(np.float32))
        enrichment = None
        if overall_rate is not None and overall_rate > 0.0 and boundary_rate is not None:
            enrichment = float(boundary_rate / overall_rate)
        feature_rows[name] = {
            "overall_rate": overall_rate,
            "boundary_rate": boundary_rate,
            "nonboundary_rate": nonboundary_rate,
            "boundary_minus_overall": None if boundary_rate is None or overall_rate is None else float(boundary_rate - overall_rate),
            "enrichment": enrichment,
        }

    source_rows: dict[str, object] = {}
    if threshold is not None and periodic is not None:
        source_rows = {
            "threshold_fraction_of_boundaries": None if int(boundary_eval.sum()) <= 0 else float((threshold & boundary_eval).sum() / boundary_eval.sum()),
            "periodic_fraction_of_boundaries": None if int(boundary_eval.sum()) <= 0 else float((periodic & boundary_eval).sum() / boundary_eval.sum()),
            "threshold_only_fraction_of_boundaries": None if int(boundary_eval.sum()) <= 0 else float(((threshold & ~periodic) & boundary_eval).sum() / boundary_eval.sum()),
            "periodic_only_fraction_of_boundaries": None if int(boundary_eval.sum()) <= 0 else float(((periodic & ~threshold) & boundary_eval).sum() / boundary_eval.sum()),
            "both_fraction_of_boundaries": None if int(boundary_eval.sum()) <= 0 else float(((periodic & threshold) & boundary_eval).sum() / boundary_eval.sum()),
        }

    flux_rows: dict[str, object] = {}
    if flux_arr is not None:
        flux_rows = {
            "overall_mean": None if int(eval_mask.sum()) <= 0 else float(flux_arr[eval_mask].mean()),
            "boundary_mean": None if int(boundary_eval.sum()) <= 0 else float(flux_arr[boundary_eval].mean()),
            "nonboundary_mean": None if int(nonboundary_eval.sum()) <= 0 else float(flux_arr[nonboundary_eval].mean()),
            "overall_p90": None if int(eval_mask.sum()) <= 0 else float(np.quantile(flux_arr[eval_mask], 0.90)),
        }

    return {
        "patches_analyzed": int(eval_mask.sum()),
        "boundaries_analyzed": int(boundary_eval.sum()),
        "boundary_rate": None if int(eval_mask.sum()) <= 0 else float(boundary_eval.sum() / eval_mask.sum()),
        "feature_alignment": feature_rows,
        "boundary_sources": source_rows,
        "flux": flux_rows,
    }
