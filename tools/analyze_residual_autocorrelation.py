#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import eval_saved_structural as evs
from residual_autocorrelation import (
    argmax_embedding_residuals,
    cosine_acf,
    detect_regime_segments,
    expected_embedding_residuals,
    scalar_acf,
    transition_window_mask,
)
from text_prosody_features import (
    BOUNDARY_STRENGTH_NAMES,
    TOKEN_CLASS_NAMES,
    bucketize_distances,
    extract_text_prosody_features,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze residual-vector autocorrelation on a saved language model artifact.")
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument(
        "--trainer-module",
        default="",
        help="Optional trainer module, e.g. train_gpt_mlx_jepa_sidecar_ref or train_gpt_mlx_sidecar_canonical.",
    )
    p.add_argument("--label", default="")
    p.add_argument("--result-json", required=True)
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--data-path", default="", help="Optional override for DATA_PATH")
    p.add_argument("--val-max-seqs", type=int, default=None)
    p.add_argument("--eval-seq-len", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=64)
    p.add_argument("--residual-mode", choices=("expected", "argmax", "both"), default="both")
    p.add_argument("--regime-layer", type=int, default=-1, help="-1 uses final hidden state; otherwise capture this layer index.")
    p.add_argument("--regime-cosine-threshold", type=float, default=None)
    p.add_argument("--regime-cosine-quantile", type=float, default=0.05)
    p.add_argument("--regime-min-segment-length", type=int, default=16)
    p.add_argument(
        "--layerwise-layers",
        default="",
        help="Optional comma-separated hidden layers to use for alternate regime segmentations, e.g. 0,2,4,-1.",
    )
    p.add_argument(
        "--transition-window",
        type=int,
        default=0,
        help="If >0, also summarize NLL/residual behavior for the first N positions after detected transitions.",
    )
    p.add_argument(
        "--probe-layers",
        default="-1",
        help="Comma-separated hidden layers for lightweight prosody probes; empty disables probes.",
    )
    p.add_argument("--probe-max-samples", type=int, default=32768)
    p.add_argument("--probe-ridge", type=float, default=1.0)
    p.add_argument("--probe-train-frac", type=float, default=0.7)
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--top-k-transitions", type=int, default=10)
    p.add_argument("--preview-radius", type=int, default=24)
    return p.parse_args()


def load_model_modules(trainer_module_name: str) -> tuple[object, object | None]:
    import train_gpt_mlx as base_mod

    base_mod = importlib.reload(base_mod)
    module_name = str(trainer_module_name or "").strip()
    if not module_name or module_name == "train_gpt_mlx":
        return base_mod, None
    trainer_mod = importlib.import_module(module_name)
    return base_mod, importlib.reload(trainer_mod)


def build_hparams_and_model(base_mod, trainer_mod, sp: spm.SentencePieceProcessor):
    if trainer_mod is None:
        hps = base_mod.Hyperparameters()
        model = base_mod.make_gpt(hps, sp)
        return hps, model
    hps_cls = getattr(trainer_mod, "Hyperparameters", base_mod.Hyperparameters)
    hps = hps_cls()
    if hasattr(trainer_mod, "make_sidecar_gpt"):
        model = trainer_mod.make_sidecar_gpt(hps, sp)
    elif hasattr(trainer_mod, "make_gpt"):
        model = trainer_mod.make_gpt(hps, sp)
    else:
        raise ValueError(f"trainer module {trainer_mod.__name__!r} does not expose make_sidecar_gpt or make_gpt")
    return hps, model


def forward_hidden_for_analysis(model, x: mx.array, *, capture_layers: tuple[int, ...], operator_codes):
    return model.forward_hidden_with_aux(x, capture_layers=capture_layers, operator_codes=operator_codes)


def logits_from_hidden(model, final_hidden: mx.array) -> mx.array:
    logits_proj = (
        final_hidden @ model.tok_emb.weight.astype(final_hidden.dtype).T
        if model.tie_embeddings
        else model.lm_head(final_hidden)
    )
    return model.softcap(logits_proj)


def acf_summary(profile: list[dict[str, float | int]], key: str) -> dict[str, float | int | None]:
    valid = [row for row in profile if int(row["count"]) > 0]
    if not valid:
        return {
            "lags": 0,
            "lag1": None,
            "mean": None,
            "positive_area": 0.0,
            "first_nonpositive_lag": None,
        }
    values = [float(row[key]) for row in valid]
    first_nonpositive = next((int(row["lag"]) for row in valid if float(row[key]) <= 0.0), None)
    return {
        "lags": len(valid),
        "lag1": float(values[0]),
        "mean": float(np.mean(values)),
        "positive_area": float(sum(max(v, 0.0) for v in values)),
        "first_nonpositive_lag": first_nonpositive,
    }


def parse_layer_list(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[int] = []
    for piece in text.split(","):
        item = piece.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def nearest_token_directions(
    vector: np.ndarray,
    embedding_table: np.ndarray,
    sp: spm.SentencePieceProcessor,
    *,
    top_k: int,
) -> dict[str, list[dict[str, float | int | str]]]:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    emb = np.asarray(embedding_table, dtype=np.float32)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm <= 1e-8:
        return {"positive": [], "negative": []}
    emb_norm = np.linalg.norm(emb, axis=1)
    sims = (emb @ vec) / np.clip(emb_norm * vec_norm, 1e-8, None)
    top_pos = np.argsort(-sims)[:top_k]
    top_neg = np.argsort(sims)[:top_k]
    render = lambda idx: {"token_id": int(idx), "piece": sp.id_to_piece(int(idx)), "cosine": float(sims[idx])}
    return {
        "positive": [render(idx) for idx in top_pos],
        "negative": [render(idx) for idx in top_neg],
    }


def summarize_transition_window_values(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if arr.shape[0] != keep.shape[0]:
        raise ValueError(f"value count {arr.shape[0]} does not match mask length {keep.shape[0]}")
    if arr.size <= 0:
        return {"count": 0, "mean": None, "std": None}
    selected = arr[keep]
    return {
        "count": int(selected.size),
        "mean": None if selected.size <= 0 else float(selected.mean()),
        "std": None if selected.size <= 0 else float(selected.std()),
    }


def decode_preview(sp: spm.SentencePieceProcessor, token_ids: np.ndarray, center: int, radius: int) -> str:
    start = max(int(center) - int(radius), 0)
    end = min(int(center) + int(radius), int(token_ids.shape[0]))
    return sp.decode_ids([int(x) for x in token_ids[start:end].tolist()])


def safe_corrcoef(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    a = np.asarray(lhs, dtype=np.float32).reshape(-1)
    b = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if a.shape[0] != b.shape[0] or a.shape[0] <= 1:
        return None
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom <= 1e-8:
        return None
    return float(np.dot(a_c, b_c) / denom)


def summarize_masked_scalar(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if arr.shape[0] != keep.shape[0]:
        raise ValueError(f"value count {arr.shape[0]} does not match mask length {keep.shape[0]}")
    selected = arr[keep]
    return {
        "count": int(selected.size),
        "mean": None if selected.size <= 0 else float(selected.mean()),
        "std": None if selected.size <= 0 else float(selected.std()),
        "sum": None if selected.size <= 0 else float(selected.sum()),
    }


def summarize_grouped_loss(
    nll_flat: np.ndarray,
    label_ids: np.ndarray,
    label_names: tuple[str, ...] | list[str],
    *,
    expected_residuals: np.ndarray | None = None,
    argmax_residuals: np.ndarray | None = None,
) -> dict[str, object]:
    nll = np.asarray(nll_flat, dtype=np.float32).reshape(-1)
    labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
    if nll.shape[0] != labels.shape[0]:
        raise ValueError(f"loss count {nll.shape[0]} does not match label count {labels.shape[0]}")
    total_sum = float(nll.sum()) if nll.size > 0 else 0.0
    expected_norms = None if expected_residuals is None else np.linalg.norm(np.asarray(expected_residuals, dtype=np.float32), axis=1)
    argmax_norms = None if argmax_residuals is None else np.linalg.norm(np.asarray(argmax_residuals, dtype=np.float32), axis=1)
    rows: dict[str, object] = {}
    for idx, name in enumerate(label_names):
        mask = labels == int(idx)
        stats = summarize_masked_scalar(nll, mask)
        loss_sum = float(stats["sum"] or 0.0)
        row: dict[str, object] = {
            "count": int(stats["count"]),
            "fraction": float(mask.mean()) if mask.size > 0 else 0.0,
            "mean_nll": stats["mean"],
            "mean_bits": None if stats["mean"] is None else float(float(stats["mean"]) / np.log(2.0)),
            "total_nll_share": 0.0 if total_sum <= 0.0 else float(loss_sum / total_sum),
        }
        if expected_norms is not None:
            row["mean_expected_residual_norm"] = summarize_masked_scalar(expected_norms, mask)["mean"]
        if argmax_norms is not None:
            row["mean_argmax_residual_norm"] = summarize_masked_scalar(argmax_norms, mask)["mean"]
        rows[str(name)] = row
    return rows


def summarize_boundary_conditioned_loss(
    nll_flat: np.ndarray,
    prev_boundary_ids: np.ndarray,
) -> dict[str, object]:
    nll = np.asarray(nll_flat, dtype=np.float32).reshape(-1)
    prev_boundary = np.asarray(prev_boundary_ids, dtype=np.int32).reshape(-1)
    exact = summarize_grouped_loss(nll, prev_boundary, BOUNDARY_STRENGTH_NAMES)
    cumulative: dict[str, object] = {}
    for idx, name in enumerate(BOUNDARY_STRENGTH_NAMES[1:], start=1):
        mask = prev_boundary >= int(idx)
        stats = summarize_masked_scalar(nll, mask)
        cumulative[f"after_{name}"] = {
            "count": int(stats["count"]),
            "fraction": float(mask.mean()) if mask.size > 0 else 0.0,
            "mean_nll": stats["mean"],
            "mean_bits": None if stats["mean"] is None else float(float(stats["mean"]) / np.log(2.0)),
        }
    return {
        "by_prev_boundary_exact": exact,
        "after_boundary_ge": cumulative,
    }


def summarize_density_quantiles(values: np.ndarray, density: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    dens = np.asarray(density, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dens.shape[0] or arr.size <= 0:
        return {}
    quantiles = np.quantile(dens, [0.25, 0.5, 0.75]).astype(np.float32)
    bins = np.digitize(dens, quantiles, right=True).astype(np.int32)
    names = ("q1_low", "q2_midlow", "q3_midhigh", "q4_high")
    return summarize_grouped_loss(arr, bins, names)


def subsample_probe_data(hidden: np.ndarray, labels: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(hidden, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"hidden rows {x.shape[0]} do not match label rows {y.shape[0]}")
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    idx = np.linspace(0, x.shape[0] - 1, num=max_samples, dtype=np.int32)
    return x[idx], y[idx]


def ridge_probe_classifier(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
    train_frac: float,
    max_samples: int,
) -> dict[str, object]:
    x_full, y_full = subsample_probe_data(hidden, labels, max_samples)
    if x_full.shape[0] <= 8:
        return {"count": int(x_full.shape[0]), "accuracy": None, "baseline_accuracy": None}
    split = int(round(float(np.clip(train_frac, 0.1, 0.9)) * x_full.shape[0]))
    split = min(max(split, 4), x_full.shape[0] - 4)
    x_train = x_full[:split]
    x_test = x_full[split:]
    y_train = y_full[:split]
    y_test = y_full[split:]
    classes = np.unique(y_train)
    if classes.size <= 1:
        return {
            "count": int(x_full.shape[0]),
            "train_count": int(x_train.shape[0]),
            "test_count": int(x_test.shape[0]),
            "accuracy": None,
            "baseline_accuracy": None,
            "num_classes": int(classes.size),
        }
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    y_train_idx = np.asarray([class_to_idx[int(v)] for v in y_train.tolist()], dtype=np.int32)
    y_test_idx = np.asarray([class_to_idx.get(int(v), -1) for v in y_test.tolist()], dtype=np.int32)
    valid = y_test_idx >= 0
    x_test = x_test[valid]
    y_test_idx = y_test_idx[valid]
    if x_test.shape[0] <= 0:
        return {
            "count": int(x_full.shape[0]),
            "train_count": int(x_train.shape[0]),
            "test_count": 0,
            "accuracy": None,
            "baseline_accuracy": None,
            "num_classes": int(classes.size),
        }
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.clip(x_train.std(axis=0, keepdims=True), 1e-4, None)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=np.float32)], axis=1)
    target = np.eye(classes.size, dtype=np.float32)[y_train_idx]
    reg = np.eye(x_train.shape[1], dtype=np.float32) * float(max(ridge, 1e-8))
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ target)
    logits = x_test @ weights
    pred = np.argmax(logits, axis=1).astype(np.int32)
    majority = int(np.bincount(y_train_idx, minlength=classes.size).argmax())
    baseline = np.full_like(y_test_idx, majority)
    return {
        "count": int(x_full.shape[0]),
        "train_count": int(x_train.shape[0]),
        "test_count": int(x_test.shape[0]),
        "num_classes": int(classes.size),
        "accuracy": float((pred == y_test_idx).mean()),
        "baseline_accuracy": float((baseline == y_test_idx).mean()),
        "lift": float((pred == y_test_idx).mean() - (baseline == y_test_idx).mean()),
    }


def summarize_prosody_probes(
    hidden_by_layer: dict[int, np.ndarray],
    prosody,
    *,
    ridge: float,
    train_frac: float,
    max_samples: int,
) -> dict[str, object]:
    probe_targets = {
        "inside_quote": prosody.quote_state,
        "sentence_distance_bucket": bucketize_distances(prosody.sentence_distance, (0, 1, 4, 16)),
        "paragraph_distance_bucket": bucketize_distances(prosody.paragraph_distance, (0, 1, 8, 32)),
        "is_noncontent": (prosody.token_class_ids != 0).astype(np.int32),
    }
    out: dict[str, object] = {}
    for layer, hidden in hidden_by_layer.items():
        layer_key = str(layer)
        out[layer_key] = {}
        for name, labels in probe_targets.items():
            out[layer_key][name] = ridge_probe_classifier(
                hidden,
                labels,
                ridge=ridge,
                train_frac=train_frac,
                max_samples=max_samples,
            )
    return out


def summarize_residual_family(
    residuals: np.ndarray,
    embedding_table: np.ndarray,
    sp: spm.SentencePieceProcessor,
    segment_ids: np.ndarray,
    *,
    max_lag: int,
    top_k_tokens: int,
    transition_positions: np.ndarray | None = None,
) -> dict[str, object]:
    resid = np.asarray(residuals, dtype=np.float32)
    overall_mean = resid.mean(axis=0)
    post_transition_mask = np.zeros((resid.shape[0],), dtype=np.bool_)
    if transition_positions is not None and transition_positions.size > 0:
        post_transition_mask[np.asarray(transition_positions, dtype=np.int32)] = True
    post_transition_mean = resid[post_transition_mask].mean(axis=0) if post_transition_mask.any() else np.zeros_like(overall_mean)
    acf_all = cosine_acf(resid, max_lag=max_lag, segment_ids=segment_ids, relation="all")
    acf_within = cosine_acf(resid, max_lag=max_lag, segment_ids=segment_ids, relation="within")
    acf_cross = cosine_acf(resid, max_lag=max_lag, segment_ids=segment_ids, relation="cross")
    return {
        "mean_residual_norm": float(np.linalg.norm(resid, axis=1).mean()),
        "mean_direction": nearest_token_directions(overall_mean, embedding_table, sp, top_k=top_k_tokens),
        "post_transition_mean_direction": nearest_token_directions(post_transition_mean, embedding_table, sp, top_k=top_k_tokens),
        "acf_all": acf_all,
        "acf_within_regime": acf_within,
        "acf_cross_regime": acf_cross,
        "acf_summary": {
            "all": acf_summary(acf_all, "mean_cosine"),
            "within_regime": acf_summary(acf_within, "mean_cosine"),
            "cross_regime": acf_summary(acf_cross, "mean_cosine"),
        },
    }


def summarize_transition_window_residuals(
    residuals: np.ndarray,
    embedding_table: np.ndarray,
    sp: spm.SentencePieceProcessor,
    *,
    mask: np.ndarray,
    top_k_tokens: int,
) -> dict[str, object]:
    resid = np.asarray(residuals, dtype=np.float32)
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if resid.shape[0] != keep.shape[0]:
        raise ValueError(f"residual count {resid.shape[0]} does not match mask length {keep.shape[0]}")
    norms = np.linalg.norm(resid, axis=1)
    selected = resid[keep]
    other = resid[~keep]
    return {
        "transition_window_count": int(selected.shape[0]),
        "outside_window_count": int(other.shape[0]),
        "transition_window_mean_residual_norm": None if selected.shape[0] <= 0 else float(np.linalg.norm(selected, axis=1).mean()),
        "outside_window_mean_residual_norm": None if other.shape[0] <= 0 else float(np.linalg.norm(other, axis=1).mean()),
        "transition_window_mean_direction": nearest_token_directions(
            selected.mean(axis=0) if selected.shape[0] > 0 else np.zeros((resid.shape[1],), dtype=np.float32),
            embedding_table,
            sp,
            top_k=top_k_tokens,
        ),
        "outside_window_mean_direction": nearest_token_directions(
            other.mean(axis=0) if other.shape[0] > 0 else np.zeros((resid.shape[1],), dtype=np.float32),
            embedding_table,
            sp,
            top_k=top_k_tokens,
        ),
        "global_mean_residual_norm": float(norms.mean()) if norms.size > 0 else None,
    }


def main() -> None:
    args = parse_args()
    evs.apply_config_env(Path(args.config_json), args)
    evs.ensure_dataset_ready(args)
    base_mod, trainer_mod = load_model_modules(args.trainer_module)

    artifact_path = Path(args.artifact).expanduser().resolve()
    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(os.environ["TOKENIZER_PATH"]))
    hps, model = build_hparams_and_model(base_mod, trainer_mod, sp)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        hps.val_max_seqs,
    )

    model.set_turbo_qat(False, 0.0)
    flat_state = evs.load_flat_state(artifact_path, base_mod)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()

    embedding_table = np.asarray(mx.stop_gradient(model.tok_emb.weight.astype(mx.float32)), dtype=np.float32)
    layerwise_layers = sorted(set(parse_layer_list(args.layerwise_layers)))
    probe_layers = sorted(set(parse_layer_list(args.probe_layers)))
    flat_actual_ids: list[np.ndarray] = []
    flat_pred_ids: list[np.ndarray] = []
    flat_nll: list[np.ndarray] = []
    residual_expected_rows: list[np.ndarray] = []
    residual_argmax_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    hidden_capture_layers = sorted(
        {
            int(layer)
            for layer in (layerwise_layers + probe_layers)
            if int(layer) >= 0
        }
    )
    capture_hidden_rows: dict[int, list[np.ndarray]] = {layer: [] for layer in hidden_capture_layers}

    positive_capture_layers = {layer for layer in hidden_capture_layers if layer >= 0}
    if int(args.regime_layer) >= 0:
        positive_capture_layers.add(int(args.regime_layer))
    capture_layers = tuple(sorted(positive_capture_layers))

    for batch_idx, (x_np, y_np) in enumerate(evs.iter_eval_batches(base_mod, hps, val_tokens), start=1):
        x = mx.array(x_np, dtype=mx.int32)
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        final_hidden, captured, _aux = forward_hidden_for_analysis(
            model,
            x,
            capture_layers=capture_layers,
            operator_codes=operator_codes,
        )
        logits = logits_from_hidden(model, final_hidden)
        logits_np = np.asarray(mx.stop_gradient(logits.astype(mx.float32)), dtype=np.float32)
        probs_np = np.asarray(mx.softmax(logits.astype(mx.float32), axis=-1), dtype=np.float32)
        nll = base_mod.nn.losses.cross_entropy(
            logits.astype(mx.float32),
            mx.array(y_np, dtype=mx.int32),
            reduction="none",
        ).astype(mx.float32)
        nll_np = np.asarray(mx.stop_gradient(nll), dtype=np.float32).reshape(-1)

        regime_hidden = final_hidden if int(args.regime_layer) < 0 else captured[int(args.regime_layer)]
        hidden_np = np.asarray(mx.stop_gradient(regime_hidden.astype(mx.float32)), dtype=np.float32).reshape(-1, regime_hidden.shape[-1])
        actual_ids = y_np.reshape(-1).astype(np.int32, copy=False)
        pred_ids = np.argmax(logits_np.reshape(-1, logits_np.shape[-1]), axis=1).astype(np.int32, copy=False)

        flat_actual_ids.append(actual_ids)
        flat_pred_ids.append(pred_ids)
        flat_nll.append(nll_np)
        hidden_rows.append(hidden_np)
        final_hidden_np = np.asarray(mx.stop_gradient(final_hidden.astype(mx.float32)), dtype=np.float32).reshape(-1, final_hidden.shape[-1])
        for layer in hidden_capture_layers:
            layer_hidden = np.asarray(
                mx.stop_gradient(captured[int(layer)].astype(mx.float32)),
                dtype=np.float32,
            ).reshape(-1, captured[int(layer)].shape[-1])
            capture_hidden_rows[layer].append(layer_hidden)

        if args.residual_mode in {"expected", "both"}:
            residual_expected_rows.append(
                expected_embedding_residuals(
                    probs_np.reshape(-1, probs_np.shape[-1]),
                    embedding_table,
                    actual_ids,
                )
            )
        if args.residual_mode in {"argmax", "both"}:
            residual_argmax_rows.append(
                argmax_embedding_residuals(
                    logits_np.reshape(-1, logits_np.shape[-1]),
                    embedding_table,
                    actual_ids,
                )
            )
        if args.analysis_max_batches > 0 and batch_idx >= args.analysis_max_batches:
            break

    if not hidden_rows:
        raise SystemExit("No eval batches were processed")

    hidden_mat = np.concatenate(hidden_rows, axis=0)
    actual_ids_flat = np.concatenate(flat_actual_ids, axis=0)
    pred_ids_flat = np.concatenate(flat_pred_ids, axis=0)
    nll_flat = np.concatenate(flat_nll, axis=0)
    expected_residuals = np.concatenate(residual_expected_rows, axis=0) if residual_expected_rows else None
    argmax_residuals = np.concatenate(residual_argmax_rows, axis=0) if residual_argmax_rows else None
    prosody = extract_text_prosody_features(sp, actual_ids_flat)

    regime = detect_regime_segments(
        hidden_mat,
        cosine_threshold=args.regime_cosine_threshold,
        cosine_quantile=args.regime_cosine_quantile,
        min_segment_length=args.regime_min_segment_length,
    )
    segment_ids = np.asarray(regime["segment_ids"], dtype=np.int32)
    transition_positions = np.asarray(regime["transition_positions"], dtype=np.int32)
    consecutive_cosines = np.asarray(regime["consecutive_cosines"], dtype=np.float32)
    transition_mask = transition_window_mask(
        hidden_mat.shape[0],
        transition_positions,
        window=max(int(args.transition_window), 0),
    )

    nll_acf_all = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=segment_ids, relation="all")
    nll_acf_within = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=segment_ids, relation="within")
    nll_acf_cross = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=segment_ids, relation="cross")

    top_transitions: list[dict[str, object]] = []
    if transition_positions.size > 0:
        sorted_positions = transition_positions[np.argsort(consecutive_cosines[transition_positions - 1])]
        for pos in sorted_positions[: args.top_k_transitions].tolist():
            top_transitions.append(
                {
                    "pos": int(pos),
                    "from_segment": int(segment_ids[pos - 1]) if pos > 0 else 0,
                    "to_segment": int(segment_ids[pos]),
                    "hidden_cosine_prev_next": float(consecutive_cosines[pos - 1]),
                    "nll_prev": float(nll_flat[pos - 1]) if pos > 0 else None,
                    "nll_next": float(nll_flat[pos]),
                    "predicted_token_next": int(pred_ids_flat[pos]),
                    "actual_token_next": int(actual_ids_flat[pos]),
                    "predicted_piece_next": sp.id_to_piece(int(pred_ids_flat[pos])),
                    "actual_piece_next": sp.id_to_piece(int(actual_ids_flat[pos])),
                    "preview": decode_preview(sp, actual_ids_flat, pos, args.preview_radius),
                }
            )

    residual_modes: dict[str, object] = {}
    if residual_expected_rows:
        expected_summary = summarize_residual_family(
            expected_residuals,
            embedding_table,
            sp,
            segment_ids,
            max_lag=args.max_lag,
            top_k_tokens=args.top_k_tokens,
            transition_positions=transition_positions,
        )
        if int(args.transition_window) > 0:
            expected_summary["transition_window"] = summarize_transition_window_residuals(
                expected_residuals,
                embedding_table,
                sp,
                mask=transition_mask,
                top_k_tokens=args.top_k_tokens,
            )
        residual_modes["expected_embedding"] = expected_summary
    if residual_argmax_rows:
        argmax_summary = summarize_residual_family(
            argmax_residuals,
            embedding_table,
            sp,
            segment_ids,
            max_lag=args.max_lag,
            top_k_tokens=args.top_k_tokens,
            transition_positions=transition_positions,
        )
        if int(args.transition_window) > 0:
            argmax_summary["transition_window"] = summarize_transition_window_residuals(
                argmax_residuals,
                embedding_table,
                sp,
                mask=transition_mask,
                top_k_tokens=args.top_k_tokens,
            )
        residual_modes["argmax_embedding"] = argmax_summary

    layerwise_regime: dict[str, object] = {}
    for layer in layerwise_layers:
        if layer < 0:
            layer_hidden_mat = hidden_mat
        else:
            layer_hidden_mat = np.concatenate(capture_hidden_rows[layer], axis=0)
        layer_regime = detect_regime_segments(
            layer_hidden_mat,
            cosine_threshold=args.regime_cosine_threshold,
            cosine_quantile=args.regime_cosine_quantile,
            min_segment_length=args.regime_min_segment_length,
        )
        layer_segment_ids = np.asarray(layer_regime["segment_ids"], dtype=np.int32)
        layer_transition_positions = np.asarray(layer_regime["transition_positions"], dtype=np.int32)
        layer_cosines = np.asarray(layer_regime["consecutive_cosines"], dtype=np.float32)
        layer_nll_acf_all = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=layer_segment_ids, relation="all")
        layer_nll_acf_within = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=layer_segment_ids, relation="within")
        layer_nll_acf_cross = scalar_acf(nll_flat, max_lag=args.max_lag, segment_ids=layer_segment_ids, relation="cross")
        layerwise_regime[str(layer)] = {
            "threshold": float(layer_regime["threshold"]),
            "transition_count": int(layer_transition_positions.size),
            "num_segments": int(layer_segment_ids.max()) + 1,
            "mean_segment_length": float(layer_hidden_mat.shape[0] / max(int(layer_segment_ids.max()) + 1, 1)),
            "mean_consecutive_hidden_cosine": float(layer_cosines.mean()) if layer_cosines.size > 0 else 1.0,
            "nll_acf_summary": {
                "all": acf_summary(layer_nll_acf_all, "corr"),
                "within_regime": acf_summary(layer_nll_acf_within, "corr"),
                "cross_regime": acf_summary(layer_nll_acf_cross, "corr"),
            },
        }

    probe_hidden_by_layer: dict[int, np.ndarray] = {}
    for layer in probe_layers:
        if layer < 0:
            probe_hidden_by_layer[layer] = hidden_mat
        elif layer in capture_hidden_rows:
            probe_hidden_by_layer[layer] = np.concatenate(capture_hidden_rows[layer], axis=0)

    result = {
        "label": args.label,
        "artifact": str(artifact_path),
        "config_json": str(Path(args.config_json).expanduser().resolve()),
        "trainer_module": str(args.trainer_module or "train_gpt_mlx"),
        "analysis_params": {
            "analysis_max_batches": int(args.analysis_max_batches),
            "max_lag": int(args.max_lag),
            "residual_mode": str(args.residual_mode),
            "regime_layer": int(args.regime_layer),
            "regime_cosine_threshold": None if args.regime_cosine_threshold is None else float(args.regime_cosine_threshold),
            "regime_cosine_quantile": float(args.regime_cosine_quantile),
            "regime_min_segment_length": int(args.regime_min_segment_length),
            "layerwise_layers": layerwise_layers,
            "transition_window": int(args.transition_window),
            "probe_layers": probe_layers,
            "probe_max_samples": int(args.probe_max_samples),
            "probe_ridge": float(args.probe_ridge),
            "probe_train_frac": float(args.probe_train_frac),
        },
        "positions_analyzed": int(hidden_mat.shape[0]),
        "state_dim": int(hidden_mat.shape[1]),
        "mean_nll": float(nll_flat.mean()),
        "token_class_loss": summarize_grouped_loss(
            nll_flat,
            prosody.token_class_ids,
            TOKEN_CLASS_NAMES,
            expected_residuals=expected_residuals,
            argmax_residuals=argmax_residuals,
        ),
        "boundary_conditioned": summarize_boundary_conditioned_loss(
            nll_flat,
            prosody.prev_boundary_strength_ids,
        ),
        "quote_conditioned": summarize_grouped_loss(
            nll_flat,
            prosody.quote_state,
            ("outside_quote", "inside_quote"),
            expected_residuals=expected_residuals,
            argmax_residuals=argmax_residuals,
        ),
        "prosody_correlations": {
            "nll_vs_prev_boundary_strength": safe_corrcoef(nll_flat, prosody.prev_boundary_strength_ids),
            "nll_vs_sentence_distance": safe_corrcoef(nll_flat, prosody.sentence_distance),
            "nll_vs_paragraph_distance": safe_corrcoef(nll_flat, prosody.paragraph_distance),
            "nll_vs_recent_punctuation_density": safe_corrcoef(nll_flat, prosody.recent_punctuation_density),
            "nll_vs_recent_noncontent_density": safe_corrcoef(nll_flat, prosody.recent_noncontent_density),
            "nll_by_recent_punctuation_density_quantile": summarize_density_quantiles(
                nll_flat,
                prosody.recent_punctuation_density,
            ),
        },
        "prosody_probes": summarize_prosody_probes(
            probe_hidden_by_layer,
            prosody,
            ridge=float(args.probe_ridge),
            train_frac=float(args.probe_train_frac),
            max_samples=int(args.probe_max_samples),
        )
        if probe_hidden_by_layer
        else {},
        "nll_acf": {
            "all": nll_acf_all,
            "within_regime": nll_acf_within,
            "cross_regime": nll_acf_cross,
            "summary": {
                "all": acf_summary(nll_acf_all, "corr"),
                "within_regime": acf_summary(nll_acf_within, "corr"),
                "cross_regime": acf_summary(nll_acf_cross, "corr"),
            },
        },
        "regime": {
            "threshold": float(regime["threshold"]),
            "transition_count": int(transition_positions.size),
            "num_segments": int(segment_ids.max()) + 1,
            "mean_segment_length": float(hidden_mat.shape[0] / max(int(segment_ids.max()) + 1, 1)),
            "mean_consecutive_hidden_cosine": float(consecutive_cosines.mean()) if consecutive_cosines.size > 0 else 1.0,
            "top_transitions": top_transitions,
        },
        "transition_window": {
            "window": int(args.transition_window),
            "transition_positions_count": int(transition_positions.size),
            "masked_positions": int(transition_mask.sum()),
            "nll": {
                "transition_window": summarize_transition_window_values(nll_flat, transition_mask),
                "outside_window": summarize_transition_window_values(nll_flat, ~transition_mask),
            },
        }
        if int(args.transition_window) > 0
        else None,
        "layerwise_regime": layerwise_regime,
        "residual_modes": residual_modes,
    }

    out_path = Path(args.result_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
