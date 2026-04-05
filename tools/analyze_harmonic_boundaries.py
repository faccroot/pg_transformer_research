#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import zlib
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

from harmonic_boundary_metrics import (
    aggregate_patch_features,
    segment_lengths_from_ids,
    summarize_boundary_alignment,
    summarize_segment_lengths,
)
from text_prosody_features import build_token_prosody_luts


FEATURE_NAMES = (
    "whitespace_like",
    "quote_like",
    "boundary_clause_plus",
    "boundary_sentence_plus",
    "boundary_paragraph_plus",
    "boundary_section",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze harmonic/segment boundary placement on a saved artifact.")
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument("--trainer-module", default="", help="Optional trainer module; defaults to config metadata")
    p.add_argument("--label", default="")
    p.add_argument("--result-json", required=True)
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--data-path", default="", help="Optional override for DATA_PATH")
    p.add_argument("--val-max-seqs", type=int, default=256)
    p.add_argument("--eval-seq-len", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=16)
    p.add_argument(
        "--patch-feature-reduction",
        choices=("first", "any"),
        default="first",
        help="How to map token prosody features onto patches. `first` best matches patch-start boundary priors.",
    )
    return p.parse_args()


def infer_trainer_module(config_json: Path, override: str) -> str:
    if override.strip():
        return override.strip()
    config = json.loads(config_json.read_text(encoding="utf-8"))
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        return "train_gpt_mlx_harmonic"
    raw = str(metadata.get("trainer_script_source") or metadata.get("trainer_script") or "").strip()
    stem = Path(raw).stem
    return stem or "train_gpt_mlx_harmonic"


def apply_config_env(config_json: Path, args: argparse.Namespace) -> None:
    config = json.loads(config_json.read_text(encoding="utf-8"))
    env = config.get("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"Expected `env` dict in {config_json}")
    for key, value in env.items():
        os.environ[str(key)] = str(value)
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = os.path.expanduser(args.tokenizer_path)
    if args.data_path:
        os.environ["DATA_PATH"] = os.path.expanduser(args.data_path)
    if args.val_max_seqs is not None:
        os.environ["VAL_MAX_SEQS"] = str(int(args.val_max_seqs))
    if args.eval_seq_len is not None:
        os.environ["EVAL_SEQ_LEN"] = str(int(args.eval_seq_len))
    if args.eval_stride is not None:
        os.environ["EVAL_STRIDE"] = str(int(args.eval_stride))
    if args.eval_batch_seqs is not None:
        os.environ["EVAL_BATCH_SEQS"] = str(int(args.eval_batch_seqs))
    os.environ["QUANT_EVAL_MAX_SEQS"] = "0"
    os.environ["VAL_LOSS_EVERY"] = "0"
    os.environ["QUANT_EVAL_EVERY"] = "0"
    os.environ["MLX_COMPILE"] = "0"
    os.environ["TRAIN_SHARDS"] = str(int(args.train_shards))


def ensure_dataset_ready(args: argparse.Namespace) -> None:
    data_path = Path(os.path.expanduser(os.environ["DATA_PATH"]))
    val_probe = data_path / "fineweb_val_000000.bin"
    if val_probe.exists():
        return
    local_cwd_script = Path.cwd() / "cached_challenge_fineweb.py"
    local_peer_script = TOOLS_ROOT / "cached_challenge_fineweb.py"
    repo_script = REPO_ROOT / "data" / "cached_challenge_fineweb.py"
    if local_cwd_script.exists():
        cache_script = local_cwd_script
    elif local_peer_script.exists():
        cache_script = local_peer_script
    else:
        cache_script = repo_script
    subprocess.run(
        [
            "/opt/homebrew/bin/python3",
            str(cache_script),
            "--variant",
            args.cache_variant,
            "--train-shards",
            str(args.train_shards),
        ],
        check=True,
    )


def load_flat_state(artifact_path: Path, base_mod) -> dict[str, object]:
    if artifact_path.suffix == ".z":
        return base_mod.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    if artifact_path.suffix == ".npz":
        npz = base_mod.mx.load(str(artifact_path))
        return dict(npz.items())
    raise ValueError(f"Unsupported artifact type: {artifact_path}")


def iter_eval_batches(base_mod, hps, val_tokens: np.ndarray):
    eval_seq_len = hps.effective_eval_seq_len
    if hps.eval_stride > 0:
        windows = base_mod.build_sliding_eval_windows(val_tokens.size - 1, eval_seq_len, hps.eval_stride)
        batch_seqs = hps.effective_eval_batch_seqs
        for batch_start in range(0, len(windows), batch_seqs):
            batch_windows = windows[batch_start : batch_start + batch_seqs]
            x_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
            y_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
            for i, (window_start, window_len, _, _) in enumerate(batch_windows):
                chunk = val_tokens[window_start : window_start + window_len + 1]
                x_np[i, :window_len] = chunk[:-1]
                y_np[i, :window_len] = chunk[1:]
            yield x_np, y_np
        return

    val_batch_tokens = hps.val_batch_size // hps.grad_accum_steps
    val_batch_seqs = max(val_batch_tokens // eval_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * eval_seq_len
        raw_end = batch_seq_end * eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        yield (
            chunk[:-1].reshape(-1, eval_seq_len).astype(np.int32, copy=False),
            chunk[1:].reshape(-1, eval_seq_len).astype(np.int32, copy=False),
        )


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
    if hasattr(trainer_mod, "make_harmonic_gpt"):
        model = trainer_mod.make_harmonic_gpt(hps, sp)
    elif hasattr(trainer_mod, "make_sidecar_gpt"):
        model = trainer_mod.make_sidecar_gpt(hps, sp)
    elif hasattr(trainer_mod, "make_gpt"):
        model = trainer_mod.make_gpt(hps, sp)
    else:
        raise ValueError(f"trainer module {trainer_mod.__name__!r} does not expose make_harmonic_gpt/make_sidecar_gpt/make_gpt")
    return hps, model


def summarize_scalar(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    keep = np.asarray(mask, dtype=np.bool_)
    if arr.shape != keep.shape:
        raise ValueError(f"values shape {arr.shape} does not match mask shape {keep.shape}")
    if int(keep.sum()) <= 0:
        return None
    return float(arr[keep].mean())


def collect_adjacent_cosines(chord_states: np.ndarray, chord_mask: np.ndarray) -> list[float]:
    states = np.asarray(chord_states, dtype=np.float32)
    mask = np.asarray(chord_mask, dtype=np.bool_)
    if states.ndim != 3 or mask.ndim != 2:
        raise ValueError(f"expected chord states [B,C,D] and mask [B,C], got {states.shape} and {mask.shape}")
    if states.shape[:2] != mask.shape:
        raise ValueError(f"shape mismatch between states {states.shape} and mask {mask.shape}")
    if states.shape[1] <= 1:
        return []
    lhs = states[:, :-1, :]
    rhs = states[:, 1:, :]
    denom = np.linalg.norm(lhs, axis=-1) * np.linalg.norm(rhs, axis=-1)
    denom = np.clip(denom, 1e-8, None)
    cosine = np.sum(lhs * rhs, axis=-1) / denom
    keep = mask[:, :-1] & mask[:, 1:]
    return [float(x) for x in cosine[keep].tolist()]


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact).expanduser().resolve()
    config_json = Path(args.config_json).expanduser().resolve()
    result_json = Path(args.result_json).expanduser().resolve()
    result_json.parent.mkdir(parents=True, exist_ok=True)

    trainer_module_name = infer_trainer_module(config_json, args.trainer_module)
    apply_config_env(config_json, args)
    ensure_dataset_ready(args)

    base_mod, trainer_mod = load_model_modules(trainer_module_name)
    sp = spm.SentencePieceProcessor(model_file=os.environ["TOKENIZER_PATH"])
    hps, model = build_hparams_and_model(base_mod, trainer_mod, sp)
    flat_state = load_flat_state(artifact_path, base_mod)
    model.update(tree_unflatten(list(flat_state.items())))

    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        hps.val_max_seqs,
    )
    prosody_luts = build_token_prosody_luts(sp, extended_binary_features=True)
    feature_index = {name: idx for idx, name in enumerate(prosody_luts.binary_feature_names)}
    selected_cols = [feature_index[name] for name in FEATURE_NAMES]
    selected_token_features = prosody_luts.binary_feature_ids[:, selected_cols]

    analyzed_batches = 0
    analyzed_sequences = 0
    all_patch_counts: list[int] = []
    all_chord_counts: list[float] = []
    all_flux_means: list[float] = []
    all_span_lengths: list[int] = []
    aggregate_alignment: list[dict[str, object]] = []
    all_eval_ce: list[float] = []
    all_eval_jepa_loss: list[float] = []
    all_patch_read_norms: list[float] = []
    all_token_read_norms: list[float] = []
    all_chord_norms: list[float] = []
    all_adjacent_chord_cosines: list[float] = []
    all_read_drop_ce_delta: list[float] = []

    for x_np, y_np in iter_eval_batches(base_mod, hps, val_tokens):
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        (
            loss_total,
            ce_loss_term,
            jepa_loss_term,
            _chord_count_metric,
            _flux_mean_metric,
            _norm_penalty_metric,
            _adjacent_cos_penalty_metric,
        ) = model.loss_terms(x, y, jepa_weight=0.0)
        del loss_total
        all_eval_ce.append(float(ce_loss_term.item()))
        all_eval_jepa_loss.append(float(jepa_loss_term.item()))

        _final_hidden, _captured, aux = model.forward_hidden_with_aux(x)
        required = (
            "harmonic_boundary_flags",
            "harmonic_segment_ids",
            "harmonic_flux",
            "harmonic_periodic_boundary_flags",
            "harmonic_threshold_boundary_flags",
            "harmonic_patch_reads",
            "harmonic_chords",
            "harmonic_chord_mask",
            "harmonic_token_reads",
            "harmonic_chord_counts",
            "harmonic_flux_means",
            "harmonic_patch_len",
        )
        missing = [name for name in required if name not in aux]
        if missing:
            raise ValueError(
                f"artifact/model does not expose harmonic boundary diagnostics keys: {missing}; "
                "run this on train_gpt_mlx_harmonic artifacts after the boundary-debug patch."
            )
        boundary_flags = np.asarray(mx.stop_gradient(aux["harmonic_boundary_flags"]).astype(mx.float32)) > 0.5
        segment_ids = np.asarray(mx.stop_gradient(aux["harmonic_segment_ids"]).astype(mx.int32))
        flux = np.asarray(mx.stop_gradient(aux["harmonic_flux"]).astype(mx.float32))
        periodic_flags = np.asarray(mx.stop_gradient(aux["harmonic_periodic_boundary_flags"]).astype(mx.float32)) > 0.5
        threshold_flags = np.asarray(mx.stop_gradient(aux["harmonic_threshold_boundary_flags"]).astype(mx.float32)) > 0.5
        patch_reads = np.asarray(mx.stop_gradient(aux["harmonic_patch_reads"]).astype(mx.float32))
        chord_states = np.asarray(mx.stop_gradient(aux["harmonic_chords"]).astype(mx.float32))
        chord_mask = np.asarray(mx.stop_gradient(aux["harmonic_chord_mask"]).astype(mx.float32)) > 0.5
        token_reads = np.asarray(mx.stop_gradient(aux["harmonic_token_reads"]).astype(mx.float32))
        chord_counts = np.asarray(mx.stop_gradient(aux["harmonic_chord_counts"]).astype(mx.float32)).reshape(-1)
        flux_means = np.asarray(mx.stop_gradient(aux["harmonic_flux_means"]).astype(mx.float32)).reshape(-1)
        patch_len = int(float(mx.stop_gradient(aux["harmonic_patch_len"]).item()))

        patch_read_norm = np.linalg.norm(patch_reads, axis=-1)
        token_read_norm = np.linalg.norm(token_reads, axis=-1)
        chord_norm = np.linalg.norm(chord_states, axis=-1)
        all_patch_read_norms.extend(float(x) for x in patch_read_norm.reshape(-1).tolist())
        all_token_read_norms.extend(float(x) for x in token_read_norm.reshape(-1).tolist())
        all_chord_norms.extend(float(x) for x in chord_norm[chord_mask].tolist())
        all_adjacent_chord_cosines.extend(collect_adjacent_cosines(chord_states, chord_mask))

        if getattr(model, "harmonic_enable_read", False):
            prev_enable_read = bool(model.harmonic_enable_read)
            try:
                model.harmonic_enable_read = False
                no_read_ce = float(model.ce_loss(x, y).item())
            finally:
                model.harmonic_enable_read = prev_enable_read
            all_read_drop_ce_delta.append(no_read_ce - all_eval_ce[-1])

        for row_idx in range(x_np.shape[0]):
            token_ids = np.asarray(x_np[row_idx], dtype=np.int32)
            patch_features = aggregate_patch_features(
                token_ids,
                selected_token_features,
                patch_len,
                reduction=args.patch_feature_reduction,
            )
            patch_count = patch_features.shape[0]
            valid_patch_mask = np.ones((patch_count,), dtype=np.bool_)
            row_segment_ids = np.asarray(segment_ids[row_idx], dtype=np.int32)[:patch_count]
            span_lengths = segment_lengths_from_ids(row_segment_ids, valid_patch_mask)
            all_span_lengths.extend(int(x) for x in span_lengths.tolist())
            aggregate_alignment.append(
                summarize_boundary_alignment(
                    np.asarray(boundary_flags[row_idx], dtype=np.bool_)[:patch_count],
                    valid_patch_mask,
                    patch_features,
                    FEATURE_NAMES,
                    threshold_boundary_flags=np.asarray(threshold_flags[row_idx], dtype=np.bool_)[:patch_count],
                    periodic_boundary_flags=np.asarray(periodic_flags[row_idx], dtype=np.bool_)[:patch_count],
                    flux=np.asarray(flux[row_idx], dtype=np.float32)[:patch_count],
                    exclude_first_patch=True,
                )
            )
            all_patch_counts.append(int(patch_count))
            analyzed_sequences += 1

        all_chord_counts.extend(float(x) for x in chord_counts.tolist())
        all_flux_means.extend(float(x) for x in flux_means.tolist())
        analyzed_batches += 1
        if args.analysis_max_batches > 0 and analyzed_batches >= args.analysis_max_batches:
            break

    total_patches = sum(int(item.get("patches_analyzed", 0) or 0) for item in aggregate_alignment)
    total_boundaries = sum(int(item.get("boundaries_analyzed", 0) or 0) for item in aggregate_alignment)

    feature_rows: dict[str, dict[str, float | int | None]] = {}
    for name in FEATURE_NAMES:
        overall_rates = []
        boundary_rates = []
        nonboundary_rates = []
        enrichments = []
        for row in aggregate_alignment:
            feature_row = row.get("feature_alignment", {}).get(name, {})
            if feature_row.get("overall_rate") is not None:
                overall_rates.append(float(feature_row["overall_rate"]))
            if feature_row.get("boundary_rate") is not None:
                boundary_rates.append(float(feature_row["boundary_rate"]))
            if feature_row.get("nonboundary_rate") is not None:
                nonboundary_rates.append(float(feature_row["nonboundary_rate"]))
            if feature_row.get("enrichment") is not None:
                enrichments.append(float(feature_row["enrichment"]))
        feature_rows[name] = {
            "overall_rate_mean": None if not overall_rates else float(np.mean(overall_rates)),
            "boundary_rate_mean": None if not boundary_rates else float(np.mean(boundary_rates)),
            "nonboundary_rate_mean": None if not nonboundary_rates else float(np.mean(nonboundary_rates)),
            "enrichment_mean": None if not enrichments else float(np.mean(enrichments)),
        }

    source_keys = (
        "threshold_fraction_of_boundaries",
        "periodic_fraction_of_boundaries",
        "threshold_only_fraction_of_boundaries",
        "periodic_only_fraction_of_boundaries",
        "both_fraction_of_boundaries",
    )
    source_rows: dict[str, float | None] = {}
    for key in source_keys:
        values = []
        for row in aggregate_alignment:
            value = row.get("boundary_sources", {}).get(key)
            if value is not None:
                values.append(float(value))
        source_rows[key] = None if not values else float(np.mean(values))

    flux_overall = []
    flux_boundary = []
    flux_nonboundary = []
    flux_p90 = []
    for row in aggregate_alignment:
        flux_row = row.get("flux", {})
        if flux_row.get("overall_mean") is not None:
            flux_overall.append(float(flux_row["overall_mean"]))
        if flux_row.get("boundary_mean") is not None:
            flux_boundary.append(float(flux_row["boundary_mean"]))
        if flux_row.get("nonboundary_mean") is not None:
            flux_nonboundary.append(float(flux_row["nonboundary_mean"]))
        if flux_row.get("overall_p90") is not None:
            flux_p90.append(float(flux_row["overall_p90"]))

    payload = {
        "artifact": str(artifact_path),
        "config_json": str(config_json),
        "label": args.label or artifact_path.stem,
        "trainer_module": trainer_module_name,
        "patch_feature_reduction": args.patch_feature_reduction,
        "analysis_max_batches": int(args.analysis_max_batches),
        "analyzed_batches": int(analyzed_batches),
        "analyzed_sequences": int(analyzed_sequences),
        "patch_count_summary": summarize_scalar([float(x) for x in all_patch_counts]),
        "chord_count_summary": summarize_scalar(all_chord_counts),
        "segment_span_summary": summarize_segment_lengths(np.asarray(all_span_lengths, dtype=np.int32)),
        "summary_path": {
            "eval_ce_summary": summarize_scalar(all_eval_ce),
            "eval_jepa_loss_summary": summarize_scalar(all_eval_jepa_loss),
            "patch_read_norm_summary": summarize_scalar(all_patch_read_norms),
            "token_read_norm_summary": summarize_scalar(all_token_read_norms),
            "chord_norm_summary": summarize_scalar(all_chord_norms),
            "adjacent_chord_cosine_summary": summarize_scalar(all_adjacent_chord_cosines),
            "read_drop_ce_delta_summary": summarize_scalar(all_read_drop_ce_delta),
        },
        "boundary_summary": {
            "patches_analyzed": int(total_patches),
            "boundaries_analyzed": int(total_boundaries),
            "boundary_rate": None if total_patches <= 0 else float(total_boundaries / total_patches),
            "boundary_sources_mean": source_rows,
            "feature_alignment_mean": feature_rows,
            "flux_alignment_mean": {
                "overall_mean": None if not flux_overall else float(np.mean(flux_overall)),
                "boundary_mean": None if not flux_boundary else float(np.mean(flux_boundary)),
                "nonboundary_mean": None if not flux_nonboundary else float(np.mean(flux_nonboundary)),
                "overall_p90_mean": None if not flux_p90 else float(np.mean(flux_p90)),
                "batch_flux_mean_summary": summarize_scalar(all_flux_means),
            },
        },
    }
    result_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
