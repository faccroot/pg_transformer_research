#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

if sys.platform.startswith("linux"):
    existing_cxxflags = str(os.environ.get("CXXFLAGS", "")).strip()
    if "-fpermissive" not in existing_cxxflags.split():
        os.environ["CXXFLAGS"] = "-fpermissive" if not existing_cxxflags else f"{existing_cxxflags} -fpermissive"
    if not str(os.environ.get("CXX", "")).strip():
        os.environ["CXX"] = "g++ -fpermissive"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "tools" else SCRIPT_PATH.parent
TOOLS_ROOT = REPO_ROOT / "tools"
PATH_CANDIDATES = [
    REPO_ROOT,
    TOOLS_ROOT,
    Path.home() / "transformer_research" / "parameter-golf",
    Path.home() / "transformer_research" / "parameter-golf" / "tools",
]
for root in PATH_CANDIDATES:
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract PCA-style logit factors and factor calibration summaries from a saved MLX artifact."
    )
    p.add_argument("--artifact", required=True, help="Path to *_mlx_model.npz or *_int8zlib.pklz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument("--label", default="", help="Optional run label")
    p.add_argument("--mode", choices=("logits", "prob_residual"), default="prob_residual")
    p.add_argument("--num-factors", type=int, default=4)
    p.add_argument("--top-tokens", type=int, default=12)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--hardmax-ablation", default="baseline")
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--data-path", default="", help="Optional override for DATA_PATH")
    p.add_argument("--val-max-seqs", type=int, default=-1, help="Optional override for VAL_MAX_SEQS")
    p.add_argument("--eval-seq-len", type=int, default=-1, help="Optional override for EVAL_SEQ_LEN")
    p.add_argument("--repo-bundle", default="", help="Optional tar/tgz of current repo Python modules for remote parity")
    p.add_argument("--summary-json", default="", help="Optional JSON summary output path")
    p.add_argument("--basis-npz", default="", help="Optional .npz basis artifact output path")
    return p.parse_args()


def maybe_stage_repo_bundle(bundle_path: str) -> Path | None:
    raw = bundle_path.strip()
    if not raw:
        return None
    bundle = Path(raw).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Repo bundle not found: {bundle}")
    temp_root = Path(tempfile.mkdtemp(prefix="logit_factor_bundle_"))
    with tarfile.open(bundle, "r:*") as tf:
        tf.extractall(temp_root)
    candidate_roots = [temp_root / "parameter-golf", temp_root]
    for root in candidate_roots:
        if root.exists():
            tools_root = root / "tools"
            for path in (root, tools_root):
                if path.exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return root
    return temp_root


def load_modules():
    import train_gpt_mlx as base_mod

    return importlib.reload(base_mod)


def load_config_env_payload(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", payload)
    if not isinstance(env_payload, dict):
        raise ValueError(f"Config env payload must be a JSON object: {config_path}")
    return env_payload


def apply_config_env(config_json: Path, args: argparse.Namespace) -> dict[str, object]:
    env = load_config_env_payload(config_json)
    for key, value in env.items():
        os.environ[str(key)] = str(value)
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = os.path.expanduser(args.tokenizer_path)
    if args.data_path:
        os.environ["DATA_PATH"] = os.path.expanduser(args.data_path)
    if args.val_max_seqs >= 0:
        os.environ["VAL_MAX_SEQS"] = str(int(args.val_max_seqs))
    if args.eval_seq_len > 0:
        os.environ["EVAL_SEQ_LEN"] = str(int(args.eval_seq_len))
    os.environ["MLX_COMPILE"] = "0"
    os.environ["VAL_LOSS_EVERY"] = "0"
    os.environ["QUANT_EVAL_EVERY"] = "0"
    os.environ["QUANT_EVAL_MAX_SEQS"] = "0"
    return env


def load_flat_state(artifact_path: Path, base_mod) -> dict[str, object]:
    if artifact_path.suffix == ".pklz":
        return base_mod.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    if artifact_path.suffix == ".npz":
        return dict(base_mod.mx.load(str(artifact_path)).items())
    raise ValueError(f"Unsupported artifact type: {artifact_path}")


def stable_factor_basis(covariance: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=1e9, neginf=-1e9)
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}")
    top_k = min(max(int(top_k), 1), int(covariance.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:top_k]
    values = np.clip(eigenvalues[order], 0.0, None).astype(np.float32)
    vectors = eigenvectors[:, order].T.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1.0e-8, None)
    return vectors, values


def top_token_rows(
    vector: np.ndarray,
    sp: spm.SentencePieceProcessor,
    *,
    top_k: int,
    descending: bool,
) -> list[dict[str, object]]:
    if vector.ndim != 1 or vector.size <= 0 or top_k <= 0:
        return []
    order = np.argsort(vector)
    if descending:
        order = order[::-1]
    rows = []
    for idx in order[:top_k]:
        token_id = int(idx)
        rows.append(
            {
                "token_id": token_id,
                "piece": sp.id_to_piece(token_id),
                "weight": float(vector[token_id]),
            }
        )
    return rows


def iter_eval_batches(
    *,
    base_mod,
    hps,
    model,
    val_tokens: np.ndarray,
    max_batches: int,
):
    eval_seq_len = int(hps.effective_eval_seq_len)
    val_batch_tokens = int(hps.val_batch_size) // int(hps.grad_accum_steps)
    if val_batch_tokens < eval_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one eval sequence; "
            f"got VAL_BATCH_SIZE={hps.val_batch_size}, GRAD_ACCUM_STEPS={hps.grad_accum_steps}, "
            f"EVAL_SEQ_LEN={eval_seq_len}"
        )
    val_batch_seqs = max(val_batch_tokens // eval_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * eval_seq_len
        raw_end = batch_seq_end * eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, eval_seq_len)
        y_np = chunk[1:].reshape(-1, eval_seq_len)
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        yield batch_idx, x_np, y_np, operator_codes


def batch_feature_matrix(
    *,
    logits: np.ndarray,
    target_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    flat_logits = logits.reshape(-1, logits.shape[-1]).astype(np.float32, copy=False)
    if mode == "logits":
        return flat_logits
    if mode != "prob_residual":
        raise ValueError(f"Unsupported factor mode {mode!r}")
    flat_targets = target_ids.reshape(-1).astype(np.int64, copy=False)
    shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    probs = exp / np.clip(exp.sum(axis=-1, keepdims=True), 1.0e-8, None)
    probs[np.arange(probs.shape[0]), flat_targets] -= 1.0
    return probs.astype(np.float32, copy=False)


def summarize_factors(
    *,
    sp: spm.SentencePieceProcessor,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    total_variance: float,
    abs_coord_sum: np.ndarray,
    coord_sum: np.ndarray,
    coord_sq_sum: np.ndarray,
    coord_nll_sum: np.ndarray,
    abs_coord_nll_sum: np.ndarray,
    coord_entropy_sum: np.ndarray,
    abs_coord_entropy_sum: np.ndarray,
    target_loading_sum: np.ndarray,
    bin_counts: np.ndarray,
    bin_nll_sums: np.ndarray,
    bin_acc_sums: np.ndarray,
    example_count: int,
    nll_sum: float,
    nll_sq_sum: float,
    entropy_sum: float,
    entropy_sq_sum: float,
    top_k_tokens: int,
) -> list[dict[str, object]]:
    factors: list[dict[str, object]] = []
    count_f = float(max(example_count, 1))
    mean_nll = nll_sum / count_f
    mean_entropy = entropy_sum / count_f
    var_nll = max(nll_sq_sum / count_f - mean_nll * mean_nll, 0.0)
    var_entropy = max(entropy_sq_sum / count_f - mean_entropy * mean_entropy, 0.0)
    std_nll = math.sqrt(var_nll)
    std_entropy = math.sqrt(var_entropy)
    total_variance = max(float(total_variance), 1.0e-8)
    for factor_idx in range(int(basis.shape[0])):
        coord_mean = float(coord_sum[factor_idx] / count_f)
        coord_var = max(float(coord_sq_sum[factor_idx] / count_f) - coord_mean * coord_mean, 0.0)
        coord_std = math.sqrt(coord_var)
        if coord_std > 0.0 and std_nll > 0.0:
            coord_nll_corr = (
                float(coord_nll_sum[factor_idx] / count_f) - coord_mean * mean_nll
            ) / max(coord_std * std_nll, 1.0e-8)
        else:
            coord_nll_corr = 0.0
        abs_coord_mean = float(abs_coord_sum[factor_idx] / count_f)
        abs_coord_var = max(float(coord_sq_sum[factor_idx] / count_f) - abs_coord_mean * abs_coord_mean, 0.0)
        abs_coord_std = math.sqrt(abs_coord_var)
        if abs_coord_std > 0.0 and std_nll > 0.0:
            abs_coord_nll_corr = (
                float(abs_coord_nll_sum[factor_idx] / count_f) - abs_coord_mean * mean_nll
            ) / max(abs_coord_std * std_nll, 1.0e-8)
        else:
            abs_coord_nll_corr = 0.0
        if coord_std > 0.0 and std_entropy > 0.0:
            coord_entropy_corr = (
                float(coord_entropy_sum[factor_idx] / count_f) - coord_mean * mean_entropy
            ) / max(coord_std * std_entropy, 1.0e-8)
        else:
            coord_entropy_corr = 0.0
        if abs_coord_std > 0.0 and std_entropy > 0.0:
            abs_coord_entropy_corr = (
                float(abs_coord_entropy_sum[factor_idx] / count_f) - abs_coord_mean * mean_entropy
            ) / max(abs_coord_std * std_entropy, 1.0e-8)
        else:
            abs_coord_entropy_corr = 0.0
        bins = []
        labels = ["low_abs", "mid_abs", "high_abs"]
        for bin_idx, label in enumerate(labels):
            denom = max(float(bin_counts[factor_idx, bin_idx]), 1.0)
            bins.append(
                {
                    "bin": label,
                    "count": int(bin_counts[factor_idx, bin_idx]),
                    "mean_nll": float(bin_nll_sums[factor_idx, bin_idx] / denom),
                    "mean_top1_acc": float(bin_acc_sums[factor_idx, bin_idx] / denom),
                }
            )
        factors.append(
            {
                "factor_idx": factor_idx,
                "eigenvalue": float(eigenvalues[factor_idx]),
                "explained_variance_ratio": float(eigenvalues[factor_idx] / total_variance),
                "coord_mean": coord_mean,
                "coord_std": coord_std,
                "coord_nll_corr": float(coord_nll_corr),
                "abs_coord_nll_corr": float(abs_coord_nll_corr),
                "coord_entropy_corr": float(coord_entropy_corr),
                "abs_coord_entropy_corr": float(abs_coord_entropy_corr),
                "mean_abs_coord": abs_coord_mean,
                "mean_target_loading": float(target_loading_sum[factor_idx] / count_f),
                "top_positive_tokens": top_token_rows(basis[factor_idx], sp, top_k=top_k_tokens, descending=True),
                "top_negative_tokens": top_token_rows(basis[factor_idx], sp, top_k=top_k_tokens, descending=False),
                "activation_bins": bins,
            }
        )
    return factors


def main() -> None:
    args = parse_args()
    maybe_stage_repo_bundle(args.repo_bundle)
    artifact_path = Path(args.artifact).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve()
    env_payload = apply_config_env(config_path, args)
    base_mod = load_modules()
    hps = base_mod.Hyperparameters()
    tokenizer_path = os.path.expanduser(args.tokenizer_path or hps.tokenizer_path)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model = base_mod.make_gpt(hps, sp)

    checkpoint_state = load_flat_state(artifact_path, base_mod)
    selected_state, checkpoint_stats = base_mod.select_compatible_flat_state(model, checkpoint_state)
    base_mod.apply_flat_arrays(model, selected_state)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()

    val_seq_len = max(hps.train_seq_len, hps.effective_eval_seq_len)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, val_seq_len),
        hps.effective_eval_seq_len,
        hps.val_max_seqs,
    )
    factor_dim = int(hps.vocab_size)
    feature_sum = np.zeros((factor_dim,), dtype=np.float64)
    feature_xx = np.zeros((factor_dim, factor_dim), dtype=np.float64)
    sample_count = 0

    ablation_spec = base_mod.resolve_hardmax_eval_ablation(args.hardmax_ablation)
    with model.hardmax_eval_ablation_scope(ablation_spec):
        for _batch_idx, x_np, y_np, operator_codes in iter_eval_batches(
            base_mod=base_mod,
            hps=hps,
            model=model,
            val_tokens=val_tokens,
            max_batches=args.max_batches,
        ):
            logits = model.forward_logits(mx.array(x_np, dtype=mx.int32), operator_codes).astype(mx.float32)
            mx.eval(logits)
            logits_np = np.asarray(logits, dtype=np.float32)
            features = batch_feature_matrix(logits=logits_np, target_ids=y_np, mode=args.mode).astype(np.float64, copy=False)
            feature_sum += features.sum(axis=0)
            feature_xx += features.T @ features
            sample_count += int(features.shape[0])

    if sample_count <= 0:
        raise ValueError("No validation samples were processed")
    mean = feature_sum / float(sample_count)
    covariance = feature_xx / float(sample_count) - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    basis, eigenvalues = stable_factor_basis(covariance, args.num_factors)
    total_variance = float(np.trace(covariance))

    coord_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    coord_sq_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    abs_coord_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    coord_nll_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    abs_coord_nll_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    coord_entropy_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    abs_coord_entropy_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    target_loading_sum = np.zeros((basis.shape[0],), dtype=np.float64)
    bin_counts = np.zeros((basis.shape[0], 3), dtype=np.int64)
    bin_nll_sums = np.zeros((basis.shape[0], 3), dtype=np.float64)
    bin_acc_sums = np.zeros((basis.shape[0], 3), dtype=np.float64)
    nll_sum = 0.0
    nll_sq_sum = 0.0
    entropy_sum = 0.0
    entropy_sq_sum = 0.0
    top1_correct_sum = 0.0

    with model.hardmax_eval_ablation_scope(ablation_spec):
        for _batch_idx, x_np, y_np, operator_codes in iter_eval_batches(
            base_mod=base_mod,
            hps=hps,
            model=model,
            val_tokens=val_tokens,
            max_batches=args.max_batches,
        ):
            logits = model.forward_logits(mx.array(x_np, dtype=mx.int32), operator_codes).astype(mx.float32)
            mx.eval(logits)
            logits_np = np.asarray(logits, dtype=np.float32)
            flat_logits = logits_np.reshape(-1, logits_np.shape[-1]).astype(np.float32, copy=False)
            flat_targets = y_np.reshape(-1).astype(np.int64, copy=False)
            shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted).astype(np.float32, copy=False)
            probs = exp / np.clip(exp.sum(axis=-1, keepdims=True), 1.0e-8, None)
            target_prob = np.clip(probs[np.arange(probs.shape[0]), flat_targets], 1.0e-8, None)
            nll = (-np.log(target_prob)).astype(np.float32, copy=False)
            entropy = (-np.sum(probs * np.log(np.clip(probs, 1.0e-8, None)), axis=-1)).astype(np.float32, copy=False)
            top1 = np.argmax(flat_logits, axis=-1)
            top1_correct = (top1 == flat_targets).astype(np.float32)
            features = batch_feature_matrix(logits=logits_np, target_ids=y_np, mode=args.mode).astype(np.float32, copy=False)
            centered = features - mean.astype(np.float32)
            coords = centered @ basis.T.astype(np.float32)
            abs_coords = np.abs(coords).astype(np.float32, copy=False)

            coord_sum += coords.sum(axis=0, dtype=np.float64)
            coord_sq_sum += np.square(coords.astype(np.float64, copy=False)).sum(axis=0, dtype=np.float64)
            abs_coord_sum += abs_coords.sum(axis=0, dtype=np.float64)
            coord_nll_sum += (coords * nll[:, None]).sum(axis=0, dtype=np.float64)
            abs_coord_nll_sum += (abs_coords * nll[:, None]).sum(axis=0, dtype=np.float64)
            coord_entropy_sum += (coords * entropy[:, None]).sum(axis=0, dtype=np.float64)
            abs_coord_entropy_sum += (abs_coords * entropy[:, None]).sum(axis=0, dtype=np.float64)
            target_loading_sum += basis[:, flat_targets].sum(axis=1, dtype=np.float64)

            nll_sum += float(nll.astype(np.float64).sum())
            nll_sq_sum += float(np.square(nll.astype(np.float64)).sum())
            entropy_sum += float(entropy.astype(np.float64).sum())
            entropy_sq_sum += float(np.square(entropy.astype(np.float64)).sum())
            top1_correct_sum += float(top1_correct.astype(np.float64).sum())

            stds = np.sqrt(np.clip(eigenvalues.astype(np.float64), 1.0e-8, None))
            low_thr = 0.5 * stds[None, :]
            high_thr = 1.5 * stds[None, :]
            low_mask = abs_coords < low_thr
            high_mask = abs_coords >= high_thr
            mid_mask = (~low_mask) & (~high_mask)
            for factor_idx in range(basis.shape[0]):
                masks = (low_mask[:, factor_idx], mid_mask[:, factor_idx], high_mask[:, factor_idx])
                for bin_idx, mask in enumerate(masks):
                    if not np.any(mask):
                        continue
                    count = int(mask.sum())
                    bin_counts[factor_idx, bin_idx] += count
                    bin_nll_sums[factor_idx, bin_idx] += float(nll[mask].astype(np.float64).sum())
                    bin_acc_sums[factor_idx, bin_idx] += float(top1_correct[mask].astype(np.float64).sum())

    mean_nll = nll_sum / float(max(sample_count, 1))
    mean_bpt = mean_nll / math.log(2.0)
    factors = summarize_factors(
        sp=sp,
        basis=basis,
        eigenvalues=eigenvalues,
        total_variance=total_variance,
        abs_coord_sum=abs_coord_sum,
        coord_sum=coord_sum,
        coord_sq_sum=coord_sq_sum,
        coord_nll_sum=coord_nll_sum,
        abs_coord_nll_sum=abs_coord_nll_sum,
        coord_entropy_sum=coord_entropy_sum,
        abs_coord_entropy_sum=abs_coord_entropy_sum,
        target_loading_sum=target_loading_sum,
        bin_counts=bin_counts,
        bin_nll_sums=bin_nll_sums,
        bin_acc_sums=bin_acc_sums,
        example_count=sample_count,
        nll_sum=nll_sum,
        nll_sq_sum=nll_sq_sum,
        entropy_sum=entropy_sum,
        entropy_sq_sum=entropy_sq_sum,
        top_k_tokens=args.top_tokens,
    )

    summary = {
        "label": args.label,
        "artifact": str(artifact_path),
        "config_json": str(config_path),
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "data_path": str(Path(os.path.expanduser(hps.data_path)).resolve()),
        "mode": args.mode,
        "hardmax_ablation": ablation_spec.name,
        "checkpoint_load_stats": checkpoint_stats,
        "num_factors": int(basis.shape[0]),
        "sample_count": int(sample_count),
        "validation_tokens": int(val_tokens.size - 1),
        "max_batches": int(args.max_batches),
        "mean_nll": float(mean_nll),
        "mean_bpt": float(mean_bpt),
        "mean_top1_acc": float(top1_correct_sum / float(max(sample_count, 1))),
        "mean_entropy": float(entropy_sum / float(max(sample_count, 1))),
        "total_variance": float(total_variance),
        "config_env_keys": sorted(str(key) for key in env_payload.keys()),
        "factors": factors,
    }

    if args.basis_npz:
        basis_path = Path(args.basis_npz).expanduser().resolve()
        basis_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            basis_path,
            basis=basis.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float32),
            mean=mean.astype(np.float32),
            covariance_diag=np.diag(covariance).astype(np.float32),
            mode=np.array([args.mode]),
            hardmax_ablation=np.array([ablation_spec.name]),
            sample_count=np.array([sample_count], dtype=np.int64),
            vocab_size=np.array([factor_dim], dtype=np.int64),
        )
        summary["basis_npz"] = str(basis_path)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
