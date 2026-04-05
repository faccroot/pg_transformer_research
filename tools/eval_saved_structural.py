#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import subprocess
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
from mlx.utils import tree_unflatten

from structural_branching import StructuralBranchingConfig, select_structural_branch_points_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved base artifact and optionally analyze structural branch points."
    )
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument("--label", default="")
    p.add_argument("--result-json", default="")
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--data-path", default="", help="Optional override for DATA_PATH")
    p.add_argument("--val-max-seqs", type=int, default=None)
    p.add_argument("--eval-seq-len", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=0)
    p.add_argument("--adapt-branch", type=int, default=0, help="If set, run score-first branch adaptation during eval.")
    p.add_argument("--adapt-lr", type=float, default=1e-5)
    p.add_argument("--adapt-beta1", type=float, default=0.9)
    p.add_argument("--adapt-beta2", type=float, default=0.999)
    p.add_argument("--adapt-ce-weight", type=float, default=0.0)
    p.add_argument("--adapt-branch-weight", type=float, default=1.0)
    p.add_argument("--adapt-branch-state-weight", type=float, default=0.0)
    p.add_argument("--adapt-max-batches", type=int, default=0)
    p.add_argument("--adapt-min-branch-points", type=int, default=1)
    p.add_argument("--adapt-rescore", type=int, default=1, help="If set, run a second full re-score after adaptation.")
    p.add_argument("--branch-max-branches", type=int, default=2)
    p.add_argument("--branch-min-structural-miss", type=float, default=0.5)
    p.add_argument("--branch-max-top1-gap", type=float, default=0.75)
    p.add_argument("--branch-max-top12-cosine", type=float, default=0.75)
    p.add_argument("--branch-min-branch-score", type=float, default=0.0)
    p.add_argument("--branch-min-top1-prob", type=float, default=0.0)
    p.add_argument("--branch-min-position-gap", type=int, default=8)
    return p.parse_args()


def load_modules():
    import train_gpt_mlx as base_mod

    return importlib.reload(base_mod)


def apply_config_env(config_json: Path, args: argparse.Namespace) -> None:
    config = json.loads(config_json.read_text())
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
    local_peer_script = Path(__file__).resolve().parent / "cached_challenge_fineweb.py"
    repo_script = Path(__file__).resolve().parents[1] / "data" / "cached_challenge_fineweb.py"
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
    if artifact_path.suffix == ".pklz":
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
            batch_windows = windows[batch_start:batch_start + batch_seqs]
            x_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
            y_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
            for i, (window_start, window_len, _, _) in enumerate(batch_windows):
                chunk = val_tokens[window_start:window_start + window_len + 1]
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


def analyze_structural_branches(base_mod, model, hps, val_tokens: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    branch_cfg = StructuralBranchingConfig(
        enabled=True,
        start_frac=0.0,
        weight=1.0,
        branch_length=hps.structural_branching_branch_length,
        max_branches=max(int(args.branch_max_branches), 0),
        min_structural_miss=float(args.branch_min_structural_miss),
        max_top1_gap=float(args.branch_max_top1_gap),
        max_top12_cosine=float(args.branch_max_top12_cosine),
        min_branch_score=float(args.branch_min_branch_score),
        min_top1_prob=float(args.branch_min_top1_prob),
        min_position_gap=int(args.branch_min_position_gap),
        margin=hps.structural_branching_margin,
    )
    embedding_table = np.asarray(base_mod.mx.stop_gradient(model.tok_emb.weight.astype(base_mod.mx.float32)))
    total_batches = 0
    total_points = 0
    miss_sum = 0.0
    cosine_sum = 0.0
    score_sum = 0.0
    top_examples: list[dict[str, float | int]] = []

    for x_np, y_np in iter_eval_batches(base_mod, hps, val_tokens):
        total_batches += 1
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        logits = model.forward_logits(base_mod.mx.array(x_np, dtype=base_mod.mx.int32), operator_codes)
        logits_np = np.asarray(base_mod.mx.stop_gradient(logits.astype(base_mod.mx.float32)))
        branch_plans = select_structural_branch_points_np(logits_np, y_np, embedding_table, branch_cfg)
        flat_plans = [plan for row in branch_plans for plan in row]
        total_points += len(flat_plans)
        for plan in flat_plans:
            miss_sum += float(plan.structural_miss)
            cosine_sum += float(plan.top12_cosine)
            score_sum += float(plan.score)
            top_examples.append(
                {
                    "batch": total_batches - 1,
                    "pos": int(plan.pos),
                    "predicted_token": int(plan.predicted_token),
                    "target_token": int(plan.target_token),
                    "alternate_token": int(plan.alternate_token),
                    "miss": float(plan.structural_miss),
                    "top12_cosine": float(plan.top12_cosine),
                    "score": float(plan.score),
                }
            )
        if args.analysis_max_batches > 0 and total_batches >= args.analysis_max_batches:
            break
    top_examples.sort(key=lambda item: float(item["score"]), reverse=True)
    denom = max(total_points, 1)
    return {
        "analyzed_batches": total_batches,
        "branch_points": total_points,
        "mean_structural_miss": miss_sum / denom,
        "mean_top12_cosine": cosine_sum / denom,
        "mean_branch_score": score_sum / denom,
        "top_examples": top_examples[:10],
    }


def _apply_model_grads(base_mod, model, optimizer, grads_tree) -> None:
    params = dict(base_mod.tree_flatten(model.parameters()))
    grads = dict(base_mod.tree_flatten(grads_tree))
    updated = optimizer.apply_gradients(grads, params)
    model.update(base_mod.tree_unflatten(list(updated.items())))
    model.clear_turbo_cache()
    base_mod.mx.eval(*updated.values())


def evaluate_with_branch_adaptation(
    base_mod,
    model,
    hps,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    mx = base_mod.mx
    nn = base_mod.nn
    optim = base_mod.optim
    branch_cfg = StructuralBranchingConfig(
        enabled=True,
        start_frac=0.0,
        weight=float(args.adapt_branch_weight),
        branch_length=hps.structural_branching_branch_length,
        max_branches=max(int(args.branch_max_branches), 0),
        min_structural_miss=float(args.branch_min_structural_miss),
        max_top1_gap=float(args.branch_max_top1_gap),
        max_top12_cosine=float(args.branch_max_top12_cosine),
        min_branch_score=float(args.branch_min_branch_score),
        min_top1_prob=float(args.branch_min_top1_prob),
        min_position_gap=int(args.branch_min_position_gap),
        margin=hps.structural_branching_margin,
        state_divergence_weight=float(args.adapt_branch_state_weight),
        state_target_max_cosine=hps.structural_branching_state_target_max_cosine,
        adaptive_depth_enabled=hps.structural_branching_adaptive_depth_enabled,
        adaptive_min_depth=hps.structural_branching_adaptive_min_depth,
        adaptive_plateau_tol=hps.structural_branching_adaptive_plateau_tol,
        adaptive_converged_divergence=hps.structural_branching_adaptive_converged_divergence,
    )
    embedding_table = np.asarray(mx.stop_gradient(model.tok_emb.weight.astype(mx.float32)))
    optimizer = optim.Adam(
        learning_rate=args.adapt_lr,
        betas=[args.adapt_beta1, args.adapt_beta2],
        eps=1e-8,
        bias_correction=True,
    )
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    total_branch_points = 0
    adapted_batches = 0
    skipped_batches = 0
    adapt_loss_sum = 0.0

    def adapt_loss(x, y, operator_codes, branch_plans):
        (
            _total,
            ce_loss,
            _seed_loss,
            _sparse_loss,
            _smooth_loss,
            _early_exit_loss,
            _prosody_aux_loss,
            _prosody_token_class_loss,
            _prosody_boundary_loss,
            _prosody_quote_loss,
            _distill_loss,
            branch_rank_loss,
            branch_state_loss,
        ) = model.loss_terms(
            x,
            y,
            operator_codes=operator_codes,
            structural_branching_cfg=branch_cfg,
            branch_plans=branch_plans,
        )
        return (
            float(args.adapt_ce_weight) * ce_loss
            + float(args.adapt_branch_weight) * branch_rank_loss
            + float(args.adapt_branch_state_weight) * branch_state_loss
        )

    for batch_idx, (x_np, y_np) in enumerate(iter_eval_batches(base_mod, hps, val_tokens), start=1):
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        logits = model.forward_logits(x, operator_codes)
        nll = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none").astype(mx.float32)
        nll_sum = mx.sum(nll)
        mx.eval(nll_sum)
        total_loss_sum += float(nll_sum.item())
        chunk_token_count = float(y_np.size)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())

        logits_np = np.asarray(mx.stop_gradient(logits.astype(mx.float32)))
        branch_plans = select_structural_branch_points_np(logits_np, y_np, embedding_table, branch_cfg)
        branch_points = sum(len(row) for row in branch_plans)
        total_branch_points += branch_points
        if branch_points >= int(args.adapt_min_branch_points):
            loss_and_grad = nn.value_and_grad(model, lambda x_i, y_i, op_i: adapt_loss(x_i, y_i, op_i, branch_plans))
            adapt_value, grads = loss_and_grad(x, y, operator_codes)
            _apply_model_grads(base_mod, model, optimizer, grads)
            adapted_batches += 1
            adapt_loss_sum += float(adapt_value.item())
        else:
            adapt_value = mx.array(0.0, dtype=mx.float32)
            skipped_batches += 1

        if batch_idx == 1 or (args.adapt_max_batches > 0 and batch_idx == args.adapt_max_batches) or batch_idx % 25 == 0:
            print(
                f"adaptive_branch_progress:{batch_idx} "
                f"branch_points:{branch_points} adapt_loss:{float(adapt_value.item()):.6f}"
            )
        if args.adapt_max_batches > 0 and batch_idx >= args.adapt_max_batches:
            break

    val_loss = total_loss_sum / max(total_tokens, 1.0)
    bits_per_token = val_loss / np.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    print(f"adaptive_branch_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    result = {
        "mode": "adaptive_branch",
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
        "batches_seen": int(batch_idx if 'batch_idx' in locals() else 0),
        "adapted_batches": int(adapted_batches),
        "skipped_batches": int(skipped_batches),
        "total_branch_points": int(total_branch_points),
        "mean_branch_points_per_batch": float(total_branch_points / max(int(batch_idx if 'batch_idx' in locals() else 0), 1)),
        "mean_branch_points_per_adapted_batch": float(total_branch_points / max(adapted_batches, 1)),
        "mean_adapt_loss": float(adapt_loss_sum / max(adapted_batches, 1)),
    }
    return result


def main() -> None:
    args = parse_args()
    apply_config_env(Path(args.config_json), args)
    ensure_dataset_ready(args)
    base_mod = load_modules()

    artifact_path = Path(args.artifact).expanduser().resolve()
    sp = spm.SentencePieceProcessor(model_file=os.path.expanduser(os.environ["TOKENIZER_PATH"]))
    hps = base_mod.Hyperparameters()
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        hps.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base_mod.build_sentencepiece_luts(sp, hps.vocab_size)

    model = base_mod.make_gpt(hps, sp)
    model.set_turbo_qat(False, 0.0)
    flat_state = load_flat_state(artifact_path, base_mod)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()

    uses_logic = (
        model.logic_sidecar is not None
        or model.polarity_detector is not None
        or model.sidecar_polarity_write
    )
    if uses_logic:
        ce_loss = lambda x, y, operator_codes=None: model.loss(x, y, operator_codes)
        forward_logits = lambda x, operator_codes=None: model.forward_logits(x, operator_codes)
    else:
        ce_loss = lambda x, y, operator_codes=None: model.loss(x, y)
        forward_logits = lambda x, operator_codes=None: model.forward_logits(x)

    if int(args.adapt_branch):
        result = evaluate_with_branch_adaptation(
            base_mod,
            model,
            hps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            args,
        )
        result.update(
            {
                "label": args.label,
                "artifact": str(artifact_path),
                "config_json": str(Path(args.config_json).expanduser().resolve()),
                "adapt_config": {
                    "adapt_lr": float(args.adapt_lr),
                    "adapt_beta1": float(args.adapt_beta1),
                    "adapt_beta2": float(args.adapt_beta2),
                    "adapt_ce_weight": float(args.adapt_ce_weight),
                    "adapt_branch_weight": float(args.adapt_branch_weight),
                    "adapt_branch_state_weight": float(args.adapt_branch_state_weight),
                    "adapt_min_branch_points": int(args.adapt_min_branch_points),
                    "adapt_max_batches": int(args.adapt_max_batches),
                },
                "branch_config": {
                    "branch_max_branches": int(args.branch_max_branches),
                    "branch_min_structural_miss": float(args.branch_min_structural_miss),
                    "branch_max_top1_gap": float(args.branch_max_top1_gap),
                    "branch_max_top12_cosine": float(args.branch_max_top12_cosine),
                    "branch_min_branch_score": float(args.branch_min_branch_score),
                    "branch_min_top1_prob": float(args.branch_min_top1_prob),
                    "branch_min_position_gap": int(args.branch_min_position_gap),
                },
            }
        )
        if int(args.adapt_rescore):
            rescore_loss, rescore_bpb = base_mod.eval_val(
                hps,
                model,
                ce_loss,
                forward_logits,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=print,
            )
            result["rescore_val_loss"] = float(rescore_loss)
            result["rescore_val_bpb"] = float(rescore_bpb)
            result["rescore_delta_bpb"] = float(rescore_bpb - result["val_bpb"])
            result["rescore_improved"] = bool(rescore_bpb < result["val_bpb"])
            print(f"adaptive_branch_rescore_exact val_loss:{rescore_loss:.8f} val_bpb:{rescore_bpb:.8f}")
            if args.analysis_max_batches > 0:
                result["post_adapt_structural_analysis"] = analyze_structural_branches(
                    base_mod,
                    model,
                    hps,
                    val_tokens,
                    args,
                )
    else:
        val_loss, val_bpb = base_mod.eval_val(
            hps,
            model,
            ce_loss,
            forward_logits,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=print,
        )
        result = {
            "label": args.label,
            "artifact": str(artifact_path),
            "config_json": str(Path(args.config_json).expanduser().resolve()),
            "mode": "exact",
            "val_loss": float(val_loss),
            "val_bpb": float(val_bpb),
        }
        print(f"eval_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if args.analysis_max_batches > 0:
        analysis = analyze_structural_branches(base_mod, model, hps, val_tokens, args)
        result["structural_analysis"] = analysis
        print(json.dumps({"structural_analysis": analysis}, indent=2))

    if args.result_json:
        out_path = Path(args.result_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
