#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import sentencepiece as spm  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from logic_register_mlx import CastedLinear, HardmaxStructuralController  # noqa: E402
import train_gpt_mlx as base  # noqa: E402


def apply_overrides_from_config(argv: list[str]) -> None:
    config_path: str | None = None
    passthrough = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        passthrough.append(arg)
        i += 1

    if config_path is None:
        return

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")

    env_payload = payload.get("env", {})
    if env_payload:
        if not isinstance(env_payload, dict):
            raise SystemExit(f"Config env payload must be a JSON object: {config_path}")
        for key, value in env_payload.items():
            if value is None:
                os.environ.pop(str(key), None)
            else:
                os.environ[str(key)] = str(value)

    args_payload = payload.get("args", {})
    if args_payload and not isinstance(args_payload, dict):
        raise SystemExit(f"Config args payload must be a JSON object: {config_path}")

    if isinstance(args_payload, dict):
        teacher_config = args_payload.get("teacher-config", args_payload.get("teacher_config"))
        teacher_checkpoint = args_payload.get("teacher-checkpoint", args_payload.get("teacher_checkpoint"))
        if teacher_config is not None and str(teacher_config).strip():
            os.environ["EXTERNAL_TEACHER_CONFIG_PATHS"] = str(teacher_config)
        if teacher_checkpoint is not None and str(teacher_checkpoint).strip():
            os.environ["EXTERNAL_TEACHER_CHECKPOINT_PATHS"] = str(teacher_checkpoint)

    expanded = list(passthrough)
    for key, value in args_payload.items():
        if value is None:
            continue
        option = f"--{str(key).strip()}"
        if isinstance(value, bool):
            if value:
                expanded.append(option)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                expanded.extend((option, str(item)))
            continue
        expanded.extend((option, str(value)))
    sys.argv[:] = expanded


apply_overrides_from_config(sys.argv)


def build_parser() -> argparse.ArgumentParser:
    defaults = base.Hyperparameters()
    parser = argparse.ArgumentParser(
        description="Train a controller-only hardmax draft model to imitate a frozen teacher's next-token certainty surface."
    )
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", ""))
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--controller-out", default="")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--seed", type=int, default=int(defaults.seed))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size in sequences.")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size in sequences.")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--teacher-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--uncertain-token-weight", type=float, default=0.05)
    parser.add_argument("--confidence-weight", type=float, default=0.10)
    parser.add_argument("--usage-balance-weight", type=float, default=1.0e-2)
    parser.add_argument("--diversity-weight", type=float, default=1.0e-2)
    parser.add_argument("--student-accept-threshold", type=float, default=0.95)
    parser.add_argument("--val-max-seqs", type=int, default=256)
    parser.add_argument(
        "--teacher-config",
        default=os.environ.get("EXTERNAL_TEACHER_CONFIG_PATHS", ""),
        help="Comma-separated frozen teacher config path(s). Defaults to EXTERNAL_TEACHER_CONFIG_PATHS.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=os.environ.get("EXTERNAL_TEACHER_CHECKPOINT_PATHS", ""),
        help="Comma-separated frozen teacher checkpoint path(s). Defaults to EXTERNAL_TEACHER_CHECKPOINT_PATHS.",
    )
    return parser


def masked_cross_entropy(logits: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
    flat_logits = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    flat_targets = targets.reshape(-1).astype(mx.int32)
    losses = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none").astype(mx.float32)
    losses = losses.reshape(targets.shape)
    denom = mx.maximum(mx.sum(weights.astype(mx.float32)), mx.array(1.0e-6, dtype=mx.float32))
    return mx.sum(losses * weights.astype(mx.float32)) / denom


def masked_accuracy(logits: mx.array, targets: mx.array, weights: mx.array) -> float:
    pred = mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)
    correct = (pred == targets.astype(mx.int32)).astype(mx.float32) * weights.astype(mx.float32)
    denom = mx.maximum(mx.sum(weights.astype(mx.float32)), mx.array(1.0e-6, dtype=mx.float32))
    return float((mx.sum(correct) / denom).item())


def mean_weighted(values: mx.array, weights: mx.array) -> mx.array:
    denom = mx.maximum(mx.sum(weights.astype(mx.float32)), mx.array(1.0e-6, dtype=mx.float32))
    return mx.sum(values.astype(mx.float32) * weights.astype(mx.float32)) / denom


class HardmaxTeacherDraft(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        model_dim: int,
        state_dim: int,
        num_states: int,
        temperature: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.mix = CastedLinear(model_dim, model_dim)
        self.controller = HardmaxStructuralController(
            model_dim,
            state_dim,
            num_states,
            temperature=temperature,
            compute_min_scale=1.0,
            compute_power=1.0,
            operator_prior_scale=0.0,
            reset_prior_scale=0.0,
        )
        self.lm_head = CastedLinear(state_dim, vocab_size)

    def forward_logits(self, input_ids: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
        seq_len = int(input_ids.shape[1])
        positions = mx.arange(seq_len, dtype=mx.int32)[None, :]
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = nn.silu(x + self.mix(x))
        _conditioned, aux = self.controller(x, None)
        logits = self.lm_head(aux["struct_state"])
        return logits, aux

    def loss_terms(
        self,
        input_ids: mx.array,
        teacher_logits: mx.array,
        *,
        teacher_confidence_threshold: float,
        uncertain_token_weight: float,
        confidence_weight: float,
        usage_balance_weight: float,
        diversity_weight: float,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        student_logits, aux = self.forward_logits(input_ids)
        teacher_logits32 = mx.stop_gradient(teacher_logits.astype(mx.float32))
        teacher_log_probs = teacher_logits32 - mx.logsumexp(teacher_logits32, axis=-1, keepdims=True)
        teacher_probs = mx.exp(teacher_log_probs)
        teacher_top1 = mx.argmax(teacher_logits32, axis=-1).astype(mx.int32)
        teacher_top1_prob = mx.max(teacher_probs, axis=-1).astype(mx.float32)
        confident_mask = (teacher_top1_prob >= mx.array(teacher_confidence_threshold, dtype=mx.float32)).astype(mx.float32)
        token_weights = confident_mask + mx.array(float(uncertain_token_weight), dtype=mx.float32) * (1.0 - confident_mask)
        token_loss = masked_cross_entropy(student_logits, teacher_top1, token_weights)
        confidence = aux["confidence"].astype(mx.float32)
        conf_loss = mx.mean(mx.square(confidence - teacher_top1_prob))
        usage_balance, diversity, entropy = self.controller.regularization_losses(aux)
        total = (
            token_loss
            + confidence_weight * conf_loss
            + usage_balance_weight * usage_balance
            + diversity_weight * diversity
        ).astype(mx.float32)
        metrics = {
            "loss_total": total,
            "loss_token": token_loss.astype(mx.float32),
            "loss_confidence": conf_loss.astype(mx.float32),
            "loss_usage_balance": usage_balance.astype(mx.float32),
            "loss_diversity": diversity.astype(mx.float32),
            "teacher_confident_frac": mx.mean(confident_mask).astype(mx.float32),
            "teacher_confidence_mean": mx.mean(teacher_top1_prob).astype(mx.float32),
            "student_confidence_mean": mx.mean(confidence).astype(mx.float32),
            "entropy": entropy.astype(mx.float32),
        }
        return total, metrics


def optimizer_step(model: nn.Module, optimizer: optim.Adam, grads_tree) -> None:
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads_tree))
    updated = optimizer.apply_gradients(grads, params)
    model.update(tree_unflatten(list(updated.items())))


def export_controller_init_state(model: HardmaxTeacherDraft) -> dict[str, mx.array]:
    return {
        f"hardmax_structural_controller.{name}": value
        for name, value in tree_flatten(model.controller.state)
    }


def export_model_state(model: HardmaxTeacherDraft) -> dict[str, mx.array]:
    return {name: value for name, value in tree_flatten(model.state)}


def sample_val_batch(
    val_tokens: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    total_seqs = max((val_tokens.size - 1) // seq_len, 1)
    rows = np.empty((batch_size, seq_len), dtype=np.int32)
    targets = np.empty((batch_size, seq_len), dtype=np.int32)
    for idx in range(batch_size):
        seq_idx = rng.randrange(total_seqs)
        start = seq_idx * seq_len
        chunk = val_tokens[start : start + seq_len + 1]
        rows[idx] = chunk[:-1]
        targets[idx] = chunk[1:]
    return rows, targets


def eval_model(
    model: HardmaxTeacherDraft,
    *,
    teacher_models: list[base.GPT],
    x_np: np.ndarray,
    logic_phase_enabled: bool,
    teacher_confidence_threshold: float,
    student_accept_threshold: float,
) -> dict[str, float]:
    x = mx.array(x_np, dtype=mx.int32)
    teacher_logits, _teacher_metrics = base.ensemble_teacher_logits_for_batch(
        teacher_models,
        x,
        x_np,
        logic_phase_enabled=logic_phase_enabled,
        collect_metrics=False,
    )
    if teacher_logits is None:
        raise ValueError("Teacher logits are required for evaluation")
    student_logits, aux = model.forward_logits(x)
    teacher_logits32 = teacher_logits.astype(mx.float32)
    teacher_probs = mx.softmax(teacher_logits32, axis=-1)
    teacher_top1 = mx.argmax(teacher_logits32, axis=-1).astype(mx.int32)
    teacher_top1_prob = mx.max(teacher_probs, axis=-1).astype(mx.float32)
    confident_mask = (teacher_top1_prob >= mx.array(teacher_confidence_threshold, dtype=mx.float32)).astype(mx.float32)
    student_conf = aux["confidence"].astype(mx.float32)
    accept_mask = (student_conf >= mx.array(student_accept_threshold, dtype=mx.float32)).astype(mx.float32)
    all_mask = mx.ones(teacher_top1_prob.shape, dtype=mx.float32)
    loss, metrics = model.loss_terms(
        x,
        teacher_logits,
        teacher_confidence_threshold=teacher_confidence_threshold,
        uncertain_token_weight=0.05,
        confidence_weight=0.0,
        usage_balance_weight=0.0,
        diversity_weight=0.0,
    )
    soft_usage = aux["soft_usage"].astype(mx.float32)
    mean_soft_usage = mx.mean(soft_usage, axis=(0, 1))
    summary = {
        "loss": float(loss.item()),
        "teacher_match_acc": masked_accuracy(student_logits, teacher_top1, all_mask),
        "teacher_match_confident_acc": masked_accuracy(student_logits, teacher_top1, confident_mask),
        "teacher_confident_frac": float(mx.mean(confident_mask).item()),
        "student_accept_frac": float(mx.mean(accept_mask).item()),
        "student_accept_acc": masked_accuracy(student_logits, teacher_top1, accept_mask),
        "teacher_confidence_mean": float(mx.mean(teacher_top1_prob).item()),
        "student_confidence_mean": float(mx.mean(student_conf).item()),
        "hard_usage_peak_frac": float(mx.max(mx.mean(aux["hard_usage"].astype(mx.float32), axis=(0, 1))).item()),
        "soft_usage_perplexity": float(
            mx.exp(
                -mx.sum(mean_soft_usage * mx.log(mx.maximum(mean_soft_usage, mx.array(1.0e-8, dtype=mx.float32))))
            ).item()
        ),
    }
    for key, value in metrics.items():
        summary[key] = float(value.item())
    return summary


def resolve_output_path(explicit: str, out_dir: str, run_id: str, suffix: str) -> Path | None:
    if explicit:
        path = Path(explicit)
    elif out_dir:
        slug = run_id or "hardmax-teacher-distill"
        path = Path(out_dir) / f"{slug}{suffix}"
    else:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    runtime_args = base.Hyperparameters()
    teacher_config_paths = str(args.teacher_config).strip() or os.environ.get("EXTERNAL_TEACHER_CONFIG_PATHS", "").strip()
    teacher_checkpoint_paths = str(args.teacher_checkpoint).strip() or os.environ.get("EXTERNAL_TEACHER_CHECKPOINT_PATHS", "").strip()
    runtime_args.external_teacher_config_paths = teacher_config_paths
    runtime_args.external_teacher_checkpoint_paths = teacher_checkpoint_paths
    runtime_args.external_teacher_allow_partial_load = bool(
        int(os.environ.get("EXTERNAL_TEACHER_ALLOW_PARTIAL_LOAD", "0"))
    )
    runtime_args.external_teacher_min_param_fraction = float(
        os.environ.get("EXTERNAL_TEACHER_MIN_PARAM_FRACTION", "0.4")
    )
    if not runtime_args.external_teacher_config_paths or not runtime_args.external_teacher_checkpoint_paths:
        raise ValueError(
            "Teacher config and checkpoint must be provided via --teacher-config/--teacher-checkpoint or env "
            f"(resolved_config={teacher_config_paths!r} resolved_checkpoint={teacher_checkpoint_paths!r} argv={sys.argv!r})"
        )

    if not runtime_args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {runtime_args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=runtime_args.tokenizer_path)
    if int(sp.vocab_size()) != int(runtime_args.vocab_size):
        raise ValueError(
            f"VOCAB_SIZE={runtime_args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_name, _actual_train_files, _expected_train_files = base.validate_dataset_tokenizer_pair(
        runtime_args.data_path,
        runtime_args.tokenizer_path,
    )
    train_loader = base.build_train_loader(runtime_args, dataset_name=dataset_name)
    val_tokens = base.limit_validation_tokens(
        base.load_validation_tokens(runtime_args.val_files, runtime_args.train_seq_len),
        runtime_args.train_seq_len,
        args.val_max_seqs,
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    teacher_models, teacher_meta = base.load_external_teacher_models(runtime_args, sp, log_fn=log)
    if not teacher_models:
        raise ValueError("Failed to load any external teacher models")
    logic_phase_enabled = any(
        getattr(teacher_model, "logic_sidecar", None) is not None
        or getattr(teacher_model, "hardmax_structural_controller", None) is not None
        for teacher_model in teacher_models
    )

    model = HardmaxTeacherDraft(
        vocab_size=int(runtime_args.vocab_size),
        max_seq_len=int(runtime_args.train_seq_len),
        model_dim=args.model_dim,
        state_dim=args.state_dim,
        num_states=args.num_states,
        temperature=args.temperature,
    )
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda x, teacher_logits: model.loss_terms(
            x,
            teacher_logits,
            teacher_confidence_threshold=args.teacher_confidence_threshold,
            uncertain_token_weight=args.uncertain_token_weight,
            confidence_weight=args.confidence_weight,
            usage_balance_weight=args.usage_balance_weight,
            diversity_weight=args.diversity_weight,
        )[0],
    )

    batch_tokens = int(args.batch_size) * int(runtime_args.train_seq_len)
    val_batch_size = max(int(args.val_batch_size), 1)
    best_val: dict[str, float] | None = None
    final_train_metrics: dict[str, float] = {}
    train_t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        x_np, _y_np = train_loader.next_batch_np(batch_tokens, runtime_args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        teacher_logits, _teacher_metrics = base.ensemble_teacher_logits_for_batch(
            teacher_models,
            x,
            x_np,
            logic_phase_enabled=logic_phase_enabled,
            collect_metrics=False,
        )
        if teacher_logits is None:
            raise ValueError("Teacher logits unexpectedly missing during training")
        loss, grads = loss_and_grad(x, teacher_logits)
        optimizer_step(model, optimizer, grads)

        if step <= 5 or step % args.log_every == 0:
            final_train_metrics = eval_model(
                model,
                teacher_models=teacher_models,
                x_np=x_np,
                logic_phase_enabled=logic_phase_enabled,
                teacher_confidence_threshold=args.teacher_confidence_threshold,
                student_accept_threshold=args.student_accept_threshold,
            )
            elapsed = time.perf_counter() - train_t0
            print(
                " ".join(
                    [
                        f"step:{step}",
                        f"loss:{float(loss.item()):.4f}",
                        f"train_match:{final_train_metrics.get('teacher_match_acc', 0.0):.4f}",
                        f"train_match_conf:{final_train_metrics.get('teacher_match_confident_acc', 0.0):.4f}",
                        f"train_accept:{final_train_metrics.get('student_accept_frac', 0.0):.4f}",
                        f"train_accept_acc:{final_train_metrics.get('student_accept_acc', 0.0):.4f}",
                        f"train_conf:{final_train_metrics.get('student_confidence_mean', 0.0):.4f}",
                        f"hard_peak:{final_train_metrics.get('hard_usage_peak_frac', 0.0):.4f}",
                        f"soft_ppl:{final_train_metrics.get('soft_usage_perplexity', 0.0):.4f}",
                        f"elapsed_s:{elapsed:.1f}",
                    ]
                ),
                flush=True,
            )

        if step == args.steps or step % args.eval_every == 0:
            x_val_np, _y_val_np = sample_val_batch(
                val_tokens,
                seq_len=runtime_args.train_seq_len,
                batch_size=val_batch_size,
                rng=rng,
            )
            val_metrics = eval_model(
                model,
                teacher_models=teacher_models,
                x_np=x_val_np,
                logic_phase_enabled=logic_phase_enabled,
                teacher_confidence_threshold=args.teacher_confidence_threshold,
                student_accept_threshold=args.student_accept_threshold,
            )
            print(
                " ".join(
                    [
                        f"eval_step:{step}",
                        f"val_loss:{val_metrics.get('loss', 0.0):.4f}",
                        f"val_match:{val_metrics.get('teacher_match_acc', 0.0):.4f}",
                        f"val_match_conf:{val_metrics.get('teacher_match_confident_acc', 0.0):.4f}",
                        f"val_accept:{val_metrics.get('student_accept_frac', 0.0):.4f}",
                        f"val_accept_acc:{val_metrics.get('student_accept_acc', 0.0):.4f}",
                        f"val_conf:{val_metrics.get('student_confidence_mean', 0.0):.4f}",
                        f"val_hard_peak:{val_metrics.get('hard_usage_peak_frac', 0.0):.4f}",
                        f"val_soft_ppl:{val_metrics.get('soft_usage_perplexity', 0.0):.4f}",
                    ]
                ),
                flush=True,
            )
            if best_val is None or val_metrics.get("loss", float("inf")) < best_val.get("loss", float("inf")):
                best_val = dict(val_metrics)

    summary = {
        "config": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "val_batch_size": int(args.val_batch_size),
            "lr": float(args.lr),
            "model_dim": int(args.model_dim),
            "state_dim": int(args.state_dim),
            "num_states": int(args.num_states),
            "temperature": float(args.temperature),
            "teacher_confidence_threshold": float(args.teacher_confidence_threshold),
            "uncertain_token_weight": float(args.uncertain_token_weight),
            "confidence_weight": float(args.confidence_weight),
            "usage_balance_weight": float(args.usage_balance_weight),
            "diversity_weight": float(args.diversity_weight),
            "student_accept_threshold": float(args.student_accept_threshold),
        },
        "teacher_meta": teacher_meta,
        "final_train": final_train_metrics,
        "best_val": {} if best_val is None else best_val,
    }

    controller_out = resolve_output_path(args.controller_out, args.out_dir, args.run_id, "_hardmax_controller_init.npz")
    if controller_out is not None:
        mx.savez(str(controller_out), **export_controller_init_state(model))
        summary["controller_out"] = str(controller_out)

    model_out = resolve_output_path(args.model_out, args.out_dir, args.run_id, "_hardmax_teacher_draft_model.npz")
    if model_out is not None:
        mx.savez(str(model_out), **export_model_state(model))
        summary["model_out"] = str(model_out)

    summary_out = resolve_output_path(args.summary_out, args.out_dir, args.run_id, ".summary.json")
    if summary_out is not None:
        summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
