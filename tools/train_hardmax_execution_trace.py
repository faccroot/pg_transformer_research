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
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from execution_trace_dataset import generate_examples  # noqa: E402
from execution_trace_pretrain_dataset import (  # noqa: E402
    TracePretrainVocab,
    build_trace_pretrain_vocab,
    encode_trace_examples,
    load_examples_jsonl,
    pad_encoded_batch,
)
from execution_trace_verifier import (  # noqa: E402
    ROLLOUT_MODES,
    build_mixed_rollout_input_batch,
    build_rollout_input_batch,
)
from logic_register_mlx import CastedLinear, HardmaxStructuralController  # noqa: E402


def parse_task_family_mixture(spec: str) -> tuple[tuple[str, float], ...]:
    spec = spec.strip()
    if not spec:
        return ()
    pairs: list[tuple[str, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mixture item {item!r}; expected family:weight")
        name, weight = item.split(":", 1)
        pairs.append((name.strip(), float(weight)))
    return tuple(pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the hardmax structural controller on execution-trace supervision.")
    parser.add_argument("--config", default="", help="Optional staged JSON config produced by prepare_mlx_sweep.py")
    parser.add_argument("--trace-jsonl", default="", help="Optional JSONL trace corpus. If omitted, synthetic traces are generated on the fly.")
    parser.add_argument("--generate-examples", type=int, default=1024, help="Synthetic example count when --trace-jsonl is empty.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-family", choices=["straight", "branch", "loop", "nested", "mixed"], default="mixed")
    parser.add_argument("--task-family-mixture", default="", help="Optional weighted mixture for mixed-family sampling, e.g. 'straight:1,branch:1,loop:1,nested:2'")
    parser.add_argument("--max-statements", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-trace-steps", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--usage-balance-weight", type=float, default=1.0e-2)
    parser.add_argument("--diversity-weight", type=float, default=1.0e-2)
    parser.add_argument("--confidence-weight", type=float, default=5.0e-2)
    parser.add_argument("--write-weight", type=float, default=1.0)
    parser.add_argument("--read-weight", type=float, default=0.75)
    parser.add_argument("--read-count-weight", type=float, default=0.35)
    parser.add_argument("--write-count-weight", type=float, default=0.35)
    parser.add_argument("--step-weight", type=float, default=0.5)
    parser.add_argument("--branch-weight", type=float, default=0.5)
    parser.add_argument("--stack-depth-weight", type=float, default=0.35)
    parser.add_argument("--env-size-weight", type=float, default=0.35)
    parser.add_argument("--delta-weight", type=float, default=0.35)
    parser.add_argument("--output-weight", type=float, default=0.35)
    parser.add_argument("--rollout-consistency-weight", type=float, default=0.0)
    parser.add_argument("--rollout-consistency-horizon", type=int, default=1)
    parser.add_argument("--rollout-consistency-mode", choices=ROLLOUT_MODES, default="predicted_all")
    parser.add_argument("--scheduled-sampling-prob", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-mode", choices=ROLLOUT_MODES, default="predicted_opcode_step_plus_sizes")
    parser.add_argument("--scheduled-sampling-prefix-keep", type=int, default=1)
    parser.add_argument("--run-id", default="", help="Optional run identifier for staged sweeps.")
    parser.add_argument("--out-dir", default="", help="Optional output directory for staged sweeps.")
    parser.add_argument("--controller-out", default="", help="Optional path to save a controller-only NPZ init artifact.")
    parser.add_argument("--model-out", default="", help="Optional path to save a full pretrainer NPZ checkpoint.")
    parser.add_argument("--summary-out", default="", help="Optional path to a final metrics JSON file.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()
    if not args.config:
        return args

    config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_args = config_payload.get("args", {})
    config_env = config_payload.get("env", {})
    if not isinstance(config_args, dict):
        raise TypeError(f"Config args must be a dict, got {type(config_args)!r}")
    if not isinstance(config_env, dict):
        raise TypeError(f"Config env must be a dict, got {type(config_env)!r}")
    for key, value in config_env.items():
        os.environ[str(key)] = str(value)

    merged = vars(args).copy()
    default_vars = vars(defaults)
    for key, value in config_args.items():
        arg_name = str(key).replace("-", "_")
        if arg_name in merged and merged[arg_name] == default_vars.get(arg_name):
            merged[arg_name] = value
    return argparse.Namespace(**merged)


def masked_cross_entropy(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    flat_logits = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    flat_targets = targets.reshape(-1).astype(mx.int32)
    losses = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none").astype(mx.float32)
    losses = losses.reshape(targets.shape)
    weights = mask.astype(mx.float32)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1.0e-6, dtype=mx.float32))


def masked_accuracy(logits: mx.array, targets: mx.array, mask: mx.array) -> float:
    pred = mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)
    correct = (pred == targets.astype(mx.int32)).astype(mx.float32) * mask.astype(mx.float32)
    denom = mx.maximum(mx.sum(mask.astype(mx.float32)), mx.array(1.0e-6, dtype=mx.float32))
    return float((mx.sum(correct) / denom).item())


LOGIT_TO_INPUT_FIELD = {
    "opcode": "opcode_ids",
    "step_type": "step_type_ids",
    "read_var": "read_var_ids",
    "write_var": "write_var_ids",
    "read_count": "read_count_ids",
    "write_count": "write_count_ids",
    "branch": "branch_ids",
    "stack_depth": "stack_depth_ids",
    "env_size": "env_size_ids",
    "env_delta": "env_delta_size_ids",
    "output": "output_flag_ids",
}


def predicted_fields_np_from_logits(logits: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {
        input_field: np.asarray(mx.argmax(logit.astype(mx.float32), axis=-1).astype(mx.int32))
        for logit_name, input_field in LOGIT_TO_INPUT_FIELD.items()
        for logit in [logits[logit_name]]
    }


class TraceHardmaxPretrainer(nn.Module):
    def __init__(
        self,
        vocab: TracePretrainVocab,
        *,
        model_dim: int,
        state_dim: int,
        num_states: int,
        temperature: float,
    ):
        super().__init__()
        self.vocab = vocab
        self.model_dim = int(model_dim)
        self.opcode_emb = nn.Embedding(len(vocab.opcode_to_id), model_dim)
        self.step_type_emb = nn.Embedding(len(vocab.step_type_to_id), model_dim)
        self.read_var_emb = nn.Embedding(len(vocab.variable_to_id), model_dim)
        self.write_var_emb = nn.Embedding(len(vocab.variable_to_id), model_dim)
        self.read_count_emb = nn.Embedding(vocab.max_memop_bucket, model_dim)
        self.write_count_emb = nn.Embedding(vocab.max_memop_bucket, model_dim)
        self.branch_emb = nn.Embedding(len(vocab.branch_to_id), model_dim)
        self.stack_depth_emb = nn.Embedding(vocab.max_stack_bucket, model_dim)
        self.env_size_emb = nn.Embedding(vocab.max_env_bucket, model_dim)
        self.env_delta_emb = nn.Embedding(vocab.max_delta_bucket, model_dim)
        self.output_flag_emb = nn.Embedding(2, model_dim)
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
        self.opcode_head = CastedLinear(state_dim, len(vocab.opcode_to_id))
        self.step_type_head = CastedLinear(state_dim, len(vocab.step_type_to_id))
        self.read_var_head = CastedLinear(state_dim, len(vocab.variable_to_id))
        self.write_var_head = CastedLinear(state_dim, len(vocab.variable_to_id))
        self.read_count_head = CastedLinear(state_dim, vocab.max_memop_bucket)
        self.write_count_head = CastedLinear(state_dim, vocab.max_memop_bucket)
        self.branch_head = CastedLinear(state_dim, len(vocab.branch_to_id))
        self.stack_depth_head = CastedLinear(state_dim, vocab.max_stack_bucket)
        self.env_size_head = CastedLinear(state_dim, vocab.max_env_bucket)
        self.delta_head = CastedLinear(state_dim, vocab.max_delta_bucket)
        self.output_head = CastedLinear(state_dim, 2)

    def forward_features(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict[str, mx.array]]:
        x = (
            self.opcode_emb(batch["opcode_ids"])
            + self.step_type_emb(batch["step_type_ids"])
            + self.read_var_emb(batch["read_var_ids"])
            + self.write_var_emb(batch["write_var_ids"])
            + self.read_count_emb(batch["read_count_ids"])
            + self.write_count_emb(batch["write_count_ids"])
            + self.branch_emb(batch["branch_ids"])
            + self.stack_depth_emb(batch["stack_depth_ids"])
            + self.env_size_emb(batch["env_size_ids"])
            + self.env_delta_emb(batch["env_delta_size_ids"])
            + self.output_flag_emb(batch["output_flag_ids"])
        )
        x = nn.silu(x + self.mix(x))
        conditioned, aux = self.controller(x, None)
        return conditioned, aux

    def forward_logits(self, batch: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        _conditioned, aux = self.forward_features(batch)
        struct_state = aux["struct_state"]
        logits = {
            "opcode": self.opcode_head(struct_state),
            "step_type": self.step_type_head(struct_state),
            "read_var": self.read_var_head(struct_state),
            "write_var": self.write_var_head(struct_state),
            "read_count": self.read_count_head(struct_state),
            "write_count": self.write_count_head(struct_state),
            "branch": self.branch_head(struct_state),
            "stack_depth": self.stack_depth_head(struct_state),
            "env_size": self.env_size_head(struct_state),
            "env_delta": self.delta_head(struct_state),
            "output": self.output_head(struct_state),
        }
        return logits, aux

    def loss_terms(
        self,
        batch: dict[str, mx.array],
        *,
        step_weight: float,
        read_weight: float,
        write_weight: float,
        read_count_weight: float,
        write_count_weight: float,
        branch_weight: float,
        stack_depth_weight: float,
        env_size_weight: float,
        delta_weight: float,
        output_weight: float,
        rollout_consistency_weight: float,
        rollout_consistency_horizon: int,
        rollout_consistency_mode: str,
        usage_balance_weight: float,
        diversity_weight: float,
        confidence_weight: float,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        logits, aux = self.forward_logits(batch)
        mask = batch["valid_mask"].astype(mx.float32)

        def prediction_losses(
            logits_dict: dict[str, mx.array],
            target_batch: dict[str, mx.array],
            target_mask: mx.array,
        ) -> tuple[mx.array, dict[str, mx.array]]:
            opcode_loss = masked_cross_entropy(logits_dict["opcode"], target_batch["target_next_opcode_ids"], target_mask)
            step_loss = masked_cross_entropy(logits_dict["step_type"], target_batch["target_next_step_type_ids"], target_mask)
            read_loss = masked_cross_entropy(logits_dict["read_var"], target_batch["target_next_read_var_ids"], target_mask)
            write_loss = masked_cross_entropy(logits_dict["write_var"], target_batch["target_next_write_var_ids"], target_mask)
            read_count_loss = masked_cross_entropy(logits_dict["read_count"], target_batch["target_next_read_count_ids"], target_mask)
            write_count_loss = masked_cross_entropy(logits_dict["write_count"], target_batch["target_next_write_count_ids"], target_mask)
            branch_loss = masked_cross_entropy(logits_dict["branch"], target_batch["target_next_branch_ids"], target_mask)
            stack_depth_loss = masked_cross_entropy(logits_dict["stack_depth"], target_batch["target_next_stack_depth_ids"], target_mask)
            env_size_loss = masked_cross_entropy(logits_dict["env_size"], target_batch["target_next_env_size_ids"], target_mask)
            delta_loss = masked_cross_entropy(logits_dict["env_delta"], target_batch["target_next_env_delta_size_ids"], target_mask)
            output_loss = masked_cross_entropy(logits_dict["output"], target_batch["target_next_output_flag_ids"], target_mask)
            total_prediction = (
                opcode_loss
                + step_weight * step_loss
                + read_weight * read_loss
                + write_weight * write_loss
                + read_count_weight * read_count_loss
                + write_count_weight * write_count_loss
                + branch_weight * branch_loss
                + stack_depth_weight * stack_depth_loss
                + env_size_weight * env_size_loss
                + delta_weight * delta_loss
                + output_weight * output_loss
            ).astype(mx.float32)
            return total_prediction, {
                "loss_opcode": opcode_loss.astype(mx.float32),
                "loss_step": step_loss.astype(mx.float32),
                "loss_read": read_loss.astype(mx.float32),
                "loss_write": write_loss.astype(mx.float32),
                "loss_read_count": read_count_loss.astype(mx.float32),
                "loss_write_count": write_count_loss.astype(mx.float32),
                "loss_branch": branch_loss.astype(mx.float32),
                "loss_stack_depth": stack_depth_loss.astype(mx.float32),
                "loss_env_size": env_size_loss.astype(mx.float32),
                "loss_delta": delta_loss.astype(mx.float32),
                "loss_output": output_loss.astype(mx.float32),
            }

        prediction_total, prediction_metrics = prediction_losses(logits, batch, mask)

        probs = [mx.softmax(logits[name].astype(mx.float32), axis=-1) for name in logits]
        norm_entropies: list[mx.array] = []
        for prob in probs:
            entropy = -mx.sum(prob * mx.log(mx.maximum(prob, mx.array(1.0e-8, dtype=mx.float32))), axis=-1)
            norm = float(np.log(max(int(prob.shape[-1]), 2)))
            norm_entropies.append(entropy / norm)
        mean_norm_entropy = sum(norm_entropies) / len(norm_entropies)
        confidence = aux["confidence"].astype(mx.float32)
        conf_sq = mx.square(confidence - mx.stop_gradient(1.0 - mean_norm_entropy.astype(mx.float32)))
        conf_loss = mx.sum(conf_sq * mask) / mx.maximum(mx.sum(mask), mx.array(1.0e-6, dtype=mx.float32))

        usage_balance, diversity, entropy = self.controller.regularization_losses(aux)
        rollout_loss = mx.array(0.0, dtype=mx.float32)
        if rollout_consistency_weight > 0.0 and rollout_consistency_horizon > 0:
            actual_batch_np = {
                key: np.asarray(value)
                for key, value in batch.items()
            }
            current_pred_by_field = predicted_fields_np_from_logits(logits)
            for rollout_step in range(1, int(rollout_consistency_horizon) + 1):
                current_rollout_inputs = build_rollout_input_batch(
                    actual_batch_np,
                    current_pred_by_field,
                    self.vocab,
                    rollout_consistency_mode,
                )
                rollout_batch_np = {
                    key: np.array(value, copy=True)
                    for key, value in actual_batch_np.items()
                }
                rollout_batch_np.update(current_rollout_inputs)
                rollout_mask_np = np.array(actual_batch_np["valid_mask"], copy=True)
                rollout_mask_np[:, : min(rollout_step, rollout_mask_np.shape[1])] = 0.0
                rollout_batch_np["valid_mask"] = rollout_mask_np.astype(np.float32, copy=False)
                rollout_batch = batch_to_mx(rollout_batch_np)
                rollout_logits, _rollout_aux = self.forward_logits(rollout_batch)
                rollout_prediction_total, _rollout_prediction_metrics = prediction_losses(
                    rollout_logits,
                    rollout_batch,
                    rollout_batch["valid_mask"].astype(mx.float32),
                )
                rollout_loss = rollout_loss + rollout_prediction_total
                current_pred_by_field = predicted_fields_np_from_logits(rollout_logits)
            rollout_loss = (rollout_loss / float(max(int(rollout_consistency_horizon), 1))).astype(mx.float32)

        total = (
            prediction_total
            + rollout_consistency_weight * rollout_loss
            + usage_balance_weight * usage_balance
            + diversity_weight * diversity
            + confidence_weight * conf_loss
        ).astype(mx.float32)
        metrics = {
            "loss_total": total,
            "loss_prediction": prediction_total.astype(mx.float32),
            **prediction_metrics,
            "loss_rollout": rollout_loss.astype(mx.float32),
            "loss_confidence": conf_loss.astype(mx.float32),
            "loss_usage_balance": usage_balance.astype(mx.float32),
            "loss_diversity": diversity.astype(mx.float32),
            "entropy": entropy.astype(mx.float32),
        }
        return total, metrics


def batch_to_mx(batch: dict[str, np.ndarray]) -> dict[str, mx.array]:
    return {key: mx.array(value) for key, value in batch.items()}


def sample_batch(
    sequences,
    vocab: TracePretrainVocab,
    *,
    batch_size: int,
    rng: random.Random,
) -> dict[str, mx.array]:
    chosen = [sequences[rng.randrange(len(sequences))] for _ in range(batch_size)]
    return batch_to_mx(pad_encoded_batch(chosen, vocab))


def maybe_apply_scheduled_sampling(
    model: TraceHardmaxPretrainer,
    batch: dict[str, mx.array],
    vocab: TracePretrainVocab,
    *,
    replace_prob: float,
    mode: str,
    min_teacher_prefix: int,
    np_rng: np.random.Generator,
) -> tuple[dict[str, mx.array], float]:
    if replace_prob <= 0.0:
        return batch, 0.0
    logits, _aux = model.forward_logits(batch)
    predicted_batch = predicted_fields_np_from_logits(logits)
    actual_batch_np = {key: np.asarray(value) for key, value in batch.items()}
    mixed_inputs, replace_mask = build_mixed_rollout_input_batch(
        actual_batch_np,
        predicted_batch,
        vocab,
        mode,
        replace_prob=float(replace_prob),
        rng=np_rng,
        min_teacher_prefix=int(min_teacher_prefix),
    )
    mixed_batch_np = {
        key: np.array(value, copy=True)
        for key, value in actual_batch_np.items()
    }
    mixed_batch_np.update(mixed_inputs)
    replaced_fraction = float(replace_mask.astype(np.float32).mean())
    return batch_to_mx(mixed_batch_np), replaced_fraction


def eval_model(
    model: TraceHardmaxPretrainer,
    sequences,
    vocab: TracePretrainVocab,
    args: argparse.Namespace,
    *,
    batch_size: int,
) -> dict[str, float]:
    if not sequences:
        return {}
    batch = batch_to_mx(pad_encoded_batch(sequences[:batch_size], vocab))
    logits, aux = model.forward_logits(batch)
    loss, metrics = model.loss_terms(
        batch,
        step_weight=args.step_weight,
        read_weight=args.read_weight,
        write_weight=args.write_weight,
        read_count_weight=args.read_count_weight,
        write_count_weight=args.write_count_weight,
        branch_weight=args.branch_weight,
        stack_depth_weight=args.stack_depth_weight,
        env_size_weight=args.env_size_weight,
        delta_weight=args.delta_weight,
        output_weight=args.output_weight,
        rollout_consistency_weight=args.rollout_consistency_weight,
        rollout_consistency_horizon=args.rollout_consistency_horizon,
        rollout_consistency_mode=args.rollout_consistency_mode,
        usage_balance_weight=args.usage_balance_weight,
        diversity_weight=args.diversity_weight,
        confidence_weight=args.confidence_weight,
    )
    mask = batch["valid_mask"].astype(mx.float32)
    summary = {
        "loss": float(loss.item()),
        "opcode_acc": masked_accuracy(logits["opcode"], batch["target_next_opcode_ids"], mask),
        "step_acc": masked_accuracy(logits["step_type"], batch["target_next_step_type_ids"], mask),
        "read_acc": masked_accuracy(logits["read_var"], batch["target_next_read_var_ids"], mask),
        "write_acc": masked_accuracy(logits["write_var"], batch["target_next_write_var_ids"], mask),
        "read_count_acc": masked_accuracy(logits["read_count"], batch["target_next_read_count_ids"], mask),
        "write_count_acc": masked_accuracy(logits["write_count"], batch["target_next_write_count_ids"], mask),
        "branch_acc": masked_accuracy(logits["branch"], batch["target_next_branch_ids"], mask),
        "stack_depth_acc": masked_accuracy(logits["stack_depth"], batch["target_next_stack_depth_ids"], mask),
        "env_size_acc": masked_accuracy(logits["env_size"], batch["target_next_env_size_ids"], mask),
        "delta_acc": masked_accuracy(logits["env_delta"], batch["target_next_env_delta_size_ids"], mask),
        "output_acc": masked_accuracy(logits["output"], batch["target_next_output_flag_ids"], mask),
        "confidence_mean": float(mx.mean(aux["confidence"].astype(mx.float32)).item()),
        "budget_mean": float(mx.mean(aux["budget"].astype(mx.float32)).item()),
        "hard_usage_peak_frac": float(
            mx.max(mx.mean(aux["hard_usage"].astype(mx.float32), axis=(0, 1))).item()
        ),
        "soft_usage_perplexity": float(
            mx.exp(
                -mx.sum(
                    mx.mean(aux["soft_usage"].astype(mx.float32), axis=(0, 1))
                    * mx.log(
                        mx.maximum(
                            mx.mean(aux["soft_usage"].astype(mx.float32), axis=(0, 1)),
                            mx.array(1.0e-8, dtype=mx.float32),
                        )
                    )
                )
            ).item()
        ),
    }
    for key, value in metrics.items():
        summary[key] = float(value.item())
    return summary


def optimizer_step(model: nn.Module, optimizer: optim.Adam, grads_tree) -> None:
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads_tree))
    updated = optimizer.apply_gradients(grads, params)
    model.update(tree_unflatten(list(updated.items())))


def export_controller_init_state(model: TraceHardmaxPretrainer) -> dict[str, mx.array]:
    return {
        f"hardmax_structural_controller.{name}": value
        for name, value in tree_flatten(model.controller.state)
    }


def export_full_model_state(model: TraceHardmaxPretrainer) -> dict[str, mx.array]:
    return {
        str(name): value
        for name, value in tree_flatten(model.state)
    }


def load_or_generate_examples(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.trace_jsonl:
        return load_examples_jsonl(args.trace_jsonl)
    return generate_examples(
        args.generate_examples,
        seed=args.seed,
        max_statements=args.max_statements,
        max_depth=args.max_depth,
        max_trace_steps=args.max_trace_steps,
        task_family=args.task_family,
        task_family_mixture=parse_task_family_mixture(args.task_family_mixture),
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    examples = load_or_generate_examples(args)
    if len(examples) < 4:
        raise ValueError("Need at least 4 examples for a trace pretrain split")
    split = max(1, int(round(len(examples) * (1.0 - args.val_fraction))))
    split = min(max(split, 1), len(examples) - 1)
    train_examples = examples[:split]
    val_examples = examples[split:]

    vocab = build_trace_pretrain_vocab(examples)
    train_sequences = encode_trace_examples(train_examples, vocab)
    val_sequences = encode_trace_examples(val_examples, vocab)

    model = TraceHardmaxPretrainer(
        vocab,
        model_dim=args.model_dim,
        state_dim=args.state_dim,
        num_states=args.num_states,
        temperature=args.temperature,
    )
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda batch: model.loss_terms(
            batch,
            step_weight=args.step_weight,
            read_weight=args.read_weight,
            write_weight=args.write_weight,
            read_count_weight=args.read_count_weight,
            write_count_weight=args.write_count_weight,
            branch_weight=args.branch_weight,
            stack_depth_weight=args.stack_depth_weight,
            env_size_weight=args.env_size_weight,
            delta_weight=args.delta_weight,
            output_weight=args.output_weight,
            rollout_consistency_weight=args.rollout_consistency_weight,
            rollout_consistency_horizon=args.rollout_consistency_horizon,
            rollout_consistency_mode=args.rollout_consistency_mode,
            usage_balance_weight=args.usage_balance_weight,
            diversity_weight=args.diversity_weight,
            confidence_weight=args.confidence_weight,
        )[0],
    )

    best_val: dict[str, float] | None = None
    train_metrics_last: dict[str, float] = {}
    train_scheduled_fraction_last = 0.0
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = sample_batch(train_sequences, vocab, batch_size=args.batch_size, rng=rng)
        batch, train_scheduled_fraction_last = maybe_apply_scheduled_sampling(
            model,
            batch,
            vocab,
            replace_prob=float(args.scheduled_sampling_prob),
            mode=str(args.scheduled_sampling_mode),
            min_teacher_prefix=int(args.scheduled_sampling_prefix_keep),
            np_rng=np_rng,
        )
        loss, grads = loss_and_grad(batch)
        optimizer_step(model, optimizer, grads)

        if step <= 5 or step % args.log_every == 0:
            train_metrics_last = eval_model(model, train_sequences, vocab, args, batch_size=min(args.batch_size, len(train_sequences)))
            elapsed = time.perf_counter() - t0
            print(
                " ".join(
                    [
                        f"step:{step}",
                        f"loss:{float(loss.item()):.4f}",
                        f"train_opcode_acc:{train_metrics_last.get('opcode_acc', 0.0):.4f}",
                        f"train_write_acc:{train_metrics_last.get('write_acc', 0.0):.4f}",
                        f"train_stack_acc:{train_metrics_last.get('stack_depth_acc', 0.0):.4f}",
                        f"train_env_acc:{train_metrics_last.get('env_size_acc', 0.0):.4f}",
                        f"train_rollout:{train_metrics_last.get('loss_rollout', 0.0):.4f}",
                        f"train_ss_frac:{train_scheduled_fraction_last:.4f}",
                        f"train_conf:{train_metrics_last.get('confidence_mean', 0.0):.4f}",
                        f"hard_peak:{train_metrics_last.get('hard_usage_peak_frac', 0.0):.4f}",
                        f"soft_ppl:{train_metrics_last.get('soft_usage_perplexity', 0.0):.4f}",
                        f"elapsed_s:{elapsed:.1f}",
                    ]
                ),
                flush=True,
            )

        if step == args.steps or step % args.eval_every == 0:
            val_metrics = eval_model(model, val_sequences, vocab, args, batch_size=min(args.batch_size, len(val_sequences)))
            print(
                " ".join(
                    [
                        f"eval_step:{step}",
                        f"val_loss:{val_metrics.get('loss', 0.0):.4f}",
                        f"val_opcode_acc:{val_metrics.get('opcode_acc', 0.0):.4f}",
                        f"val_write_acc:{val_metrics.get('write_acc', 0.0):.4f}",
                        f"val_stack_acc:{val_metrics.get('stack_depth_acc', 0.0):.4f}",
                        f"val_env_acc:{val_metrics.get('env_size_acc', 0.0):.4f}",
                        f"val_rollout:{val_metrics.get('loss_rollout', 0.0):.4f}",
                        f"val_conf:{val_metrics.get('confidence_mean', 0.0):.4f}",
                        f"val_hard_peak:{val_metrics.get('hard_usage_peak_frac', 0.0):.4f}",
                        f"val_soft_ppl:{val_metrics.get('soft_usage_perplexity', 0.0):.4f}",
                    ]
                ),
                flush=True,
            )
            if best_val is None or val_metrics.get("loss", float("inf")) < best_val.get("loss", float("inf")):
                best_val = dict(val_metrics)

    summary = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "config": {
            "model_dim": args.model_dim,
            "state_dim": args.state_dim,
            "num_states": args.num_states,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "rollout_consistency_weight": args.rollout_consistency_weight,
            "rollout_consistency_horizon": args.rollout_consistency_horizon,
            "rollout_consistency_mode": args.rollout_consistency_mode,
            "scheduled_sampling_prob": args.scheduled_sampling_prob,
            "scheduled_sampling_mode": args.scheduled_sampling_mode,
            "scheduled_sampling_prefix_keep": args.scheduled_sampling_prefix_keep,
        },
        "generation_config": {
            "trace_jsonl": args.trace_jsonl,
            "generate_examples": args.generate_examples,
            "seed": args.seed,
            "task_family": args.task_family,
            "task_family_mixture": args.task_family_mixture,
            "max_statements": args.max_statements,
            "max_depth": args.max_depth,
            "max_trace_steps": args.max_trace_steps,
            "val_fraction": args.val_fraction,
        },
        "final_train": train_metrics_last,
        "best_val": {} if best_val is None else best_val,
        "vocab_sizes": {
            "opcode": len(vocab.opcode_to_id),
            "step_type": len(vocab.step_type_to_id),
            "variable": len(vocab.variable_to_id),
            "branch": len(vocab.branch_to_id),
        },
    }
    model_out_str = args.model_out
    if not model_out_str and args.out_dir:
        run_slug = args.run_id or "hardmax-trace-pretrain"
        model_out_str = str(Path(args.out_dir) / f"{run_slug}_trace_pretrain_model.npz")
    if model_out_str:
        model_out = Path(model_out_str)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(str(model_out), **export_full_model_state(model))
        summary["model_out"] = str(model_out)
    controller_out_str = args.controller_out
    if not controller_out_str and args.out_dir:
        run_slug = args.run_id or "hardmax-trace-pretrain"
        controller_out_str = str(Path(args.out_dir) / f"{run_slug}_hardmax_controller_init.npz")
    if controller_out_str:
        controller_out = Path(controller_out_str)
        controller_out.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(str(controller_out), **export_controller_init_state(model))
        summary["controller_out"] = str(controller_out)
    summary_path_str = args.summary_out
    if not summary_path_str and args.out_dir:
        run_slug = args.run_id or "hardmax-trace-pretrain"
        summary_path_str = str(Path(args.out_dir) / f"{run_slug}.summary.json")
    if summary_path_str:
        summary_path = Path(summary_path_str)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
