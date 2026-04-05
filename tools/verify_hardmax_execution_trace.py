#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

from execution_trace_dataset import generate_examples  # noqa: E402
from execution_trace_pretrain_dataset import (  # noqa: E402
    TracePretrainVocab,
    build_trace_pretrain_vocab,
    encode_trace_examples,
    load_examples_jsonl,
    pad_encoded_batch,
)
from execution_trace_verifier import (  # noqa: E402
    INPUT_FIELDS,
    PREDICTED_FIELD_TARGET_PAIRS,
    ROLLOUT_MODES,
    TEACHER_FORCED_ABLATION_MODES,
    apply_teacher_forced_input_ablation,
    build_rollout_next_input_row,
    predicted_fields_exact_for_step,
    summarize_rollout_failures,
)
import train_hardmax_execution_trace as trace_train  # noqa: E402


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

SEQUENCE_ARRAY_FIELDS = INPUT_FIELDS + tuple(target for _field, target in PREDICTED_FIELD_TARGET_PAIRS) + ("valid_mask",)


def build_parser() -> argparse.ArgumentParser:
    parser = trace_train.build_parser()
    parser.description = (
        "Verify whether a pretrained hardmax trace controller generalizes on held-out programs, "
        "including input ablations and semi-open-loop rollouts."
    )
    parser.add_argument("--artifact", required=True, help="Path to a full TraceHardmaxPretrainer NPZ checkpoint.")
    parser.add_argument("--result-json", required=True, help="Output path for verification results.")
    parser.add_argument("--label", default="", help="Optional run label.")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--rollout-max-sequences", type=int, default=64)
    parser.add_argument(
        "--teacher-forced-modes",
        default="none,drop_memops,drop_branch,drop_delta_output,opcode_step_plus_sizes,opcode_step_only",
        help="Comma-separated teacher-forced ablation modes.",
    )
    parser.add_argument(
        "--rollout-modes",
        default="predicted_all,predicted_opcode_step_plus_sizes,predicted_opcode_step_only,predicted_all_oracle_sizes,predicted_opcode_step_only_oracle_sizes",
        help="Comma-separated rollout modes.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    defaults = parser.parse_args(["--artifact", "_", "--result-json", "_"])
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


def parse_mode_list(spec: str, allowed: tuple[str, ...]) -> list[str]:
    items = [item.strip() for item in str(spec).split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown modes {unknown!r}; allowed={allowed!r}")
    return items


def load_examples(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.trace_jsonl:
        return load_examples_jsonl(args.trace_jsonl)
    return generate_examples(
        int(args.generate_examples),
        seed=int(args.seed),
        max_statements=int(args.max_statements),
        max_depth=int(args.max_depth),
        max_trace_steps=int(args.max_trace_steps),
        task_family=str(args.task_family),
        task_family_mixture=trace_train.parse_task_family_mixture(str(args.task_family_mixture)),
    )


def split_examples(examples: list[dict[str, object]], val_fraction: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(examples) < 4:
        raise ValueError("Need at least 4 examples for a trace pretrain split")
    split = max(1, int(round(len(examples) * (1.0 - float(val_fraction)))))
    split = min(max(split, 1), len(examples) - 1)
    return examples[:split], examples[split:]


def build_model(args: argparse.Namespace, vocab: TracePretrainVocab, artifact_path: Path):
    model = trace_train.TraceHardmaxPretrainer(
        vocab,
        model_dim=int(args.model_dim),
        state_dim=int(args.state_dim),
        num_states=int(args.num_states),
        temperature=float(args.temperature),
    )
    flat = np.load(str(artifact_path))
    keys = set(flat.files)
    required = {
        "opcode_emb.weight",
        "step_type_emb.weight",
        "read_var_emb.weight",
        "write_var_emb.weight",
        "controller.state_book",
        "opcode_head.weight",
    }
    if not required.issubset(keys):
        missing = sorted(required - keys)
        raise ValueError(
            "Artifact does not look like a full TraceHardmaxPretrainer checkpoint; "
            f"missing keys: {missing}. Controller-only init exports are insufficient for verification."
        )
    state_items = [(name, mx.array(value)) for name, value in flat.items()]
    model.update(tree_unflatten(state_items))
    return model


def batch_predictions_np(model, batch_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    logits, _aux = model.forward_logits(trace_train.batch_to_mx(batch_np))
    out: dict[str, np.ndarray] = {}
    for logit_name, input_field in LOGIT_TO_INPUT_FIELD.items():
        out[input_field] = np.asarray(
            mx.argmax(logits[logit_name].astype(mx.float32), axis=-1).astype(mx.int32)
        )
    return out


def teacher_forced_metrics(
    model,
    sequences,
    vocab: TracePretrainVocab,
    *,
    batch_size: int,
    modes: list[str],
) -> dict[str, object]:
    results: dict[str, object] = {}
    field_names = [field for field, _target in PREDICTED_FIELD_TARGET_PAIRS]
    for mode in modes:
        correct_by_field = {field: 0.0 for field in field_names}
        total_by_field = {field: 0.0 for field in field_names}
        seq_exact = 0
        seq_total = 0
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            if not batch_seqs:
                continue
            actual_batch = pad_encoded_batch(batch_seqs, vocab)
            input_batch = apply_teacher_forced_input_ablation(actual_batch, vocab, mode)
            pred_by_field = batch_predictions_np(model, input_batch)
            mask = np.asarray(actual_batch["valid_mask"], dtype=np.float32)
            batch_exact = np.ones(mask.shape, dtype=np.bool_)
            for field, target_name in PREDICTED_FIELD_TARGET_PAIRS:
                preds = pred_by_field[field]
                target = np.asarray(actual_batch[target_name], dtype=np.int32)
                correct = (preds == target) & (mask > 0.0)
                correct_by_field[field] += float(correct.sum())
                total_by_field[field] += float((mask > 0.0).sum())
                batch_exact &= ((preds == target) | ~(mask > 0.0))
            seq_exact += int(batch_exact.all(axis=1).sum())
            seq_total += int(batch_exact.shape[0])
        results[mode] = {
            "field_accuracy": {
                field: (correct_by_field[field] / total_by_field[field] if total_by_field[field] > 0.0 else None)
                for field in field_names
            },
            "sequence_exact_fraction": (float(seq_exact) / float(seq_total)) if seq_total > 0 else None,
            "sequence_count": int(seq_total),
        }
    return results


def sequence_to_np(seq) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(getattr(seq, field), dtype=np.int32 if field != "valid_mask" else np.float32)
        for field in SEQUENCE_ARRAY_FIELDS
    }


def prefix_batch_from_rows(prefix_rows: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray([values], dtype=np.int32)
        for field, values in prefix_rows.items()
    }


def last_step_prediction(model, prefix_rows: dict[str, list[int]]) -> dict[str, int]:
    batch_np = prefix_batch_from_rows(prefix_rows)
    pred = batch_predictions_np(model, batch_np)
    return {field: int(values[0, -1]) for field, values in pred.items()}


def rollout_metrics(
    model,
    sequences,
    vocab: TracePretrainVocab,
    *,
    modes: list[str],
    max_sequences: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    field_names = [field for field, _target in PREDICTED_FIELD_TARGET_PAIRS]
    selected = list(sequences[:max_sequences])
    for mode in modes:
        correct_by_field = {field: 0.0 for field in field_names}
        total_by_field = {field: 0.0 for field in field_names}
        full_trace_exact = 0
        first_failures: list[int] = []
        step_exact_by_depth: dict[int, list[int]] = {}
        evaluated = 0
        for seq in selected:
            actual = sequence_to_np(seq)
            seq_len = int(actual["opcode_ids"].shape[0])
            valid_steps = int(np.asarray(actual["valid_mask"], dtype=np.float32).sum())
            if valid_steps <= 0:
                continue
            prefix_rows = {
                field: [int(actual[field][0])]
                for field in INPUT_FIELDS
            }
            step_exact_flags: list[bool] = []
            for step in range(valid_steps):
                predicted_next = last_step_prediction(model, prefix_rows)
                for field, _target in PREDICTED_FIELD_TARGET_PAIRS:
                    total_by_field[field] += 1.0
                    if int(predicted_next[field]) == int(actual[field][step + 1]):
                        correct_by_field[field] += 1.0
                step_exact = predicted_fields_exact_for_step(predicted_next, actual, step)
                step_exact_flags.append(step_exact)
                depth = step + 1
                step_exact_by_depth.setdefault(depth, [0, 0])
                step_exact_by_depth[depth][0] += int(step_exact)
                step_exact_by_depth[depth][1] += 1
                if step + 1 >= seq_len - 1:
                    continue
                next_row = build_rollout_next_input_row(actual, step + 1, predicted_next, vocab, mode)
                for field, value in next_row.items():
                    prefix_rows[field].append(int(value))
            summary = summarize_rollout_failures(step_exact_flags)
            full_trace_exact += int(summary.full_trace_exact)
            if summary.first_failure_step is not None:
                first_failures.append(int(summary.first_failure_step))
            evaluated += 1
        results[mode] = {
            "field_accuracy": {
                field: (correct_by_field[field] / total_by_field[field] if total_by_field[field] > 0.0 else None)
                for field in field_names
            },
            "full_trace_exact_fraction": (float(full_trace_exact) / float(evaluated)) if evaluated > 0 else None,
            "first_failure_step_mean": (float(np.mean(first_failures)) if first_failures else None),
            "step_exact_by_depth": {
                str(depth): (float(correct) / float(total) if total > 0 else None)
                for depth, (correct, total) in sorted(step_exact_by_depth.items())
            },
            "sequence_count": int(evaluated),
        }
    return results


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact).expanduser().resolve()
    examples = load_examples(args)
    train_examples, val_examples = split_examples(examples, float(args.val_fraction))
    vocab = build_trace_pretrain_vocab(examples)
    train_sequences = encode_trace_examples(train_examples, vocab)
    val_sequences = encode_trace_examples(val_examples, vocab)

    if args.split == "train":
        sequences = train_sequences
    elif args.split == "val":
        sequences = val_sequences
    else:
        sequences = train_sequences + val_sequences
    if int(args.max_sequences) > 0:
        sequences = sequences[: int(args.max_sequences)]

    model = build_model(args, vocab, artifact_path)
    teacher_modes = parse_mode_list(str(args.teacher_forced_modes), TEACHER_FORCED_ABLATION_MODES)
    rollout_modes = parse_mode_list(str(args.rollout_modes), ROLLOUT_MODES)
    teacher_results = teacher_forced_metrics(
        model,
        sequences,
        vocab,
        batch_size=max(1, min(int(args.batch_size), len(sequences) or 1)),
        modes=teacher_modes,
    )
    rollout_results = rollout_metrics(
        model,
        sequences,
        vocab,
        modes=rollout_modes,
        max_sequences=max(1, min(int(args.rollout_max_sequences), len(sequences) or 1)),
    )

    payload = {
        "label": str(args.label),
        "artifact": str(artifact_path),
        "split": str(args.split),
        "counts": {
            "examples_total": int(len(examples)),
            "examples_train": int(len(train_examples)),
            "examples_val": int(len(val_examples)),
            "sequences_evaluated": int(len(sequences)),
        },
        "model_config": {
            "model_dim": int(args.model_dim),
            "state_dim": int(args.state_dim),
            "num_states": int(args.num_states),
            "temperature": float(args.temperature),
        },
        "generation_config": {
            "trace_jsonl": str(args.trace_jsonl),
            "generate_examples": int(args.generate_examples),
            "seed": int(args.seed),
            "task_family": str(args.task_family),
            "task_family_mixture": str(args.task_family_mixture),
            "max_statements": int(args.max_statements),
            "max_depth": int(args.max_depth),
            "max_trace_steps": int(args.max_trace_steps),
            "val_fraction": float(args.val_fraction),
        },
        "task_boundary": {
            "supports_raw_program_execution_claim": False,
            "reason": (
                "Current pretrainer consumes rich current-trace features, not raw bytecode alone, "
                "and it does not predict next stack depth or env size. The strongest current test is "
                "held-out transition dynamics plus semi-open-loop rollout with oracle size carry."
            ),
        },
        "teacher_forced": teacher_results,
        "semi_open_loop_rollout": rollout_results,
    }
    result_path = Path(args.result_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
