from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_trace_pretrain_dataset import TracePretrainVocab


PREDICTED_FIELD_TARGET_PAIRS: tuple[tuple[str, str], ...] = (
    ("opcode_ids", "target_next_opcode_ids"),
    ("step_type_ids", "target_next_step_type_ids"),
    ("read_var_ids", "target_next_read_var_ids"),
    ("write_var_ids", "target_next_write_var_ids"),
    ("read_count_ids", "target_next_read_count_ids"),
    ("write_count_ids", "target_next_write_count_ids"),
    ("branch_ids", "target_next_branch_ids"),
    ("stack_depth_ids", "target_next_stack_depth_ids"),
    ("env_size_ids", "target_next_env_size_ids"),
    ("env_delta_size_ids", "target_next_env_delta_size_ids"),
    ("output_flag_ids", "target_next_output_flag_ids"),
)

INPUT_FIELDS: tuple[str, ...] = (
    "opcode_ids",
    "step_type_ids",
    "read_var_ids",
    "write_var_ids",
    "read_count_ids",
    "write_count_ids",
    "branch_ids",
    "stack_depth_ids",
    "env_size_ids",
    "env_delta_size_ids",
    "output_flag_ids",
)


TEACHER_FORCED_ABLATION_MODES: tuple[str, ...] = (
    "none",
    "drop_memops",
    "drop_branch",
    "drop_delta_output",
    "opcode_step_plus_sizes",
    "opcode_step_only",
)


ROLLOUT_MODES: tuple[str, ...] = (
    "predicted_all",
    "predicted_opcode_step_plus_sizes",
    "predicted_opcode_step_only",
    "predicted_all_oracle_sizes",
    "predicted_opcode_step_only_oracle_sizes",
)

SIZE_INPUT_FIELDS: tuple[str, ...] = ("stack_depth_ids", "env_size_ids")


@dataclass(frozen=True)
class RolloutFailureSummary:
    full_trace_exact: bool
    first_failure_step: int | None


def copy_batch_np(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in batch.items()}


def apply_teacher_forced_input_ablation(
    batch: dict[str, np.ndarray],
    vocab: TracePretrainVocab,
    mode: str,
) -> dict[str, np.ndarray]:
    out = copy_batch_np(batch)
    if mode == "none":
        return out
    if mode in {"drop_memops", "opcode_step_plus_sizes", "opcode_step_only"}:
        out["read_var_ids"].fill(vocab.none_variable_id)
        out["write_var_ids"].fill(vocab.none_variable_id)
        out["read_count_ids"].fill(0)
        out["write_count_ids"].fill(0)
    if mode in {"drop_branch", "opcode_step_plus_sizes", "opcode_step_only"}:
        out["branch_ids"].fill(vocab.none_branch_id)
    if mode in {"drop_delta_output", "opcode_step_plus_sizes", "opcode_step_only"}:
        out["env_delta_size_ids"].fill(0)
        out["output_flag_ids"].fill(0)
    if mode == "opcode_step_only":
        out["stack_depth_ids"].fill(0)
        out["env_size_ids"].fill(0)
    if mode not in TEACHER_FORCED_ABLATION_MODES:
        raise ValueError(f"Unsupported teacher-forced ablation mode: {mode}")
    return out


def neutral_input_row(
    actual_batch: dict[str, np.ndarray],
    next_index: int,
    vocab: TracePretrainVocab,
    *,
    keep_stack_env: bool,
) -> dict[str, int]:
    row = {
        "opcode_ids": int(actual_batch["opcode_ids"][next_index]),
        "step_type_ids": int(actual_batch["step_type_ids"][next_index]),
        "read_var_ids": int(vocab.none_variable_id),
        "write_var_ids": int(vocab.none_variable_id),
        "read_count_ids": 0,
        "write_count_ids": 0,
        "branch_ids": int(vocab.none_branch_id),
        "stack_depth_ids": int(actual_batch["stack_depth_ids"][next_index]) if keep_stack_env else 0,
        "env_size_ids": int(actual_batch["env_size_ids"][next_index]) if keep_stack_env else 0,
        "env_delta_size_ids": 0,
        "output_flag_ids": 0,
    }
    return row


def build_rollout_next_input_row(
    actual_batch: dict[str, np.ndarray],
    next_index: int,
    predicted_next: dict[str, int],
    vocab: TracePretrainVocab,
    mode: str,
) -> dict[str, int]:
    if mode == "predicted_all":
        row = neutral_input_row(actual_batch, next_index, vocab, keep_stack_env=False)
        for field, _target in PREDICTED_FIELD_TARGET_PAIRS:
            row[field] = int(predicted_next[field])
        return row
    if mode == "predicted_opcode_step_plus_sizes":
        row = neutral_input_row(actual_batch, next_index, vocab, keep_stack_env=False)
        row["opcode_ids"] = int(predicted_next["opcode_ids"])
        row["step_type_ids"] = int(predicted_next["step_type_ids"])
        row["stack_depth_ids"] = int(predicted_next["stack_depth_ids"])
        row["env_size_ids"] = int(predicted_next["env_size_ids"])
        return row
    if mode == "predicted_opcode_step_only":
        row = neutral_input_row(actual_batch, next_index, vocab, keep_stack_env=False)
        row["opcode_ids"] = int(predicted_next["opcode_ids"])
        row["step_type_ids"] = int(predicted_next["step_type_ids"])
        return row
    if mode == "predicted_all_oracle_sizes":
        row = neutral_input_row(actual_batch, next_index, vocab, keep_stack_env=True)
        for field, _target in PREDICTED_FIELD_TARGET_PAIRS:
            if field in SIZE_INPUT_FIELDS:
                continue
            row[field] = int(predicted_next[field])
        return row
    if mode == "predicted_opcode_step_only_oracle_sizes":
        row = neutral_input_row(actual_batch, next_index, vocab, keep_stack_env=True)
        row["opcode_ids"] = int(predicted_next["opcode_ids"])
        row["step_type_ids"] = int(predicted_next["step_type_ids"])
        return row
    raise ValueError(f"Unsupported rollout mode: {mode}")


def build_rollout_input_batch(
    actual_batch: dict[str, np.ndarray],
    predicted_batch: dict[str, np.ndarray],
    vocab: TracePretrainVocab,
    mode: str,
) -> dict[str, np.ndarray]:
    out = {
        field: np.array(actual_batch[field], copy=True)
        for field in INPUT_FIELDS
    }
    batch_size, seq_len = out["opcode_ids"].shape
    for batch_idx in range(batch_size):
        actual_1d = {
            field: np.asarray(actual_batch[field][batch_idx])
            for field in INPUT_FIELDS
        }
        for next_index in range(1, seq_len):
            predicted_next = {
                field: int(predicted_batch[field][batch_idx, next_index - 1])
                for field, _target in PREDICTED_FIELD_TARGET_PAIRS
            }
            row = build_rollout_next_input_row(actual_1d, next_index, predicted_next, vocab, mode)
            for field, value in row.items():
                out[field][batch_idx, next_index] = int(value)
    return out


def build_mixed_rollout_input_batch(
    actual_batch: dict[str, np.ndarray],
    predicted_batch: dict[str, np.ndarray],
    vocab: TracePretrainVocab,
    mode: str,
    *,
    replace_prob: float,
    rng: np.random.Generator,
    min_teacher_prefix: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if replace_prob <= 0.0:
        replace_mask = np.zeros_like(actual_batch["valid_mask"], dtype=np.bool_)
        return (
            {
                field: np.array(actual_batch[field], copy=True)
                for field in INPUT_FIELDS
            },
            replace_mask,
        )
    rollout_inputs = build_rollout_input_batch(actual_batch, predicted_batch, vocab, mode)
    replace_mask = rng.random(actual_batch["valid_mask"].shape) < float(replace_prob)
    prefix = max(int(min_teacher_prefix), 0)
    if prefix > 0:
        replace_mask[:, :prefix] = False
    prev_valid = np.zeros_like(replace_mask, dtype=np.bool_)
    if prev_valid.shape[1] > 1:
        prev_valid[:, 1:] = np.asarray(actual_batch["valid_mask"][:, :-1] > 0.0, dtype=np.bool_)
    replace_mask &= prev_valid
    out = {}
    for field in INPUT_FIELDS:
        out[field] = np.where(replace_mask, rollout_inputs[field], actual_batch[field]).astype(np.int32, copy=False)
    return out, replace_mask


def predicted_fields_exact_for_step(
    predicted_next: dict[str, int],
    actual_batch: dict[str, np.ndarray],
    step_index: int,
) -> bool:
    next_index = step_index + 1
    for field, _target in PREDICTED_FIELD_TARGET_PAIRS:
        if int(predicted_next[field]) != int(actual_batch[field][next_index]):
            return False
    return True


def summarize_rollout_failures(step_exact_flags: list[bool]) -> RolloutFailureSummary:
    if all(step_exact_flags):
        return RolloutFailureSummary(full_trace_exact=True, first_failure_step=None)
    for idx, is_exact in enumerate(step_exact_flags, start=1):
        if not is_exact:
            return RolloutFailureSummary(full_trace_exact=False, first_failure_step=idx)
    return RolloutFailureSummary(full_trace_exact=False, first_failure_step=None)
