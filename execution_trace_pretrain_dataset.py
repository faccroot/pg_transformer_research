from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NONE_TOKEN = "<none>"
PAD_TOKEN = "<pad>"


@dataclass(frozen=True)
class TracePretrainVocab:
    opcode_to_id: dict[str, int]
    step_type_to_id: dict[str, int]
    variable_to_id: dict[str, int]
    branch_to_id: dict[str, int]
    max_stack_bucket: int = 16
    max_env_bucket: int = 16
    max_delta_bucket: int = 8
    max_memop_bucket: int = 8

    @property
    def pad_variable_id(self) -> int:
        return self.variable_to_id[PAD_TOKEN]

    @property
    def none_variable_id(self) -> int:
        return self.variable_to_id[NONE_TOKEN]

    @property
    def none_branch_id(self) -> int:
        return self.branch_to_id["none"]


@dataclass(frozen=True)
class EncodedTraceSequence:
    example_id: str
    opcode_ids: np.ndarray
    step_type_ids: np.ndarray
    read_var_ids: np.ndarray
    write_var_ids: np.ndarray
    read_count_ids: np.ndarray
    write_count_ids: np.ndarray
    branch_ids: np.ndarray
    stack_depth_ids: np.ndarray
    env_size_ids: np.ndarray
    env_delta_size_ids: np.ndarray
    output_flag_ids: np.ndarray
    target_next_opcode_ids: np.ndarray
    target_next_step_type_ids: np.ndarray
    target_next_read_var_ids: np.ndarray
    target_next_write_var_ids: np.ndarray
    target_next_read_count_ids: np.ndarray
    target_next_write_count_ids: np.ndarray
    target_next_branch_ids: np.ndarray
    target_next_stack_depth_ids: np.ndarray
    target_next_env_size_ids: np.ndarray
    target_next_env_delta_size_ids: np.ndarray
    target_next_output_flag_ids: np.ndarray
    valid_mask: np.ndarray

    @property
    def length(self) -> int:
        return int(self.opcode_ids.shape[0])


def load_examples_jsonl(path: str | Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload in {path}, got {type(payload)!r}")
            examples.append(payload)
    return examples


def _sorted_union(items: set[str]) -> list[str]:
    return sorted(item for item in items if item)


def build_trace_pretrain_vocab(
    examples: list[dict[str, object]],
    *,
    max_stack_bucket: int = 16,
    max_env_bucket: int = 16,
    max_delta_bucket: int = 8,
    max_memop_bucket: int = 8,
) -> TracePretrainVocab:
    opcodes: set[str] = {PAD_TOKEN}
    step_types: set[str] = {PAD_TOKEN}
    variables: set[str] = {PAD_TOKEN, NONE_TOKEN}

    for example in examples:
        trace = example.get("trace")
        if not isinstance(trace, list):
            continue
        for event in trace:
            if not isinstance(event, dict):
                continue
            opcode = event.get("opcode")
            step_type = event.get("step_type")
            if isinstance(opcode, str):
                opcodes.add(opcode)
            if isinstance(step_type, str):
                step_types.add(step_type)
            reads = event.get("memory_reads")
            if isinstance(reads, list):
                for name in reads:
                    if isinstance(name, str):
                        variables.add(name)
            writes = event.get("memory_writes")
            if isinstance(writes, dict):
                for name in writes:
                    if isinstance(name, str):
                        variables.add(name)
            env_after = event.get("env_after")
            if isinstance(env_after, dict):
                for name in env_after:
                    if isinstance(name, str):
                        variables.add(name)

    branch_to_id = {"none": 0, "false": 1, "true": 2}
    return TracePretrainVocab(
        opcode_to_id={name: idx for idx, name in enumerate(_sorted_union(opcodes))},
        step_type_to_id={name: idx for idx, name in enumerate(_sorted_union(step_types))},
        variable_to_id={name: idx for idx, name in enumerate(_sorted_union(variables))},
        branch_to_id=branch_to_id,
        max_stack_bucket=int(max_stack_bucket),
        max_env_bucket=int(max_env_bucket),
        max_delta_bucket=int(max_delta_bucket),
        max_memop_bucket=int(max_memop_bucket),
    )


def _branch_label(value: object) -> str:
    if value is None:
        return "none"
    if bool(value):
        return "true"
    return "false"


def _first_read_var(event: dict[str, object]) -> str:
    reads = event.get("memory_reads")
    if isinstance(reads, list):
        for name in reads:
            if isinstance(name, str):
                return name
    return NONE_TOKEN


def _first_write_var(event: dict[str, object]) -> str:
    writes = event.get("memory_writes")
    if isinstance(writes, dict):
        for name in sorted(writes):
            return str(name)
    return NONE_TOKEN


def _bucket_size(value: int, cap: int) -> int:
    return min(max(int(value), 0), max(int(cap) - 1, 0))


def _read_count(event: dict[str, object]) -> int:
    reads = event.get("memory_reads")
    return len(reads) if isinstance(reads, list) else 0


def _write_count(event: dict[str, object]) -> int:
    writes = event.get("memory_writes")
    return len(writes) if isinstance(writes, dict) else 0


def encode_trace_example(
    example: dict[str, object],
    vocab: TracePretrainVocab,
) -> EncodedTraceSequence:
    trace = example.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("Trace example is missing a non-empty 'trace' list")

    pad_opcode_id = vocab.opcode_to_id[PAD_TOKEN]
    pad_step_type_id = vocab.step_type_to_id[PAD_TOKEN]
    pad_var_id = vocab.pad_variable_id
    none_var_id = vocab.none_variable_id
    none_branch_id = vocab.none_branch_id

    current_opcode_ids: list[int] = []
    current_step_type_ids: list[int] = []
    current_read_var_ids: list[int] = []
    current_write_var_ids: list[int] = []
    current_read_count_ids: list[int] = []
    current_write_count_ids: list[int] = []
    current_branch_ids: list[int] = []
    current_stack_depth_ids: list[int] = []
    current_env_size_ids: list[int] = []
    current_env_delta_size_ids: list[int] = []
    current_output_flag_ids: list[int] = []

    next_opcode_ids: list[int] = []
    next_step_type_ids: list[int] = []
    next_read_var_ids: list[int] = []
    next_write_var_ids: list[int] = []
    next_read_count_ids: list[int] = []
    next_write_count_ids: list[int] = []
    next_branch_ids: list[int] = []
    next_stack_depth_ids: list[int] = []
    next_env_size_ids: list[int] = []
    next_env_delta_size_ids: list[int] = []
    next_output_flag_ids: list[int] = []
    valid_mask: list[float] = []

    for idx, event in enumerate(trace):
        if not isinstance(event, dict):
            raise TypeError(f"Trace event must be a dict, got {type(event)!r}")
        opcode = str(event.get("opcode", PAD_TOKEN))
        step_type = str(event.get("step_type", PAD_TOKEN))
        read_var = _first_read_var(event)
        write_var = _first_write_var(event)
        branch = _branch_label(event.get("branch_taken"))
        stack_after = event.get("stack_after")
        env_after = event.get("env_after")
        env_delta = event.get("env_delta")
        output_delta = event.get("output_delta")

        current_opcode_ids.append(vocab.opcode_to_id.get(opcode, pad_opcode_id))
        current_step_type_ids.append(vocab.step_type_to_id.get(step_type, pad_step_type_id))
        current_read_var_ids.append(vocab.variable_to_id.get(read_var, none_var_id))
        current_write_var_ids.append(vocab.variable_to_id.get(write_var, none_var_id))
        current_read_count_ids.append(_bucket_size(_read_count(event), vocab.max_memop_bucket))
        current_write_count_ids.append(_bucket_size(_write_count(event), vocab.max_memop_bucket))
        current_branch_ids.append(vocab.branch_to_id.get(branch, none_branch_id))
        current_stack_depth_ids.append(
            _bucket_size(len(stack_after) if isinstance(stack_after, list) else 0, vocab.max_stack_bucket)
        )
        current_env_size_ids.append(
            _bucket_size(len(env_after) if isinstance(env_after, dict) else 0, vocab.max_env_bucket)
        )
        current_env_delta_size_ids.append(
            _bucket_size(len(env_delta) if isinstance(env_delta, dict) else 0, vocab.max_delta_bucket)
        )
        current_output_flag_ids.append(1 if isinstance(output_delta, list) and output_delta else 0)

        has_next = idx + 1 < len(trace)
        valid_mask.append(1.0 if has_next else 0.0)
        if not has_next:
            next_opcode_ids.append(pad_opcode_id)
            next_step_type_ids.append(pad_step_type_id)
            next_read_var_ids.append(pad_var_id)
            next_write_var_ids.append(pad_var_id)
            next_read_count_ids.append(0)
            next_write_count_ids.append(0)
            next_branch_ids.append(none_branch_id)
            next_stack_depth_ids.append(0)
            next_env_size_ids.append(0)
            next_env_delta_size_ids.append(0)
            next_output_flag_ids.append(0)
            continue

        next_event = trace[idx + 1]
        if not isinstance(next_event, dict):
            raise TypeError(f"Next trace event must be a dict, got {type(next_event)!r}")
        next_opcode = str(next_event.get("opcode", PAD_TOKEN))
        next_step_type = str(next_event.get("step_type", PAD_TOKEN))
        next_read_var = _first_read_var(next_event)
        next_write_var = _first_write_var(next_event)
        next_branch = _branch_label(next_event.get("branch_taken"))
        next_stack_after = next_event.get("stack_after")
        next_env_after = next_event.get("env_after")
        next_env_delta = next_event.get("env_delta")
        next_output_delta = next_event.get("output_delta")

        next_opcode_ids.append(vocab.opcode_to_id.get(next_opcode, pad_opcode_id))
        next_step_type_ids.append(vocab.step_type_to_id.get(next_step_type, pad_step_type_id))
        next_read_var_ids.append(vocab.variable_to_id.get(next_read_var, none_var_id))
        next_write_var_ids.append(vocab.variable_to_id.get(next_write_var, none_var_id))
        next_read_count_ids.append(_bucket_size(_read_count(next_event), vocab.max_memop_bucket))
        next_write_count_ids.append(_bucket_size(_write_count(next_event), vocab.max_memop_bucket))
        next_branch_ids.append(vocab.branch_to_id.get(next_branch, none_branch_id))
        next_stack_depth_ids.append(
            _bucket_size(len(next_stack_after) if isinstance(next_stack_after, list) else 0, vocab.max_stack_bucket)
        )
        next_env_size_ids.append(
            _bucket_size(len(next_env_after) if isinstance(next_env_after, dict) else 0, vocab.max_env_bucket)
        )
        next_env_delta_size_ids.append(
            _bucket_size(len(next_env_delta) if isinstance(next_env_delta, dict) else 0, vocab.max_delta_bucket)
        )
        next_output_flag_ids.append(1 if isinstance(next_output_delta, list) and next_output_delta else 0)

    example_id = str(example.get("example_id", "unknown"))
    return EncodedTraceSequence(
        example_id=example_id,
        opcode_ids=np.asarray(current_opcode_ids, dtype=np.int32),
        step_type_ids=np.asarray(current_step_type_ids, dtype=np.int32),
        read_var_ids=np.asarray(current_read_var_ids, dtype=np.int32),
        write_var_ids=np.asarray(current_write_var_ids, dtype=np.int32),
        read_count_ids=np.asarray(current_read_count_ids, dtype=np.int32),
        write_count_ids=np.asarray(current_write_count_ids, dtype=np.int32),
        branch_ids=np.asarray(current_branch_ids, dtype=np.int32),
        stack_depth_ids=np.asarray(current_stack_depth_ids, dtype=np.int32),
        env_size_ids=np.asarray(current_env_size_ids, dtype=np.int32),
        env_delta_size_ids=np.asarray(current_env_delta_size_ids, dtype=np.int32),
        output_flag_ids=np.asarray(current_output_flag_ids, dtype=np.int32),
        target_next_opcode_ids=np.asarray(next_opcode_ids, dtype=np.int32),
        target_next_step_type_ids=np.asarray(next_step_type_ids, dtype=np.int32),
        target_next_read_var_ids=np.asarray(next_read_var_ids, dtype=np.int32),
        target_next_write_var_ids=np.asarray(next_write_var_ids, dtype=np.int32),
        target_next_read_count_ids=np.asarray(next_read_count_ids, dtype=np.int32),
        target_next_write_count_ids=np.asarray(next_write_count_ids, dtype=np.int32),
        target_next_branch_ids=np.asarray(next_branch_ids, dtype=np.int32),
        target_next_stack_depth_ids=np.asarray(next_stack_depth_ids, dtype=np.int32),
        target_next_env_size_ids=np.asarray(next_env_size_ids, dtype=np.int32),
        target_next_env_delta_size_ids=np.asarray(next_env_delta_size_ids, dtype=np.int32),
        target_next_output_flag_ids=np.asarray(next_output_flag_ids, dtype=np.int32),
        valid_mask=np.asarray(valid_mask, dtype=np.float32),
    )


def encode_trace_examples(
    examples: list[dict[str, object]],
    vocab: TracePretrainVocab,
) -> list[EncodedTraceSequence]:
    return [encode_trace_example(example, vocab) for example in examples]


def pad_encoded_batch(
    sequences: list[EncodedTraceSequence],
    vocab: TracePretrainVocab,
) -> dict[str, np.ndarray]:
    if not sequences:
        raise ValueError("pad_encoded_batch requires at least one sequence")
    batch_size = len(sequences)
    max_len = max(seq.length for seq in sequences)

    def alloc(fill: int | float, *, dtype: np.dtype) -> np.ndarray:
        return np.full((batch_size, max_len), fill, dtype=dtype)

    batch = {
        "opcode_ids": alloc(vocab.opcode_to_id[PAD_TOKEN], dtype=np.int32),
        "step_type_ids": alloc(vocab.step_type_to_id[PAD_TOKEN], dtype=np.int32),
        "read_var_ids": alloc(vocab.pad_variable_id, dtype=np.int32),
        "write_var_ids": alloc(vocab.pad_variable_id, dtype=np.int32),
        "read_count_ids": alloc(0, dtype=np.int32),
        "write_count_ids": alloc(0, dtype=np.int32),
        "branch_ids": alloc(vocab.none_branch_id, dtype=np.int32),
        "stack_depth_ids": alloc(0, dtype=np.int32),
        "env_size_ids": alloc(0, dtype=np.int32),
        "env_delta_size_ids": alloc(0, dtype=np.int32),
        "output_flag_ids": alloc(0, dtype=np.int32),
        "target_next_opcode_ids": alloc(vocab.opcode_to_id[PAD_TOKEN], dtype=np.int32),
        "target_next_step_type_ids": alloc(vocab.step_type_to_id[PAD_TOKEN], dtype=np.int32),
        "target_next_read_var_ids": alloc(vocab.pad_variable_id, dtype=np.int32),
        "target_next_write_var_ids": alloc(vocab.pad_variable_id, dtype=np.int32),
        "target_next_read_count_ids": alloc(0, dtype=np.int32),
        "target_next_write_count_ids": alloc(0, dtype=np.int32),
        "target_next_branch_ids": alloc(vocab.none_branch_id, dtype=np.int32),
        "target_next_stack_depth_ids": alloc(0, dtype=np.int32),
        "target_next_env_size_ids": alloc(0, dtype=np.int32),
        "target_next_env_delta_size_ids": alloc(0, dtype=np.int32),
        "target_next_output_flag_ids": alloc(0, dtype=np.int32),
        "valid_mask": alloc(0.0, dtype=np.float32),
    }

    for row, seq in enumerate(sequences):
        length = seq.length
        for field in batch:
            batch[field][row, :length] = getattr(seq, field)
    return batch
