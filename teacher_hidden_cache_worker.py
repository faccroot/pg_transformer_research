#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sentencepiece as spm

from replay_signal_runtime import FileReplayBuffer, ReplayExample
from snapshot_signal_runtime import StudentSnapshotRuntime
from teacher_signal_runtime import FileTeacherHiddenCache, teacher_window_key
from train_gpt_mlx import (
    Hyperparameters,
    append_teacher_hidden_cache_for_batch,
    ensemble_teacher_hidden_for_batch,
    load_external_teacher_models,
)


def _pending_examples(
    replay_buffer: FileReplayBuffer,
    cache: FileTeacherHiddenCache,
    *,
    batch_rows: int,
) -> list[ReplayExample]:
    pending: list[ReplayExample] = []
    seen_keys: set[str] = set()
    for example in replay_buffer.examples:
        tokens = np.asarray(example.tokens, dtype=np.int32)
        key = teacher_window_key(
            tokens,
            layer_index=cache.layer_index,
            hidden_dim=cache.hidden_dim,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if cache.entry_path(key).is_file():
            continue
        pending.append(example)
        if len(pending) >= batch_rows:
            break
    return pending


def _examples_to_xy(examples: list[ReplayExample], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = np.stack(
        [np.asarray(example.tokens[:-1], dtype=np.int32) for example in examples],
        axis=0,
    )
    y_rows = np.stack(
        [np.asarray(example.tokens[1:], dtype=np.int32) for example in examples],
        axis=0,
    )
    if x_rows.shape[1] != seq_len or y_rows.shape[1] != seq_len:
        raise ValueError(
            f"Replay example seq_len mismatch: expected {seq_len}, got x={x_rows.shape}, y={y_rows.shape}"
        )
    return x_rows, y_rows


def main() -> None:
    args = Hyperparameters()
    if not args.replay_queue_path:
        raise SystemExit("Set REPLAY_QUEUE_PATH for teacher hidden cache worker")
    if not args.external_teacher_config_paths or not args.external_teacher_checkpoint_paths:
        raise SystemExit("Set EXTERNAL_TEACHER_CONFIG_PATHS and EXTERNAL_TEACHER_CHECKPOINT_PATHS for teacher hidden cache worker")
    if not args.external_teacher_hidden_cache_dir:
        raise SystemExit("Set EXTERNAL_TEACHER_HIDDEN_CACHE_DIR for teacher hidden cache worker")

    poll_seconds = max(float(os.environ.get("TEACHER_HIDDEN_WORKER_POLL_SECONDS", "5.0")), 0.1)
    batch_rows = max(int(os.environ.get("TEACHER_HIDDEN_WORKER_BATCH_ROWS", "16")), 1)
    once = bool(int(os.environ.get("TEACHER_HIDDEN_WORKER_ONCE", "0")))
    max_idle_polls = max(int(os.environ.get("TEACHER_HIDDEN_WORKER_MAX_IDLE_POLLS", "0")), 0)
    logic_phase_enabled = bool(int(os.environ.get("TEACHER_HIDDEN_WORKER_LOGIC_ENABLED", "1")))
    worker_source = os.environ.get("TEACHER_HIDDEN_WORKER_SOURCE", "teacher_hidden_worker").strip() or "teacher_hidden_worker"
    snapshot_bus_dir = os.environ.get("SNAPSHOT_BUS_DIR", "").strip() or args.student_snapshot_dir
    snapshot_runtime = (
        StudentSnapshotRuntime(Path(snapshot_bus_dir), run_id=args.run_id)
        if snapshot_bus_dir
        else None
    )

    print(
        f"teacher_hidden_worker:start run_id={args.run_id} replay_queue={args.replay_queue_path} "
        f"cache_dir={args.external_teacher_hidden_cache_dir} hidden_layer={args.external_teacher_hidden_layer} "
        f"batch_rows={batch_rows} poll_seconds={poll_seconds:.2f}",
        flush=True,
    )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    teacher_models, teacher_meta = load_external_teacher_models(args, sp, log_fn=print)
    print(
        f"teacher_hidden_worker:teachers count={len(teacher_models)} "
        f"matched_param_frac={[round(float(meta['matched_param_fraction']), 3) for meta in teacher_meta]}",
        flush=True,
    )

    cache = FileTeacherHiddenCache(
        Path(args.external_teacher_hidden_cache_dir),
        layer_index=args.external_teacher_hidden_layer,
        hidden_dim=args.model_dim,
        max_entries=args.external_teacher_hidden_cache_max_entries,
        log_fn=print,
    )
    replay_buffer = FileReplayBuffer(
        args.replay_queue_path,
        seq_len=args.train_seq_len,
        refresh_every_steps=1,
        max_cached_examples=max(args.replay_max_cached_examples, batch_rows * 8),
        seed=args.seed,
        log_fn=print,
    )

    def write_status(**payload: object) -> None:
        if snapshot_runtime is None:
            return
        snapshot_runtime.write_helper_status(
            worker_source,
            {
                "state": str(payload.get("state", "running")),
                "step": int(payload.get("step", 0)),
                "cache_entries": int(payload.get("cache_entries", cache.summary()["teacher_hidden_cache_entries"])),
                "replay_cached_examples": int(payload.get("replay_cached_examples", replay_buffer.available_count())),
                "last_written": int(payload.get("last_written", 0)),
                "elapsed_ms": float(payload.get("elapsed_ms", 0.0)),
                "note": str(payload.get("note", "")),
            },
        )

    def append_proposal(**payload: object) -> None:
        if snapshot_runtime is None:
            return
        snapshot_runtime.append_helper_proposal(
            worker_source,
            {
                "proposal_id": str(payload.get("proposal_id", f"{worker_source}-{int(time.time() * 1000)}")),
                "kind": str(payload.get("kind", "teacher_hidden_cache_batch")),
                "step": int(payload.get("step", 0)),
                "cache_entries": int(payload.get("cache_entries", 0)),
                "last_written": int(payload.get("last_written", 0)),
                "suggested_distill_weight_mult": float(payload.get("suggested_distill_weight_mult", 1.0)),
                "score_min": float(payload.get("score_min", 0.0)),
                "score_max": float(payload.get("score_max", 0.0)),
                "note": str(payload.get("note", "")),
            },
        )

    write_status(
        state="starting",
        note=(
            f"teachers={len(teacher_models)} hidden_layer={args.external_teacher_hidden_layer} "
            f"batch_rows={batch_rows}"
        ),
    )

    idle_polls = 0
    refresh_step = 0
    while True:
        replay_buffer.refresh(refresh_step)
        refresh_step += 1
        batch = _pending_examples(replay_buffer, cache, batch_rows=batch_rows)
        if not batch:
            idle_polls += 1
            write_status(
                state="idle",
                step=refresh_step,
                replay_cached_examples=replay_buffer.available_count(),
                note=f"idle_polls={idle_polls}",
            )
            if once or (max_idle_polls > 0 and idle_polls >= max_idle_polls):
                print(
                    f"teacher_hidden_worker:stop reason=no_pending idle_polls={idle_polls} "
                    f"cached_examples={replay_buffer.available_count()} cache_entries={cache.summary()['teacher_hidden_cache_entries']}",
                    flush=True,
                )
                write_status(
                    state="stopped",
                    step=refresh_step,
                    replay_cached_examples=replay_buffer.available_count(),
                    note=f"no_pending idle_polls={idle_polls}",
                )
                return
            time.sleep(poll_seconds)
            continue

        idle_polls = 0
        x_np, y_np = _examples_to_xy(batch, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        hidden_t0 = time.perf_counter()
        teacher_hidden = ensemble_teacher_hidden_for_batch(
            teacher_models,
            x,
            x_np,
            logic_phase_enabled=logic_phase_enabled,
            hidden_layer=args.external_teacher_hidden_layer,
        )
        if teacher_hidden is None:
            raise RuntimeError("Teacher hidden worker failed to produce hidden states")
        written = append_teacher_hidden_cache_for_batch(
            cache,
            x_np,
            y_np,
            teacher_hidden,
            step=max(int(example.step) for example in batch),
            source_run_id=f"{worker_source}:{args.run_id}",
            cached_rows=None,
        )
        mx.synchronize()
        max_score = max(float(example.score) for example in batch)
        min_score = min(float(example.score) for example in batch)
        cache_entries = int(cache.summary()["teacher_hidden_cache_entries"])
        suggested_distill_weight_mult = 1.0
        if cache_entries >= max(batch_rows * 2, 32):
            suggested_distill_weight_mult = 1.10
        if cache_entries >= max(batch_rows * 8, 128):
            suggested_distill_weight_mult = 1.20
        print(
            f"teacher_hidden_worker:batch batch_rows={len(batch)} written={written} "
            f"score_range=[{min_score:.4f},{max_score:.4f}] "
            f"elapsed_ms={1000.0 * (time.perf_counter() - hidden_t0):.1f}",
            flush=True,
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - hidden_t0)
        write_status(
            state="active",
            step=max(int(example.step) for example in batch),
            cache_entries=cache_entries,
            replay_cached_examples=replay_buffer.available_count(),
            last_written=written,
            elapsed_ms=elapsed_ms,
            note=f"score_range=[{min_score:.4f},{max_score:.4f}]",
        )
        append_proposal(
            proposal_id=f"{worker_source}-step{max(int(example.step) for example in batch):07d}",
            kind="teacher_hidden_cache_batch",
            step=max(int(example.step) for example in batch),
            cache_entries=cache_entries,
            last_written=written,
            suggested_distill_weight_mult=suggested_distill_weight_mult,
            score_min=min_score,
            score_max=max_score,
            note=f"batch_rows={len(batch)} elapsed_ms={elapsed_ms:.1f}",
        )
        if once:
            write_status(
                state="stopped",
                step=max(int(example.step) for example in batch),
                cache_entries=cache_entries,
                replay_cached_examples=replay_buffer.available_count(),
                last_written=written,
                elapsed_ms=elapsed_ms,
                note="once",
            )
            return


if __name__ == "__main__":
    main()
