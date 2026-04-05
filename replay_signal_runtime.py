from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappush, heapreplace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReplayExample:
    tokens: tuple[int, ...]
    score: float
    seq_len: int
    source_run_id: str
    step: int
    kind: str = "student_row_nll"
    metadata: dict[str, float | int | str] | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tokens": list(int(token) for token in self.tokens),
            "score": float(self.score),
            "seq_len": int(self.seq_len),
            "source_run_id": self.source_run_id,
            "step": int(self.step),
            "kind": self.kind,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> ReplayExample | None:
        tokens_raw = payload.get("tokens")
        if not isinstance(tokens_raw, list) or not tokens_raw:
            return None
        seq_len = int(payload.get("seq_len", 0))
        tokens = tuple(int(token) for token in tokens_raw)
        if seq_len <= 0 or len(tokens) != seq_len + 1:
            return None
        score = float(payload.get("score", 0.0))
        step = int(payload.get("step", 0))
        source_run_id = str(payload.get("source_run_id", "unknown"))
        kind = str(payload.get("kind", "student_row_nll"))
        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else None
        return cls(
            tokens=tokens,
            score=score,
            seq_len=seq_len,
            source_run_id=source_run_id,
            step=step,
            kind=kind,
            metadata=metadata_dict,
        )


class FileReplayBuffer:
    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        refresh_every_steps: int = 50,
        max_cached_examples: int = 2048,
        seed: int = 1337,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.path = Path(path).expanduser()
        self.seq_len = int(seq_len)
        self.refresh_every_steps = max(int(refresh_every_steps), 1)
        self.max_cached_examples = max(int(max_cached_examples), 1)
        self.rng = np.random.default_rng(seed)
        self.log_fn = log_fn

        self.examples: list[ReplayExample] = []
        self.last_refresh_step = -1
        self.last_refresh_mtime_ns = -1
        self.total_emitted = 0
        self.total_sampled = 0

    def available_count(self) -> int:
        return len(self.examples)

    def maybe_refresh(self, step: int) -> None:
        if step <= 0:
            return
        if self.last_refresh_step >= 0 and (step - self.last_refresh_step) < self.refresh_every_steps:
            return
        self.refresh(step)

    def refresh(self, step: int = 0) -> None:
        self.last_refresh_step = int(step)
        if not self.path.is_file():
            self.examples = []
            self.last_refresh_mtime_ns = -1
            return
        stat = self.path.stat()
        if stat.st_mtime_ns == self.last_refresh_mtime_ns:
            return
        self.last_refresh_mtime_ns = stat.st_mtime_ns
        heap: list[tuple[float, int, ReplayExample]] = []
        line_index = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                example = ReplayExample.from_json(payload)
                if example is None or example.seq_len != self.seq_len:
                    continue
                score = float(example.score)
                item = (score, line_index, example)
                if len(heap) < self.max_cached_examples:
                    heappush(heap, item)
                elif score > heap[0][0]:
                    heapreplace(heap, item)
                line_index += 1
        heap.sort(key=lambda item: (item[0], item[1]), reverse=True)
        self.examples = [item[2] for item in heap]
        if self.log_fn is not None:
            self.log_fn(
                f"replay_buffer:refresh path:{self.path} cached_examples:{len(self.examples)} "
                f"mtime_ns:{self.last_refresh_mtime_ns}"
            )

    def append_examples(self, examples: list[ReplayExample]) -> None:
        if not examples:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_json(), sort_keys=True))
                handle.write("\n")
        self.total_emitted += len(examples)

    def sample_windows(self, count: int) -> list[np.ndarray]:
        if count <= 0 or not self.examples:
            return []
        scores = np.asarray(
            [max(float(example.score), 1.0e-6) for example in self.examples],
            dtype=np.float64,
        )
        denom = float(scores.sum())
        probs = None if not math.isfinite(denom) or denom <= 0.0 else scores / denom
        replace = len(self.examples) < count
        indices = self.rng.choice(len(self.examples), size=count, replace=replace, p=probs)
        self.total_sampled += int(np.size(indices))
        windows: list[np.ndarray] = []
        for idx in np.asarray(indices, dtype=np.int32).reshape(-1):
            windows.append(np.asarray(self.examples[int(idx)].tokens, dtype=np.int32))
        return windows

    def summary(self) -> dict[str, object]:
        return {
            "replay_queue_path": str(self.path),
            "replay_cached_examples": int(len(self.examples)),
            "replay_total_emitted": int(self.total_emitted),
            "replay_total_sampled": int(self.total_sampled),
        }


class ReplayMixTokenLoader:
    def __init__(
        self,
        base_loader,
        replay_buffer: FileReplayBuffer,
        *,
        mix_fraction: float,
        seed: int = 1337,
    ):
        self.base_loader = base_loader
        self.replay_buffer = replay_buffer
        self.mix_fraction = float(np.clip(float(mix_fraction), 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self._last_batch_metrics: dict[str, float | int] = {}

    def set_mix_fraction(self, mix_fraction: float) -> None:
        self.mix_fraction = float(np.clip(float(mix_fraction), 0.0, 1.0))

    def maybe_refresh(self, step: int) -> None:
        self.replay_buffer.maybe_refresh(step)

    def summary(self) -> dict[str, object]:
        base_summary = self.base_loader.summary() if hasattr(self.base_loader, "summary") else {}
        return {
            **base_summary,
            "replay_mix_fraction": float(self.mix_fraction),
            **self.replay_buffer.summary(),
        }

    def current_phase(self):
        if hasattr(self.base_loader, "current_phase"):
            return self.base_loader.current_phase()
        raise AttributeError("current_phase")

    def begin_step(self):
        if hasattr(self.base_loader, "begin_step"):
            return self.base_loader.begin_step()
        raise AttributeError("begin_step")

    def end_step(self) -> None:
        if hasattr(self.base_loader, "end_step"):
            self.base_loader.end_step()

    def last_step_metrics(self) -> dict[str, float | int]:
        if hasattr(self.base_loader, "last_step_metrics"):
            return dict(self.base_loader.last_step_metrics())
        return {}

    def last_batch_metrics(self) -> dict[str, float | int]:
        return dict(self._last_batch_metrics)

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        expected_replay = batch_seqs * self.mix_fraction
        requested_replay = int(math.floor(expected_replay))
        frac = expected_replay - requested_replay
        if frac > 0.0 and self.rng.random() < frac:
            requested_replay += 1
        requested_replay = min(max(requested_replay, 0), batch_seqs)
        replay_windows = self.replay_buffer.sample_windows(requested_replay)
        replay_rows = len(replay_windows)
        live_rows = batch_seqs - replay_rows

        live_x = np.zeros((0, seq_len), dtype=np.int32)
        live_y = np.zeros((0, seq_len), dtype=np.int32)
        if live_rows > 0:
            live_x, live_y = self.base_loader.next_batch_np(live_rows * seq_len, seq_len)

        if replay_rows > 0:
            replay_x = np.stack([window[:-1] for window in replay_windows], axis=0).astype(np.int32, copy=False)
            replay_y = np.stack([window[1:] for window in replay_windows], axis=0).astype(np.int32, copy=False)
            if live_rows > 0:
                x = np.concatenate([live_x, replay_x], axis=0)
                y = np.concatenate([live_y, replay_y], axis=0)
                order = self.rng.permutation(x.shape[0])
                x = x[order]
                y = y[order]
            else:
                x, y = replay_x, replay_y
        else:
            x, y = live_x, live_y

        metrics = self.base_loader.last_batch_metrics() if hasattr(self.base_loader, "last_batch_metrics") else {}
        metrics = dict(metrics)
        metrics.update(
            {
                "replay_requested_rows": int(requested_replay),
                "replay_rows": int(replay_rows),
                "live_rows": int(live_rows),
                "replay_frac": float(replay_rows / max(batch_seqs, 1)),
                "replay_cache_entries": int(self.replay_buffer.available_count()),
            }
        )
        self._last_batch_metrics = metrics
        return x, y
