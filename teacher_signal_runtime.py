from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def window_tokens_from_xy(x_row: np.ndarray, y_row: np.ndarray) -> np.ndarray:
    if x_row.ndim != 1 or y_row.ndim != 1:
        raise ValueError("Expected 1D token rows")
    if x_row.shape != y_row.shape:
        raise ValueError(f"Mismatched x/y shapes: {x_row.shape} vs {y_row.shape}")
    return np.concatenate([x_row.astype(np.int32, copy=False), y_row[-1:].astype(np.int32, copy=False)], axis=0)


def teacher_window_key(tokens: np.ndarray, *, layer_index: int, hidden_dim: int) -> str:
    tokens_i32 = np.asarray(tokens, dtype=np.int32)
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(tokens_i32.tobytes())
    hasher.update(f"layer={int(layer_index)};hidden_dim={int(hidden_dim)}".encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class TeacherHiddenExample:
    key: str
    tokens: np.ndarray
    hidden: np.ndarray
    layer_index: int
    step: int
    source_run_id: str


class FileTeacherHiddenCache:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        layer_index: int,
        hidden_dim: int,
        max_entries: int = 1024,
        log_fn=None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.layer_index = int(layer_index)
        self.hidden_dim = int(hidden_dim)
        self.max_entries = max(int(max_entries), 1)
        self.log_fn = log_fn
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    @property
    def entries_dir(self) -> Path:
        return self.root_dir / "hidden_entries"

    @property
    def index_path(self) -> Path:
        return self.root_dir / "hidden_index.jsonl"

    def entry_path(self, key: str) -> Path:
        return self.entries_dir / f"{key}.npz"

    def summary(self) -> dict[str, object]:
        return {
            "teacher_hidden_cache_dir": str(self.root_dir),
            "teacher_hidden_cache_entries": int(len(list(self.entries_dir.glob("*.npz")))),
            "teacher_hidden_cache_layer": int(self.layer_index),
            "teacher_hidden_cache_hidden_dim": int(self.hidden_dim),
            "teacher_hidden_cache_max_entries": int(self.max_entries),
        }

    def lookup_rows(self, x_np: np.ndarray, y_np: np.ndarray) -> tuple[list[np.ndarray | None], dict[str, float]]:
        hits: list[np.ndarray | None] = []
        hit_count = 0
        for x_row, y_row in zip(x_np, y_np, strict=True):
            tokens = window_tokens_from_xy(x_row, y_row)
            key = teacher_window_key(tokens, layer_index=self.layer_index, hidden_dim=self.hidden_dim)
            path = self.entry_path(key)
            if not path.is_file():
                hits.append(None)
                continue
            try:
                with np.load(path, allow_pickle=False) as payload:
                    cached_tokens = np.asarray(payload["tokens"], dtype=np.int32)
                    hidden = np.asarray(payload["hidden"], dtype=np.float32)
            except Exception:
                hits.append(None)
                continue
            if cached_tokens.shape != tokens.shape or not np.array_equal(cached_tokens, tokens):
                hits.append(None)
                continue
            if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
                hits.append(None)
                continue
            hits.append(hidden)
            hit_count += 1
        total = max(len(hits), 1)
        return hits, {
            "teacher_hidden_cache_hits": float(hit_count),
            "teacher_hidden_cache_rows": float(len(hits)),
            "teacher_hidden_cache_hit_frac": float(hit_count / total),
            "teacher_hidden_cache_full_hit": float(1.0 if hit_count == len(hits) and hits else 0.0),
        }

    def append_examples(self, examples: list[TeacherHiddenExample]) -> int:
        written = 0
        if not examples:
            return written
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as index_handle:
            for example in examples:
                path = self.entry_path(example.key)
                np.savez_compressed(
                    path,
                    tokens=np.asarray(example.tokens, dtype=np.int32),
                    hidden=np.asarray(example.hidden, dtype=np.float16),
                    layer_index=np.asarray(example.layer_index, dtype=np.int32),
                    step=np.asarray(example.step, dtype=np.int32),
                )
                index_handle.write(
                    json.dumps(
                        {
                            "key": example.key,
                            "path": str(path),
                            "layer_index": int(example.layer_index),
                            "step": int(example.step),
                            "source_run_id": str(example.source_run_id),
                        },
                        sort_keys=True,
                    )
                )
                index_handle.write("\n")
                written += 1
        self.prune_old_entries()
        if self.log_fn is not None:
            self.log_fn(
                f"teacher_hidden_cache:append dir:{self.root_dir} written:{written} "
                f"layer:{self.layer_index} hidden_dim:{self.hidden_dim}"
            )
        return written

    def prune_old_entries(self) -> None:
        files = sorted(self.entries_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime_ns)
        stale = files[:-self.max_entries]
        for path in stale:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

