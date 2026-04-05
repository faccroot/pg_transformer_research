from __future__ import annotations

import glob
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curriculum import (
    SHOW_NEVER,
    ChunkFeatures,
    CurriculumPhase,
    classify_replay_buckets,
    default_phase_plan,
    order_chunk_indices,
    phase_for_progress,
    validate_phase_plan,
)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


@dataclass(frozen=True)
class CurriculumRuntimeConfig:
    features_path: str
    phase_plan_path: str = ""
    cached_shards: int = 2
    min_compressibility: float = -1.0


def load_phase_plan(path: str | None) -> list[CurriculumPhase]:
    if not path:
        phases = default_phase_plan()
        validate_phase_plan(phases)
        return phases
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Curriculum phase plan must be a JSON list: {path}")
    phases = [CurriculumPhase(**entry) for entry in payload]
    validate_phase_plan(phases)
    return phases


class ShardChunkStore:
    def __init__(self, files: list[Path], seq_len: int, max_cached_shards: int = 2):
        self.files = files
        self.seq_len = seq_len
        self.max_cached_shards = max(max_cached_shards, 1)
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

    def _get_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = load_data_shard(self.files[shard_idx])
            self._cache_order.append(shard_idx)
            while len(self._cache_order) > self.max_cached_shards:
                evict_idx = self._cache_order.pop(0)
                self._cache.pop(evict_idx, None)
        else:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
        return self._cache[shard_idx]

    def fetch_sequence(self, shard_idx: int, chunk_idx: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = self._get_shard(shard_idx)
        start = chunk_idx * self.seq_len
        end = start + self.seq_len + 1
        if end > tokens.size:
            raise ValueError(
                f"Curriculum chunk exceeds shard bounds: shard_idx={shard_idx} chunk_idx={chunk_idx} "
                f"seq_len={self.seq_len} shard_tokens={tokens.size}"
            )
        chunk = tokens[start:end]
        return chunk[:-1], chunk[1:]


class CurriculumTokenLoader:
    def __init__(
        self,
        pattern: str,
        *,
        seq_len: int,
        total_train_tokens: int,
        runtime_config: CurriculumRuntimeConfig,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(path) for path in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.seq_len = seq_len
        self.total_train_tokens = max(int(total_train_tokens), int(seq_len))
        self.runtime_config = runtime_config
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.phase_plan = load_phase_plan(runtime_config.phase_plan_path or None)

        payload = np.load(runtime_config.features_path)
        chunk_size = int(np.asarray(payload["chunk_size"]).reshape(-1)[0]) if "chunk_size" in payload else seq_len
        if chunk_size != seq_len:
            raise ValueError(
                f"Curriculum feature chunk_size={chunk_size} must match TRAIN_SEQ_LEN={seq_len}"
            )
        self.chunk_size = chunk_size

        self.features = ChunkFeatures(
            cluster_ids=np.asarray(payload["cluster_ids"], dtype=np.int32) if "cluster_ids" in payload else None,
            operator_density=np.asarray(payload["operator_density"], dtype=np.float32) if "operator_density" in payload else None,
            difficulty=np.asarray(payload["difficulty"], dtype=np.float32) if "difficulty" in payload else None,
            duplicate_score=np.asarray(payload["duplicate_score"], dtype=np.float32) if "duplicate_score" in payload else None,
            compressibility_ratio=np.asarray(payload["compressibility_ratio"], dtype=np.float32) if "compressibility_ratio" in payload else None,
            learnability_score=np.asarray(payload["learnability_score"], dtype=np.float32) if "learnability_score" in payload else None,
            quality_score=np.asarray(payload["quality_score"], dtype=np.float32) if "quality_score" in payload else None,
            confidence=np.asarray(payload["confidence"], dtype=np.float32) if "confidence" in payload else None,
        )
        self.shard_index = np.asarray(payload["shard_index"], dtype=np.int32)
        self.chunk_index = np.asarray(payload["chunk_index"], dtype=np.int32)
        if self.shard_index.shape[0] != len(self.features) or self.chunk_index.shape[0] != len(self.features):
            raise ValueError("Curriculum feature metadata length mismatch")
        if self.shard_index.size and int(self.shard_index.max()) >= len(self.files):
            raise ValueError(
                f"Curriculum metadata references shard {int(self.shard_index.max())}, but only {len(self.files)} files matched"
            )

        self.replay_bucket = (
            np.asarray(payload["replay_bucket"], dtype=np.int8)
            if "replay_bucket" in payload
            else classify_replay_buckets(self.features)
        )
        self.compressibility = (
            1.0 - np.clip(np.asarray(self.features.compressibility_ratio, dtype=np.float32), 0.0, 1.0)
            if self.features.compressibility_ratio is not None
            else None
        )
        self.filtered_low_compressibility_mask = np.zeros((len(self.features),), dtype=bool)
        if runtime_config.min_compressibility >= 0.0:
            if self.compressibility is None:
                raise ValueError(
                    "CURRICULUM_MIN_COMPRESSIBILITY requires compressibility_ratio in the features file"
                )
            self.filtered_low_compressibility_mask = self.compressibility < float(runtime_config.min_compressibility)
            self.replay_bucket = np.where(self.filtered_low_compressibility_mask, SHOW_NEVER, self.replay_bucket).astype(np.int8)
        self.max_uses = np.where(self.replay_bucket == SHOW_NEVER, 0, np.where(self.replay_bucket == 2, 2, 1)).astype(np.int16)
        self.use_counts = np.zeros((len(self.features),), dtype=np.int16)
        self.phase_orders = {
            phase.name: order_chunk_indices(self.features, phase).astype(np.int32)
            for phase in self.phase_plan
        }
        self.phase_cursors = {phase.name: 0 for phase in self.phase_plan}
        self.store = ShardChunkStore(self.files, seq_len=seq_len, max_cached_shards=runtime_config.cached_shards)
        self.tokens_served = 0
        self.epoch = 1
        self.active_phase_name: str | None = None
        self._pinned_phase: CurriculumPhase | None = None
        self._step_indices: list[int] = []
        self._last_step_metrics: dict[str, float | int] = {}
        self._last_batch_metrics: dict[str, float | int] = {}

    def summary(self) -> dict[str, object]:
        return {
            "features_path": self.runtime_config.features_path,
            "phase_plan_path": self.runtime_config.phase_plan_path,
            "chunk_size": self.chunk_size,
            "num_chunks": len(self.features),
            "num_shards": len(self.files),
            "repeatable_chunks": int(np.count_nonzero(self.max_uses > 1)),
            "skipped_chunks": int(np.count_nonzero(self.max_uses <= 0)),
            "min_compressibility": float(self.runtime_config.min_compressibility),
            "filtered_low_compressibility_chunks": int(np.count_nonzero(self.filtered_low_compressibility_mask)),
        }

    def progress_frac(self) -> float:
        return min(max(self.tokens_served / float(self.total_train_tokens), 0.0), 1.0)

    def current_phase(self) -> CurriculumPhase:
        phase = phase_for_progress(self.progress_frac(), self.phase_plan)
        if phase.name != self.active_phase_name:
            self.active_phase_name = phase.name
            if self.log_fn is not None:
                self.log_fn(
                    f"curriculum:phase name:{phase.name} "
                    f"focus:{phase.focus} progress:{self.progress_frac():.3f} "
                    f"logic:{int(phase.enable_logic_sidecar)} jepa:{int(phase.enable_jepa)} "
                    f"qat:{int(phase.enable_qat)} ema:{int(phase.enable_ema)} "
                    f"focal:{phase.focal_loss_weight:.3f}"
                )
        return phase

    def begin_step(self) -> CurriculumPhase:
        self._pinned_phase = self.current_phase()
        self._step_indices = []
        return self._pinned_phase

    def end_step(self) -> None:
        self._last_step_metrics = self._summarize_indices(self._step_indices)
        self._pinned_phase = None
        self._step_indices = []

    def last_step_metrics(self) -> dict[str, float | int]:
        return dict(self._last_step_metrics)

    def last_batch_metrics(self) -> dict[str, float | int]:
        return dict(self._last_batch_metrics)

    def _summarize_indices(self, indices: list[int]) -> dict[str, float | int]:
        if not indices:
            return {}
        idx = np.asarray(indices, dtype=np.int32)
        buckets = self.replay_bucket[idx]
        metrics: dict[str, float | int] = {
            "chunks": int(idx.shape[0]),
            "unique_chunk_frac": float(np.unique(idx).shape[0] / max(idx.shape[0], 1)),
            "repeat_bucket_frac": float(np.mean(buckets == 2, dtype=np.float64)),
            "once_bucket_frac": float(np.mean(buckets == 1, dtype=np.float64)),
            "repeat_reuse_frac": float(np.mean(self.use_counts[idx] > 1, dtype=np.float64)),
        }

        def maybe_mean(name: str, values: np.ndarray | None) -> None:
            if values is None:
                return
            metrics[name] = float(np.asarray(values[idx], dtype=np.float32).mean(dtype=np.float64))

        maybe_mean("mean_difficulty", self.features.difficulty)
        maybe_mean("mean_operator_density", self.features.operator_density)
        maybe_mean("mean_compressibility", self.features.compressibility_ratio)
        if self.compressibility is not None:
            metrics["mean_human_compressibility"] = float(np.asarray(self.compressibility[idx], dtype=np.float32).mean(dtype=np.float64))
        maybe_mean("mean_learnability", self.features.learnability_score)
        maybe_mean("mean_quality", self.features.quality_score)
        return metrics

    def _reset_epoch(self) -> None:
        self.epoch += 1
        self.use_counts.fill(0)
        self.phase_cursors = {phase.name: 0 for phase in self.phase_plan}
        if self.log_fn is not None:
            self.log_fn(
                f"WARNING: curriculum epoch reset epoch:{self.epoch} "
                f"dataset:{self.dataset_name} chunks:{len(self.features)}"
            )

    def _take_indices_for_phase(self, phase: CurriculumPhase, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.int32)
        order = self.phase_orders[phase.name]
        cursor = self.phase_cursors[phase.name]
        picked: list[int] = []

        def scan_once(start_cursor: int) -> tuple[list[int], int]:
            local_picked: list[int] = []
            cursor_now = start_cursor
            scanned = 0
            total = int(order.shape[0])
            while len(local_picked) < count and scanned < total:
                idx = int(order[cursor_now])
                cursor_now = (cursor_now + 1) % total
                scanned += 1
                if self.max_uses[idx] <= 0 or self.use_counts[idx] >= self.max_uses[idx]:
                    continue
                local_picked.append(idx)
            return local_picked, cursor_now

        picked, cursor = scan_once(cursor)
        if len(picked) < count:
            self._reset_epoch()
            picked, cursor = scan_once(0)
        if len(picked) < count:
            raise RuntimeError(
                f"Curriculum loader could not find {count} eligible chunks in phase {phase.name!r}"
            )

        for idx in picked:
            self.use_counts[idx] += 1
        self.phase_cursors[phase.name] = cursor
        return np.asarray(picked, dtype=np.int32)

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        if seq_len != self.seq_len:
            raise ValueError(f"Curriculum loader was built for seq_len={self.seq_len}, got {seq_len}")
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        phase = self._pinned_phase if self._pinned_phase is not None else self.current_phase()
        indices = self._take_indices_for_phase(phase, batch_seqs)
        x = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        y = np.zeros((batch_seqs, seq_len), dtype=np.int32)
        for row_idx, feature_idx in enumerate(indices):
            shard_idx = int(self.shard_index[feature_idx])
            chunk_idx = int(self.chunk_index[feature_idx])
            x_row, y_row = self.store.fetch_sequence(shard_idx, chunk_idx)
            x[row_idx] = x_row
            y[row_idx] = y_row
        self._last_batch_metrics = self._summarize_indices(indices.tolist())
        self._step_indices.extend(int(value) for value in indices.tolist())
        self.tokens_served += usable
        return x, y
