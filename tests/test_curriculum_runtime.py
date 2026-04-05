from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from curriculum_runtime import CurriculumRuntimeConfig, CurriculumTokenLoader, load_phase_plan


def write_fake_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros((256,), dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    payload = np.asarray(tokens, dtype="<u2")
    with path.open("wb") as handle:
        header.tofile(handle)
        payload.tofile(handle)


class CurriculumRuntimeTests(unittest.TestCase):
    def test_load_phase_plan_defaults(self) -> None:
        phases = load_phase_plan(None)
        self.assertEqual(phases[0].name, "structural-foundation")
        self.assertEqual(phases[-1].name, "consolidation")

    def test_loader_uses_operator_dense_phase_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            shard_path = tmpdir / "fineweb_train_000000.bin"
            write_fake_shard(shard_path, list(range(14)))

            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                chunk_size=np.array([4], dtype=np.int32),
                cluster_ids=np.array([0, 1, 2], dtype=np.int32),
                operator_density=np.array([0.0, 0.8, 0.1], dtype=np.float32),
                difficulty=np.array([0.2, 0.9, 0.5], dtype=np.float32),
                duplicate_score=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                replay_bucket=np.array([1, 1, 1], dtype=np.int8),
                shard_index=np.array([0, 0, 0], dtype=np.int32),
                chunk_index=np.array([0, 1, 2], dtype=np.int32),
            )

            loader = CurriculumTokenLoader(
                str(tmpdir / "fineweb_train_*.bin"),
                seq_len=4,
                total_train_tokens=40,
                runtime_config=CurriculumRuntimeConfig(features_path=str(features_path)),
            )
            loader.tokens_served = 8
            x, y = loader.next_batch_np(batch_tokens=4, seq_len=4)
            np.testing.assert_array_equal(x[0], np.array([4, 5, 6, 7], dtype=np.int32))
            np.testing.assert_array_equal(y[0], np.array([5, 6, 7, 8], dtype=np.int32))

    def test_loader_uses_easy_phase_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            shard_path = tmpdir / "fineweb_train_000000.bin"
            write_fake_shard(shard_path, list(range(14)))

            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                chunk_size=np.array([4], dtype=np.int32),
                cluster_ids=np.array([0, 1, 2], dtype=np.int32),
                operator_density=np.array([0.0, 0.8, 0.1], dtype=np.float32),
                difficulty=np.array([0.1, 0.9, 0.5], dtype=np.float32),
                duplicate_score=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                replay_bucket=np.array([1, 1, 1], dtype=np.int8),
                shard_index=np.array([0, 0, 0], dtype=np.int32),
                chunk_index=np.array([0, 1, 2], dtype=np.int32),
            )

            loader = CurriculumTokenLoader(
                str(tmpdir / "fineweb_train_*.bin"),
                seq_len=4,
                total_train_tokens=40,
                runtime_config=CurriculumRuntimeConfig(features_path=str(features_path)),
            )
            loader.tokens_served = 24
            x, y = loader.next_batch_np(batch_tokens=4, seq_len=4)
            np.testing.assert_array_equal(x[0], np.array([0, 1, 2, 3], dtype=np.int32))
            np.testing.assert_array_equal(y[0], np.array([1, 2, 3, 4], dtype=np.int32))

    def test_loader_pins_phase_for_full_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            shard_path = tmpdir / "fineweb_train_000000.bin"
            write_fake_shard(shard_path, list(range(18)))

            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                chunk_size=np.array([4], dtype=np.int32),
                cluster_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                operator_density=np.array([0.1, 0.9, 0.7, 0.0], dtype=np.float32),
                difficulty=np.array([0.2, 0.3, 0.4, 0.95], dtype=np.float32),
                duplicate_score=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                replay_bucket=np.array([1, 1, 1, 1], dtype=np.int8),
                shard_index=np.array([0, 0, 0, 0], dtype=np.int32),
                chunk_index=np.array([0, 1, 2, 3], dtype=np.int32),
            )

            loader = CurriculumTokenLoader(
                str(tmpdir / "fineweb_train_*.bin"),
                seq_len=4,
                total_train_tokens=40,
                runtime_config=CurriculumRuntimeConfig(features_path=str(features_path)),
            )
            loader.tokens_served = 15
            phase = loader.begin_step()
            self.assertEqual(phase.name, "logic-polarity")
            x_first, _y_first = loader.next_batch_np(batch_tokens=4, seq_len=4)
            x_second, _y_second = loader.next_batch_np(batch_tokens=4, seq_len=4)
            loader.end_step()
            np.testing.assert_array_equal(x_first[0], np.array([4, 5, 6, 7], dtype=np.int32))
            np.testing.assert_array_equal(x_second[0], np.array([8, 9, 10, 11], dtype=np.int32))

    def test_loader_step_metrics_report_served_chunk_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            shard_path = tmpdir / "fineweb_train_000000.bin"
            write_fake_shard(shard_path, list(range(18)))

            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                chunk_size=np.array([4], dtype=np.int32),
                cluster_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                operator_density=np.array([0.1, 0.8, 0.7, 0.0], dtype=np.float32),
                difficulty=np.array([0.2, 0.3, 0.4, 0.95], dtype=np.float32),
                compressibility_ratio=np.array([0.5, 0.2, 0.4, 0.9], dtype=np.float32),
                learnability_score=np.array([0.8, 0.7, 0.6, 0.2], dtype=np.float32),
                quality_score=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
                duplicate_score=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                replay_bucket=np.array([1, 2, 2, 1], dtype=np.int8),
                shard_index=np.array([0, 0, 0, 0], dtype=np.int32),
                chunk_index=np.array([0, 1, 2, 3], dtype=np.int32),
            )

            loader = CurriculumTokenLoader(
                str(tmpdir / "fineweb_train_*.bin"),
                seq_len=4,
                total_train_tokens=40,
                runtime_config=CurriculumRuntimeConfig(features_path=str(features_path)),
            )
            loader.tokens_served = 15
            loader.begin_step()
            loader.next_batch_np(batch_tokens=8, seq_len=4)
            loader.end_step()
            metrics = loader.last_step_metrics()
            self.assertEqual(metrics["chunks"], 2)
            self.assertAlmostEqual(float(metrics["unique_chunk_frac"]), 1.0)
            self.assertAlmostEqual(float(metrics["repeat_bucket_frac"]), 1.0)
            self.assertGreater(float(metrics["mean_operator_density"]), 0.7)

    def test_loader_filters_low_compressibility_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            shard_path = tmpdir / "fineweb_train_000000.bin"
            write_fake_shard(shard_path, list(range(18)))

            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                chunk_size=np.array([4], dtype=np.int32),
                cluster_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                operator_density=np.array([0.0, 0.9, 0.7, 0.1], dtype=np.float32),
                difficulty=np.array([0.2, 0.6, 0.5, 0.3], dtype=np.float32),
                compressibility_ratio=np.array([0.30, 0.75, 0.35, 0.40], dtype=np.float32),
                duplicate_score=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                replay_bucket=np.array([1, 1, 1, 1], dtype=np.int8),
                shard_index=np.array([0, 0, 0, 0], dtype=np.int32),
                chunk_index=np.array([0, 1, 2, 3], dtype=np.int32),
            )

            loader = CurriculumTokenLoader(
                str(tmpdir / "fineweb_train_*.bin"),
                seq_len=4,
                total_train_tokens=40,
                runtime_config=CurriculumRuntimeConfig(
                    features_path=str(features_path),
                    min_compressibility=0.40,
                ),
            )
            summary = loader.summary()
            self.assertEqual(int(summary["filtered_low_compressibility_chunks"]), 1)
            self.assertEqual(int(summary["skipped_chunks"]), 1)

            loader.tokens_served = 8
            x, y = loader.next_batch_np(batch_tokens=4, seq_len=4)
            np.testing.assert_array_equal(x[0], np.array([8, 9, 10, 11], dtype=np.int32))
            np.testing.assert_array_equal(y[0], np.array([9, 10, 11, 12], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
