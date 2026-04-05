from __future__ import annotations

import unittest

import numpy as np

from curriculum import (
    SHOW_NEVER,
    SHOW_ONCE,
    SHOW_REPEAT,
    ChunkFeatures,
    CurriculumPhase,
    classify_replay_buckets,
    cosine_kmeans,
    default_phase_plan,
    hashed_token_histograms,
    operator_density,
    phase_for_progress,
    score_chunk_priority,
    validate_phase_plan,
    zlib_compressibility_ratio,
)


class CurriculumTests(unittest.TestCase):
    def test_default_phase_plan_is_contiguous(self) -> None:
        validate_phase_plan(default_phase_plan())

    def test_phase_for_progress_switches_on_boundaries(self) -> None:
        phases = default_phase_plan()
        self.assertEqual(phase_for_progress(0.00, phases).name, "structural-foundation")
        self.assertEqual(phase_for_progress(0.20, phases).name, "logic-polarity")
        self.assertEqual(phase_for_progress(0.95, phases).name, "consolidation")

    def test_hashed_token_histograms_normalize_and_ignore_tail(self) -> None:
        tokens = np.array([0, 1, 1, 5, 6], dtype=np.int32)
        hist = hashed_token_histograms(tokens, chunk_size=2, num_bins=4)
        self.assertEqual(hist.shape, (2, 4))
        np.testing.assert_allclose(hist.sum(axis=1), np.ones((2,), dtype=np.float32))
        np.testing.assert_allclose(hist[0], np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(hist[1], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    def test_operator_density_counts_fraction_per_chunk(self) -> None:
        chunks = np.array(
            [
                [1, 2, 3, 4],
                [4, 4, 4, 5],
            ],
            dtype=np.int32,
        )
        density = operator_density(chunks, chunk_size=4, operator_token_ids=[4, 5])
        np.testing.assert_allclose(density, np.array([0.25, 1.0], dtype=np.float32))

    def test_zlib_compressibility_ratio_prefers_repetition(self) -> None:
        chunks = np.array(
            [
                [7] * 32,
                list(range(32)),
            ],
            dtype=np.int32,
        )
        ratios = zlib_compressibility_ratio(chunks, chunk_size=32, level=6)
        self.assertLess(ratios[0], ratios[1])

    def test_cosine_kmeans_splits_simple_modes(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        assignments, _centers = cosine_kmeans(points, num_clusters=2, iterations=6, seed=7)
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])

    def test_diverse_phase_prefers_unseen_clusters(self) -> None:
        phase = CurriculumPhase("structural-foundation", 0.0, 1.0, focus="diverse")
        features = ChunkFeatures(
            cluster_ids=np.array([0, 0, 1], dtype=np.int32),
            difficulty=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            operator_density=np.zeros((3,), dtype=np.float32),
        )
        scores = score_chunk_priority(features, phase, seen_clusters=np.array([True, False], dtype=bool))
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[2], scores[1])

    def test_easy_and_hard_phases_pull_in_opposite_directions(self) -> None:
        features = ChunkFeatures(
            difficulty=np.array([0.1, 0.9], dtype=np.float32),
            compressibility_ratio=np.array([0.2, 0.8], dtype=np.float32),
            operator_density=np.zeros((2,), dtype=np.float32),
        )
        easy_scores = score_chunk_priority(
            features,
            CurriculumPhase("qat-geometry", 0.0, 1.0, focus="easy"),
        )
        hard_scores = score_chunk_priority(
            features,
            CurriculumPhase("hard-content", 0.0, 1.0, focus="hard"),
        )
        self.assertGreater(easy_scores[0], easy_scores[1])
        self.assertGreater(hard_scores[1], hard_scores[0])

    def test_hard_phase_prefers_learnable_chunks_over_irreducible_noise(self) -> None:
        features = ChunkFeatures(
            difficulty=np.array([0.8, 0.8], dtype=np.float32),
            learnability_score=np.array([0.9, 0.1], dtype=np.float32),
            quality_score=np.array([0.9, 0.9], dtype=np.float32),
        )
        hard_scores = score_chunk_priority(
            features,
            CurriculumPhase("hard-content", 0.0, 1.0, focus="hard"),
        )
        self.assertGreater(hard_scores[0], hard_scores[1])

    def test_replay_bucket_classifier_matches_expected_cases(self) -> None:
        features = ChunkFeatures(
            difficulty=np.array([0.1, 0.8, 0.5], dtype=np.float32),
            operator_density=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            duplicate_score=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            confidence=np.array([0.95, 0.2, 0.6], dtype=np.float32),
        )
        buckets = classify_replay_buckets(features)
        np.testing.assert_array_equal(
            buckets,
            np.array([SHOW_NEVER, SHOW_REPEAT, SHOW_ONCE], dtype=np.int8),
        )

    def test_replay_bucket_drops_low_learnability_chunks(self) -> None:
        features = ChunkFeatures(
            difficulty=np.array([0.7, 0.7], dtype=np.float32),
            learnability_score=np.array([0.05, 0.8], dtype=np.float32),
            quality_score=np.array([0.9, 0.9], dtype=np.float32),
        )
        buckets = classify_replay_buckets(features)
        np.testing.assert_array_equal(
            buckets,
            np.array([SHOW_NEVER, SHOW_REPEAT], dtype=np.int8),
        )


if __name__ == "__main__":
    unittest.main()
