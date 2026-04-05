import unittest

import numpy as np

from harmonic_boundary_metrics import (
    aggregate_patch_features,
    segment_lengths_from_ids,
    summarize_boundary_alignment,
    summarize_segment_lengths,
)


class HarmonicBoundaryMetricsTests(unittest.TestCase):
    def test_aggregate_patch_features_first_and_any(self) -> None:
        token_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        feature_lut = np.array(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
                [0, 0],
            ],
            dtype=np.int32,
        )
        first = aggregate_patch_features(token_ids, feature_lut, 2, reduction="first")
        any_feats = aggregate_patch_features(token_ids, feature_lut, 2, reduction="any")
        np.testing.assert_array_equal(first, np.array([[1, 0], [0, 0], [0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(any_feats, np.array([[1, 1], [1, 1], [0, 0]], dtype=np.float32))

    def test_segment_lengths_from_ids(self) -> None:
        segment_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
        lengths = segment_lengths_from_ids(segment_ids)
        np.testing.assert_array_equal(lengths, np.array([2, 3, 1], dtype=np.int32))
        summary = summarize_segment_lengths(lengths)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["max"], 3)
        self.assertAlmostEqual(float(summary["mean"]), 2.0)

    def test_summarize_boundary_alignment_reports_enrichment_and_sources(self) -> None:
        boundary_flags = np.array([1, 0, 1, 0], dtype=np.bool_)
        valid_mask = np.array([1, 1, 1, 1], dtype=np.bool_)
        features = np.array(
            [
                [1, 0],
                [0, 0],
                [1, 1],
                [0, 0],
            ],
            dtype=np.float32,
        )
        summary = summarize_boundary_alignment(
            boundary_flags,
            valid_mask,
            features,
            ("whitespace_like", "quote_like"),
            threshold_boundary_flags=np.array([0, 0, 1, 0], dtype=np.bool_),
            periodic_boundary_flags=np.array([1, 0, 0, 0], dtype=np.bool_),
            flux=np.array([0.0, 0.1, 0.9, 0.2], dtype=np.float32),
            exclude_first_patch=True,
        )
        self.assertEqual(summary["patches_analyzed"], 3)
        self.assertEqual(summary["boundaries_analyzed"], 1)
        self.assertAlmostEqual(float(summary["boundary_rate"]), 1.0 / 3.0)
        self.assertAlmostEqual(
            float(summary["feature_alignment"]["whitespace_like"]["boundary_rate"]),
            1.0,
        )
        self.assertGreater(
            float(summary["feature_alignment"]["whitespace_like"]["enrichment"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(summary["boundary_sources"]["threshold_fraction_of_boundaries"]),
            1.0,
        )
        self.assertAlmostEqual(float(summary["flux"]["boundary_mean"]), 0.9)


if __name__ == "__main__":
    unittest.main()
