from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.extract_model_representation import (
    chunk_projection_summary,
    dominant_directions_from_covariance,
    require_chunk_projection_coverage,
)


class ExtractModelRepresentationTests(unittest.TestCase):
    def test_dominant_directions_sanitizes_non_finite_covariance(self) -> None:
        covariance = np.array(
            [
                [1.0, np.inf, 0.0],
                [np.inf, 2.0, np.nan],
                [0.0, np.nan, 3.0],
            ],
            dtype=np.float64,
        )
        directions, scales = dominant_directions_from_covariance(covariance, top_k=2)
        self.assertEqual(directions.shape, (2, 3))
        self.assertEqual(scales.shape, (2,))
        self.assertTrue(np.isfinite(directions).all())
        self.assertTrue(np.isfinite(scales).all())

    def test_chunk_projection_summary_reports_missing_layers(self) -> None:
        summary = chunk_projection_summary(
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={1: np.ones((2, 4), dtype=np.float32)},
            requested_layers=[1, 4],
        )
        self.assertEqual(summary["projection_layers"], [1])
        self.assertEqual(summary["missing_projection_layers"], [4])
        self.assertEqual(summary["expected_chunk_count"], 2)

    def test_require_chunk_projection_coverage_raises_on_missing_layers(self) -> None:
        with self.assertRaises(RuntimeError):
            require_chunk_projection_coverage(
                chunk_ids=["c1", "c2"],
                chunk_layer_projections={1: np.ones((2, 4), dtype=np.float32)},
                requested_layers=[1, 4],
            )


if __name__ == "__main__":
    unittest.main()
