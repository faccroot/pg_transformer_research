import unittest

import numpy as np

from residual_autocorrelation import (
    argmax_embedding_residuals,
    cosine_acf,
    detect_regime_segments,
    expected_embedding_residuals,
    factorize_residual_pca,
    scalar_acf,
    transition_window_mask,
)


class ResidualAutocorrelationTests(unittest.TestCase):
    def test_expected_embedding_residuals_matches_manual(self) -> None:
        probs = np.array([[0.25, 0.75], [0.90, 0.10]], dtype=np.float32)
        embed = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        actual = np.array([0, 1], dtype=np.int32)
        residuals = expected_embedding_residuals(probs, embed, actual)
        expected = np.array([[0.75, -0.75], [-0.90, 0.90]], dtype=np.float32)
        np.testing.assert_allclose(residuals, expected, atol=1e-6)

    def test_argmax_embedding_residuals_matches_manual(self) -> None:
        logits = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        embed = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        actual = np.array([0, 1], dtype=np.int32)
        residuals = argmax_embedding_residuals(logits, embed, actual)
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(residuals, expected, atol=1e-6)

    def test_cosine_acf_on_constant_direction_is_one(self) -> None:
        vecs = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (8, 1))
        acf = cosine_acf(vecs, max_lag=3)
        self.assertEqual([row["lag"] for row in acf], [1, 2, 3])
        for row in acf:
            self.assertAlmostEqual(float(row["mean_cosine"]), 1.0, places=6)

    def test_scalar_acf_detects_positive_serial_correlation(self) -> None:
        values = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
        acf = scalar_acf(values, max_lag=2)
        self.assertGreater(float(acf[0]["corr"]), 0.9)
        self.assertGreater(float(acf[1]["corr"]), 0.9)

    def test_detect_regime_segments_marks_low_cosine_breaks(self) -> None:
        hidden = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.0, 0.9],
                [-1.0, 0.0],
                [-0.9, 0.0],
            ],
            dtype=np.float32,
        )
        result = detect_regime_segments(
            hidden,
            cosine_threshold=0.2,
            min_segment_length=2,
        )
        np.testing.assert_array_equal(result["transition_positions"], np.array([2, 4], dtype=np.int32))
        np.testing.assert_array_equal(result["segment_ids"], np.array([0, 0, 1, 1, 2, 2], dtype=np.int32))

    def test_cosine_acf_within_vs_cross_respects_segments(self) -> None:
        vecs = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        segs = np.array([0, 0, 1, 1], dtype=np.int32)
        within = cosine_acf(vecs, max_lag=1, segment_ids=segs, relation="within")
        cross = cosine_acf(vecs, max_lag=1, segment_ids=segs, relation="cross")
        self.assertAlmostEqual(float(within[0]["mean_cosine"]), 1.0, places=6)
        self.assertAlmostEqual(float(cross[0]["mean_cosine"]), 0.0, places=6)

    def test_transition_window_mask_marks_post_transition_ranges(self) -> None:
        mask = transition_window_mask(10, np.array([2, 7], dtype=np.int32), window=2)
        expected = np.array([False, False, True, True, False, False, False, True, True, False], dtype=np.bool_)
        np.testing.assert_array_equal(mask, expected)

    def test_factorize_residual_pca_recovers_rank1_direction(self) -> None:
        base = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        coeffs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        vecs = coeffs[:, None] * base[None, :]
        factored = factorize_residual_pca(vecs, max_factors=2)
        self.assertEqual(tuple(factored["scores"].shape), (5, 2))
        self.assertGreater(float(factored["explained_variance_ratio"][0]), 0.99)
        lead = np.asarray(factored["components"][0], dtype=np.float32)
        cosine = float(np.dot(lead, base) / (np.linalg.norm(lead) * np.linalg.norm(base)))
        self.assertGreater(abs(cosine), 0.99)


if __name__ == "__main__":
    unittest.main()
