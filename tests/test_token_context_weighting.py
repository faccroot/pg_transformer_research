import unittest

import numpy as np

from token_context_weighting import normalize_context_delta_scores_np


class TokenContextWeightingTests(unittest.TestCase):
    def test_positive_delta_boosts_weights(self) -> None:
        long_nll = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        short_nll = np.array([[1.1, 2.8, 3.0]], dtype=np.float32)
        weights = normalize_context_delta_scores_np(long_nll, short_nll)
        np.testing.assert_allclose(weights, np.array([[1.1, 1.8, 1.0]], dtype=np.float32))

    def test_use_absolute_delta_keeps_negative_gaps(self) -> None:
        long_nll = np.array([[1.0, 2.0]], dtype=np.float32)
        short_nll = np.array([[0.8, 2.0]], dtype=np.float32)
        abs_weights = normalize_context_delta_scores_np(long_nll, short_nll, use_absolute_delta=True)
        pos_weights = normalize_context_delta_scores_np(long_nll, short_nll, use_absolute_delta=False)
        np.testing.assert_allclose(abs_weights, np.array([[1.2, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(pos_weights, np.array([[1.0, 1.0]], dtype=np.float32))

    def test_topk_fraction_masks_nonselected_tokens(self) -> None:
        long_nll = np.zeros((1, 4), dtype=np.float32)
        short_nll = np.array([[0.1, 1.0, 0.3, 0.2]], dtype=np.float32)
        weights = normalize_context_delta_scores_np(long_nll, short_nll, topk_fraction=0.25)
        np.testing.assert_allclose(weights, np.array([[0.0, 2.0, 0.0, 0.0]], dtype=np.float32))

    def test_max_multiplier_clips_large_scores(self) -> None:
        long_nll = np.zeros((1, 2), dtype=np.float32)
        short_nll = np.array([[1.0, 10.0]], dtype=np.float32)
        weights = normalize_context_delta_scores_np(long_nll, short_nll, max_multiplier=3.0)
        np.testing.assert_allclose(weights, np.array([[2.0, 3.0]], dtype=np.float32))

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_context_delta_scores_np(
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
