from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.extract_jacobian_spectral_representation import leading_spectral_directions_from_jtj


class ExtractJacobianSpectralRepresentationTests(unittest.TestCase):
    def test_recovers_top_linear_operator_directions(self) -> None:
        matrix = np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )

        def apply_jtj(vector: np.ndarray) -> np.ndarray:
            return matrix.T @ matrix @ vector

        directions, scales = leading_spectral_directions_from_jtj(
            apply_jtj,
            dim=3,
            top_k=2,
            power_steps=8,
            seed=123,
        )
        self.assertEqual(directions.shape, (2, 3))
        self.assertEqual(scales.shape, (2,))
        self.assertTrue(np.all(scales[:-1] >= scales[1:]))
        top_alignment = float(np.abs(directions[0] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        second_alignment = float(np.abs(directions[1] @ np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        self.assertGreater(top_alignment, 0.99)
        self.assertGreater(second_alignment, 0.99)
        self.assertAlmostEqual(float(scales[0]), 3.0, places=2)
        self.assertAlmostEqual(float(scales[1]), 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
