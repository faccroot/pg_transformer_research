import unittest

import numpy as np

from sidecar_transition import (
    SidecarTransitionResetController,
    blend_transition_reset_signals,
    transition_reset_prior_from_cosine,
)


class SidecarTransitionTests(unittest.TestCase):
    def test_lower_cosine_means_higher_prior(self) -> None:
        controller = SidecarTransitionResetController(enabled=True, cosine_threshold=0.8, cosine_sharpness=10.0)
        low = transition_reset_prior_from_cosine(0.2, controller)
        high = transition_reset_prior_from_cosine(0.95, controller)
        self.assertGreater(float(low), float(high))

    def test_disabled_controller_returns_zero_prior(self) -> None:
        controller = SidecarTransitionResetController(enabled=False)
        prior = transition_reset_prior_from_cosine(np.array([0.1, 0.9], dtype=np.float32), controller)
        np.testing.assert_array_equal(prior, np.zeros((2,), dtype=np.float32))

    def test_blend_is_clamped_and_scaled(self) -> None:
        controller = SidecarTransitionResetController(
            enabled=True,
            prior_weight=1.0,
            learned_weight=1.0,
            max_gate=0.75,
        )
        gate = blend_transition_reset_signals(
            np.array([0.9], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            controller,
        )
        self.assertAlmostEqual(float(gate[0]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
