import unittest

from sidecar_aux import SidecarAuxBudgetController, derive_sidecar_aux_scale


class SidecarAuxTests(unittest.TestCase):
    def test_operator_dense_phase_increases_scale(self) -> None:
        controller = SidecarAuxBudgetController(enabled=True)
        scale = derive_sidecar_aux_scale(
            base_scale=1.0,
            phase_focus="operator_dense",
            controller=controller,
            mean_operator_density=0.03,
            mean_human_compressibility=0.40,
        )
        self.assertGreater(scale, 1.0)

    def test_easy_high_compressibility_reduces_scale(self) -> None:
        controller = SidecarAuxBudgetController(enabled=True)
        scale = derive_sidecar_aux_scale(
            base_scale=1.0,
            phase_focus="easy",
            controller=controller,
            mean_operator_density=0.001,
            mean_human_compressibility=0.65,
        )
        self.assertLess(scale, 1.0)

    def test_scale_is_clamped(self) -> None:
        controller = SidecarAuxBudgetController(enabled=True, min_scale=0.8, max_scale=1.1)
        scale = derive_sidecar_aux_scale(
            base_scale=1.0,
            phase_focus="operator_dense",
            controller=controller,
            mean_operator_density=1.0,
            mean_human_compressibility=0.10,
        )
        self.assertLessEqual(scale, 1.1)
        self.assertGreaterEqual(scale, 0.8)


if __name__ == "__main__":
    unittest.main()
