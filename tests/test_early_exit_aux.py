import unittest

import numpy as np

from early_exit_aux import (
    EarlyExitBudgetController,
    aligned_horizon_views_np,
    derive_early_exit_aux_weight,
    parse_horizons,
    select_contiguous_draft_horizons,
)


class EarlyExitAuxTests(unittest.TestCase):
    def test_parse_horizons_dedupes_and_preserves_order(self) -> None:
        self.assertEqual(parse_horizons("1, 2, 3, 2"), (1, 2, 3))

    def test_parse_horizons_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            parse_horizons("1,0,2")

    def test_aligned_horizon_views_shift_targets(self) -> None:
        targets = np.array([[10, 11, 12, 13]], dtype=np.int32)
        weights = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        shifted_targets, shifted_weights = aligned_horizon_views_np(targets, 3, weights)
        np.testing.assert_array_equal(shifted_targets, np.array([[12, 13]], dtype=np.int32))
        assert shifted_weights is not None
        np.testing.assert_allclose(shifted_weights, np.array([[3.0, 4.0]], dtype=np.float32))

    def test_aligned_horizon_views_empty_when_out_of_range(self) -> None:
        targets = np.array([[1, 2]], dtype=np.int32)
        shifted_targets, shifted_weights = aligned_horizon_views_np(targets, 4)
        self.assertEqual(shifted_targets.shape, (1, 0))
        self.assertIsNone(shifted_weights)

    def test_derive_early_exit_aux_weight_operator_dense_boosts(self) -> None:
        controller = EarlyExitBudgetController(enabled=True)
        weight = derive_early_exit_aux_weight(
            0.10,
            phase_focus="operator_dense",
            controller=controller,
            mean_operator_density=0.03,
            mean_human_compressibility=0.40,
        )
        self.assertGreater(weight, 0.10)

    def test_derive_early_exit_aux_weight_easy_high_compressibility_reduces(self) -> None:
        controller = EarlyExitBudgetController(enabled=True)
        weight = derive_early_exit_aux_weight(
            0.10,
            phase_focus="easy",
            controller=controller,
            mean_operator_density=0.001,
            mean_human_compressibility=0.60,
        )
        self.assertLess(weight, 0.10)

    def test_derive_early_exit_aux_weight_clamps_scale(self) -> None:
        controller = EarlyExitBudgetController(enabled=True, min_scale=0.8, max_scale=1.2)
        weight = derive_early_exit_aux_weight(
            0.10,
            phase_focus="operator_dense",
            controller=controller,
            mean_operator_density=0.20,
            mean_human_compressibility=0.10,
        )
        self.assertAlmostEqual(weight, 0.12, places=6)

    def test_select_contiguous_draft_horizons_accepts_prefix_only(self) -> None:
        accepted = select_contiguous_draft_horizons(
            (1, 2, 3),
            (0.95, 0.91, 0.40),
            threshold=0.70,
            max_tokens=3,
        )
        self.assertEqual(accepted, (1, 2))

    def test_select_contiguous_draft_horizons_stops_on_gap(self) -> None:
        accepted = select_contiguous_draft_horizons(
            (1, 3, 4),
            (0.95, 0.99, 0.99),
            threshold=0.70,
            max_tokens=4,
        )
        self.assertEqual(accepted, (1,))

    def test_base_exportable_state_strips_early_exit_heads(self) -> None:
        try:
            import train_gpt_mlx as base
        except ModuleNotFoundError as exc:
            self.skipTest(f"trainer import unavailable in unit environment: {exc}")
        model = base.GPT(
            vocab_size=32,
            num_layers=4,
            num_layer_templates=4,
            dim=16,
            num_heads=4,
            num_kv_heads=4,
            mlp_mult=2,
            mlp_leaky_slope=0.5,
            tie_embeddings=False,
            logit_chunk_tokens=0,
            logit_softcap=30.0,
            rope_base=10000.0,
            tied_embed_init_std=0.02,
            qk_gain_init=1.0,
            early_exit_layer_index=1,
            early_exit_horizons=(1, 2, 3),
            early_exit_aux_weight=0.1,
            early_exit_head_init_std=0.005,
            early_exit_cascaded_enabled=True,
            early_exit_condition_init_std=0.001,
        )
        self.assertEqual(len(model.early_exit_heads), 3)
        self.assertEqual(len(model.early_exit_condition_projs), 3)
        flat = base.exportable_flat_state(model)
        self.assertTrue(flat)
        self.assertFalse(any("early_exit_heads" in key for key in flat))
        self.assertFalse(any("early_exit_condition_projs" in key for key in flat))


if __name__ == "__main__":
    unittest.main()
