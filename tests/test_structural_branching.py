import unittest

import numpy as np

from structural_branching import (
    adaptive_branch_length_from_divergence,
    branch_state_divergence_penalty_np,
    StructuralBranchBudgetController,
    StructuralBranchBudgetSignals,
    StructuralBranchingConfig,
    derive_structural_branching_config,
    select_structural_branch_points_np,
)


class StructuralBranchingTests(unittest.TestCase):
    def test_selects_high_miss_small_gap_branch(self) -> None:
        logits = np.array(
            [
                [
                    [0.1, 3.0, 2.8, 0.0],
                    [0.1, 2.5, 2.4, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        targets = np.array([[2, 1]], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        )
        cfg = StructuralBranchingConfig(
            enabled=True,
            max_branches=1,
            min_structural_miss=0.5,
            max_top1_gap=0.5,
            min_position_gap=1,
        )
        plans = select_structural_branch_points_np(logits, targets, embeddings, cfg)
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(plans[0]), 1)
        plan = plans[0][0]
        self.assertEqual(plan.pos, 0)
        self.assertEqual(plan.predicted_token, 1)
        self.assertEqual(plan.target_token, 2)
        self.assertGreater(plan.structural_miss, 0.9)

    def test_skips_surface_like_miss(self) -> None:
        logits = np.array([[[0.0, 3.0, 2.9, 0.1]]], dtype=np.float32)
        targets = np.array([[2]], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cfg = StructuralBranchingConfig(enabled=True, min_structural_miss=0.2, max_top1_gap=0.5)
        plans = select_structural_branch_points_np(logits, targets, embeddings, cfg)
        self.assertEqual(plans, [[]])

    def test_respects_branch_count_and_spacing(self) -> None:
        logits = np.array(
            [
                [
                    [0.0, 3.0, 2.7, 0.1],
                    [0.0, 3.0, 2.7, 0.1],
                    [0.0, 3.0, 2.7, 0.1],
                    [0.0, 3.0, 2.7, 0.1],
                ]
            ],
            dtype=np.float32,
        )
        targets = np.array([[2, 2, 2, 2]], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        )
        cfg = StructuralBranchingConfig(
            enabled=True,
            max_branches=2,
            min_structural_miss=0.5,
            max_top1_gap=0.5,
            min_position_gap=2,
        )
        plans = select_structural_branch_points_np(logits, targets, embeddings, cfg)
        self.assertEqual([plan.pos for plan in plans[0]], [0, 2])

    def test_top12_cosine_gate_skips_trivial_ambiguity(self) -> None:
        logits = np.array([[[0.0, 3.0, 2.95, 0.1]]], dtype=np.float32)
        targets = np.array([[3]], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cfg = StructuralBranchingConfig(
            enabled=True,
            min_structural_miss=0.5,
            max_top1_gap=0.5,
            max_top12_cosine=0.95,
        )
        plans = select_structural_branch_points_np(logits, targets, embeddings, cfg)
        self.assertEqual(plans, [[]])

    def test_adaptive_branch_length_stops_on_convergence(self) -> None:
        divergence = np.array([0.40, 0.22, 0.04, 0.03], dtype=np.float32)
        depth = adaptive_branch_length_from_divergence(
            divergence,
            min_depth=2,
            plateau_tol=0.02,
            converged_divergence=0.05,
        )
        self.assertEqual(depth, 3)

    def test_adaptive_branch_length_stops_on_plateau(self) -> None:
        divergence = np.array([0.20, 0.42, 0.43, 0.44, 0.45], dtype=np.float32)
        depth = adaptive_branch_length_from_divergence(
            divergence,
            min_depth=2,
            plateau_tol=0.02,
            converged_divergence=0.05,
        )
        self.assertEqual(depth, 4)

    def test_branch_state_divergence_penalty_zero_below_target(self) -> None:
        real_hidden = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        wrong_hidden = np.array([[0.95, 0.0], [0.0, 0.90]], dtype=np.float32)
        penalty = branch_state_divergence_penalty_np(
            real_hidden,
            wrong_hidden,
            effective_len=2,
            target_max_cosine=1.0,
        )
        self.assertEqual(penalty, 0.0)

    def test_branch_state_divergence_penalty_positive_above_target(self) -> None:
        real_hidden = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        wrong_hidden = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        penalty = branch_state_divergence_penalty_np(
            real_hidden,
            wrong_hidden,
            effective_len=2,
            target_max_cosine=0.25,
        )
        self.assertGreater(penalty, 0.7)

    def test_dynamic_budget_disables_easy_phase(self) -> None:
        cfg = StructuralBranchingConfig(enabled=True, max_branches=2, min_structural_miss=0.5, max_top1_gap=0.75)
        controller = StructuralBranchBudgetController(enabled=True)
        derived = derive_structural_branching_config(
            cfg,
            StructuralBranchBudgetSignals(
                phase_focus="easy",
                mean_operator_density=0.001,
                mean_human_compressibility=0.60,
            ),
            controller,
        )
        self.assertFalse(derived.enabled)
        self.assertEqual(derived.max_branches, 0)

    def test_dynamic_budget_expands_operator_dense_phase(self) -> None:
        cfg = StructuralBranchingConfig(enabled=True, max_branches=1, min_structural_miss=0.5, max_top1_gap=0.75)
        controller = StructuralBranchBudgetController(enabled=True, operator_density_high=0.02)
        derived = derive_structural_branching_config(
            cfg,
            StructuralBranchBudgetSignals(
                phase_focus="operator_dense",
                mean_operator_density=0.03,
                mean_human_compressibility=0.40,
            ),
            controller,
        )
        self.assertTrue(derived.enabled)
        self.assertGreaterEqual(derived.max_branches, 2)
        self.assertLess(derived.min_structural_miss, cfg.min_structural_miss)
        self.assertGreater(derived.max_top1_gap, cfg.max_top1_gap)

    def test_dynamic_budget_keeps_hard_low_compressibility_selective(self) -> None:
        cfg = StructuralBranchingConfig(enabled=True, max_branches=3, min_structural_miss=0.5, max_top1_gap=0.75)
        controller = StructuralBranchBudgetController(enabled=True, low_human_compressibility=0.33)
        derived = derive_structural_branching_config(
            cfg,
            StructuralBranchBudgetSignals(
                phase_focus="hard",
                mean_operator_density=0.01,
                mean_human_compressibility=0.20,
            ),
            controller,
        )
        self.assertTrue(derived.enabled)
        self.assertEqual(derived.max_branches, 1)
        self.assertGreater(derived.min_structural_miss, cfg.min_structural_miss)
        self.assertLess(derived.max_top1_gap, cfg.max_top1_gap)


if __name__ == "__main__":
    unittest.main()
