from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.runtime_mlx import (
    construct_matrix,
    parameter_key_for_role,
    parse_prior_targets,
    summarize_matrix_update,
)


class RepresentationLearningRuntimeMlxTests(unittest.TestCase):
    def test_parse_prior_targets_expands_aliases(self) -> None:
        self.assertEqual(parse_prior_targets("qkvo"), ["q", "k", "v", "o"])
        self.assertEqual(parse_prior_targets("qk,mlp"), ["q", "k", "mlp_fc", "mlp_proj"])
        self.assertEqual(parse_prior_targets("attn,mlp_out"), ["q", "k", "v", "o", "mlp_proj"])

    def test_parameter_key_for_role(self) -> None:
        self.assertEqual(parameter_key_for_role(3, "q"), "blocks.3.attn.c_q.weight")
        self.assertEqual(parameter_key_for_role(1, "o"), "blocks.1.attn.proj.weight")
        self.assertEqual(parameter_key_for_role(5, "mlp_fc"), "blocks.5.mlp.fc.weight")

    def test_construct_matrix_random_and_svd_carrier(self) -> None:
        directions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        scales = np.array([2.0, 0.5], dtype=np.float32)
        current = np.array(
            [
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.125],
            ],
            dtype=np.float32,
        )
        random_matrix = construct_matrix(
            directions,
            scales,
            out_dim=6,
            in_dim=6,
            seed=11,
            current_matrix=current,
            adapter_mode="random",
        )
        carrier_matrix = construct_matrix(
            directions,
            scales,
            out_dim=6,
            in_dim=6,
            seed=11,
            current_matrix=current,
            adapter_mode="svd_carrier",
        )
        matched_matrix = construct_matrix(
            directions,
            scales,
            out_dim=6,
            in_dim=6,
            seed=11,
            current_matrix=current,
            adapter_mode="svd_carrier_matched",
        )
        self.assertEqual(random_matrix.shape, (6, 6))
        self.assertEqual(carrier_matrix.shape, (6, 6))
        self.assertEqual(matched_matrix.shape, (6, 6))
        self.assertFalse(np.allclose(random_matrix, carrier_matrix))
        self.assertFalse(np.allclose(carrier_matrix, matched_matrix))
        self.assertAlmostEqual(
            float(np.linalg.norm(matched_matrix)),
            float(np.linalg.norm(current)),
            places=5,
        )

    def test_summarize_matrix_update(self) -> None:
        current = np.eye(3, dtype=np.float32)
        target = np.eye(3, dtype=np.float32) * 2.0
        updated = 0.5 * current + 0.5 * target
        summary = summarize_matrix_update(
            current,
            target,
            updated,
            role="q",
            weight_key="blocks.0.attn.c_q.weight",
            blend_strength=0.5,
        )
        self.assertEqual(summary["role"], "q")
        self.assertEqual(summary["weight_key"], "blocks.0.attn.c_q.weight")
        self.assertAlmostEqual(float(summary["blend_strength"]), 0.5, places=6)
        self.assertGreater(float(summary["delta_norm"]), 0.0)
        self.assertGreater(float(summary["target_ratio_vs_current"]), 0.0)


if __name__ == "__main__":
    unittest.main()
