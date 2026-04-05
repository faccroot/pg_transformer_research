from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.merge_cascade_representations import (
    build_model_zoo_matrix,
    greedy_model_order,
    merge_cascade_representations,
)
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class MergeCascadeRepresentationsTests(unittest.TestCase):
    def test_build_model_zoo_matrix_has_expected_shape(self) -> None:
        rep = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([2.0], dtype=np.float32),
                )
            },
        )
        matrix = build_model_zoo_matrix([rep], canonical_dim=4, num_layers=2, top_k=2)
        self.assertEqual(matrix.shape, (1, (2 * (2 * 4 + 2)) + 2))

    def test_greedy_order_includes_all_models(self) -> None:
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        order = greedy_model_order(matrix, ["a", "b", "c"])
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(len(order), 3)

    def test_merge_keeps_novel_residual_contribution(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([5.0], dtype=np.float32),
                )
            },
        )
        rep_b = ModelRepresentation(
            model_id="model/b",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[0.99, 0.01, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
        )
        rep_c = ModelRepresentation(
            model_id="model/c",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([3.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_cascade_representations(
            [rep_a, rep_b, rep_c],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            min_residual_ratio=0.1,
            max_base_directions=2,
            max_residual_directions=1,
        )
        layer = geometry.layer_geometries[1]
        self.assertEqual(layer.directions.shape, (2, 4))
        contrib = layer.metadata["contribution_by_model"]
        self.assertIn("model/a", contrib)
        self.assertIn("model/c", contrib)
        self.assertGreaterEqual(contrib["model/c"]["accepted_direction_count"], 1)
        self.assertLessEqual(contrib["model/c"]["accepted_direction_count"], 1)
        self.assertEqual(geometry.metadata["selection_method"], "cascade_sparse_residual_merge")
        self.assertEqual(len(geometry.metadata["model_order"]), 3)
        self.assertIn("novelty_weight_by_model", geometry.metadata)
        self.assertEqual(layer.metadata["selection_method"], "cascade_sparse_residual_merge")

    def test_residual_cap_limits_later_models(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.eye(4, dtype=np.float32),
                    scales=np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32),
                )
            },
        )
        rep_b = ModelRepresentation(
            model_id="model/b",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array(
                        [
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    scales=np.array([6.0, 5.0, 4.0, 3.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_cascade_representations(
            [rep_a, rep_b],
            canonical_dim=4,
            num_layers=1,
            top_k=3,
            max_base_directions=2,
            max_residual_directions=1,
            min_residual_ratio=0.0,
            min_model_score_ratio=0.0,
        )
        contrib = geometry.layer_geometries[1].metadata["contribution_by_model"]
        self.assertEqual(contrib["model/a"]["accepted_direction_count"], 2)
        self.assertEqual(contrib["model/b"]["accepted_direction_count"], 1)


if __name__ == "__main__":
    unittest.main()
