from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.merge_moonshot_representations import merge_moonshot_representations
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class MergeMoonshotRepresentationsTests(unittest.TestCase):
    def test_moonshot_merge_records_shared_and_unique_budgets(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    scales=np.array([5.0, 3.0], dtype=np.float32),
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
                            [0.99, 0.01, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    scales=np.array([4.5, 4.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_moonshot_representations(
            [rep_a, rep_b],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.9,
            shared_fraction=0.5,
        )
        self.assertEqual(geometry.metadata["selection_method"], "moonshot_stratified_hardmax_merge")
        layer = geometry.layer_geometries[1]
        self.assertEqual(layer.metadata["selection_method"], "moonshot_stratified_hardmax_merge")
        self.assertEqual(layer.metadata["shared_budget"], 1)
        self.assertEqual(layer.metadata["unique_budget"], 1)
        cluster_types = [cluster["cluster_type"] for cluster in layer.metadata["clusters"]]
        self.assertIn("shared", cluster_types)
        self.assertIn("unique", cluster_types)

    def test_moonshot_merge_keeps_unique_model_cluster(self) -> None:
        rep_shared = ModelRepresentation(
            model_id="model/shared",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
        )
        rep_shared2 = ModelRepresentation(
            model_id="model/shared2",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.98, 0.02, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([3.5], dtype=np.float32),
                )
            },
        )
        rep_unique = ModelRepresentation(
            model_id="model/unique",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([5.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_moonshot_representations(
            [rep_shared, rep_shared2, rep_unique],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.9,
            shared_fraction=0.5,
        )
        champion_models = [cluster["champion_model"] for cluster in geometry.layer_geometries[1].metadata["clusters"]]
        self.assertIn("model/unique", champion_models)


if __name__ == "__main__":
    unittest.main()
