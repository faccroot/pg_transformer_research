from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.merge_hardmax_hybrid_representations import merge_hardmax_hybrid_representations
from tools.representation_learning.merge_hybrid_cascade_representations import merge_hybrid_cascade_representations
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class MergeHybridCascadeRepresentationsTests(unittest.TestCase):
    def test_hybrid_merge_records_novelty_and_order(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
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
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
        )
        rep_c = ModelRepresentation(
            model_id="model/c",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
                    scales=np.array([6.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_hybrid_cascade_representations(
            [rep_a, rep_b, rep_c],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.5,
        )
        self.assertEqual(geometry.metadata["selection_method"], "hybrid_cascade_guided_spectral_merge")
        self.assertEqual(len(geometry.metadata["model_order"]), 3)
        self.assertIn("novelty_weight_by_model", geometry.metadata)
        layer = geometry.layer_geometries[1]
        self.assertEqual(layer.metadata["selection_method"], "hybrid_cascade_guided_spectral_merge")
        self.assertEqual(layer.directions.shape, (2, 4))
        self.assertEqual(len(layer.metadata["clusters"]), 2)
        self.assertIn("winner_margin", layer.metadata["clusters"][0])

    def test_hybrid_merge_prefers_unique_cluster_when_enabled(self) -> None:
        rep_shared_a = ModelRepresentation(
            model_id="model/shared_a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([3.0], dtype=np.float32),
                )
            },
        )
        rep_shared_b = ModelRepresentation(
            model_id="model/shared_b",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.98, 0.02, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([3.0], dtype=np.float32),
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
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_hybrid_cascade_representations(
            [rep_shared_a, rep_shared_b, rep_unique],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.9,
            novelty_power=1.5,
            unique_bonus=1.0,
            shared_support_bonus=0.0,
        )
        champion_models = [cluster["champion_model"] for cluster in geometry.layer_geometries[1].metadata["clusters"]]
        self.assertIn("model/unique", champion_models)

    def test_hardmax_wrapper_sets_selection_method(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
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
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
        )
        geometry = merge_hardmax_hybrid_representations(
            [rep_a, rep_b],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.5,
        )
        self.assertEqual(geometry.metadata["selection_method"], "hardmax_cascade_guided_spectral_merge")
        self.assertEqual(
            geometry.layer_geometries[1].metadata["selection_method"],
            "hardmax_cascade_guided_spectral_merge",
        )


if __name__ == "__main__":
    unittest.main()
