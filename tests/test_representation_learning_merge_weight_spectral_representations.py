from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.merge_weight_spectral_representations import merge_weight_spectral_representations
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class MergeWeightSpectralRepresentationsTests(unittest.TestCase):
    def test_selects_stronger_direction_champions(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([5.0, 1.0], dtype=np.float32),
                )
            },
            metadata={"extraction_method": "weight_svd"},
        )
        rep_b = ModelRepresentation(
            model_id="model/b",
            architecture_family="toy",
            num_parameters=12,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[0.99, 0.01, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
                    scales=np.array([3.0, 4.0], dtype=np.float32),
                )
            },
            metadata={"extraction_method": "weight_svd"},
        )
        geometry = merge_weight_spectral_representations(
            [rep_a, rep_b],
            canonical_dim=4,
            num_layers=1,
            top_k=2,
            similarity_threshold=0.95,
        )
        layer = geometry.layer_geometries[1]
        self.assertEqual(layer.directions.shape, (2, 4))
        self.assertEqual(layer.scales.shape, (2,))
        clusters = layer.metadata["clusters"]
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["champion_model"], "model/a")
        self.assertAlmostEqual(clusters[0]["champion_scale"], 5.0, places=6)
        self.assertEqual(clusters[1]["champion_model"], "model/b")
        self.assertAlmostEqual(clusters[1]["champion_scale"], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
