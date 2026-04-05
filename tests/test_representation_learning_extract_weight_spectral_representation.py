from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.extract_weight_spectral_representation import (
    WeightMatrixSpec,
    build_weight_spectral_representation_from_layer_weights,
)


class ExtractWeightSpectralRepresentationTests(unittest.TestCase):
    def test_builds_hidden_space_weight_representation(self) -> None:
        representation = build_weight_spectral_representation_from_layer_weights(
            model_id="model/weight",
            architecture_family="toy",
            num_parameters=42,
            hidden_dim=4,
            num_layers=2,
            layer_weights={
                1: [
                    WeightMatrixSpec(
                        module_name="mlp.up_proj",
                        weight=np.array(
                            [
                                [3.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                        side="input",
                    ),
                    WeightMatrixSpec(
                        module_name="mlp.down_proj",
                        weight=np.array(
                            [
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [4.0, 0.0],
                                [0.0, 2.0],
                            ],
                            dtype=np.float32,
                        ),
                        side="output",
                    ),
                ]
            },
            top_k=3,
            top_k_per_module=2,
        )
        self.assertEqual(representation.metadata["extraction_method"], "weight_svd")
        self.assertIn(1, representation.layer_geometries)
        layer = representation.layer_geometries[1]
        self.assertEqual(layer.directions.shape, (3, 4))
        self.assertEqual(layer.scales.shape, (3,))
        self.assertTrue(np.all(layer.scales[:-1] >= layer.scales[1:]))
        self.assertTrue(np.isfinite(layer.covariance).all())
        self.assertTrue(np.isfinite(layer.coactivation).all())
        self.assertEqual(layer.metadata["direction_sources"][0]["module_name"], "mlp.down_proj")
        self.assertEqual(layer.metadata["direction_sources"][0]["singular_value"], 4.0)


if __name__ == "__main__":
    unittest.main()
