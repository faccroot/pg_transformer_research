from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.compare_representation_views import build_view_comparison_report
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class CompareRepresentationViewsTests(unittest.TestCase):
    def test_compares_activation_and_weight_views(self) -> None:
        activation_rep = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([2.0, 1.0], dtype=np.float32),
                )
            },
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.7,
                    "layers": {
                        "1": {
                            "relative_depth": 0.5,
                            "direction": [1.0, 0.0, 0.0, 0.0],
                            "layer_score": 0.6,
                        }
                    },
                }
            },
            metadata={"extraction_method": "activation_probe"},
        )
        weight_rep = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
                    scales=np.array([5.0, 3.0], dtype=np.float32),
                )
            },
            metadata={"extraction_method": "weight_svd"},
        )
        report = build_view_comparison_report(activation_rep, weight_rep, num_layers=1)
        self.assertEqual(report["primary_extraction_method"], "activation_probe")
        self.assertEqual(report["secondary_extraction_method"], "weight_svd")
        self.assertEqual(len(report["layers"]), 1)
        self.assertGreater(report["summary"]["mean_subspace_overlap"], 0.4)
        self.assertGreater(report["summary"]["mean_concept_alignment"], 0.99)
        concept = report["concepts"]["negation"]
        self.assertEqual(concept["layers"][0]["best_direction_idx"], 0)
        self.assertAlmostEqual(concept["layers"][0]["best_direction_scale"], 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
