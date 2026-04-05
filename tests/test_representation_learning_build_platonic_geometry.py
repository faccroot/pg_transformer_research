from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.build_platonic_geometry import build_platonic_geometry
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class BuildPlatonicGeometryTests(unittest.TestCase):
    def test_builds_shared_geometry_from_multiple_representations(self) -> None:
        rep_a = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(relative_depth=0.5, directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), scales=np.array([1.0], dtype=np.float32)),
            },
            chunk_losses=np.array([1.0, 2.0], dtype=np.float32),
            chunk_ids=["c1", "c2"],
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.7,
                    "num_pairs": 4,
                    "layers": {
                        "1": {
                            "relative_depth": 0.5,
                            "direction": [1.0, 0.0, 0.0, 0.0],
                            "layer_score": 0.6,
                        }
                    },
                }
            },
        )
        rep_b = ModelRepresentation(
            model_id="model/b",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(relative_depth=0.5, directions=np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32), scales=np.array([1.0], dtype=np.float32)),
            },
            chunk_losses=np.array([0.5, 1.5], dtype=np.float32),
            chunk_ids=["c1", "c2"],
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.9,
                    "num_pairs": 4,
                    "layers": {
                        "1": {
                            "relative_depth": 0.5,
                            "direction": [0.0, 1.0, 0.0, 0.0],
                            "layer_score": 0.8,
                        }
                    },
                }
            },
        )
        geometry = build_platonic_geometry([rep_a, rep_b], canonical_dim=4, num_layers=1, top_k=1)
        self.assertEqual(geometry.canonical_dim, 4)
        self.assertEqual(geometry.source_models, ["model/a", "model/b"])
        self.assertIn(1, geometry.layer_geometries)
        self.assertEqual(geometry.layer_geometries[1].directions.shape, (1, 4))
        np.testing.assert_allclose(geometry.frontier_floor, np.array([0.75, 1.75], dtype=np.float32))
        self.assertEqual(geometry.concept_profiles["negation"]["best_model"], "model/b")
        self.assertEqual(geometry.concept_profiles["negation"]["layers"]["1"]["model_id"], "model/b")


if __name__ == "__main__":
    unittest.main()
