from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.compare_representation_view_cohort import build_view_cohort_report
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class CompareRepresentationViewCohortTests(unittest.TestCase):
    def test_builds_pairwise_view_reports(self) -> None:
        def _make_rep(method: str, directions: np.ndarray) -> ModelRepresentation:
            return ModelRepresentation(
                model_id="model/a",
                architecture_family="toy",
                num_parameters=10,
                hidden_dim=4,
                num_layers=2,
                layer_geometries={
                    1: LayerGeometry(
                        relative_depth=0.5,
                        directions=directions,
                        scales=np.array([2.0, 1.0], dtype=np.float32),
                    )
                },
                concept_profiles={
                    "negation": {
                        "description": "negation probe",
                        "layers": {
                            "1": {
                                "relative_depth": 0.5,
                                "direction": [1.0, 0.0, 0.0, 0.0],
                                "layer_score": 0.8,
                            }
                        },
                    }
                },
                metadata={"extraction_method": method},
            )

        report = build_view_cohort_report(
            [
                _make_rep("activation_probe", np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)),
                _make_rep("weight_svd", np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)),
                _make_rep("jacobian_svd", np.array([[0.99, 0.01, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)),
            ],
            num_layers=1,
        )
        self.assertEqual(len(report["views"]), 3)
        self.assertEqual(len(report["pairwise"]), 3)
        self.assertGreater(report["summary"]["mean_pairwise_subspace_overlap"], 0.3)
        self.assertGreater(report["summary"]["mean_pairwise_concept_alignment"], 0.9)


if __name__ == "__main__":
    unittest.main()
