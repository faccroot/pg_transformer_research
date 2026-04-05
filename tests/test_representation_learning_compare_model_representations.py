from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.compare_model_representations import build_comparison_report
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class CompareModelRepresentationsTests(unittest.TestCase):
    def test_builds_pairwise_overlap_and_chunk_loss_report(self) -> None:
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
                    scales=np.array([2.0], dtype=np.float32),
                    covariance=np.eye(4, dtype=np.float32),
                    importance=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                ),
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
                            "layer_score": 0.65,
                        }
                    },
                }
            },
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
                    directions=np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.5], dtype=np.float32),
                    covariance=np.eye(4, dtype=np.float32) * 2.0,
                    importance=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
                ),
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
                            "direction": [0.9, 0.1, 0.0, 0.0],
                            "layer_score": 0.8,
                        }
                    },
                }
            },
        )
        report = build_comparison_report([rep_a, rep_b], canonical_dim=4, num_layers=1)
        self.assertEqual(report["best_mean_chunk_loss_model"], "model/b")
        self.assertEqual(len(report["layers"]), 1)
        layer = report["layers"][0]
        self.assertEqual(layer["best_scale_energy_model"], "model/a")
        self.assertEqual(len(layer["pairwise_subspace_overlap"]), 1)
        overlap = layer["pairwise_subspace_overlap"][0]["subspace_overlap"]
        self.assertGreater(overlap, 0.9)
        pairwise = report["pairwise_summary"][0]
        self.assertEqual(pairwise["shared_chunk_count"], 2)
        self.assertAlmostEqual(pairwise["shared_chunk_loss_mae"], 0.5, places=6)
        self.assertIn("pairwise_advantages", report)
        self.assertEqual(len(report["pairwise_advantages"]), 1)
        advantage = report["pairwise_advantages"][0]
        self.assertEqual(advantage["shared_chunk_count"], 2)
        self.assertEqual(advantage["model_b_better"]["win_count"], 2)
        self.assertEqual(advantage["model_a_better"]["win_count"], 0)
        self.assertEqual(report["concepts"]["negation"]["best_model"], "model/b")
        self.assertEqual(report["concepts"]["negation"]["layers"][0]["best_model"], "model/b")
        self.assertIn("latent_factors", report)
        self.assertGreaterEqual(len(report["latent_factors"]), 1)
        first_factor = report["latent_factors"][0]
        self.assertIn(first_factor["best_model"], {"model/a", "model/b"})
        self.assertEqual(first_factor["top_named_anchors"][0]["concept"], "negation")
        self.assertIn("residual_latent_factors", report)
        self.assertIsInstance(report["residual_latent_factors"], list)


if __name__ == "__main__":
    unittest.main()
