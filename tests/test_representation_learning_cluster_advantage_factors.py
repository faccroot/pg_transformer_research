from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.cluster_advantage_factors import build_advantage_cluster_report
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class ClusterAdvantageFactorsTests(unittest.TestCase):
    def test_builds_pairwise_advantage_clusters_from_chunk_projections(self) -> None:
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
                    scales=np.array([1.0], dtype=np.float32),
                )
            },
            chunk_losses=np.array([1.0, 2.0], dtype=np.float32),
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={
                1: np.array([[1.0], [0.1]], dtype=np.float32),
            },
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.7,
                    "num_pairs": 2,
                    "layers": {
                        "1": {
                            "relative_depth": 1.0,
                            "direction": [1.0, 0.0, 0.0, 0.0],
                            "layer_score": 0.7,
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
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.0], dtype=np.float32),
                )
            },
            chunk_losses=np.array([1.5, 1.0], dtype=np.float32),
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={
                1: np.array([[0.1], [1.0]], dtype=np.float32),
            },
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.9,
                    "num_pairs": 2,
                    "layers": {
                        "1": {
                            "relative_depth": 1.0,
                            "direction": [1.0, 0.0, 0.0, 0.0],
                            "layer_score": 0.9,
                        }
                    },
                }
            },
        )
        calibration_lookup = {
            "c1": {"chunk_id": "c1", "text": "alpha chunk", "cluster_id": 1},
            "c2": {"chunk_id": "c2", "text": "beta chunk", "cluster_id": 2},
        }
        report = build_advantage_cluster_report(
            [rep_a, rep_b],
            calibration_lookup=calibration_lookup,
            canonical_dim=4,
            num_layers=1,
            similarity_threshold=0.8,
            family_overlap_threshold=0.8,
            chunk_family_similarity_threshold=0.8,
            min_advantage=0.01,
            top_chunks=4,
            top_named_anchors=3,
            max_clusters=16,
        )
        self.assertEqual(report["representations"], ["model/a", "model/b"])
        self.assertEqual(len(report["pairwise_advantage_clusters"]), 1)
        pair = report["pairwise_advantage_clusters"][0]
        self.assertEqual(pair["model_a"], "model/a")
        self.assertEqual(pair["model_b"], "model/b")
        self.assertEqual(len(pair["sides"]), 2)
        a_better = next(side for side in pair["sides"] if side["winner_model"] == "model/a")
        b_better = next(side for side in pair["sides"] if side["winner_model"] == "model/b")
        self.assertEqual(a_better["difference_record_count"], 1)
        self.assertEqual(b_better["difference_record_count"], 1)
        self.assertGreaterEqual(a_better["cluster_count"], 1)
        self.assertGreaterEqual(b_better["cluster_count"], 1)
        self.assertGreaterEqual(a_better["family_count"], 1)
        self.assertGreaterEqual(b_better["family_count"], 1)
        self.assertGreaterEqual(a_better["chunk_family_count"], 1)
        self.assertGreaterEqual(b_better["chunk_family_count"], 1)
        self.assertEqual(a_better["clusters"][0]["top_chunks"][0]["chunk_id"], "c1")
        self.assertEqual(b_better["clusters"][0]["top_chunks"][0]["chunk_id"], "c2")
        self.assertEqual(a_better["families"][0]["top_chunks"][0]["chunk_id"], "c1")
        self.assertEqual(b_better["families"][0]["top_chunks"][0]["chunk_id"], "c2")
        self.assertEqual(a_better["chunk_families"][0]["top_chunks"][0]["chunk_id"], "c1")
        self.assertEqual(b_better["chunk_families"][0]["top_chunks"][0]["chunk_id"], "c2")

    def test_collapses_cross_layer_duplicates_into_one_family(self) -> None:
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
                    scales=np.array([1.0], dtype=np.float32),
                ),
                2: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.0], dtype=np.float32),
                ),
            },
            chunk_losses=np.array([1.0], dtype=np.float32),
            chunk_ids=["c1"],
            chunk_layer_projections={
                1: np.array([[1.0]], dtype=np.float32),
                2: np.array([[1.0]], dtype=np.float32),
            },
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.7,
                    "num_pairs": 1,
                    "layers": {
                        "1": {"relative_depth": 0.5, "direction": [1.0, 0.0, 0.0, 0.0], "layer_score": 0.7},
                        "2": {"relative_depth": 1.0, "direction": [0.0, 1.0, 0.0, 0.0], "layer_score": 0.6},
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
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.0], dtype=np.float32),
                ),
                2: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.0], dtype=np.float32),
                ),
            },
            chunk_losses=np.array([2.0], dtype=np.float32),
            chunk_ids=["c1"],
            chunk_layer_projections={
                1: np.array([[0.1]], dtype=np.float32),
                2: np.array([[0.1]], dtype=np.float32),
            },
            concept_profiles={
                "negation": {
                    "description": "negation probe",
                    "sharpness": 0.9,
                    "num_pairs": 1,
                    "layers": {
                        "1": {"relative_depth": 0.5, "direction": [1.0, 0.0, 0.0, 0.0], "layer_score": 0.9},
                        "2": {"relative_depth": 1.0, "direction": [0.0, 1.0, 0.0, 0.0], "layer_score": 0.8},
                    },
                }
            },
        )
        report = build_advantage_cluster_report(
            [rep_a, rep_b],
            calibration_lookup={"c1": {"chunk_id": "c1", "text": "alpha chunk", "cluster_id": 1}},
            canonical_dim=8,
            num_layers=2,
            similarity_threshold=0.99,
            family_overlap_threshold=0.8,
            chunk_family_similarity_threshold=0.8,
            min_advantage=0.01,
            top_chunks=4,
            top_named_anchors=3,
            max_clusters=16,
        )
        pair = report["pairwise_advantage_clusters"][0]
        a_better = next(side for side in pair["sides"] if side["winner_model"] == "model/a")
        self.assertGreaterEqual(a_better["cluster_count"], 2)
        self.assertEqual(a_better["family_count"], 1)
        self.assertEqual(a_better["chunk_family_count"], 1)
        self.assertEqual(a_better["families"][0]["unique_chunk_count"], 1)
        self.assertAlmostEqual(a_better["families"][0]["coverage_of_winner_chunks"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
