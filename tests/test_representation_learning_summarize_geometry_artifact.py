from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.schemas import LayerGeometry, PlatonicGeometry
from tools.representation_learning.summarize_geometry_artifact import summarize_geometry


class SummarizeGeometryArtifactTests(unittest.TestCase):
    def test_summarize_geometry_with_cluster_metadata(self) -> None:
        geometry = PlatonicGeometry(
            canonical_dim=4,
            source_models=["a", "b"],
            metadata={"selection_method": "hybrid_cascade_guided_spectral_merge"},
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([2.0], dtype=np.float32),
                    metadata={
                        "clusters": [
                            {
                                "champion_model": "a",
                                "champion_score": 3.0,
                                "support": 2,
                            },
                            {
                                "champion_model": "b",
                                "champion_score": 1.0,
                                "support": 1,
                            },
                        ]
                    },
                )
            },
        )
        summary = summarize_geometry(geometry)
        self.assertEqual(summary["selection_method"], "hybrid_cascade_guided_spectral_merge")
        self.assertEqual(summary["ownership_by_model"]["a"]["count"], 1)
        self.assertEqual(summary["ownership_by_model"]["a"]["total_scale"], 3.0)
        self.assertEqual(summary["total_shared_cluster_count"], 1)
        self.assertEqual(summary["total_unique_cluster_count"], 1)

    def test_summarize_geometry_with_contribution_metadata(self) -> None:
        geometry = PlatonicGeometry(
            canonical_dim=4,
            source_models=["a", "b"],
            metadata={"selection_method": "cascade_sparse_residual_merge"},
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    metadata={
                        "contribution_by_model": {
                            "a": {"accepted_direction_count": 2, "total_effective_scale": 5.0},
                            "b": {"accepted_direction_count": 1, "total_effective_scale": 2.0},
                        }
                    },
                )
            },
        )
        summary = summarize_geometry(geometry)
        self.assertEqual(summary["total_contribution_like_entries"], 3)
        self.assertIn("contribution_by_model", summary["per_layer"]["1"])


if __name__ == "__main__":
    unittest.main()
