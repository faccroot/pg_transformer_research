from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.merge_gauge_fixed_spectral_representations import activation_gauge_rotation, merge_gauge_fixed_spectral_representations
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class MergeGaugeFixedSpectralRepresentationsTests(unittest.TestCase):
    def test_activation_gauge_rotation_recovers_simple_rotation(self) -> None:
        anchor = ModelRepresentation(
            model_id="anchor",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=2,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    scales=np.array([1.0, 1.0], dtype=np.float32),
                )
            },
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={1: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)},
        )
        other = ModelRepresentation(
            model_id="other",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=2,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.0, 1.0], dtype=np.float32),
                )
            },
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={1: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)},
        )
        rotation = activation_gauge_rotation(anchor, other, target_layer=1, canonical_dim=2, num_layers=1)
        self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(2, dtype=np.float32), atol=1e-5))
        self.assertTrue(np.allclose(np.abs(rotation), np.eye(2, dtype=np.float32), atol=1e-5))

    def test_gauge_fixed_merge_preserves_anchor_champion_after_alignment(self) -> None:
        anchor = ModelRepresentation(
            model_id="anchor",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=2,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0]], dtype=np.float32),
                    scales=np.array([5.0], dtype=np.float32),
                )
            },
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={1: np.array([[1.0], [0.0]], dtype=np.float32)},
        )
        other = ModelRepresentation(
            model_id="other",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=2,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[0.0, 1.0]], dtype=np.float32),
                    scales=np.array([4.0], dtype=np.float32),
                )
            },
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={1: np.array([[0.0], [1.0]], dtype=np.float32)},
        )
        geometry = merge_gauge_fixed_spectral_representations(
            [anchor, other],
            canonical_dim=2,
            num_layers=1,
            top_k=1,
            similarity_threshold=0.9,
            incremental=False,
        )
        cluster = geometry.layer_geometries[1].metadata["clusters"][0]
        self.assertEqual(cluster["champion_model"], "anchor")


if __name__ == "__main__":
    unittest.main()
