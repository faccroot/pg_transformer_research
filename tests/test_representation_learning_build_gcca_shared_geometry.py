from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.build_gcca_shared_geometry import build_gcca_shared_geometry
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, SharedLatentGeometry


def _toy_rep(model_id: str, projections: np.ndarray) -> ModelRepresentation:
    return ModelRepresentation(
        model_id=model_id,
        architecture_family="toy",
        num_parameters=128,
        hidden_dim=4,
        num_layers=1,
        layer_geometries={
            1: LayerGeometry(
                relative_depth=1.0,
                directions=np.eye(2, 4, dtype=np.float32),
                scales=np.ones((2,), dtype=np.float32),
            )
        },
        chunk_ids=["c1", "c2", "c3"],
        chunk_losses=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        chunk_layer_projections={1: np.asarray(projections, dtype=np.float32)},
        metadata={"source": "unit"},
    )


class BuildGCCASharedGeometryTests(unittest.TestCase):
    def test_builds_shared_latent_artifact(self) -> None:
        rep_a = _toy_rep(
            "m1",
            np.array(
                [
                    [1.0, 0.0],
                    [0.5, 0.5],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        rep_b = _toy_rep(
            "m2",
            np.array(
                [
                    [0.0, 1.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        artifact = build_gcca_shared_geometry(
            [rep_a, rep_b],
            canonical_dim=4,
            latent_dim=2,
            num_layers=1,
            min_shared_chunks=2,
        )
        self.assertEqual(artifact.source_models, ["m1", "m2"])
        self.assertEqual(artifact.latent_dim, 2)
        self.assertIn(1, artifact.layers)
        layer = artifact.layers[1]
        self.assertEqual(layer.chunk_ids, ["c1", "c2", "c3"])
        self.assertEqual(layer.shared_latents.shape[0], 3)
        self.assertGreaterEqual(layer.shared_latents.shape[1], 1)
        self.assertLessEqual(layer.shared_latents.shape[1], 2)
        self.assertEqual(layer.model_projections["m1"].shape[0], 4)
        self.assertEqual(layer.model_projections["m1"].shape[1], layer.shared_latents.shape[1])
        self.assertEqual(layer.model_means["m2"].shape, (4,))
        self.assertEqual(layer.aligned_latents["m2"].shape, layer.shared_latents.shape)
        self.assertIn("view_residuals", layer.metadata)
        self.assertTrue(np.isfinite(layer.metadata["view_residuals"]["m1"]))

    def test_roundtrips_built_artifact(self) -> None:
        rep_a = _toy_rep("m1", np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
        rep_b = _toy_rep("m2", np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32))
        artifact = build_gcca_shared_geometry(
            [rep_a, rep_b],
            canonical_dim=4,
            latent_dim=2,
            num_layers=1,
            min_shared_chunks=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shared_latent.npz"
            artifact.save(path)
            loaded = SharedLatentGeometry.load(path)
        self.assertEqual(loaded.source_models, ["m1", "m2"])
        np.testing.assert_allclose(loaded.layers[1].shared_latents, artifact.layers[1].shared_latents)
        np.testing.assert_allclose(loaded.layers[1].model_projections["m1"], artifact.layers[1].model_projections["m1"])


if __name__ == "__main__":
    unittest.main()
