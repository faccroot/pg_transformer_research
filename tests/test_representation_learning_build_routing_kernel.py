from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.build_routing_kernel import build_routing_kernel
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, VerificationTable


def _make_rep(model_id: str, projections: np.ndarray) -> ModelRepresentation:
    return ModelRepresentation(
        model_id=model_id,
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
        chunk_losses=np.array([0.1, 0.2], dtype=np.float32),
        chunk_ids=["c1", "c2"],
        chunk_layer_projections={1: np.asarray(projections, dtype=np.float32)},
    )


class BuildRoutingKernelTests(unittest.TestCase):
    def test_builds_clustered_rules_from_verified_winners(self) -> None:
        rep_a = _make_rep("m1", np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32))
        rep_b = _make_rep("m2", np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32))
        table = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.6,
                    "verified": True,
                },
                {
                    "probe_id": "p2",
                    "chunk_id": "c2",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.4,
                    "verified": True,
                },
            ],
        )
        kernel = build_routing_kernel(
            [rep_a, rep_b],
            table,
            canonical_dim=2,
            num_layers=1,
            similarity_threshold=0.8,
            min_support=2,
        )
        self.assertEqual(len(kernel.rules), 1)
        rule = kernel.rules[0]
        self.assertEqual(rule["winner_model"], "m1")
        self.assertEqual(rule["support"], 2)
        self.assertEqual(rule["target_layer"], 1)
        self.assertAlmostEqual(rule["mean_verification_confidence"], 0.5, places=6)

    def test_preserves_fallback_rules_when_chunk_projections_are_missing(self) -> None:
        rep_a = ModelRepresentation(
            model_id="m1",
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
            chunk_losses=np.array([0.1, 0.2], dtype=np.float32),
            chunk_ids=["c1", "c2"],
            chunk_layer_projections={},
        )
        rep_b = _make_rep("m2", np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32))
        table = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "confidence_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.6,
                    "verified": True,
                },
                {
                    "probe_id": "p2",
                    "chunk_id": "c2",
                    "probe_type": "confidence_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.4,
                    "verified": True,
                },
            ],
        )
        kernel = build_routing_kernel(
            [rep_a, rep_b],
            table,
            canonical_dim=2,
            num_layers=1,
            similarity_threshold=0.8,
            min_support=2,
        )
        self.assertEqual(len(kernel.rules), 1)
        rule = kernel.rules[0]
        self.assertEqual(rule["routing_mode"], "chunk_lookup_fallback")
        self.assertIsNone(rule["target_layer"])
        self.assertEqual(rule["support"], 2)
        self.assertEqual(rule["winner_model"], "m1")


if __name__ == "__main__":
    unittest.main()
