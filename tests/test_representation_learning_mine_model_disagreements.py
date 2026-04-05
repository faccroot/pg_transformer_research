from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.mine_model_disagreements import mine_pairwise_disagreements
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


def _make_rep(model_id: str, losses: list[float]) -> ModelRepresentation:
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
        chunk_losses=np.asarray(losses, dtype=np.float32),
        chunk_ids=["c1", "c2", "c3"],
        chunk_layer_projections={1: np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)},
    )


class MineModelDisagreementsTests(unittest.TestCase):
    def test_mines_typed_pairwise_probes(self) -> None:
        rep_a = _make_rep("model/a", [0.1, 0.3, 2.5])
        rep_b = _make_rep("model/b", [0.9, 0.35, 3.5])
        probe_set = mine_pairwise_disagreements(
            [rep_a, rep_b],
            calibration_lookup={"c1": {"text": "alpha"}, "c3": {"text": "omega"}},
            top_k_per_pair=8,
            min_disagreement_score=0.05,
            easy_loss_quantile=0.34,
            hard_loss_quantile=0.8,
            confidence_divergence_threshold=0.5,
        )
        self.assertEqual(probe_set.source_models, ["model/a", "model/b"])
        probe_types = {probe["chunk_id"]: probe["probe_type"] for probe in probe_set.probes}
        self.assertEqual(probe_types["c1"], "confidence_divergence")
        self.assertEqual(probe_types["c3"], "joint_uncertainty")
        self.assertEqual(probe_set.probes[0]["text_preview"], "omega")


if __name__ == "__main__":
    unittest.main()
