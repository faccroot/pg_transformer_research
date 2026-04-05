from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.schemas import EcologyTrainingSet
from tools.representation_learning.train_ecology_model import (
    FORWARD_SIGNATURE_FEATURES,
    _feature_selection,
    train_one_mode,
)


def _example(example_id: str, chunk_id: str, winner_model: str, projection_a: float, projection_b: float) -> dict:
    return {
        "example_id": example_id,
        "probe_id": example_id,
        "chunk_id": chunk_id,
        "probe_type": "directional_divergence" if winner_model == "m1" else "joint_uncertainty",
        "target_layer": 1,
        "winner_model": winner_model,
        "verification_confidence": 0.8,
        "candidate_model_ids": ["m1", "m2"],
        "candidate_event_ids": [f"{example_id}::m1", f"{example_id}::m2"],
        "candidates": [
            {
                "event_id": f"{example_id}::m1",
                "model_id": "m1",
                "features": {
                    "loss": 0.4 if winner_model == "m1" else 0.6,
                    "projection_norm": projection_a,
                    "relative_depth": 1.0,
                    "has_projection": 1.0,
                    "concept_sharpness_max": projection_a,
                },
                "embedding": [projection_a, 0.0],
                "source_layer_idx": 1,
            },
            {
                "event_id": f"{example_id}::m2",
                "model_id": "m2",
                "features": {
                    "loss": 0.4 if winner_model == "m2" else 0.6,
                    "projection_norm": projection_b,
                    "relative_depth": 1.0,
                    "has_projection": 1.0,
                    "concept_sharpness_max": projection_b,
                },
                "embedding": [projection_b, 0.0],
                "source_layer_idx": 1,
            },
        ],
        "context_features": {
            "mean_loss": 0.5,
            "disagreement_score": 0.2,
            "num_models_present": 2.0,
        },
    }


class TrainEcologyModelTests(unittest.TestCase):
    def test_feature_modes_split_geometry_and_forward_signatures(self) -> None:
        feature_names = [
            "loss",
            "projection_norm",
            "relative_depth",
            "has_projection",
            "concept_sharpness_max",
            "attention_entropy",
            "cross_model_topk_jaccard_mean",
            "factor_1_proj",
            "factor_1_share",
        ]
        geometry_selected, geometry_use_embeddings = _feature_selection(feature_names, "geometry_only")
        forward_selected, forward_use_embeddings = _feature_selection(feature_names, "forward_only")
        structure_selected, structure_use_embeddings = _feature_selection(feature_names, "structure_no_embeddings")
        factor_selected, factor_use_embeddings = _feature_selection(feature_names, "factor_only")
        self.assertTrue(geometry_use_embeddings)
        self.assertFalse(forward_use_embeddings)
        self.assertFalse(structure_use_embeddings)
        self.assertFalse(factor_use_embeddings)
        self.assertIn("projection_norm", geometry_selected)
        self.assertNotIn("attention_entropy", geometry_selected)
        self.assertEqual(set(forward_selected), FORWARD_SIGNATURE_FEATURES.intersection(feature_names))
        self.assertIn("attention_entropy", structure_selected)
        self.assertIn("projection_norm", structure_selected)
        self.assertEqual(set(factor_selected), {"factor_1_proj", "factor_1_share"})

    def test_train_one_mode_runs_and_writes_checkpoint(self) -> None:
        examples = []
        for idx in range(12):
            if idx % 2 == 0:
                examples.append(_example(f"ex{idx}", f"chunk{idx}", "m1", 0.9, 0.2))
            else:
                examples.append(_example(f"ex{idx}", f"chunk{idx}", "m2", 0.2, 0.9))
        training_set = EcologyTrainingSet(
            source_models=["m1", "m2"],
            feature_names=[
                "loss",
                "projection_norm",
                "relative_depth",
                "has_projection",
                "concept_sharpness_max",
            ],
            embedding_dim=2,
            examples=examples,
            metadata={"stage": "unit"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_one_mode(
                training_set,
                mode="structure_only",
                output_dir=Path(tmpdir),
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                batch_size=4,
                epochs=3,
                learning_rate=2e-3,
                val_fraction=0.25,
                seed=7,
            )
            self.assertIn("val_metrics", report)
            self.assertGreater(report["train_examples"], 0)
            self.assertGreater(report["val_examples"], 0)
            self.assertTrue(Path(report["checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
