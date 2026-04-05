from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.build_ecology_training_data import FEATURE_NAMES, build_ecology_training_data
from tools.representation_learning.schemas import ForwardSignatureDataset, LayerGeometry, ModelRepresentation, VerificationTable


def _make_rep(
    model_id: str,
    *,
    losses: list[float],
    chunk_ids: list[str],
    directions: np.ndarray,
    projections: np.ndarray | None,
    sharpness: float,
    hidden_dim: int = 4,
) -> ModelRepresentation:
    return ModelRepresentation(
        model_id=model_id,
        architecture_family="toy",
        num_parameters=128,
        hidden_dim=hidden_dim,
        num_layers=1,
        layer_geometries={
            1: LayerGeometry(
                relative_depth=1.0,
                directions=np.asarray(directions, dtype=np.float32),
                scales=np.ones(directions.shape[0], dtype=np.float32),
            )
        },
        chunk_losses=np.asarray(losses, dtype=np.float32),
        chunk_ids=list(chunk_ids),
        chunk_layer_projections={} if projections is None else {1: np.asarray(projections, dtype=np.float32)},
        concept_profiles={"probe": {"sharpness": sharpness}},
    )


class BuildEcologyTrainingDataTests(unittest.TestCase):
    def test_builds_event_and_example_artifacts(self) -> None:
        rep_a = _make_rep(
            "m1",
            losses=[0.2, 0.4],
            chunk_ids=["c1", "c2"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            sharpness=0.8,
        )
        rep_b = _make_rep(
            "m2",
            losses=[0.5, 0.3],
            chunk_ids=["c1", "c2"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32),
            sharpness=0.4,
        )
        verification = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.7,
                    "verified": True,
                }
            ],
        )
        event_dataset, training_set = build_ecology_training_data(
            [rep_a, rep_b],
            verification,
            canonical_dim=4,
            num_layers=1,
        )
        self.assertEqual(event_dataset.source_models, ["m1", "m2"])
        self.assertEqual(event_dataset.feature_names, FEATURE_NAMES)
        self.assertEqual(len(event_dataset.events), 2)
        self.assertEqual(len(training_set.examples), 1)
        self.assertEqual(training_set.embedding_dim, 4)
        example = training_set.examples[0]
        self.assertEqual(example["winner_model"], "m1")
        self.assertEqual(example["candidate_model_ids"], ["m1", "m2"])
        self.assertEqual(len(example["candidate_event_ids"]), 2)
        self.assertEqual(len(example["candidates"]), 2)
        event = event_dataset.events[0]
        self.assertIn("loss", event["features"])
        self.assertIn("projection_norm", event["features"])
        self.assertIn("factor_1_proj", event["features"])
        self.assertIn("factor_1_share", event["features"])
        self.assertIn("num_disagreement_factors", training_set.metadata)
        self.assertEqual(len(event["embedding"]), 4)

    def test_marks_missing_chunk_projections_in_features(self) -> None:
        rep_a = _make_rep(
            "m1",
            losses=[0.2],
            chunk_ids=["c1"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=None,
            sharpness=0.9,
        )
        rep_b = _make_rep(
            "m2",
            losses=[0.3],
            chunk_ids=["c1"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[0.0, 1.0]], dtype=np.float32),
            sharpness=0.5,
        )
        verification = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "confidence_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.5,
                    "verified": True,
                }
            ],
        )
        event_dataset, training_set = build_ecology_training_data(
            [rep_a, rep_b],
            verification,
            canonical_dim=4,
            num_layers=1,
        )
        by_model = {event["model_id"]: event for event in event_dataset.events}
        self.assertEqual(by_model["m1"]["features"]["has_projection"], 0.0)
        self.assertEqual(by_model["m1"]["embedding"], [])
        self.assertEqual(by_model["m2"]["features"]["has_projection"], 1.0)
        self.assertEqual(training_set.examples[0]["winner_model"], "m1")

    def test_incorporates_forward_signature_features_when_available(self) -> None:
        rep_a = _make_rep(
            "m1",
            losses=[0.2],
            chunk_ids=["c1"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[1.0, 0.0]], dtype=np.float32),
            sharpness=0.8,
        )
        rep_b = _make_rep(
            "m2",
            losses=[0.3],
            chunk_ids=["c1"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[0.0, 1.0]], dtype=np.float32),
            sharpness=0.4,
        )
        forward_a = ForwardSignatureDataset(
            model_id="m1",
            chunk_ids=["c1"],
            top_k=2,
            global_features={
                "last_token_entropy": np.array([1.2], dtype=np.float32),
                "sequence_mean_entropy": np.array([1.0], dtype=np.float32),
                "last_token_top1_prob": np.array([0.7], dtype=np.float32),
                "last_token_margin": np.array([0.2], dtype=np.float32),
                "last_token_topk_mass": np.array([0.9], dtype=np.float32),
            },
            layer_features={1: {"attention_entropy": np.array([0.4], dtype=np.float32), "attention_peak_frac": np.array([0.6], dtype=np.float32)}},
            topk_token_ids=np.array([[10, 11]], dtype=np.int32),
            topk_token_probs=np.array([[0.7, 0.2]], dtype=np.float32),
        )
        forward_b = ForwardSignatureDataset(
            model_id="m2",
            chunk_ids=["c1"],
            top_k=2,
            global_features={
                "last_token_entropy": np.array([1.5], dtype=np.float32),
                "sequence_mean_entropy": np.array([1.3], dtype=np.float32),
                "last_token_top1_prob": np.array([0.5], dtype=np.float32),
                "last_token_margin": np.array([0.1], dtype=np.float32),
                "last_token_topk_mass": np.array([0.75], dtype=np.float32),
            },
            layer_features={1: {"attention_entropy": np.array([0.7], dtype=np.float32), "attention_peak_frac": np.array([0.4], dtype=np.float32)}},
            topk_token_ids=np.array([[10, 12]], dtype=np.int32),
            topk_token_probs=np.array([[0.5, 0.25]], dtype=np.float32),
        )
        verification = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.8,
                    "verified": True,
                }
            ],
        )
        event_dataset, training_set = build_ecology_training_data(
            [rep_a, rep_b],
            verification,
            canonical_dim=4,
            num_layers=1,
            forward_signatures=[forward_a, forward_b],
        )
        by_model = {event["model_id"]: event for event in event_dataset.events}
        self.assertIn("attention_entropy", event_dataset.feature_names)
        self.assertEqual(by_model["m1"]["features"]["has_forward_signature"], 1.0)
        self.assertAlmostEqual(by_model["m1"]["features"]["attention_entropy"], 0.4, places=6)
        self.assertAlmostEqual(by_model["m1"]["features"]["cross_model_topk_jaccard_mean"], 1.0 / 3.0, places=6)
        self.assertGreater(by_model["m1"]["features"]["cross_model_topk_prob_l1_mean"], 0.0)
        self.assertEqual(training_set.metadata["forward_signature_models"], ["m1", "m2"])

    def test_adds_finite_disagreement_factor_features(self) -> None:
        rep_a = _make_rep(
            "m1",
            losses=[0.2, 0.6],
            chunk_ids=["c1", "c2"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
            sharpness=0.9,
        )
        rep_b = _make_rep(
            "m2",
            losses=[0.5, 0.1],
            chunk_ids=["c1", "c2"],
            directions=np.eye(2, 4, dtype=np.float32),
            projections=np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32),
            sharpness=0.3,
        )
        verification = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m1",
                    "verification_confidence": 0.8,
                    "verified": True,
                },
                {
                    "probe_id": "p2",
                    "chunk_id": "c2",
                    "probe_type": "directional_divergence",
                    "verified_winner_model": "m2",
                    "verification_confidence": 0.8,
                    "verified": True,
                },
            ],
        )
        event_dataset, training_set = build_ecology_training_data(
            [rep_a, rep_b],
            verification,
            canonical_dim=4,
            num_layers=1,
            num_disagreement_factors=3,
        )
        self.assertIn("factor_1_proj", event_dataset.feature_names)
        self.assertIn("factor_3_next_align", event_dataset.feature_names)
        self.assertEqual(training_set.metadata["num_disagreement_factors"], 3)
        factor_metadata = training_set.metadata["disagreement_factor_metadata"]
        self.assertGreaterEqual(factor_metadata["1"]["sample_count"], 2)
        for event in event_dataset.events:
            for key in ("factor_1_proj", "factor_1_abs", "factor_1_share", "factor_1_prev_align", "factor_1_next_align"):
                self.assertTrue(np.isfinite(event["features"][key]))


if __name__ == "__main__":
    unittest.main()
