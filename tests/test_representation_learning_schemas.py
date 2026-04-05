from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.schemas import (
    ActivationEventDataset,
    DisagreementProbeSet,
    EcologyTrainingSet,
    ForwardSignatureDataset,
    KernelTeacherDataset,
    KernelTeacherTextDataset,
    LayerGeometry,
    ModelRepresentation,
    PlatonicGeometry,
    RoutingKernel,
    SharedLatentGeometry,
    SharedLatentLayer,
    VerificationTable,
)


class RepresentationLearningSchemaTests(unittest.TestCase):
    def test_model_representation_roundtrip(self) -> None:
        representation = ModelRepresentation(
            model_id="test/model",
            architecture_family="toy",
            num_parameters=123,
            hidden_dim=4,
            num_layers=2,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([2.0], dtype=np.float32),
                    covariance=np.eye(1, dtype=np.float32),
                    coactivation=np.eye(1, dtype=np.float32),
                    importance=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    metadata={"probe": "unit"},
                )
            },
            chunk_losses=np.array([1.25, 0.75], dtype=np.float32),
            chunk_ids=["chunk_a", "chunk_b"],
            chunk_layer_projections={
                1: np.array([[0.5], [0.25]], dtype=np.float32),
            },
            concept_profiles={
                "negation": {
                    "sharpness": 0.91,
                    "layers": {
                        "1": {
                            "relative_depth": 0.5,
                            "direction": [1.0, 0.0, 0.0, 0.0],
                            "layer_score": 0.75,
                        }
                    },
                }
            },
            metadata={"note": "roundtrip"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "representation.npz"
            representation.save(path)
            loaded = ModelRepresentation.load(path)
        self.assertEqual(loaded.model_id, "test/model")
        self.assertEqual(loaded.architecture_family, "toy")
        np.testing.assert_allclose(loaded.chunk_losses, np.array([1.25, 0.75], dtype=np.float32))
        self.assertEqual(loaded.chunk_ids, ["chunk_a", "chunk_b"])
        np.testing.assert_allclose(
            loaded.chunk_layer_projections[1],
            np.array([[0.5], [0.25]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            loaded.layer_geometries[1].directions,
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(loaded.layer_geometries[1].metadata["probe"], "unit")
        self.assertAlmostEqual(loaded.concept_profiles["negation"]["sharpness"], 0.91, places=6)
        self.assertEqual(loaded.concept_profiles["negation"]["layers"]["1"]["direction"], [1.0, 0.0, 0.0, 0.0])

    def test_platonic_geometry_roundtrip(self) -> None:
        geometry = PlatonicGeometry(
            canonical_dim=4,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    scales=np.array([1.5], dtype=np.float32),
                    coactivation=np.eye(1, dtype=np.float32),
                )
            },
            source_models=["m1", "m2"],
            frontier_floor=np.array([0.9, 1.1], dtype=np.float32),
            concept_profiles={
                "conditionality": {
                    "best_model": "m2",
                    "layers": {
                        "1": {
                            "model_id": "m2",
                            "direction": [0.0, 1.0, 0.0, 0.0],
                            "layer_score": 0.66,
                        }
                    },
                }
            },
            metadata={"version": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "geometry.npz"
            geometry.save(path)
            loaded = PlatonicGeometry.load(path)
        self.assertEqual(loaded.canonical_dim, 4)
        self.assertEqual(loaded.source_models, ["m1", "m2"])
        np.testing.assert_allclose(loaded.frontier_floor, np.array([0.9, 1.1], dtype=np.float32))
        np.testing.assert_allclose(
            loaded.layer_geometries[1].directions,
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(loaded.concept_profiles["conditionality"]["best_model"], "m2")

    def test_probe_related_artifact_roundtrip(self) -> None:
        probe_set = DisagreementProbeSet(
            source_models=["m1", "m2"],
            probes=[
                {
                    "probe_id": "p1",
                    "chunk_id": "chunk_a",
                    "probe_type": "directional_divergence",
                    "winner_model": "m1",
                    "loss_by_model": {"m1": 0.4, "m2": 0.8},
                }
            ],
            metadata={"version": 1},
        )
        verification = VerificationTable(
            source_models=["m1", "m2"],
            entries=[
                {
                    "probe_id": "p1",
                    "verified_winner_model": "m1",
                    "verified": True,
                    "verification_confidence": 0.5,
                }
            ],
            metadata={"method": "heldout_chunk_loss"},
        )
        kernel = RoutingKernel(
            source_models=["m1", "m2"],
            rules=[
                {
                    "rule_id": "r1",
                    "winner_model": "m1",
                    "probe_type": "directional_divergence",
                    "target_layer": 1,
                    "centroid": [1.0, 0.0, 0.0, 0.0],
                }
            ],
            metadata={"canonical_dim": 4},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            probe_path = tmp / "probe_set.npz"
            verification_path = tmp / "verification.npz"
            kernel_path = tmp / "kernel.npz"
            probe_set.save(probe_path)
            verification.save(verification_path)
            kernel.save(kernel_path)
            loaded_probe_set = DisagreementProbeSet.load(probe_path)
            loaded_verification = VerificationTable.load(verification_path)
            loaded_kernel = RoutingKernel.load(kernel_path)
        self.assertEqual(loaded_probe_set.probes[0]["winner_model"], "m1")
        self.assertTrue(loaded_verification.entries[0]["verified"])
        self.assertEqual(loaded_kernel.rules[0]["rule_id"], "r1")

    def test_ecology_related_artifact_roundtrip(self) -> None:
        event_dataset = ActivationEventDataset(
            source_models=["m1", "m2"],
            feature_names=["loss", "projection_norm"],
            embedding_dim=4,
            events=[
                {
                    "event_id": "e1",
                    "chunk_id": "c1",
                    "model_id": "m1",
                    "target_layer": 1,
                    "features": {"loss": 0.1, "projection_norm": 0.9},
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                }
            ],
            metadata={"stage": "bootstrap"},
        )
        training_set = EcologyTrainingSet(
            source_models=["m1", "m2"],
            feature_names=["loss", "projection_norm"],
            embedding_dim=4,
            examples=[
                {
                    "example_id": "x1",
                    "candidate_event_ids": ["e1"],
                    "winner_model": "m1",
                    "probe_type": "directional_divergence",
                }
            ],
            metadata={"split": "train"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_path = tmp / "events.npz"
            training_path = tmp / "training.npz"
            event_dataset.save(event_path)
            training_set.save(training_path)
            loaded_events = ActivationEventDataset.load(event_path)
            loaded_training = EcologyTrainingSet.load(training_path)
        self.assertEqual(loaded_events.feature_names, ["loss", "projection_norm"])
        self.assertEqual(loaded_events.events[0]["event_id"], "e1")
        self.assertEqual(loaded_training.examples[0]["winner_model"], "m1")
        self.assertEqual(loaded_training.embedding_dim, 4)

    def test_kernel_teacher_related_artifact_roundtrip(self) -> None:
        teacher_dataset = KernelTeacherDataset(
            source_models=["m1", "m2"],
            embedding_dim=2,
            examples=[
                {
                    "example_id": "k1",
                    "chunk_id": "chunk_1",
                    "candidate_model_ids": ["m1", "m2"],
                    "candidate_weights": [0.8, 0.2],
                    "cleared_embedding": [1.0, 0.0],
                }
            ],
            metadata={"stage": "teacher"},
        )
        teacher_text_dataset = KernelTeacherTextDataset(
            source_models=["m1", "m2"],
            embedding_dim=2,
            examples=[
                {
                    "example_id": "k1",
                    "chunk_id": "chunk_1",
                    "text": "hello world",
                    "cleared_embedding": [1.0, 0.0],
                }
            ],
            metadata={"stage": "teacher_text"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            teacher_path = tmp / "teacher.npz"
            teacher_text_path = tmp / "teacher_text.npz"
            teacher_dataset.save(teacher_path)
            teacher_text_dataset.save(teacher_text_path)
            loaded_teacher = KernelTeacherDataset.load(teacher_path)
            loaded_teacher_text = KernelTeacherTextDataset.load(teacher_text_path)
        self.assertEqual(loaded_teacher.embedding_dim, 2)
        self.assertEqual(loaded_teacher.examples[0]["chunk_id"], "chunk_1")
        self.assertEqual(loaded_teacher_text.examples[0]["text"], "hello world")

    def test_forward_signature_dataset_roundtrip(self) -> None:
        dataset = ForwardSignatureDataset(
            model_id="m1",
            chunk_ids=["c1", "c2"],
            top_k=3,
            global_features={
                "last_token_entropy": np.array([1.0, 2.0], dtype=np.float32),
                "last_token_top1_prob": np.array([0.9, 0.8], dtype=np.float32),
            },
            layer_features={
                1: {
                    "attention_entropy": np.array([0.2, 0.3], dtype=np.float32),
                    "attention_peak_frac": np.array([0.7, 0.6], dtype=np.float32),
                }
            },
            topk_token_ids=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            topk_token_probs=np.array([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]], dtype=np.float32),
            metadata={"stage": "forward"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "forward_signatures.npz"
            dataset.save(path)
            loaded = ForwardSignatureDataset.load(path)
        self.assertEqual(loaded.model_id, "m1")
        self.assertEqual(loaded.chunk_ids, ["c1", "c2"])
        self.assertEqual(loaded.top_k, 3)
        np.testing.assert_allclose(loaded.global_features["last_token_entropy"], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(loaded.layer_features[1]["attention_entropy"], np.array([0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(loaded.topk_token_ids, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    def test_shared_latent_geometry_roundtrip(self) -> None:
        artifact = SharedLatentGeometry(
            latent_dim=2,
            input_dim=4,
            source_models=["m1", "m2"],
            layers={
                1: SharedLatentLayer(
                    relative_depth=0.5,
                    chunk_ids=["c1", "c2"],
                    shared_latents=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
                    model_projections={
                        "m1": np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.2], [0.0, 0.0]], dtype=np.float32),
                        "m2": np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.1], [0.0, 0.0]], dtype=np.float32),
                    },
                    model_means={
                        "m1": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                        "m2": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
                    },
                    aligned_latents={
                        "m1": np.array([[0.9, 0.1], [0.45, 0.55]], dtype=np.float32),
                        "m2": np.array([[0.1, 0.9], [0.55, 0.45]], dtype=np.float32),
                    },
                    metadata={"method": "gcca"},
                )
            },
            metadata={"version": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shared_latent_geometry.npz"
            artifact.save(path)
            loaded = SharedLatentGeometry.load(path)
        self.assertEqual(loaded.latent_dim, 2)
        self.assertEqual(loaded.input_dim, 4)
        self.assertEqual(loaded.source_models, ["m1", "m2"])
        self.assertEqual(loaded.layers[1].chunk_ids, ["c1", "c2"])
        np.testing.assert_allclose(loaded.layers[1].shared_latents, np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
        np.testing.assert_allclose(loaded.layers[1].model_means["m1"], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        np.testing.assert_allclose(loaded.layers[1].aligned_latents["m2"], np.array([[0.1, 0.9], [0.55, 0.45]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
