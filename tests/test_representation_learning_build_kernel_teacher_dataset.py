from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.build_kernel_teacher_dataset import build_kernel_teacher_dataset
from tools.representation_learning.schemas import EcologyTrainingSet, KernelTeacherDataset
from tools.representation_learning.train_ecology_model import train_one_mode


def _toy_example(index: int) -> dict[str, object]:
    winner = "m1" if index % 2 == 0 else "m2"
    loser = "m2" if winner == "m1" else "m1"
    winner_signal = 2.0 + 0.1 * index
    loser_signal = -2.0 - 0.1 * index
    return {
        "example_id": f"ex{index}",
        "chunk_id": f"chunk_{index:02d}",
        "probe_id": f"probe_{index:02d}",
        "probe_type": "directional_divergence",
        "winner_model": winner,
        "verification_confidence": 1.0,
        "target_layer": 1,
        "context_features": {},
        "candidates": [
            {
                "model_id": winner,
                "features": {"signal": winner_signal},
                "embedding": [1.0, 0.0] if winner == "m1" else [0.0, 1.0],
            },
            {
                "model_id": loser,
                "features": {"signal": loser_signal},
                "embedding": [1.0, 0.0] if loser == "m1" else [0.0, 1.0],
            },
        ],
    }


class BuildKernelTeacherDatasetTests(unittest.TestCase):
    def test_exports_teacher_weights_and_cleared_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training_set_path = root / "toy_training_set.npz"
            training_set = EcologyTrainingSet(
                source_models=["m1", "m2"],
                feature_names=["signal"],
                embedding_dim=2,
                examples=[_toy_example(index) for index in range(12)],
                metadata={"toy": True},
            )
            training_set.save(training_set_path)

            report = train_one_mode(
                training_set,
                mode="full",
                output_dir=root / "ecology_model",
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                batch_size=4,
                epochs=12,
                learning_rate=1e-2,
                val_fraction=0.25,
                seed=7,
            )
            output_path = root / "kernel_teacher_dataset.npz"
            summary = build_kernel_teacher_dataset(
                output_path=output_path,
                ecology_training_set=training_set_path,
                ecology_checkpoint=report["checkpoint_path"],
                batch_size=4,
                device="cpu",
            )

            artifact = KernelTeacherDataset.load(output_path)
            summary_json = json.loads(output_path.with_suffix(".npz.summary.json").read_text(encoding="utf-8"))

        self.assertEqual(artifact.source_models, ["m1", "m2"])
        self.assertEqual(artifact.embedding_dim, 2)
        self.assertEqual(len(artifact.examples), 12)
        self.assertEqual(summary["example_count"], 12)
        self.assertGreaterEqual(float(summary["teacher_metrics"]["accuracy"]), 0.8)
        self.assertEqual(summary_json["example_count"], 12)
        first = artifact.examples[0]
        self.assertEqual(sorted(first["source_model_weights"]), ["m1", "m2"])
        self.assertEqual(len(first["candidate_weights"]), 2)
        self.assertEqual(len(first["cleared_embedding"]), 2)
        self.assertAlmostEqual(sum(first["candidate_weights"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
