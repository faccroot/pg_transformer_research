from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.build_kernel_teacher_baseline_dataset import (
    build_kernel_teacher_baseline_dataset,
)
from tools.representation_learning.schemas import KernelTeacherDataset


class BuildKernelTeacherBaselineDatasetTests(unittest.TestCase):
    def test_builds_fixed_source_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.npz"
            output_path = root / "baseline_fixed.npz"
            teacher = KernelTeacherDataset(
                source_models=["m1", "m2"],
                embedding_dim=2,
                examples=[
                    {
                        "example_id": "ex1",
                        "chunk_id": "chunk_a",
                        "winner_model": "m2",
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.3, 0.7],
                        "source_model_weights": {"m1": 0.3, "m2": 0.7},
                        "candidate_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                        "cleared_embedding": [0.3, 0.7],
                    }
                ],
                metadata={"toy": True},
            )
            teacher.save(input_path)

            summary = build_kernel_teacher_baseline_dataset(
                output_path=output_path,
                kernel_teacher_dataset=input_path,
                mode="fixed_source",
                fixed_source_model="m1",
            )
            artifact = KernelTeacherDataset.load(output_path)
            summary_json = json.loads(output_path.with_suffix(".npz.summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["mode"], "fixed_source")
        self.assertEqual(summary["fixed_source_model"], "m1")
        self.assertEqual(summary_json["teacher_accuracy"], 0.0)
        self.assertEqual(artifact.examples[0]["predicted_model"], "m1")
        self.assertEqual(artifact.examples[0]["candidate_weights"], [1.0, 0.0])
        self.assertEqual(artifact.examples[0]["cleared_embedding"], [1.0, 0.0])

    def test_builds_teacher_mean_static_mix_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "teacher.npz"
            output_path = root / "baseline_mix.npz"
            teacher = KernelTeacherDataset(
                source_models=["m1", "m2"],
                embedding_dim=2,
                examples=[
                    {
                        "example_id": "ex1",
                        "chunk_id": "chunk_a",
                        "winner_model": "m1",
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.8, 0.2],
                        "source_model_weights": {"m1": 0.8, "m2": 0.2},
                        "candidate_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                        "cleared_embedding": [0.8, 0.2],
                    },
                    {
                        "example_id": "ex2",
                        "chunk_id": "chunk_b",
                        "winner_model": "m2",
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.4, 0.6],
                        "source_model_weights": {"m1": 0.4, "m2": 0.6},
                        "candidate_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                        "cleared_embedding": [0.4, 0.6],
                    },
                ],
                metadata={"toy": True},
            )
            teacher.save(input_path)

            summary = build_kernel_teacher_baseline_dataset(
                output_path=output_path,
                kernel_teacher_dataset=input_path,
                mode="static_mix",
                static_mix_strategy="teacher_mean",
            )
            artifact = KernelTeacherDataset.load(output_path)

        self.assertEqual(summary["mode"], "static_mix")
        self.assertEqual(summary["static_mix_strategy"], "teacher_mean")
        self.assertAlmostEqual(summary["static_mix_source_weights"]["m1"], 0.6, places=5)
        self.assertAlmostEqual(summary["static_mix_source_weights"]["m2"], 0.4, places=5)
        self.assertEqual(artifact.examples[0]["candidate_weights"], [0.6000000238418579, 0.4000000059604645])
        self.assertEqual(artifact.examples[1]["candidate_weights"], [0.6000000238418579, 0.4000000059604645])
        self.assertEqual(artifact.examples[0]["predicted_model"], "m1")


if __name__ == "__main__":
    unittest.main()
