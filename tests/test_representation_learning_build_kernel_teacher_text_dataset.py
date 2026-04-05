from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.build_kernel_teacher_text_dataset import build_kernel_teacher_text_dataset
from tools.representation_learning.schemas import KernelTeacherDataset, KernelTeacherTextDataset


class BuildKernelTeacherTextDatasetTests(unittest.TestCase):
    def test_joins_teacher_examples_with_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            teacher_path = root / "teacher.npz"
            calibration_path = root / "calibration.jsonl"
            output_path = root / "teacher_text.npz"

            teacher = KernelTeacherDataset(
                source_models=["m1", "m2"],
                embedding_dim=2,
                examples=[
                    {
                        "example_id": "ex1",
                        "chunk_id": "chunk_a",
                        "winner_model": "m1",
                        "predicted_model": "m1",
                        "cleared_embedding": [1.0, 0.0],
                    }
                ],
                metadata={"toy": True},
            )
            teacher.save(teacher_path)
            calibration_path.write_text(
                json.dumps({"chunk_id": "chunk_a", "text": "hello world", "difficulty": 0.2}) + "\n",
                encoding="utf-8",
            )

            summary = build_kernel_teacher_text_dataset(
                output_path=output_path,
                kernel_teacher_dataset=teacher_path,
                calibration_jsonl=calibration_path,
            )

            artifact = KernelTeacherTextDataset.load(output_path)
            summary_json = json.loads(output_path.with_suffix(".npz.summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(summary_json["example_count"], 1)
        self.assertEqual(artifact.examples[0]["text"], "hello world")
        self.assertEqual(artifact.examples[0]["calibration_metadata"]["difficulty"], 0.2)


if __name__ == "__main__":
    unittest.main()
