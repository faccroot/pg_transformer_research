from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.schemas import KernelTeacherTextDataset


TOKENIZER_PATH = Path("/home/zaytor/transformer_research/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
HAS_MLX = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(HAS_MLX, "mlx is required for kernel teacher student training tests")
class TrainKernelTeacherStudentTests(unittest.TestCase):
    def test_trains_small_student_on_teacher_text_dataset(self) -> None:
        from tools.representation_learning.train_kernel_teacher_student import train_kernel_teacher_student

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            teacher_path = root / "teacher_text.npz"
            output_dir = root / "student_out"
            teacher_dataset = KernelTeacherTextDataset(
                source_models=["m1", "m2"],
                embedding_dim=4,
                examples=[
                    {
                        "example_id": "e1",
                        "chunk_id": "c1",
                        "text": "The cat sat on the mat.",
                        "cleared_embedding": [1.0, 0.0, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.9, 0.1],
                        "candidate_embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        "winner_model": "m1",
                        "predicted_model": "m1",
                        "winner_probability": 0.9,
                        "verification_confidence": 0.8,
                    },
                    {
                        "example_id": "e2",
                        "chunk_id": "c2",
                        "text": "A kitten chased a red ball across the floor.",
                        "cleared_embedding": [0.9, 0.1, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.85, 0.15],
                        "candidate_embeddings": [[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
                        "winner_model": "m1",
                        "predicted_model": "m1",
                        "winner_probability": 0.85,
                        "verification_confidence": 0.7,
                    },
                    {
                        "example_id": "e3",
                        "chunk_id": "c3",
                        "text": "Dogs bark loudly at the passing truck.",
                        "cleared_embedding": [0.0, 1.0, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.1, 0.9],
                        "candidate_embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        "winner_model": "m2",
                        "predicted_model": "m2",
                        "winner_probability": 0.92,
                        "verification_confidence": 0.75,
                    },
                    {
                        "example_id": "e4",
                        "chunk_id": "c4",
                        "text": "A puppy wagged its tail near the door.",
                        "cleared_embedding": [0.1, 0.9, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.2, 0.8],
                        "candidate_embeddings": [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
                        "winner_model": "m2",
                        "predicted_model": "m2",
                        "winner_probability": 0.88,
                        "verification_confidence": 0.72,
                    },
                ],
                metadata={"toy": True},
            )
            teacher_dataset.save(teacher_path)

            summary = train_kernel_teacher_student(
                teacher_dataset_path=teacher_path,
                tokenizer_path=TOKENIZER_PATH,
                output_dir=output_dir,
                seed=3,
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                max_seq_len=24,
                val_fraction=0.25,
                ce_weight=0.1,
                distill_weight=1.0,
                readout_mode="mean_last",
                freeze_backbone=True,
                model_dim=32,
                num_layers=1,
                num_layer_templates=1,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
            )

            summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["teacher_dim"], 4)
        self.assertEqual(summary_json["teacher_dim"], 4)
        self.assertEqual(summary["readout_mode"], "mean_last")
        self.assertTrue((output_dir / "best_kernel_teacher_student.npz").is_file())
        self.assertTrue((output_dir / "kernel_teacher_student_final.npz").is_file())
        self.assertIn("loss_total", summary["final_val_metrics"])
        self.assertIn("mean_teacher_cosine", summary["final_val_metrics"])
        self.assertIn("loss_ce_probe", summary["final_val_metrics"])
        self.assertIn("bpb_probe", summary["final_val_metrics"])

    def test_trains_factorized_student_on_teacher_text_dataset(self) -> None:
        from tools.representation_learning.train_kernel_teacher_student import train_kernel_teacher_student

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            teacher_path = root / "teacher_text_factorized.npz"
            output_dir = root / "student_out_factorized"
            teacher_dataset = KernelTeacherTextDataset(
                source_models=["m1", "m2"],
                embedding_dim=4,
                examples=[
                    {
                        "example_id": "f1",
                        "chunk_id": "c1",
                        "text": "A formal explanation about rainfall and clouds.",
                        "cleared_embedding": [1.0, 0.0, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.95, 0.05],
                        "candidate_embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        "winner_model": "m1",
                        "predicted_model": "m1",
                        "winner_probability": 0.95,
                        "verification_confidence": 0.9,
                    },
                    {
                        "example_id": "f2",
                        "chunk_id": "c2",
                        "text": "An informal story about a dog in a backyard.",
                        "cleared_embedding": [0.0, 1.0, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.05, 0.95],
                        "candidate_embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        "winner_model": "m2",
                        "predicted_model": "m2",
                        "winner_probability": 0.95,
                        "verification_confidence": 0.9,
                    },
                    {
                        "example_id": "f3",
                        "chunk_id": "c3",
                        "text": "A short paragraph comparing cause and effect.",
                        "cleared_embedding": [0.8, 0.2, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.8, 0.2],
                        "candidate_embeddings": [[0.8, 0.2, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
                        "winner_model": "m1",
                        "predicted_model": "m1",
                        "winner_probability": 0.8,
                        "verification_confidence": 0.8,
                    },
                    {
                        "example_id": "f4",
                        "chunk_id": "c4",
                        "text": "A conversational exchange about planning tomorrow.",
                        "cleared_embedding": [0.2, 0.8, 0.0, 0.0],
                        "candidate_model_ids": ["m1", "m2"],
                        "candidate_weights": [0.2, 0.8],
                        "candidate_embeddings": [[0.8, 0.2, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
                        "winner_model": "m2",
                        "predicted_model": "m2",
                        "winner_probability": 0.8,
                        "verification_confidence": 0.8,
                    },
                ],
                metadata={"toy": True, "factorized": True},
            )
            teacher_dataset.save(teacher_path)

            summary = train_kernel_teacher_student(
                teacher_dataset_path=teacher_path,
                tokenizer_path=TOKENIZER_PATH,
                output_dir=output_dir,
                seed=5,
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                max_seq_len=24,
                val_fraction=0.25,
                ce_weight=0.0,
                distill_weight=1.0,
                projection_mode="factorized",
                factor_count=2,
                factor_loss_weight=1.0,
                readout_mode="last",
                freeze_backbone=True,
                model_dim=32,
                num_layers=1,
                num_layer_templates=1,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
            )

            summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["projection_mode"], "factorized")
        self.assertEqual(summary_json["projection_mode"], "factorized")
        self.assertEqual(summary["factor_count"], 2)
        self.assertIn("loss_factor", summary["final_val_metrics"])
        self.assertIn("loss_ce_probe", summary["final_val_metrics"])
        self.assertIn("bpb_probe", summary["final_val_metrics"])
        self.assertTrue((output_dir / "best_kernel_teacher_student.npz").is_file())


if __name__ == "__main__":
    unittest.main()
