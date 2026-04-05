from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.schemas import SharedLatentGeometry, SharedLatentLayer
from tools.representation_learning.verify_model_stitching_cohort import build_stitching_cohort_report


def _text_feature_vector(text: str) -> np.ndarray:
    index = int(text.rsplit(" ", 1)[-1])
    return np.asarray([1.0, float(index), float(index * index)], dtype=np.float32)


class _FakeAdapter:
    num_layers = 2
    _hidden_params = {
        "m1": (
            np.asarray(
                [
                    [0.4, -0.1, 0.2],
                    [0.3, 0.2, -0.2],
                    [0.1, 0.3, 0.4],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.2, -0.3, 0.1], dtype=np.float32),
        ),
        "m2": (
            np.asarray(
                [
                    [0.5, -0.2, 0.1],
                    [-0.1, 0.4, 0.2],
                    [0.2, 0.1, 0.5],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.1, -0.2, 0.05], dtype=np.float32),
        ),
        "m3": (
            np.asarray(
                [
                    [0.3, 0.1, -0.1],
                    [0.2, -0.3, 0.4],
                    [0.1, 0.2, 0.2],
                ],
                dtype=np.float32,
            ),
            np.asarray([-0.1, 0.05, 0.2], dtype=np.float32),
        ),
    }
    _logit_heads = {
        "m1": (np.asarray([[0.5, -0.1, 0.2], [0.1, 0.3, -0.2], [0.0, -0.2, 0.4]], dtype=np.float32), np.asarray([0.1, -0.1, 0.05], dtype=np.float32)),
        "m2": (np.asarray([[0.2, -0.3, 0.4], [0.1, 0.2, -0.1], [0.3, 0.0, -0.1]], dtype=np.float32), np.asarray([0.05, -0.15, 0.2], dtype=np.float32)),
        "m3": (np.asarray([[0.1, 0.2, -0.2], [-0.2, 0.1, 0.4], [0.3, -0.1, 0.2]], dtype=np.float32), np.asarray([-0.05, 0.1, 0.0], dtype=np.float32)),
    }

    def __init__(self, model_id: str, **_kwargs) -> None:
        self.model_id = model_id

    def get_sequence_representations(
        self,
        texts: list[str],
        *,
        layers: list[int] | None = None,
        capture_full_sequences: bool = False,
        max_length: int | None = None,
    ):
        del max_length
        hidden_weight, hidden_bias = self._hidden_params[self.model_id]
        head_weight, head_bias = self._logit_heads[self.model_id]
        base = np.stack([_text_feature_vector(text) for text in texts], axis=0).astype(np.float32)
        last_hidden = base @ hidden_weight + hidden_bias
        mean_hidden = 0.5 * last_hidden
        last_logits = last_hidden @ head_weight + head_bias
        from tools.representation_learning.model_adapter import SequenceRepresentationBatch

        return SequenceRepresentationBatch(
            mean_hidden=mean_hidden.astype(np.float32),
            last_hidden=last_hidden.astype(np.float32),
            last_logits=last_logits.astype(np.float32),
            attention_mask=np.ones((len(texts), 3), dtype=np.int32),
            layer_last_hidden={},
            layer_hidden_sequences={},
        )

    def project_hidden_to_logits(self, hidden):
        head_weight, head_bias = self._logit_heads[self.model_id]
        return np.asarray(hidden, dtype=np.float32) @ head_weight + head_bias


class VerifyModelStitchingCohortTests(unittest.TestCase):
    def test_builds_pairwise_cohort_summary(self) -> None:
        import tools.representation_learning.verify_model_stitching_cohort as cohort_mod

        prev_adapter = cohort_mod.HFCausalLMAdapter
        cohort_mod.HFCausalLMAdapter = _FakeAdapter
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                calibration_path = root / "calibration.jsonl"
                with calibration_path.open("w", encoding="utf-8") as handle:
                    for index in range(6):
                        handle.write(json.dumps({"chunk_id": f"c{index}", "text": f"sample {index}"}) + "\n")
                shared_path = root / "shared.npz"
                geometry = SharedLatentGeometry(
                    latent_dim=2,
                    input_dim=4,
                    source_models=["m1", "m2", "m3"],
                    layers={
                        1: SharedLatentLayer(
                            relative_depth=1.0,
                            chunk_ids=["c0", "c1", "c2", "c3", "c4"],
                            shared_latents=np.asarray(
                                [[1.0, 0.0], [0.0, 1.0], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8]],
                                dtype=np.float32,
                            ),
                            model_projections={model_id: np.eye(4, 2, dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                            model_means={model_id: np.zeros((4,), dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                        )
                    },
                )
                geometry.save(shared_path)
                output_dir = root / "reports"
                report = build_stitching_cohort_report(
                    shared_geometry_path=shared_path,
                    calibration_jsonl=calibration_path,
                    output_dir=output_dir,
                    shared_layer=1,
                    max_examples=5,
                    train_fraction=0.6,
                    seed=5,
                    batch_size=2,
                    max_length=64,
                    ridge_lambda=0.0,
                )
                persisted = json.loads((output_dir / "cohort_summary.json").read_text(encoding="utf-8"))
                first_report_exists = Path(report["pairwise"][0]["report_path"]).exists()
        finally:
            cohort_mod.HFCausalLMAdapter = prev_adapter

        self.assertEqual(report["summary"]["pair_count"], 6)
        self.assertEqual(len(report["pairwise"]), 6)
        self.assertEqual(report["selected_chunk_ids"], ["c0", "c1", "c2", "c3", "c4"])
        self.assertIn("m1", report["summary"]["best_source_for_target"])
        self.assertEqual(persisted["summary"]["shared_layer"], 1)
        self.assertTrue(first_report_exists)


if __name__ == "__main__":
    unittest.main()
