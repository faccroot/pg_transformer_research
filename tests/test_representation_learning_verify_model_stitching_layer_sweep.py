from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.schemas import SharedLatentGeometry, SharedLatentLayer
from tools.representation_learning.verify_model_stitching_layer_sweep import build_stitching_layer_sweep_report


def _text_feature_vector(text: str) -> np.ndarray:
    index = int(text.rsplit(" ", 1)[-1])
    return np.asarray([1.0, float(index), float(index * index)], dtype=np.float32)


class _FakeAdapter:
    num_layers = 2
    _layer_hidden_params = {
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
    _continuation_params = {
        "m1": (
            np.asarray(
                [
                    [0.8, 0.0, 0.1],
                    [0.1, 0.7, 0.0],
                    [0.0, 0.2, 0.9],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.05, -0.05, 0.1], dtype=np.float32),
        ),
        "m2": (
            np.asarray(
                [
                    [0.7, 0.1, 0.0],
                    [0.0, 0.9, 0.1],
                    [0.1, 0.0, 0.8],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.0, 0.1, -0.05], dtype=np.float32),
        ),
        "m3": (
            np.asarray(
                [
                    [0.9, -0.1, 0.0],
                    [0.1, 0.8, 0.1],
                    [0.0, 0.1, 0.7],
                ],
                dtype=np.float32,
            ),
            np.asarray([-0.05, 0.0, 0.05], dtype=np.float32),
        ),
    }
    _logit_heads = {
        "m1": (np.asarray([[0.5, -0.1, 0.2], [0.1, 0.3, -0.2], [0.0, -0.2, 0.4]], dtype=np.float32), np.asarray([0.1, -0.1, 0.05], dtype=np.float32)),
        "m2": (np.asarray([[0.2, -0.3, 0.4], [0.1, 0.2, -0.1], [0.3, 0.0, -0.1]], dtype=np.float32), np.asarray([0.05, -0.15, 0.2], dtype=np.float32)),
        "m3": (np.asarray([[0.1, 0.2, -0.2], [-0.2, 0.1, 0.4], [0.3, -0.1, 0.2]], dtype=np.float32), np.asarray([-0.05, 0.1, 0.0], dtype=np.float32)),
    }

    def __init__(self, model_id: str, **_kwargs) -> None:
        self.model_id = model_id

    def _layer_last_hidden(self, base: np.ndarray, layer_idx: int) -> np.ndarray:
        hidden_weight, hidden_bias = self._layer_hidden_params[self.model_id]
        layer_one = base @ hidden_weight + hidden_bias
        if int(layer_idx) == 1:
            return layer_one.astype(np.float32)
        cont_weight, cont_bias = self._continuation_params[self.model_id]
        return (layer_one @ cont_weight + cont_bias).astype(np.float32)

    def get_sequence_representations(
        self,
        texts: list[str],
        *,
        layers: list[int] | None = None,
        capture_full_sequences: bool = False,
        max_length: int | None = None,
    ):
        del max_length
        head_weight, head_bias = self._logit_heads[self.model_id]
        base = np.stack([_text_feature_vector(text) for text in texts], axis=0).astype(np.float32)
        last_hidden = self._layer_last_hidden(base, 2)
        mean_hidden = 0.5 * last_hidden
        last_logits = last_hidden @ head_weight + head_bias
        from tools.representation_learning.model_adapter import SequenceRepresentationBatch

        attention_mask = np.ones((len(texts), 3), dtype=np.int32)
        layer_last_hidden = {}
        layer_hidden_sequences = {}
        for layer_idx in layers or []:
            layer_hidden = self._layer_last_hidden(base, int(layer_idx))
            layer_last_hidden[int(layer_idx)] = layer_hidden.astype(np.float32)
            if capture_full_sequences:
                seq_hidden = np.zeros((len(texts), 3, layer_hidden.shape[1]), dtype=np.float32)
                seq_hidden[:, -1, :] = layer_hidden
                layer_hidden_sequences[int(layer_idx)] = seq_hidden

        return SequenceRepresentationBatch(
            mean_hidden=mean_hidden.astype(np.float32),
            last_hidden=last_hidden.astype(np.float32),
            last_logits=last_logits.astype(np.float32),
            attention_mask=attention_mask,
            layer_last_hidden=layer_last_hidden,
            layer_hidden_sequences=layer_hidden_sequences,
        )

    def project_hidden_to_logits(self, hidden):
        head_weight, head_bias = self._logit_heads[self.model_id]
        return np.asarray(hidden, dtype=np.float32) @ head_weight + head_bias

    def continue_from_layer_hidden_sequence(self, hidden_sequence, attention_mask, *, layer_idx: int):
        del attention_mask
        hidden_sequence = np.asarray(hidden_sequence, dtype=np.float32)
        last_hidden = hidden_sequence[:, -1, :]
        if int(layer_idx) < self.num_layers:
            cont_weight, cont_bias = self._continuation_params[self.model_id]
            last_hidden = last_hidden @ cont_weight + cont_bias
        head_weight, head_bias = self._logit_heads[self.model_id]
        last_logits = last_hidden @ head_weight + head_bias
        from tools.representation_learning.model_adapter import SequenceRepresentationBatch

        return SequenceRepresentationBatch(
            mean_hidden=(0.5 * last_hidden).astype(np.float32),
            last_hidden=last_hidden.astype(np.float32),
            last_logits=last_logits.astype(np.float32),
            attention_mask=np.ones(hidden_sequence.shape[:2], dtype=np.int32),
        )


class VerifyModelStitchingLayerSweepTests(unittest.TestCase):
    def test_builds_layer_sweep_on_common_chunk_cohort(self) -> None:
        import tools.representation_learning.verify_model_stitching_layer_sweep as sweep_mod

        prev_adapter = sweep_mod.HFCausalLMAdapter
        sweep_mod.HFCausalLMAdapter = _FakeAdapter
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
                            relative_depth=0.5,
                            chunk_ids=["c0", "c1", "c2", "c3", "c4"],
                            shared_latents=np.asarray(
                                [[1.0, 0.0], [0.0, 1.0], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8]],
                                dtype=np.float32,
                            ),
                            model_projections={model_id: np.eye(4, 2, dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                            model_means={model_id: np.zeros((4,), dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                            aligned_latents={
                                "m1": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32),
                                "m2": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.4, 0.6], [0.3, 0.7]], dtype=np.float32),
                                "m3": np.asarray([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
                            },
                            metadata={"view_residuals": {"m1": 0.4, "m2": 0.5, "m3": 0.7}},
                        ),
                        2: SharedLatentLayer(
                            relative_depth=1.0,
                            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
                            shared_latents=np.asarray(
                                [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]],
                                dtype=np.float32,
                            ),
                            model_projections={model_id: np.eye(4, 2, dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                            model_means={model_id: np.zeros((4,), dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                            aligned_latents={
                                "m1": np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]], dtype=np.float32),
                                "m2": np.asarray([[0.7, 0.3], [0.3, 0.7], [0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=np.float32),
                                "m3": np.asarray([[0.8, 0.2], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
                            },
                            metadata={"view_residuals": {"m1": 0.45, "m2": 0.35, "m3": 0.55}},
                        ),
                    },
                )
                geometry.save(shared_path)
                output_dir = root / "reports"
                report = build_stitching_layer_sweep_report(
                    shared_geometry_path=shared_path,
                    calibration_jsonl=calibration_path,
                    output_dir=output_dir,
                    max_examples=3,
                    train_fraction=0.6,
                    seed=5,
                    batch_size=2,
                    max_length=64,
                    ridge_lambda=0.0,
                )
                persisted = json.loads((output_dir / "layer_sweep_summary.json").read_text(encoding="utf-8"))
                layer_01_exists = (output_dir / "layer_01" / "cohort_summary.json").exists()
                layer_02_exists = (output_dir / "layer_02" / "gcca_vs_stitch.json").exists()
        finally:
            sweep_mod.HFCausalLMAdapter = prev_adapter

        self.assertEqual(report["summary"]["layer_count"], 2)
        self.assertEqual(report["summary"]["evaluated_layers"], [1, 2])
        self.assertEqual(report["summary"]["selected_chunk_ids"], ["c1", "c2", "c3"])
        self.assertEqual(set(report["layers"]), {"1", "2"})
        self.assertTrue(layer_01_exists)
        self.assertTrue(layer_02_exists)
        self.assertIn("layer", report["summary"]["best_layer_by_mean_eval_target_logit_kl"])
        self.assertEqual(persisted["summary"]["selected_chunk_ids"], ["c1", "c2", "c3"])


if __name__ == "__main__":
    unittest.main()
