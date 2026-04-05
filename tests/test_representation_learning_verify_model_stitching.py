from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.schemas import SharedLatentGeometry, SharedLatentLayer
from tools.representation_learning.verify_model_stitching import (
    apply_affine_stitch,
    fit_affine_stitch,
    verify_model_stitching,
)


def _text_feature_vector(text: str) -> np.ndarray:
    index = int(text.rsplit(" ", 1)[-1])
    return np.asarray([1.0, float(index), float(index * index)], dtype=np.float32)


class _FakeAdapter:
    num_layers = 2
    _layer_hidden_params = {
        "source/model": (
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
        "target/model": (
            np.asarray(
                [
                    [0.5, -0.2, 0.1, 0.3],
                    [-0.1, 0.4, 0.2, -0.2],
                    [0.2, 0.1, 0.5, 0.1],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.1, -0.2, 0.05, 0.3], dtype=np.float32),
        ),
    }
    _continuation_params = {
        "source/model": (
            np.asarray(
                [
                    [0.7, 0.1, 0.0],
                    [0.0, 0.8, 0.1],
                    [0.1, 0.0, 0.9],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.05, -0.05, 0.1], dtype=np.float32),
        ),
        "target/model": (
            np.asarray(
                [
                    [0.8, 0.1, -0.1, 0.0],
                    [0.0, 0.7, 0.1, 0.2],
                    [0.1, 0.0, 0.9, 0.1],
                    [-0.1, 0.2, 0.0, 0.8],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.05, -0.1, 0.1, 0.05], dtype=np.float32),
        ),
    }
    _logit_heads = {
        "source/model": (
            np.asarray(
                [
                    [0.5, -0.1, 0.2, 0.0],
                    [0.1, 0.3, -0.2, 0.1],
                    [0.0, -0.2, 0.4, 0.3],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.1, -0.1, 0.05, 0.2], dtype=np.float32),
        ),
        "target/model": (
            np.asarray(
                [
                    [0.2, -0.3, 0.4, 0.1, 0.0],
                    [0.1, 0.2, -0.1, 0.3, -0.2],
                    [0.0, 0.1, 0.2, -0.2, 0.4],
                    [0.3, 0.0, -0.1, 0.2, 0.1],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.05, -0.15, 0.2, 0.1, -0.05], dtype=np.float32),
        ),
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


class _TorchReturningFakeAdapter(_FakeAdapter):
    def project_hidden_to_logits(self, hidden):
        import torch

        head_weight, head_bias = self._logit_heads[self.model_id]
        tensor = torch.as_tensor(np.asarray(hidden, dtype=np.float32))
        return tensor @ torch.as_tensor(head_weight) + torch.as_tensor(head_bias)


class VerifyModelStitchingTests(unittest.TestCase):
    def test_fit_affine_stitch_recovers_known_map(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(32, 3)).astype(np.float32)
        weight = np.asarray(
            [
                [0.5, -0.1, 0.3, 0.2],
                [-0.2, 0.4, 0.1, -0.3],
                [0.1, 0.2, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        bias = np.asarray([0.3, -0.2, 0.1, 0.05], dtype=np.float32)
        target = source @ weight + bias
        fit = fit_affine_stitch(source, target, ridge_lambda=0.0)
        stitched = apply_affine_stitch(source, fit)
        np.testing.assert_allclose(stitched, target, atol=1e-5, rtol=1e-5)
        self.assertGreaterEqual(fit.rank, 3)

    def test_verify_model_stitching_honors_shared_geometry_and_reports_transfer(self) -> None:
        import tools.representation_learning.verify_model_stitching as stitching_mod

        prev_adapter = stitching_mod.HFCausalLMAdapter
        stitching_mod.HFCausalLMAdapter = _FakeAdapter
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                calibration_path = root / "calibration.jsonl"
                with calibration_path.open("w", encoding="utf-8") as handle:
                    for index in range(6):
                        handle.write(
                            json.dumps(
                                {
                                    "chunk_id": f"c{index}",
                                    "text": f"sample {index}",
                                }
                            )
                            + "\n"
                        )
                shared_path = root / "shared.npz"
                geometry = SharedLatentGeometry(
                    latent_dim=2,
                    input_dim=4,
                    source_models=["source/model", "target/model"],
                    layers={
                        1: SharedLatentLayer(
                            relative_depth=1.0,
                            chunk_ids=["c1", "c2", "c3", "c4"],
                            shared_latents=np.asarray(
                                [
                                    [1.0, 0.0],
                                    [0.0, 1.0],
                                    [0.5, 0.5],
                                    [0.2, 0.8],
                                ],
                                dtype=np.float32,
                            ),
                            model_projections={
                                "source/model": np.eye(4, 2, dtype=np.float32),
                                "target/model": np.eye(4, 2, dtype=np.float32),
                            },
                            model_means={
                                "source/model": np.zeros((4,), dtype=np.float32),
                                "target/model": np.zeros((4,), dtype=np.float32),
                            },
                        )
                    },
                )
                geometry.save(shared_path)
                output_path = root / "report.json"
                report = verify_model_stitching(
                    source_model_id="source/model",
                    target_model_id="target/model",
                    calibration_jsonl=calibration_path,
                    output_path=output_path,
                    shared_geometry_path=shared_path,
                    shared_layer=1,
                    max_examples=4,
                    train_fraction=0.75,
                    seed=3,
                    batch_size=2,
                    max_length=64,
                    ridge_lambda=0.0,
                )
                persisted = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            stitching_mod.HFCausalLMAdapter = prev_adapter

        self.assertEqual(report["num_examples"], 4)
        self.assertEqual(report["metadata"]["chunk_ids"], ["c1", "c2", "c3", "c4"])
        self.assertLess(report["metrics"]["eval"]["hidden_mse"], 1e-6)
        self.assertGreater(report["metrics"]["eval"]["hidden_cosine_mean"], 0.999)
        self.assertGreater(report["metrics"]["eval"]["target_top1_agreement"], 0.999)
        self.assertEqual(persisted["shared_layer"], 1)
        self.assertEqual(persisted["source_model_id"], "source/model")

    def test_verify_model_stitching_accepts_torch_logit_projection_outputs(self) -> None:
        import tools.representation_learning.verify_model_stitching as stitching_mod

        prev_adapter = stitching_mod.HFCausalLMAdapter
        stitching_mod.HFCausalLMAdapter = _TorchReturningFakeAdapter
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                calibration_path = root / "calibration.jsonl"
                with calibration_path.open("w", encoding="utf-8") as handle:
                    for index in range(6):
                        handle.write(json.dumps({"chunk_id": f"c{index}", "text": f"sample {index}"}) + "\n")
                output_path = root / "report.json"
                report = verify_model_stitching(
                    source_model_id="source/model",
                    target_model_id="target/model",
                    calibration_jsonl=calibration_path,
                    output_path=output_path,
                    max_examples=6,
                    train_fraction=0.5,
                    seed=1,
                    batch_size=2,
                    max_length=64,
                    ridge_lambda=0.0,
                )
        finally:
            stitching_mod.HFCausalLMAdapter = prev_adapter

        self.assertGreater(report["metrics"]["eval"]["target_top1_agreement"], 0.999)

    def test_verify_model_stitching_supports_layer_continuation_mode(self) -> None:
        import tools.representation_learning.verify_model_stitching as stitching_mod

        prev_adapter = stitching_mod.HFCausalLMAdapter
        stitching_mod.HFCausalLMAdapter = _FakeAdapter
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
                    source_models=["source/model", "target/model"],
                    layers={
                        1: SharedLatentLayer(
                            relative_depth=0.5,
                            chunk_ids=["c0", "c1", "c2", "c3", "c4"],
                            shared_latents=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32),
                            model_projections={
                                "source/model": np.eye(4, 2, dtype=np.float32),
                                "target/model": np.eye(4, 2, dtype=np.float32),
                            },
                            model_means={
                                "source/model": np.zeros((4,), dtype=np.float32),
                                "target/model": np.zeros((4,), dtype=np.float32),
                            },
                        )
                    },
                )
                geometry.save(shared_path)
                output_path = root / "report.json"
                report = verify_model_stitching(
                    source_model_id="source/model",
                    target_model_id="target/model",
                    calibration_jsonl=calibration_path,
                    output_path=output_path,
                    shared_geometry_path=shared_path,
                    shared_layer=1,
                    max_examples=5,
                    train_fraction=0.6,
                    seed=3,
                    batch_size=2,
                    max_length=64,
                    ridge_lambda=0.0,
                    representation_mode="layer_last_hidden_continuation",
                )
        finally:
            stitching_mod.HFCausalLMAdapter = prev_adapter

        self.assertEqual(report["metadata"]["representation_mode"], "layer_last_hidden_continuation")
        self.assertEqual(report["source_layer_idx"], 1)
        self.assertEqual(report["target_layer_idx"], 1)
        self.assertGreater(report["metrics"]["eval"]["target_top1_agreement"], 0.999)


if __name__ == "__main__":
    unittest.main()
