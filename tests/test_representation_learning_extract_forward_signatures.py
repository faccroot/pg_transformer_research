from __future__ import annotations

import unittest

import numpy as np

from tools.representation_learning.extract_forward_signatures import extract_forward_signatures


class _FakeAdapter:
    def __init__(self) -> None:
        self.model_id = "fake/model"
        self.num_layers = 2
        self.device = "cpu"
        self.calls: list[tuple[list[str], list[int], int, int]] = []

    def get_forward_signatures(
        self,
        texts: list[str],
        *,
        layers: list[int],
        max_length: int | None = None,
        top_k: int = 8,
    ):
        self.calls.append((list(texts), list(layers), int(max_length or 0), int(top_k)))

        class _Batch:
            pass

        batch = _Batch()
        batch.global_features = {
            "last_token_entropy": np.array([float(len(text)) for text in texts], dtype=np.float32),
            "sequence_mean_entropy": np.array([float(len(text)) + 0.5 for text in texts], dtype=np.float32),
            "last_token_top1_prob": np.array([0.7] * len(texts), dtype=np.float32),
            "last_token_margin": np.array([0.2] * len(texts), dtype=np.float32),
            "last_token_topk_mass": np.array([0.9] * len(texts), dtype=np.float32),
        }
        batch.layer_features = {
            int(layer_idx): {
                "attention_entropy": np.array([0.1 * layer_idx] * len(texts), dtype=np.float32),
                "attention_peak_frac": np.array([0.9 - 0.1 * layer_idx] * len(texts), dtype=np.float32),
            }
            for layer_idx in layers
        }
        batch.topk_token_ids = np.tile(np.array([[1, 2, 3]], dtype=np.int32), (len(texts), 1))
        batch.topk_token_probs = np.tile(np.array([[0.6, 0.3, 0.1]], dtype=np.float32), (len(texts), 1))
        return batch


class ExtractForwardSignaturesTests(unittest.TestCase):
    def test_extracts_and_stacks_batches(self) -> None:
        adapter = _FakeAdapter()
        dataset = extract_forward_signatures(
            [
                {"chunk_id": "c1", "text": "alpha"},
                {"chunk_id": "c2", "text": "beta"},
            ],
            adapter=adapter,
            calibration_jsonl="calibration.jsonl",
            layers=[1, 2],
            batch_size=1,
            max_length=32,
            top_k=3,
            torch_dtype="float16",
        )
        self.assertEqual(dataset.model_id, "fake/model")
        self.assertEqual(dataset.chunk_ids, ["c1", "c2"])
        np.testing.assert_allclose(dataset.global_features["last_token_entropy"], np.array([5.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(dataset.layer_features[2]["attention_entropy"], np.array([0.2, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(dataset.topk_token_ids, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32))
        self.assertEqual(adapter.calls[0][1], [1, 2])


if __name__ == "__main__":
    unittest.main()
