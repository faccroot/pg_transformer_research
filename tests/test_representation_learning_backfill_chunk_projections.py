from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.backfill_chunk_projections import (
    backfill_representation,
    candidate_calibration_paths,
    compute_chunk_projections,
    resolve_calibration_path,
    select_records,
)
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int], int]] = []

    def get_mean_pooled_hidden_states(
        self,
        texts: list[str],
        *,
        layers: list[int],
        max_length: int | None = None,
    ) -> dict[int, np.ndarray]:
        self.calls.append((list(texts), list(layers), int(max_length or 0)))
        result: dict[int, np.ndarray] = {}
        for layer_idx in layers:
            rows = []
            for text in texts:
                base = 1.0 if text == "alpha" else 2.0
                rows.append(np.array([base + layer_idx, base * 10.0 + layer_idx, 1.0], dtype=np.float32))
            result[int(layer_idx)] = np.stack(rows, axis=0).astype(np.float32)
        return result


def _make_representation() -> ModelRepresentation:
    return ModelRepresentation(
        model_id="fake/model",
        architecture_family="toy",
        num_parameters=42,
        hidden_dim=3,
        num_layers=2,
        layer_geometries={
            1: LayerGeometry(
                relative_depth=0.5,
                directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                scales=np.array([1.0, 1.0], dtype=np.float32),
            ),
            2: LayerGeometry(
                relative_depth=1.0,
                directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                scales=np.array([1.0, 1.0], dtype=np.float32),
            ),
        },
        chunk_losses=np.array([0.1, 0.2], dtype=np.float32),
        chunk_ids=["c1", "c2"],
        metadata={"batch_size": 2, "max_length": 32},
    )


class BackfillChunkProjectionsTests(unittest.TestCase):
    def test_select_records_preserves_artifact_order(self) -> None:
        records = [
            {"chunk_id": "c2", "text": "beta"},
            {"chunk_id": "c1", "text": "alpha"},
        ]
        selected = select_records(records, ["c1", "c2"])
        self.assertEqual([record["text"] for record in selected], ["alpha", "beta"])

    def test_compute_chunk_projections_fills_missing_layers(self) -> None:
        rep = _make_representation()
        adapter = _FakeAdapter()
        records = [
            {"chunk_id": "c1", "text": "alpha"},
            {"chunk_id": "c2", "text": "beta"},
        ]
        projections, summary = compute_chunk_projections(
            rep,
            adapter=adapter,
            records=records,
            batch_size=2,
            max_length=32,
        )
        np.testing.assert_allclose(
            projections[1],
            np.array([[2.0, 11.0], [3.0, 21.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            projections[2],
            np.array([[12.0, 1.0], [22.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(summary["computed_layers"], [1, 2])
        self.assertEqual(summary["reused_layers"], [])
        self.assertEqual(len(adapter.calls), 1)

    def test_backfill_representation_uses_local_calibration_fallback(self) -> None:
        rep = _make_representation()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact = tmp_path / "model_representation.npz"
            calibration = tmp_path / "calibration.jsonl"
            rep.save(artifact)
            with calibration.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"chunk_id": "c1", "text": "alpha"}) + "\n")
                handle.write(json.dumps({"chunk_id": "c2", "text": "beta"}) + "\n")

            import tools.representation_learning.backfill_chunk_projections as module

            original_adapter = module.HFCausalLMAdapter
            module.HFCausalLMAdapter = lambda *args, **kwargs: _FakeAdapter()
            try:
                updated, summary = backfill_representation(artifact)
            finally:
                module.HFCausalLMAdapter = original_adapter

            self.assertEqual(summary["calibration_jsonl"], str(calibration.resolve()))
            self.assertEqual(sorted(updated.chunk_layer_projections), [1, 2])
            reloaded = ModelRepresentation.load(artifact)
            self.assertEqual(sorted(reloaded.chunk_layer_projections), [1, 2])

    def test_resolve_calibration_path_checks_metadata_and_repo_fallbacks(self) -> None:
        rep = _make_representation()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact = tmp_path / "model_representation.npz"
            local_calibration = tmp_path / "calibration.jsonl"
            local_calibration.write_text("", encoding="utf-8")
            rep.save(artifact)
            candidates = candidate_calibration_paths(artifact, rep, None)
            self.assertIn(local_calibration, candidates)
            resolved = resolve_calibration_path(artifact, rep, None)
            self.assertEqual(resolved, local_calibration.resolve())


if __name__ == "__main__":
    unittest.main()
