from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.eval_mlx_representation_init import (
    inspect_artifact,
    parse_artifact_spec,
    rank_results,
    select_best_label,
    summarize_against_reference,
    summarize_paired_deltas,
)
from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation, PlatonicGeometry


class EvalMlxRepresentationInitTests(unittest.TestCase):
    def test_parse_artifact_spec(self) -> None:
        label, path = parse_artifact_spec("merge=/tmp/priors.npz")
        self.assertEqual(label, "merge")
        self.assertEqual(path, "/tmp/priors.npz")

    def test_parse_artifact_spec_requires_label_and_path(self) -> None:
        with self.assertRaises(ValueError):
            parse_artifact_spec("broken")

    def test_summarize_paired_deltas(self) -> None:
        summary = summarize_paired_deltas(
            baseline_batches=[1.0, 2.0, 3.0, 4.0],
            candidate_batches=[0.5, 2.5, 3.0, 3.0],
        )
        self.assertEqual(summary["wins_vs_baseline"], 2)
        self.assertEqual(summary["losses_vs_baseline"], 1)
        self.assertEqual(summary["ties_vs_baseline"], 1)
        self.assertAlmostEqual(summary["mean_delta_bpb_vs_baseline"], -0.25, places=6)

    def test_summarize_against_reference(self) -> None:
        summary = summarize_against_reference(
            {
                "best_single": [1.0, 2.0, 3.0],
                "candidate_a": [0.5, 2.5, 3.0],
                "candidate_b": [1.5, 2.5, 3.5],
            },
            reference_label="best_single",
        )
        self.assertEqual(summary["candidate_a"]["reference_label"], "best_single")
        self.assertEqual(summary["candidate_a"]["wins_vs_reference"], 1)
        self.assertEqual(summary["candidate_b"]["losses_vs_reference"], 3)

    def test_rank_and_select_best_label(self) -> None:
        results = {
            "baseline": {"mean_val_bpb": 1.5},
            "qwen": {"mean_val_bpb": 1.2},
            "olmo": {"mean_val_bpb": 1.1},
            "merge": {"mean_val_bpb": 1.0},
        }
        artifact_details = {
            "qwen": {"category": "single_source"},
            "olmo": {"category": "single_source"},
            "merge": {"category": "merged"},
        }
        ranking = rank_results(results)
        self.assertEqual([row["label"] for row in ranking], ["merge", "olmo", "qwen"])
        self.assertEqual(select_best_label(results, artifact_details, category="single_source"), "olmo")
        self.assertEqual(select_best_label(results, artifact_details, category="merged"), "merge")

    def test_inspect_artifact(self) -> None:
        rep = ModelRepresentation(
            model_id="model/a",
            architecture_family="toy",
            num_parameters=10,
            hidden_dim=4,
            num_layers=1,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
            },
        )
        geometry = PlatonicGeometry(
            canonical_dim=4,
            source_models=["model/a", "model/b"],
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=1.0,
                    directions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            rep_path = Path(tmpdir) / "rep.npz"
            geom_path = Path(tmpdir) / "geom.npz"
            rep.save(rep_path)
            geometry.save(geom_path)
            rep_meta = inspect_artifact(rep_path)
            geom_meta = inspect_artifact(geom_path)
        self.assertEqual(rep_meta["category"], "single_source")
        self.assertEqual(rep_meta["kind"], "model_representation")
        self.assertEqual(geom_meta["category"], "merged")
        self.assertEqual(geom_meta["num_source_models"], 2)


if __name__ == "__main__":
    unittest.main()
