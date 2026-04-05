from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.representation_learning.compare_gcca_stitching import compare_gcca_stitching
from tools.representation_learning.schemas import SharedLatentGeometry, SharedLatentLayer


class CompareGCCAStitchingTests(unittest.TestCase):
    def test_joins_pairwise_transfer_with_shared_geometry_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_path = root / "shared.npz"
            stitching_path = root / "stitching.json"
            geometry = SharedLatentGeometry(
                latent_dim=2,
                input_dim=4,
                source_models=["m1", "m2", "m3"],
                layers={
                    1: SharedLatentLayer(
                        relative_depth=1.0,
                        chunk_ids=["c1", "c2", "c3"],
                        shared_latents=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
                        model_projections={model_id: np.eye(4, 2, dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                        model_means={model_id: np.zeros((4,), dtype=np.float32) for model_id in ["m1", "m2", "m3"]},
                        aligned_latents={
                            "m1": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
                            "m2": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6]], dtype=np.float32),
                            "m3": np.asarray([[0.2, 0.8], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32),
                        },
                        metadata={"view_residuals": {"m1": 0.4, "m2": 0.5, "m3": 0.7}},
                    )
                },
            )
            geometry.save(shared_path)
            stitching_path.write_text(
                json.dumps(
                    {
                        "summary": {"shared_layer": 1},
                        "pairwise": [
                            {
                                "source_model_id": "m1",
                                "target_model_id": "m2",
                                "eval_target_logit_kl_mean": 0.2,
                                "eval_target_logit_js_mean": 0.1,
                                "eval_hidden_cosine_mean": 0.9,
                                "eval_target_top1_agreement": 0.8,
                                "report_path": "/tmp/m1_to_m2.json",
                            },
                            {
                                "source_model_id": "m3",
                                "target_model_id": "m2",
                                "eval_target_logit_kl_mean": 0.6,
                                "eval_target_logit_js_mean": 0.3,
                                "eval_hidden_cosine_mean": 0.4,
                                "eval_target_top1_agreement": 0.2,
                                "report_path": "/tmp/m3_to_m2.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = compare_gcca_stitching(
                shared_geometry_path=shared_path,
                stitching_cohort_report=stitching_path,
            )

        self.assertEqual(report["shared_layer"], 1)
        self.assertEqual(report["summary"]["pair_count"], 2)
        self.assertEqual(report["pairwise"][0]["source_residual"], 0.4)
        self.assertTrue(np.isfinite(report["pairwise"][0]["aligned_latent_cosine_mean"]))


if __name__ == "__main__":
    unittest.main()
