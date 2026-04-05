from __future__ import annotations

import unittest

from tools.representation_learning.summarize_init_eval_report import build_summary


class SummarizeInitEvalReportTests(unittest.TestCase):
    def test_build_summary_extracts_best_labels_and_deltas(self) -> None:
        payload = {
            "results": {
                "baseline": {"mean_val_bpb": 1.5},
                "qwen": {"mean_val_bpb": 1.2},
                "merge_hybrid": {"mean_val_bpb": 1.1},
            },
            "ranking": [
                {"label": "merge_hybrid", "mean_val_bpb": 1.1},
                {"label": "qwen", "mean_val_bpb": 1.2},
            ],
            "best_single_source_label": "qwen",
            "best_merged_label": "merge_hybrid",
            "paired_vs_baseline": {
                "qwen": {"mean_delta_bpb_vs_baseline": -0.3},
                "merge_hybrid": {"mean_delta_bpb_vs_baseline": -0.4},
            },
            "paired_vs_best_single_source": {
                "merge_hybrid": {"mean_delta_bpb_vs_reference": -0.1},
            },
        }
        summary = build_summary(payload)
        self.assertEqual(summary["top_ranked_label"], "merge_hybrid")
        self.assertEqual(summary["best_single_source_label"], "qwen")
        self.assertEqual(summary["best_merged_label"], "merge_hybrid")
        self.assertEqual(summary["deltas"]["best_single_vs_baseline"]["mean_delta_bpb_vs_baseline"], -0.3)
        self.assertEqual(summary["deltas"]["best_merged_vs_best_single"]["mean_delta_bpb_vs_reference"], -0.1)


if __name__ == "__main__":
    unittest.main()
