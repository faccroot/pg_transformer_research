from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.representation_learning.summarize_kernel_teacher_student_reports import (
    summarize_kernel_teacher_student_reports,
)


class SummarizeKernelTeacherStudentReportsTests(unittest.TestCase):
    def test_sorts_by_bpb_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, bpb, loss in [
                ("run_a", 10.1, 0.8),
                ("run_b", 9.9, 0.7),
                ("run_c", 10.0, 0.6),
            ]:
                run_dir = root / name
                run_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "teacher_dataset_path": f"/tmp/{name}.npz",
                    "projection_mode": "direct",
                    "readout_mode": "mean_last",
                    "ce_weight": 0.0,
                    "distill_weight": 1.0,
                    "final_val_metrics": {
                        "loss_total": loss,
                        "loss_distill": loss,
                        "loss_ce_probe": 6.9,
                        "bpb_probe": bpb,
                        "mean_teacher_cosine": 0.3,
                    },
                }
                (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            rows = summarize_kernel_teacher_student_reports(root, sort_metric="bpb_probe")

        self.assertEqual([row["run"] for row in rows], ["run_b", "run_c", "run_a"])
        self.assertEqual(rows[0]["val_bpb_probe"], 9.9)


if __name__ == "__main__":
    unittest.main()
