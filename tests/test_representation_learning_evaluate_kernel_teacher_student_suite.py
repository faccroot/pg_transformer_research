from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.representation_learning.evaluate_kernel_teacher_student_suite import (
    evaluate_kernel_teacher_student_suite,
)


class EvaluateKernelTeacherStudentSuiteTests(unittest.TestCase):
    def test_sorts_suite_by_bpb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_dir = root / "reports"
            out_dir = root / "evals"
            for run in ("a", "b", "c"):
                run_dir = report_dir / run
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(json.dumps({"stub": run}), encoding="utf-8")

            def fake_eval(*, summary_path, eval_jsonl, output_path, tokenizer_path, text_key, batch_size, max_examples, max_seq_len):
                name = Path(summary_path).parent.name
                bpb = {"a": 2.0, "b": 1.5, "c": 3.0}[name]
                payload = {"summary_path": str(summary_path), "bpb": bpb}
                Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with mock.patch(
                "tools.representation_learning.evaluate_kernel_teacher_student_suite._evaluate_checkpoint",
                side_effect=fake_eval,
            ):
                rows = evaluate_kernel_teacher_student_suite(
                    report_dir=report_dir,
                    eval_jsonl=root / "eval.jsonl",
                    output_dir=out_dir,
                )
                comparison = json.loads((out_dir / "comparison_v1.json").read_text(encoding="utf-8"))

        self.assertEqual([row["run"] for row in rows], ["b", "a", "c"])
        self.assertEqual([row["run"] for row in comparison], ["b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
