#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _evaluate_checkpoint(**kwargs):
    from tools.representation_learning.evaluate_kernel_teacher_student_checkpoint import (
        evaluate_kernel_teacher_student_checkpoint,
    )
    return evaluate_kernel_teacher_student_checkpoint(**kwargs)


def evaluate_kernel_teacher_student_suite(
    *,
    report_dir: str | Path,
    eval_jsonl: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    text_key: str = "text",
    batch_size: int = 4,
    max_examples: int = 0,
    max_seq_len: int | None = None,
) -> list[dict[str, Any]]:
    report_dir = Path(report_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(report_dir.glob("*/summary.json")):
        run_name = summary_path.parent.name
        output_path = output_dir / f"{run_name}.json"
        result = _evaluate_checkpoint(
            summary_path=summary_path,
            eval_jsonl=eval_jsonl,
            output_path=output_path,
            tokenizer_path=tokenizer_path,
            text_key=text_key,
            batch_size=batch_size,
            max_examples=max_examples,
            max_seq_len=max_seq_len,
        )
        result = dict(result)
        result["run"] = run_name
        rows.append(result)
    rows.sort(key=lambda row: float(row["bpb"]))
    (output_dir / "comparison_v1.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all kernel-teacher-student summaries under a report dir on a common JSONL text surface.")
    parser.add_argument("report_dir", help="Directory containing per-run summary.json files")
    parser.add_argument("--eval-jsonl", required=True, help="JSONL file containing evaluation text")
    parser.add_argument("--output-dir", required=True, help="Directory for per-run eval JSON and comparison_v1.json")
    parser.add_argument("--tokenizer-path", default="", help="Optional tokenizer override")
    parser.add_argument("--text-key", default="text", help="JSON key containing text")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    args = parser.parse_args()

    rows = evaluate_kernel_teacher_student_suite(
        report_dir=args.report_dir,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        tokenizer_path=(args.tokenizer_path or None),
        text_key=args.text_key,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        max_seq_len=(args.max_seq_len if args.max_seq_len > 0 else None),
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
