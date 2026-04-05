#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric_value(payload: dict[str, Any], metric: str) -> float:
    final_metrics = payload.get("final_val_metrics", {})
    if metric in final_metrics:
        return float(final_metrics[metric])
    value = payload.get(metric)
    if value is None:
        raise KeyError(metric)
    return float(value)


def summarize_kernel_teacher_student_reports(
    report_dir: str | Path,
    *,
    sort_metric: str = "loss_total",
    ascending: bool = True,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    report_dir = Path(report_dir).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        final_metrics = payload.get("final_val_metrics", {})
        row = {
            "run": path.parent.name,
            "summary_path": str(path),
            "teacher_dataset_path": payload.get("teacher_dataset_path", ""),
            "projection_mode": payload.get("projection_mode", ""),
            "readout_mode": payload.get("readout_mode", ""),
            "ce_weight": float(payload.get("ce_weight", 0.0)),
            "distill_weight": float(payload.get("distill_weight", 0.0)),
            "val_loss_total": float(final_metrics.get("loss_total", 0.0)),
            "val_loss_distill": float(final_metrics.get("loss_distill", 0.0)),
            "val_loss_ce_probe": float(final_metrics.get("loss_ce_probe", 0.0)),
            "val_bpb_probe": float(final_metrics.get("bpb_probe", 0.0)),
            "val_mean_teacher_cosine": float(final_metrics.get("mean_teacher_cosine", 0.0)),
        }
        rows.append(row)
    rows.sort(key=lambda row: _metric_value({"final_val_metrics": {
        "loss_total": row["val_loss_total"],
        "loss_distill": row["val_loss_distill"],
        "loss_ce_probe": row["val_loss_ce_probe"],
        "bpb_probe": row["val_bpb_probe"],
        "mean_teacher_cosine": row["val_mean_teacher_cosine"],
    }}, sort_metric), reverse=not ascending)
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize kernel-teacher-student summary.json files under a report directory.")
    parser.add_argument("report_dir", help="Directory containing per-run summary.json files")
    parser.add_argument("--sort-metric", default="loss_total", choices=[
        "loss_total",
        "loss_distill",
        "loss_ce_probe",
        "bpb_probe",
        "mean_teacher_cosine",
    ])
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    rows = summarize_kernel_teacher_student_reports(
        args.report_dir,
        sort_metric=args.sort_metric,
        ascending=not args.descending,
        output_path=args.output or None,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
