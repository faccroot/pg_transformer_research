#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary(payload: dict[str, object]) -> dict[str, object]:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("report.results must be a JSON object")
    ranking = payload.get("ranking", [])
    if not isinstance(ranking, list):
        raise ValueError("report.ranking must be a JSON array")
    baseline = results.get("baseline", {})
    if not isinstance(baseline, dict):
        raise ValueError("report.results.baseline must be a JSON object")
    paired_vs_baseline = payload.get("paired_vs_baseline", {})
    paired_vs_best_single = payload.get("paired_vs_best_single_source", {})
    if not isinstance(paired_vs_baseline, dict) or not isinstance(paired_vs_best_single, dict):
        raise ValueError("paired summaries must be JSON objects")

    top_ranked = ranking[0] if ranking else None
    best_single_label = payload.get("best_single_source_label")
    best_merged_label = payload.get("best_merged_label")
    best_single_metrics = results.get(best_single_label, {}) if isinstance(best_single_label, str) else {}
    best_merged_metrics = results.get(best_merged_label, {}) if isinstance(best_merged_label, str) else {}
    summary: dict[str, object] = {
        "baseline_bpb": float(baseline.get("mean_val_bpb", 0.0)),
        "top_ranked_label": top_ranked.get("label") if isinstance(top_ranked, dict) else None,
        "top_ranked_bpb": float(top_ranked.get("mean_val_bpb", 0.0)) if isinstance(top_ranked, dict) else None,
        "best_single_source_label": best_single_label,
        "best_single_source_bpb": float(best_single_metrics.get("mean_val_bpb", 0.0)) if isinstance(best_single_metrics, dict) else None,
        "best_merged_label": best_merged_label,
        "best_merged_bpb": float(best_merged_metrics.get("mean_val_bpb", 0.0)) if isinstance(best_merged_metrics, dict) else None,
        "ranking": ranking,
        "deltas": {},
    }

    deltas = summary["deltas"]
    assert isinstance(deltas, dict)
    if isinstance(best_single_label, str) and best_single_label in paired_vs_baseline:
        deltas["best_single_vs_baseline"] = paired_vs_baseline[best_single_label]
    if isinstance(best_merged_label, str) and best_merged_label in paired_vs_baseline:
        deltas["best_merged_vs_baseline"] = paired_vs_baseline[best_merged_label]
    if isinstance(best_merged_label, str) and best_merged_label in paired_vs_best_single:
        deltas["best_merged_vs_best_single"] = paired_vs_best_single[best_merged_label]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a representation-learning MLX init-eval report.")
    parser.add_argument("report", help="Path to eval_mlx_representation_init.py JSON output")
    parser.add_argument("--output", default="", help="Optional path to write summary JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = Path(args.report).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = build_summary(payload)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
