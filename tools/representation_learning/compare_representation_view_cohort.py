#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.compare_representation_views import build_view_comparison_report
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from compare_representation_views import build_view_comparison_report  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def build_view_cohort_report(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int | None = None,
    num_layers: int | None = None,
) -> dict[str, object]:
    if len(representations) < 2:
        raise ValueError("Need at least two representation views to build a cohort report")
    pairwise: list[dict[str, object]] = []
    for left_idx in range(len(representations)):
        for right_idx in range(left_idx + 1, len(representations)):
            report = build_view_comparison_report(
                representations[left_idx],
                representations[right_idx],
                canonical_dim=canonical_dim,
                num_layers=num_layers,
            )
            pairwise.append(report)
    return {
        "views": [
            {
                "index": int(index),
                "model_id": rep.model_id,
                "extraction_method": str(rep.metadata.get("extraction_method", "unknown")),
            }
            for index, rep in enumerate(representations)
        ],
        "pairwise": pairwise,
        "summary": {
            "mean_pairwise_subspace_overlap": float(
                sum(float(item["summary"]["mean_subspace_overlap"]) for item in pairwise) / max(len(pairwise), 1)
            ),
            "mean_pairwise_concept_alignment": float(
                sum(float(item["summary"]["mean_concept_alignment"]) for item in pairwise) / max(len(pairwise), 1)
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a cohort of representation views, for example activation, weight, and Jacobian extractions of the same model.")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=0, help="Shared dimension when hidden sizes differ; 0 means direct comparison when possible")
    parser.add_argument("--num-layers", type=int, default=0, help="Number of relative-depth bins; 0 uses the max source depth")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    report = build_view_cohort_report(
        representations,
        canonical_dim=None if args.canonical_dim <= 0 else args.canonical_dim,
        num_layers=None if args.num_layers <= 0 else args.num_layers,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "pair_count": len(report["pairwise"]),
                "mean_pairwise_concept_alignment": report["summary"]["mean_pairwise_concept_alignment"],
                "mean_pairwise_subspace_overlap": report["summary"]["mean_pairwise_subspace_overlap"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
