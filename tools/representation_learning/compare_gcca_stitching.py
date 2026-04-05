#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import SharedLatentGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import SharedLatentGeometry  # type: ignore[no-redef]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _pairwise_cosine_lookup(layer) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    model_ids = sorted(layer.aligned_latents)
    for idx, model_a in enumerate(model_ids):
        a = np.asarray(layer.aligned_latents[model_a], dtype=np.float32)
        for model_b in model_ids[idx + 1:]:
            b = np.asarray(layer.aligned_latents[model_b], dtype=np.float32)
            count = min(a.shape[0], b.shape[0])
            if count <= 0:
                cosine = 0.0
            else:
                dot = np.sum(a[:count] * b[:count], axis=1)
                denom = np.linalg.norm(a[:count], axis=1) * np.linalg.norm(b[:count], axis=1)
                cosine = float(np.mean(dot / np.clip(denom, 1e-8, None)))
            lookup[(model_a, model_b)] = cosine
            lookup[(model_b, model_a)] = cosine
    return lookup


def compare_gcca_stitching(
    *,
    shared_geometry_path: str | Path,
    stitching_cohort_report: str | Path,
) -> dict[str, Any]:
    geometry = SharedLatentGeometry.load(shared_geometry_path)
    cohort = _read_json(stitching_cohort_report)
    summary = cohort.get("summary", {})
    shared_layer = int(summary.get("shared_layer") or max(geometry.layers))
    if shared_layer not in geometry.layers:
        raise ValueError(f"shared_layer={shared_layer} not present in {shared_geometry_path}")
    layer = geometry.layers[shared_layer]
    layer_residuals = {
        model_id: float(layer.metadata.get("view_residuals", {}).get(model_id, float("nan")))
        for model_id in geometry.source_models
    }

    pair_records: list[dict[str, Any]] = []
    cosine_lookup = _pairwise_cosine_lookup(layer)

    for record in cohort.get("pairwise", []):
        if not isinstance(record, dict):
            continue
        source_model_id = str(record.get("source_model_id"))
        target_model_id = str(record.get("target_model_id"))
        pair_records.append(
            {
                "source_model_id": source_model_id,
                "target_model_id": target_model_id,
                "source_residual": layer_residuals.get(source_model_id),
                "target_residual": layer_residuals.get(target_model_id),
                "mean_pair_residual": float(
                    np.nanmean(
                        np.asarray(
                            [layer_residuals.get(source_model_id, np.nan), layer_residuals.get(target_model_id, np.nan)],
                            dtype=np.float32,
                        )
                    )
                ),
                "aligned_latent_cosine_mean": cosine_lookup.get((source_model_id, target_model_id)),
                "eval_target_logit_kl_mean": float(record.get("eval_target_logit_kl_mean", float("nan"))),
                "eval_target_logit_js_mean": float(record.get("eval_target_logit_js_mean", float("nan"))),
                "eval_hidden_cosine_mean": float(record.get("eval_hidden_cosine_mean", float("nan"))),
                "eval_target_top1_agreement": float(record.get("eval_target_top1_agreement", float("nan"))),
                "report_path": str(record.get("report_path", "")),
            }
        )

    def _corr(key_x: str, key_y: str) -> float | None:
        xs: list[float] = []
        ys: list[float] = []
        for record in pair_records:
            x = record.get(key_x)
            y = record.get(key_y)
            if x is None or y is None:
                continue
            xf = float(x)
            yf = float(y)
            if not (np.isfinite(xf) and np.isfinite(yf)):
                continue
            xs.append(xf)
            ys.append(yf)
        if len(xs) < 2:
            return None
        return float(np.corrcoef(np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32))[0, 1])

    return {
        "shared_geometry_path": str(Path(shared_geometry_path).resolve()),
        "stitching_cohort_report": str(Path(stitching_cohort_report).resolve()),
        "shared_layer": shared_layer,
        "layer_view_residuals": layer_residuals,
        "pairwise": pair_records,
        "summary": {
            "pair_count": len(pair_records),
            "corr_aligned_latent_cosine_vs_logit_kl": _corr("aligned_latent_cosine_mean", "eval_target_logit_kl_mean"),
            "corr_aligned_latent_cosine_vs_top1_agreement": _corr("aligned_latent_cosine_mean", "eval_target_top1_agreement"),
            "corr_mean_pair_residual_vs_logit_kl": _corr("mean_pair_residual", "eval_target_logit_kl_mean"),
            "corr_mean_pair_residual_vs_top1_agreement": _corr("mean_pair_residual", "eval_target_top1_agreement"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GCCA shared-geometry diagnostics against stitching cohort transfer metrics.")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--shared-geometry", required=True, help="SharedLatentGeometry artifact")
    parser.add_argument("--stitching-report", required=True, help="Cohort stitching summary JSON")
    args = parser.parse_args()

    report = compare_gcca_stitching(
        shared_geometry_path=args.shared_geometry,
        stitching_cohort_report=args.stitching_report,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
