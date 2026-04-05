#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.compare_model_representations import _load_calibration_lookup
    from tools.representation_learning.schemas import DisagreementProbeSet, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from compare_model_representations import _load_calibration_lookup  # type: ignore[no-redef]
    from schemas import DisagreementProbeSet, ModelRepresentation  # type: ignore[no-redef]


def _chunk_loss_lookup(rep: ModelRepresentation) -> dict[str, float]:
    if rep.chunk_losses is None or rep.chunk_ids is None:
        return {}
    losses = np.asarray(rep.chunk_losses, dtype=np.float32).reshape(-1)
    count = min(len(rep.chunk_ids), int(losses.shape[0]))
    return {str(rep.chunk_ids[idx]): float(losses[idx]) for idx in range(count)}


def _threshold(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    clipped = min(max(float(quantile), 0.0), 1.0)
    return float(np.quantile(np.asarray(values, dtype=np.float32), clipped))


def _probe_type(
    *,
    mean_loss: float,
    disagreement_score: float,
    best_loss: float,
    easy_threshold: float,
    hard_threshold: float,
    confidence_divergence_threshold: float,
) -> str:
    if mean_loss >= hard_threshold:
        return "joint_uncertainty"
    if disagreement_score >= confidence_divergence_threshold and best_loss <= easy_threshold:
        return "confidence_divergence"
    return "directional_divergence"


def mine_pairwise_disagreements(
    representations: list[ModelRepresentation],
    *,
    calibration_lookup: dict[str, dict[str, object]],
    top_k_per_pair: int,
    min_disagreement_score: float,
    easy_loss_quantile: float,
    hard_loss_quantile: float,
    confidence_divergence_threshold: float,
) -> DisagreementProbeSet:
    if len(representations) < 2:
        raise ValueError("need at least two model representations")
    losses_by_model = {rep.model_id: _chunk_loss_lookup(rep) for rep in representations}
    probes: list[dict[str, object]] = []
    pair_summaries: list[dict[str, object]] = []

    for i, rep_a in enumerate(representations):
        for rep_b in representations[i + 1:]:
            losses_a = losses_by_model[rep_a.model_id]
            losses_b = losses_by_model[rep_b.model_id]
            shared_ids = sorted(set(losses_a).intersection(losses_b))
            if not shared_ids:
                continue
            pair_rows: list[dict[str, object]] = []
            mean_losses = []
            disagreement_scores = []
            for chunk_id in shared_ids:
                loss_a = float(losses_a[chunk_id])
                loss_b = float(losses_b[chunk_id])
                mean_loss = 0.5 * (loss_a + loss_b)
                margin = abs(loss_a - loss_b)
                score = float(margin / max(mean_loss, 1e-6))
                mean_losses.append(mean_loss)
                disagreement_scores.append(score)
                pair_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "loss_a": loss_a,
                        "loss_b": loss_b,
                        "mean_loss": mean_loss,
                        "loss_margin": margin,
                        "disagreement_score": score,
                    }
                )
            easy_threshold = _threshold(mean_losses, easy_loss_quantile)
            hard_threshold = _threshold(mean_losses, hard_loss_quantile)

            typed_rows: list[dict[str, object]] = []
            for row in pair_rows:
                chunk_id = str(row["chunk_id"])
                loss_a = float(row["loss_a"])
                loss_b = float(row["loss_b"])
                mean_loss = float(row["mean_loss"])
                disagreement_score = float(row["disagreement_score"])
                if disagreement_score < float(min_disagreement_score) and mean_loss < hard_threshold:
                    continue
                winner_model = rep_a.model_id if loss_a <= loss_b else rep_b.model_id
                loser_model = rep_b.model_id if winner_model == rep_a.model_id else rep_a.model_id
                best_loss = min(loss_a, loss_b)
                probe_type = _probe_type(
                    mean_loss=mean_loss,
                    disagreement_score=disagreement_score,
                    best_loss=best_loss,
                    easy_threshold=easy_threshold,
                    hard_threshold=hard_threshold,
                    confidence_divergence_threshold=confidence_divergence_threshold,
                )
                calibration = calibration_lookup.get(chunk_id, {})
                text_preview = None
                if "text" in calibration:
                    text_preview = " ".join(str(calibration["text"]).split())[:240]
                typed_rows.append(
                    {
                        "probe_id": f"{rep_a.model_id}__{rep_b.model_id}__{chunk_id}",
                        "pair_id": f"{rep_a.model_id}__{rep_b.model_id}",
                        "probe_type": probe_type,
                        "chunk_id": chunk_id,
                        "source_models": [rep_a.model_id, rep_b.model_id],
                        "winner_model": winner_model,
                        "loser_model": loser_model,
                        "loss_by_model": {
                            rep_a.model_id: loss_a,
                            rep_b.model_id: loss_b,
                        },
                        "winner_margin": abs(loss_a - loss_b),
                        "disagreement_score": disagreement_score,
                        "mean_loss": mean_loss,
                        "text_preview": text_preview,
                        "metadata": {
                            key: value
                            for key, value in calibration.items()
                            if key != "text"
                        },
                    }
                )
            typed_rows.sort(
                key=lambda row: (
                    0 if str(row["probe_type"]) == "joint_uncertainty" else 1,
                    float(row["mean_loss"]) if str(row["probe_type"]) == "joint_uncertainty" else float(row["disagreement_score"]),
                    float(row["winner_margin"]),
                    str(row["chunk_id"]),
                ),
                reverse=True,
            )
            kept = typed_rows[: max(int(top_k_per_pair), 0)]
            probes.extend(kept)
            pair_summaries.append(
                {
                    "pair_id": f"{rep_a.model_id}__{rep_b.model_id}",
                    "shared_chunk_count": len(shared_ids),
                    "retained_probe_count": len(kept),
                    "easy_loss_threshold": easy_threshold,
                    "hard_loss_threshold": hard_threshold,
                    "mean_disagreement_score": float(np.mean(disagreement_scores)) if disagreement_scores else None,
                    "probe_type_counts": {
                        probe_type: sum(1 for row in kept if str(row["probe_type"]) == probe_type)
                        for probe_type in ("confidence_divergence", "directional_divergence", "joint_uncertainty")
                    },
                }
            )

    probes.sort(
        key=lambda row: (
            str(row["probe_type"]),
            float(row["disagreement_score"]),
            float(row["winner_margin"]),
            str(row["chunk_id"]),
        ),
        reverse=True,
    )
    return DisagreementProbeSet(
        source_models=[rep.model_id for rep in representations],
        probes=probes,
        metadata={
            "probe_mining_method": "pairwise_chunk_loss_disagreement",
            "top_k_per_pair": int(top_k_per_pair),
            "min_disagreement_score": float(min_disagreement_score),
            "easy_loss_quantile": float(easy_loss_quantile),
            "hard_loss_quantile": float(hard_loss_quantile),
            "confidence_divergence_threshold": float(confidence_divergence_threshold),
            "pair_summaries": pair_summaries,
            "probe_type_counts": {
                probe_type: sum(1 for row in probes if str(row["probe_type"]) == probe_type)
                for probe_type in ("confidence_divergence", "directional_divergence", "joint_uncertainty")
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine typed disagreement probes from extracted model representations.")
    parser.add_argument("output", help="Output .npz path for the disagreement probe set")
    parser.add_argument("representations", nargs="+", help="Input model representation artifacts")
    parser.add_argument("--calibration-jsonl", default=None, help="Optional calibration JSONL for text previews and metadata")
    parser.add_argument("--top-k-per-pair", type=int, default=128)
    parser.add_argument("--min-disagreement-score", type=float, default=0.05)
    parser.add_argument("--easy-loss-quantile", type=float, default=0.3)
    parser.add_argument("--hard-loss-quantile", type=float, default=0.8)
    parser.add_argument("--confidence-divergence-threshold", type=float, default=0.2)
    args = parser.parse_args()

    reps = [ModelRepresentation.load(path) for path in args.representations]
    calibration_lookup = _load_calibration_lookup(args.calibration_jsonl)
    probe_set = mine_pairwise_disagreements(
        reps,
        calibration_lookup=calibration_lookup,
        top_k_per_pair=args.top_k_per_pair,
        min_disagreement_score=args.min_disagreement_score,
        easy_loss_quantile=args.easy_loss_quantile,
        hard_loss_quantile=args.hard_loss_quantile,
        confidence_divergence_threshold=args.confidence_divergence_threshold,
    )
    probe_set.save(args.output)
    summary = {
        "output": str(Path(args.output).resolve()),
        "probe_count": len(probe_set.probes),
        "source_models": probe_set.source_models,
        "probe_type_counts": probe_set.metadata.get("probe_type_counts", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
