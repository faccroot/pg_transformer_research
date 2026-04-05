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
    from tools.representation_learning.build_routing_kernel import build_routing_kernel
    from tools.representation_learning.compare_model_representations import _load_calibration_lookup
    from tools.representation_learning.mine_model_disagreements import mine_pairwise_disagreements
    from tools.representation_learning.schemas import ModelRepresentation
    from tools.representation_learning.verify_probe_outcomes import verify_probe_set
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from build_routing_kernel import build_routing_kernel  # type: ignore[no-redef]
    from compare_model_representations import _load_calibration_lookup  # type: ignore[no-redef]
    from mine_model_disagreements import mine_pairwise_disagreements  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]
    from verify_probe_outcomes import verify_probe_set  # type: ignore[no-redef]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the typed disagreement -> verification -> routing-kernel bootstrap pipeline.")
    parser.add_argument("output_dir", help="Directory for the generated probe set, verification table, and routing kernel")
    parser.add_argument("representations", nargs="+", help="Input model representation artifacts")
    parser.add_argument("--calibration-jsonl", default=None)
    parser.add_argument("--probe-top-k-per-pair", type=int, default=96)
    parser.add_argument("--probe-min-disagreement-score", type=float, default=0.05)
    parser.add_argument("--probe-easy-loss-quantile", type=float, default=0.3)
    parser.add_argument("--probe-hard-loss-quantile", type=float, default=0.8)
    parser.add_argument("--probe-confidence-divergence-threshold", type=float, default=0.2)
    parser.add_argument("--verification-min-confidence", type=float, default=0.05)
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=1)
    args = parser.parse_args()

    reps = [ModelRepresentation.load(path) for path in args.representations]
    calibration_lookup = _load_calibration_lookup(args.calibration_jsonl)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_set = mine_pairwise_disagreements(
        reps,
        calibration_lookup=calibration_lookup,
        top_k_per_pair=args.probe_top_k_per_pair,
        min_disagreement_score=args.probe_min_disagreement_score,
        easy_loss_quantile=args.probe_easy_loss_quantile,
        hard_loss_quantile=args.probe_hard_loss_quantile,
        confidence_divergence_threshold=args.probe_confidence_divergence_threshold,
    )
    probe_path = output_dir / "disagreement_probe_set.npz"
    probe_set.save(probe_path)

    verification = verify_probe_set(
        probe_set,
        min_verification_confidence=args.verification_min_confidence,
    )
    verification_path = output_dir / "verification_table.npz"
    verification.save(verification_path)

    routing_kernel = build_routing_kernel(
        reps,
        verification,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        similarity_threshold=args.similarity_threshold,
        min_support=args.min_support,
    )
    kernel_path = output_dir / "routing_kernel.npz"
    routing_kernel.save(kernel_path)

    summary = {
        "output_dir": str(output_dir),
        "probe_path": str(probe_path),
        "verification_path": str(verification_path),
        "routing_kernel_path": str(kernel_path),
        "probe_count": len(probe_set.probes),
        "verified_entry_count": len(verification.entries),
        "routing_rule_count": len(routing_kernel.rules),
        "source_models": routing_kernel.source_models,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
