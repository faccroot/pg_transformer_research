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
    from tools.representation_learning.schemas import DisagreementProbeSet, VerificationTable
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import DisagreementProbeSet, VerificationTable  # type: ignore[no-redef]


def verify_probe_set(
    probe_set: DisagreementProbeSet,
    *,
    min_verification_confidence: float,
) -> VerificationTable:
    entries: list[dict[str, object]] = []
    for probe in probe_set.probes:
        losses = probe.get("loss_by_model", {})
        if not isinstance(losses, dict) or not losses:
            continue
        ordered = sorted(
            ((str(model_id), float(loss)) for model_id, loss in losses.items()),
            key=lambda item: (item[1], item[0]),
        )
        winner_model, winner_loss = ordered[0]
        runner_up_model, runner_up_loss = ordered[1] if len(ordered) > 1 else (winner_model, winner_loss)
        confidence = float((runner_up_loss - winner_loss) / max(0.5 * (runner_up_loss + winner_loss), 1e-6))
        verified = confidence >= float(min_verification_confidence) or str(probe.get("probe_type")) == "joint_uncertainty"
        entries.append(
            {
                "probe_id": str(probe["probe_id"]),
                "chunk_id": str(probe["chunk_id"]),
                "probe_type": str(probe.get("probe_type", "unknown")),
                "verified_winner_model": winner_model,
                "runner_up_model": runner_up_model,
                "verification_method": "heldout_chunk_loss",
                "verification_confidence": confidence,
                "verified": bool(verified),
                "winner_loss": winner_loss,
                "runner_up_loss": runner_up_loss,
                "loss_by_model": losses,
                "source_models": list(probe.get("source_models", [])),
                "text_preview": probe.get("text_preview"),
                "metadata": dict(probe.get("metadata", {})) if isinstance(probe.get("metadata"), dict) else {},
            }
        )
    return VerificationTable(
        source_models=list(probe_set.source_models),
        entries=entries,
        metadata={
            "verification_method": "heldout_chunk_loss",
            "min_verification_confidence": float(min_verification_confidence),
            "verified_count": sum(1 for entry in entries if bool(entry["verified"])),
            "unverified_count": sum(1 for entry in entries if not bool(entry["verified"])),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify disagreement probes using a deterministic held-out chunk-loss rule.")
    parser.add_argument("output", help="Output .npz path for the verification table")
    parser.add_argument("probe_set", help="Input disagreement probe set artifact")
    parser.add_argument("--min-verification-confidence", type=float, default=0.05)
    args = parser.parse_args()

    probe_set = DisagreementProbeSet.load(args.probe_set)
    table = verify_probe_set(
        probe_set,
        min_verification_confidence=args.min_verification_confidence,
    )
    table.save(args.output)
    summary = {
        "output": str(Path(args.output).resolve()),
        "entry_count": len(table.entries),
        "verified_count": table.metadata.get("verified_count", 0),
        "unverified_count": table.metadata.get("unverified_count", 0),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
