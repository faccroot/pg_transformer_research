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
    from tools.representation_learning.merge_hybrid_cascade_representations import merge_hybrid_cascade_representations
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from merge_hybrid_cascade_representations import merge_hybrid_cascade_representations  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def merge_hardmax_hybrid_representations(
    representations: list[ModelRepresentation],
    *,
    canonical_dim: int,
    num_layers: int,
    top_k: int,
    similarity_threshold: float,
    novelty_power: float = 1.5,
    unique_bonus: float = 1.0,
    shared_support_bonus: float = 0.1,
    margin_power: float = 1.5,
):
    return merge_hybrid_cascade_representations(
        representations,
        canonical_dim=canonical_dim,
        num_layers=num_layers,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        novelty_power=novelty_power,
        unique_bonus=unique_bonus,
        shared_support_bonus=shared_support_bonus,
        margin_power=margin_power,
        selection_method="hardmax_cascade_guided_spectral_merge",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ModelRepresentation artifacts with a harder winner-take-most hybrid that rewards decisive cluster champions.",
    )
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--novelty-power", type=float, default=1.5)
    parser.add_argument("--unique-bonus", type=float, default=1.0)
    parser.add_argument("--shared-support-bonus", type=float, default=0.1)
    parser.add_argument("--margin-power", type=float, default=1.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_hardmax_hybrid_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        novelty_power=args.novelty_power,
        unique_bonus=args.unique_bonus,
        shared_support_bonus=args.shared_support_bonus,
        margin_power=args.margin_power,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "selection_method": geometry.metadata["selection_method"],
                "model_order": geometry.metadata["model_order"],
                "source_models": geometry.source_models,
                "top_k": geometry.metadata["top_k"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
