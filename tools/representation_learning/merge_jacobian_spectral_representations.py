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
    from tools.representation_learning.merge_weight_spectral_representations import merge_spectral_representations
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from merge_weight_spectral_representations import merge_spectral_representations  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Jacobian-spectral ModelRepresentation artifacts by selecting the strongest aligned direction per cluster.")
    parser.add_argument("output", help="Output PlatonicGeometry .npz path")
    parser.add_argument("representations", nargs="+", help="Input Jacobian-spectral ModelRepresentation .npz files")
    parser.add_argument("--canonical-dim", type=int, default=64, help="Canonical dimension used for cross-model comparison")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of relative-depth bins in the merged geometry")
    parser.add_argument("--top-k", type=int, default=16, help="Directions to retain per merged layer")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="Absolute cosine threshold for clustering source directions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = [ModelRepresentation.load(path) for path in args.representations]
    geometry = merge_spectral_representations(
        representations,
        canonical_dim=args.canonical_dim,
        num_layers=args.num_layers,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
    )
    geometry.metadata["selection_method"] = "jacobian_spectral_argmax"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(output_path)
    print(
        json.dumps(
            {
                "canonical_dim": args.canonical_dim,
                "num_layers": args.num_layers,
                "output": str(output_path),
                "selection_method": geometry.metadata["selection_method"],
                "similarity_threshold": args.similarity_threshold,
                "source_models": geometry.source_models,
                "top_k": args.top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
