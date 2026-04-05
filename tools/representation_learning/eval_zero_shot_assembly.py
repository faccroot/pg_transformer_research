#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.assemble_reasoning_core import ReasoningCoreAssembler
    from tools.representation_learning.schemas import ModelRepresentation, PlatonicGeometry
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from assemble_reasoning_core import ReasoningCoreAssembler  # type: ignore[no-redef]
    from schemas import ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int64, copy=False)


def load_geometry(path: str | Path):
    payload = np.load(Path(path), allow_pickle=False)
    kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
    if kind == "platonic_geometry":
        return PlatonicGeometry.load(path)
    if kind == "model_representation":
        return ModelRepresentation.load(path)
    raise ValueError(f"Unsupported geometry kind {kind!r} in {path}")


def iter_eval_batches(files: list[Path], seq_len: int, batch_size: int, num_batches: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    current: list[np.ndarray] = []
    for path in files:
        tokens = load_data_shard(path)
        usable = (tokens.shape[0] // seq_len) * seq_len
        if usable <= 0:
            continue
        chunks = tokens[:usable].reshape(-1, seq_len)
        for chunk in chunks:
            current.append(chunk)
            if len(current) >= batch_size:
                batches.append(torch.tensor(np.stack(current, axis=0), dtype=torch.long))
                current = []
                if len(batches) >= num_batches:
                    return batches
    if current and len(batches) < num_batches:
        batches.append(torch.tensor(np.stack(current, axis=0), dtype=torch.long))
    return batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a first zero-shot assembly probe from an extracted geometry artifact.")
    parser.add_argument("geometry_path", help="Path to a ModelRepresentation or PlatonicGeometry .npz artifact")
    parser.add_argument("input_glob", help="Shard glob used for zero-shot evaluation")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for the zero-shot probe")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the zero-shot probe")
    parser.add_argument("--num-batches", type=int, default=8, help="Number of batches to score")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Assembled model hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Assembled model layer count")
    parser.add_argument("--num-heads", type=int, default=8, help="Assembled model attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=2.0, help="Assembled model MLP ratio")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Assembled model vocabulary size")
    parser.add_argument("--seed", type=int, default=17, help="Assembly seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geometry = load_geometry(args.geometry_path)
    assembler = ReasoningCoreAssembler(geometry)
    model = assembler.assemble(
        target_hidden_dim=args.hidden_dim,
        target_num_layers=args.num_layers,
        target_vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.seq_len,
        seed=args.seed,
    )
    files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")
    batches = iter_eval_batches(files, seq_len=args.seq_len, batch_size=args.batch_size, num_batches=args.num_batches)
    if not batches:
        raise RuntimeError("No evaluation batches available for the requested zero-shot probe")
    losses = [assembler.evaluate_zero_shot(model, batch) for batch in batches]
    bits_per_token = [float(loss / np.log(2.0)) for loss in losses]
    summary = {
        "batch_size": args.batch_size,
        "geometry_path": str(Path(args.geometry_path).resolve()),
        "hidden_dim": args.hidden_dim,
        "input_glob": args.input_glob,
        "mean_loss_nats": float(np.mean(losses)),
        "mean_bits_per_token": float(np.mean(bits_per_token)),
        "num_batches": len(losses),
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
