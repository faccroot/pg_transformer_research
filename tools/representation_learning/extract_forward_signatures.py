#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.extract_model_representation import batched, parse_layers, read_calibration_records
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import ForwardSignatureDataset
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from extract_model_representation import batched, parse_layers, read_calibration_records  # type: ignore[no-redef]
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import ForwardSignatureDataset  # type: ignore[no-redef]


def extract_forward_signatures(
    records: list[dict[str, object]],
    *,
    adapter: HFCausalLMAdapter,
    calibration_jsonl: str | Path,
    layers: list[int],
    batch_size: int,
    max_length: int,
    top_k: int,
    torch_dtype: str,
) -> ForwardSignatureDataset:
    chunk_ids = [str(record["chunk_id"]) for record in records]
    global_parts: dict[str, list[np.ndarray]] = {}
    layer_parts: dict[int, dict[str, list[np.ndarray]]] = {int(layer_idx): {} for layer_idx in layers}
    topk_ids_parts: list[np.ndarray] = []
    topk_probs_parts: list[np.ndarray] = []
    for batch in batched(records, batch_size):
        texts = [str(record["text"]) for record in batch]
        stats = adapter.get_forward_signatures(texts, layers=layers, max_length=max_length, top_k=top_k)
        for name, values in stats.global_features.items():
            global_parts.setdefault(str(name), []).append(np.asarray(values, dtype=np.float32))
        for layer_idx, payload in stats.layer_features.items():
            for name, values in payload.items():
                layer_parts[int(layer_idx)].setdefault(str(name), []).append(np.asarray(values, dtype=np.float32))
        topk_ids_parts.append(np.asarray(stats.topk_token_ids, dtype=np.int32))
        topk_probs_parts.append(np.asarray(stats.topk_token_probs, dtype=np.float32))
    global_features = {
        name: np.concatenate(parts, axis=0).astype(np.float32)
        for name, parts in global_parts.items()
    }
    layer_features = {
        int(layer_idx): {
            name: np.concatenate(parts, axis=0).astype(np.float32)
            for name, parts in payload.items()
        }
        for layer_idx, payload in layer_parts.items()
    }
    return ForwardSignatureDataset(
        model_id=adapter.model_id,
        chunk_ids=chunk_ids,
        top_k=top_k,
        global_features=global_features,
        layer_features=layer_features,
        topk_token_ids=np.concatenate(topk_ids_parts, axis=0).astype(np.int32),
        topk_token_probs=np.concatenate(topk_probs_parts, axis=0).astype(np.float32),
        metadata={
            "batch_size": batch_size,
            "calibration_jsonl": str(Path(calibration_jsonl).resolve()),
            "device": adapter.device,
            "layers": layers,
            "max_examples": len(records),
            "max_length": max_length,
            "top_k": top_k,
            "torch_dtype": torch_dtype,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract forward-pass attention and top-k signature sidecars for ecology training.")
    parser.add_argument("model_id")
    parser.add_argument("calibration_jsonl")
    parser.add_argument("output")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k last-token distribution to retain")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layers", default="", help="Comma-separated layers; default is all hidden layers")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = read_calibration_records(args.calibration_jsonl, max_examples=args.max_examples)
    adapter = HFCausalLMAdapter(
        args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
    layers = parse_layers(args.layers, adapter.num_layers)
    dataset = extract_forward_signatures(
        records,
        adapter=adapter,
        calibration_jsonl=args.calibration_jsonl,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.top_k,
        torch_dtype=args.torch_dtype,
    )
    output_path = Path(args.output).resolve()
    dataset.save(output_path)
    summary = {
        "model_id": args.model_id,
        "output": str(output_path),
        "chunk_count": len(dataset.chunk_ids),
        "top_k": dataset.top_k,
        "global_features": sorted(dataset.global_features),
        "layer_indices": sorted(dataset.layer_features),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
