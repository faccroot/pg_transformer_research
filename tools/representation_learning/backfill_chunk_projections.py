#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import ModelRepresentation  # type: ignore[no-redef]


def read_calibration_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or "text" not in payload or "chunk_id" not in payload:
                raise ValueError(f"Malformed calibration record: {line[:120]!r}")
            records.append(payload)
    if not records:
        raise ValueError(f"No calibration records found in {path}")
    return records


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[offset: offset + batch_size] for offset in range(0, len(items), batch_size)]


def candidate_calibration_paths(
    artifact_path: Path,
    representation: ModelRepresentation,
    explicit_path: str | Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    raw_metadata_path = str(representation.metadata.get("calibration_jsonl", "")).strip()
    if raw_metadata_path:
        metadata_path = Path(raw_metadata_path).expanduser()
        candidates.append(metadata_path)
        candidates.append(artifact_path.parent / metadata_path.name)
        candidates.append(ROOT / "research" / "representation_learning" / "calibration" / metadata_path.name)
    candidates.append(artifact_path.parent / "calibration.jsonl")
    candidates.append(ROOT / "research" / "representation_learning" / "calibration" / "calibration.jsonl")
    candidates.append(ROOT / "research" / "representation_learning" / "calibration" / "compare_v1_calibration.jsonl")
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_calibration_path(
    artifact_path: Path,
    representation: ModelRepresentation,
    explicit_path: str | Path | None,
) -> Path:
    candidates = candidate_calibration_paths(artifact_path, representation, explicit_path)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Could not locate calibration JSONL. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def select_records(records: list[dict[str, Any]], chunk_ids: list[str]) -> list[dict[str, Any]]:
    by_chunk_id = {str(record["chunk_id"]): record for record in records}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for chunk_id in chunk_ids:
        record = by_chunk_id.get(str(chunk_id))
        if record is None:
            missing.append(str(chunk_id))
        else:
            selected.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Calibration records missing {len(missing)} chunk ids, including: {preview}")
    return selected


def compute_chunk_projections(
    representation: ModelRepresentation,
    *,
    adapter: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layers: list[int] | None = None,
    overwrite: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    target_layers = sorted(int(layer_idx) for layer_idx in (layers or list(representation.layer_geometries)))
    projection_parts: dict[int, list[np.ndarray]] = {layer_idx: [] for layer_idx in target_layers}
    reused_layers: list[int] = []
    computed_layers: list[int] = []
    chunk_count = len(records)
    for layer_idx in target_layers:
        existing = representation.chunk_layer_projections.get(int(layer_idx))
        if existing is not None and not overwrite:
            if existing.shape[0] != chunk_count:
                raise ValueError(
                    f"Existing projections for layer {layer_idx} have {existing.shape[0]} rows, expected {chunk_count}"
                )
            projection_parts[layer_idx].append(np.asarray(existing, dtype=np.float32))
            reused_layers.append(layer_idx)

    pending_layers = [layer_idx for layer_idx in target_layers if layer_idx not in reused_layers]
    for batch in batched(records, max(int(batch_size), 1)):
        texts = [str(record["text"]) for record in batch]
        pooled_states = adapter.get_mean_pooled_hidden_states(texts, layers=pending_layers, max_length=max_length)
        for layer_idx in pending_layers:
            directions = np.asarray(representation.layer_geometries[int(layer_idx)].directions, dtype=np.float32)
            pooled = np.asarray(pooled_states[int(layer_idx)], dtype=np.float32)
            projection_parts[layer_idx].append((pooled @ directions.T).astype(np.float32))
    computed_layers.extend(pending_layers)

    updated = dict(representation.chunk_layer_projections)
    for layer_idx in target_layers:
        parts = projection_parts[int(layer_idx)]
        if not parts:
            continue
        combined = np.concatenate(parts, axis=0).astype(np.float32)
        if combined.shape[0] != chunk_count:
            raise ValueError(
                f"Computed projections for layer {layer_idx} have {combined.shape[0]} rows, expected {chunk_count}"
            )
        updated[int(layer_idx)] = combined

    summary = {
        "chunk_count": chunk_count,
        "computed_layers": computed_layers,
        "reused_layers": reused_layers,
        "target_layers": target_layers,
    }
    return updated, summary


def backfill_representation(
    artifact_path: str | Path,
    *,
    output_path: str | Path | None = None,
    calibration_jsonl: str | Path | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    layers: list[int] | None = None,
    overwrite: bool = False,
    device: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[ModelRepresentation, dict[str, Any]]:
    artifact_path = Path(artifact_path).resolve()
    representation = ModelRepresentation.load(artifact_path)
    resolved_calibration_path = resolve_calibration_path(artifact_path, representation, calibration_jsonl)
    calibration_records = read_calibration_records(resolved_calibration_path)
    selected_records = select_records(calibration_records, representation.chunk_ids or [])
    effective_batch_size = int(batch_size or representation.metadata.get("batch_size", 1) or 1)
    effective_max_length = int(max_length or representation.metadata.get("max_length", 256) or 256)
    adapter = HFCausalLMAdapter(
        representation.model_id,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    updated_projections, projection_summary = compute_chunk_projections(
        representation,
        adapter=adapter,
        records=selected_records,
        batch_size=effective_batch_size,
        max_length=effective_max_length,
        layers=layers,
        overwrite=overwrite,
    )
    updated_metadata = dict(representation.metadata)
    updated_metadata["chunk_projection_backfill"] = {
        "calibration_jsonl": str(resolved_calibration_path),
        "batch_size": effective_batch_size,
        "max_length": effective_max_length,
        "layers": projection_summary["target_layers"],
        "overwrite": bool(overwrite),
    }
    updated = ModelRepresentation(
        model_id=representation.model_id,
        architecture_family=representation.architecture_family,
        num_parameters=representation.num_parameters,
        hidden_dim=representation.hidden_dim,
        num_layers=representation.num_layers,
        layer_geometries=representation.layer_geometries,
        chunk_losses=representation.chunk_losses,
        chunk_ids=representation.chunk_ids,
        chunk_layer_projections=updated_projections,
        concept_profiles=representation.concept_profiles,
        metadata=updated_metadata,
    )
    destination = Path(output_path).resolve() if output_path is not None else artifact_path
    updated.save(destination)
    summary = {
        "artifact_path": str(artifact_path),
        "output_path": str(destination),
        "calibration_jsonl": str(resolved_calibration_path),
        "computed_layers": projection_summary["computed_layers"],
        "reused_layers": projection_summary["reused_layers"],
        "chunk_count": projection_summary["chunk_count"],
        "num_projection_layers": len(updated.chunk_layer_projections),
        "projection_layers": sorted(updated.chunk_layer_projections),
    }
    return updated, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill missing per-chunk layer projections in a ModelRepresentation artifact.")
    parser.add_argument("artifact", help="Input model_representation.npz")
    parser.add_argument("--output", default="", help="Output path; defaults to overwriting the input artifact")
    parser.add_argument("--calibration-jsonl", default="", help="Optional local calibration JSONL override")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size for pooling")
    parser.add_argument("--max-length", type=int, default=0, help="Override tokenizer truncation length")
    parser.add_argument("--layers", default="", help="Optional comma-separated subset of layers to backfill")
    parser.add_argument("--overwrite", action="store_true", help="Recompute projections even if they already exist")
    parser.add_argument("--device", default="auto", help="Device passed to the Hugging Face adapter")
    parser.add_argument("--torch-dtype", default="auto", help="torch dtype passed to from_pretrained")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to transformers.from_pretrained")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary output path")
    return parser


def parse_layers(spec: str) -> list[int] | None:
    raw = [part.strip() for part in spec.split(",") if part.strip()]
    if not raw:
        return None
    return sorted({int(part) for part in raw})


def main() -> None:
    args = build_parser().parse_args()
    _, summary = backfill_representation(
        args.artifact,
        output_path=args.output or None,
        calibration_jsonl=args.calibration_jsonl or None,
        batch_size=args.batch_size or None,
        max_length=args.max_length or None,
        layers=parse_layers(args.layers),
        overwrite=args.overwrite,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
