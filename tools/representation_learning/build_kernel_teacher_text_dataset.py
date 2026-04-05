#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.schemas import KernelTeacherDataset, KernelTeacherTextDataset
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from schemas import KernelTeacherDataset, KernelTeacherTextDataset  # type: ignore[no-redef]


def _load_calibration_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunk_id = str(payload.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            rows[chunk_id] = payload
    return rows


def build_kernel_teacher_text_dataset(
    *,
    output_path: str | Path,
    kernel_teacher_dataset: str | Path,
    calibration_jsonl: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    teacher_path = Path(kernel_teacher_dataset).resolve()
    calibration_path = Path(calibration_jsonl).resolve()
    teacher = KernelTeacherDataset.load(teacher_path)
    calibration_rows = _load_calibration_rows(calibration_path)

    text_examples: list[dict[str, Any]] = []
    missing_chunk_ids: list[str] = []
    for example in teacher.examples:
        chunk_id = str(example.get("chunk_id", "")).strip()
        calibration = calibration_rows.get(chunk_id)
        if calibration is None:
            missing_chunk_ids.append(chunk_id)
            if strict:
                continue
            text = ""
            calibration_metadata: dict[str, Any] = {}
        else:
            text = str(calibration.get("text", ""))
            calibration_metadata = {
                key: value
                for key, value in calibration.items()
                if key not in {"text"}
            }
        payload = dict(example)
        payload["text"] = text
        payload["calibration_metadata"] = calibration_metadata
        text_examples.append(payload)

    if strict and missing_chunk_ids:
        raise ValueError(
            "Missing calibration rows for kernel-teacher chunk ids: "
            f"{sorted(set(missing_chunk_ids))[:8]}"
        )

    artifact = KernelTeacherTextDataset(
        source_models=list(teacher.source_models),
        embedding_dim=int(teacher.embedding_dim),
        examples=text_examples,
        metadata={
            **teacher.metadata,
            "builder": "build_kernel_teacher_text_dataset",
            "kernel_teacher_dataset": str(teacher_path),
            "calibration_jsonl": str(calibration_path),
            "example_count": int(len(text_examples)),
            "missing_chunk_ids": sorted(set(missing_chunk_ids)),
            "strict": bool(strict),
        },
    )
    artifact.save(output_path)

    summary = {
        "kernel_teacher_dataset": str(teacher_path),
        "calibration_jsonl": str(calibration_path),
        "example_count": int(len(text_examples)),
        "embedding_dim": int(artifact.embedding_dim),
        "source_models": list(artifact.source_models),
        "missing_chunk_ids": sorted(set(missing_chunk_ids)),
        "strict": bool(strict),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Join a KernelTeacherDataset with calibration text to produce a text-conditioned student dataset.")
    parser.add_argument("output", help="Output .npz path for the KernelTeacherTextDataset artifact")
    parser.add_argument("--kernel-teacher-dataset", required=True, help="Input KernelTeacherDataset artifact")
    parser.add_argument("--calibration-jsonl", required=True, help="Calibration JSONL containing chunk_id/text rows")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing chunk ids and emit empty text rows instead of failing",
    )
    args = parser.parse_args()

    summary = build_kernel_teacher_text_dataset(
        output_path=args.output,
        kernel_teacher_dataset=args.kernel_teacher_dataset,
        calibration_jsonl=args.calibration_jsonl,
        strict=not args.allow_missing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
