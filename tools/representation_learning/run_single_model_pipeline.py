#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.representation_learning.assemble_reasoning_core import ReasoningCoreAssembler
    from tools.representation_learning.build_platonic_geometry import build_platonic_geometry
    from tools.representation_learning.calibration_set import build_records, load_operator_ids
    from tools.representation_learning.eval_zero_shot_assembly import iter_eval_batches
    from tools.representation_learning.extract_model_representation import (
        build_model_representation,
        chunk_projection_summary,
        load_concept_probe_specs,
        parse_layers,
        read_calibration_records,
        require_chunk_projection_coverage,
    )
    from tools.representation_learning.model_adapter import HFCausalLMAdapter
    from tools.representation_learning.schemas import LayerGeometry, ModelRepresentation
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from assemble_reasoning_core import ReasoningCoreAssembler  # type: ignore[no-redef]
    from build_platonic_geometry import build_platonic_geometry  # type: ignore[no-redef]
    from calibration_set import build_records, load_operator_ids  # type: ignore[no-redef]
    from eval_zero_shot_assembly import iter_eval_batches  # type: ignore[no-redef]
    from extract_model_representation import (  # type: ignore[no-redef]
        build_model_representation,
        chunk_projection_summary,
        load_concept_probe_specs,
        parse_layers,
        read_calibration_records,
        require_chunk_projection_coverage,
    )
    from model_adapter import HFCausalLMAdapter  # type: ignore[no-redef]
    from schemas import LayerGeometry, ModelRepresentation  # type: ignore[no-redef]


def write_calibration_jsonl(records, summary: dict[str, object], output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, summary_path


def extract_representation(
    *,
    model_id: str,
    calibration_jsonl: Path,
    output_path: Path,
    top_k: int,
    batch_size: int,
    max_examples: int,
    max_length: int,
    layers_spec: str,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
    concept_probes: str,
    probe_batch_size: int,
    require_chunk_projections: bool,
) -> ModelRepresentation:
    records = read_calibration_records(calibration_jsonl, max_examples=max_examples)
    print(
        f"[rep] extracting model_id={model_id} examples={len(records)} batch_size={batch_size} "
        f"max_length={max_length} device={device} dtype={torch_dtype}",
        flush=True,
    )
    adapter = HFCausalLMAdapter(
        model_id,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    layers = parse_layers(layers_spec, adapter.num_layers)
    print(
        f"[rep] model loaded architecture={adapter.architecture_family} "
        f"num_layers={adapter.num_layers} hidden_dim={adapter.hidden_dim} params={adapter.num_parameters}",
        flush=True,
    )
    print(f"[rep] extracting layers={layers}", flush=True)
    concept_probe_specs = load_concept_probe_specs(concept_probes)
    representation = build_model_representation(
        records,
        adapter=adapter,
        calibration_jsonl=calibration_jsonl,
        layers=layers,
        top_k=top_k,
        batch_size=batch_size,
        max_length=max_length,
        torch_dtype=torch_dtype,
        concept_probe_specs=concept_probe_specs,
        concept_probe_batch_size=probe_batch_size,
    )
    if require_chunk_projections:
        require_chunk_projection_coverage(
            chunk_ids=representation.chunk_ids,
            chunk_layer_projections=representation.chunk_layer_projections,
            requested_layers=layers,
        )
    representation.save(output_path)
    return representation


def summarize_representation(rep: ModelRepresentation) -> dict[str, object]:
    layer_summaries: list[dict[str, object]] = []
    for layer_idx in sorted(rep.layer_geometries):
        layer = rep.layer_geometries[layer_idx]
        direction_norms = np.linalg.norm(layer.directions, axis=1)
        scales = np.asarray(layer.scales if layer.scales is not None else np.ones((layer.directions.shape[0],), dtype=np.float32))
        covariance = layer.covariance if layer.covariance is not None else np.zeros((0, 0), dtype=np.float32)
        coactivation = layer.coactivation if layer.coactivation is not None else np.zeros((0, 0), dtype=np.float32)
        symmetry_error = 0.0
        if coactivation.size > 0:
            symmetry_error = float(np.max(np.abs(coactivation - coactivation.T)))
        layer_summaries.append(
            {
                "layer_idx": layer_idx,
                "relative_depth": layer.relative_depth,
                "num_directions": int(layer.directions.shape[0]),
                "direction_norm_mean": float(direction_norms.mean()) if direction_norms.size else 0.0,
                "direction_norm_min": float(direction_norms.min()) if direction_norms.size else 0.0,
                "direction_norm_max": float(direction_norms.max()) if direction_norms.size else 0.0,
                "scale_min": float(scales.min()) if scales.size else 0.0,
                "scale_max": float(scales.max()) if scales.size else 0.0,
                "scale_mean": float(scales.mean()) if scales.size else 0.0,
                "covariance_trace": float(np.trace(covariance)) if covariance.size else 0.0,
                "coactivation_symmetry_error": symmetry_error,
                "has_nan": bool(np.isnan(layer.directions).any() or np.isnan(scales).any() or np.isnan(covariance).any()),
            }
        )
    chunk_loss_summary = None
    if rep.chunk_losses is not None and rep.chunk_losses.size > 0:
        chunk_loss_summary = {
            "min": float(np.min(rep.chunk_losses)),
            "mean": float(np.mean(rep.chunk_losses)),
            "max": float(np.max(rep.chunk_losses)),
            "std": float(np.std(rep.chunk_losses)),
        }
    return {
        "model_id": rep.model_id,
        "architecture_family": rep.architecture_family,
        "num_layers": rep.num_layers,
        "hidden_dim": rep.hidden_dim,
        "num_parameters": rep.num_parameters,
        "chunk_loss_summary": chunk_loss_summary,
        "chunk_projection_summary": chunk_projection_summary(
            chunk_ids=rep.chunk_ids,
            chunk_layer_projections=rep.chunk_layer_projections,
            requested_layers=list(rep.metadata.get("layers", sorted(rep.layer_geometries))),
        ),
        "layers": layer_summaries,
        "concept_profiles": {
            concept_name: {
                "sharpness": float(profile.get("sharpness", 0.0)),
                "num_pairs": int(profile.get("num_pairs", 0)),
                "best_layer": max(
                    (
                        (int(layer_idx), float(layer_payload.get("layer_score", 0.0)))
                        for layer_idx, layer_payload in dict(profile.get("layers", {})).items()
                    ),
                    key=lambda item: item[1],
                    default=(None, 0.0),
                )[0],
            }
            for concept_name, profile in sorted(rep.concept_profiles.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the single-model representation-learning pipeline end to end.")
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen3-4B")
    parser.add_argument("input_glob", help="Training shard glob, for example data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("--work-dir", default=str(ROOT / "research" / "representation_learning" / "runs" / "single_model"), help="Output directory")
    parser.add_argument("--tokenizer-path", default=str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--histogram-bins", type=int, default=256)
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=50000)
    parser.add_argument("--zlib-level", type=int, default=6)
    parser.add_argument("--operator-token-json", default="")
    parser.add_argument("--operator-token-ids", default="")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--extract-max-examples", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layers", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--concept-probes", default="default")
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--require-chunk-projections", action="store_true")
    parser.add_argument("--canonical-dim", type=int, default=64)
    parser.add_argument("--combined-layers", type=int, default=12)
    parser.add_argument("--zero-shot-seq-len", type=int, default=128)
    parser.add_argument("--zero-shot-batch-size", type=int, default=4)
    parser.add_argument("--zero-shot-num-batches", type=int, default=8)
    parser.add_argument("--assembled-hidden-dim", type=int, default=512)
    parser.add_argument("--assembled-num-layers", type=int, default=6)
    parser.add_argument("--assembled-num-heads", type=int, default=8)
    parser.add_argument("--assembled-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--assembled-vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    input_glob = str(Path(args.input_glob).expanduser())
    tokenizer_path = str(Path(args.tokenizer_path).expanduser())
    work_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = work_dir / "calibration.jsonl"
    representation_path = work_dir / "model_representation.npz"
    representation_summary_path = work_dir / "model_representation.summary.json"
    geometry_path = work_dir / "platonic_geometry.npz"
    zero_shot_summary_path = work_dir / "zero_shot_summary.json"
    pipeline_summary_path = work_dir / "pipeline_summary.json"

    t0 = time.perf_counter()
    print(f"[rep] work_dir={work_dir}", flush=True)
    operator_ids = load_operator_ids(args.operator_token_ids, args.operator_token_json)
    records, calibration_summary = build_records(
        input_glob=input_glob,
        tokenizer_path=tokenizer_path,
        chunk_size=args.chunk_size,
        histogram_bins=args.histogram_bins,
        num_clusters=args.num_clusters,
        kmeans_iterations=args.kmeans_iterations,
        sample_size=args.sample_size,
        max_shards=args.max_shards,
        max_chunks=args.max_chunks,
        zlib_level=args.zlib_level,
        operator_ids=operator_ids,
    )
    calibration_path, calibration_summary_path = write_calibration_jsonl(records, calibration_summary, calibration_path)
    print(
        f"[rep] calibration built records={len(records)} path={calibration_path} "
        f"seconds={time.perf_counter() - t0:.1f}",
        flush=True,
    )

    t1 = time.perf_counter()
    representation = extract_representation(
        model_id=args.model_id,
        calibration_jsonl=calibration_path,
        output_path=representation_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_examples=args.extract_max_examples,
        max_length=args.max_length,
        layers_spec=args.layers,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        concept_probes=args.concept_probes,
        probe_batch_size=args.probe_batch_size,
        require_chunk_projections=args.require_chunk_projections,
    )
    representation_summary = summarize_representation(representation)
    representation_summary_path.write_text(json.dumps(representation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[rep] representation saved path={representation_path} "
        f"seconds={time.perf_counter() - t1:.1f}",
        flush=True,
    )

    t2 = time.perf_counter()
    geometry = build_platonic_geometry(
        [representation],
        canonical_dim=args.canonical_dim,
        num_layers=args.combined_layers,
        top_k=args.top_k,
    )
    geometry.save(geometry_path)
    print(
        f"[rep] geometry saved path={geometry_path} "
        f"seconds={time.perf_counter() - t2:.1f}",
        flush=True,
    )

    t3 = time.perf_counter()
    assembler = ReasoningCoreAssembler(geometry)
    assembled = assembler.assemble(
        target_hidden_dim=args.assembled_hidden_dim,
        target_num_layers=args.assembled_num_layers,
        target_vocab_size=args.assembled_vocab_size,
        num_heads=args.assembled_num_heads,
        mlp_ratio=args.assembled_mlp_ratio,
        max_seq_len=args.zero_shot_seq_len,
        seed=args.seed,
    )
    eval_files = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not eval_files:
        raise FileNotFoundError(f"No files matched for zero-shot evaluation: {input_glob}")
    batches = iter_eval_batches(
        eval_files,
        seq_len=args.zero_shot_seq_len,
        batch_size=args.zero_shot_batch_size,
        num_batches=args.zero_shot_num_batches,
    )
    if not batches:
        raise RuntimeError("No zero-shot evaluation batches were produced")
    zero_shot_losses = [assembler.evaluate_zero_shot(assembled, batch) for batch in batches]
    zero_shot_bits = [float(loss / np.log(2.0)) for loss in zero_shot_losses]
    zero_shot_summary = {
        "mean_loss_nats": float(np.mean(zero_shot_losses)),
        "min_loss_nats": float(np.min(zero_shot_losses)),
        "max_loss_nats": float(np.max(zero_shot_losses)),
        "mean_bits_per_token": float(np.mean(zero_shot_bits)),
        "min_bits_per_token": float(np.min(zero_shot_bits)),
        "max_bits_per_token": float(np.max(zero_shot_bits)),
        "num_batches": len(zero_shot_losses),
        "seq_len": args.zero_shot_seq_len,
        "batch_size": args.zero_shot_batch_size,
    }
    zero_shot_summary_path.write_text(json.dumps(zero_shot_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[rep] zero-shot saved path={zero_shot_summary_path} "
        f"seconds={time.perf_counter() - t3:.1f}",
        flush=True,
    )

    t4 = time.perf_counter()
    pipeline_summary = {
        "model_id": args.model_id,
        "work_dir": str(work_dir),
        "input_glob": input_glob,
        "calibration_jsonl": str(calibration_path),
        "calibration_summary": str(calibration_summary_path),
        "model_representation": str(representation_path),
        "model_representation_summary": str(representation_summary_path),
        "platonic_geometry": str(geometry_path),
        "zero_shot_summary": str(zero_shot_summary_path),
        "timings_seconds": {
            "calibration": round(t1 - t0, 3),
            "extraction": round(t2 - t1, 3),
            "geometry": round(t3 - t2, 3),
            "zero_shot": round(t4 - t3, 3),
            "total": round(t4 - t0, 3),
        },
    }
    pipeline_summary_path.write_text(json.dumps(pipeline_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**pipeline_summary, "representation_summary": representation_summary, "zero_shot": zero_shot_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
