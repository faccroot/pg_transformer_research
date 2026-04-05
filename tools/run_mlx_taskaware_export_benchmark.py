#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    try:
        sys.path.remove(str(root))
    except ValueError:
        pass
for root in (REPO_ROOT, TOOLS_ROOT):
    sys.path.insert(0, str(root))

import analyze_mlx_quant_export as aqe


DEFAULT_DATA_PATH = REPO_ROOT / "data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "research/iterations/generated/analysis"
DEFAULT_GAUGE_INIT = DEFAULT_ANALYSIS_ROOT / "rope_gauge_hybrid_blockkv_all9_s8.json"


def log(message: str) -> None:
    print(message, flush=True)


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_ANALYSIS_ROOT / f"taskaware_export_benchmark_{stamp}"


def summary_bpb(summary: dict[str, object]) -> float | None:
    roundtrip = summary.get("confirm_roundtrip_eval")
    if isinstance(roundtrip, dict) and "val_bpb" in roundtrip:
        return float(roundtrip["val_bpb"])
    confirm = summary.get("confirm_eval")
    if isinstance(confirm, dict) and "val_bpb" in confirm:
        return float(confirm["val_bpb"])
    return None


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def infer_per_tensor_targets(checkpoint: Path) -> list[str]:
    flat_state = aqe.load_flat_state(checkpoint)
    blocks = aqe.discover_attention_blocks(flat_state)
    return [str(block["k_name"]) for block in blocks]


def build_quant_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--scheme",
        args.scheme,
        "--turbo-mse-patterns",
        args.turbo_mse_patterns,
        "--turbo-prod-patterns",
        args.turbo_prod_patterns,
    ]
    if args.turbo_embed_export:
        out.append("--turbo-embed-export")
    return out


def build_data_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-path",
        str(args.data_path),
        "--tokenizer-path",
        str(args.tokenizer_path),
    ]


def build_train_seq_arg(args: argparse.Namespace) -> list[str]:
    return ["--train-seq-len", str(args.train_seq_len)]


def build_confirm_args(args: argparse.Namespace) -> list[str]:
    if args.confirm_val_max_seqs <= 0:
        return []
    return [
        "--confirm-val-max-seqs",
        str(args.confirm_val_max_seqs),
        "--confirm-val-batch-size",
        str(args.confirm_val_batch_size),
        "--confirm-eval-seq-len",
        str(args.confirm_eval_seq_len),
        "--confirm-eval-stride",
        str(args.confirm_eval_stride),
        "--confirm-eval-batch-seqs",
        str(args.confirm_eval_batch_seqs),
    ]


def run_json_stage(
    *,
    name: str,
    cmd: list[str],
    out_path: Path,
    force: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    if out_path.exists() and not force:
        log(f"[reuse] {name}: {out_path}")
        return load_json(out_path), {"seconds": 0.0, "reused": True, "command": cmd}
    log(f"[start] {name}")
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {proc.returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if out_path.exists():
        data = load_json(out_path)
    else:
        data = json.loads(proc.stdout)
    log(f"[done] {name}: {elapsed:.2f}s")
    return data, {"seconds": float(elapsed), "reused": False, "command": cmd}


def export_stage_record(
    stage_name: str,
    export_summary: dict[str, object],
    export_meta: dict[str, object],
    baseline_bpb: float | None,
    baseline_bytes: int | None,
) -> dict[str, object]:
    artifact_bytes = int(export_summary["artifact_bytes"])
    bpb = summary_bpb(export_summary)
    record = {
        "stage": stage_name,
        "artifact_summary": export_summary,
        "runtime": export_meta,
        "artifact_bytes": artifact_bytes,
        "val_bpb": bpb,
    }
    if baseline_bpb is not None and bpb is not None:
        record["delta_bpb_vs_baseline"] = float(bpb - baseline_bpb)
    if baseline_bytes is not None:
        record["delta_bytes_vs_baseline"] = int(artifact_bytes - baseline_bytes)
    return record


def run_checkpoint_benchmark(args: argparse.Namespace, checkpoint: Path, output_root: Path) -> dict[str, object]:
    py = sys.executable
    quant_args = build_quant_args(args)
    data_args = build_data_args(args)
    train_seq_arg = build_train_seq_arg(args)
    confirm_args = build_confirm_args(args)
    ckpt_dir = output_root / checkpoint.stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    targets = infer_per_tensor_targets(checkpoint)

    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "per_tensor_targets": targets,
        "stage_order": ["baseline", "gauge", "sensitivity", "codebook_global", "codebook_per_tensor", "bitalloc"],
        "stages": {},
    }
    log(f"[checkpoint] {checkpoint}")

    baseline_ptz = ckpt_dir / "baseline_turbo.ptz"
    baseline_summary_path = ckpt_dir / "baseline_turbo_summary.json"
    baseline_summary, baseline_meta = run_json_stage(
        name="baseline_export",
        cmd=[
            py,
            str(TOOLS_ROOT / "export_mlx_taskaware_turbo.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            *confirm_args,
            "--out",
            str(baseline_ptz),
            "--summary-out",
            str(baseline_summary_path),
        ],
        out_path=baseline_summary_path,
        force=args.force,
    )
    baseline_bpb = summary_bpb(baseline_summary)
    baseline_bytes = int(baseline_summary["artifact_bytes"])
    result["stages"]["baseline"] = export_stage_record(
        "baseline",
        baseline_summary,
        baseline_meta,
        baseline_bpb,
        baseline_bytes,
    )

    gauge_json_path = ckpt_dir / "gauge_continuous.json"
    gauge_json, gauge_search_meta = run_json_stage(
        name="gauge_search",
        cmd=[
            py,
            str(TOOLS_ROOT / "optimize_mlx_rope_gauge_continuous.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            "--transform",
            args.gauge_transform,
            "--num-bands",
            str(args.gauge_num_bands),
            "--proxy-seq-len",
            str(args.proxy_seq_len),
            "--proxy-max-seqs",
            str(args.proxy_max_seqs),
            "--proxy-batch-seqs",
            str(args.proxy_batch_seqs),
            "--bytes-weight-per-kib",
            str(args.gauge_bytes_weight_per_kib),
            "--initial-step",
            str(args.gauge_initial_step),
            "--min-step",
            str(args.gauge_min_step),
            "--step-decay",
            str(args.gauge_step_decay),
            "--max-rounds-per-step",
            str(args.gauge_max_rounds_per_step),
            *confirm_args,
            *(
                ["--init-result", str(args.gauge_init_result)]
                if args.gauge_init_result is not None
                else []
            ),
            "--out",
            str(gauge_json_path),
        ],
        out_path=gauge_json_path,
        force=args.force,
    )
    gauge_export_ptz = ckpt_dir / "gauge_turbo.ptz"
    gauge_export_summary_path = ckpt_dir / "gauge_turbo_summary.json"
    gauge_export_summary, gauge_export_meta = run_json_stage(
        name="gauge_export",
        cmd=[
            py,
            str(TOOLS_ROOT / "export_mlx_taskaware_turbo.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            *confirm_args,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--out",
            str(gauge_export_ptz),
            "--summary-out",
            str(gauge_export_summary_path),
        ],
        out_path=gauge_export_summary_path,
        force=args.force,
    )
    gauge_record = export_stage_record(
        "gauge",
        gauge_export_summary,
        {
            "search_seconds": gauge_search_meta["seconds"],
            "export_seconds": gauge_export_meta["seconds"],
            "total_seconds": float(gauge_search_meta["seconds"] + gauge_export_meta["seconds"]),
            "search_reused": gauge_search_meta["reused"],
            "export_reused": gauge_export_meta["reused"],
            "search_command": gauge_search_meta["command"],
            "export_command": gauge_export_meta["command"],
        },
        baseline_bpb,
        baseline_bytes,
    )
    gauge_record["search_result"] = gauge_json
    result["stages"]["gauge"] = gauge_record

    sensitivity_json_path = ckpt_dir / "gauge_sensitivity.json"
    sensitivity_json, sensitivity_meta = run_json_stage(
        name="gauge_sensitivity",
        cmd=[
            py,
            str(TOOLS_ROOT / "analyze_mlx_rope_gauge_sensitivity.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            "--gauge-result",
            str(gauge_json_path),
            "--transform",
            args.gauge_transform,
            "--num-bands",
            str(args.gauge_num_bands),
            "--proxy-seq-len",
            str(args.proxy_seq_len),
            "--proxy-max-seqs",
            str(args.proxy_max_seqs),
            "--proxy-batch-seqs",
            str(args.proxy_batch_seqs),
            "--out",
            str(sensitivity_json_path),
        ],
        out_path=sensitivity_json_path,
        force=args.force,
    )
    result["stages"]["sensitivity"] = {
        "stage": "sensitivity",
        "result": sensitivity_json,
        "runtime": sensitivity_meta,
    }

    codebook_global_json_path = ckpt_dir / "codebook_global.json"
    codebook_global_json, codebook_global_meta = run_json_stage(
        name="codebook_global",
        cmd=[
            py,
            str(TOOLS_ROOT / "refine_mlx_turbo_codebook_ce.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--proxy-seq-len",
            str(args.proxy_seq_len),
            "--proxy-max-seqs",
            str(args.proxy_max_seqs),
            "--proxy-batch-seqs",
            str(args.proxy_batch_seqs),
            "--bytes-weight-per-kib",
            str(args.codebook_bytes_weight_per_kib),
            "--initial-step",
            str(args.codebook_initial_step),
            "--min-step",
            str(args.codebook_min_step),
            "--step-decay",
            str(args.codebook_step_decay),
            "--max-rounds-per-step",
            str(args.codebook_max_rounds_per_step),
            *confirm_args,
            "--out",
            str(codebook_global_json_path),
        ],
        out_path=codebook_global_json_path,
        force=args.force,
    )
    codebook_global_export_ptz = ckpt_dir / "taskaware_global_turbo.ptz"
    codebook_global_export_summary_path = ckpt_dir / "taskaware_global_turbo_summary.json"
    codebook_global_export_summary, codebook_global_export_meta = run_json_stage(
        name="codebook_global_export",
        cmd=[
            py,
            str(TOOLS_ROOT / "export_mlx_taskaware_turbo.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            *confirm_args,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--codebook-result",
            str(codebook_global_json_path),
            "--out",
            str(codebook_global_export_ptz),
            "--summary-out",
            str(codebook_global_export_summary_path),
        ],
        out_path=codebook_global_export_summary_path,
        force=args.force,
    )
    codebook_global_record = export_stage_record(
        "codebook_global",
        codebook_global_export_summary,
        {
            "search_seconds": codebook_global_meta["seconds"],
            "export_seconds": codebook_global_export_meta["seconds"],
            "total_seconds": float(codebook_global_meta["seconds"] + codebook_global_export_meta["seconds"]),
            "search_reused": codebook_global_meta["reused"],
            "export_reused": codebook_global_export_meta["reused"],
            "search_command": codebook_global_meta["command"],
            "export_command": codebook_global_export_meta["command"],
        },
        baseline_bpb,
        baseline_bytes,
    )
    codebook_global_record["search_result"] = codebook_global_json
    result["stages"]["codebook_global"] = codebook_global_record

    codebook_per_tensor_json_path = ckpt_dir / "codebook_per_tensor.json"
    codebook_per_tensor_json, codebook_per_tensor_meta = run_json_stage(
        name="codebook_per_tensor",
        cmd=[
            py,
            str(TOOLS_ROOT / "refine_mlx_turbo_codebook_ce.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--per-tensor",
            "--target-tensors",
            ",".join(targets),
            "--proxy-seq-len",
            str(args.proxy_seq_len),
            "--proxy-max-seqs",
            str(args.proxy_max_seqs),
            "--proxy-batch-seqs",
            str(args.proxy_batch_seqs),
            "--bytes-weight-per-kib",
            str(args.codebook_bytes_weight_per_kib),
            "--initial-step",
            str(args.per_tensor_initial_step),
            "--min-step",
            str(args.per_tensor_min_step),
            "--step-decay",
            str(args.codebook_step_decay),
            "--max-rounds-per-step",
            str(args.per_tensor_max_rounds_per_step),
            *confirm_args,
            "--out",
            str(codebook_per_tensor_json_path),
        ],
        out_path=codebook_per_tensor_json_path,
        force=args.force,
    )
    codebook_per_tensor_export_ptz = ckpt_dir / "taskaware_per_tensor_turbo.ptz"
    codebook_per_tensor_export_summary_path = ckpt_dir / "taskaware_per_tensor_turbo_summary.json"
    codebook_per_tensor_export_summary, codebook_per_tensor_export_meta = run_json_stage(
        name="codebook_per_tensor_export",
        cmd=[
            py,
            str(TOOLS_ROOT / "export_mlx_taskaware_turbo.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            *confirm_args,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--codebook-result",
            str(codebook_per_tensor_json_path),
            "--out",
            str(codebook_per_tensor_export_ptz),
            "--summary-out",
            str(codebook_per_tensor_export_summary_path),
        ],
        out_path=codebook_per_tensor_export_summary_path,
        force=args.force,
    )
    codebook_per_tensor_record = export_stage_record(
        "codebook_per_tensor",
        codebook_per_tensor_export_summary,
        {
            "search_seconds": codebook_per_tensor_meta["seconds"],
            "export_seconds": codebook_per_tensor_export_meta["seconds"],
            "total_seconds": float(codebook_per_tensor_meta["seconds"] + codebook_per_tensor_export_meta["seconds"]),
            "search_reused": codebook_per_tensor_meta["reused"],
            "export_reused": codebook_per_tensor_export_meta["reused"],
            "search_command": codebook_per_tensor_meta["command"],
            "export_command": codebook_per_tensor_export_meta["command"],
        },
        baseline_bpb,
        baseline_bytes,
    )
    codebook_per_tensor_record["search_result"] = codebook_per_tensor_json
    result["stages"]["codebook_per_tensor"] = codebook_per_tensor_record

    bitalloc_json_path = ckpt_dir / "bitalloc.json"
    bitalloc_json, bitalloc_meta = run_json_stage(
        name="bitalloc",
        cmd=[
            py,
            str(TOOLS_ROOT / "optimize_mlx_turbo_bit_allocation.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--codebook-result",
            str(codebook_per_tensor_json_path),
            "--sensitivity-result",
            str(sensitivity_json_path),
            "--candidate-prod-bits",
            args.bitalloc_candidate_prod_bits,
            "--top-k-units",
            str(args.bitalloc_top_k_units),
            "--bytes-weight-per-kib",
            str(args.bitalloc_bytes_weight_per_kib),
            "--max-rounds",
            str(args.bitalloc_max_rounds),
            "--max-pair-rounds",
            str(args.bitalloc_max_pair_rounds),
            "--max-extra-zlib-bytes",
            str(args.bitalloc_max_extra_zlib_bytes),
            "--proxy-seq-len",
            str(args.proxy_seq_len),
            "--proxy-max-seqs",
            str(args.proxy_max_seqs),
            "--proxy-batch-seqs",
            str(args.proxy_batch_seqs),
            *confirm_args,
            "--out",
            str(bitalloc_json_path),
        ],
        out_path=bitalloc_json_path,
        force=args.force,
    )
    bitalloc_export_ptz = ckpt_dir / "taskaware_bitalloc_turbo.ptz"
    bitalloc_export_summary_path = ckpt_dir / "taskaware_bitalloc_turbo_summary.json"
    bitalloc_export_summary, bitalloc_export_meta = run_json_stage(
        name="bitalloc_export",
        cmd=[
            py,
            str(TOOLS_ROOT / "export_mlx_taskaware_turbo.py"),
            str(checkpoint),
            *quant_args,
            *data_args,
            *train_seq_arg,
            *confirm_args,
            "--gauge-result",
            str(gauge_json_path),
            "--gauge-transform",
            args.gauge_transform,
            "--codebook-result",
            str(codebook_per_tensor_json_path),
            "--bitalloc-result",
            str(bitalloc_json_path),
            "--out",
            str(bitalloc_export_ptz),
            "--summary-out",
            str(bitalloc_export_summary_path),
        ],
        out_path=bitalloc_export_summary_path,
        force=args.force,
    )
    bitalloc_record = export_stage_record(
        "bitalloc",
        bitalloc_export_summary,
        {
            "search_seconds": bitalloc_meta["seconds"],
            "export_seconds": bitalloc_export_meta["seconds"],
            "total_seconds": float(bitalloc_meta["seconds"] + bitalloc_export_meta["seconds"]),
            "search_reused": bitalloc_meta["reused"],
            "export_reused": bitalloc_export_meta["reused"],
            "search_command": bitalloc_meta["command"],
            "export_command": bitalloc_export_meta["command"],
        },
        baseline_bpb,
        baseline_bytes,
    )
    bitalloc_record["search_result"] = bitalloc_json
    result["stages"]["bitalloc"] = bitalloc_record
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full task-aware Turbo export benchmark stack on one or more MLX checkpoints."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-max-seqs", type=int, default=8)
    parser.add_argument("--proxy-batch-seqs", type=int, default=1)
    parser.add_argument("--confirm-val-max-seqs", type=int, default=64)
    parser.add_argument("--confirm-val-batch-size", type=int, default=262144)
    parser.add_argument("--confirm-eval-seq-len", type=int, default=1024)
    parser.add_argument("--confirm-eval-stride", type=int, default=0)
    parser.add_argument("--confirm-eval-batch-seqs", type=int, default=0)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument(
        "--turbo-mse-patterns",
        default="attn.c_q.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight",
    )
    parser.add_argument("--turbo-prod-patterns", default="attn.c_k.weight")
    parser.add_argument("--gauge-transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--gauge-init-result", type=Path, default=DEFAULT_GAUGE_INIT)
    parser.add_argument("--gauge-num-bands", type=int, default=4)
    parser.add_argument("--gauge-bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--gauge-initial-step", type=float, default=0.15)
    parser.add_argument("--gauge-min-step", type=float, default=0.03)
    parser.add_argument("--gauge-step-decay", type=float, default=0.5)
    parser.add_argument("--gauge-max-rounds-per-step", type=int, default=1)
    parser.add_argument("--codebook-bytes-weight-per-kib", type=float, default=1.0e-4)
    parser.add_argument("--codebook-initial-step", type=float, default=0.005)
    parser.add_argument("--codebook-min-step", type=float, default=0.0005)
    parser.add_argument("--codebook-step-decay", type=float, default=0.5)
    parser.add_argument("--codebook-max-rounds-per-step", type=int, default=2)
    parser.add_argument("--per-tensor-initial-step", type=float, default=0.0025)
    parser.add_argument("--per-tensor-min-step", type=float, default=0.000625)
    parser.add_argument("--per-tensor-max-rounds-per-step", type=int, default=2)
    parser.add_argument("--bitalloc-candidate-prod-bits", default="3,4,5")
    parser.add_argument("--bitalloc-top-k-units", type=int, default=24)
    parser.add_argument("--bitalloc-bytes-weight-per-kib", type=float, default=0.0)
    parser.add_argument("--bitalloc-max-rounds", type=int, default=3)
    parser.add_argument("--bitalloc-max-pair-rounds", type=int, default=0)
    parser.add_argument("--bitalloc-max-extra-zlib-bytes", type=int, default=768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log(f"[output_root] {args.output_root}")

    checkpoints: list[dict[str, object]] = []
    for checkpoint in args.checkpoints:
        checkpoints.append(run_checkpoint_benchmark(args, checkpoint, args.output_root))

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(args.output_root),
        "checkpoints": checkpoints,
    }
    summary_path = args.output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
