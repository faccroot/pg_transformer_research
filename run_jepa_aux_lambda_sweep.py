#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLUSTER_ROOT = Path.home() / "cluster"
DISPATCH = CLUSTER_ROOT / "dispatch.sh"
STATUS = CLUSTER_ROOT / "status.sh"
LEDGER_PATH = ROOT / "research" / "experiment_ledger.csv"
GENERATED_ROOT = ROOT / "research" / "iterations" / "generated"
SUPPORT_FILES = [
    ROOT / "data" / "cached_challenge_fineweb.py",
    ROOT / "train_gpt_mlx.py",
    ROOT / "train_gpt_mlx_jepa_aux.py",
    ROOT / "logic_register_mlx.py",
    ROOT / "turbo_quant_mlx.py",
]

JOB_ID_RE = re.compile(r"\[(job_[^\]]+)\]")
RUN_HOST_RE = re.compile(r"\[(job_[^\]]+)\] Running on (\S+)\.\.\.")
RUN_ID_RE = re.compile(r"^run_id:(\S+)$", re.MULTILINE)
FINAL_EXACT_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+)")
RAW_VAL_RE = re.compile(r"step:(\d+)/\d+ val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+)")

LEDGER_COLUMNS = [
    "timestamp_utc",
    "suite_id",
    "status",
    "job_id",
    "host",
    "mode",
    "run_id",
    "run_suffix",
    "duration_seconds",
    "train_shards",
    "val_max_seqs",
    "turbo_profile",
    "mlx_compile",
    "jepa_pred_weight",
    "jepa_sigreg_weight",
    "metric_name",
    "val_loss",
    "val_bpb",
    "returncode",
    "dispatch_log",
]


@dataclass
class ActiveRun:
    host: str
    pred_weight: float
    sig_weight: float
    run_suffix: str
    dispatch_log: Path
    process: subprocess.Popen[str]


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ready_hosts() -> list[str]:
    out = subprocess.check_output(["bash", str(STATUS)], text=True)
    hosts: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("mini") and parts[1] == "READY":
            hosts.append(parts[0])
    return hosts


def ensure_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()


def append_ledger(path: Path, row: dict[str, object]) -> None:
    ensure_ledger(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})


def slugify_weight(value: float) -> str:
    return f"{int(round(value * 1000)):04d}"


def parse_run_result(dispatch_log: Path, fallback_host: str) -> dict[str, object]:
    text = dispatch_log.read_text(encoding="utf-8", errors="replace")
    job_match = JOB_ID_RE.search(text)
    host_match = RUN_HOST_RE.search(text)
    run_id_match = RUN_ID_RE.search(text)
    final_matches = FINAL_EXACT_RE.findall(text)
    raw_matches = RAW_VAL_RE.findall(text)
    metric_name = ""
    val_loss = ""
    val_bpb = ""
    status = "failed"
    if final_matches:
        loss, bpb = final_matches[-1]
        metric_name = "final_int8_zlib_roundtrip_exact"
        val_loss = loss
        val_bpb = bpb
        status = "completed"
    elif raw_matches:
        _, loss, bpb = raw_matches[-1]
        metric_name = "val"
        val_loss = loss
        val_bpb = bpb
    return {
        "status": status,
        "job_id": job_match.group(1) if job_match else "",
        "host": host_match.group(2) if host_match else fallback_host,
        "run_id": run_id_match.group(1) if run_id_match else "",
        "metric_name": metric_name,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
    }


def launch_run(
    *,
    host: str,
    pred_weight: float,
    sig_weight: float,
    duration_seconds: int,
    train_shards: int,
    val_max_seqs: int,
    turbo_profile: str,
    mlx_compile: str,
    jepa_aux_start_frac: float,
    jepa_aux_ramp_frac: float,
    jepa_summary_mode: str,
    jepa_pred_mode: str,
    jepa_pred_init_std: float,
    jepa_grad_scrub_nonfinite: int,
    suite_id: str,
    logs_dir: Path,
) -> ActiveRun:
    run_suffix = f"{suite_id}_p{slugify_weight(pred_weight)}_s{slugify_weight(sig_weight)}"
    dispatch_log = logs_dir / f"{run_suffix}.log"
    cmd = [
        str(DISPATCH),
        "--host",
        host,
        str(ROOT / "run_mini_pg_job.py"),
        "--mode",
        "jepa_aux",
        "--duration-seconds",
        str(duration_seconds),
        "--train-shards",
        str(train_shards),
        "--turbo-profile",
        turbo_profile,
        "--val-max-seqs",
        str(val_max_seqs),
        "--mlx-compile",
        mlx_compile,
        "--run-suffix",
        run_suffix,
        "--jepa-pred-weight",
        str(pred_weight),
        "--jepa-sigreg-weight",
        str(sig_weight),
        "--jepa-aux-start-frac",
        str(jepa_aux_start_frac),
        "--jepa-aux-ramp-frac",
        str(jepa_aux_ramp_frac),
        "--jepa-summary-mode",
        jepa_summary_mode,
        "--jepa-pred-mode",
        jepa_pred_mode,
        "--jepa-pred-init-std",
        str(jepa_pred_init_std),
        "--jepa-grad-scrub-nonfinite",
        str(jepa_grad_scrub_nonfinite),
        *[str(path) for path in SUPPORT_FILES],
    ]
    f = dispatch_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=ROOT.parent,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveRun(
        host=host,
        pred_weight=pred_weight,
        sig_weight=sig_weight,
        run_suffix=run_suffix,
        dispatch_log=dispatch_log,
        process=process,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JEPA auxiliary lambda sweep on the mini cluster")
    parser.add_argument("--duration-seconds", type=int, default=42)
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--val-max-seqs", type=int, default=256)
    parser.add_argument("--turbo-profile", default="none")
    parser.add_argument("--mlx-compile", default="0")
    parser.add_argument("--hosts", default="")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--pred-values", default="0.0,0.05,0.1,0.5")
    parser.add_argument("--sig-values", default="0.0,0.005,0.01,0.05")
    parser.add_argument("--jepa-aux-start-frac", type=float, default=0.0)
    parser.add_argument("--jepa-aux-ramp-frac", type=float, default=0.0)
    parser.add_argument("--jepa-summary-mode", default="mean")
    parser.add_argument("--jepa-pred-mode", default="residual_linear")
    parser.add_argument("--jepa-pred-init-std", type=float, default=1e-4)
    parser.add_argument("--jepa-grad-scrub-nonfinite", type=int, default=1)
    args = parser.parse_args()

    pred_values = [float(x) for x in args.pred_values.split(",") if x]
    sig_values = [float(x) for x in args.sig_values.split(",") if x]
    suite_id = datetime.now(timezone.utc).strftime("jepa_aux_lambda_%Y%m%d_%H%M%S")
    currently_ready = ready_hosts()
    if args.hosts:
        requested_hosts = [h for h in args.hosts.split(",") if h]
        unavailable_hosts = [h for h in requested_hosts if h not in currently_ready]
        hosts = [h for h in requested_hosts if h in currently_ready]
        if unavailable_hosts:
            print(
                f"Skipping unavailable hosts: {', '.join(unavailable_hosts)}",
                file=sys.stderr,
            )
    else:
        requested_hosts = []
        unavailable_hosts = []
        hosts = currently_ready
    if not hosts:
        raise SystemExit("No READY mini hosts available for the sweep")
    max_parallel = args.max_parallel or len(hosts)
    suite_dir = GENERATED_ROOT / suite_id
    logs_dir = suite_dir / "dispatch_logs"
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ensure_ledger(LEDGER_PATH)

    configs = [{"pred_weight": p, "sig_weight": s} for p, s in itertools.product(pred_values, sig_values)]
    metadata = {
        "created_at_utc": iso_utc_now(),
        "suite_id": suite_id,
        "duration_seconds": args.duration_seconds,
        "train_shards": args.train_shards,
        "val_max_seqs": args.val_max_seqs,
        "turbo_profile": args.turbo_profile,
        "mlx_compile": args.mlx_compile,
        "jepa_aux_start_frac": args.jepa_aux_start_frac,
        "jepa_aux_ramp_frac": args.jepa_aux_ramp_frac,
        "jepa_summary_mode": args.jepa_summary_mode,
        "jepa_pred_mode": args.jepa_pred_mode,
        "jepa_pred_init_std": args.jepa_pred_init_std,
        "jepa_grad_scrub_nonfinite": args.jepa_grad_scrub_nonfinite,
        "requested_hosts": requested_hosts,
        "unavailable_hosts": unavailable_hosts,
        "hosts": hosts,
        "pred_values": pred_values,
        "sig_values": sig_values,
    }
    (suite_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    pending = list(configs)
    free_hosts = list(hosts)
    active: list[ActiveRun] = []
    results: list[dict[str, object]] = []

    while pending or active:
        while pending and free_hosts and len(active) < max_parallel:
            cfg = pending.pop(0)
            host = free_hosts.pop(0)
            run = launch_run(
                host=host,
                pred_weight=cfg["pred_weight"],
                sig_weight=cfg["sig_weight"],
                duration_seconds=args.duration_seconds,
                train_shards=args.train_shards,
                val_max_seqs=args.val_max_seqs,
                turbo_profile=args.turbo_profile,
                mlx_compile=args.mlx_compile,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
                suite_id=suite_id,
                logs_dir=logs_dir,
            )
            active.append(run)

        if not active:
            time.sleep(args.poll_seconds)
            continue

        time.sleep(args.poll_seconds)
        still_active: list[ActiveRun] = []
        for run in active:
            returncode = run.process.poll()
            if returncode is None:
                still_active.append(run)
                continue
            parsed = parse_run_result(run.dispatch_log, run.host)
            row: dict[str, object] = {
                "timestamp_utc": iso_utc_now(),
                "suite_id": suite_id,
                "status": parsed["status"] if returncode == 0 else "failed",
                "job_id": parsed["job_id"],
                "host": parsed["host"],
                "mode": "jepa_aux",
                "run_id": parsed["run_id"],
                "run_suffix": run.run_suffix,
                "duration_seconds": args.duration_seconds,
                "train_shards": args.train_shards,
                "val_max_seqs": args.val_max_seqs,
                "turbo_profile": args.turbo_profile,
                "mlx_compile": args.mlx_compile,
                "jepa_pred_weight": run.pred_weight,
                "jepa_sigreg_weight": run.sig_weight,
                "metric_name": parsed["metric_name"],
                "val_loss": parsed["val_loss"],
                "val_bpb": parsed["val_bpb"],
                "returncode": returncode,
                "dispatch_log": str(run.dispatch_log.relative_to(ROOT)),
            }
            append_ledger(LEDGER_PATH, row)
            results.append(row)
            free_hosts.append(run.host)
        active = still_active

    sorted_results = sorted(
        results,
        key=lambda row: float(row["val_bpb"]) if row["val_bpb"] not in {"", None} else float("inf"),
    )
    with (suite_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2)
        f.write("\n")
    with (suite_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_results)

    if sorted_results:
        best = next((row for row in sorted_results if row["status"] == "completed"), sorted_results[0])
        print(
            json.dumps(
                {
                    "suite_id": suite_id,
                    "best_run_suffix": best["run_suffix"],
                    "best_val_bpb": best["val_bpb"],
                    "best_pred_weight": best["jepa_pred_weight"],
                    "best_sig_weight": best["jepa_sigreg_weight"],
                    "results_csv": str((suite_dir / "results.csv").relative_to(ROOT)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
