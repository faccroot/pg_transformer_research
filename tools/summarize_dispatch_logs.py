#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOB_RE = re.compile(r"\[(job_[^\]]+)\]")
RUN_ID_RE = re.compile(r"^run_id:(\S+)$", re.MULTILINE)
RUNNING_HOST_RE = re.compile(r"Running on (\S+)\.\.\.")
ITER_RE = re.compile(r"^iterations:(.*)$", re.MULTILINE)
JEPA_RE = re.compile(r"^jepa_aux:(.*)$", re.MULTILINE)
COMPUTE_RE = re.compile(r"^compute_dtype:(.*)$", re.MULTILINE)
FINAL_EXACT_RE = re.compile(r"^(final_[^ ]+_exact) val_loss:([0-9.]+) val_bpb:([0-9.]+)$", re.MULTILINE)
RELEASED_RE = re.compile(r"Released (\S+)$", re.MULTILINE)


def parse_value(raw: str):
    if raw.endswith("ms"):
        raw = raw[:-2]
    if raw in {"True", "False"}:
        return raw == "True"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_keyvals(blob: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in blob.split():
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        out[k] = parse_value(v)
    return out


def infer_mode(run_id: str) -> str:
    if run_id.startswith("jepa_aux_"):
        return "jepa_aux"
    if run_id.startswith("baseline_"):
        return "baseline"
    if run_id.startswith("state_jepa_"):
        return "state_jepa"
    return "unknown"


def summarize_log(path: Path, suite_id: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")

    job_id_match = JOB_RE.search(text)
    run_id_match = RUN_ID_RE.search(text)
    host_match = RUNNING_HOST_RE.search(text)
    iter_match = ITER_RE.search(text)
    jepa_match = JEPA_RE.search(text)
    compute_match = COMPUTE_RE.search(text)
    final_match = list(FINAL_EXACT_RE.finditer(text))

    if not job_id_match:
        raise ValueError(f"Could not parse job id from {path}")
    if not run_id_match:
        raise ValueError(f"Could not parse run_id from {path}")
    if not final_match:
        raise ValueError(f"Could not find final exact metric in {path}")

    run_id = run_id_match.group(1)
    host = host_match.group(1) if host_match else ""
    iterations = parse_keyvals(iter_match.group(1)) if iter_match else {}
    jepa_cfg = parse_keyvals(jepa_match.group(1)) if jepa_match else {}
    compute_cfg = parse_keyvals(compute_match.group(1)) if compute_match else {}
    final = final_match[-1]

    duration_seconds = int(float(iterations.get("max_wallclock_seconds", 0) or 0))
    train_shards = 1
    val_max_seqs = int(iterations.get("val_seqs", 0) or 0)
    mlx_compile = int(compute_cfg.get("compile_mode", 0) or 0)
    pred_weight = float(jepa_cfg.get("pred_weight", 0.0) or 0.0)
    sigreg_weight = float(jepa_cfg.get("sigreg_weight", 0.0) or 0.0)
    metric_name = final.group(1)
    val_loss = float(final.group(2))
    val_bpb = float(final.group(3))
    returncode = 0 if RELEASED_RE.search(text) else 1
    timestamp_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    row: dict[str, object] = {
        "timestamp_utc": timestamp_utc,
        "suite_id": suite_id,
        "status": "completed" if returncode == 0 else "unknown",
        "job_id": job_id_match.group(1),
        "host": host,
        "mode": infer_mode(run_id),
        "run_id": run_id,
        "run_suffix": run_id,
        "duration_seconds": duration_seconds,
        "train_shards": train_shards,
        "val_max_seqs": val_max_seqs,
        "turbo_profile": "none",
        "mlx_compile": mlx_compile,
        "jepa_pred_weight": pred_weight,
        "jepa_sigreg_weight": sigreg_weight,
        "metric_name": metric_name,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "returncode": returncode,
        "dispatch_log": str(path.relative_to(ROOT)),
    }
    return row


def append_to_ledger(ledger_path: Path, rows: list[dict[str, object]]) -> int:
    existing_keys: set[tuple[str, str]] = set()
    with ledger_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Ledger has no header: {ledger_path}")
        for row in reader:
            existing_keys.add((row.get("suite_id", ""), row.get("job_id", "")))

    new_rows = [row for row in rows if (str(row["suite_id"]), str(row["job_id"])) not in existing_keys]
    if not new_rows:
        return 0

    with ledger_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--write-results-csv", action="store_true")
    args = parser.parse_args()

    iteration_dir = Path(args.iteration_dir)
    suite_id = iteration_dir.name
    dispatch_dir = iteration_dir / "dispatch_logs"
    logs = sorted(dispatch_dir.glob("*.log"))
    if not logs:
        raise SystemExit(f"No dispatch logs found in {dispatch_dir}")

    rows = [summarize_log(log, suite_id) for log in logs]
    rows.sort(key=lambda row: (str(row["host"]), str(row["run_id"])))

    if args.write_results_csv:
        write_results_csv(iteration_dir / "results.csv", rows)

    if args.ledger:
        appended = append_to_ledger(Path(args.ledger), rows)
        print(f"appended_ledger_rows:{appended}")

    for row in rows:
        print(
            f'{row["host"]},{row["job_id"]},{row["run_id"]},{row["metric_name"]},'
            f'{row["val_loss"]:.8f},{row["val_bpb"]:.8f}'
        )


if __name__ == "__main__":
    main()
