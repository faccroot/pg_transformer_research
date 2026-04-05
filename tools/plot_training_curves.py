#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^step:(\d+)/(\d+)\s+(.*)$")
FINAL_RE = re.compile(r"^(final_[^ ]+)\s+(.*)$")


def parse_value(raw: str):
    if raw.endswith("ms"):
        raw = raw[:-2]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_keyvals(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in text.split():
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        out[key] = parse_value(value)
    return out


def lr_mul(step: int, elapsed_ms: float, iterations: int, max_wallclock_seconds: float, warmdown_iters: int, warmdown_fraction: float) -> float:
    if warmdown_iters <= 0:
        if warmdown_fraction <= 0.0 or max_wallclock_seconds <= 0:
            return 1.0
    if max_wallclock_seconds <= 0:
        warmdown_start = max(iterations - warmdown_iters, 0)
        if warmdown_start <= step < iterations:
            return max((iterations - step) / max(warmdown_iters, 1), 0.0)
        return 1.0
    if warmdown_fraction > 0.0:
        warmdown_ms = 1000.0 * max_wallclock_seconds * warmdown_fraction
        remaining_ms = max(1000.0 * max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(1000.0 * max_wallclock_seconds - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


def parse_log(path: Path) -> dict[str, object]:
    cfg: dict[str, object] = {
        "iterations": 1_000_000,
        "warmdown_iters": 1200,
        "warmdown_fraction": 0.18,
        "max_wallclock_seconds": 3600.0,
        "matrix_lr": 0.04,
        "scalar_lr": 0.04,
        "embed_lr": 0.05,
        "run_id": path.stem,
    }
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    finals: dict[str, dict[str, object]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("run_id:"):
                cfg["run_id"] = line.split(":", 1)[1]
                continue
            if line.startswith("iterations:"):
                cfg.update(parse_keyvals(line))
                continue
            if line.startswith("optimizer:"):
                cfg.update(parse_keyvals(line))
                continue

            m = STEP_RE.match(line)
            if m:
                step = int(m.group(1))
                total_iters = int(m.group(2))
                kv = parse_keyvals(m.group(3))
                kv["step"] = step
                kv["iterations"] = total_iters
                elapsed_ms = float(kv.get("train_time", 0.0))
                kv["elapsed_min"] = elapsed_ms / 60000.0
                kv["lr_mul_reconstructed"] = lr_mul(
                    step=step,
                    elapsed_ms=elapsed_ms,
                    iterations=int(cfg.get("iterations", total_iters)),
                    max_wallclock_seconds=float(cfg.get("max_wallclock_seconds", 0.0)),
                    warmdown_iters=int(cfg.get("warmdown_iters", 1200)),
                    warmdown_fraction=float(cfg.get("warmdown_fraction", 0.18)),
                )
                kv["matrix_lr_effective"] = float(cfg.get("matrix_lr", 0.04)) * float(kv["lr_mul_reconstructed"])
                kv["scalar_lr_effective"] = float(cfg.get("scalar_lr", 0.04)) * float(kv["lr_mul_reconstructed"])
                kv["embed_lr_effective"] = float(cfg.get("embed_lr", 0.05)) * float(kv["lr_mul_reconstructed"])
                if "val_loss" in kv:
                    val_rows.append(kv)
                elif "train_loss" in kv:
                    train_rows.append(kv)
                continue

            m = FINAL_RE.match(line)
            if m:
                finals[m.group(1)] = parse_keyvals(m.group(2))

    return {
        "config": cfg,
        "train": train_rows,
        "val": val_rows,
        "finals": finals,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(runs: list[dict[str, object]], out_path: Path, x_key: str, y_key: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False
    for run in runs:
        rows = [row for row in run["train"] if y_key in row and x_key in row]  # type: ignore[index]
        if not rows:
            continue
        xs = [float(row[x_key]) for row in rows]
        ys = [float(row[y_key]) for row in rows]
        plt.plot(xs, ys, label=run["label"], linewidth=2)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Elapsed Minutes" if x_key == "elapsed_min" else x_key.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_final_bpb(runs: list[dict[str, object]], out_path: Path) -> None:
    labels: list[str] = []
    vals: list[float] = []
    for run in runs:
        finals = run["finals"]  # type: ignore[index]
        metric = finals.get("final_int8_zlib_roundtrip_exact") or finals.get("final_raw_export_ready_exact")
        if not metric or "val_bpb" not in metric:
            continue
        labels.append(run["label"])  # type: ignore[index]
        vals.append(float(metric["val_bpb"]))
    if not vals:
        return
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals)
    plt.ylabel("Final BPB")
    plt.title("Final BPB By Run")
    plt.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run", action="append", required=True, help="label=/abs/path/to/log.txt")
    parser.add_argument("--title", default="Training Curves")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        parsed = parse_log(path)
        parsed["label"] = label
        runs.append(parsed)
        run_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
        write_csv(out_dir / f"{run_slug}_train.csv", parsed["train"])  # type: ignore[arg-type]
        write_csv(out_dir / f"{run_slug}_val.csv", parsed["val"])  # type: ignore[arg-type]

    plot_metric(runs, out_dir / "train_loss_vs_minutes.png", "elapsed_min", "train_loss", args.title + " - Train Loss", "Train Loss")
    plot_metric(runs, out_dir / "ce_vs_minutes.png", "elapsed_min", "ce", args.title + " - Token CE", "CE")
    plot_metric(runs, out_dir / "lr_mul_vs_minutes.png", "elapsed_min", "lr_mul_reconstructed", args.title + " - LR Multiplier", "LR Multiplier")
    plot_metric(runs, out_dir / "jepa_pred_loss_vs_minutes.png", "elapsed_min", "jepa_pred", args.title + " - JEPA Prediction Loss", "JEPA Pred Loss")
    plot_metric(runs, out_dir / "jepa_sigreg_vs_minutes.png", "elapsed_min", "jepa_sigreg", args.title + " - SIGReg", "SIGReg")
    plot_metric(runs, out_dir / "jepa_pred_weight_vs_minutes.png", "elapsed_min", "jepa_pred_weight", args.title + " - JEPA Pred Weight", "Pred Weight")
    plot_final_bpb(runs, out_dir / "final_bpb_bar.png")

    summary: dict[str, object] = {"title": args.title, "runs": []}
    for run in runs:
        finals = run["finals"]  # type: ignore[index]
        summary["runs"].append(
            {
                "label": run["label"],
                "run_id": run["config"].get("run_id"),  # type: ignore[index]
                "final_raw_export_ready_exact": finals.get("final_raw_export_ready_exact"),
                "final_int8_zlib_roundtrip_exact": finals.get("final_int8_zlib_roundtrip_exact"),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
