#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_WARMDOWN_FRACTION = 0.18


@dataclass
class ScheduleConfig:
    max_wallclock_seconds: float
    warmdown_fraction: float
    embed_lr: float
    matrix_lr: float
    scalar_lr: float


STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms "
    r"step_avg:(?P<step_avg_ms>[0-9.]+)ms "
    r"tok_s:(?P<tok_s>[0-9.]+)"
    r"(?P<rest>.*)$"
)
KV_RE = re.compile(r"([a-zA-Z0-9_]+):([^\s]+)")
FINAL_RE = re.compile(r"final_(raw_export_ready|int8_zlib_roundtrip)_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")
STOP_VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:([0-9.]+)ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MLX training/lr curves from dispatch logs.")
    parser.add_argument("suite_dir", help="Suite directory containing dispatch_logs/ and optionally configs/.")
    parser.add_argument(
        "--out-dir",
        help="Directory to write plots into. Defaults to <suite_dir>/plots.",
    )
    return parser.parse_args()


def load_manifest(suite_dir: Path) -> dict:
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text())


def load_config_env(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text()).get("env", {})


def schedule_from_env(env: dict) -> ScheduleConfig:
    return ScheduleConfig(
        max_wallclock_seconds=float(env.get("MAX_WALLCLOCK_SECONDS", 0.0)),
        warmdown_fraction=float(env.get("WARMDOWN_FRACTION", DEFAULT_WARMDOWN_FRACTION)),
        embed_lr=float(env.get("TIED_EMBED_LR", env.get("EMBED_LR", 0.05))),
        matrix_lr=float(env.get("MATRIX_LR", 0.04)),
        scalar_lr=float(env.get("SCALAR_LR", 0.04)),
    )


def compute_lr_mul(train_time_ms: float, schedule: ScheduleConfig) -> float:
    if schedule.max_wallclock_seconds <= 0:
        return 1.0
    if schedule.warmdown_fraction <= 0.0:
        return 1.0
    total_ms = 1000.0 * schedule.max_wallclock_seconds
    warmdown_ms = total_ms * schedule.warmdown_fraction
    remaining_ms = max(total_ms - train_time_ms, 0.0)
    if remaining_ms > warmdown_ms:
        return 1.0
    return max(remaining_ms / max(warmdown_ms, 1e-9), 0.0)


def parse_step_lines(log_text: str, schedule: ScheduleConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for line in log_text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {k: float(v) for k, v in match.groupdict(default="").items() if k not in {"rest"}}
        row["step"] = int(match.group("step"))
        row["iterations"] = int(match.group("iterations"))
        rest = match.group("rest")
        for key, value in KV_RE.findall(rest):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        lr_mul = compute_lr_mul(row["train_time_ms"], schedule)
        row["lr_mul_reconstructed"] = lr_mul
        row["embed_lr_effective"] = schedule.embed_lr * lr_mul
        row["matrix_lr_effective"] = schedule.matrix_lr * lr_mul
        row["scalar_lr_effective"] = schedule.scalar_lr * lr_mul
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def parse_final_metrics(log_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for kind, _, bpb in FINAL_RE.findall(log_text):
        out[f"final_{kind}_bpb"] = float(bpb)
    stop = STOP_VAL_RE.findall(log_text)
    if stop:
        step, iterations, _, bpb, train_time_ms = stop[-1]
        out["stop_step"] = int(step)
        out["stop_iterations"] = int(iterations)
        out["stop_val_bpb"] = float(bpb)
        out["stop_train_time_ms"] = float(train_time_ms)
    return out


def plot_run(run_name: str, df: pd.DataFrame, finals: dict[str, float], out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(df["step"], df["train_loss"], label="train_loss", linewidth=1.8)
    if "ce" in df.columns:
        axes[0].plot(df["step"], df["ce"], label="ce", linewidth=1.4)
    if "jepa_pred" in df.columns:
        axes[0].plot(df["step"], df["jepa_pred"], label="jepa_pred", linewidth=1.0, alpha=0.9)
    if "jepa_sigreg" in df.columns:
        axes[0].plot(df["step"], df["jepa_sigreg"], label="jepa_sigreg", linewidth=1.0, alpha=0.9)
    axes[0].set_ylabel("Loss")
    axes[0].set_title(run_name)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(df["step"], df["embed_lr_effective"], label="embed_lr", linewidth=1.8)
    axes[1].plot(df["step"], df["matrix_lr_effective"], label="matrix_lr", linewidth=1.8)
    axes[1].plot(df["step"], df["scalar_lr_effective"], label="scalar_lr", linewidth=1.8)
    axes[1].set_ylabel("Effective LR")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(df["step"], df["tok_s"], label="tok_s", linewidth=1.8)
    if "turbo_qat_scale" in df.columns:
        axes[2].plot(df["step"], df["turbo_qat_scale"], label="turbo_qat_scale", linewidth=1.5)
    if "jepa_aux_scale" in df.columns:
        axes[2].plot(df["step"], df["jepa_aux_scale"], label="jepa_aux_scale", linewidth=1.5)
    if "jepa_pred_weight" in df.columns:
        axes[2].plot(df["step"], df["jepa_pred_weight"], label="jepa_pred_weight", linewidth=1.2)
    if "jepa_sigreg_weight" in df.columns:
        axes[2].plot(df["step"], df["jepa_sigreg_weight"], label="jepa_sigreg_weight", linewidth=1.2)
    axes[2].set_ylabel("Schedule / Throughput")
    axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right", ncol=2)

    footer = []
    if "stop_val_bpb" in finals:
        footer.append(f"stop_val_bpb={finals['stop_val_bpb']:.4f}")
    if "final_raw_export_ready_bpb" in finals:
        footer.append(f"final_raw={finals['final_raw_export_ready_bpb']:.4f}")
    if "final_int8_zlib_roundtrip_bpb" in finals:
        footer.append(f"final_quant={finals['final_int8_zlib_roundtrip_bpb']:.4f}")
    if footer:
        fig.text(0.01, 0.01, " | ".join(footer), fontsize=10)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / f"{run_name}_curves.png", dpi=180)
    plt.close(fig)


def plot_suite_comparison(run_frames: dict[str, pd.DataFrame], run_finals: dict[str, dict[str, float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    for run_name, df in run_frames.items():
        axes[0].plot(df["step"], df["train_loss"], label=run_name, linewidth=1.6)
    axes[0].set_title("Train Loss Comparison")
    axes[0].set_ylabel("train_loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    ranking = sorted(
        (
            (name, finals.get("stop_val_bpb"), finals.get("final_int8_zlib_roundtrip_bpb"))
            for name, finals in run_finals.items()
        ),
        key=lambda x: (math.inf if x[2] is None else x[2]),
    )
    labels = [x[0] for x in ranking]
    stop_vals = [x[1] for x in ranking]
    quant_vals = [x[2] for x in ranking]
    xpos = range(len(labels))
    axes[1].bar([x - 0.18 for x in xpos], stop_vals, width=0.36, label="stop_val_bpb")
    axes[1].bar([x + 0.18 for x in xpos], quant_vals, width=0.36, label="final_quant_bpb")
    axes[1].set_xticks(list(xpos), labels, rotation=10)
    axes[1].set_ylabel("BPB")
    axes[1].set_title("Validation / Final Quantized BPB")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "suite_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else suite_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(suite_dir)
    run_to_config = {}
    for run in manifest.get("runs", []):
        run_to_config[Path(run["config_path"]).stem] = Path(run["config_path"])

    run_frames: dict[str, pd.DataFrame] = {}
    run_finals: dict[str, dict[str, float]] = {}
    summary_rows: list[dict] = []

    for log_path in sorted((suite_dir / "dispatch_logs").glob("*.log")):
        run_name = log_path.stem
        cfg_env = load_config_env(run_to_config.get(run_name, Path()))
        schedule = schedule_from_env(cfg_env)
        text = log_path.read_text()
        df = parse_step_lines(text, schedule)
        if df.empty:
            continue
        finals = parse_final_metrics(text)
        run_frames[run_name] = df
        run_finals[run_name] = finals
        plot_run(run_name, df, finals, out_dir)
        df.to_csv(out_dir / f"{run_name}_parsed.csv", index=False)
        summary_rows.append(
            {
                "run": run_name,
                "max_step_logged": int(df["step"].max()),
                "stop_val_bpb": finals.get("stop_val_bpb"),
                "final_raw_export_ready_bpb": finals.get("final_raw_export_ready_bpb"),
                "final_int8_zlib_roundtrip_bpb": finals.get("final_int8_zlib_roundtrip_bpb"),
            }
        )

    if run_frames:
        plot_suite_comparison(run_frames, run_finals, out_dir)
    summary = pd.DataFrame(summary_rows).sort_values("final_int8_zlib_roundtrip_bpb", na_position="last")
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(summary.to_json(orient="records", indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
