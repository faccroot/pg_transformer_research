#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project measured Mini runs into a hypothetical 8xH100 10-minute regime."
    )
    parser.add_argument("scenario", type=Path, help="Path to scenario JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_log_curve(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [math.log(tokens) for tokens, _ in points if tokens > 0]
    ys = [bpb for tokens, bpb in points if tokens > 0]
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return intercept, slope


def project_h100_tok_s(run: dict, model: dict) -> float | None:
    if "h100_tok_s" in run:
        return float(run["h100_tok_s"])
    if "mini_to_h100_multiplier" in model:
        return float(run["mini_tok_s"]) * float(model["mini_to_h100_multiplier"])
    if all(k in model for k in ("anchor_h100_tok_s", "anchor_params", "param_exponent")) and "params" in run:
        anchor_h100_tok_s = float(model["anchor_h100_tok_s"])
        anchor_params = float(model["anchor_params"])
        param_exponent = float(model["param_exponent"])
        params = float(run["params"])
        if anchor_params <= 0.0 or params <= 0.0:
            return None
        return anchor_h100_tok_s * (params / anchor_params) ** (-param_exponent)
    return None


def project_bpb(run: dict, h100_tokens: float, family_fit: tuple[float, float] | None, global_slope: float | None) -> float | None:
    measured_bpb = run.get("final_bpb")
    mini_tokens = float(run["mini_tok_s"]) * float(run["mini_wallclock_s"])
    if measured_bpb is None or mini_tokens <= 0.0 or h100_tokens <= 0.0:
        return None
    measured_bpb = float(measured_bpb)
    if family_fit is not None:
        intercept, slope = family_fit
        return intercept + slope * math.log(h100_tokens)
    if global_slope is not None:
        return measured_bpb + global_slope * math.log(h100_tokens / mini_tokens)
    return None


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    scenario = load_json(args.scenario)
    model = dict(scenario.get("throughput_model", {}))
    runs = list(scenario.get("runs", []))
    target_wallclock_s = float(model.get("target_wallclock_s", 600.0))
    global_log_slope = scenario.get("global_log_slope_bpb_per_ln_token")
    global_log_slope = None if global_log_slope is None else float(global_log_slope)

    family_points: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        family = str(run.get("family", "default"))
        if run.get("final_bpb") is None:
            continue
        tokens = float(run["mini_tok_s"]) * float(run["mini_wallclock_s"])
        family_points.setdefault(family, []).append((tokens, float(run["final_bpb"])))

    family_fits = {family: fit_log_curve(points) for family, points in family_points.items()}

    rows = []
    for run in runs:
        mini_tok_s = float(run["mini_tok_s"])
        mini_wallclock_s = float(run["mini_wallclock_s"])
        mini_tokens = mini_tok_s * mini_wallclock_s
        h100_tok_s = project_h100_tok_s(run, model)
        h100_tokens = None if h100_tok_s is None else h100_tok_s * target_wallclock_s
        required_h100_tok_s = mini_tokens / target_wallclock_s if target_wallclock_s > 0 else None
        token_ratio = None
        if h100_tokens is not None and mini_tokens > 0:
            token_ratio = h100_tokens / mini_tokens
        family = str(run.get("family", "default"))
        predicted_bpb = None
        if h100_tokens is not None:
            predicted_bpb = project_bpb(run, h100_tokens, family_fits.get(family), global_log_slope)
        rows.append(
            {
                "slug": run["slug"],
                "family": family,
                "params": run.get("params"),
                "mini_tok_s": mini_tok_s,
                "mini_wallclock_s": mini_wallclock_s,
                "mini_tokens": mini_tokens,
                "final_bpb": run.get("final_bpb"),
                "required_h100_tok_s_to_match_mini": required_h100_tok_s,
                "projected_h100_tok_s": h100_tok_s,
                "projected_h100_tokens": h100_tokens,
                "token_ratio_vs_mini": token_ratio,
                "projected_h100_bpb": predicted_bpb,
            }
        )

    if args.json:
        payload = {
            "scenario": str(args.scenario),
            "throughput_model": model,
            "global_log_slope_bpb_per_ln_token": global_log_slope,
            "family_fits": {
                family: None if fit is None else {"intercept": fit[0], "slope": fit[1]}
                for family, fit in family_fits.items()
            },
            "rows": rows,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"scenario: {args.scenario}")
    print(f"target_wallclock_s: {target_wallclock_s:.0f}")
    if "mini_to_h100_multiplier" in model:
        print(f"mini_to_h100_multiplier: {model['mini_to_h100_multiplier']}")
    elif "anchor_h100_tok_s" in model:
        print(
            "throughput_model:"
            f" anchor_h100_tok_s={model['anchor_h100_tok_s']}"
            f" anchor_params={model.get('anchor_params')}"
            f" param_exponent={model.get('param_exponent')}"
        )
    if global_log_slope is not None:
        print(f"global_log_slope_bpb_per_ln_token: {global_log_slope:+.6f}")
    print()
    print(
        "slug".ljust(18),
        "mini_tok/s".rjust(10),
        "mini_tokens".rjust(12),
        "need_h100".rjust(10),
        "proj_h100".rjust(10),
        "ratio".rjust(8),
        "mini_bpb".rjust(10),
        "proj_bpb".rjust(10),
    )
    for row in rows:
        print(
            str(row["slug"]).ljust(18),
            format_float(row["mini_tok_s"], 0).rjust(10),
            format_float(row["mini_tokens"], 0).rjust(12),
            format_float(row["required_h100_tok_s_to_match_mini"], 0).rjust(10),
            format_float(row["projected_h100_tok_s"], 0).rjust(10),
            format_float(row["token_ratio_vs_mini"], 2).rjust(8),
            format_float(None if row["final_bpb"] is None else float(row["final_bpb"]), 6).rjust(10),
            format_float(row["projected_h100_bpb"], 6).rjust(10),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
