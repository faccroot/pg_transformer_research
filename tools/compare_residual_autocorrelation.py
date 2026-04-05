#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two residual-autocorrelation JSON outputs.")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-label", default="")
    p.add_argument("--right-label", default="")
    p.add_argument("--result-json", default="")
    return p.parse_args()


def load_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def pick(d: dict[str, object], *path: str) -> float:
    cur: object = d
    for key in path:
        if not isinstance(cur, dict):
            raise KeyError(path)
        cur = cur[key]
    return float(cur)


def compare_metric(left: dict[str, object], right: dict[str, object], *path: str) -> dict[str, object]:
    lv = pick(left, *path)
    rv = pick(right, *path)
    return {
        "path": ".".join(path),
        "left": lv,
        "right": rv,
        "delta": rv - lv,
    }


def main() -> None:
    args = parse_args()
    left = load_json(args.left)
    right = load_json(args.right)
    left_label = args.left_label or str(left.get("label", "left"))
    right_label = args.right_label or str(right.get("label", "right"))

    metrics = [
        ("mean_nll",),
        ("nll_acf", "summary", "all", "mean"),
        ("nll_acf", "summary", "within_regime", "mean"),
        ("nll_acf", "summary", "cross_regime", "mean"),
        ("residual_modes", "expected_embedding", "acf_summary", "all", "mean"),
        ("residual_modes", "expected_embedding", "acf_summary", "within_regime", "mean"),
        ("residual_modes", "expected_embedding", "acf_summary", "cross_regime", "mean"),
        ("residual_modes", "expected_embedding", "acf_summary", "all", "positive_area"),
        ("residual_modes", "argmax_embedding", "acf_summary", "all", "mean"),
        ("residual_modes", "argmax_embedding", "acf_summary", "within_regime", "mean"),
        ("residual_modes", "argmax_embedding", "acf_summary", "cross_regime", "mean"),
        ("residual_modes", "argmax_embedding", "acf_summary", "all", "positive_area"),
    ]
    rows = [compare_metric(left, right, *metric) for metric in metrics]

    result = {
        "left_label": left_label,
        "right_label": right_label,
        "left": str(Path(args.left).expanduser().resolve()),
        "right": str(Path(args.right).expanduser().resolve()),
        "metrics": rows,
    }

    if args.result_json:
        out = Path(args.result_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{left_label} -> {right_label}")
    for row in rows:
        print(
            f"{row['path']}: "
            f"{row['left']:.6f} -> {row['right']:.6f} "
            f"(delta={row['delta']:+.6f})"
        )


if __name__ == "__main__":
    main()
