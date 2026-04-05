#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep residual-autocorrelation analysis over multiple eval context lengths.")
    p.add_argument("--artifact", required=True)
    p.add_argument("--config-json", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--eval-seq-lens", required=True, help="Comma-separated eval sequence lengths, e.g. 128,256,512,1024")
    p.add_argument("--label", default="")
    p.add_argument("--trainer-module", default="")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--data-path", default="")
    p.add_argument("--val-max-seqs", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=64)
    p.add_argument("--residual-mode", choices=("expected", "argmax", "both"), default="both")
    p.add_argument("--regime-layer", type=int, default=-1)
    p.add_argument("--regime-cosine-threshold", type=float, default=None)
    p.add_argument("--regime-cosine-quantile", type=float, default=0.05)
    p.add_argument("--regime-min-segment-length", type=int, default=16)
    p.add_argument("--layerwise-layers", default="")
    p.add_argument("--transition-window", type=int, default=0)
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--top-k-transitions", type=int, default=10)
    p.add_argument("--preview-radius", type=int, default=24)
    return p.parse_args()


def parse_seq_lens(raw: str) -> list[int]:
    out: list[int] = []
    for piece in str(raw or "").split(","):
        item = piece.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("need at least one eval sequence length")
    return out


def maybe_add(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([flag, str(value)])


def metric_at(payload: dict[str, object], path: list[str]):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "mean_nll": payload.get("mean_nll"),
        "nll_acf_all_mean": metric_at(payload, ["nll_acf", "summary", "all", "mean"]),
        "nll_acf_within_mean": metric_at(payload, ["nll_acf", "summary", "within_regime", "mean"]),
        "nll_acf_cross_mean": metric_at(payload, ["nll_acf", "summary", "cross_regime", "mean"]),
        "expected_acf_all_mean": metric_at(payload, ["residual_modes", "expected_embedding", "acf_summary", "all", "mean"]),
        "expected_acf_within_mean": metric_at(payload, ["residual_modes", "expected_embedding", "acf_summary", "within_regime", "mean"]),
        "expected_acf_cross_mean": metric_at(payload, ["residual_modes", "expected_embedding", "acf_summary", "cross_regime", "mean"]),
        "argmax_acf_all_mean": metric_at(payload, ["residual_modes", "argmax_embedding", "acf_summary", "all", "mean"]),
        "argmax_acf_within_mean": metric_at(payload, ["residual_modes", "argmax_embedding", "acf_summary", "within_regime", "mean"]),
        "argmax_acf_cross_mean": metric_at(payload, ["residual_modes", "argmax_embedding", "acf_summary", "cross_regime", "mean"]),
        "transition_count": metric_at(payload, ["regime", "transition_count"]),
        "mean_segment_length": metric_at(payload, ["regime", "mean_segment_length"]),
        "transition_window_mean_nll": metric_at(payload, ["transition_window", "nll", "transition_window", "mean"]),
        "outside_window_mean_nll": metric_at(payload, ["transition_window", "nll", "outside_window", "mean"]),
    }


def main() -> None:
    args = parse_args()
    eval_seq_lens = parse_seq_lens(args.eval_seq_lens)
    root = Path(__file__).resolve().parent
    analyzer = root / "analyze_residual_autocorrelation.py"
    out_path = Path(args.result_json).expanduser().resolve()
    out_dir = out_path.parent / f"{out_path.stem}_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for eval_seq_len in eval_seq_lens:
        run_json = out_dir / f"residual_ctx_{int(eval_seq_len)}.json"
        cmd = [
            sys.executable,
            str(analyzer),
            "--artifact",
            str(args.artifact),
            "--config-json",
            str(args.config_json),
            "--result-json",
            str(run_json),
            "--eval-seq-len",
            str(int(eval_seq_len)),
            "--label",
            f"{args.label or 'residual_ctx'}_L{int(eval_seq_len)}",
            "--cache-variant",
            str(args.cache_variant),
            "--train-shards",
            str(int(args.train_shards)),
            "--analysis-max-batches",
            str(int(args.analysis_max_batches)),
            "--max-lag",
            str(int(args.max_lag)),
            "--residual-mode",
            str(args.residual_mode),
            "--regime-layer",
            str(int(args.regime_layer)),
            "--regime-cosine-quantile",
            str(float(args.regime_cosine_quantile)),
            "--regime-min-segment-length",
            str(int(args.regime_min_segment_length)),
            "--layerwise-layers",
            str(args.layerwise_layers),
            "--transition-window",
            str(int(args.transition_window)),
            "--top-k-tokens",
            str(int(args.top_k_tokens)),
            "--top-k-transitions",
            str(int(args.top_k_transitions)),
            "--preview-radius",
            str(int(args.preview_radius)),
        ]
        maybe_add(cmd, "--trainer-module", args.trainer_module)
        maybe_add(cmd, "--tokenizer-path", args.tokenizer_path)
        maybe_add(cmd, "--data-path", args.data_path)
        maybe_add(cmd, "--val-max-seqs", args.val_max_seqs)
        maybe_add(cmd, "--eval-stride", args.eval_stride)
        maybe_add(cmd, "--eval-batch-seqs", args.eval_batch_seqs)
        maybe_add(cmd, "--regime-cosine-threshold", args.regime_cosine_threshold)
        subprocess.run(cmd, check=True)
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        runs.append(
            {
                "eval_seq_len": int(eval_seq_len),
                "result_json": str(run_json),
                "summary": build_summary(payload),
            }
        )

    result = {
        "artifact": str(Path(args.artifact).expanduser().resolve()),
        "config_json": str(Path(args.config_json).expanduser().resolve()),
        "trainer_module": str(args.trainer_module or "train_gpt_mlx"),
        "eval_seq_lens": eval_seq_lens,
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
