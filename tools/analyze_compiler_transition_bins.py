#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from text_prosody_features import BOUNDARY_STRENGTH_TO_ID, build_token_prosody_luts, extract_text_prosody_features
from torch_artifact_eval_utils import build_probe_windows, forward_logits_with_prefix, make_model_from_checkpoint, per_token_nll
from train_gpt import limit_validation_tokens, load_validation_tokens
from train_prefix_compiler_ablation import band_means, load_compiler_artifact, parse_bands


def build_transition_examples(
    model,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    stride: int,
    max_windows: int,
    device: torch.device,
) -> list[dict[str, object]]:
    windows = build_probe_windows(int(tokens.numel() - 1), seq_len, stride)[:max_windows]
    examples: list[dict[str, object]] = []
    with torch.no_grad():
        prev_pack: dict[str, object] | None = None
        for eval_index, (window_start, window_len) in enumerate(windows):
            local = tokens[window_start : window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)
            logits, hidden = forward_logits_with_prefix(model, x)
            nll = per_token_nll(logits, y)
            cur_pack = {
                "eval_index": int(eval_index),
                "window_start": int(window_start),
                "x": x.detach(),
                "y": y.detach(),
                "hidden": hidden.detach(),
                "nll": nll.detach(),
                "token_ids": x.squeeze(0).detach().cpu(),
            }
            if prev_pack is not None:
                examples.append(
                    {
                        "eval_index": int(len(examples)),
                        "prev_window_start": int(prev_pack["window_start"]),
                        "window_start": int(window_start),
                        "prev_x": prev_pack["x"],
                        "prev_hidden": prev_pack["hidden"],
                        "prev_nll": prev_pack["nll"],
                        "prev_token_ids": prev_pack["token_ids"],
                        "x_cur": cur_pack["x"],
                        "y_cur": cur_pack["y"],
                        "token_ids_cur": cur_pack["token_ids"],
                    }
                )
            prev_pack = cur_pack
    return examples


def classify_transition_bin(
    sp: spm.SentencePieceProcessor,
    prev_token_ids: torch.Tensor,
    curr_token_ids: torch.Tensor,
    *,
    seam_tokens: int,
    density_window: int,
    curr_none_nll: torch.Tensor,
) -> tuple[str, dict[str, float | int]]:
    prev_np = prev_token_ids.cpu().numpy().astype(np.int32, copy=False).reshape(-1)
    curr_np = curr_token_ids.cpu().numpy().astype(np.int32, copy=False).reshape(-1)
    prev_feat = extract_text_prosody_features(sp, prev_np, density_window=density_window)
    curr_feat = extract_text_prosody_features(sp, curr_np, density_window=density_window)

    tail = min(int(seam_tokens), int(prev_np.shape[0]))
    head = min(int(seam_tokens), int(curr_np.shape[0]))
    sentence_id = int(BOUNDARY_STRENGTH_TO_ID["sentence"])
    paragraph_id = int(BOUNDARY_STRENGTH_TO_ID["paragraph"])

    prev_tail_boundary_max = int(prev_feat.boundary_strength_ids[-tail:].max(initial=0))
    curr_head_boundary_max = int(curr_feat.boundary_strength_ids[:head].max(initial=0))
    prev_tail_sentence_frac = float((prev_feat.boundary_strength_ids[-tail:] >= sentence_id).mean())
    curr_head_sentence_frac = float((curr_feat.boundary_strength_ids[:head] >= sentence_id).mean())
    curr_head_paragraph_frac = float((curr_feat.boundary_strength_ids[:head] >= paragraph_id).mean())
    curr_head_punct_density = float(curr_feat.recent_punctuation_density[:head].mean())
    curr_head_noncontent_density = float(curr_feat.recent_noncontent_density[:head].mean())
    curr_head_surprise = float(curr_none_nll[:head].mean().item())
    quote_flip = int(curr_feat.quote_state[0] != prev_feat.quote_state[-1])

    strong_prev = prev_tail_boundary_max >= sentence_id or prev_tail_sentence_frac >= 0.125
    strong_curr = (
        curr_head_boundary_max >= sentence_id
        or curr_head_sentence_frac >= 0.125
        or curr_head_paragraph_frac >= 0.0625
    )

    if strong_prev and not strong_curr:
        bucket = "post_boundary"
    elif strong_curr or quote_flip or curr_head_punct_density >= 0.22 or curr_head_noncontent_density >= 0.60:
        bucket = "near_boundary"
    else:
        bucket = "stable"

    metrics = {
        "prev_tail_boundary_max": prev_tail_boundary_max,
        "curr_head_boundary_max": curr_head_boundary_max,
        "prev_tail_sentence_frac": prev_tail_sentence_frac,
        "curr_head_sentence_frac": curr_head_sentence_frac,
        "curr_head_paragraph_frac": curr_head_paragraph_frac,
        "curr_head_punct_density": curr_head_punct_density,
        "curr_head_noncontent_density": curr_head_noncontent_density,
        "curr_head_surprise": curr_head_surprise,
        "quote_flip": quote_flip,
    }
    return bucket, metrics


def compute_reset_scale(
    prev_token_ids: torch.Tensor,
    prev_nll: torch.Tensor,
    *,
    reset_prior_lut: np.ndarray | None,
    seam_tokens: int,
    reset_policy: str,
    reset_threshold: float,
    min_scale: float,
) -> tuple[float, dict[str, float]]:
    tail = min(int(seam_tokens), int(prev_token_ids.numel()))
    tail_ids = prev_token_ids[-tail:].cpu().numpy().astype(np.int64, copy=False)
    tail_surprise = float(prev_nll[:, -tail:].mean().item())
    if reset_prior_lut is None or reset_policy == "none":
        return 1.0, {"tail_reset_prior_max": 0.0, "tail_reset_prior_mean": 0.0, "tail_surprise_mean": tail_surprise}

    tail_reset = np.asarray(reset_prior_lut[tail_ids], dtype=np.float32).reshape(-1)
    prior_max = float(tail_reset.max(initial=0.0))
    prior_mean = float(tail_reset.mean()) if tail_reset.size else 0.0
    if reset_policy == "hard":
        scale = 0.0 if prior_max >= float(reset_threshold) else 1.0
    elif reset_policy == "soft":
        scale = max(float(min_scale), 1.0 - prior_max)
    else:
        raise ValueError(f"Unsupported reset_policy: {reset_policy}")
    return scale, {
        "tail_reset_prior_max": prior_max,
        "tail_reset_prior_mean": prior_mean,
        "tail_surprise_mean": tail_surprise,
    }


def summarize_rows(rows: list[dict[str, object]], bands: list[tuple[int, int]]) -> dict[str, object]:
    bins = ["stable", "near_boundary", "post_boundary"]
    summary: dict[str, object] = {}
    for bucket in bins:
        bucket_rows = [row for row in rows if row["transition_bin"] == bucket]
        bucket_payload = {"count": len(bucket_rows), "bands": []}
        for band_idx, (start, end) in enumerate(bands):
            none_vals = []
            raw_vals = []
            comp_vals = []
            for row in bucket_rows:
                none_vals.append(float(row["none_band_means"][band_idx]["mean_nll"]))
                raw_vals.append(float(row["raw_prev_all_band_means"][band_idx]["mean_nll"]))
                comp_vals.append(float(row["compiler_band_means"][band_idx]["mean_nll"]))
            band_payload: dict[str, object] = {"start": start, "end": end}
            if none_vals:
                band_payload.update(
                    {
                        "none_mean_nll": float(sum(none_vals) / len(none_vals)),
                        "raw_mean_nll": float(sum(raw_vals) / len(raw_vals)),
                        "compiler_mean_nll": float(sum(comp_vals) / len(comp_vals)),
                        "raw_minus_none": float(sum(r - n for r, n in zip(raw_vals, none_vals)) / len(none_vals)),
                        "compiler_minus_none": float(
                            sum(c - n for c, n in zip(comp_vals, none_vals)) / len(none_vals)
                        ),
                    }
                )
            else:
                band_payload.update(
                    {
                        "none_mean_nll": None,
                        "raw_mean_nll": None,
                        "compiler_mean_nll": None,
                        "raw_minus_none": None,
                        "compiler_minus_none": None,
                    }
                )
            bucket_payload["bands"].append(band_payload)
        summary[bucket] = bucket_payload
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compiler-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--eval-windows", type=int, default=6)
    p.add_argument("--seam-tokens", type=int, default=16)
    p.add_argument("--density-window", type=int, default=16)
    p.add_argument("--reset-policy", choices=["none", "hard", "soft"], default="none")
    p.add_argument("--reset-threshold", type=float, default=0.35)
    p.add_argument("--reset-min-scale", type=float, default=0.0)
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    compiler, compiler_payload = load_compiler_artifact(Path(args.compiler_path), device=device)
    sp = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path).expanduser()))
    reset_prior_lut = None
    if args.reset_policy != "none":
        reset_prior_lut = build_token_prosody_luts(sp).reset_prior_values
    bands = parse_bands(args.bands, args.seq_len)
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.eval_windows)
    examples = build_transition_examples(
        model,
        val_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_windows=args.eval_windows,
        device=device,
    )
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for example in examples:
            none_logits, _ = forward_logits_with_prefix(model, example["x_cur"])
            none_nll = per_token_nll(none_logits, example["y_cur"]).squeeze(0).detach().cpu().to(torch.float64)
            raw_logits, _ = forward_logits_with_prefix(model, example["x_cur"], prefix_input_ids=example["prev_x"])
            raw_nll = per_token_nll(raw_logits, example["y_cur"]).squeeze(0).detach().cpu().to(torch.float64)
            prefix = compiler(example["prev_hidden"], example["prev_nll"])
            reset_scale, reset_metrics = compute_reset_scale(
                example["prev_token_ids"],
                example["prev_nll"],
                reset_prior_lut=reset_prior_lut,
                seam_tokens=args.seam_tokens,
                reset_policy=args.reset_policy,
                reset_threshold=args.reset_threshold,
                min_scale=args.reset_min_scale,
            )
            prefix = prefix * float(reset_scale)
            comp_logits, _ = forward_logits_with_prefix(model, example["x_cur"], prefix_embeddings=prefix)
            comp_nll = per_token_nll(comp_logits, example["y_cur"]).squeeze(0).detach().cpu().to(torch.float64)

            transition_bin, transition_metrics = classify_transition_bin(
                sp,
                example["prev_token_ids"],
                example["token_ids_cur"],
                seam_tokens=args.seam_tokens,
                density_window=args.density_window,
                curr_none_nll=none_nll,
            )
            rows.append(
                {
                    "eval_index": int(example["eval_index"]),
                    "prev_window_start": int(example["prev_window_start"]),
                    "window_start": int(example["window_start"]),
                    "transition_bin": transition_bin,
                    "transition_metrics": transition_metrics,
                    "reset_policy": args.reset_policy,
                    "reset_scale": float(reset_scale),
                    "reset_metrics": reset_metrics,
                    "none_band_means": band_means(none_nll, bands),
                    "raw_prev_all_band_means": band_means(raw_nll, bands),
                    "compiler_band_means": band_means(comp_nll, bands),
                }
            )

    payload = {
        "checkpoint": args.checkpoint,
        "compiler_path": args.compiler_path,
        "compiler_config": compiler_payload.get("config", {}),
        "compiler_extra": compiler_payload.get("extra", {}),
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser()),
        "device": str(device),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "eval_windows": int(args.eval_windows),
        "seam_tokens": int(args.seam_tokens),
        "density_window": int(args.density_window),
        "reset_policy": args.reset_policy,
        "reset_threshold": float(args.reset_threshold),
        "reset_min_scale": float(args.reset_min_scale),
        "summary_by_bin": summarize_rows(rows, bands),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
