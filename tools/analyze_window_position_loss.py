#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torch_artifact_eval_utils import build_probe_windows, forward_logits_with_prefix, make_model_from_checkpoint, per_token_nll
from train_gpt import limit_validation_tokens, load_validation_tokens


def parse_bands(raw: str, seq_len: int) -> list[tuple[int, int]]:
    if not raw.strip():
        return [(0, min(seq_len - 1, 31)), (32, min(seq_len - 1, 63)), (64, min(seq_len - 1, 127)), (128, seq_len - 1)]
    bands: list[tuple[int, int]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        start_s, end_s = piece.split("-", 1)
        bands.append((int(start_s), min(seq_len - 1, int(end_s))))
    return bands


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255,256-511,512-1023")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.max_windows)
    windows = build_probe_windows(int(val_tokens.numel() - 1), args.seq_len, args.stride)[: args.max_windows]
    bands = parse_bands(args.bands, args.seq_len)

    pos_loss_sum = torch.zeros((args.seq_len,), dtype=torch.float64)
    pos_count = torch.zeros((args.seq_len,), dtype=torch.int64)

    model.eval()
    with torch.inference_mode():
        for window_start, window_len in windows:
            local = val_tokens[window_start:window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _hidden = forward_logits_with_prefix(model, x)
                nll = per_token_nll(logits, y).squeeze(0).detach().cpu().to(torch.float64)
            pos_loss_sum[:window_len] += nll
            pos_count[:window_len] += 1

    pos_mean: list[float | None] = []
    for idx in range(args.seq_len):
        if int(pos_count[idx].item()) <= 0:
            pos_mean.append(None)
        else:
            pos_mean.append(float((pos_loss_sum[idx] / pos_count[idx].to(torch.float64)).item()))

    band_means: list[dict[str, object]] = []
    for start, end in bands:
        if start >= args.seq_len or end < start:
            continue
        counts = pos_count[start : end + 1]
        mass = int(counts.sum().item())
        if mass <= 0:
            mean_value = None
        else:
            mean_value = float((pos_loss_sum[start : end + 1].sum() / counts.sum().to(torch.float64)).item())
        band_means.append({"start": start, "end": end, "count": mass, "mean_nll": mean_value})

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "num_windows": len(windows),
        "band_means": band_means,
        "position_mean_nll": pos_mean,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
