#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def meanpool_slots(hidden: torch.Tensor, slots: int) -> torch.Tensor:
    seq_len = int(hidden.size(0))
    slots = max(min(slots, seq_len), 1)
    edges = torch.linspace(0, seq_len, slots + 1, dtype=torch.int64, device=hidden.device)
    out: list[torch.Tensor] = []
    for i in range(slots):
        start = int(edges[i].item())
        end = max(int(edges[i + 1].item()), start + 1)
        out.append(hidden[start:end].mean(dim=0))
    return torch.stack(out, dim=0)


def svd_slots(hidden: torch.Tensor, slots: int) -> torch.Tensor:
    seq_len, dim = hidden.shape
    slots = max(min(slots, seq_len, dim), 1)
    mean = hidden.mean(dim=0, keepdim=True)
    centered = hidden - mean
    u, s, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    out = [mean.squeeze(0)]
    usable = max(min(slots - 1, int(vh.size(0))), 0)
    if usable > 0:
        scale = s[:usable] / math.sqrt(max(seq_len, 1))
        comps = vh[:usable] * scale.unsqueeze(1)
        for row in comps:
            out.append(mean.squeeze(0) + row.to(dtype=hidden.dtype, device=hidden.device))
    while len(out) < slots:
        out.append(mean.squeeze(0))
    return torch.stack(out[:slots], dim=0)


def toploss_slots(hidden: torch.Tensor, nll: torch.Tensor, slots: int) -> torch.Tensor:
    slots = max(min(slots, int(hidden.size(0))), 1)
    topk = torch.topk(nll.float(), k=slots, largest=True)
    idx = torch.sort(topk.indices).values
    return hidden[idx]


def band_summary(loss_sum: torch.Tensor, count: torch.Tensor, bands: list[tuple[int, int]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for start, end in bands:
        counts = count[start : end + 1]
        total = int(counts.sum().item())
        if total <= 0:
            mean_value = None
        else:
            mean_value = float((loss_sum[start : end + 1].sum() / counts.sum().to(torch.float64)).item())
        out.append({"start": start, "end": end, "count": total, "mean_nll": mean_value})
    return out


def extract_mode_prefix(
    mode: str,
    prev_x: torch.Tensor,
    prev_hidden: torch.Tensor,
    prev_nll: torch.Tensor,
    *,
    prefix_slots: int,
    raw_tail_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mode == "none":
        return None, None
    if mode == "raw_prev_all":
        return prev_x, None
    if mode == "raw_tail":
        tail = max(min(raw_tail_tokens, int(prev_x.size(1))), 1)
        return prev_x[:, -tail:], None
    seq_hidden = prev_hidden.squeeze(0)
    seq_nll = prev_nll.squeeze(0)
    if mode == "hidden_prev_all":
        return None, seq_hidden
    if mode == "hidden_tail":
        tail = max(min(prefix_slots, int(seq_hidden.size(0))), 1)
        return None, seq_hidden[-tail:]
    if mode == "hidden_meanpool":
        return None, meanpool_slots(seq_hidden, prefix_slots)
    if mode == "hidden_svd":
        return None, svd_slots(seq_hidden, prefix_slots)
    if mode == "hidden_toploss":
        return None, toploss_slots(seq_hidden, seq_nll, prefix_slots)
    raise ValueError(f"Unsupported mode {mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--prefix-slots", type=int, default=32)
    p.add_argument("--raw-tail-tokens", type=int, default=32)
    p.add_argument("--modes", default="none,raw_tail,raw_prev_all,hidden_prev_all,hidden_tail,hidden_meanpool,hidden_svd,hidden_toploss")
    p.add_argument("--device", default="auto")
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255,256-511,512-1023")
    p.add_argument("--out", default="")
    args = p.parse_args()

    modes = [piece.strip() for piece in args.modes.split(",") if piece.strip()]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.max_windows)
    windows = build_probe_windows(int(val_tokens.numel() - 1), args.seq_len, args.stride)[: args.max_windows]
    bands = parse_bands(args.bands, args.seq_len)

    results = {
        mode: {
            "loss_sum": torch.zeros((args.seq_len,), dtype=torch.float64),
            "count": torch.zeros((args.seq_len,), dtype=torch.int64),
        }
        for mode in modes
    }

    model.eval()
    prev_x: torch.Tensor | None = None
    prev_hidden: torch.Tensor | None = None
    prev_nll: torch.Tensor | None = None

    with torch.inference_mode():
        for window_idx, (window_start, window_len) in enumerate(windows):
            local = val_tokens[window_start:window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                base_logits, base_hidden = forward_logits_with_prefix(model, x)
                base_nll = per_token_nll(base_logits, y)
            base_nll_cpu = base_nll.squeeze(0).detach().cpu().to(torch.float64)

            for mode in modes:
                if mode == "none" or prev_x is None or prev_hidden is None or prev_nll is None:
                    nll = base_nll_cpu
                else:
                    prefix_input_ids, prefix_embeddings = extract_mode_prefix(
                        mode,
                        prev_x,
                        prev_hidden,
                        prev_nll,
                        prefix_slots=args.prefix_slots,
                        raw_tail_tokens=args.raw_tail_tokens,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits, _hidden = forward_logits_with_prefix(
                            model,
                            x,
                            prefix_input_ids=prefix_input_ids,
                            prefix_embeddings=prefix_embeddings,
                        )
                        nll = per_token_nll(logits, y).squeeze(0).detach().cpu().to(torch.float64)
                results[mode]["loss_sum"][:window_len] += nll
                results[mode]["count"][:window_len] += 1

            prev_x = x
            prev_hidden = base_hidden.detach()
            prev_nll = base_nll.detach()

    none_band = band_summary(results["none"]["loss_sum"], results["none"]["count"], bands) if "none" in results else []
    payload: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "num_windows": len(windows),
        "prefix_slots": args.prefix_slots,
        "raw_tail_tokens": args.raw_tail_tokens,
        "modes": {},
    }

    for mode in modes:
        mode_bands = band_summary(results[mode]["loss_sum"], results[mode]["count"], bands)
        delta_vs_none: list[dict[str, object]] = []
        if none_band:
            for cur, base in zip(mode_bands, none_band):
                delta = None
                if cur["mean_nll"] is not None and base["mean_nll"] is not None:
                    delta = float(cur["mean_nll"] - base["mean_nll"])
                delta_vs_none.append(
                    {
                        "start": cur["start"],
                        "end": cur["end"],
                        "delta_mean_nll": delta,
                    }
                )
        payload["modes"][mode] = {
            "band_means": mode_bands,
            "delta_vs_none": delta_vs_none,
        }

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
