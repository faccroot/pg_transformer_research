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


def band_means(nll: torch.Tensor, bands: list[tuple[int, int]]) -> list[dict[str, float | int | None]]:
    out: list[dict[str, float | int | None]] = []
    for start, end in bands:
        if start >= int(nll.numel()) or end < start:
            out.append({"start": start, "end": end, "mean_nll": None})
            continue
        segment = nll[start : end + 1]
        out.append({"start": start, "end": end, "mean_nll": float(segment.mean().item())})
    return out


def evaluate_mode(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    prefix_input_ids: torch.Tensor | None = None,
    prefix_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, hidden = forward_logits_with_prefix(
        model,
        x,
        prefix_input_ids=prefix_input_ids,
        prefix_embeddings=prefix_embeddings,
    )
    nll = per_token_nll(logits, y).squeeze(0).detach().cpu().to(torch.float64)
    return nll, hidden.detach()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max-windows", type=int, default=3)
    p.add_argument("--prefix-slots", type=int, default=32)
    p.add_argument("--opt-steps", type=int, default=20)
    p.add_argument("--opt-lr", type=float, default=0.05)
    p.add_argument("--focus-start", type=int, default=0)
    p.add_argument("--focus-end", type=int, default=127)
    p.add_argument("--init", default="meanpool", choices=["zeros", "meanpool"])
    p.add_argument("--device", default="auto")
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255,256-511")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    for param in model.parameters():
        param.requires_grad_(False)

    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.max_windows)
    windows = build_probe_windows(int(val_tokens.numel() - 1), args.seq_len, args.stride)[: args.max_windows]
    bands = parse_bands(args.bands, args.seq_len)

    per_window: list[dict[str, object]] = []
    with torch.no_grad():
        prev_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for window_start, window_len in windows:
            local = val_tokens[window_start:window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)
            none_nll, hidden = evaluate_mode(model, x, y)
            prev_cache.append((x.detach(), y.detach(), hidden.detach()))

    for window_idx in range(1, len(prev_cache)):
        x_prev, _y_prev, hidden_prev = prev_cache[window_idx - 1]
        x_cur, y_cur, _hidden_cur = prev_cache[window_idx]

        none_nll, _ = evaluate_mode(model, x_cur, y_cur)
        raw_nll, _ = evaluate_mode(model, x_cur, y_cur, prefix_input_ids=x_prev)

        if args.init == "zeros":
            init_prefix = torch.zeros((1, args.prefix_slots, model.tok_emb.embedding_dim), device=device, dtype=torch.float32)
        else:
            init_prefix = meanpool_slots(hidden_prev.squeeze(0).to(device=device, dtype=torch.float32), args.prefix_slots).unsqueeze(0)
        prefix = torch.nn.Parameter(init_prefix)
        optimizer = torch.optim.Adam([prefix], lr=args.opt_lr)
        focus_start = max(min(args.focus_start, x_cur.size(1) - 1), 0)
        focus_end = max(min(args.focus_end, x_cur.size(1) - 1), focus_start)

        history: list[float] = []
        for _step in range(args.opt_steps):
            optimizer.zero_grad(set_to_none=True)
            logits, _hidden = forward_logits_with_prefix(model, x_cur, prefix_embeddings=prefix)
            nll = per_token_nll(logits, y_cur)
            loss = nll[:, focus_start : focus_end + 1].mean()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().item()))

        opt_nll, _ = evaluate_mode(model, x_cur, y_cur, prefix_embeddings=prefix.detach())
        per_window.append(
            {
                "window_index": window_idx,
                "focus_start": focus_start,
                "focus_end": focus_end,
                "none_band_means": band_means(none_nll, bands),
                "raw_prev_all_band_means": band_means(raw_nll, bands),
                "optimized_prefix_band_means": band_means(opt_nll, bands),
                "opt_history": history,
            }
        )

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "prefix_slots": args.prefix_slots,
        "opt_steps": args.opt_steps,
        "opt_lr": args.opt_lr,
        "focus_start": args.focus_start,
        "focus_end": args.focus_end,
        "init": args.init,
        "per_window": per_window,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
