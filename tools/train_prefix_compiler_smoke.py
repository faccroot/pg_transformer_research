#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

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


def band_means(nll: torch.Tensor, bands: list[tuple[int, int]]) -> list[dict[str, float | int | None]]:
    out: list[dict[str, float | int | None]] = []
    for start, end in bands:
        if start >= int(nll.numel()) or end < start:
            out.append({"start": start, "end": end, "mean_nll": None})
            continue
        segment = nll[start : end + 1]
        out.append({"start": start, "end": end, "mean_nll": float(segment.mean().item())})
    return out


class PrefixCompiler(nn.Module):
    def __init__(self, model_dim: int, prefix_slots: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(prefix_slots, model_dim) * 0.02)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, prev_hidden: torch.Tensor) -> torch.Tensor:
        # prev_hidden: [B, T, D]
        query = self.query.unsqueeze(0).expand(prev_hidden.size(0), -1, -1)
        keys = self.k_proj(prev_hidden)
        values = self.v_proj(prev_hidden)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(prev_hidden.size(-1))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=prev_hidden.dtype)
        prefix = torch.matmul(attn, values)
        return self.out_proj(prefix)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--train-examples", type=int, default=8)
    p.add_argument("--eval-examples", type=int, default=2)
    p.add_argument("--prefix-slots", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--focus-start", type=int, default=0)
    p.add_argument("--focus-end", type=int, default=127)
    p.add_argument("--device", default="auto")
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.max_windows)
    windows = build_probe_windows(int(val_tokens.numel() - 1), args.seq_len, args.stride)[: args.max_windows]
    bands = parse_bands(args.bands, args.seq_len)

    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        prev_x = None
        prev_hidden = None
        for window_start, window_len in windows:
            local = val_tokens[window_start:window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)
            _logits, hidden = forward_logits_with_prefix(model, x)
            if prev_x is not None and prev_hidden is not None:
                examples.append((prev_x.detach(), prev_hidden.detach(), x.detach(), y.detach()))
            prev_x = x
            prev_hidden = hidden

    train_examples = examples[: args.train_examples]
    eval_examples = examples[args.train_examples : args.train_examples + args.eval_examples]
    if not train_examples or not eval_examples:
        raise ValueError("Need both train and eval examples")

    compiler = PrefixCompiler(model.tok_emb.embedding_dim, args.prefix_slots).to(device)
    optimizer = torch.optim.AdamW(compiler.parameters(), lr=args.lr)
    focus_start = max(min(args.focus_start, args.seq_len - 1), 0)
    focus_end = max(min(args.focus_end, args.seq_len - 1), focus_start)
    train_history: list[float] = []

    for _step in range(args.train_steps):
        total_loss = 0.0
        for _prev_x, prev_hidden, x_cur, y_cur in train_examples:
            optimizer.zero_grad(set_to_none=True)
            prefix = compiler(prev_hidden)
            logits, _hidden = forward_logits_with_prefix(model, x_cur, prefix_embeddings=prefix)
            nll = per_token_nll(logits, y_cur)
            loss = nll[:, focus_start : focus_end + 1].mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
        train_history.append(total_loss / len(train_examples))

    results: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, (prev_x, prev_hidden, x_cur, y_cur) in enumerate(eval_examples):
            none_logits, _ = forward_logits_with_prefix(model, x_cur)
            none_nll = per_token_nll(none_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            raw_logits, _ = forward_logits_with_prefix(model, x_cur, prefix_input_ids=prev_x)
            raw_nll = per_token_nll(raw_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            prefix = compiler(prev_hidden)
            comp_logits, _ = forward_logits_with_prefix(model, x_cur, prefix_embeddings=prefix)
            comp_nll = per_token_nll(comp_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            results.append(
                {
                    "eval_index": idx,
                    "none_band_means": band_means(none_nll, bands),
                    "raw_prev_all_band_means": band_means(raw_nll, bands),
                    "compiler_band_means": band_means(comp_nll, bands),
                }
            )

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "prefix_slots": args.prefix_slots,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "focus_start": focus_start,
        "focus_end": focus_end,
        "train_history": train_history,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
