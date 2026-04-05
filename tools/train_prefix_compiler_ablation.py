#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_artifact_eval_utils import build_probe_windows, forward_logits_with_prefix, make_model_from_checkpoint, per_token_nll
from train_gpt import limit_validation_tokens, load_data_shard, load_validation_tokens


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

    def forward(self, prev_hidden: torch.Tensor, prev_nll: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(prev_hidden.size(0), -1, -1)
        keys = self.k_proj(prev_hidden)
        values = self.v_proj(prev_hidden)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(prev_hidden.size(-1))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=prev_hidden.dtype)
        prefix = torch.matmul(attn, values)
        return self.out_proj(prefix)


class TypedPrefixCompiler(nn.Module):
    def __init__(
        self,
        model_dim: int,
        prefix_slots: int,
        *,
        block_count: int,
        topk_blocks: int,
        utility_source: str,
        hierarchical: bool,
    ):
        super().__init__()
        self.block_count = max(int(block_count), 1)
        self.topk_blocks = max(int(topk_blocks), 1)
        self.utility_source = utility_source
        self.hierarchical = bool(hierarchical)
        self.query = nn.Parameter(torch.randn(prefix_slots, model_dim) * 0.02)
        self.summary_proj = nn.Linear(model_dim * 2, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def _segments(self, seq_len: int, count: int, device: torch.device) -> list[tuple[int, int]]:
        count = max(min(int(count), seq_len), 1)
        edges = torch.linspace(0, seq_len, count + 1, dtype=torch.int64, device=device)
        out: list[tuple[int, int]] = []
        for i in range(count):
            start = int(edges[i].item())
            end = max(int(edges[i + 1].item()), start + 1)
            out.append((start, end))
        return out

    def _segment_feature(self, hidden_seg: torch.Tensor) -> torch.Tensor:
        mean = hidden_seg.mean(dim=0)
        last = hidden_seg[-1]
        return torch.cat([mean, last], dim=0)

    def _utility_scores(self, hidden: torch.Tensor, nll: torch.Tensor, segments: list[tuple[int, int]]) -> torch.Tensor:
        hidden_scores = []
        surprise_scores = []
        for start, end in segments:
            hidden_seg = hidden[start:end]
            nll_seg = nll[start:end]
            hidden_scores.append(hidden_seg.mean(dim=0).float().norm())
            surprise_scores.append(nll_seg.float().mean())
        hidden_tensor = torch.stack(hidden_scores)
        surprise_tensor = torch.stack(surprise_scores)
        if self.utility_source == "hidden":
            return hidden_tensor
        if self.utility_source == "surprise":
            return surprise_tensor
        hidden_z = (hidden_tensor - hidden_tensor.mean()) / hidden_tensor.std(unbiased=False).clamp_min(1e-6)
        surprise_z = (surprise_tensor - surprise_tensor.mean()) / surprise_tensor.std(unbiased=False).clamp_min(1e-6)
        return hidden_z + surprise_z

    def _build_sources_single(self, hidden: torch.Tensor, nll: torch.Tensor) -> torch.Tensor:
        seq_len, model_dim = hidden.shape
        micro_segments = self._segments(seq_len, self.block_count, hidden.device)
        scores = self._utility_scores(hidden, nll, micro_segments)
        keep = min(self.topk_blocks, len(micro_segments))
        selected_idx = torch.topk(scores, k=keep, largest=True).indices.sort().values.tolist()
        features = [self._segment_feature(hidden[micro_segments[idx][0] : micro_segments[idx][1]]) for idx in selected_idx]

        if self.hierarchical:
            macro_count = max(2, min(4, self.block_count))
            macro_segments = self._segments(seq_len, macro_count, hidden.device)
            features.extend(self._segment_feature(hidden[start:end]) for start, end in macro_segments)
            features.append(self._segment_feature(hidden))

        stacked = torch.stack(features, dim=0)
        summaries = self.summary_proj(stacked)
        return F.rms_norm(summaries, (model_dim,))

    def forward(self, prev_hidden: torch.Tensor, prev_nll: torch.Tensor) -> torch.Tensor:
        batch_sources = []
        max_sources = 0
        for batch_idx in range(prev_hidden.size(0)):
            source = self._build_sources_single(prev_hidden[batch_idx], prev_nll[batch_idx])
            batch_sources.append(source)
            max_sources = max(max_sources, int(source.size(0)))
        padded = []
        for source in batch_sources:
            if int(source.size(0)) < max_sources:
                pad = source.new_zeros((max_sources - int(source.size(0)), source.size(1)))
                source = torch.cat([source, pad], dim=0)
            padded.append(source)
        source_tensor = torch.stack(padded, dim=0)
        query = self.query.unsqueeze(0).expand(source_tensor.size(0), -1, -1)
        keys = self.k_proj(source_tensor)
        values = self.v_proj(source_tensor)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(source_tensor.size(-1))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=source_tensor.dtype)
        prefix = torch.matmul(attn, values)
        return self.out_proj(prefix)


def build_compiler(
    model_dim: int,
    *,
    prefix_slots: int,
    compiler_mode: str,
    block_count: int = 8,
    topk_blocks: int = 2,
    utility_source: str = "hybrid",
    hierarchical: bool = False,
) -> nn.Module:
    if compiler_mode == "full":
        return PrefixCompiler(model_dim, prefix_slots)
    return TypedPrefixCompiler(
        model_dim,
        prefix_slots,
        block_count=block_count,
        topk_blocks=topk_blocks,
        utility_source=utility_source,
        hierarchical=bool(hierarchical),
    )


def compiler_arch_config(
    *,
    model_dim: int,
    prefix_slots: int,
    compiler_mode: str,
    block_count: int,
    topk_blocks: int,
    utility_source: str,
    hierarchical: int | bool,
) -> dict[str, int | str]:
    return {
        "model_dim": int(model_dim),
        "prefix_slots": int(prefix_slots),
        "compiler_mode": str(compiler_mode),
        "block_count": int(block_count),
        "topk_blocks": int(topk_blocks),
        "utility_source": str(utility_source),
        "hierarchical": int(bool(hierarchical)),
    }


def save_compiler_artifact(
    path: Path,
    compiler: nn.Module,
    config: dict[str, int | str],
    *,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "config": dict(config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in compiler.state_dict().items()},
    }
    if extra:
        payload["extra"] = dict(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_compiler_artifact(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    config = dict(payload["config"])
    compiler = build_compiler(
        int(config["model_dim"]),
        prefix_slots=int(config["prefix_slots"]),
        compiler_mode=str(config["compiler_mode"]),
        block_count=int(config["block_count"]),
        topk_blocks=int(config["topk_blocks"]),
        utility_source=str(config["utility_source"]),
        hierarchical=bool(int(config["hierarchical"])),
    )
    compiler.load_state_dict(payload["state_dict"], strict=True)
    compiler.to(device)
    compiler.eval()
    return compiler, payload


def load_train_prefix_tokens(pattern: str, needed_tokens: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    chunks: list[torch.Tensor] = []
    total = 0
    for file in files:
        shard = load_data_shard(file)
        chunks.append(shard)
        total += int(shard.numel())
        if total >= needed_tokens:
            break
    if total < needed_tokens:
        raise ValueError(f"Training data too short for requested window budget: needed {needed_tokens}, got {total}")
    tokens = torch.cat(chunks).contiguous()
    return tokens[:needed_tokens]


def build_examples_from_tokens(
    model,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    stride: int,
    max_windows: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    windows = build_probe_windows(int(tokens.numel() - 1), seq_len, stride)[:max_windows]
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        prev_x = None
        prev_hidden = None
        prev_nll = None
        for window_start, window_len in windows:
            local = tokens[window_start:window_start + window_len + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(1, window_len)
            y = local[1:].reshape(1, window_len)
            logits, hidden = forward_logits_with_prefix(model, x)
            nll = per_token_nll(logits, y)
            if prev_x is not None and prev_hidden is not None and prev_nll is not None:
                examples.append((prev_x.detach(), prev_hidden.detach(), prev_nll.detach(), x.detach(), y.detach()))
            prev_x = x
            prev_hidden = hidden
            prev_nll = nll
    return examples


def evaluate_examples(model, compiler, examples, bands):
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, (prev_x, prev_hidden, prev_nll, x_cur, y_cur) in enumerate(examples):
            none_logits, _ = forward_logits_with_prefix(model, x_cur)
            none_nll = per_token_nll(none_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            raw_logits, _ = forward_logits_with_prefix(model, x_cur, prefix_input_ids=prev_x)
            raw_nll = per_token_nll(raw_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            prefix = compiler(prev_hidden, prev_nll)
            comp_logits, _ = forward_logits_with_prefix(model, x_cur, prefix_embeddings=prefix)
            comp_nll = per_token_nll(comp_logits, y_cur).squeeze(0).detach().cpu().to(torch.float64)
            rows.append(
                {
                    "eval_index": idx,
                    "none_band_means": band_means(none_nll, bands),
                    "raw_prev_all_band_means": band_means(raw_nll, bands),
                    "compiler_band_means": band_means(comp_nll, bands),
                }
            )
    return rows


def summarize_deltas(results: list[dict[str, object]]) -> list[dict[str, object]]:
    if not results:
        return []
    num_bands = len(results[0]["none_band_means"])
    out: list[dict[str, object]] = []
    for band_idx in range(num_bands):
        none_vals = []
        raw_vals = []
        comp_vals = []
        start = results[0]["none_band_means"][band_idx]["start"]
        end = results[0]["none_band_means"][band_idx]["end"]
        for row in results:
            none_vals.append(float(row["none_band_means"][band_idx]["mean_nll"]))
            raw_vals.append(float(row["raw_prev_all_band_means"][band_idx]["mean_nll"]))
            comp_vals.append(float(row["compiler_band_means"][band_idx]["mean_nll"]))
        out.append(
            {
                "start": start,
                "end": end,
                "compiler_minus_none": float(sum(c - n for c, n in zip(comp_vals, none_vals)) / len(none_vals)),
                "raw_minus_none": float(sum(r - n for r, n in zip(raw_vals, none_vals)) / len(none_vals)),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-files", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--train-windows", type=int, default=12)
    p.add_argument("--eval-windows", type=int, default=4)
    p.add_argument("--prefix-slots", type=int, default=16)
    p.add_argument("--compiler-mode", choices=["full", "typed"], default="full")
    p.add_argument("--block-count", type=int, default=8)
    p.add_argument("--topk-blocks", type=int, default=2)
    p.add_argument("--utility-source", choices=["hidden", "surprise", "hybrid"], default="hybrid")
    p.add_argument("--hierarchical", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--focus-start", type=int, default=0)
    p.add_argument("--focus-end", type=int, default=127)
    p.add_argument("--device", default="auto")
    p.add_argument("--bands", default="0-31,32-63,64-127,128-255")
    p.add_argument("--save-compiler-path", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    needed_train_tokens = args.stride * max(args.train_windows - 1, 0) + args.seq_len + 1
    train_tokens = load_train_prefix_tokens(args.train_files, needed_train_tokens)
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.eval_windows)
    bands = parse_bands(args.bands, args.seq_len)

    train_examples = build_examples_from_tokens(
        model, train_tokens, seq_len=args.seq_len, stride=args.stride, max_windows=args.train_windows, device=device
    )
    eval_examples = build_examples_from_tokens(
        model, val_tokens, seq_len=args.seq_len, stride=args.stride, max_windows=args.eval_windows, device=device
    )
    if not train_examples or not eval_examples:
        raise ValueError("Need both train and eval examples")

    arch_config = compiler_arch_config(
        model_dim=model.tok_emb.embedding_dim,
        prefix_slots=args.prefix_slots,
        compiler_mode=args.compiler_mode,
        block_count=args.block_count,
        topk_blocks=args.topk_blocks,
        utility_source=args.utility_source,
        hierarchical=args.hierarchical,
    )
    compiler = build_compiler(
        model.tok_emb.embedding_dim,
        prefix_slots=args.prefix_slots,
        compiler_mode=args.compiler_mode,
        block_count=args.block_count,
        topk_blocks=args.topk_blocks,
        utility_source=args.utility_source,
        hierarchical=bool(args.hierarchical),
    ).to(device)
    optimizer = torch.optim.AdamW(compiler.parameters(), lr=args.lr)
    focus_start = max(min(args.focus_start, args.seq_len - 1), 0)
    focus_end = max(min(args.focus_end, args.seq_len - 1), focus_start)
    train_history: list[float] = []

    for _step in range(args.train_steps):
        total_loss = 0.0
        for _prev_x, prev_hidden, prev_nll, x_cur, y_cur in train_examples:
            optimizer.zero_grad(set_to_none=True)
            prefix = compiler(prev_hidden, prev_nll)
            logits, _hidden = forward_logits_with_prefix(model, x_cur, prefix_embeddings=prefix)
            nll = per_token_nll(logits, y_cur)
            loss = nll[:, focus_start : focus_end + 1].mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
        train_history.append(total_loss / len(train_examples))

    compiler.eval()
    if args.save_compiler_path:
        save_compiler_artifact(
            Path(args.save_compiler_path),
            compiler,
            arch_config,
            extra={
                "checkpoint": args.checkpoint,
                "seq_len": int(args.seq_len),
                "stride": int(args.stride),
                "train_windows": int(args.train_windows),
                "eval_windows": int(args.eval_windows),
                "focus_start": int(focus_start),
                "focus_end": int(focus_end),
                "train_steps": int(args.train_steps),
                "lr": float(args.lr),
                "train_history": list(train_history),
            },
        )

    results = evaluate_examples(model, compiler, eval_examples, bands)
    summary = summarize_deltas(results)
    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "train_windows": args.train_windows,
        "eval_windows": args.eval_windows,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "prefix_slots": args.prefix_slots,
        "compiler_mode": args.compiler_mode,
        "block_count": int(args.block_count),
        "topk_blocks": int(args.topk_blocks),
        "utility_source": args.utility_source,
        "hierarchical": int(args.hierarchical),
        "compiler_arch_config": arch_config,
        "save_compiler_path": args.save_compiler_path,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "focus_start": focus_start,
        "focus_end": focus_end,
        "train_history": train_history,
        "summary_deltas": summary,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
