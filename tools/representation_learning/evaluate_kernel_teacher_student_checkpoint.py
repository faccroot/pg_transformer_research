#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import sentencepiece as spm  # noqa: E402

import train_gpt_mlx as base  # noqa: E402


def _load_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path).resolve()
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_checkpoint_path(summary: dict[str, Any], summary_path: Path) -> Path:
    candidates = [
        summary_path.parent / "best_kernel_teacher_student.npz",
        summary_path.parent / "kernel_teacher_student_final.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raw = str(summary.get("best_checkpoint_path") or summary.get("final_checkpoint_path") or "").strip()
    if raw:
        path = Path(raw)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve a kernel-student checkpoint from {summary_path}")


def _build_backbone(summary: dict[str, Any], tokenizer_path: Path, checkpoint_state: dict[str, mx.array]):
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tie_embeddings = bool(summary.get("tie_embeddings", "backbone.lm_head.weight" not in checkpoint_state))
    hp = base.Hyperparameters()
    hp.tokenizer_path = str(tokenizer_path)
    hp.vocab_size = int(sp.vocab_size())
    hp.train_seq_len = int(summary["max_seq_len"])
    hp.num_layers = int(summary["num_layers"])
    hp.num_layer_templates = int(summary["num_layer_templates"])
    hp.model_dim = int(summary["model_dim"])
    hp.num_heads = int(summary["num_heads"])
    hp.num_kv_heads = int(summary["num_kv_heads"])
    hp.mlp_mult = int(summary.get("mlp_mult", 2))
    hp.mlp_leaky_slope = float(summary.get("mlp_leaky_slope", 0.0))
    hp.tie_embeddings = tie_embeddings
    hp.logit_softcap = float(summary.get("logit_softcap", 30.0))
    hp.tied_embed_init_std = float(summary.get("tied_embed_init_std", 0.02))
    hp.qk_gain_init = float(summary.get("qk_gain_init", 1.0))
    hp.seed = int(summary.get("seed", 17))
    model = base.make_gpt(hp, sp)
    model.set_turbo_qat(False, 0.0)
    return model, sp


def _strip_backbone_state(checkpoint_state: dict[str, mx.array]) -> dict[str, mx.array]:
    stripped: dict[str, mx.array] = {}
    for name, value in checkpoint_state.items():
        if not str(name).startswith("backbone."):
            continue
        stripped[str(name)[len("backbone."):]] = value
    if not stripped:
        raise ValueError("Checkpoint does not contain backbone.* parameters")
    return stripped


def _iter_texts(path: str | Path, *, text_key: str, max_examples: int) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = str(payload.get(text_key, ""))
            if not text.strip():
                continue
            rows.append(text)
            if max_examples > 0 and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"No non-empty text rows found in {path}")
    return rows


def _batch_tokenized(
    token_lists: list[list[int]],
    *,
    start: int,
    batch_size: int,
    pad_id: int,
    max_seq_len: int,
) -> tuple[mx.array, mx.array, mx.array, int]:
    batch = token_lists[start: start + max(int(batch_size), 1)]
    if not batch:
        raise ValueError("Empty batch")
    seq_lens = [max(1, min(len(tokens) - 1, int(max_seq_len))) for tokens in batch]
    width = max(seq_lens)
    x_np = np.full((len(batch), width), int(pad_id), dtype=np.int32)
    y_np = np.full((len(batch), width), int(pad_id), dtype=np.int32)
    weights_np = np.zeros((len(batch), width), dtype=np.float32)
    token_count = 0
    for row_idx, (tokens, seq_len) in enumerate(zip(batch, seq_lens, strict=True)):
        arr = np.asarray(tokens[: seq_len + 1], dtype=np.int32)
        x_np[row_idx, :seq_len] = arr[:-1]
        y_np[row_idx, :seq_len] = arr[1:]
        weights_np[row_idx, :seq_len] = 1.0
        token_count += int(seq_len)
    return (
        mx.array(x_np, dtype=mx.int32),
        mx.array(y_np, dtype=mx.int32),
        mx.array(weights_np, dtype=mx.float32),
        token_count,
    )


def evaluate_kernel_teacher_student_checkpoint(
    *,
    summary_path: str | Path,
    eval_jsonl: str | Path,
    output_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    text_key: str = "text",
    batch_size: int = 4,
    max_examples: int = 0,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = _load_summary(summary_path)
    resolved_tokenizer = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else Path(str(summary["tokenizer_path"])).expanduser().resolve()
    checkpoint_path = _resolve_checkpoint_path(summary, summary_path)
    checkpoint_state = {name: value for name, value in mx.load(str(checkpoint_path)).items()}
    backbone, sp = _build_backbone(summary, resolved_tokenizer, checkpoint_state)
    backbone_state = _strip_backbone_state(checkpoint_state)
    if hasattr(base, "select_compatible_flat_state"):
        compatible_state, load_stats = base.select_compatible_flat_state(backbone, backbone_state)
    else:
        compatible_state = backbone_state
        load_stats = {
            "matched_tensors": len(compatible_state),
            "matched_params": int(sum(np.prod(value.shape) for value in compatible_state.values())),
            "coverage": 1.0,
            "missing_tensors": 0,
            "mismatched_tensors": 0,
            "unexpected_tensors": 0,
        }
    base.apply_flat_arrays(backbone, compatible_state)
    mx.eval(backbone.parameters())

    texts = _iter_texts(eval_jsonl, text_key=text_key, max_examples=max_examples)
    token_lists = [list(sp.encode(text, out_type=int)) for text in texts]
    token_lists = [tokens for tokens in token_lists if len(tokens) >= 2]
    if not token_lists:
        raise ValueError("All evaluation texts were too short after tokenization")
    effective_max_seq_len = int(max_seq_len) if max_seq_len is not None else int(summary["max_seq_len"])
    pad_id = int(sp.pad_id()) if int(sp.pad_id()) >= 0 else 0

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(token_lists), max(int(batch_size), 1)):
        x, y, token_weights, token_count = _batch_tokenized(
            token_lists,
            start=start,
            batch_size=batch_size,
            pad_id=pad_id,
            max_seq_len=effective_max_seq_len,
        )
        hidden, _captured, _aux = backbone.forward_hidden_with_aux(x)
        nll = backbone.token_nll_from_hidden(hidden, y, token_weights=token_weights)
        nll_sum = float(mx.sum(nll.astype(mx.float32)).item())
        total_nll += nll_sum
        total_tokens += int(token_count)
    mean_nll = float(total_nll / max(total_tokens, 1))
    bpb = float(mean_nll / np.log(2.0))
    result = {
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "text_key": str(text_key),
        "num_examples": int(len(token_lists)),
        "num_tokens": int(total_tokens),
        "mean_nll": mean_nll,
        "bpb": bpb,
        "load_stats": load_stats,
        "projection_mode": str(summary.get("projection_mode", "")),
        "readout_mode": str(summary.get("readout_mode", "")),
        "ce_weight": float(summary.get("ce_weight", 0.0)),
        "distill_weight": float(summary.get("distill_weight", 0.0)),
    }
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved kernel-teacher-student checkpoint on arbitrary JSONL text.")
    parser.add_argument("summary_path", nargs="?", default="", help="summary.json for a completed kernel-teacher-student run")
    parser.add_argument("--summary-path", dest="summary_path_opt", default="", help="summary.json for a completed kernel-teacher-student run")
    parser.add_argument("--eval-jsonl", required=True, help="JSONL file containing evaluation text")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    parser.add_argument("--tokenizer-path", default="", help="Optional tokenizer override")
    parser.add_argument("--text-key", default="text", help="JSON key containing text")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    args = parser.parse_args()

    resolved_summary_path = str(args.summary_path_opt or args.summary_path).strip()
    if not resolved_summary_path:
        raise SystemExit("summary_path is required")

    result = evaluate_kernel_teacher_student_checkpoint(
        summary_path=resolved_summary_path,
        eval_jsonl=args.eval_jsonl,
        output_path=args.output or None,
        tokenizer_path=args.tokenizer_path or None,
        text_key=args.text_key,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        max_seq_len=(args.max_seq_len if args.max_seq_len > 0 else None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
