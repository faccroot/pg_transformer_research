#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curriculum_runtime import load_data_shard
from text_prosody_features import (
    is_emoji_like_piece,
    is_markup_delimiter_piece,
    is_quote_like,
    is_url_like_piece,
    piece_to_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit tokenizer fragmentation tax on tokenized FineWeb shards.")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--shard-glob", default="", help="Explicit shard glob, e.g. data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    p.add_argument("--data-path", default="", help="Dataset dir containing fineweb_{train,val}_*.bin")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--max-shards", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=262144)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--result-json", required=True)
    return p.parse_args()


def shard_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.shard_glob:
        paths = sorted(Path().glob(args.shard_glob))
    elif args.data_path:
        paths = sorted(Path(args.data_path).expanduser().glob(f"fineweb_{args.split}_*.bin"))
    else:
        raise ValueError("either --shard-glob or --data-path is required")
    if not paths:
        raise FileNotFoundError("no shards found for requested dataset selection")
    return [Path(path) for path in paths[: max(int(args.max_shards), 1)]]


def take_token_sample(shard_paths: list[Path], max_tokens: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    remaining = max(int(max_tokens), 1)
    for path in shard_paths:
        if remaining <= 0:
            break
        shard = load_data_shard(path)
        take = min(int(shard.shape[0]), remaining)
        rows.append(np.asarray(shard[:take], dtype=np.int32))
        remaining -= take
    if not rows:
        return np.zeros((0,), dtype=np.int32)
    return rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)


def contains_whitespace(text: str) -> bool:
    return any(ch.isspace() for ch in str(text))


def whitespace_only(text: str) -> bool:
    core = str(text)
    return bool(core) and all(ch.isspace() for ch in core)


def construct_kind_for_whitespace(text: str) -> str:
    newline_count = int(str(text).count("\n"))
    if newline_count >= 2:
        return "paragraph_break"
    if newline_count == 1:
        trailing = str(text).split("\n")[-1]
        if trailing and all(ch in {" ", "\t"} for ch in trailing):
            return "newline_plus_indent"
        return "single_newline"
    return "space_or_tab_run"


def summarize_examples(records: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    ranked = sorted(
        records,
        key=lambda row: (
            int(row["token_count"]),
            int(row["char_count"]),
            len(str(row["text"])),
        ),
        reverse=True,
    )
    return ranked[: max(int(top_k), 0)]


def summarize_construct_records(records: list[dict[str, object]], *, sample_tokens: int, top_k: int) -> dict[str, object]:
    token_counts = np.asarray([int(row["token_count"]) for row in records], dtype=np.int32)
    char_counts = np.asarray([int(row["char_count"]) for row in records], dtype=np.int32)
    if token_counts.size <= 0:
        return {
            "count": 0,
            "token_share": 0.0,
            "mean_tokens": None,
            "median_tokens": None,
            "p90_tokens": None,
            "mean_chars": None,
            "multi_token_fraction": 0.0,
            "extra_token_tax": 0,
            "extra_token_tax_share": 0.0,
            "examples": [],
        }
    extra_token_tax = int(np.maximum(token_counts - 1, 0).sum())
    return {
        "count": int(token_counts.size),
        "token_share": float(token_counts.sum() / max(int(sample_tokens), 1)),
        "mean_tokens": float(token_counts.mean()),
        "median_tokens": float(np.median(token_counts)),
        "p90_tokens": float(np.quantile(token_counts, 0.90)),
        "mean_chars": float(char_counts.mean()),
        "multi_token_fraction": float((token_counts > 1).mean()),
        "extra_token_tax": extra_token_tax,
        "extra_token_tax_share": float(extra_token_tax / max(int(sample_tokens), 1)),
        "examples": summarize_examples(records, top_k),
    }


def scan_constructs(piece_texts: list[str], pieces: list[str]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {
        "paragraph_break": [],
        "single_newline": [],
        "newline_plus_indent": [],
        "space_or_tab_run": [],
        "url_span": [],
        "emoji_span": [],
        "quote_span": [],
        "markup_delim_span": [],
    }
    n = len(piece_texts)
    i = 0
    while i < n:
        raw = piece_texts[i]
        piece = pieces[i]
        if whitespace_only(raw):
            j = i + 1
            text = raw
            while j < n and whitespace_only(piece_texts[j]):
                text += piece_texts[j]
                j += 1
            kind = construct_kind_for_whitespace(text)
            out[kind].append(
                {
                    "start": int(i),
                    "end": int(j),
                    "token_count": int(j - i),
                    "char_count": len(text),
                    "text": text.replace("\n", "\\n").replace("\t", "\\t"),
                }
            )
            i = j
            continue
        if is_url_like_piece(piece):
            j = i + 1
            text = raw
            while (
                j < n
                and not contains_whitespace(piece_texts[j])
                and not is_emoji_like_piece(pieces[j])
                and not is_quote_like(pieces[j])
                and not is_markup_delimiter_piece(pieces[j])
            ):
                text += piece_texts[j]
                j += 1
            out["url_span"].append(
                {
                    "start": int(i),
                    "end": int(j),
                    "token_count": int(j - i),
                    "char_count": len(text),
                    "text": text[:160],
                }
            )
            i = j
            continue
        if is_emoji_like_piece(piece):
            j = i + 1
            text = raw
            while j < n and is_emoji_like_piece(pieces[j]) and not contains_whitespace(piece_texts[j]):
                text += piece_texts[j]
                j += 1
            out["emoji_span"].append(
                {
                    "start": int(i),
                    "end": int(j),
                    "token_count": int(j - i),
                    "char_count": len(text),
                    "text": text,
                }
            )
            i = j
            continue
        if is_quote_like(piece):
            j = i + 1
            text = raw
            while j < n and is_quote_like(pieces[j]) and not contains_whitespace(piece_texts[j]):
                text += piece_texts[j]
                j += 1
            out["quote_span"].append(
                {
                    "start": int(i),
                    "end": int(j),
                    "token_count": int(j - i),
                    "char_count": len(text),
                    "text": text,
                }
            )
            i = j
            continue
        if is_markup_delimiter_piece(piece):
            j = i + 1
            text = raw
            while j < n and is_markup_delimiter_piece(pieces[j]) and not contains_whitespace(piece_texts[j]):
                text += piece_texts[j]
                j += 1
            out["markup_delim_span"].append(
                {
                    "start": int(i),
                    "end": int(j),
                    "token_count": int(j - i),
                    "char_count": len(text),
                    "text": text,
                }
            )
            i = j
            continue
        i += 1
    return out


def summarize_piece_level_flags(pieces: list[str], piece_texts: list[str]) -> dict[str, object]:
    flags = {
        "quote_like": np.asarray([int(is_quote_like(piece)) for piece in pieces], dtype=np.int32),
        "markup_delim": np.asarray([int(is_markup_delimiter_piece(piece)) for piece in pieces], dtype=np.int32),
        "url_like": np.asarray([int(is_url_like_piece(piece)) for piece in pieces], dtype=np.int32),
        "emoji_like": np.asarray([int(is_emoji_like_piece(piece)) for piece in pieces], dtype=np.int32),
        "whitespace_only": np.asarray([int(whitespace_only(text)) for text in piece_texts], dtype=np.int32),
    }
    out: dict[str, object] = {}
    total = max(len(pieces), 1)
    for name, arr in flags.items():
        idx = np.flatnonzero(arr > 0)
        out[name] = {
            "count": int(idx.size),
            "fraction": float(idx.size / total),
            "examples": [pieces[int(i)] for i in idx[:8].tolist()],
        }
    return out


def main() -> None:
    args = parse_args()
    shard_paths = shard_paths_from_args(args)
    tokens = take_token_sample(shard_paths, args.max_tokens)
    if tokens.size <= 0:
        raise ValueError("no tokens loaded for audit")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path).expanduser()))
    pieces = [sp.id_to_piece(int(tok)) for tok in tokens.tolist()]
    piece_texts = [piece_to_text(piece) for piece in pieces]
    construct_records = scan_constructs(piece_texts, pieces)
    construct_summary = {
        name: summarize_construct_records(records, sample_tokens=int(tokens.size), top_k=args.top_k)
        for name, records in construct_records.items()
    }
    total_tax = int(sum(int(summary["extra_token_tax"]) for summary in construct_summary.values()))
    result = {
        "sample": {
            "shards": [str(path) for path in shard_paths],
            "num_tokens": int(tokens.size),
            "tokenizer_path": str(Path(args.tokenizer_path).expanduser()),
        },
        "constructs": construct_summary,
        "piece_level_flags": summarize_piece_level_flags(pieces, piece_texts),
        "totals": {
            "construct_extra_token_tax": total_tax,
            "construct_extra_token_tax_share": float(total_tax / max(int(tokens.size), 1)),
        },
    }
    output_path = Path(args.result_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
