#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_curriculum_corpus import allocate_samples, list_shards, sample_chunk_indices, shard_chunk_count
from tools.build_curriculum_features import load_data_shard

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NUMBER_RE = re.compile(r"^\d+(?:[.,:/-]\d+)*%?$")
IDENT_RE = re.compile(r"^(?:[A-Za-z]*\d[A-Za-z0-9_-]*|[A-Fa-f0-9]{8,}|[A-Za-z0-9_-]{10,})$")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TOKEN_RE = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
    r"\d+(?:[.,:/-]\d+)*%?|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[^\w\s]",
    re.UNICODE,
)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)

PRONOUNS = frozenset(
    {
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        "this", "that", "these", "those", "someone", "anyone", "everyone", "nobody", "nothing",
        "who", "whom", "whose", "which", "what",
    }
)
DETERMINERS = frozenset({"a", "an", "the", "some", "any", "each", "every", "no", "another", "such"})
PREPOSITIONS = frozenset(
    {
        "in", "on", "at", "from", "to", "for", "with", "without", "by", "of", "into", "onto", "through",
        "across", "after", "before", "under", "over", "between", "about", "against", "within", "around", "during",
        "toward", "towards", "per", "via", "upon", "among", "amongst", "beside", "besides", "beneath",
    }
)
AUXILIARIES = frozenset(
    {
        "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
        "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    }
)
OP_NOT = frozenset(
    {
        "not", "n't", "no", "never", "neither", "nor", "without", "except", "unless", "cannot", "can't", "won't",
        "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    }
)
OP_AND = frozenset({"and", "also", "plus", "both", "furthermore", "moreover", "additionally"})
OP_OR = frozenset({"or", "either", "otherwise", "alternatively", "else"})
OP_IF = frozenset({"if", "unless", "when", "whenever", "whether", "while"})
OP_CAUSE = frozenset({"because", "since", "therefore", "thus", "hence", "so"})
PUNCT_MAP = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QMARK",
    "!": "EMARK",
    ":": "COLON",
    ";": "SEMI",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACK",
    "]": "RBRACK",
    "{": "LBRACE",
    "}": "RBRACE",
    "\"": "QUOTE",
    "'": "QUOTE",
    "-": "DASH",
    "/": "SLASH",
}
CONTENT_TAGS = frozenset({"ENTITY", "NUMBER", "URL", "EMAIL", "IDENT", "CONTENT"})
END_TAGS = frozenset({"PERIOD", "QMARK", "EMARK"})
BREAK_TAGS = frozenset({"COMMA", "COLON", "SEMI"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sentence-shape and operator-sequence motifs in FineWeb.")
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--train-sample-chunks", type=int, default=8192)
    parser.add_argument("--val-sample-chunks", type=int, default=8192)
    parser.add_argument("--sample-shards", type=int, default=0)
    parser.add_argument("--operator-window", type=int, default=4)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-sentence-chars", type=int, default=220)
    parser.add_argument("--substantive-min-tokens", type=int, default=8)
    return parser


def decode_chunk_text(sp: spm.SentencePieceProcessor, chunk_tokens: np.ndarray) -> str:
    return sp.decode_ids([int(x) for x in chunk_tokens.reshape(-1)])


def split_sentences(text: str) -> list[str]:
    sentences = [piece.strip() for piece in SENTENCE_RE.findall(text) if piece.strip()]
    return [piece for piece in sentences if WORD_RE.search(piece)]


def tokenize_sentence(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_entity_like(token: str) -> bool:
    if not token:
        return False
    if token.isupper() and len(token) > 1 and any(ch.isalpha() for ch in token):
        return True
    return token[:1].isupper() and token[1:].islower()


def classify_token(token: str) -> str:
    if not token:
        return "EMPTY"
    lower = token.casefold()
    if URL_RE.fullmatch(token):
        return "URL"
    if EMAIL_RE.fullmatch(token):
        return "EMAIL"
    if NUMBER_RE.fullmatch(token):
        return "NUMBER"
    if IDENT_RE.fullmatch(token):
        return "IDENT"
    if lower in OP_NOT:
        return "OP_NOT"
    if lower in OP_AND:
        return "OP_AND"
    if lower in OP_OR:
        return "OP_OR"
    if lower in OP_IF:
        return "OP_IF"
    if lower in OP_CAUSE:
        return "OP_CAUSE"
    if lower in PRONOUNS:
        return "PRON"
    if lower in DETERMINERS:
        return "DET"
    if lower in PREPOSITIONS:
        return "PREP"
    if lower in AUXILIARIES:
        return "AUX"
    if token in PUNCT_MAP:
        return PUNCT_MAP[token]
    if is_entity_like(token):
        return "ENTITY"
    if WORD_RE.fullmatch(token):
        return "CONTENT"
    if re.fullmatch(r"[^\w\s]", token, re.UNICODE):
        return "PUNCT"
    return "OTHER"


def compress_tags(tags: list[str], *, exact_counts: bool = True) -> list[str]:
    if not tags:
        return []
    out: list[str] = []
    current = tags[0]
    count = 1
    for tag in tags[1:]:
        if tag == current:
            count += 1
            continue
        if count > 1:
            out.append(f"{current}x{count}" if exact_counts else f"{current}+")
        else:
            out.append(current)
        current = tag
        count = 1
    if count > 1:
        out.append(f"{current}x{count}" if exact_counts else f"{current}+")
    else:
        out.append(current)
    return out


def operator_signature(tags: list[str]) -> str:
    ops = [tag for tag in tags if tag.startswith("OP_")]
    if not ops:
        return "none"
    compact: list[str] = []
    for op in ops:
        if not compact or compact[-1] != op:
            compact.append(op)
    return " > ".join(compact)


def classify_sentence(sentence: str) -> tuple[list[str], str]:
    tags = [classify_token(token) for token in tokenize_sentence(sentence)]
    tags = [tag for tag in tags if tag != "EMPTY"]
    return tags, " ".join(compress_tags(tags, exact_counts=True))


def operator_windows(tags: list[str], width: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for idx, tag in enumerate(tags):
        if not tag.startswith("OP_"):
            continue
        start = max(0, idx - width)
        end = min(len(tags), idx + width + 1)
        window = tags[start:end]
        center = idx - start
        rendered = list(window)
        rendered[center] = f"[{rendered[center]}]"
        rows.append((tag, " ".join(rendered)))
    return rows


def category_ngrams(tags: list[str], n: int) -> list[str]:
    if n <= 0 or len(tags) < n:
        return []
    return [" ".join(tags[i : i + n]) for i in range(len(tags) - n + 1)]


def shape_family(tags: list[str]) -> str:
    family = []
    for tag in tags:
        if tag.startswith("OP_"):
            family.append(tag)
        elif tag in {"PRON", "DET", "PREP", "AUX"}:
            family.append(tag)
        elif tag == "COMMA":
            family.append("COMMA")
        elif tag in END_TAGS:
            family.append("END")
        elif tag in CONTENT_TAGS or tag in {"ENTITY", "CONTENT", "OTHER", "PUNCT", "COLON", "SEMI", "LPAREN", "RPAREN", "QUOTE", "DASH", "SLASH"}:
            family.append("ARG")
        else:
            family.append("ARG")
    return " ".join(compress_tags(family, exact_counts=False))


def shape_spine(tags: list[str]) -> str:
    spine = []
    for tag in tags:
        if tag.startswith("OP_"):
            spine.append(tag)
        elif tag in {"PRON", "DET", "PREP", "AUX"}:
            spine.append(tag)
        elif tag in BREAK_TAGS:
            spine.append("BREAK")
        elif tag in END_TAGS:
            spine.append("END")
    if not spine:
        return "LEXICAL_ONLY"
    return " ".join(compress_tags(spine, exact_counts=False))


def preview_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def process_split_shard(
    task: tuple[str, int, str, int, int, int, str, int, int, int]
) -> tuple[list[dict[str, object]], Counter[str], Counter[str], Counter[str]]:
    split_name, shard_idx, shard_path_str, sample_count, chunk_size, seed, tokenizer_path_str, operator_window, ngram_size, max_chars = task
    if sample_count <= 0:
        return [], Counter(), Counter(), Counter()
    path = Path(shard_path_str)
    tokens = load_data_shard(path)
    num_chunks = max((tokens.size - 1) // chunk_size, 0)
    if num_chunks <= 0:
        return [], Counter(), Counter(), Counter()

    selected = sample_chunk_indices(num_chunks, sample_count, seed)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path_str)
    rows: list[dict[str, object]] = []
    operator_counter: Counter[str] = Counter()
    ngram_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()

    for chunk_idx in selected.tolist():
        start = chunk_idx * chunk_size
        chunk_tokens = tokens[start : start + chunk_size].astype(np.int32, copy=False)
        text = decode_chunk_text(sp, chunk_tokens)
        for sentence_idx, sentence in enumerate(split_sentences(text)):
            tags, shape = classify_sentence(sentence)
            if not tags:
                continue
            family = shape_family(tags)
            rows.append(
                {
                    "split": split_name,
                    "shard_id": shard_idx,
                    "shard_name": path.name,
                    "chunk_index": int(chunk_idx),
                    "sentence_index": int(sentence_idx),
                    "sentence_length_tokens": int(len(tags)),
                    "shape_length": int(len(shape.split())),
                    "operator_signature": operator_signature(tags),
                    "shape": shape,
                    "shape_family": family,
                    "shape_spine": shape_spine(tags),
                    "sentence_preview": preview_text(sentence, max_chars),
                }
            )
            family_counter[family] += 1
            for operator, window_shape in operator_windows(tags, operator_window):
                operator_counter[f"{operator}\t{window_shape}"] += 1
            for ngram in category_ngrams(tags, ngram_size):
                ngram_counter[ngram] += 1
    return rows, operator_counter, ngram_counter, family_counter


def summarize_shapes(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(key, as_index=False)
        .agg(
            count=(key, "size"),
            mean_sentence_length=("sentence_length_tokens", "mean"),
            operator_signature=("operator_signature", lambda s: s.mode().iloc[0] if not s.mode().empty else "none"),
            example=("sentence_preview", "first"),
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = max(int(grouped["count"].sum()), 1)
    grouped["fraction"] = grouped["count"] / total
    grouped["rank"] = np.arange(1, len(grouped) + 1, dtype=np.int32)
    return grouped


def summarize_families(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    grouped = (
        frame.groupby("shape_family", as_index=False)
        .agg(count=("shape_family", "size"), example=("sentence_preview", "first"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = max(int(grouped["count"].sum()), 1)
    grouped["fraction"] = grouped["count"] / total
    grouped["rank"] = np.arange(1, len(grouped) + 1, dtype=np.int32)
    return grouped.head(top_k).copy()


def summarize_operator_windows(counter: Counter[str], top_k: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = max(sum(counter.values()), 1)
    for rank, (key, count) in enumerate(counter.most_common(top_k), start=1):
        operator, window_shape = key.split("\t", 1)
        rows.append(
            {
                "rank": rank,
                "operator": operator,
                "window_shape": window_shape,
                "count": int(count),
                "fraction": float(count / total),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_ngrams(counter: Counter[str], top_k: int) -> pd.DataFrame:
    total = max(sum(counter.values()), 1)
    rows = [
        {"rank": rank, "ngram": ngram, "count": int(count), "fraction": float(count / total)}
        for rank, (ngram, count) in enumerate(counter.most_common(top_k), start=1)
    ]
    return pd.DataFrame.from_records(rows)


def coverage(counts: pd.Series, k: int) -> float:
    if counts.empty:
        return 0.0
    total = max(float(counts.sum()), 1.0)
    return float(counts.head(k).sum() / total)


def overlap_table(train_df: pd.DataFrame, val_df: pd.DataFrame, key: str, top_k: int) -> pd.DataFrame:
    merged = train_df[[key, "count"]].rename(columns={"count": "train_count"}).merge(
        val_df[[key, "count"]].rename(columns={"count": "val_count"}),
        on=key,
        how="outer",
    )
    merged["train_count"] = merged["train_count"].fillna(0).astype(np.int64)
    merged["val_count"] = merged["val_count"].fillna(0).astype(np.int64)
    merged["combined"] = merged["train_count"] + merged["val_count"]
    merged["delta"] = merged["val_count"] - merged["train_count"]
    merged = merged.sort_values(["combined", "val_count"], ascending=False).reset_index(drop=True)
    return merged.head(top_k).copy()


def shape_metrics(
    frame: pd.DataFrame,
    shape_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    spine_summary: pd.DataFrame,
    *,
    substantive_min_tokens: int,
) -> dict[str, float | int]:
    counts = shape_summary["count"] if not shape_summary.empty else pd.Series([], dtype=np.int64)
    family_counts = family_summary["count"] if not family_summary.empty else pd.Series([], dtype=np.int64)
    spine_counts = spine_summary["count"] if not spine_summary.empty else pd.Series([], dtype=np.int64)
    operator_fraction = float((frame["operator_signature"] != "none").mean()) if not frame.empty else 0.0
    substantive_fraction = float((frame["sentence_length_tokens"] >= substantive_min_tokens).mean()) if not frame.empty else 0.0
    return {
        "sentence_count": int(len(frame)),
        "unique_shapes": int(frame["shape"].nunique()) if not frame.empty else 0,
        "unique_shape_families": int(frame["shape_family"].nunique()) if not frame.empty else 0,
        "unique_shape_spines": int(frame["shape_spine"].nunique()) if not frame.empty else 0,
        "operator_sentence_fraction": operator_fraction,
        "substantive_sentence_fraction": substantive_fraction,
        "top10_shape_coverage": coverage(counts, 10),
        "top50_shape_coverage": coverage(counts, 50),
        "top100_shape_coverage": coverage(counts, 100),
        "top10_family_coverage": coverage(family_counts, 10),
        "top50_family_coverage": coverage(family_counts, 50),
        "top10_spine_coverage": coverage(spine_counts, 10),
        "top50_spine_coverage": coverage(spine_counts, 50),
        "avg_sentence_length_tokens": float(frame["sentence_length_tokens"].mean()) if not frame.empty else 0.0,
    }


def render_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_html_report(
    out_path: Path,
    *,
    train_metrics: dict[str, float | int],
    val_metrics: dict[str, float | int],
    train_shapes: pd.DataFrame,
    val_shapes: pd.DataFrame,
    train_families: pd.DataFrame,
    val_families: pd.DataFrame,
    train_spines: pd.DataFrame,
    val_spines: pd.DataFrame,
    train_substantive_spines: pd.DataFrame,
    val_substantive_spines: pd.DataFrame,
    train_operator_spines: pd.DataFrame,
    val_operator_spines: pd.DataFrame,
    train_windows: pd.DataFrame,
    val_windows: pd.DataFrame,
    train_ngrams: pd.DataFrame,
    val_ngrams: pd.DataFrame,
    shape_overlap: pd.DataFrame,
    spine_overlap: pd.DataFrame,
) -> None:
    css = """
    body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4efe7;color:#1f2937;margin:0;padding:32px}
    .card{background:#fffdf8;border:1px solid #ded7cb;border-radius:18px;padding:24px;margin:0 0 20px 0;box-shadow:0 4px 18px rgba(0,0,0,0.04)}
    h1,h2,h3{margin:0 0 12px 0;color:#111827}
    p{line-height:1.5}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border-bottom:1px solid #ece5d9;padding:8px 10px;text-align:left;vertical-align:top}
    th{background:#f7f2ea}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
    .metric{font-size:14px;color:#475569;margin:4px 0}
    code{background:#f1ece2;padding:1px 4px;border-radius:4px}
    """
    def metric_block(title: str, metrics: dict[str, float | int]) -> str:
        lines = [f"<div class='card'><h3>{html.escape(title)}</h3>"]
        for key, value in metrics.items():
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"<div class='metric'><strong>{html.escape(key)}</strong>: {html.escape(rendered)}</div>")
        lines.append("</div>")
        return "".join(lines)

    content = f"""
    <html><head><meta charset="utf-8"/><title>Sequence Shape Deep Dive</title><style>{css}</style></head><body>
    <div class="card">
      <h1>Sequence Shape Deep Dive</h1>
      <p>This review studies sentence-level structural templates, operator-local windows, and symbolic token-class ngrams rather than raw lexical strings.</p>
    </div>
    <div class="grid">
      {metric_block("Train Sample", train_metrics)}
      {metric_block("Validation Sample", val_metrics)}
    </div>
    <div class="grid">
      <div class="card"><h2>Top Train Shapes</h2>{render_table(train_shapes, ["rank", "shape", "count", "fraction", "operator_signature", "example"])}</div>
      <div class="card"><h2>Top Val Shapes</h2>{render_table(val_shapes, ["rank", "shape", "count", "fraction", "operator_signature", "example"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Top Train Shape Families</h2>{render_table(train_families, ["rank", "shape_family", "count", "fraction", "example"])}</div>
      <div class="card"><h2>Top Val Shape Families</h2>{render_table(val_families, ["rank", "shape_family", "count", "fraction", "example"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Top Train Structural Spines</h2>{render_table(train_spines, ["rank", "shape_spine", "count", "fraction", "example"])}</div>
      <div class="card"><h2>Top Val Structural Spines</h2>{render_table(val_spines, ["rank", "shape_spine", "count", "fraction", "example"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Substantive Sentence Spines (Train)</h2>{render_table(train_substantive_spines, ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])}</div>
      <div class="card"><h2>Substantive Sentence Spines (Val)</h2>{render_table(val_substantive_spines, ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Operator Sentence Spines (Train)</h2>{render_table(train_operator_spines, ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])}</div>
      <div class="card"><h2>Operator Sentence Spines (Val)</h2>{render_table(val_operator_spines, ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Top Train Operator Windows</h2>{render_table(train_windows, ["rank", "operator", "window_shape", "count", "fraction"])}</div>
      <div class="card"><h2>Top Val Operator Windows</h2>{render_table(val_windows, ["rank", "operator", "window_shape", "count", "fraction"])}</div>
    </div>
    <div class="grid">
      <div class="card"><h2>Top Train Category Ngrams</h2>{render_table(train_ngrams, ["rank", "ngram", "count", "fraction"])}</div>
      <div class="card"><h2>Top Val Category Ngrams</h2>{render_table(val_ngrams, ["rank", "ngram", "count", "fraction"])}</div>
    </div>
    <div class="card"><h2>Train vs Val Overlap</h2>{render_table(shape_overlap, ["shape", "train_count", "val_count", "delta"])}</div>
    <div class="card"><h2>Structural Spine Overlap</h2>{render_table(spine_overlap, ["shape_spine", "train_count", "val_count", "delta"])}</div>
    </body></html>
    """
    out_path.write_text(content, encoding="utf-8")


def write_markdown_report(
    out_path: Path,
    *,
    train_metrics: dict[str, float | int],
    val_metrics: dict[str, float | int],
    train_shapes: pd.DataFrame,
    val_shapes: pd.DataFrame,
    train_families: pd.DataFrame,
    val_families: pd.DataFrame,
    train_spines: pd.DataFrame,
    val_spines: pd.DataFrame,
    train_substantive_spines: pd.DataFrame,
    val_substantive_spines: pd.DataFrame,
    train_operator_spines: pd.DataFrame,
    val_operator_spines: pd.DataFrame,
    train_windows: pd.DataFrame,
    val_windows: pd.DataFrame,
    train_ngrams: pd.DataFrame,
    val_ngrams: pd.DataFrame,
    shape_overlap: pd.DataFrame,
    spine_overlap: pd.DataFrame,
) -> None:
    lines = [
        "# Sequence Shape Deep Dive",
        "",
        "This pass studies symbolic sentence templates, operator-local windows, and token-class sequences rather than literal strings.",
        "",
        "## Sample Metrics",
        "",
        "### Train",
    ]
    for key, value in train_metrics.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- {key}: {rendered}")
    lines += ["", "### Validation"]
    for key, value in val_metrics.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- {key}: {rendered}")

    def append_frame(title: str, df: pd.DataFrame, columns: list[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        if df.empty:
            lines.append("_No rows._")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in df.iterrows():
            vals = [str(row[col]) for col in columns]
            lines.append("| " + " | ".join(vals) + " |")

    append_frame("Top Train Shapes", train_shapes.head(20), ["rank", "shape", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Validation Shapes", val_shapes.head(20), ["rank", "shape", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Train Shape Families", train_families.head(20), ["rank", "shape_family", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Validation Shape Families", val_families.head(20), ["rank", "shape_family", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Train Structural Spines", train_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Validation Structural Spines", val_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Substantive Sentence Spines (Train)", train_substantive_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Substantive Sentence Spines (Validation)", val_substantive_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Operator Sentence Spines (Train)", train_operator_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Operator Sentence Spines (Validation)", val_operator_spines.head(20), ["rank", "shape_spine", "count", "fraction", "operator_signature", "example"])
    append_frame("Top Train Operator Windows", train_windows.head(20), ["rank", "operator", "window_shape", "count", "fraction"])
    append_frame("Top Validation Operator Windows", val_windows.head(20), ["rank", "operator", "window_shape", "count", "fraction"])
    append_frame("Top Train Category Ngrams", train_ngrams.head(20), ["rank", "ngram", "count", "fraction"])
    append_frame("Top Validation Category Ngrams", val_ngrams.head(20), ["rank", "ngram", "count", "fraction"])
    append_frame("Train vs Validation Shape Overlap", shape_overlap.head(30), ["shape", "train_count", "val_count", "delta"])
    append_frame("Train vs Validation Spine Overlap", spine_overlap.head(30), ["shape_spine", "train_count", "val_count", "delta"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split(
    *,
    split_name: str,
    glob_pattern: str,
    sample_chunks: int,
    tokenizer_path: Path,
    chunk_size: int,
    sample_shards: int,
    operator_window: int,
    ngram_size: int,
    num_workers: int,
    seed: int,
    max_chars: int,
) -> tuple[pd.DataFrame, Counter[str], Counter[str], Counter[str]]:
    shard_paths = list_shards(glob_pattern, sample_shards)
    chunk_counts = [shard_chunk_count(path, chunk_size) for path in shard_paths]
    alloc = allocate_samples(chunk_counts, sample_chunks)
    tasks = [
        (
            split_name,
            shard_idx,
            path.as_posix(),
            count,
            chunk_size,
            seed + shard_idx * 1009,
            tokenizer_path.as_posix(),
            operator_window,
            ngram_size,
            max_chars,
        )
        for shard_idx, (path, count) in enumerate(zip(shard_paths, alloc))
        if count > 0
    ]
    rows: list[dict[str, object]] = []
    operator_counter: Counter[str] = Counter()
    ngram_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    if num_workers <= 1:
        results = [process_split_shard(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(process_split_shard, tasks))
    for split_rows, split_ops, split_ngrams, split_families in results:
        rows.extend(split_rows)
        operator_counter.update(split_ops)
        ngram_counter.update(split_ngrams)
        family_counter.update(split_families)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise SystemExit(f"No sentence rows produced for split {split_name}")
    return frame, operator_counter, ngram_counter, family_counter


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer_path)

    train_frame, train_ops, train_ngrams, _train_families_counter = run_split(
        split_name="train",
        glob_pattern=args.train_glob,
        sample_chunks=args.train_sample_chunks,
        tokenizer_path=tokenizer_path,
        chunk_size=args.chunk_size,
        sample_shards=args.sample_shards,
        operator_window=args.operator_window,
        ngram_size=args.ngram_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_chars=args.max_sentence_chars,
    )
    val_frame, val_ops, val_ngrams, _val_families_counter = run_split(
        split_name="val",
        glob_pattern=args.val_glob,
        sample_chunks=args.val_sample_chunks,
        tokenizer_path=tokenizer_path,
        chunk_size=args.chunk_size,
        sample_shards=args.sample_shards,
        operator_window=args.operator_window,
        ngram_size=args.ngram_size,
        num_workers=args.num_workers,
        seed=args.seed + 9973,
        max_chars=args.max_sentence_chars,
    )

    train_shape_full = summarize_shapes(train_frame, "shape")
    val_shape_full = summarize_shapes(val_frame, "shape")
    train_shapes = train_shape_full.head(args.top_k).copy()
    val_shapes = val_shape_full.head(args.top_k).copy()
    train_family_full = summarize_shapes(train_frame, "shape_family")
    val_family_full = summarize_shapes(val_frame, "shape_family")
    train_families = train_family_full.head(args.top_k).copy()
    val_families = val_family_full.head(args.top_k).copy()
    train_spine_full = summarize_shapes(train_frame, "shape_spine")
    val_spine_full = summarize_shapes(val_frame, "shape_spine")
    train_spines = train_spine_full.head(args.top_k).copy()
    val_spines = val_spine_full.head(args.top_k).copy()
    train_substantive_spines = summarize_shapes(
        train_frame[train_frame["sentence_length_tokens"] >= args.substantive_min_tokens], "shape_spine"
    ).head(args.top_k).copy()
    val_substantive_spines = summarize_shapes(
        val_frame[val_frame["sentence_length_tokens"] >= args.substantive_min_tokens], "shape_spine"
    ).head(args.top_k).copy()
    train_operator_spines = summarize_shapes(train_frame[train_frame["operator_signature"] != "none"], "shape_spine").head(args.top_k).copy()
    val_operator_spines = summarize_shapes(val_frame[val_frame["operator_signature"] != "none"], "shape_spine").head(args.top_k).copy()
    train_windows = summarize_operator_windows(train_ops, args.top_k)
    val_windows = summarize_operator_windows(val_ops, args.top_k)
    train_ngram_df = summarize_ngrams(train_ngrams, args.top_k)
    val_ngram_df = summarize_ngrams(val_ngrams, args.top_k)
    shape_overlap = overlap_table(train_shape_full, val_shape_full, "shape", args.top_k)
    family_overlap = overlap_table(
        train_family_full.rename(columns={"shape_family": "shape_family_key"}),
        val_family_full.rename(columns={"shape_family": "shape_family_key"}),
        "shape_family_key",
        args.top_k,
    )
    spine_overlap = overlap_table(
        train_spine_full.rename(columns={"shape_spine": "shape_spine_key"}),
        val_spine_full.rename(columns={"shape_spine": "shape_spine_key"}),
        "shape_spine_key",
        args.top_k,
    )
    train_metrics = shape_metrics(
        train_frame,
        train_shape_full,
        train_family_full,
        train_spine_full,
        substantive_min_tokens=args.substantive_min_tokens,
    )
    val_metrics = shape_metrics(
        val_frame,
        val_shape_full,
        val_family_full,
        val_spine_full,
        substantive_min_tokens=args.substantive_min_tokens,
    )

    summary = {
        "config": {
            "chunk_size": args.chunk_size,
            "train_sample_chunks": args.train_sample_chunks,
            "val_sample_chunks": args.val_sample_chunks,
            "sample_shards": args.sample_shards,
            "operator_window": args.operator_window,
            "ngram_size": args.ngram_size,
            "seed": args.seed,
            "top_k": args.top_k,
            "substantive_min_tokens": args.substantive_min_tokens,
        },
        "train": train_metrics,
        "val": val_metrics,
        "overlap": {
            "top_shape_intersection": int(len(set(train_shapes["shape"]) & set(val_shapes["shape"]))),
            "top_family_intersection": int(len(set(train_families["shape_family"]) & set(val_families["shape_family"]))),
            "top_spine_intersection": int(len(set(train_spines["shape_spine"]) & set(val_spines["shape_spine"]))),
            "val_shape_seen_in_train_fraction": float(val_frame["shape"].isin(set(train_frame["shape"])).mean()),
            "val_shape_family_seen_in_train_fraction": float(val_frame["shape_family"].isin(set(train_frame["shape_family"])).mean()),
            "val_shape_spine_seen_in_train_fraction": float(val_frame["shape_spine"].isin(set(train_frame["shape_spine"])).mean()),
            "top_operator_window_intersection": int(
                len(set(train_windows["window_shape"]) & set(val_windows["window_shape"]))
            ) if not train_windows.empty and not val_windows.empty else 0,
        },
    }

    train_frame.to_csv(out_dir / "train_sentence_rows.csv", index=False)
    val_frame.to_csv(out_dir / "val_sentence_rows.csv", index=False)
    train_shapes.to_csv(out_dir / "train_sentence_shapes.csv", index=False)
    val_shapes.to_csv(out_dir / "val_sentence_shapes.csv", index=False)
    train_families.to_csv(out_dir / "train_shape_families.csv", index=False)
    val_families.to_csv(out_dir / "val_shape_families.csv", index=False)
    train_spines.to_csv(out_dir / "train_shape_spines.csv", index=False)
    val_spines.to_csv(out_dir / "val_shape_spines.csv", index=False)
    train_substantive_spines.to_csv(out_dir / "train_substantive_shape_spines.csv", index=False)
    val_substantive_spines.to_csv(out_dir / "val_substantive_shape_spines.csv", index=False)
    train_operator_spines.to_csv(out_dir / "train_operator_sentence_spines.csv", index=False)
    val_operator_spines.to_csv(out_dir / "val_operator_sentence_spines.csv", index=False)
    train_windows.to_csv(out_dir / "train_operator_windows.csv", index=False)
    val_windows.to_csv(out_dir / "val_operator_windows.csv", index=False)
    train_ngram_df.to_csv(out_dir / "train_category_ngrams.csv", index=False)
    val_ngram_df.to_csv(out_dir / "val_category_ngrams.csv", index=False)
    shape_overlap.to_csv(out_dir / "shape_overlap.csv", index=False)
    family_overlap.to_csv(out_dir / "shape_family_overlap.csv", index=False)
    spine_overlap.rename(columns={"shape_spine_key": "shape_spine"}).to_csv(out_dir / "shape_spine_overlap.csv", index=False)
    (out_dir / "sequence_shape_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_markdown_report(
        out_dir / "sequence_shape_deep_dive.md",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_shapes=train_shapes,
        val_shapes=val_shapes,
        train_families=train_families,
        val_families=val_families,
        train_spines=train_spines,
        val_spines=val_spines,
        train_substantive_spines=train_substantive_spines,
        val_substantive_spines=val_substantive_spines,
        train_operator_spines=train_operator_spines,
        val_operator_spines=val_operator_spines,
        train_windows=train_windows,
        val_windows=val_windows,
        train_ngrams=train_ngram_df,
        val_ngrams=val_ngram_df,
        shape_overlap=shape_overlap,
        spine_overlap=spine_overlap.rename(columns={"shape_spine_key": "shape_spine"}),
    )
    write_html_report(
        out_dir / "sequence_shape_deep_dive.html",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_shapes=train_shapes,
        val_shapes=val_shapes,
        train_families=train_families,
        val_families=val_families,
        train_spines=train_spines,
        val_spines=val_spines,
        train_substantive_spines=train_substantive_spines,
        val_substantive_spines=val_substantive_spines,
        train_operator_spines=train_operator_spines,
        val_operator_spines=val_operator_spines,
        train_windows=train_windows,
        val_windows=val_windows,
        train_ngrams=train_ngram_df,
        val_ngrams=val_ngram_df,
        shape_overlap=shape_overlap,
        spine_overlap=spine_overlap.rename(columns={"shape_spine_key": "shape_spine"}),
    )

    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
