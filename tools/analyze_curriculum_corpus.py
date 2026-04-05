#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import html
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curriculum import cosine_kmeans, hashed_token_histograms
from tools.build_curriculum_features import load_data_shard


DEFAULT_OPERATOR_TERMS = ("not", "and", "or", "if", "because")
DEFAULT_CODE_TERMS = ("def", "function", "class", "return", "import", "const", "let", "var", "public", "private")
DEFAULT_LIST_PATTERNS = (
    re.compile(r"(?m)^\s*[-*]\s+"),
    re.compile(r"(?m)^\s*\d+[.)]\s+"),
    re.compile(r"(?m)^\s*[A-Za-z][.)]\s+"),
)
DEFAULT_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
DEFAULT_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DEFAULT_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEFAULT_UPPER_RE = re.compile(r"[A-Z]")
DEFAULT_ALPHA_RE = re.compile(r"[A-Za-z]")
DEFAULT_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
DEFAULT_QUESTION_RE = re.compile(r"\?")
DEFAULT_CODE_BLOCK_RE = re.compile(r"```|<code>|</code>|^\s{4,}\S", re.IGNORECASE | re.MULTILINE)
DEFAULT_DIGIT_RE = re.compile(r"\d")
DEFAULT_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
DEFAULT_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    fraction: float
    auto_label: str
    mean_compressibility: float
    mean_operator_density: float
    mean_code_marker_count: float
    mean_baseline_loss: float
    mean_sidecar_delta: float
    representative_sample: str
    representative_shard: int
    representative_chunk: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a curriculum review pack from FineWeb chunk statistics.")
    p.add_argument("--data-glob", required=True, help="Glob for tokenized train shards")
    p.add_argument("--tokenizer-path", required=True, help="SentencePiece tokenizer model")
    p.add_argument("--output-dir", required=True, help="Directory for CSV/JSON/HTML/SVG outputs")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--sample-chunks", type=int, default=102400, help="Total chunks to sample across all shards")
    p.add_argument("--sample-shards", type=int, default=0, help="Optional cap on number of shards to consider")
    p.add_argument("--histogram-bins", type=int, default=256)
    p.add_argument("--num-clusters", type=int, default=256)
    p.add_argument("--kmeans-iterations", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-workers", type=int, default=8, help="Parallel worker count for per-shard decoding/statistics")
    p.add_argument("--max-cluster-examples", type=int, default=24, help="How many cluster examples to spotlight in markdown/html")
    return p


def list_shards(pattern: str, sample_shards: int) -> list[Path]:
    import glob

    files = [Path(path) for path in sorted(glob.glob(pattern))]
    if sample_shards > 0:
        files = files[:sample_shards]
    if not files:
        raise SystemExit(f"No shards matched {pattern!r}")
    return files


def shard_chunk_count(path: Path, chunk_size: int) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    return max((num_tokens - 1) // chunk_size, 0)


def allocate_samples(chunk_counts: list[int], total_samples: int) -> list[int]:
    total_chunks = int(sum(chunk_counts))
    if total_chunks <= 0:
        raise ValueError("No chunks available")
    target = min(int(total_samples), total_chunks)
    raw = np.asarray(chunk_counts, dtype=np.float64)
    alloc = np.floor(raw / raw.sum() * target).astype(np.int64)
    alloc = np.minimum(alloc, np.asarray(chunk_counts, dtype=np.int64))
    remaining = int(target - int(alloc.sum()))
    if remaining > 0:
        remainders = raw / raw.sum() * target - alloc
        order = np.argsort(-remainders)
        for idx in order:
            if remaining <= 0:
                break
            if alloc[idx] >= chunk_counts[idx]:
                continue
            alloc[idx] += 1
            remaining -= 1
    return [int(x) for x in alloc.tolist()]


def sample_chunk_indices(num_chunks: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0,), dtype=np.int32)
    if sample_count >= num_chunks:
        return np.arange(num_chunks, dtype=np.int32)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(num_chunks, size=sample_count, replace=False).astype(np.int32))
    return selected


def chunk_offsets(indices: np.ndarray, chunk_size: int) -> np.ndarray:
    return indices.astype(np.int64) * int(chunk_size)


def decode_chunk_text(sp: spm.SentencePieceProcessor, chunk_tokens: np.ndarray) -> str:
    return sp.decode_ids([int(x) for x in chunk_tokens.reshape(-1)])


def count_operator_terms(text: str) -> dict[str, int]:
    lower = text.casefold()
    counts: dict[str, int] = {}
    for term in DEFAULT_OPERATOR_TERMS:
        counts[term] = len(re.findall(rf"\b{re.escape(term)}\b", lower))
    return counts


def count_code_markers(text: str) -> int:
    lower = text.casefold()
    total = text.count("{") + text.count("}") + text.count(";")
    for term in DEFAULT_CODE_TERMS:
        total += len(re.findall(rf"\b{re.escape(term)}\b", lower))
    return total


def count_list_markers(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in DEFAULT_LIST_PATTERNS)


def sentence_lengths(text: str) -> list[int]:
    pieces = [part.strip() for part in DEFAULT_SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not pieces:
        return []
    out: list[int] = []
    for piece in pieces:
        words = DEFAULT_WORD_RE.findall(piece)
        if words:
            out.append(len(words))
    return out


def text_features(text: str) -> dict[str, float | int | bool]:
    raw_bytes = text.encode("utf-8", errors="ignore")
    raw_byte_length = len(raw_bytes)
    zlib_length = len(__import__("zlib").compress(raw_bytes))
    words = DEFAULT_WORD_RE.findall(text)
    num_words = len(words)
    sentence_lens = sentence_lengths(text)
    digit_count = len(DEFAULT_DIGIT_RE.findall(text))
    alpha_count = len(DEFAULT_ALPHA_RE.findall(text))
    upper_count = len(DEFAULT_UPPER_RE.findall(text))
    punct_count = len(DEFAULT_PUNCT_RE.findall(text))
    operator_counts = count_operator_terms(text)
    list_marker_count = count_list_markers(text)
    code_marker_count = count_code_markers(text)
    question_mark_count = len(DEFAULT_QUESTION_RE.findall(text))
    has_url = bool(DEFAULT_URL_RE.search(text))
    has_email = bool(DEFAULT_EMAIL_RE.search(text))
    has_code_blocks = bool(DEFAULT_CODE_BLOCK_RE.search(text) or code_marker_count >= 8)
    whitespace_split = [part for part in DEFAULT_WHITESPACE_RE.split(text.strip()) if part]
    avg_word_length = float(np.mean([len(word) for word in words])) if words else 0.0
    avg_sentence_length = float(np.mean(sentence_lens)) if sentence_lens else 0.0
    uppercase_fraction = float(upper_count / max(alpha_count, 1))
    numbers_heavy = bool(digit_count / max(len(text), 1) > 0.10)
    vocabulary_rarity = 0.0
    return {
        "raw_byte_length": raw_byte_length,
        "zlib_compressed_length": zlib_length,
        "compressibility": 1.0 - (zlib_length / max(raw_byte_length, 1)),
        "unique_word_count": len(set(word.casefold() for word in words)),
        "punctuation_density": punct_count / max(len(text), 1),
        "avg_sentence_length": avg_sentence_length,
        "list_marker_count": list_marker_count,
        "code_marker_count": code_marker_count,
        "question_mark_count": question_mark_count,
        "has_url": has_url,
        "has_email": has_email,
        "has_numbers_heavy": numbers_heavy,
        "has_code_blocks": has_code_blocks,
        "avg_word_length": avg_word_length,
        "uppercase_fraction": uppercase_fraction,
        "operator_not_count": operator_counts["not"],
        "operator_and_count": operator_counts["and"],
        "operator_or_count": operator_counts["or"],
        "operator_if_count": operator_counts["if"],
        "operator_because_count": operator_counts["because"],
        "operator_total_count": int(sum(operator_counts.values())),
        "word_count": num_words,
        "whitespace_token_count": len(whitespace_split),
        "digit_char_count": digit_count,
    }


def process_shard_sample(task: tuple[int, str, int, int, int, str, int]) -> tuple[list[dict[str, object]], np.ndarray]:
    shard_idx, shard_path_str, sample_count, chunk_size, seed, tokenizer_path_str, histogram_bins = task
    if sample_count <= 0:
        return [], np.zeros((0, histogram_bins), dtype=np.float32)
    path = Path(shard_path_str)
    tokens = load_data_shard(path)
    num_chunks = max((tokens.size - 1) // chunk_size, 0)
    if num_chunks <= 0:
        return [], np.zeros((0, histogram_bins), dtype=np.float32)
    chosen = sample_chunk_indices(num_chunks, sample_count, seed)
    starts = chunk_offsets(chosen, chunk_size)
    selected_chunks = np.stack([tokens[start : start + chunk_size] for start in starts], axis=0).astype(np.int32)
    hist = hashed_token_histograms(selected_chunks, chunk_size, histogram_bins)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path_str)
    rows: list[dict[str, object]] = []
    for local_idx, chunk_idx in enumerate(chosen.tolist()):
        chunk_tokens = selected_chunks[local_idx]
        text = decode_chunk_text(sp, chunk_tokens)
        tf = text_features(text)
        token_count = int(chunk_tokens.size)
        piece_counts = np.unique(chunk_tokens, return_counts=True)
        unique_token_count = int(piece_counts[0].size)
        token_freqs = piece_counts[1]
        vocabulary_rarity = float(np.mean(token_freqs <= 2)) if token_freqs.size else 0.0
        type_token_ratio = float(unique_token_count / max(token_count, 1))
        operator_density = float(tf["operator_total_count"]) / max(token_count, 1)
        rows.append(
            {
                "shard_id": shard_idx,
                "shard_name": path.name,
                "chunk_index": int(chunk_idx),
                "seed": seed,
                "text_preview": text[:240].replace("\n", " ").strip(),
                "token_count": token_count,
                "bytes_per_token": float(tf["raw_byte_length"]) / max(token_count, 1),
                "unique_token_count": unique_token_count,
                "type_token_ratio": type_token_ratio,
                "vocabulary_rarity": vocabulary_rarity,
                "operator_density": operator_density,
                **tf,
            }
        )
    return rows, hist


def build_sample_frame(
    *,
    shard_paths: list[Path],
    sample_alloc: list[int],
    chunk_size: int,
    tokenizer_path: Path,
    seed: int,
    histogram_bins: int,
    num_workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    histogram_parts: list[np.ndarray] = []
    tasks = [
        (shard_idx, path.as_posix(), sample_count, chunk_size, seed + shard_idx * 1009, tokenizer_path.as_posix(), histogram_bins)
        for shard_idx, (path, sample_count) in enumerate(zip(shard_paths, sample_alloc))
        if sample_count > 0
    ]
    if num_workers <= 1:
        results = [process_shard_sample(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(process_shard_sample, tasks))
    for shard_rows, hist in results:
        if shard_rows:
            rows.extend(shard_rows)
        if hist.size:
            histogram_parts.append(hist)
    if not rows:
        raise SystemExit("No sampled chunks were produced")
    frame = pd.DataFrame.from_records(rows)
    hist = np.concatenate(histogram_parts, axis=0).astype(np.float32)
    if hist.shape[0] != len(frame):
        raise RuntimeError(f"Histogram/sample mismatch: hist={hist.shape[0]} rows={len(frame)}")
    frame["sample_index"] = np.arange(len(frame), dtype=np.int32)
    frame["sample_seed"] = seed
    frame.attrs["histograms"] = hist
    return frame


def run_clustering(frame: pd.DataFrame, num_clusters: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    hist = np.asarray(frame.attrs["histograms"], dtype=np.float32)
    if num_clusters <= 0:
        return np.zeros((len(frame),), dtype=np.int32), np.zeros((0, hist.shape[1]), dtype=np.float32)
    return cosine_kmeans(hist, num_clusters=num_clusters, iterations=iterations, seed=seed)


def pca_projection(values: np.ndarray, out_dim: int = 2) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 0:
        return np.zeros((0, out_dim), dtype=np.float32)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    comps = vh[:out_dim].T
    return (x @ comps).astype(np.float32)


def auto_label_cluster(row: pd.Series) -> str:
    if row["mean_code_marker_count"] >= 12 or row["code_fraction"] >= 0.35:
        return "code-heavy docs"
    if row["url_fraction"] >= 0.30:
        return "url / web boilerplate"
    if row["email_fraction"] >= 0.10:
        return "contact / account text"
    if row["numbers_fraction"] >= 0.35:
        return "number-heavy records"
    if row["mean_list_marker_count"] >= 2.5:
        return "lists / outlines"
    if row["mean_operator_density"] >= 0.020:
        return "logic / formal prose"
    if row["mean_punctuation_density"] >= 0.16:
        return "punctuation-dense fragments"
    if row["mean_compressibility"] >= 0.60:
        return "template-heavy web prose"
    return "general web prose"


def cluster_summaries(frame: pd.DataFrame, centers: np.ndarray) -> list[ClusterSummary]:
    hist = np.asarray(frame.attrs["histograms"], dtype=np.float32)
    norm_hist = hist / np.clip(np.linalg.norm(hist, axis=1, keepdims=True), 1e-12, None)
    summaries: list[ClusterSummary] = []
    total = len(frame)
    grouped = frame.groupby("cluster_id", sort=True)
    for cluster_id, group in grouped:
        idx = group.index.to_numpy(dtype=np.int64)
        if centers.size:
            center = centers[int(cluster_id)]
            sims = norm_hist[idx] @ center.astype(np.float32)
            best_pos = int(idx[int(np.argmax(sims))])
        else:
            best_pos = int(idx[0])
        cluster_row = pd.Series(
            {
                "mean_compressibility": float(group["compressibility"].mean()),
                "mean_operator_density": float(group["operator_density"].mean()),
                "mean_code_marker_count": float(group["code_marker_count"].mean()),
                "code_fraction": float(group["has_code_blocks"].mean()),
                "url_fraction": float(group["has_url"].mean()),
                "email_fraction": float(group["has_email"].mean()),
                "numbers_fraction": float(group["has_numbers_heavy"].mean()),
                "mean_list_marker_count": float(group["list_marker_count"].mean()),
                "mean_punctuation_density": float(group["punctuation_density"].mean()),
            }
        )
        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                size=int(len(group)),
                fraction=float(len(group) / max(total, 1)),
                auto_label=auto_label_cluster(cluster_row),
                mean_compressibility=float(group["compressibility"].mean()),
                mean_operator_density=float(group["operator_density"].mean()),
                mean_code_marker_count=float(group["code_marker_count"].mean()),
                mean_baseline_loss=float(group["baseline_loss"].mean()) if "baseline_loss" in group else math.nan,
                mean_sidecar_delta=float(group["sidecar_delta"].mean()) if "sidecar_delta" in group else math.nan,
                representative_sample=str(frame.loc[best_pos, "text_preview"]),
                representative_shard=int(frame.loc[best_pos, "shard_id"]),
                representative_chunk=int(frame.loc[best_pos, "chunk_index"]),
            )
        )
    summaries.sort(key=lambda item: (-item.size, item.cluster_id))
    return summaries


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    with gzip.open(path, "wb") as handle:
        handle.write(csv_bytes)


def quantile(values: pd.Series, q: float) -> float:
    if values.empty:
        return 0.0
    return float(np.quantile(values.to_numpy(dtype=np.float64), q))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pca_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "compressibility",
        "bytes_per_token",
        "type_token_ratio",
        "vocabulary_rarity",
        "operator_density",
        "punctuation_density",
        "avg_sentence_length",
        "list_marker_count",
        "code_marker_count",
        "avg_word_length",
        "uppercase_fraction",
    ]
    matrix = frame[cols].to_numpy(dtype=np.float64)
    matrix = (matrix - matrix.mean(axis=0, keepdims=True)) / np.clip(matrix.std(axis=0, keepdims=True), 1e-6, None)
    xy = pca_projection(matrix, out_dim=2)
    out = frame.copy()
    out["pca_x"] = xy[:, 0]
    out["pca_y"] = xy[:, 1]
    return out


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        (
            "<style>"
            "text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#233043}"
            ".title{font-size:24px;font-weight:700;fill:#142033}"
            ".subtitle{font-size:12px;fill:#5b6675}"
            ".axis{font-size:12px;font-weight:600;fill:#334155}"
            ".tick{font-size:11px;fill:#64748b}"
            ".small{font-size:11px;fill:#64748b}"
            ".legend{font-size:12px;fill:#334155}"
            ".annot{font-size:11px;font-weight:700;fill:#142033}"
            ".grid{stroke:#d8e1ea;stroke-width:1}"
            ".grid-soft{stroke:#ebf0f5;stroke-width:1}"
            ".frame{stroke:#9aa9bc;fill:#fffdf8;stroke-width:1.5}"
            ".panel{fill:#fffdf8;stroke:#d9e1ea;stroke-width:1.2}"
            ".pill{fill:#f2efe7;stroke:#d8ccb2;stroke-width:1}"
            ".legend-swatch{stroke:#fffdf8;stroke-width:1.5}"
            "</style>"
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f4efe7"/>',
    ]


def palette(cluster_id: int) -> str:
    colors = [
        "#266dd3", "#ca6702", "#2a9d8f", "#d1495b", "#7c6aaf", "#a44a3f",
        "#c56f8f", "#5f6f81", "#8f9800", "#1b9aaa", "#0f766e", "#9a3412",
        "#6d28d9", "#be123c", "#15803d", "#334155",
    ]
    return colors[cluster_id % len(colors)]


LABEL_COLORS = {
    "general web prose": "#4472c4",
    "logic / formal prose": "#d97706",
    "contact / account text": "#c2410c",
    "url / web boilerplate": "#0f766e",
    "number-heavy records": "#b91c1c",
    "lists / outlines": "#7c3aed",
    "code-heavy docs": "#1d4ed8",
    "punctuation-dense fragments": "#64748b",
    "template-heavy web prose": "#ad7c3f",
}


def label_color(label: str) -> str:
    return LABEL_COLORS.get(str(label), "#475569")


def linear_map(values: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
    if hi <= lo:
        return np.full_like(values, (out_lo + out_hi) * 0.5, dtype=np.float64)
    scaled = (values - lo) / (hi - lo)
    return out_lo + scaled * (out_hi - out_lo)


def tick_values(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    return np.linspace(lo, hi, count).tolist()


def draw_legend(lines: list[str], *, entries: list[tuple[str, str]], x: float, y: float, row_h: float = 22.0) -> None:
    panel_h = 18 + row_h * len(entries)
    panel_w = 228
    lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{panel_w}" height="{panel_h:.1f}" rx="14" class="panel"/>')
    lines.append(f'<text x="{x + 14:.1f}" y="{y + 18:.1f}" class="legend">Corpus labels</text>')
    for idx, (label, color) in enumerate(entries):
        yy = y + 18 + row_h * (idx + 1)
        lines.append(f'<circle cx="{x + 18:.1f}" cy="{yy - 4:.1f}" r="5.5" fill="{color}" class="legend-swatch"/>')
        lines.append(f'<text x="{x + 32:.1f}" y="{yy:.1f}" class="tick">{html.escape(label)}</text>')


def short_label(label: str, limit: int = 24) -> str:
    value = str(label)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def interesting_cluster_ids(cluster_df: pd.DataFrame, limit: int = 12) -> list[int]:
    picked: list[int] = []
    for subset in (
        cluster_df.sort_values("size", ascending=False).head(8),
        cluster_df.sort_values("mean_operator_density", ascending=False).head(4),
        cluster_df.sort_values("mean_compressibility", ascending=False).head(3),
        cluster_df.sort_values("mean_compressibility", ascending=True).head(3),
    ):
        for value in subset["cluster_id"].tolist():
            ivalue = int(value)
            if ivalue not in picked:
                picked.append(ivalue)
            if len(picked) >= limit:
                return picked
    return picked[:limit]


def scatter_svg(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    subtitle: str,
    size_col: str = "size",
    color_col: str = "auto_label",
    annotate_ids: list[int] | None = None,
) -> None:
    width, height = 1120, 760
    margin_left, margin_right, margin_top, margin_bottom = 88, 260, 92, 76
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = frame[x_col].to_numpy(dtype=np.float64)
    ys = frame[y_col].to_numpy(dtype=np.float64)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    px = linear_map(xs, x0, x1, margin_left, margin_left + plot_w)
    py = linear_map(ys, y0, y1, margin_top + plot_h, margin_top)
    size_vals = frame[size_col].to_numpy(dtype=np.float64)
    radii = linear_map(np.sqrt(size_vals), float(np.sqrt(size_vals).min()), float(np.sqrt(size_vals).max()), 7.0, 22.0)
    lines = svg_header(width, height)
    lines.append(f'<rect x="28" y="24" width="{width - 56}" height="{height - 48}" rx="24" class="panel"/>')
    lines.append(f'<text x="{margin_left}" y="56" class="title">{html.escape(title)}</text>')
    lines.append(f'<text x="{margin_left}" y="78" class="subtitle">{html.escape(subtitle)}</text>')
    lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" rx="18" class="frame"/>')
    for i in range(5):
        gy = margin_top + plot_h * i / 4
        gx = margin_left + plot_w * i / 4
        lines.append(f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_w}" y2="{gy:.1f}" class="grid-soft"/>')
        lines.append(f'<line x1="{gx:.1f}" y1="{margin_top}" x2="{gx:.1f}" y2="{margin_top + plot_h}" class="grid-soft"/>')
    for tick in tick_values(x0, x1, 5):
        x = float(linear_map(np.asarray([tick]), x0, x1, margin_left, margin_left + plot_w)[0])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top + plot_h}" x2="{x:.1f}" y2="{margin_top + plot_h + 8}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" class="tick">{tick:.3f}</text>')
    for tick in tick_values(y0, y1, 5):
        y = float(linear_map(np.asarray([tick]), y0, y1, margin_top + plot_h, margin_top)[0])
        lines.append(f'<line x1="{margin_left - 8}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left - 14:.1f}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.3f}</text>')
    order = np.argsort(-radii)
    ordered = frame.iloc[order]
    for row, x, y, r in zip(ordered.itertuples(index=False), px[order], py[order], radii[order]):
        fill = label_color(getattr(row, color_col))
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" fill-opacity="0.68" stroke="#fffdf8" stroke-width="1.4"/>')
    if annotate_ids is None:
        annotate_ids = interesting_cluster_ids(frame)
    annotated = frame[frame["cluster_id"].isin(annotate_ids)].copy()
    for row in annotated.itertuples(index=False):
        x = float(linear_map(np.asarray([getattr(row, x_col)]), x0, x1, margin_left, margin_left + plot_w)[0])
        y = float(linear_map(np.asarray([getattr(row, y_col)]), y0, y1, margin_top + plot_h, margin_top)[0])
        label = f"C{int(row.cluster_id)}"
        label_w = 12 + 7 * len(label)
        lines.append(f'<rect x="{x + 10:.1f}" y="{y - 15:.1f}" width="{label_w}" height="18" rx="9" fill="#fff8e9" stroke="#d8ccb2" stroke-width="1"/>')
        lines.append(f'<text x="{x + 16:.1f}" y="{y - 2:.1f}" class="annot">{html.escape(label)}</text>')
    lines.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" class="axis">{html.escape(x_label)}</text>')
    lines.append(f'<text x="26" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 26 {margin_top + plot_h / 2:.1f})" text-anchor="middle" class="axis">{html.escape(y_label)}</text>')
    lines.append(
        f'<text x="{margin_left}" y="{height - 44}" class="small">Bubble size tracks cluster share. Labels call out the largest and most structurally distinct clusters.</text>'
    )
    lines.append(
        f'<text x="{margin_left}" y="{height - 24}" class="small">x range: {x0:.4f} .. {x1:.4f}    y range: {y0:.4f} .. {y1:.4f}</text>'
    )
    legend_entries = [(label, label_color(label)) for label in frame[color_col].dropna().astype(str).drop_duplicates().tolist()]
    draw_legend(lines, entries=legend_entries[:8], x=width - 236, y=margin_top + 10)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hist_svg(values: pd.Series, title: str, x_label: str, output_path: Path, bins: int = 40, subtitle: str = "") -> None:
    width, height = 1120, 520
    margin_left, margin_right, margin_top, margin_bottom = 88, 56, 92, 78
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    arr = values.to_numpy(dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    ymax = max(int(counts.max()), 1)
    lines = svg_header(width, height)
    lines.append(f'<rect x="28" y="24" width="{width - 56}" height="{height - 48}" rx="24" class="panel"/>')
    lines.append(f'<text x="{margin_left}" y="56" class="title">{html.escape(title)}</text>')
    if subtitle:
        lines.append(f'<text x="{margin_left}" y="78" class="subtitle">{html.escape(subtitle)}</text>')
    lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" rx="18" class="frame"/>')
    for i in range(5):
        gy = margin_top + plot_h * i / 4
        lines.append(f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_w}" y2="{gy:.1f}" class="grid-soft"/>')
        tick = int(round(ymax * (4 - i) / 4))
        lines.append(f'<text x="{margin_left - 14:.1f}" y="{gy + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    for i in range(len(counts)):
        x0 = margin_left + plot_w * i / len(counts)
        x1 = margin_left + plot_w * (i + 1) / len(counts)
        bar_h = plot_h * counts[i] / ymax
        y = margin_top + plot_h - bar_h
        fill = "#2855a6" if i < len(counts) * 0.65 else "#d97706"
        lines.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{max((x1 - x0) - 2, 1):.2f}" height="{bar_h:.2f}" fill="{fill}" fill-opacity="0.82" rx="3"/>')
    p50 = float(np.median(arr))
    p95 = float(np.quantile(arr, 0.95))
    for tick in tick_values(float(arr.min()), float(arr.max()), 6):
        x = float(linear_map(np.asarray([tick]), float(arr.min()), float(arr.max()), margin_left, margin_left + plot_w)[0])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top + plot_h}" x2="{x:.1f}" y2="{margin_top + plot_h + 8}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" class="tick">{tick:.3f}</text>')
    for value, label, stroke in ((p50, "p50", "#0f766e"), (p95, "p95", "#c2410c")):
        x = float(linear_map(np.asarray([value]), float(arr.min()), float(arr.max()), margin_left, margin_left + plot_w)[0])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" stroke="{stroke}" stroke-width="2.2" stroke-dasharray="7 5"/>')
        lines.append(f'<rect x="{x + 8:.1f}" y="{margin_top + 10:.1f}" width="44" height="20" rx="10" fill="#fff8e9" stroke="#d8ccb2" stroke-width="1"/>')
        lines.append(f'<text x="{x + 18:.1f}" y="{margin_top + 24:.1f}" class="annot">{label}</text>')
    lines.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" class="axis">{html.escape(x_label)}</text>')
    lines.append(
        f'<text x="{margin_left}" y="{height - 44}" class="small">Most mass sits near the median. The right tail is where curriculum filtering and replay policy become interesting.</text>'
    )
    lines.append(
        f'<text x="{margin_left}" y="{height - 24}" class="small">min: {arr.min():.4f}    p50: {p50:.4f}    p95: {p95:.4f}    max: {arr.max():.4f}    peak bin count: {ymax}</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bar_svg(cluster_df: pd.DataFrame, title: str, output_path: Path, top_n: int = 16) -> None:
    width, height = 1120, 780
    margin_left, margin_right, margin_top, margin_bottom = 320, 96, 92, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    top = cluster_df.sort_values("size", ascending=False).head(top_n).reset_index(drop=True)
    xmax = max(int(top["size"].max()), 1)
    lines = svg_header(width, height)
    lines.append(f'<rect x="28" y="24" width="{width - 56}" height="{height - 48}" rx="24" class="panel"/>')
    lines.append(f'<text x="{64}" y="56" class="title">{html.escape(title)}</text>')
    lines.append(
        '<text x="64" y="78" class="subtitle">The large clusters dominate the corpus. This is the practical menu the scheduler is drawing from.</text>'
    )
    lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" rx="18" class="frame"/>')
    for tick in tick_values(0.0, float(xmax), 5):
        x = float(linear_map(np.asarray([tick]), 0.0, float(xmax), margin_left, margin_left + plot_w)[0])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" class="grid-soft"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 24:.1f}" text-anchor="middle" class="tick">{int(round(tick))}</text>')
    for i, row in top.iterrows():
        row_h = plot_h / len(top)
        y = margin_top + i * row_h + 8
        bar_h = row_h - 12
        x1 = float(linear_map(np.asarray([float(row["size"])]), 0.0, float(xmax), margin_left, margin_left + plot_w)[0])
        fill = label_color(str(row["auto_label"]))
        lines.append(f'<rect x="{margin_left:.2f}" y="{y:.2f}" width="{x1 - margin_left:.2f}" height="{bar_h:.2f}" fill="{fill}" fill-opacity="0.86" rx="9"/>')
        lines.append(f'<text x="{margin_left - 18:.1f}" y="{y + 16:.1f}" text-anchor="end" class="annot">C{int(row["cluster_id"])}</text>')
        lines.append(f'<text x="{margin_left - 18:.1f}" y="{y + 32:.1f}" text-anchor="end" class="tick">{html.escape(short_label(str(row["auto_label"]), 22))}</text>')
        lines.append(f'<text x="{x1 + 10:.1f}" y="{y + 16:.1f}" class="annot">{int(row["size"]):,}</text>')
        lines.append(f'<text x="{x1 + 10:.1f}" y="{y + 32:.1f}" class="tick">{format_percent(float(row["fraction"]))}</text>')
    lines.append(f'<text x="{margin_left}" y="{height - 16}" class="small">Horizontal bars read much more clearly here than rotated labels because cluster names are semantic, not numeric.</text>')
    legend_entries = [(label, label_color(label)) for label in top["auto_label"].astype(str).drop_duplicates().tolist()]
    draw_legend(lines, entries=legend_entries[:8], x=62, y=112)
    lines.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{margin_top + plot_h + 48:.1f}" text-anchor="middle" class="axis">sampled chunks</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_mix_svg(label_df: pd.DataFrame, title: str, output_path: Path) -> None:
    width, height = 1120, 420
    margin_left, margin_right, margin_top, margin_bottom = 56, 56, 96, 64
    plot_w = width - margin_left - margin_right
    bar_y = 180
    bar_h = 54
    lines = svg_header(width, height)
    lines.append(f'<rect x="28" y="24" width="{width - 56}" height="{height - 48}" rx="24" class="panel"/>')
    lines.append(f'<text x="{margin_left}" y="56" class="title">{html.escape(title)}</text>')
    lines.append(
        '<text x="56" y="78" class="subtitle">At a glance: most of the corpus is ordinary web prose, with a meaningful but much smaller formal-logic slice.</text>'
    )
    lines.append(f'<rect x="{margin_left}" y="{bar_y}" width="{plot_w}" height="{bar_h}" rx="18" class="frame"/>')
    cursor = margin_left
    for row in label_df.itertuples(index=False):
        seg_w = plot_w * float(row.fraction)
        fill = label_color(str(row.auto_label))
        lines.append(f'<rect x="{cursor:.2f}" y="{bar_y:.2f}" width="{seg_w:.2f}" height="{bar_h:.2f}" fill="{fill}" fill-opacity="0.90" rx="10"/>')
        if seg_w > 96:
            lines.append(
                f'<text x="{cursor + seg_w / 2:.1f}" y="{bar_y + 22:.1f}" text-anchor="middle" class="annot">{html.escape(short_label(str(row.auto_label), 20))}</text>'
            )
            lines.append(
                f'<text x="{cursor + seg_w / 2:.1f}" y="{bar_y + 40:.1f}" text-anchor="middle" class="tick">{format_percent(float(row.fraction))}</text>'
            )
        cursor += seg_w
    draw_legend(lines, entries=[(row.auto_label, label_color(row.auto_label)) for row in label_df.itertuples(index=False)], x=56, y=256, row_h=20)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_cluster_markdown(cluster_df: pd.DataFrame, limit: int) -> str:
    lines = ["# Cluster Review", ""]
    top = cluster_df.sort_values("size", ascending=False).head(limit)
    for _, row in top.iterrows():
        lines.extend(
            [
                f"## Cluster {int(row['cluster_id'])} - {row['auto_label']}",
                f"- Size: {int(row['size'])} chunks ({row['fraction'] * 100:.2f}%)",
                f"- Mean compressibility: {row['mean_compressibility']:.3f}",
                f"- Mean operator density: {row['mean_operator_density']:.4f}",
                f"- Mean code markers: {row['mean_code_marker_count']:.2f}",
                f"- Representative shard/chunk: {int(row['representative_shard'])}:{int(row['representative_chunk'])}",
                f"- Sample: `{str(row['representative_sample'])}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_review_html(
    output_path: Path,
    summary: dict[str, object],
    cluster_df: pd.DataFrame,
    label_df: pd.DataFrame,
    plot_files: list[Path],
) -> None:
    top_clusters = cluster_df.sort_values("size", ascending=False).head(16)
    metric_cards = [
        ("Sampled chunks", f"{int(summary['sample_chunks']):,}"),
        ("Median compressibility", f"{float(summary['sample_compressibility_p50']):.3f}"),
        ("95th percentile compressibility", f"{float(summary['sample_compressibility_p95']):.3f}"),
        ("Median operator density", f"{float(summary['sample_operator_density_p50']):.4f}"),
        ("URL-heavy chunks", format_percent(float(summary["url_fraction"]))),
        ("Email-heavy chunks", format_percent(float(summary["email_fraction"]))),
    ]
    top_label = label_df.iloc[0]
    second_label = label_df.iloc[1] if len(label_df) > 1 else None
    takeaways = [
        (
            "The corpus is overwhelmingly ordinary web prose.",
            f"{top_label['auto_label']} covers {format_percent(float(top_label['fraction']))} of the sampled chunks, so broad prose remains the default training environment.",
        ),
        (
            "The formal / logic slice is real, but not dominant.",
            f"{second_label['auto_label']} covers {format_percent(float(second_label['fraction']))} of the sample, large enough to target with a phase, but too small to drive the whole run."
            if second_label is not None
            else "Only one major label was present in the current sample.",
        ),
        (
            "The low-compressibility tail is where filtering pressure belongs.",
            "That tail looks multilingual, garbled, or boilerplate-heavy in cluster examples, which matches the current learnability-filter hypothesis.",
        ),
    ]
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Curriculum Corpus Review</title>",
        (
            "<style>"
            "body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "max-width:1280px;margin:0 auto;padding:28px 18px 56px;background:#f4efe7;color:#172132}"
            "h1,h2,h3{margin:0 0 10px 0;color:#142033}"
            "h1{font-size:40px;line-height:1.05;font-family:Georgia,'Times New Roman',serif}"
            "h2{font-size:22px;margin-top:28px}"
            "p{line-height:1.55;color:#495568}"
            ".hero,.section,.plot-card,.metric,.takeaway,.table-wrap{background:#fffdf8;border:1px solid #d9e1ea;border-radius:22px;box-shadow:0 10px 28px rgba(20,32,51,0.06)}"
            ".hero{padding:28px 30px;background:linear-gradient(135deg,#fffaf2 0%,#fffdf8 55%,#edf4fb 100%)}"
            ".eyebrow{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#8b5e34}"
            ".subtitle{max-width:860px;font-size:17px;margin-top:10px}"
            ".grid{display:grid;gap:16px}"
            ".metrics{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:18px}"
            ".takeaways{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:20px}"
            ".metric,.takeaway{padding:18px 18px 16px}"
            ".metric .label{font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#7a5b35}"
            ".metric .value{font-size:28px;font-weight:700;margin-top:8px}"
            ".takeaway h3{font-size:17px;margin-bottom:8px}"
            ".plot-grid{display:grid;grid-template-columns:1fr;gap:18px}"
            ".plot-card{padding:18px 18px 10px}"
            ".plot-caption{font-size:14px;color:#5b6675;margin:0 0 10px 0}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border-bottom:1px solid #e5ebf1;padding:10px 12px;text-align:left;vertical-align:top}"
            "th{background:#f8fafc;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#5b6675}"
            "code{background:#f4efe7;padding:2px 6px;border-radius:6px}"
            "svg{display:block;width:100%;height:auto;border-radius:18px}"
            ".table-wrap{padding:12px 14px 18px;margin-top:18px}"
            ".pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#f4efe7;color:#704f28;font-size:12px;font-weight:700}"
            "@media (max-width: 960px){.metrics,.takeaways{grid-template-columns:1fr}}"
            "</style></head><body>"
        ),
        "<section class='hero'>",
        "<div class='eyebrow'>Parameter Golf Data Review</div>",
        "<h1>Curriculum Corpus Map</h1>",
        "<p class='subtitle'>A human-readable picture of what the FineWeb training stream actually looks like, where the structured slices live, and which regions are plausible curriculum targets.</p>",
        "<div class='grid metrics'>",
    ]
    for label, value in metric_cards:
        parts.append(
            f"<div class='metric'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
        )
    parts.extend(["</div>", "<div class='grid takeaways'>"])
    for heading, text in takeaways:
        parts.append(f"<div class='takeaway'><h3>{html.escape(heading)}</h3><p>{html.escape(text)}</p></div>")
    parts.extend(["</div>", "</section>", "<section class='section'>", "<h2>Visual Review</h2>", "<p>The plots below have been simplified to emphasize stable structure over raw point clouds. Cluster maps use centroid bubbles, not thousands of overlapping samples.</p>", "<div class='plot-grid'>"])
    for plot in plot_files:
        title = plot.stem.replace("_", " ").title()
        parts.append(f"<div class='plot-card'><h3>{html.escape(title)}</h3>{plot.read_text(encoding='utf-8')}</div>")
    parts.extend(
        [
            "</div>",
            "</section>",
            "<section class='table-wrap'>",
            "<h2>Top Clusters</h2>",
            "<p><span class='pill'>Cluster examples</span> The table is still the fastest way to inspect representative snippets, but the plots should now do the heavy lifting for shape and composition.</p>",
            "<table><thead><tr><th>Cluster</th><th>Label</th><th>Size</th><th>Compressibility</th><th>Operator density</th><th>Code markers</th><th>Sample</th></tr></thead><tbody>",
        ]
    )
    for _, row in top_clusters.iterrows():
        parts.append(
            "<tr>"
            f"<td>{int(row['cluster_id'])}</td>"
            f"<td>{html.escape(str(row['auto_label']))}</td>"
            f"<td>{int(row['size'])} ({row['fraction'] * 100:.2f}%)</td>"
            f"<td>{row['mean_compressibility']:.3f}</td>"
            f"<td>{row['mean_operator_density']:.4f}</td>"
            f"<td>{row['mean_code_marker_count']:.2f}</td>"
            f"<td><code>{html.escape(str(row['representative_sample']))}</code></td>"
            "</tr>"
        )
    parts.extend(["</tbody></table>", "</section>", "</body></html>"])
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = list_shards(args.data_glob, args.sample_shards)
    chunk_counts = [shard_chunk_count(path, args.chunk_size) for path in shard_paths]
    sample_alloc = allocate_samples(chunk_counts, args.sample_chunks)
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    frame = build_sample_frame(
        shard_paths=shard_paths,
        sample_alloc=sample_alloc,
        chunk_size=args.chunk_size,
        tokenizer_path=tokenizer_path,
        seed=args.seed,
        histogram_bins=args.histogram_bins,
        num_workers=args.num_workers,
    )

    cluster_ids, centers = run_clustering(frame, args.num_clusters, args.kmeans_iterations, args.seed)
    frame["cluster_id"] = cluster_ids.astype(np.int32)
    pca_df = pca_frame(frame)
    hist = np.asarray(frame.attrs["histograms"], dtype=np.float32)
    np.savez_compressed(out_dir / "sample_histograms.npz", histograms=hist, cluster_centers=centers, cluster_ids=cluster_ids)

    summaries = cluster_summaries(frame, centers)
    cluster_df = pd.DataFrame.from_records([asdict(item) for item in summaries])
    cluster_df["baseline_loss"] = np.nan
    cluster_df["sidecar_delta"] = np.nan
    centroid_df = (
        pca_df.groupby("cluster_id", as_index=False)
        .agg(
            pca_x=("pca_x", "mean"),
            pca_y=("pca_y", "mean"),
            mean_bytes_per_token=("bytes_per_token", "mean"),
            mean_type_token_ratio=("type_token_ratio", "mean"),
            mean_operator_density_raw=("operator_density", "mean"),
            mean_compressibility_raw=("compressibility", "mean"),
        )
    )
    cluster_df = cluster_df.merge(centroid_df, on="cluster_id", how="left")
    label_map = cluster_df[["cluster_id", "auto_label"]].copy()
    pca_df = pca_df.merge(label_map, on="cluster_id", how="left")
    label_df = (
        cluster_df.groupby("auto_label", as_index=False)["size"]
        .sum()
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    label_df["fraction"] = label_df["size"] / max(int(len(frame)), 1)

    sample_csv = out_dir / "curriculum_data_map_sample.csv.gz"
    save_dataframe(pca_df, sample_csv)
    cluster_csv = out_dir / "curriculum_cluster_summary.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    cluster_md = out_dir / "curriculum_cluster_examples.md"
    cluster_md.write_text(top_cluster_markdown(cluster_df, args.max_cluster_examples), encoding="utf-8")

    summary = {
        "data_glob": args.data_glob,
        "tokenizer_path": str(tokenizer_path),
        "num_shards": len(shard_paths),
        "sample_chunks": int(len(frame)),
        "sample_chunk_target": int(args.sample_chunks),
        "chunk_size": int(args.chunk_size),
        "num_clusters": int(args.num_clusters),
        "num_workers": int(args.num_workers),
        "sample_bytes_per_token_p50": quantile(frame["bytes_per_token"], 0.50),
        "sample_bytes_per_token_p95": quantile(frame["bytes_per_token"], 0.95),
        "sample_compressibility_p50": quantile(frame["compressibility"], 0.50),
        "sample_compressibility_p95": quantile(frame["compressibility"], 0.95),
        "sample_operator_density_p50": quantile(frame["operator_density"], 0.50),
        "sample_operator_density_p95": quantile(frame["operator_density"], 0.95),
        "url_fraction": float(frame["has_url"].mean()),
        "email_fraction": float(frame["has_email"].mean()),
        "code_fraction": float(frame["has_code_blocks"].mean()),
        "numbers_heavy_fraction": float(frame["has_numbers_heavy"].mean()),
    }
    write_json(out_dir / "curriculum_data_map_summary.json", summary)

    plot_files: list[Path] = []
    hist_path = plot_dir / "compressibility_histogram.svg"
    hist_svg(
        frame["compressibility"],
        "Compressibility Distribution",
        "compressibility",
        hist_path,
        subtitle="Higher compressibility usually means more reusable document structure. The noisy tail is the main candidate for filtering.",
    )
    plot_files.append(hist_path)
    hist_path = plot_dir / "bytes_per_token_histogram.svg"
    hist_svg(
        frame["bytes_per_token"],
        "Bytes Per Token Distribution",
        "bytes per token",
        hist_path,
        subtitle="This is the tokenizer-efficiency view: most chunks cluster tightly, with only a modest tail of unusually expensive tokenization.",
    )
    plot_files.append(hist_path)
    hist_path = plot_dir / "operator_density_histogram.svg"
    hist_svg(
        frame["operator_density"],
        "Operator Density Distribution",
        "operators / token",
        hist_path,
        subtitle="Operator-heavy text exists, but it is not the default. That makes dedicated logic phases valuable and also capacity-limited.",
    )
    plot_files.append(hist_path)
    mix_path = plot_dir / "label_mix.svg"
    label_mix_svg(label_df, "Corpus Mix By High-Level Label", mix_path)
    plot_files.append(mix_path)
    bar_path = plot_dir / "top_cluster_sizes.svg"
    bar_svg(cluster_df, "Top Cluster Sizes", bar_path)
    plot_files.append(bar_path)
    scatter_path = plot_dir / "pca_cluster_map.svg"
    scatter_svg(
        cluster_df,
        x_col="pca_x",
        y_col="pca_y",
        title="Feature Landscape: Cluster Centroids In PCA Space",
        x_label="PCA 1",
        y_label="PCA 2",
        output_path=scatter_path,
        subtitle="Each bubble is one structural cluster. Distance reflects feature similarity; bubble size reflects how much of the corpus lives there.",
        annotate_ids=interesting_cluster_ids(cluster_df, limit=14),
    )
    plot_files.append(scatter_path)
    scatter_path = plot_dir / "compressibility_vs_operator_density.svg"
    scatter_svg(
        cluster_df.rename(columns={"mean_compressibility": "compressibility", "mean_operator_density": "operator_density"}),
        x_col="compressibility",
        y_col="operator_density",
        title="Structural Opportunity Map",
        x_label="cluster mean compressibility",
        y_label="cluster mean operator density",
        output_path=scatter_path,
        subtitle="Upper-right clusters are the best candidates for logic-aware or structure-aware scheduling. Lower-left clusters look more like noise or generic web tail.",
        annotate_ids=interesting_cluster_ids(cluster_df, limit=14),
    )
    plot_files.append(scatter_path)

    write_review_html(out_dir / "curriculum_data_map_review.html", summary, cluster_df, label_df, plot_files)
    print(json.dumps({"output_dir": out_dir.as_posix(), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
