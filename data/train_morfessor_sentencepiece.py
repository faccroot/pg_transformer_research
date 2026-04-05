#!/usr/bin/env python3
"""Train a Morfessor-seeded SentencePiece BPE tokenizer.

This is a filtered Option B pipeline:
- Count raw word frequencies from docs_selected.jsonl.
- Train Morfessor only on morphologically interesting words rather than on the
  whole Zipf-heavy vocabulary.
- Rank candidate symbols by segmented frequency and by reuse across distinct
  words.
- Screen candidates against raw substring bleed before protecting them as
  SentencePiece user_defined_symbols.

The output is a normal SentencePiece `.model` file plus JSON metadata, so the
existing shard/export pipeline can reuse it without changes to trainer code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sentencepiece as spm


WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", flags=re.UNICODE)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iter_doc_texts(path: Path, *, max_docs: int | None = None) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_docs is not None and i >= max_docs:
                break
            payload = json.loads(line)
            text = str(payload["text"]).replace("\x00", " ").strip()
            if text:
                yield text


def iter_words(
    text: str,
    *,
    lowercase: bool,
    min_word_chars: int,
    max_word_chars: int,
) -> Iterable[str]:
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        if lowercase:
            word = word.lower()
        if not (min_word_chars <= len(word) <= max_word_chars):
            continue
        yield word


def build_word_counts(
    docs_jsonl: Path,
    *,
    max_docs: int | None,
    lowercase: bool,
    min_word_chars: int,
    max_word_chars: int,
    min_word_count: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in iter_doc_texts(docs_jsonl, max_docs=max_docs):
        counts.update(
            iter_words(
                text,
                lowercase=lowercase,
                min_word_chars=min_word_chars,
                max_word_chars=max_word_chars,
            )
        )
    if min_word_count <= 1:
        return counts
    return Counter({word: count for word, count in counts.items() if count >= min_word_count})


def filter_word_counts(
    word_counts: Counter[str],
    *,
    min_word_chars: int,
    max_word_chars: int,
    min_word_count: int,
    max_word_count: int | None,
    alpha_only: bool,
) -> Counter[str]:
    filtered: Counter[str] = Counter()
    for word, count in word_counts.items():
        if len(word) < min_word_chars or len(word) > max_word_chars:
            continue
        if count < min_word_count:
            continue
        if max_word_count is not None and count > max_word_count:
            continue
        if alpha_only and not word.isalpha():
            continue
        filtered[word] = count
    return filtered


def iter_sentencepiece_text(docs_jsonl: Path, *, max_docs: int | None = None) -> Iterable[str]:
    yield from iter_doc_texts(docs_jsonl, max_docs=max_docs)


@dataclass
class MorphCandidate:
    symbol: str
    segmented_total: int = 0
    segmented_word_count: int = 0
    segmented_prefix: int = 0
    segmented_suffix: int = 0
    segmented_whole: int = 0
    segmented_interior: int = 0
    raw_total: int = 0
    raw_prefix: int = 0
    raw_suffix: int = 0
    raw_interior: int = 0
    sanctioned_ratio: float = 0.0
    boundary_bias: float = 0.0
    selected: bool = False


def train_morfessor_model(
    word_counts: Counter[str],
    *,
    finish_threshold: float,
    max_epochs: int | None,
):
    try:
        import morfessor
    except ImportError as exc:
        raise RuntimeError("morfessor is required for this script; install it with `pip install morfessor`") from exc

    model = morfessor.BaselineModel()
    data = ((count, tuple(word)) for word, count in word_counts.items())
    model.load_data(data, freqthreshold=1)
    model.train_batch(finish_threshold=finish_threshold, max_epochs=max_epochs)
    return model


def segment_word(model, word: str, *, max_segment_len: int) -> list[str]:
    pieces, _ = model.viterbi_segment(tuple(word), maxlen=max_segment_len)
    return ["".join(piece) for piece in pieces]


def collect_segmented_candidates(
    model,
    word_counts: Counter[str],
    *,
    max_segment_len: int,
) -> dict[str, MorphCandidate]:
    candidates: dict[str, MorphCandidate] = {}
    for word, count in word_counts.items():
        pieces = segment_word(model, word, max_segment_len=max_segment_len)
        piece_count = len(pieces)
        seen_symbols_in_word: set[str] = set()
        for idx, piece in enumerate(pieces):
            candidate = candidates.setdefault(piece, MorphCandidate(symbol=piece))
            candidate.segmented_total += count
            if piece not in seen_symbols_in_word:
                candidate.segmented_word_count += 1
                seen_symbols_in_word.add(piece)
            if piece_count == 1:
                candidate.segmented_whole += count
            elif idx == 0:
                candidate.segmented_prefix += count
            elif idx == piece_count - 1:
                candidate.segmented_suffix += count
            else:
                candidate.segmented_interior += count
    return candidates


def add_raw_substring_stats(
    candidates: dict[str, MorphCandidate],
    word_counts: Counter[str],
) -> None:
    symbols = sorted(candidates.keys(), key=len, reverse=True)
    for word, count in word_counts.items():
        for symbol in symbols:
            start = 0
            while True:
                idx = word.find(symbol, start)
                if idx < 0:
                    break
                candidate = candidates[symbol]
                candidate.raw_total += count
                if idx == 0:
                    candidate.raw_prefix += count
                if idx + len(symbol) == len(word):
                    candidate.raw_suffix += count
                if idx > 0 and idx + len(symbol) < len(word):
                    candidate.raw_interior += count
                start = idx + 1


def finalize_candidate_stats(candidates: dict[str, MorphCandidate]) -> list[MorphCandidate]:
    rows = list(candidates.values())
    for row in rows:
        if row.raw_total > 0:
            row.sanctioned_ratio = row.segmented_total / row.raw_total
        if row.segmented_total > 0:
            row.boundary_bias = (
                row.segmented_prefix + row.segmented_suffix + row.segmented_whole
            ) / row.segmented_total
    rows.sort(key=lambda row: (-row.segmented_total, row.symbol))
    return rows


def select_seed_symbols(
    candidates: list[MorphCandidate],
    *,
    top_morphemes: int,
    min_symbol_chars: int,
    max_symbol_chars: int,
    min_segmented_count: int,
    min_reuse_word_count: int,
    min_sanctioned_ratio: float,
    min_boundary_bias: float,
    max_raw_interior_ratio: float,
    max_whole_ratio: float,
    alpha_symbols_only: bool,
) -> list[MorphCandidate]:
    selected: list[MorphCandidate] = []
    for row in candidates:
        if len(row.symbol) < min_symbol_chars or len(row.symbol) > max_symbol_chars:
            continue
        if alpha_symbols_only and not row.symbol.isalpha():
            continue
        if row.segmented_total < min_segmented_count:
            continue
        if row.segmented_word_count < min_reuse_word_count:
            continue
        if row.raw_total <= 0:
            continue
        if row.sanctioned_ratio < min_sanctioned_ratio:
            continue
        if row.boundary_bias < min_boundary_bias:
            continue
        if (row.raw_interior / row.raw_total) > max_raw_interior_ratio:
            continue
        if (row.segmented_whole / row.segmented_total) > max_whole_ratio:
            continue
        row.selected = True
        selected.append(row)
        if len(selected) >= top_morphemes:
            break
    return selected


def train_sentencepiece(
    *,
    docs_jsonl: Path,
    output_prefix: Path,
    vocab_size: int,
    tokenizer_train_docs: int | None,
    user_defined_symbols: list[str],
) -> tuple[Path, Path]:
    model_path = output_prefix.with_suffix(".model")
    vocab_path = output_prefix.with_suffix(".vocab")
    for path in (model_path, vocab_path):
        if path.exists():
            path.unlink()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_sentencepiece_text(docs_jsonl, max_docs=tokenizer_train_docs),
        model_prefix=str(output_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
        user_defined_symbols=user_defined_symbols,
    )
    return model_path, vocab_path


def evaluate_tokenizer(
    model_path: Path,
    docs_jsonl: Path,
    *,
    max_docs: int | None,
    protected_symbols: set[str],
) -> dict[str, object]:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    total_bytes = 0
    total_tokens = 0
    protected_token_count = 0
    protected_piece_counts: Counter[str] = Counter()

    for text in iter_doc_texts(docs_jsonl, max_docs=max_docs):
        pieces = sp.encode(text, out_type=str)
        total_bytes += len(text.encode("utf-8", errors="replace"))
        total_tokens += len(pieces)
        for piece in pieces:
            if piece in protected_symbols:
                protected_token_count += 1
                protected_piece_counts[piece] += 1

    bytes_per_token = (total_bytes / total_tokens) if total_tokens else None
    return {
        "docs": max_docs,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": bytes_per_token,
        "protected_token_fraction": (protected_token_count / total_tokens) if total_tokens else None,
        "top_emitted_protected_symbols": protected_piece_counts.most_common(25),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Morfessor-seeded SentencePiece tokenizer")
    parser.add_argument("--docs-jsonl", required=True, help="Path to docs_selected.jsonl")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for .model/.vocab and metadata")
    parser.add_argument("--vocab-size", type=int, default=4096, help="SentencePiece vocab size")
    parser.add_argument("--morfessor-train-docs", type=int, default=200000, help="Docs used for word-count and Morfessor training")
    parser.add_argument("--tokenizer-train-docs", type=int, default=200000, help="Docs used for SentencePiece training")
    parser.add_argument("--eval-docs", type=int, default=2000, help="Docs used for bytes-per-token profiling")
    parser.add_argument("--top-morphemes", type=int, default=500, help="Maximum number of protected morphemes to keep")
    parser.add_argument("--candidate-pool-multiplier", type=int, default=8, help="How many ranked Morfessor candidates to inspect before filtering")
    parser.add_argument("--count-min-word-count", type=int, default=1, help="Minimum count retained in the raw word-count table")
    parser.add_argument("--count-min-word-chars", type=int, default=2, help="Minimum raw word length to count")
    parser.add_argument("--count-max-word-chars", type=int, default=48, help="Maximum raw word length to count")
    parser.add_argument("--morfessor-min-word-count", type=int, default=50, help="Minimum count for words included in Morfessor training")
    parser.add_argument("--morfessor-max-word-count", type=int, default=500000, help="Maximum count for words included in Morfessor training")
    parser.add_argument("--morfessor-min-word-chars", type=int, default=6, help="Minimum word length for Morfessor training")
    parser.add_argument("--morfessor-max-word-chars", type=int, default=48, help="Maximum word length for Morfessor training")
    parser.add_argument("--allow-nonalpha-morfessor-words", action="store_true", help="Allow non-alphabetic words in Morfessor training candidates")
    parser.add_argument("--preserve-case", action="store_true", help="Keep original casing when building word counts")
    parser.add_argument("--min-symbol-chars", type=int, default=2, help="Minimum selected morpheme length")
    parser.add_argument("--max-symbol-chars", type=int, default=8, help="Maximum selected morpheme length")
    parser.add_argument("--min-segmented-count", type=int, default=100, help="Minimum Morfessor segmented count for a seed")
    parser.add_argument("--min-reuse-word-count", type=int, default=10, help="Minimum number of distinct words a candidate must appear in")
    parser.add_argument("--min-sanctioned-ratio", type=float, default=0.7, help="Minimum segmented/raw substring ratio")
    parser.add_argument("--min-boundary-bias", type=float, default=0.7, help="Minimum boundary-biased segmented usage")
    parser.add_argument("--max-raw-interior-ratio", type=float, default=0.10, help="Maximum fraction of raw matches that are strictly interior")
    parser.add_argument("--max-whole-ratio", type=float, default=0.80, help="Maximum fraction of segmented uses allowed as whole-word matches")
    parser.add_argument("--allow-nonalpha-symbols", action="store_true", help="Allow non-alphabetic protected symbols")
    parser.add_argument("--morfessor-max-segment-len", type=int, default=30, help="Maximum Morfessor segment length during Viterbi segmentation")
    parser.add_argument("--morfessor-finish-threshold", type=float, default=0.005, help="Batch-training finish threshold for Morfessor")
    parser.add_argument("--morfessor-max-epochs", type=int, default=None, help="Optional max epoch cap for Morfessor batch training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    lowercase = not args.preserve_case
    raw_word_counts = build_word_counts(
        docs_jsonl,
        max_docs=args.morfessor_train_docs,
        lowercase=lowercase,
        min_word_chars=args.count_min_word_chars,
        max_word_chars=args.count_max_word_chars,
        min_word_count=args.count_min_word_count,
    )
    if not raw_word_counts:
        raise ValueError("No word counts were collected; adjust the corpus or filters.")

    morfessor_word_counts = filter_word_counts(
        raw_word_counts,
        min_word_chars=args.morfessor_min_word_chars,
        max_word_chars=args.morfessor_max_word_chars,
        min_word_count=args.morfessor_min_word_count,
        max_word_count=args.morfessor_max_word_count,
        alpha_only=not args.allow_nonalpha_morfessor_words,
    )
    if not morfessor_word_counts:
        raise ValueError("No Morfessor training words survived filtering; relax the morphology candidate filters.")

    model = train_morfessor_model(
        morfessor_word_counts,
        finish_threshold=args.morfessor_finish_threshold,
        max_epochs=args.morfessor_max_epochs,
    )
    candidates = collect_segmented_candidates(
        model,
        morfessor_word_counts,
        max_segment_len=args.morfessor_max_segment_len,
    )
    ranked = finalize_candidate_stats(candidates)
    candidate_pool_size = max(args.top_morphemes * args.candidate_pool_multiplier, args.top_morphemes)
    candidate_pool = ranked[:candidate_pool_size]
    filtered_lookup = {row.symbol: row for row in candidate_pool}
    add_raw_substring_stats(filtered_lookup, raw_word_counts)
    candidate_pool = finalize_candidate_stats(filtered_lookup)
    selected = select_seed_symbols(
        candidate_pool,
        top_morphemes=args.top_morphemes,
        min_symbol_chars=args.min_symbol_chars,
        max_symbol_chars=args.max_symbol_chars,
        min_segmented_count=args.min_segmented_count,
        min_reuse_word_count=args.min_reuse_word_count,
        min_sanctioned_ratio=args.min_sanctioned_ratio,
        min_boundary_bias=args.min_boundary_bias,
        max_raw_interior_ratio=args.max_raw_interior_ratio,
        max_whole_ratio=args.max_whole_ratio,
        alpha_symbols_only=not args.allow_nonalpha_symbols,
    )
    user_defined_symbols = [row.symbol for row in selected]
    model_path: Path | None = None
    vocab_path: Path | None = None
    evaluation: dict[str, object] | None = None

    if user_defined_symbols:
        model_path, vocab_path = train_sentencepiece(
            docs_jsonl=docs_jsonl,
            output_prefix=output_prefix,
            vocab_size=args.vocab_size,
            tokenizer_train_docs=args.tokenizer_train_docs,
            user_defined_symbols=user_defined_symbols,
        )
        evaluation = evaluate_tokenizer(
            model_path,
            docs_jsonl,
            max_docs=args.eval_docs,
            protected_symbols=set(user_defined_symbols),
        )

    summary = {
        "created_at_utc": now_utc(),
        "docs_jsonl": str(docs_jsonl),
        "output_prefix": str(output_prefix),
        "vocab_size": args.vocab_size,
        "morfessor_train_docs": args.morfessor_train_docs,
        "tokenizer_train_docs": args.tokenizer_train_docs,
        "eval_docs": args.eval_docs,
        "raw_word_counts_unique": len(raw_word_counts),
        "raw_word_counts_total": int(sum(raw_word_counts.values())),
        "morfessor_word_counts_unique": len(morfessor_word_counts),
        "morfessor_word_counts_total": int(sum(morfessor_word_counts.values())),
        "selected_symbol_count": len(user_defined_symbols),
        "selected_symbols_path": str(output_prefix.with_suffix(".user_symbols.txt")),
        "sentencepiece_model_path": None if model_path is None else str(model_path),
        "sentencepiece_vocab_path": None if vocab_path is None else str(vocab_path),
        "evaluation": evaluation,
        "morfessor_candidate_filters": {
            "min_word_chars": args.morfessor_min_word_chars,
            "max_word_chars": args.morfessor_max_word_chars,
            "min_word_count": args.morfessor_min_word_count,
            "max_word_count": args.morfessor_max_word_count,
            "alpha_only": not args.allow_nonalpha_morfessor_words,
        },
        "selection_thresholds": {
            "min_symbol_chars": args.min_symbol_chars,
            "max_symbol_chars": args.max_symbol_chars,
            "min_segmented_count": args.min_segmented_count,
            "min_reuse_word_count": args.min_reuse_word_count,
            "min_sanctioned_ratio": args.min_sanctioned_ratio,
            "min_boundary_bias": args.min_boundary_bias,
            "max_raw_interior_ratio": args.max_raw_interior_ratio,
            "max_whole_ratio": args.max_whole_ratio,
            "alpha_symbols_only": not args.allow_nonalpha_symbols,
        },
        "selected_symbols": user_defined_symbols,
        "candidate_pool": [asdict(row) for row in candidate_pool],
    }

    output_prefix.with_suffix(".user_symbols.txt").write_text("\n".join(user_defined_symbols) + ("\n" if user_defined_symbols else ""), encoding="utf-8")
    write_json(output_prefix.with_suffix(".summary.json"), summary)
    if not user_defined_symbols:
        raise ValueError("No safe Morfessor seed symbols survived filtering; see the written .summary.json candidate pool for diagnostics.")
    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "selected_symbol_count": len(user_defined_symbols),
                "bytes_per_token": evaluation["bytes_per_token"],
                "protected_token_fraction": evaluation["protected_token_fraction"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
