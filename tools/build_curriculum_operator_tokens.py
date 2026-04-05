#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm


DEFAULT_WORD_TERMS = [
    "if",
    "and",
    "or",
    "not",
    "because",
    "then",
    "but",
    "therefore",
    "however",
    "unless",
    "while",
    "whereas",
    "thus",
    "hence",
]
DEFAULT_PUNCT_TERMS = [":", ";", "(", ")", "[", "]", "{", "}"]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalized_word_piece(piece: str) -> str:
    return piece.lstrip("▁").casefold()


def byte_piece_for_char(ch: str) -> str | None:
    if len(ch) != 1:
        return None
    code = ord(ch)
    if code < 0 or code > 255:
        return None
    return f"<0x{code:02X}>"


def build_operator_payload(
    sp: spm.SentencePieceProcessor,
    word_terms: list[str],
    punct_terms: list[str],
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    matched_ids: set[int] = set()
    unmatched_words: list[str] = []
    unmatched_punct: list[str] = []

    vocab_size = sp.get_piece_size()
    pieces = [sp.id_to_piece(idx) for idx in range(vocab_size)]

    for term in word_terms:
        target = term.casefold()
        term_matches = []
        for idx, piece in enumerate(pieces):
            if normalized_word_piece(piece) == target:
                term_matches.append((idx, piece))
        if not term_matches:
            unmatched_words.append(term)
            continue
        for idx, piece in term_matches:
            matched_ids.add(idx)
            matches.append({"id": idx, "piece": piece, "target": term, "kind": "word"})

    for term in punct_terms:
        exact_matches = []
        byte_piece = byte_piece_for_char(term)
        for idx, piece in enumerate(pieces):
            if piece == term or (byte_piece is not None and piece == byte_piece):
                exact_matches.append((idx, piece))
        if not exact_matches:
            unmatched_punct.append(term)
            continue
        for idx, piece in exact_matches:
            matched_ids.add(idx)
            matches.append({"id": idx, "piece": piece, "target": term, "kind": "punct"})

    matches.sort(key=lambda item: (int(item["id"]), str(item["target"])))
    return {
        "token_ids": sorted(matched_ids),
        "matched_pieces": matches,
        "word_terms": word_terms,
        "punct_terms": punct_terms,
        "unmatched_word_terms": unmatched_words,
        "unmatched_punct_terms": unmatched_punct,
        "vocab_size": vocab_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tokenizer-aligned operator token id artifact for curriculum scoring.")
    parser.add_argument("tokenizer_path", help="SentencePiece .model path")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--word-terms", default=",".join(DEFAULT_WORD_TERMS), help="Comma-separated reasoning lexemes")
    parser.add_argument("--punct-terms", default=",".join(DEFAULT_PUNCT_TERMS), help="Comma-separated punctuation operators")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    payload = build_operator_payload(
        sp,
        word_terms=parse_csv(args.word_terms),
        punct_terms=parse_csv(args.punct_terms),
    )
    payload["tokenizer_path"] = tokenizer_path.as_posix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
