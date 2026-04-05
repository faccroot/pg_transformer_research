from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sentencepiece as spm

from logic_register_mlx import (
    AND_OPERATOR_SURFACES,
    NOT_OPERATOR_SURFACES,
    OR_OPERATOR_SURFACES,
)


IF_FAMILY_SURFACES = frozenset(
    {
        "if",
        "whether",
        "provided",
        "provided that",
        "assuming",
        "assuming that",
        "supposing",
        "supposing that",
        "in case",
    }
)

BECAUSE_FAMILY_SURFACES = frozenset(
    {
        "because",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "as a result",
    }
)

MAYBE_FAMILY_SURFACES = frozenset(
    {
        "maybe",
        "perhaps",
        "possibly",
        "probably",
        "likely",
        "unlikely",
        "might",
        "may",
    }
)


DEFAULT_FAMILY_SURFACES: dict[str, frozenset[str]] = {
    "not": NOT_OPERATOR_SURFACES,
    "and": AND_OPERATOR_SURFACES,
    "or": OR_OPERATOR_SURFACES,
    "if": IF_FAMILY_SURFACES,
    "because": BECAUSE_FAMILY_SURFACES,
    "maybe": MAYBE_FAMILY_SURFACES,
}


@dataclass(frozen=True)
class LexicalFamilyRoutingSpec:
    family_names: tuple[str, ...]
    lookup: np.ndarray
    piece_starts_word: np.ndarray
    piece_has_alnum: np.ndarray
    patterns_by_first_token: dict[int, tuple[tuple[np.ndarray, int], ...]]


def parse_family_list(raw: str) -> tuple[str, ...]:
    text = (raw or "").strip().lower()
    if not text or text == "operators":
        return ("not", "and", "or")
    if text in {"all", "control", "nsm"}:
        return tuple(DEFAULT_FAMILY_SURFACES.keys())
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in DEFAULT_FAMILY_SURFACES]
    if unknown:
        raise ValueError(f"Unknown lexical families: {unknown}")
    return names


def _normalize_piece(piece: str) -> str:
    return piece.lower().replace("’", "'").lstrip("▁").strip()


def _extract_pattern_ids(
    sp: spm.SentencePieceProcessor,
    text: str,
    target_begin: int,
    target_end: int,
) -> tuple[int, ...]:
    proto = sp.encode_as_immutable_proto(text)
    return tuple(
        int(piece.id)
        for piece in proto.pieces
        if not (int(piece.end) <= target_begin or int(piece.begin) >= target_end)
    )


def _surface_pattern_sequences(sp: spm.SentencePieceProcessor, surface: str) -> set[tuple[int, ...]]:
    patterns: set[tuple[int, ...]] = set()
    raw_ids = _extract_pattern_ids(sp, surface, 0, len(surface))
    if raw_ids:
        patterns.add(raw_ids)
    sentence_text = f"a {surface} b"
    sentence_ids = _extract_pattern_ids(sp, sentence_text, 1, 2 + len(surface))
    if sentence_ids:
        patterns.add(sentence_ids)
    return patterns


def build_family_routing_spec(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    family_names: tuple[str, ...],
) -> LexicalFamilyRoutingSpec:
    family_ids = {name: idx + 1 for idx, name in enumerate(family_names)}
    lookup = np.zeros((vocab_size,), dtype=np.int32)
    piece_starts_word = np.zeros((vocab_size,), dtype=np.int32)
    piece_has_alnum = np.zeros((vocab_size,), dtype=np.int32)

    single_piece_map: dict[str, int] = {}
    for name in family_names:
        code = family_ids[name]
        for surface in DEFAULT_FAMILY_SURFACES[name]:
            normalized = _normalize_piece(surface)
            if normalized:
                single_piece_map.setdefault(normalized, code)

    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        normalized = _normalize_piece(piece)
        piece_starts_word[token_id] = int(piece.startswith("▁"))
        piece_has_alnum[token_id] = int(any(ch.isalnum() for ch in normalized))
        if piece.startswith("▁"):
            code = single_piece_map.get(normalized)
            if code is not None:
                lookup[token_id] = np.int32(code)

    pattern_groups: dict[int, dict[tuple[int, ...], int]] = {}
    for name in family_names:
        code = family_ids[name]
        for surface in sorted(DEFAULT_FAMILY_SURFACES[name]):
            for pattern in _surface_pattern_sequences(sp, surface):
                existing = pattern_groups.setdefault(len(pattern), {}).get(pattern)
                if existing is not None and existing != code:
                    continue
                pattern_groups.setdefault(len(pattern), {})[pattern] = code

    patterns_by_first_token: dict[int, list[tuple[np.ndarray, int]]] = {}
    for _, mapping in sorted(pattern_groups.items()):
        for token_ids, code in mapping.items():
            token_arr = np.array(token_ids, dtype=np.int32)
            patterns_by_first_token.setdefault(int(token_arr[0]), []).append((token_arr, int(code)))
    patterns_by_first_token_final = {
        first_token: tuple(sorted(candidates, key=lambda item: (int(item[0].shape[0]), tuple(int(v) for v in item[0]))))
        for first_token, candidates in patterns_by_first_token.items()
    }

    return LexicalFamilyRoutingSpec(
        family_names=family_names,
        lookup=lookup,
        piece_starts_word=piece_starts_word,
        piece_has_alnum=piece_has_alnum,
        patterns_by_first_token=patterns_by_first_token_final,
    )


def detect_family_codes_np(input_ids: np.ndarray, routing: LexicalFamilyRoutingSpec) -> np.ndarray:
    input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
    family_codes = np.take(routing.lookup, input_ids, axis=0)
    if not routing.patterns_by_first_token:
        return family_codes

    batch_size, seq_len = input_ids.shape
    piece_starts_word = routing.piece_starts_word
    piece_has_alnum = routing.piece_has_alnum
    for batch_idx in range(batch_size):
        row = input_ids[batch_idx]
        row_codes = family_codes[batch_idx]
        for start in range(seq_len):
            token_id = int(row[start])
            candidates = routing.patterns_by_first_token.get(token_id)
            if not candidates:
                continue
            if start > 0 and piece_starts_word[token_id] <= 0 and piece_has_alnum[int(row[start - 1])] != 0:
                continue
            for token_ids, code in candidates:
                length = int(token_ids.shape[0])
                end = start + length
                if end > seq_len:
                    continue
                if end < seq_len:
                    next_id = int(row[end])
                    if piece_starts_word[next_id] <= 0 and piece_has_alnum[next_id] != 0:
                        continue
                if length > 1 and not np.array_equal(row[start:end], token_ids):
                    continue
                row_codes[end - 1] = np.int32(code)
    return family_codes
