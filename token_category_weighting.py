from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


URL_HINTS = (
    "http",
    "www.",
    "://",
    ".com",
    ".org",
    ".net",
    ".edu",
    ".gov",
    "@",
    "/",
)

STRUCTURAL_PIECES = {
    "",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
}

ALPHA_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
ALNUM_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class TokenCategoryWeightingConfig:
    enabled: bool = False
    url_like_weight: float = 0.2
    identifier_like_weight: float = 0.4
    repeat_content_weight: float = 2.0


@dataclass(frozen=True)
class TokenCategoryLuts:
    url_like: np.ndarray
    identifier_like: np.ndarray
    repeat_candidate: np.ndarray


def normalize_piece(piece: str) -> str:
    text = piece.replace("▁", " ").replace("Ġ", " ").strip()
    return re.sub(r"\s+", " ", text).strip()


def classify_piece(piece: str) -> tuple[bool, bool, bool]:
    text = normalize_piece(piece)
    lower = text.casefold()
    compact = lower.replace(" ", "")
    has_alpha = bool(ALPHA_RE.search(text))
    has_digit = bool(DIGIT_RE.search(text))
    alnum_chars = "".join(ch for ch in compact if ALNUM_RE.match(ch))
    url_like = any(hint in compact for hint in URL_HINTS) or compact.startswith("//")
    identifier_like = False
    if has_digit:
        digit_count = sum(ch.isdigit() for ch in compact)
        identifier_like = (
            (has_alpha and len(alnum_chars) >= 4)
            or digit_count >= 4
            or any(ch in compact for ch in "-_./")
        )
    repeat_candidate = (
        (
            has_alpha
            and len(re.sub(r"[^a-z]", "", lower)) >= 3
            and compact not in STRUCTURAL_PIECES
        )
        or identifier_like
    ) and not url_like
    return url_like, identifier_like, repeat_candidate


def build_token_category_luts(sp) -> TokenCategoryLuts:
    vocab_size = int(sp.vocab_size())
    url_like = np.zeros((vocab_size,), dtype=np.bool_)
    identifier_like = np.zeros((vocab_size,), dtype=np.bool_)
    repeat_candidate = np.zeros((vocab_size,), dtype=np.bool_)
    for token_id in range(vocab_size):
        piece = str(sp.id_to_piece(int(token_id)))
        url_flag, identifier_flag, repeat_flag = classify_piece(piece)
        url_like[token_id] = url_flag
        identifier_like[token_id] = identifier_flag
        repeat_candidate[token_id] = repeat_flag
    return TokenCategoryLuts(
        url_like=url_like,
        identifier_like=identifier_like,
        repeat_candidate=repeat_candidate,
    )


def compute_token_category_weights(
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    luts: TokenCategoryLuts,
    config: TokenCategoryWeightingConfig,
) -> np.ndarray:
    del input_ids
    weights = np.ones_like(target_ids, dtype=np.float32)
    if not config.enabled:
        return weights
    url_mask = luts.url_like[target_ids]
    identifier_mask = luts.identifier_like[target_ids]
    if config.url_like_weight != 1.0:
        weights[url_mask] = np.minimum(weights[url_mask], np.float32(config.url_like_weight))
    if config.identifier_like_weight != 1.0:
        weights[identifier_mask] = np.minimum(weights[identifier_mask], np.float32(config.identifier_like_weight))
    if config.repeat_content_weight != 1.0:
        repeat_candidate = luts.repeat_candidate[target_ids] & ~url_mask & ~identifier_mask
        for row_idx in range(target_ids.shape[0]):
            seen: set[int] = set()
            for col_idx in range(target_ids.shape[1]):
                token_id = int(target_ids[row_idx, col_idx])
                if repeat_candidate[row_idx, col_idx] and token_id in seen:
                    weights[row_idx, col_idx] = max(float(weights[row_idx, col_idx]), float(config.repeat_content_weight))
                seen.add(token_id)
    return weights
