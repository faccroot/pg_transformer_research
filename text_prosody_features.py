from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOKEN_CLASS_NAMES = (
    "content",
    "punctuation",
    "whitespace",
    "markup",
    "quote",
)
TOKEN_CLASS_TO_ID = {name: idx for idx, name in enumerate(TOKEN_CLASS_NAMES)}

BOUNDARY_STRENGTH_NAMES = (
    "none",
    "clause",
    "sentence",
    "paragraph",
    "section",
)
BOUNDARY_STRENGTH_TO_ID = {name: idx for idx, name in enumerate(BOUNDARY_STRENGTH_NAMES)}
PUNCTUATION_ROLE_NAMES = (
    "none",
    "comma_clause",
    "colon_semicolon",
    "dash_interrupt",
    "ellipsis",
    "sentence_terminal",
    "question_terminal",
    "exclamation_terminal",
    "bracket",
    "quote_mark",
    "slash_pipe",
    "markup_delim",
    "other",
)
PUNCTUATION_ROLE_TO_ID = {name: idx for idx, name in enumerate(PUNCTUATION_ROLE_NAMES)}
BASE_PROSODY_BINARY_FEATURE_NAMES = (
    "punctuation_like",
    "whitespace_like",
    "quote_like",
    "markup_like",
    "url_like",
    "emoji_like",
    "boundary_clause_plus",
    "boundary_sentence_plus",
    "boundary_paragraph_plus",
    "boundary_section",
)
PROSODY_BINARY_FEATURE_NAMES = (
    *BASE_PROSODY_BINARY_FEATURE_NAMES[:6],
    "interruption_like",
    "ellipsis_like",
    "list_like",
    "heading_like",
    "code_like",
    "emphasis_like",
    *BASE_PROSODY_BINARY_FEATURE_NAMES[6:],
)
PROSODY_BINARY_FEATURE_TO_ID = {name: idx for idx, name in enumerate(PROSODY_BINARY_FEATURE_NAMES)}
BASE_PROSODY_BINARY_FEATURE_TO_ID = {name: idx for idx, name in enumerate(BASE_PROSODY_BINARY_FEATURE_NAMES)}

_DOUBLE_QUOTE_CHARS = {'"', "“", "”", "«", "»", "„", "‟", "❝", "❞"}
_CLAUSE_PUNCT = {",", ";", ":", "—", "–", "-", "…", "...", "(", ")", "[", "]"}
_SENTENCE_PUNCT = {".", "!", "?", "。", "！", "？"}
_MARKUP_DELIMS = {"*", "_", "`", "#", "~", "|"}
_URL_PREFIXES = ("http://", "https://", "www.")
_DASH_CHARS = {"—", "–", "-", "--"}
_ELLIPSIS_CHARS = {"…", "..."}
_BRACKET_CHARS = {"(", ")", "[", "]", "{", "}"}
_LIST_BULLETS = {"-", "*", "+", "•", "·"}
_EMOJI_CODEPOINT_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
)


def piece_to_text(piece: str) -> str:
    return str(piece).replace("▁", " ")


def _core_text(piece: str) -> str:
    return piece_to_text(piece).strip(" ")


def is_markup_like(piece: str) -> bool:
    text = _core_text(piece)
    if not text:
        return False
    if text.startswith("<") and text.endswith(">"):
        return True
    if text.startswith("</") or text.endswith("/>"):
        return True
    if text.startswith("http://") or text.startswith("https://") or text.startswith("www."):
        return True
    if text.startswith("```") or text in {"```", "~~~", "---", "***"}:
        return True
    if text.startswith("#") and len(text) <= 6 and set(text) == {"#"}:
        return True
    return False


def is_url_like_piece(piece: str) -> bool:
    text = _core_text(piece)
    if not text:
        return False
    return text.startswith(_URL_PREFIXES)


def is_emoji_like_text(text: str) -> bool:
    core = str(text or "").strip()
    if not core:
        return False
    saw_symbol = False
    for ch in core:
        if ch.isspace():
            continue
        codepoint = ord(ch)
        if ch in {"\ufe0f", "\u200d"}:
            continue
        if any(lo <= codepoint <= hi for lo, hi in _EMOJI_CODEPOINT_RANGES):
            saw_symbol = True
            continue
        return False
    return saw_symbol


def is_emoji_like_piece(piece: str) -> bool:
    return is_emoji_like_text(_core_text(piece))


def is_markup_delimiter_piece(piece: str) -> bool:
    text = _core_text(piece)
    if not text:
        return False
    return all(ch in _MARKUP_DELIMS for ch in text)


def is_heading_like_piece(piece: str) -> bool:
    text = piece_to_text(piece).lstrip(" ")
    core = text.strip()
    if not core:
        return False
    if core.startswith("<h") and core.endswith(">"):
        return True
    if core.startswith("#") and set(core.split()[0]) == {"#"}:
        return True
    return False


def is_list_like_piece(piece: str) -> bool:
    text = piece_to_text(piece).lstrip(" ")
    core = text.strip()
    if not core:
        return False
    if core in _LIST_BULLETS:
        return True
    if any(core.startswith(f"{mark} ") for mark in _LIST_BULLETS):
        return True
    if len(core) >= 2 and core[0].isdigit() and core[1] in {".", ")"}:
        return True
    return False


def is_code_like_piece(piece: str) -> bool:
    core = _core_text(piece)
    if not core:
        return False
    if "`" in core:
        return True
    if core.startswith(("```", "~~~", "<code", "</code", "def ", "class ")):
        return True
    return False


def is_emphasis_like_piece(piece: str) -> bool:
    core = _core_text(piece)
    if not core:
        return False
    if is_markup_delimiter_piece(piece) and any(ch in {"*", "_", "~"} for ch in core):
        return True
    if len(core) >= 2 and core.isalpha() and core.upper() == core and core.lower() != core:
        return True
    return False


def is_ellipsis_like_piece(piece: str) -> bool:
    core = _core_text(piece)
    return bool(core) and any(mark in core for mark in _ELLIPSIS_CHARS)


def is_interruption_like_piece(piece: str) -> bool:
    core = _core_text(piece)
    if not core:
        return False
    if any(mark in core for mark in _DASH_CHARS):
        return True
    if any(mark in core for mark in _BRACKET_CHARS):
        return True
    return False


def is_quote_like(piece: str) -> bool:
    text = _core_text(piece)
    if not text:
        return False
    if all(ch in _DOUBLE_QUOTE_CHARS or ch == "`" for ch in text):
        return True
    return text in {"''", "``"}


def classify_piece(piece: str) -> str:
    raw = piece_to_text(piece)
    core = raw.strip(" ")
    if not core:
        return "whitespace"
    if is_markup_like(piece):
        return "markup"
    if is_quote_like(piece):
        return "quote"
    if not any(ch.isalnum() for ch in core):
        if all(ch.isspace() for ch in core):
            return "whitespace"
        return "punctuation"
    return "content"


def punctuation_role_for_piece(piece: str) -> int:
    token_class = classify_piece(piece)
    if token_class in {"content", "whitespace"}:
        return PUNCTUATION_ROLE_TO_ID["none"]
    if is_quote_like(piece):
        return PUNCTUATION_ROLE_TO_ID["quote_mark"]
    if is_markup_delimiter_piece(piece):
        return PUNCTUATION_ROLE_TO_ID["markup_delim"]
    core = _core_text(piece)
    if not core:
        return PUNCTUATION_ROLE_TO_ID["none"]
    if is_ellipsis_like_piece(piece):
        return PUNCTUATION_ROLE_TO_ID["ellipsis"]
    if any(ch in core for ch in {"?", "？"}):
        return PUNCTUATION_ROLE_TO_ID["question_terminal"]
    if any(ch in core for ch in {"!", "！"}):
        return PUNCTUATION_ROLE_TO_ID["exclamation_terminal"]
    if any(ch in core for ch in {".", "。"}):
        return PUNCTUATION_ROLE_TO_ID["sentence_terminal"]
    if "," in core:
        return PUNCTUATION_ROLE_TO_ID["comma_clause"]
    if ";" in core or ":" in core:
        return PUNCTUATION_ROLE_TO_ID["colon_semicolon"]
    if any(mark in core for mark in _DASH_CHARS):
        return PUNCTUATION_ROLE_TO_ID["dash_interrupt"]
    if any(mark in core for mark in _BRACKET_CHARS):
        return PUNCTUATION_ROLE_TO_ID["bracket"]
    if any(mark in core for mark in {"/", "\\", "|"}):
        return PUNCTUATION_ROLE_TO_ID["slash_pipe"]
    if token_class in {"punctuation", "quote", "markup"}:
        return PUNCTUATION_ROLE_TO_ID["other"]
    return PUNCTUATION_ROLE_TO_ID["none"]


def boundary_strength_for_piece(piece: str) -> int:
    text = piece_to_text(piece)
    core = text.strip(" ")
    newline_count = text.count("\n")
    if newline_count >= 3:
        return BOUNDARY_STRENGTH_TO_ID["section"]
    if newline_count >= 2:
        return BOUNDARY_STRENGTH_TO_ID["paragraph"]
    if newline_count == 1:
        return BOUNDARY_STRENGTH_TO_ID["sentence"]
    if any(mark in core for mark in _SENTENCE_PUNCT):
        return BOUNDARY_STRENGTH_TO_ID["sentence"]
    if core in _CLAUSE_PUNCT or any(mark in core for mark in _CLAUSE_PUNCT):
        return BOUNDARY_STRENGTH_TO_ID["clause"]
    return BOUNDARY_STRENGTH_TO_ID["none"]


def classify_pieces(pieces: list[str]) -> np.ndarray:
    return np.asarray([TOKEN_CLASS_TO_ID[classify_piece(piece)] for piece in pieces], dtype=np.int32)


def punctuation_roles_for_pieces(pieces: list[str]) -> np.ndarray:
    return np.asarray([punctuation_role_for_piece(piece) for piece in pieces], dtype=np.int32)


def boundary_strengths_for_pieces(pieces: list[str]) -> np.ndarray:
    return np.asarray([boundary_strength_for_piece(piece) for piece in pieces], dtype=np.int32)


def quote_state_for_pieces(pieces: list[str]) -> np.ndarray:
    state = np.zeros((len(pieces),), dtype=np.int32)
    inside = 0
    for idx, piece in enumerate(pieces):
        state[idx] = inside
        text = piece_to_text(piece)
        toggles = sum(text.count(ch) for ch in _DOUBLE_QUOTE_CHARS)
        toggles += text.count("``")
        toggles += text.count("''")
        if toggles % 2 == 1:
            inside = 1 - inside
    return state


def distance_to_next(mask: np.ndarray, *, default_distance: int | None = None) -> np.ndarray:
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    n = int(keep.shape[0])
    fill_value = int(default_distance if default_distance is not None else max(n, 1))
    out = np.full((n,), fill_value, dtype=np.int32)
    next_pos: int | None = None
    for idx in range(n - 1, -1, -1):
        if keep[idx]:
            next_pos = idx
            out[idx] = 0
        elif next_pos is not None:
            out[idx] = int(next_pos - idx)
    return out


def rolling_mean(values: np.ndarray, *, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if window <= 1 or arr.size <= 0:
        return arr.astype(np.float32, copy=True)
    csum = np.cumsum(arr, dtype=np.float32)
    out = np.empty_like(arr)
    for idx in range(arr.shape[0]):
        start = max(0, idx - window + 1)
        total = csum[idx] - (csum[start - 1] if start > 0 else 0.0)
        out[idx] = total / float(idx - start + 1)
    return out


def bucketize_distances(distances: np.ndarray, edges: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(distances, dtype=np.int32).reshape(-1)
    return np.asarray(np.digitize(arr, np.asarray(edges, dtype=np.int32), right=True), dtype=np.int32)


@dataclass(frozen=True)
class TokenProsodyLuts:
    token_class_ids: np.ndarray
    boundary_strength_ids: np.ndarray
    punctuation_role_ids: np.ndarray
    quote_like_ids: np.ndarray
    markup_like_ids: np.ndarray
    url_like_ids: np.ndarray
    emoji_like_ids: np.ndarray
    binary_feature_ids: np.ndarray
    binary_feature_names: tuple[str, ...]
    reset_prior_values: np.ndarray


def build_binary_feature_stack(
    token_class_ids: np.ndarray,
    boundary_strength_ids: np.ndarray,
    quote_like_ids: np.ndarray,
    markup_like_ids: np.ndarray,
    url_like_ids: np.ndarray,
    emoji_like_ids: np.ndarray,
    *,
    feature_names: tuple[str, ...] = PROSODY_BINARY_FEATURE_NAMES,
    punctuation_role_ids: np.ndarray | None = None,
    list_like_ids: np.ndarray | None = None,
    heading_like_ids: np.ndarray | None = None,
    code_like_ids: np.ndarray | None = None,
    emphasis_like_ids: np.ndarray | None = None,
) -> np.ndarray:
    token_class_ids = np.asarray(token_class_ids, dtype=np.int32).reshape(-1)
    boundary_strength_ids = np.asarray(boundary_strength_ids, dtype=np.int32).reshape(-1)
    quote_like_ids = np.asarray(quote_like_ids, dtype=np.int32).reshape(-1)
    markup_like_ids = np.asarray(markup_like_ids, dtype=np.int32).reshape(-1)
    url_like_ids = np.asarray(url_like_ids, dtype=np.int32).reshape(-1)
    emoji_like_ids = np.asarray(emoji_like_ids, dtype=np.int32).reshape(-1)
    punctuation_role_ids = (
        np.zeros_like(token_class_ids)
        if punctuation_role_ids is None
        else np.asarray(punctuation_role_ids, dtype=np.int32).reshape(-1)
    )
    list_like_ids = (
        np.zeros_like(token_class_ids)
        if list_like_ids is None
        else np.asarray(list_like_ids, dtype=np.int32).reshape(-1)
    )
    heading_like_ids = (
        np.zeros_like(token_class_ids)
        if heading_like_ids is None
        else np.asarray(heading_like_ids, dtype=np.int32).reshape(-1)
    )
    code_like_ids = (
        np.zeros_like(token_class_ids)
        if code_like_ids is None
        else np.asarray(code_like_ids, dtype=np.int32).reshape(-1)
    )
    emphasis_like_ids = (
        np.zeros_like(token_class_ids)
        if emphasis_like_ids is None
        else np.asarray(emphasis_like_ids, dtype=np.int32).reshape(-1)
    )
    features = np.zeros((token_class_ids.shape[0], len(feature_names)), dtype=np.int32)
    feature_values = {
        "punctuation_like": (token_class_ids == TOKEN_CLASS_TO_ID["punctuation"]).astype(np.int32),
        "whitespace_like": (token_class_ids == TOKEN_CLASS_TO_ID["whitespace"]).astype(np.int32),
        "quote_like": quote_like_ids,
        "markup_like": markup_like_ids,
        "url_like": url_like_ids,
        "emoji_like": emoji_like_ids,
        "interruption_like": np.isin(
            punctuation_role_ids,
            np.asarray(
                [
                    PUNCTUATION_ROLE_TO_ID["dash_interrupt"],
                    PUNCTUATION_ROLE_TO_ID["bracket"],
                ],
                dtype=np.int32,
            ),
        ).astype(np.int32),
        "ellipsis_like": (punctuation_role_ids == PUNCTUATION_ROLE_TO_ID["ellipsis"]).astype(np.int32),
        "list_like": list_like_ids,
        "heading_like": heading_like_ids,
        "code_like": code_like_ids,
        "emphasis_like": emphasis_like_ids,
        "boundary_clause_plus": (boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["clause"]).astype(np.int32),
        "boundary_sentence_plus": (boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["sentence"]).astype(np.int32),
        "boundary_paragraph_plus": (boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["paragraph"]).astype(np.int32),
        "boundary_section": (boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["section"]).astype(np.int32),
    }
    for idx, name in enumerate(feature_names):
        features[:, idx] = feature_values[name]
    return features


def build_reset_prior_values(
    binary_feature_ids: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    feats = np.asarray(binary_feature_ids, dtype=np.float32)
    names = tuple(feature_names or (BASE_PROSODY_BINARY_FEATURE_NAMES if feats.shape[1] == len(BASE_PROSODY_BINARY_FEATURE_NAMES) else PROSODY_BINARY_FEATURE_NAMES))
    if feats.ndim != 2 or feats.shape[1] != len(names):
        raise ValueError(
            f"binary_feature_ids must have shape [N, {len(names)}], got {feats.shape}"
        )
    weight_lookup = {
        "punctuation_like": 0.08,
        "whitespace_like": 0.18,
        "quote_like": 0.18,
        "markup_like": 0.10,
        "url_like": 0.04,
        "emoji_like": 0.08,
        "interruption_like": 0.12,
        "ellipsis_like": 0.16,
        "list_like": 0.22,
        "heading_like": 0.28,
        "code_like": 0.18,
        "emphasis_like": 0.10,
        "boundary_clause_plus": 0.12,
        "boundary_sentence_plus": 0.28,
        "boundary_paragraph_plus": 0.55,
        "boundary_section": 0.85,
    }
    weights = np.asarray([weight_lookup[name] for name in names], dtype=np.float32)
    return np.clip(feats @ weights, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class TextProsodyFeatures:
    pieces: list[str]
    token_class_ids: np.ndarray
    boundary_strength_ids: np.ndarray
    prev_boundary_strength_ids: np.ndarray
    punctuation_role_ids: np.ndarray
    quote_state: np.ndarray
    sentence_distance: np.ndarray
    paragraph_distance: np.ndarray
    section_distance: np.ndarray
    recent_punctuation_density: np.ndarray
    recent_noncontent_density: np.ndarray


def build_token_prosody_luts(sp, *, extended_binary_features: bool = False) -> TokenProsodyLuts:
    vocab_size = int(sp.get_piece_size())
    pieces = [sp.id_to_piece(idx) for idx in range(vocab_size)]
    token_class_ids = classify_pieces(pieces)
    boundary_strength_ids = boundary_strengths_for_pieces(pieces)
    punctuation_role_ids = punctuation_roles_for_pieces(pieces)
    quote_like_ids = np.asarray([int(is_quote_like(piece)) for piece in pieces], dtype=np.int32)
    markup_like_ids = np.asarray([int(is_markup_like(piece)) for piece in pieces], dtype=np.int32)
    url_like_ids = np.asarray([int(is_url_like_piece(piece)) for piece in pieces], dtype=np.int32)
    emoji_like_ids = np.asarray([int(is_emoji_like_piece(piece)) for piece in pieces], dtype=np.int32)
    list_like_ids = np.asarray([int(is_list_like_piece(piece)) for piece in pieces], dtype=np.int32)
    heading_like_ids = np.asarray([int(is_heading_like_piece(piece)) for piece in pieces], dtype=np.int32)
    code_like_ids = np.asarray([int(is_code_like_piece(piece)) for piece in pieces], dtype=np.int32)
    emphasis_like_ids = np.asarray([int(is_emphasis_like_piece(piece)) for piece in pieces], dtype=np.int32)
    binary_feature_names = PROSODY_BINARY_FEATURE_NAMES if extended_binary_features else BASE_PROSODY_BINARY_FEATURE_NAMES
    binary_feature_ids = build_binary_feature_stack(
        token_class_ids,
        boundary_strength_ids,
        quote_like_ids,
        markup_like_ids,
        url_like_ids,
        emoji_like_ids,
        feature_names=binary_feature_names,
        punctuation_role_ids=punctuation_role_ids,
        list_like_ids=list_like_ids,
        heading_like_ids=heading_like_ids,
        code_like_ids=code_like_ids,
        emphasis_like_ids=emphasis_like_ids,
    )
    return TokenProsodyLuts(
        token_class_ids=token_class_ids,
        boundary_strength_ids=boundary_strength_ids,
        punctuation_role_ids=punctuation_role_ids,
        quote_like_ids=quote_like_ids,
        markup_like_ids=markup_like_ids,
        url_like_ids=url_like_ids,
        emoji_like_ids=emoji_like_ids,
        binary_feature_ids=binary_feature_ids,
        binary_feature_names=binary_feature_names,
        reset_prior_values=build_reset_prior_values(binary_feature_ids, feature_names=binary_feature_names),
    )


def extract_text_prosody_features_from_pieces(
    pieces: list[str],
    *,
    density_window: int = 16,
) -> TextProsodyFeatures:
    token_class_ids = classify_pieces(pieces)
    boundary_strength_ids = boundary_strengths_for_pieces(pieces)
    punctuation_role_ids = punctuation_roles_for_pieces(pieces)
    prev_boundary_strength_ids = np.zeros_like(boundary_strength_ids)
    if boundary_strength_ids.shape[0] > 1:
        prev_boundary_strength_ids[1:] = boundary_strength_ids[:-1]
    quote_state = quote_state_for_pieces(pieces)
    sentence_distance = distance_to_next(boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["sentence"])
    paragraph_distance = distance_to_next(boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["paragraph"])
    section_distance = distance_to_next(boundary_strength_ids >= BOUNDARY_STRENGTH_TO_ID["section"])
    punctuation_mask = token_class_ids == TOKEN_CLASS_TO_ID["punctuation"]
    noncontent_mask = token_class_ids != TOKEN_CLASS_TO_ID["content"]
    recent_punctuation_density = rolling_mean(punctuation_mask.astype(np.float32), window=density_window)
    recent_noncontent_density = rolling_mean(noncontent_mask.astype(np.float32), window=density_window)
    return TextProsodyFeatures(
        pieces=list(pieces),
        token_class_ids=token_class_ids,
        boundary_strength_ids=boundary_strength_ids,
        prev_boundary_strength_ids=prev_boundary_strength_ids,
        punctuation_role_ids=punctuation_role_ids,
        quote_state=quote_state,
        sentence_distance=sentence_distance,
        paragraph_distance=paragraph_distance,
        section_distance=section_distance,
        recent_punctuation_density=recent_punctuation_density,
        recent_noncontent_density=recent_noncontent_density,
    )


def extract_text_prosody_features(sp, token_ids: np.ndarray, *, density_window: int = 16) -> TextProsodyFeatures:
    ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    pieces = [sp.id_to_piece(int(tok)) for tok in ids.tolist()]
    return extract_text_prosody_features_from_pieces(pieces, density_window=density_window)
