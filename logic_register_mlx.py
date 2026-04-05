from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn


NOT_OPERATOR_SURFACES = frozenset(
    {
        "not",
        "n't",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "except",
        "unless",
        "lack",
        "absence",
        "fail",
        "cannot",
        "can't",
        "won't",
        "don't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
    }
)
AND_OPERATOR_SURFACES = frozenset(
    {
        "and",
        "also",
        "moreover",
        "additionally",
        "furthermore",
        "both",
        "plus",
        "together",
    }
)
OR_OPERATOR_SURFACES = frozenset(
    {
        "or",
        "either",
        "alternatively",
        "otherwise",
        "else",
    }
)


@dataclass(frozen=True)
class OperatorPatternBank:
    token_ids: np.ndarray
    codes: np.ndarray


@dataclass(frozen=True)
class OperatorRoutingSpec:
    lookup: np.ndarray
    piece_starts_word: np.ndarray
    piece_has_alnum: np.ndarray
    pattern_banks: tuple[OperatorPatternBank, ...]
    patterns_by_first_token: dict[int, tuple[tuple[np.ndarray, int], ...]]


def _normalize_piece(piece: str) -> str:
    return piece.lower().replace("’", "'").lstrip("▁").strip()


def operator_code_for_piece(piece: str) -> int:
    normalized = _normalize_piece(piece)
    if not normalized:
        return 0
    if normalized in NOT_OPERATOR_SURFACES or normalized.endswith("n't"):
        return 1
    if normalized in AND_OPERATOR_SURFACES:
        return 2
    if normalized in OR_OPERATOR_SURFACES:
        return 3
    return 0


def build_operator_lookup(sp: spm.SentencePieceProcessor, vocab_size: int) -> np.ndarray:
    table = np.zeros((vocab_size,), dtype=np.int32)
    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        if not piece.startswith("▁"):
            continue
        table[token_id] = operator_code_for_piece(piece)
    return table


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


def build_operator_routing_spec(sp: spm.SentencePieceProcessor, vocab_size: int) -> OperatorRoutingSpec:
    lookup = build_operator_lookup(sp, vocab_size)
    piece_starts_word = np.zeros((vocab_size,), dtype=np.int32)
    piece_has_alnum = np.zeros((vocab_size,), dtype=np.int32)

    for token_id in range(min(vocab_size, int(sp.vocab_size()))):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        piece_starts_word[token_id] = int(piece.startswith("▁"))
        piece_has_alnum[token_id] = int(any(ch.isalnum() for ch in _normalize_piece(piece)))

    pattern_groups: dict[int, dict[tuple[int, ...], int]] = {}
    for code, surfaces in (
        (1, NOT_OPERATOR_SURFACES),
        (2, AND_OPERATOR_SURFACES),
        (3, OR_OPERATOR_SURFACES),
    ):
        for surface in sorted(surfaces):
            for pattern in _surface_pattern_sequences(sp, surface):
                existing = pattern_groups.setdefault(len(pattern), {}).get(pattern)
                if existing is not None and existing != code:
                    raise ValueError(f"operator pattern collision for ids={pattern}: {existing} vs {code}")
                pattern_groups.setdefault(len(pattern), {})[pattern] = code

    pattern_banks: list[OperatorPatternBank] = []
    for _, mapping in sorted(pattern_groups.items()):
        token_ids = np.array(list(mapping.keys()), dtype=np.int32)
        codes = np.array(list(mapping.values()), dtype=np.int32)
        pattern_banks.append(OperatorPatternBank(token_ids=token_ids, codes=codes))

    patterns_by_first_token: dict[int, list[tuple[np.ndarray, int]]] = {}
    for bank in pattern_banks:
        for token_ids, code in zip(bank.token_ids, bank.codes, strict=False):
            patterns_by_first_token.setdefault(int(token_ids[0]), []).append(
                (np.ascontiguousarray(token_ids, dtype=np.int32), int(code))
            )
    patterns_by_first_token_final = {
        first_token: tuple(sorted(candidates, key=lambda item: (int(item[0].shape[0]), tuple(int(v) for v in item[0]))))
        for first_token, candidates in patterns_by_first_token.items()
    }

    return OperatorRoutingSpec(
        lookup=lookup,
        piece_starts_word=piece_starts_word,
        piece_has_alnum=piece_has_alnum,
        pattern_banks=tuple(pattern_banks),
        patterns_by_first_token=patterns_by_first_token_final,
    )


def detect_operator_codes_np(input_ids: np.ndarray, routing: OperatorRoutingSpec) -> np.ndarray:
    input_ids = np.ascontiguousarray(input_ids, dtype=np.int32)
    operator_codes = np.take(routing.lookup, input_ids, axis=0)
    if not routing.patterns_by_first_token:
        return operator_codes

    batch_size, seq_len = input_ids.shape
    piece_starts_word = routing.piece_starts_word
    piece_has_alnum = routing.piece_has_alnum
    for batch_idx in range(batch_size):
        row = input_ids[batch_idx]
        row_codes = operator_codes[batch_idx]
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
    return operator_codes


def route_operator_codes(operator_codes: np.ndarray | mx.array) -> np.ndarray | mx.array:
    if operator_codes.shape[1] <= 0:
        return operator_codes
    if isinstance(operator_codes, np.ndarray):
        routed = np.zeros_like(operator_codes)
        routed[:, 1:] = operator_codes[:, :-1]
        return routed
    zero = mx.zeros((operator_codes.shape[0], 1), dtype=operator_codes.dtype)
    return mx.concatenate([zero, operator_codes[:, :-1]], axis=1)


def pad_operator_codes(operator_codes: np.ndarray | mx.array, num_registers: int) -> np.ndarray | mx.array:
    if num_registers <= 0:
        return operator_codes
    if isinstance(operator_codes, np.ndarray):
        pad = np.zeros((operator_codes.shape[0], num_registers), dtype=operator_codes.dtype)
        return np.concatenate([pad, operator_codes], axis=1)
    pad = mx.zeros((operator_codes.shape[0], num_registers), dtype=operator_codes.dtype)
    return mx.concatenate([pad, operator_codes], axis=1)


def interleaved_register_block_count(token_len: int, register_stride: int) -> int:
    if token_len <= 0:
        return 0
    if register_stride <= 0:
        raise ValueError(f"register_stride must be positive, got {register_stride}")
    return max(0, (token_len - 1) // register_stride)


def total_sequence_len_with_registers(
    token_len: int,
    num_registers: int,
    *,
    layout: str = "prefix",
    register_stride: int = 256,
) -> int:
    if num_registers <= 0:
        return token_len
    if layout == "prefix":
        return token_len + num_registers
    if layout == "interleaved":
        return token_len + interleaved_register_block_count(token_len, register_stride) * num_registers
    raise ValueError(f"Unsupported register layout {layout!r}")


def register_position_mask(
    total_len: int,
    num_registers: int,
    *,
    layout: str = "prefix",
    register_stride: int = 256,
) -> np.ndarray:
    mask = np.zeros((total_len,), dtype=np.bool_)
    if num_registers <= 0:
        return mask
    if layout == "prefix":
        mask[:num_registers] = True
        return mask
    if layout != "interleaved":
        raise ValueError(f"Unsupported register layout {layout!r}")
    pos = 0
    while pos < total_len:
        take = min(register_stride, total_len - pos)
        pos += take
        remaining = total_len - pos
        if remaining <= 0:
            break
        skip = min(num_registers, remaining)
        mask[pos : pos + skip] = True
        pos += skip
    return mask


def _strip_interleaved_axis1(x: np.ndarray | mx.array, num_registers: int, register_stride: int) -> np.ndarray | mx.array:
    parts: list[np.ndarray | mx.array] = []
    pos = 0
    seq_len = int(x.shape[1])
    while pos < seq_len:
        take = min(register_stride, seq_len - pos)
        if take > 0:
            parts.append(x[:, pos : pos + take, ...])
        pos += take
        remaining = seq_len - pos
        if remaining <= 0:
            break
        pos += min(num_registers, remaining)
    if not parts:
        return x[:, :0, ...]
    if len(parts) == 1:
        return parts[0]
    if isinstance(x, np.ndarray):
        return np.concatenate(parts, axis=1)
    return mx.concatenate(parts, axis=1)


def strip_register_positions(
    x: np.ndarray | mx.array,
    num_registers: int,
    *,
    layout: str = "prefix",
    register_stride: int = 256,
) -> np.ndarray | mx.array:
    if num_registers <= 0:
        return x
    if layout == "prefix":
        return x[:, num_registers:, ...]
    if layout == "interleaved":
        return _strip_interleaved_axis1(x, num_registers, register_stride)
    raise ValueError(f"Unsupported register layout {layout!r}")


def interleave_operator_codes(
    operator_codes: np.ndarray,
    num_registers: int,
    *,
    layout: str = "prefix",
    register_stride: int = 256,
) -> np.ndarray:
    if num_registers <= 0:
        return operator_codes
    if layout == "prefix":
        return pad_operator_codes(operator_codes, num_registers)
    if layout != "interleaved":
        raise ValueError(f"Unsupported register layout {layout!r}")
    batch, token_len = operator_codes.shape
    zero_block = np.zeros((batch, num_registers), dtype=operator_codes.dtype)
    parts: list[np.ndarray] = []
    for start in range(0, token_len, register_stride):
        end = min(start + register_stride, token_len)
        parts.append(operator_codes[:, start:end])
        if end < token_len:
            parts.append(zero_block)
    return np.concatenate(parts, axis=1) if parts else operator_codes[:, :0]


def build_register_attention_mask(total_len: int, num_registers: int) -> mx.array:
    return build_register_attention_mask_with_mode(total_len, num_registers, mode="bidirectional")


def build_register_attention_mask_with_mode(total_len: int, num_registers: int, mode: str = "bidirectional") -> mx.array:
    if mode not in {"bidirectional", "causal"}:
        raise ValueError(f"Unsupported register mask mode {mode!r}")
    allowed = np.tri(total_len, total_len, dtype=np.bool_)
    if num_registers > 0 and mode == "bidirectional":
        allowed[:num_registers, :] = True
    additive = np.where(allowed, 0.0, -1e9).astype(np.float32, copy=False)
    return mx.array(additive[None, None, :, :], dtype=mx.bfloat16)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class PolarityDetector(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int = 0, init_bias: float = -4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = CastedLinear(model_dim, hidden_dim) if hidden_dim > 0 else None
        self.out = CastedLinear(hidden_dim if hidden_dim > 0 else model_dim, 1)
        self.bias = mx.full((1,), init_bias, dtype=mx.float32)

    def __call__(self, h: mx.array) -> mx.array:
        x = nn.silu(self.fc(h)) if self.fc is not None else h
        logits = self.out(x).astype(mx.float32)
        return logits[..., 0] + self.bias.astype(mx.float32)


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, model_dim: int, init_std: float = 0.02):
        super().__init__()
        self.num_registers = num_registers
        self.registers = (mx.random.normal((num_registers, model_dim), dtype=mx.float32) * init_std).astype(mx.float32)

    def _expanded_regs(self, x: mx.array) -> mx.array:
        return mx.broadcast_to(self.registers.astype(x.dtype)[None, :, :], (x.shape[0], self.num_registers, x.shape[2]))

    def prepend(self, x: mx.array) -> mx.array:
        regs = self._expanded_regs(x)
        return mx.concatenate([regs, x], axis=1)

    def interleave(self, x: mx.array, register_stride: int) -> mx.array:
        parts: list[mx.array] = []
        regs = self._expanded_regs(x)
        token_len = int(x.shape[1])
        for start in range(0, token_len, register_stride):
            end = min(start + register_stride, token_len)
            parts.append(x[:, start:end, :])
            if end < token_len:
                parts.append(regs)
        return mx.concatenate(parts, axis=1) if parts else x[:, :0, :]

    def inject(self, x: mx.array, *, layout: str = "prefix", register_stride: int = 256) -> mx.array:
        if layout == "prefix":
            return self.prepend(x)
        if layout == "interleaved":
            return self.interleave(x, register_stride)
        raise ValueError(f"Unsupported register layout {layout!r}")

    def strip(self, x: mx.array, *, layout: str = "prefix", register_stride: int = 256) -> mx.array:
        stripped = strip_register_positions(x, self.num_registers, layout=layout, register_stride=register_stride)
        if isinstance(stripped, np.ndarray):
            raise TypeError("RegisterTokens.strip expects an mlx array")
        return stripped


class LogicSideCar(nn.Module):
    def __init__(self, model_dim: int, logic_dim: int, operator_mode: str = "all"):
        super().__init__()
        if operator_mode not in {"all", "not_only"}:
            raise ValueError(f"Unsupported logic operator mode {operator_mode!r}")
        self.logic_dim = logic_dim
        self.operator_mode = operator_mode
        self.proj_in = CastedLinear(model_dim, logic_dim)
        self.proj_out = CastedLinear(logic_dim, model_dim)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

    def __call__(
        self,
        h: mx.array,
        operator_codes: mx.array | None,
        polarity_scores: mx.array | None = None,
    ) -> mx.array:
        if operator_codes is None and polarity_scores is None:
            return h
        z = self.proj_in(h)
        z_logic = mx.zeros_like(z)
        if polarity_scores is not None:
            z_logic = z_logic + mx.expand_dims(polarity_scores.astype(z.dtype), axis=-1) * (-z)
        elif operator_codes is not None:
            not_mask = mx.expand_dims(operator_codes == 1, axis=-1)
            z_logic = mx.where(not_mask, -z, z_logic)
        if self.operator_mode == "all" and operator_codes is not None:
            zero = mx.zeros_like(z[:, :1, :])
            lhs = mx.concatenate([zero, zero, z[:, :-2, :]], axis=1) if z.shape[1] > 1 else mx.zeros_like(z)
            and_mask = mx.expand_dims(operator_codes == 2, axis=-1)
            or_mask = mx.expand_dims(operator_codes == 3, axis=-1)
            z_logic = mx.where(and_mask, mx.minimum(lhs, z), z_logic)
            z_logic = mx.where(or_mask, mx.maximum(lhs, z), z_logic)
        gate = mx.tanh(self.gate).astype(h.dtype)[None, None, :]
        return h + gate * self.proj_out(z_logic)


class HardmaxStructuralController(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        num_states: int,
        *,
        init_std: float = 0.02,
        temperature: float = 1.0,
        compute_min_scale: float = 0.35,
        compute_power: float = 1.0,
        operator_prior_scale: float = 1.0,
        reset_prior_scale: float = 1.0,
        simvq_enabled: bool = False,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_states <= 0:
            raise ValueError(f"num_states must be positive, got {num_states}")
        self.state_dim = int(state_dim)
        self.num_states = int(num_states)
        self.temperature = max(float(temperature), 1.0e-4)
        self.compute_min_scale = min(max(float(compute_min_scale), 0.0), 1.0)
        self.compute_power = max(float(compute_power), 1.0e-4)
        self.operator_prior_scale = float(operator_prior_scale)
        self.reset_prior_scale = max(float(reset_prior_scale), 0.0)
        self.simvq_enabled = bool(simvq_enabled)

        self.proj_in = CastedLinear(model_dim, state_dim)
        self.recur = CastedLinear(state_dim, state_dim)
        self.state_logits = CastedLinear(state_dim, num_states)
        self.proj_out = CastedLinear(state_dim, model_dim)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

        self.state_book = (mx.random.normal((num_states, state_dim), dtype=mx.float32) * init_std).astype(mx.float32)
        self.state_book_proj = CastedLinear(state_dim, state_dim) if self.simvq_enabled else None
        if self.state_book_proj is not None:
            self.state_book_proj.weight = mx.eye(state_dim, dtype=mx.float32)
        self.operator_state = (mx.random.normal((4, state_dim), dtype=mx.float32) * init_std).astype(mx.float32)
        self.polarity_state = (mx.random.normal((state_dim,), dtype=mx.float32) * init_std).astype(mx.float32)
        self._eye = mx.eye(num_states, dtype=mx.float32)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1.0e-4)

    def effective_state_book(self) -> mx.array:
        book = self.state_book.astype(mx.float32)
        if self.state_book_proj is None:
            return book
        return self.state_book_proj(book.astype(book.dtype)).astype(mx.float32)

    def _hard_assign(self, probs: mx.array) -> tuple[mx.array, mx.array]:
        state_idx = mx.argmax(probs, axis=-1).astype(mx.int32)
        hard = self._eye[state_idx]
        assign = probs + mx.stop_gradient(hard - probs)
        return assign.astype(probs.dtype), state_idx

    def _confidence_and_budget(self, probs: mx.array) -> tuple[mx.array, mx.array]:
        probs_f = probs.astype(mx.float32)
        if self.num_states <= 1:
            confidence = mx.ones(probs_f.shape[:-1], dtype=mx.float32)
        else:
            sorted_probs = mx.sort(probs_f, axis=-1)
            confidence = sorted_probs[..., -1] - sorted_probs[..., -2]
        budget = self.compute_min_scale + (1.0 - self.compute_min_scale) * mx.power(
            mx.maximum(1.0 - confidence, mx.array(0.0, dtype=mx.float32)),
            self.compute_power,
        )
        return confidence, budget

    def regularization_losses(self, aux: dict[str, mx.array] | None) -> tuple[mx.array, mx.array, mx.array]:
        zero = mx.array(0.0, dtype=mx.float32)
        if aux is None:
            return zero, zero, zero
        soft_usage = aux.get("soft_usage")
        if soft_usage is None:
            return zero, zero, zero
        mean_usage = mx.mean(soft_usage.astype(mx.float32), axis=(0, 1))
        target = mx.full(mean_usage.shape, 1.0 / max(self.num_states, 1), dtype=mx.float32)
        usage_balance = mx.mean(mx.square(mean_usage - target))
        entropy = -mx.mean(
            mx.sum(
                soft_usage.astype(mx.float32)
                * mx.log(mx.maximum(soft_usage.astype(mx.float32), mx.array(1.0e-8, dtype=mx.float32))),
                axis=-1,
            )
        )
        book = self.effective_state_book().astype(mx.float32)
        denom = mx.maximum(mx.linalg.norm(book, axis=-1, keepdims=True), mx.array(1.0e-6, dtype=mx.float32))
        norm_book = book / denom
        cosine = norm_book @ norm_book.T
        offdiag = mx.ones((self.num_states, self.num_states), dtype=mx.float32) - mx.eye(self.num_states, dtype=mx.float32)
        diversity = mx.sum(mx.square(cosine * offdiag)) / mx.maximum(mx.sum(offdiag), mx.array(1.0, dtype=mx.float32))
        return usage_balance, diversity, entropy

    def runtime_stats(self, aux: dict[str, mx.array] | None) -> dict[str, float]:
        if aux is None:
            return {
                "confidence_mean": 0.0,
                "budget_mean": 0.0,
                "entropy": 0.0,
            }
        confidence = aux.get("confidence")
        budget = aux.get("budget")
        _usage_balance, _diversity, entropy = self.regularization_losses(aux)
        return {
            "confidence_mean": 0.0 if confidence is None else float(mx.mean(confidence.astype(mx.float32)).item()),
            "budget_mean": 0.0 if budget is None else float(mx.mean(budget.astype(mx.float32)).item()),
            "entropy": float(entropy.item()),
        }

    def __call__(
        self,
        h: mx.array,
        operator_codes: mx.array | None = None,
        *,
        polarity_scores: mx.array | None = None,
        reset_prior: mx.array | None = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        if h.shape[1] <= 0:
            zero_conf = mx.zeros(h.shape[:2], dtype=mx.float32)
            zero_budget = mx.ones((*h.shape[:2], 1), dtype=mx.float32)
            zero_usage = mx.zeros((*h.shape[:2], self.num_states), dtype=mx.float32)
            return h, {
                "confidence": zero_conf,
                "budget": zero_budget,
                "soft_usage": zero_usage,
                "hard_usage": zero_usage,
                "state_index": mx.zeros(h.shape[:2], dtype=mx.int32),
                "struct_state": mx.zeros((h.shape[0], h.shape[1], self.state_dim), dtype=h.dtype),
            }

        z_in = self.proj_in(h)
        if operator_codes is not None:
            op_ids = mx.minimum(
                mx.maximum(operator_codes.astype(mx.int32), mx.array(0, dtype=mx.int32)),
                mx.array(3, dtype=mx.int32),
            )
            z_in = z_in + self.operator_prior_scale * self.operator_state[op_ids].astype(z_in.dtype)
        if polarity_scores is not None:
            z_in = z_in + polarity_scores.astype(z_in.dtype)[..., None] * self.polarity_state.astype(z_in.dtype)[None, None, :]

        batch, seqlen, _ = z_in.shape
        prev_state = mx.zeros((batch, self.state_dim), dtype=z_in.dtype)
        struct_steps: list[mx.array] = []
        conf_steps: list[mx.array] = []
        budget_steps: list[mx.array] = []
        soft_steps: list[mx.array] = []
        hard_steps: list[mx.array] = []
        idx_steps: list[mx.array] = []

        for t in range(seqlen):
            if reset_prior is not None:
                keep = 1.0 - mx.clip(
                    self.reset_prior_scale * reset_prior[:, t : t + 1].astype(z_in.dtype),
                    0.0,
                    1.0,
                )
                prev_state = prev_state * keep
            proposal = nn.silu(z_in[:, t, :] + self.recur(prev_state))
            logits = self.state_logits(proposal).astype(mx.float32)
            probs = mx.softmax(logits / self.temperature, axis=-1).astype(mx.float32)
            assign, state_idx = self._hard_assign(probs)
            state_basis = assign @ self.effective_state_book().astype(assign.dtype)
            struct_state = mx.tanh(proposal + state_basis.astype(proposal.dtype) + prev_state)
            confidence, budget = self._confidence_and_budget(probs)
            prev_state = struct_state

            struct_steps.append(struct_state[:, None, :])
            conf_steps.append(confidence[:, None])
            budget_steps.append(budget[:, None, None].astype(mx.float32))
            soft_steps.append(probs[:, None, :])
            hard_steps.append(assign[:, None, :])
            idx_steps.append(state_idx[:, None])

        struct_seq = mx.concatenate(struct_steps, axis=1)
        confidence_seq = mx.concatenate(conf_steps, axis=1)
        budget_seq = mx.concatenate(budget_steps, axis=1)
        soft_usage = mx.concatenate(soft_steps, axis=1)
        hard_usage = mx.concatenate(hard_steps, axis=1)
        state_index = mx.concatenate(idx_steps, axis=1)
        gate = mx.tanh(self.gate).astype(h.dtype)[None, None, :]
        conditioned = h + gate * self.proj_out(struct_seq)
        return conditioned, {
            "confidence": confidence_seq,
            "budget": budget_seq,
            "soft_usage": soft_usage,
            "hard_usage": hard_usage,
            "state_index": state_index,
            "struct_state": struct_seq,
        }


class StaticStructuralAdapter(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        *,
        init_std: float = 0.02,
        operator_prior_scale: float = 1.0,
        reset_prior_scale: float = 1.0,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        self.state_dim = int(state_dim)
        self.num_states = 1
        self.temperature = 1.0
        self.compute_min_scale = 1.0
        self.compute_power = 1.0
        self.operator_prior_scale = float(operator_prior_scale)
        self.reset_prior_scale = max(float(reset_prior_scale), 0.0)

        self.proj_in = CastedLinear(model_dim, state_dim)
        self.mix = CastedLinear(state_dim, state_dim)
        self.proj_out = CastedLinear(state_dim, model_dim)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

        self.state_book = (mx.random.normal((1, state_dim), dtype=mx.float32) * init_std).astype(mx.float32)
        self.operator_state = (mx.random.normal((4, state_dim), dtype=mx.float32) * init_std).astype(mx.float32)
        self.polarity_state = (mx.random.normal((state_dim,), dtype=mx.float32) * init_std).astype(mx.float32)
        self.reset_state = (mx.random.normal((state_dim,), dtype=mx.float32) * init_std).astype(mx.float32)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1.0e-4)

    def effective_state_book(self) -> mx.array:
        return self.state_book.astype(mx.float32)

    def regularization_losses(self, aux: dict[str, mx.array] | None) -> tuple[mx.array, mx.array, mx.array]:
        zero = mx.array(0.0, dtype=mx.float32)
        if aux is None:
            return zero, zero, zero
        confidence = aux.get("confidence")
        if confidence is None:
            return zero, zero, zero
        confidence_f = confidence.astype(mx.float32)
        entropy = -mx.mean(
            confidence_f * mx.log(mx.maximum(confidence_f, mx.array(1.0e-8, dtype=mx.float32)))
            + (1.0 - confidence_f) * mx.log(mx.maximum(1.0 - confidence_f, mx.array(1.0e-8, dtype=mx.float32)))
        )
        return zero, zero, entropy

    def runtime_stats(self, aux: dict[str, mx.array] | None) -> dict[str, float]:
        if aux is None:
            return {
                "confidence_mean": 0.0,
                "budget_mean": 0.0,
                "entropy": 0.0,
            }
        confidence = aux.get("confidence")
        budget = aux.get("budget")
        _usage_balance, _diversity, entropy = self.regularization_losses(aux)
        return {
            "confidence_mean": 0.0 if confidence is None else float(mx.mean(confidence.astype(mx.float32)).item()),
            "budget_mean": 0.0 if budget is None else float(mx.mean(budget.astype(mx.float32)).item()),
            "entropy": float(entropy.item()),
        }

    def __call__(
        self,
        h: mx.array,
        operator_codes: mx.array | None = None,
        *,
        polarity_scores: mx.array | None = None,
        reset_prior: mx.array | None = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        z = self.proj_in(h)
        if operator_codes is not None:
            op_ids = mx.minimum(
                mx.maximum(operator_codes.astype(mx.int32), mx.array(0, dtype=mx.int32)),
                mx.array(3, dtype=mx.int32),
            )
            z = z + self.operator_prior_scale * self.operator_state[op_ids].astype(z.dtype)
        if polarity_scores is not None:
            z = z + polarity_scores.astype(z.dtype)[..., None] * self.polarity_state.astype(z.dtype)[None, None, :]
        if reset_prior is not None:
            z = z + self.reset_prior_scale * reset_prior.astype(z.dtype)[..., None] * self.reset_state.astype(z.dtype)[None, None, :]

        hidden = nn.silu(z)
        struct_state = mx.tanh(hidden + self.mix(hidden))
        confidence = mx.sigmoid(mx.mean(mx.abs(struct_state.astype(mx.float32)), axis=-1))
        budget = mx.ones((*h.shape[:2], 1), dtype=mx.float32)
        usage = mx.ones((*h.shape[:2], 1), dtype=mx.float32)
        state_index = mx.zeros(h.shape[:2], dtype=mx.int32)
        gate = mx.tanh(self.gate).astype(h.dtype)[None, None, :]
        conditioned = h + gate * self.proj_out(struct_state)
        return conditioned, {
            "confidence": confidence,
            "budget": budget,
            "soft_usage": usage,
            "hard_usage": usage,
            "state_index": state_index,
            "struct_state": struct_state,
        }
