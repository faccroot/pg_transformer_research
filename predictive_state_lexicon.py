from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_statelex_split_spec(
    sp,
    vocab_size: int,
    *,
    split_pieces: str = "",
    split_ids: str = "",
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    token_ids: list[int] = []
    token_labels: list[str] = []
    seen: set[int] = set()

    for raw in _parse_csv(split_ids):
        token_id = int(raw)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"STATELEX_SPLIT_IDS contains out-of-range token id {token_id}")
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
        token_labels.append(sp.id_to_piece(int(token_id)))

    for raw_piece in _parse_csv(split_pieces):
        candidates = [raw_piece]
        if not raw_piece.startswith("▁"):
            candidates.append("▁" + raw_piece)
        resolved_id: int | None = None
        resolved_piece: str | None = None
        for piece in candidates:
            token_id = int(sp.piece_to_id(piece))
            if 0 <= token_id < vocab_size and sp.id_to_piece(token_id) == piece:
                resolved_id = token_id
                resolved_piece = piece
                break
        if resolved_id is None or resolved_piece is None:
            raise ValueError(f"STATELEX_SPLIT_PIECES could not resolve piece {raw_piece!r}")
        if resolved_id in seen:
            continue
        seen.add(resolved_id)
        token_ids.append(resolved_id)
        token_labels.append(resolved_piece)

    return tuple(token_ids), tuple(token_labels)


class PredictiveStateLexicon(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_dim: int,
        state_dim: int,
        num_states: int,
        split_token_ids: Iterable[int],
        router_hidden_dim: int = 0,
        init_std: float = 0.02,
        delta_scale: float = 0.25,
    ):
        super().__init__()
        split_ids = tuple(int(token_id) for token_id in split_token_ids)
        if not split_ids:
            raise ValueError("PredictiveStateLexicon requires at least one split token id")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_states <= 1:
            raise ValueError(f"num_states must be > 1, got {num_states}")
        if delta_scale < 0.0:
            raise ValueError(f"delta_scale must be non-negative, got {delta_scale}")

        token_to_slot = torch.full((vocab_size,), -1, dtype=torch.long)
        for slot, token_id in enumerate(split_ids):
            token_to_slot[token_id] = slot
        self.register_buffer("token_to_slot", token_to_slot, persistent=False)
        self.register_buffer("split_token_ids", torch.tensor(split_ids, dtype=torch.long), persistent=False)

        self.state_dim = int(state_dim)
        self.num_states = int(num_states)
        self.delta_scale = float(delta_scale)

        self.state_in = nn.Linear(model_dim, state_dim, bias=False)
        self.state_h = nn.Linear(state_dim, state_dim, bias=False)
        self.gate_in = nn.Linear(model_dim, state_dim, bias=False)
        self.gate_h = nn.Linear(state_dim, state_dim, bias=False)

        router_in_dim = model_dim + state_dim
        self.router_hidden = (
            nn.Linear(router_in_dim, router_hidden_dim, bias=False)
            if router_hidden_dim > 0 else None
        )
        self.router_out = nn.Linear(router_hidden_dim if router_hidden_dim > 0 else router_in_dim, num_states, bias=False)
        self.delta_table = nn.Parameter(torch.empty(len(split_ids), num_states, model_dim))

        self._reset_parameters(init_std)

    def _reset_parameters(self, init_std: float) -> None:
        for module in [self.state_in, self.state_h, self.gate_in, self.gate_h, self.router_hidden, self.router_out]:
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.delta_table, mean=0.0, std=init_std)

    def forward(self, input_ids: Tensor, base_embeddings: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, seq_len, _ = base_embeddings.shape
        prefix_state = base_embeddings.new_zeros(batch_size, self.state_dim)
        prefix_states: list[Tensor] = []
        for pos in range(seq_len):
            prefix_states.append(prefix_state)
            token_emb = base_embeddings[:, pos, :]
            proposal = torch.tanh(self.state_in(token_emb) + self.state_h(prefix_state))
            gate = torch.sigmoid(self.gate_in(token_emb) + self.gate_h(prefix_state))
            prefix_state = gate * proposal + (1.0 - gate) * prefix_state
        prefix = torch.stack(prefix_states, dim=1)

        slot = self.token_to_slot[input_ids]
        active = slot >= 0
        route_features = torch.cat((prefix, base_embeddings), dim=-1)
        if self.router_hidden is not None:
            route_hidden = torch.tanh(self.router_hidden(route_features))
            router_logits = self.router_out(route_hidden)
        else:
            router_logits = self.router_out(route_features)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(base_embeddings.dtype)

        mixed_delta = torch.zeros_like(base_embeddings)
        active_count = int(active.sum().item())
        if active_count > 0:
            active_slots = slot[active]
            active_probs = router_probs[active]
            active_deltas = self.delta_table[active_slots].to(active_probs.dtype)
            active_mixed = torch.sum(active_probs.unsqueeze(-1) * active_deltas, dim=-2)
            mixed_delta[active] = active_mixed.to(base_embeddings.dtype)

        output = base_embeddings + self.delta_scale * mixed_delta

        if active_count > 0:
            active_probs_f = router_probs[active].float()
            entropy = -(active_probs_f * torch.log(active_probs_f.clamp_min(1e-8))).sum(dim=-1).mean()
        else:
            entropy = base_embeddings.new_zeros((), dtype=torch.float32)
        stats = {
            "statelex_entropy": entropy,
            "statelex_active_fraction": active.float().mean(),
        }
        return output, stats
