from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from tools.representation_learning.schemas import LayerGeometry
    except ModuleNotFoundError:
        from schemas import LayerGeometry  # type: ignore[no-redef]
else:
    from .schemas import LayerGeometry


class _GeometryLike(Protocol):
    layer_geometries: dict[int, LayerGeometry]


def _orthonormal_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    q, _r = torch.linalg.qr(matrix.T, mode="reduced")
    return q.T


def _random_orthogonal(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    base = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(base, mode="reduced")
    sign = torch.sign(torch.diag(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(0)


def _mapped_directions(directions: np.ndarray, target_dim: int, seed: int) -> torch.Tensor:
    source = torch.tensor(np.asarray(directions, dtype=np.float32))
    if source.shape[0] == 0:
        return torch.zeros((0, target_dim), dtype=torch.float32)
    if source.shape[1] == target_dim:
        return _orthonormal_rows(source)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn((source.shape[1], target_dim), generator=generator, dtype=torch.float32)
    projected = source @ projection / max(float(np.sqrt(source.shape[1])), 1.0)
    return _orthonormal_rows(projected)


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / max(self.head_dim ** 0.5, 1e-6)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        return self.o_proj(y)


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = TinySelfAttention(hidden_dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.down_proj(F.gelu(self.up_proj(self.ln2(x))))
        return x


class TinyReasoningCore(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList(
            TinyBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim)
            for _ in range(num_layers)
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_ln(x))
        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="mean",
        )
        return logits, loss


@dataclass
class ReasoningCoreAssembler:
    geometry: _GeometryLike

    def _select_layer(self, relative_depth: float) -> LayerGeometry:
        candidates = self.geometry.layer_geometries
        if not candidates:
            raise ValueError("Geometry has no layers to assemble from")
        best_idx = min(candidates, key=lambda idx: abs(candidates[idx].relative_depth - relative_depth))
        return candidates[best_idx]

    def _construct_q_weight(self, layer: LayerGeometry, target_dim: int, *, strength: float, seed: int) -> torch.Tensor:
        basis = _mapped_directions(layer.directions, target_dim=target_dim, seed=seed)
        if basis.shape[0] == 0:
            return torch.eye(target_dim, dtype=torch.float32)
        scales = layer.scales
        if scales is None:
            scale_vector = torch.ones((basis.shape[0],), dtype=torch.float32)
        else:
            scale_vector = torch.tensor(scales[: basis.shape[0]], dtype=torch.float32)
            scale_vector = scale_vector / scale_vector.abs().mean().clamp_min(1e-6)
        update = basis.T @ torch.diag(scale_vector) @ basis
        weight = torch.eye(target_dim, dtype=torch.float32) + float(strength) * update
        spectral = torch.linalg.matrix_norm(weight, ord=2).clamp_min(1.0)
        return weight / spectral

    def _construct_mlp_weights(
        self,
        layer: LayerGeometry,
        *,
        target_dim: int,
        mlp_hidden_dim: int,
        strength: float,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = _mapped_directions(layer.directions, target_dim=target_dim, seed=seed)
        up = torch.zeros((mlp_hidden_dim, target_dim), dtype=torch.float32)
        down = torch.zeros((target_dim, mlp_hidden_dim), dtype=torch.float32)
        if basis.shape[0] == 0:
            filler = _random_orthogonal(target_dim, seed=seed)
            rows = min(mlp_hidden_dim, target_dim)
            up[:rows, :] = filler[:rows, :]
            down[:, :rows] = filler[:, :rows]
            return up, down
        active = min(int(basis.shape[0]), int(mlp_hidden_dim))
        graph = layer.coactivation
        if graph is None:
            graph_tensor = torch.eye(active, dtype=torch.float32)
        else:
            graph_tensor = torch.tensor(np.asarray(graph, dtype=np.float32)[:active, :active])
            graph_norm = torch.linalg.matrix_norm(graph_tensor, ord=2).clamp_min(1e-6)
            graph_tensor = graph_tensor / graph_norm
        up[:active, :] = basis[:active, :]
        down[:, :active] = float(strength) * (basis[:active, :].T @ graph_tensor)
        return up, down

    def assemble(
        self,
        *,
        target_hidden_dim: int,
        target_num_layers: int,
        target_vocab_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        max_seq_len: int = 1024,
        q_strength: float = 1.0,
        mlp_strength: float = 1.0,
        seed: int = 17,
    ) -> TinyReasoningCore:
        mlp_hidden_dim = max(int(round(target_hidden_dim * mlp_ratio)), target_hidden_dim)
        model = TinyReasoningCore(
            vocab_size=target_vocab_size,
            hidden_dim=target_hidden_dim,
            num_layers=target_num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            max_seq_len=max_seq_len,
        )
        with torch.no_grad():
            embed = _random_orthogonal(target_hidden_dim, seed=seed)
            vocab_rows = min(target_vocab_size, target_hidden_dim)
            model.token_emb.weight.zero_()
            model.token_emb.weight[:vocab_rows, :] = embed[:vocab_rows, :]
            model.pos_emb.weight.zero_()
            for layer_idx, block in enumerate(model.blocks):
                relative_depth = (layer_idx + 1) / max(target_num_layers, 1)
                geometry_layer = self._select_layer(relative_depth)
                block.attn.q_proj.weight.copy_(
                    self._construct_q_weight(geometry_layer, target_dim=target_hidden_dim, strength=q_strength, seed=seed + layer_idx)
                )
                block.attn.k_proj.weight.copy_(_random_orthogonal(target_hidden_dim, seed=seed + 1000 + layer_idx))
                block.attn.v_proj.weight.copy_(_random_orthogonal(target_hidden_dim, seed=seed + 2000 + layer_idx))
                block.attn.o_proj.weight.copy_(_random_orthogonal(target_hidden_dim, seed=seed + 3000 + layer_idx))
                up, down = self._construct_mlp_weights(
                    geometry_layer,
                    target_dim=target_hidden_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    strength=mlp_strength,
                    seed=seed + 4000 + layer_idx,
                )
                block.up_proj.weight.copy_(up)
                block.down_proj.weight.copy_(down)
        return model

    def evaluate_zero_shot(self, model: TinyReasoningCore, input_ids: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            _logits, loss = model(input_ids, labels=input_ids)
        return float(loss.item())
