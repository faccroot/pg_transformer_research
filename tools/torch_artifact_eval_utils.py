#!/usr/bin/env python3
from __future__ import annotations

import importlib
import math
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("numpy._core.numeric", importlib.import_module("numpy.core.numeric"))

from train_gpt import GPT, dequantize_turbo_tensor


def _to_torch(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    return obj


def _convert_meta(meta: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in meta.items():
        if isinstance(value, dict):
            out[key] = _convert_meta(value)
        else:
            out[key] = _to_torch(value)
    return out


def load_quantized_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = pickle.loads(zlib.decompress(path.read_bytes()))
    state: dict[str, torch.Tensor] = {}
    for name, tensor in payload.get("quantized", {}).items():
        q = _to_torch(tensor).to(dtype=torch.float32)
        scale = _to_torch(payload["scales"][name]).to(dtype=torch.float32)
        qmeta = payload.get("qmeta", {}).get(name, {})
        if qmeta.get("scheme") == "per_row" or scale.ndim > 0:
            restored = q * scale.view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            restored = q * float(scale.item())
        dtype_name = payload.get("dtypes", {}).get(name, "float32")
        state[name] = restored.to(dtype=getattr(torch, dtype_name)).contiguous()
    for name, tensor in payload.get("passthrough", {}).items():
        value = _to_torch(tensor).contiguous()
        if isinstance(value, torch.Tensor):
            state[name] = value
    for name, meta in payload.get("turbo", {}).items():
        state[name] = dequantize_turbo_tensor(_convert_meta(meta)).contiguous()
    return state


def infer_model_config(state: dict[str, torch.Tensor]) -> dict[str, int | bool | float]:
    vocab_size, model_dim = map(int, state["tok_emb.weight"].shape)
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in state
            if name.startswith("blocks.") and name.split(".")[1].isdigit()
        }
    )
    if not block_ids:
        raise ValueError("Could not infer block count from state dict")
    num_layers = max(block_ids) + 1
    num_layer_templates = len(block_ids)
    num_heads = int(state["blocks.0.attn.q_gain"].numel())
    head_dim = model_dim // num_heads
    num_kv_heads = int(state["blocks.0.attn.c_k.weight"].shape[0] // head_dim)
    mlp_mult = int(state["blocks.0.mlp.fc.weight"].shape[0] // model_dim)
    tie_embeddings = "lm_head.weight" not in state
    cfg: dict[str, int | bool | float] = {
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_layer_templates": num_layer_templates,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_mult": mlp_mult,
        "tie_embeddings": tie_embeddings,
        "tied_embed_init_std": 0.005,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.5,
    }
    statelex_split: list[int] = []
    router_weight = state.get("statelex.router.0.weight")
    if router_weight is not None:
        cfg["statelex_router_hidden_dim"] = int(router_weight.shape[0])
    router_out = state.get("statelex.router.2.weight")
    if router_out is not None:
        cfg["statelex_num_states"] = int(router_out.shape[0])
        cfg["statelex_state_dim"] = int(router_out.shape[1])
    token_mask = state.get("statelex.split_token_mask")
    if token_mask is not None:
        statelex_split = torch.nonzero(token_mask > 0, as_tuple=False).view(-1).tolist()
    if statelex_split:
        cfg["statelex_split_token_ids"] = tuple(int(v) for v in statelex_split)
    statelex_delta = state.get("statelex.state_delta")
    if statelex_delta is not None and cfg.get("statelex_num_states", 0):
        denom = max(float(cfg["statelex_num_states"]), 1.0)
        cfg["statelex_delta_scale"] = float(statelex_delta.float().std().item() * math.sqrt(denom))
    return cfg


def make_model_from_checkpoint(checkpoint_path: Path, device: torch.device | str = "cpu") -> GPT:
    state = load_quantized_checkpoint(checkpoint_path)
    cfg = infer_model_config(state)
    model = GPT(**cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_probe_windows(total_tokens: int, seq_len: int, stride: int) -> list[tuple[int, int]]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    windows: list[tuple[int, int]] = []
    for window_start in range(0, total_tokens, stride):
        window_end = min(window_start + seq_len, total_tokens)
        window_len = window_end - window_start
        if window_len <= 0:
            continue
        windows.append((window_start, window_len))
        if window_end >= total_tokens:
            break
    if not windows:
        raise ValueError("No probe windows could be built")
    return windows


def _expand_prefix(
    prefix: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if prefix.ndim == 2:
        prefix = prefix.unsqueeze(0)
    if prefix.ndim != 3:
        raise ValueError(f"Expected prefix tensor rank 2 or 3, got {tuple(prefix.shape)}")
    if prefix.size(0) == 1 and batch_size > 1:
        prefix = prefix.expand(batch_size, -1, -1)
    if prefix.size(0) != batch_size:
        raise ValueError(f"Prefix batch {prefix.size(0)} != input batch {batch_size}")
    return prefix.to(device=device, dtype=dtype)


def forward_logits_with_prefix(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    prefix_input_ids: torch.Tensor | None = None,
    prefix_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix_input_ids is not None and prefix_embeddings is not None:
        raise ValueError("Provide at most one prefix source")

    x_real = model.embed_inputs(input_ids)
    batch_size = x_real.size(0)
    prefix_len = 0

    if prefix_input_ids is not None:
        prefix = model.embed_inputs(prefix_input_ids.to(device=input_ids.device, dtype=torch.int64))
        prefix_len = int(prefix.size(1))
        x = torch.cat([prefix, x_real], dim=1)
    elif prefix_embeddings is not None:
        prefix = _expand_prefix(
            prefix_embeddings,
            batch_size=batch_size,
            device=x_real.device,
            dtype=x_real.dtype,
        )
        prefix_len = int(prefix.size(1))
        x = torch.cat([prefix, x_real], dim=1)
    else:
        x = x_real

    x0 = x
    skips: list[torch.Tensor] = []

    for i in range(model.num_encoder_layers):
        x = model._block_for_step(i)(x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model._block_for_step(model.num_encoder_layers + i)(x, x0)

    hidden = model.final_norm(x)
    hidden_real = hidden[:, prefix_len:, :]
    if model.tie_embeddings:
        logits_proj = F.linear(hidden_real, model.tok_emb.weight)
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings=False")
        logits_proj = model.lm_head(hidden_real)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return logits, hidden_real


def per_token_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    return F.cross_entropy(
        logits.reshape(batch * seq_len, vocab).float(),
        target_ids.reshape(batch * seq_len),
        reduction="none",
    ).reshape(batch, seq_len)
