#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("numpy._core.numeric", importlib.import_module("numpy.core.numeric"))

from train_gpt import (
    GPT,
    apply_rotary_emb,
    build_sentencepiece_luts,
    dequantize_turbo_tensor,
    eval_val_single_process,
    infer_turbo_mode,
    limit_validation_tokens,
    load_validation_tokens,
)


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
    return {
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


class KVAblationWrapper(torch.nn.Module):
    def __init__(self, base_model: GPT, token_scale_lut: torch.Tensor):
        super().__init__()
        self.base = base_model
        self.register_buffer("token_scale_lut", token_scale_lut.to(dtype=torch.float32), persistent=False)

    def _attn_forward(self, attn, x: torch.Tensor, kv_scale: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = attn.c_q(x).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.c_k(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.c_v(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]
        scale = kv_scale[:, None, :, None].to(dtype=k.dtype)
        k = k * scale
        v = v * scale
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(attn.num_kv_heads != attn.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return attn.proj(y)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        kv_scale = self.token_scale_lut[input_ids]
        x = self.base.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[torch.Tensor] = []
        for i in range(self.base.num_encoder_layers):
            block = self.base._block_for_step(i)
            mix = block.resid_mix.to(dtype=x.dtype)
            xb = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = self._attn_forward(block.attn, block.attn_norm(xb), kv_scale)
            x = xb + block.attn_scale.to(dtype=xb.dtype)[None, None, :] * attn_out
            x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))
            skips.append(x)
        for i in range(self.base.num_decoder_layers):
            block = self.base._block_for_step(self.base.num_encoder_layers + i)
            if skips:
                x = x + self.base.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            mix = block.resid_mix.to(dtype=x.dtype)
            xb = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = self._attn_forward(block.attn, block.attn_norm(xb), kv_scale)
            x = xb + block.attn_scale.to(dtype=xb.dtype)[None, None, :] * attn_out
            x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))
        x = self.base.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.base.tie_embeddings:
            logits_proj = F.linear(x, self.base.tok_emb.weight)
        else:
            logits_proj = self.base.lm_head(x)
        logits = self.base.logit_softcap * torch.tanh(logits_proj / self.base.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def make_model_from_checkpoint(checkpoint_path: Path) -> GPT:
    state = load_quantized_checkpoint(checkpoint_path)
    cfg = infer_model_config(state)
    model = GPT(**cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_model(model: torch.nn.Module, *, seq_len: int, batch_tokens: int, val_tokens: torch.Tensor, sp) -> tuple[float, float]:
    args = SimpleNamespace(train_seq_len=seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, sp.vocab_size(), torch.device("cpu"))
    return eval_val_single_process(
        args,
        model,
        torch.device("cpu"),
        batch_tokens,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )


def resolve_token_ids(sp, pieces_csv: str) -> list[tuple[int, str]]:
    pieces = [p.strip() for p in pieces_csv.split(",") if p.strip()]
    out: list[tuple[int, str]] = []
    for raw in pieces:
        candidates = [raw]
        if not raw.startswith("▁"):
            candidates.append("▁" + raw)
        token_id = -1
        token_piece = raw
        for piece in candidates:
            pid = int(sp.piece_to_id(piece))
            if pid >= 0 and sp.id_to_piece(pid) == piece:
                token_id = pid
                token_piece = piece
                break
        if token_id < 0:
            raise ValueError(f"Could not resolve piece {raw!r}")
        out.append((token_id, token_piece))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", default=str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    parser.add_argument("--val-files", default=str(ROOT / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_val_*.bin"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--token-pieces", default="the,that,to,as,if,not,because,and,or")
    parser.add_argument("--low-scale", type=float, default=0.0)
    parser.add_argument("--combined-k", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    model = make_model_from_checkpoint(Path(args.checkpoint))
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.max_seqs)
    baseline_loss, baseline_bpb = evaluate_model(model, seq_len=args.seq_len, batch_tokens=args.batch_tokens, val_tokens=val_tokens, sp=sp)

    token_info = resolve_token_ids(sp, args.token_pieces)
    results: list[dict[str, object]] = []
    token_scale_lut = torch.ones((sp.vocab_size(),), dtype=torch.float32)
    for token_id, piece in token_info:
        lut = token_scale_lut.clone()
        lut[token_id] = args.low_scale
        wrapped = KVAblationWrapper(model, lut)
        loss, bpb = evaluate_model(wrapped, seq_len=args.seq_len, batch_tokens=args.batch_tokens, val_tokens=val_tokens, sp=sp)
        count = int((val_tokens[:-1] == token_id).sum().item())
        results.append(
            {
                "token_id": token_id,
                "piece": piece,
                "count": count,
                "loss": loss,
                "bpb": bpb,
                "delta_bpb": bpb - baseline_bpb,
                "delta_bpb_per_occurrence": (bpb - baseline_bpb) / max(count, 1),
            }
        )

    ranked = sorted(results, key=lambda item: float(item["delta_bpb"]))
    low_utility = ranked[: max(args.combined_k, 0)]
    combined_lut = token_scale_lut.clone()
    for item in low_utility:
        combined_lut[int(item["token_id"])] = args.low_scale
    combined_wrapped = KVAblationWrapper(model, combined_lut)
    combined_loss, combined_bpb = evaluate_model(
        combined_wrapped,
        seq_len=args.seq_len,
        batch_tokens=args.batch_tokens,
        val_tokens=val_tokens,
        sp=sp,
    )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seq_len": args.seq_len,
        "max_seqs": args.max_seqs,
        "batch_tokens": args.batch_tokens,
        "low_scale": args.low_scale,
        "baseline": {"loss": baseline_loss, "bpb": baseline_bpb},
        "token_results": ranked,
        "combined_low_utility": {
            "tokens": low_utility,
            "loss": combined_loss,
            "bpb": combined_bpb,
            "delta_bpb": combined_bpb - baseline_bpb,
        },
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
