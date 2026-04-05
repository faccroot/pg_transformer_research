#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictive_state_lexicon import resolve_statelex_split_spec
from train_gpt import GPT, load_data_shard


def build_batches(tokens: torch.Tensor, *, seq_len: int, batch_size: int, num_steps: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    max_start = int(tokens.numel()) - (seq_len + 1)
    if max_start <= 0:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}")
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_steps):
        starts = [rng.randrange(0, max_start) for _ in range(batch_size)]
        x_rows = [tokens[s : s + seq_len].to(dtype=torch.long) for s in starts]
        y_rows = [tokens[s + 1 : s + 1 + seq_len].to(dtype=torch.long) for s in starts]
        batches.append((torch.stack(x_rows, dim=0), torch.stack(y_rows, dim=0)))
    return batches


def make_model(
    *,
    vocab_size: int,
    statelex_split_ids: tuple[int, ...],
    enable_statelex: bool,
    model_dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    num_states: int,
    state_dim: int,
    router_hidden_dim: int,
    init_std: float,
    delta_scale: float,
) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_layer_templates=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        statelex_num_states=num_states if enable_statelex else 0,
        statelex_state_dim=state_dim if enable_statelex else 0,
        statelex_router_hidden_dim=router_hidden_dim if enable_statelex else 0,
        statelex_split_token_ids=statelex_split_ids if enable_statelex else (),
        statelex_init_std=init_std,
        statelex_delta_scale=delta_scale,
    ).float()


def copy_common_weights(src: GPT, dst: GPT) -> None:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for name, tensor in src_state.items():
        if name in dst_state and dst_state[name].shape == tensor.shape:
            dst_state[name].copy_(tensor)
    dst.load_state_dict(dst_state)


def train_steps(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]], *, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    parser.add_argument("--tokenizer-path", default=str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    parser.add_argument("--split-pieces", default="the,that,to,as,if,not,because,and,or")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--statelex-num-states", type=int, default=4)
    parser.add_argument("--statelex-state-dim", type=int, default=32)
    parser.add_argument("--statelex-router-hidden-dim", type=int, default=32)
    parser.add_argument("--statelex-init-std", type=float, default=0.02)
    parser.add_argument("--statelex-delta-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    split_ids, split_labels = resolve_statelex_split_spec(
        sp,
        int(sp.vocab_size()),
        split_pieces=args.split_pieces,
    )
    train_file = sorted(Path(args.data_path).glob("fineweb_train_*.bin"))[0]
    tokens = load_data_shard(train_file)
    batches = build_batches(tokens, seq_len=args.seq_len, batch_size=args.batch_size, num_steps=args.steps, seed=args.seed)

    baseline = make_model(
        vocab_size=int(sp.vocab_size()),
        statelex_split_ids=split_ids,
        enable_statelex=False,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        num_states=args.statelex_num_states,
        state_dim=args.statelex_state_dim,
        router_hidden_dim=args.statelex_router_hidden_dim,
        init_std=args.statelex_init_std,
        delta_scale=args.statelex_delta_scale,
    )
    statelex = make_model(
        vocab_size=int(sp.vocab_size()),
        statelex_split_ids=split_ids,
        enable_statelex=True,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        num_states=args.statelex_num_states,
        state_dim=args.statelex_state_dim,
        router_hidden_dim=args.statelex_router_hidden_dim,
        init_std=args.statelex_init_std,
        delta_scale=args.statelex_delta_scale,
    )
    copy_common_weights(baseline, statelex)

    baseline_losses = train_steps(baseline, batches, lr=args.lr)
    statelex_losses = train_steps(statelex, batches, lr=args.lr)

    print("split_labels", ",".join(split_labels))
    print("baseline_losses", ",".join(f"{loss:.4f}" for loss in baseline_losses))
    print("statelex_losses", ",".join(f"{loss:.4f}" for loss in statelex_losses))
    print("baseline_final", f"{baseline_losses[-1]:.6f}")
    print("statelex_final", f"{statelex_losses[-1]:.6f}")
    print("delta_final", f"{statelex_losses[-1] - baseline_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
