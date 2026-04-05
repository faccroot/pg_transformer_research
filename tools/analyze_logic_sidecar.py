#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import analyze_mlx_quant_export as aq
import train_gpt_mlx as tg


def scalar_accumulator() -> dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "max": 0.0}


def update_scalar(acc: dict[str, float], values: np.ndarray) -> None:
    if values.size == 0:
        return
    vals = np.asarray(values, dtype=np.float32)
    acc["count"] += float(vals.size)
    acc["sum"] += float(vals.sum(dtype=np.float64))
    acc["sum_sq"] += float(np.square(vals, dtype=np.float64).sum())
    acc["max"] = max(acc["max"], float(np.max(vals)))


def finalize_scalar(acc: dict[str, float]) -> dict[str, float]:
    denom = max(acc["count"], 1.0)
    return {
        "mean": acc["sum"] / denom,
        "rms": math.sqrt(acc["sum_sq"] / denom),
        "max": acc["max"],
        "count": acc["count"],
    }


def expand_kv_heads(x: mx.array, num_heads: int, num_kv_heads: int) -> mx.array:
    if num_heads == num_kv_heads:
        return x
    repeats = num_heads // num_kv_heads
    return mx.repeat(x, repeats=repeats, axis=1)


def attention_logits(attn: tg.CausalSelfAttention, x: mx.array) -> mx.array:
    bsz, seqlen, dim = x.shape
    q = attn.c_q(x).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = attn.c_k(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    q = attn.rope(tg.rms_norm(q).astype(tg.COMPUTE_DTYPE))
    k = attn.rope(tg.rms_norm(k).astype(tg.COMPUTE_DTYPE))
    q = q * attn.q_gain.astype(q.dtype)[None, :, None, None]
    k = expand_kv_heads(k, attn.num_heads, attn.num_kv_heads)
    return (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * float(attn.scale)


def resolve_attention_mask(model: tg.GPT, total_len: int) -> mx.array:
    mask = model.attention_mask(total_len)
    if isinstance(mask, str):
        return tg.build_register_attention_mask(total_len, 0)
    return mask


def attention_probs(attn: tg.CausalSelfAttention, x: mx.array, mask: mx.array) -> mx.array:
    return mx.softmax(attention_logits(attn, x) + mask.astype(mx.float32), axis=-1).astype(mx.float32)


def infer_logic_config(
    flat_state: dict[str, mx.array],
    logic_layer_index: int | None,
    logic_route_to_next_token: bool,
) -> dict[str, object]:
    config = dict(aq.infer_model_config(flat_state))
    register_state = flat_state.get("register_tokens.registers")
    logic_proj = flat_state.get("logic_sidecar.proj_in.weight")
    config["num_registers"] = int(register_state.shape[0]) if register_state is not None else 0
    config["logic_dim"] = int(logic_proj.shape[0]) if logic_proj is not None else 0
    config["logic_layer_index"] = (
        logic_layer_index
        if logic_layer_index is not None
        else (int(config["num_layers"]) // 3 if int(config["logic_dim"]) > 0 else None)
    )
    config["logic_route_to_next_token"] = logic_route_to_next_token
    return config


def build_model(config: dict[str, object], sp: spm.SentencePieceProcessor) -> tg.GPT:
    operator_routing = (
        tg.build_operator_routing_spec(sp, int(config["vocab_size"]))
        if int(config["logic_dim"]) > 0
        else None
    )
    model = tg.GPT(
        vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_layers"]),
        num_layer_templates=int(config["num_layer_templates"]),
        dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        mlp_mult=int(config["mlp_mult"]),
        mlp_leaky_slope=float(config.get("mlp_leaky_slope", 0.0)),
        tie_embeddings=bool(config["tie_embeddings"]),
        logit_chunk_tokens=0,
        logit_softcap=float(config["logit_softcap"]),
        rope_base=float(config["rope_base"]),
        tied_embed_init_std=float(config["tied_embed_init_std"]),
        qk_gain_init=float(config["qk_gain_init"]),
        num_registers=int(config["num_registers"]),
        logic_dim=int(config["logic_dim"]),
        logic_layer_index=config["logic_layer_index"],
        logic_route_to_next_token=bool(config["logic_route_to_next_token"]),
        operator_routing=operator_routing,
    )
    return model


def subset_eval(
    model: tg.GPT,
    val_tokens: np.ndarray,
    train_seq_len: int,
    seq_indices: np.ndarray,
    val_batch_size: int,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> dict[str, float | int] | None:
    if seq_indices.size == 0:
        return None
    batch_seqs = max(val_batch_size // train_seq_len, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for start in range(0, int(seq_indices.size), batch_seqs):
        batch_ids = seq_indices[start : start + batch_seqs]
        x_batch = []
        y_batch = []
        for seq_idx in batch_ids:
            raw_start = int(seq_idx) * train_seq_len
            chunk = val_tokens[raw_start : raw_start + train_seq_len + 1]
            x_batch.append(chunk[:-1])
            y_batch.append(chunk[1:])
        x_np = np.stack(x_batch, axis=0)
        y_np = np.stack(y_batch, axis=0)
        operator_codes = model.operator_codes_for_numpy(x_np)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        batch_loss = model.loss(
            x,
            y,
            None if operator_codes is None else mx.array(operator_codes, dtype=mx.int32),
        ).astype(mx.float32)
        mx.eval(batch_loss)
        token_count = float(y.size)
        total_loss_sum += float(batch_loss.item()) * token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
    val_loss = total_loss_sum / max(total_tokens, 1.0)
    bits_per_token = val_loss / math.log(2.0)
    return {
        "seqs": int(seq_indices.size),
        "val_loss": float(val_loss),
        "val_bpb": float(bits_per_token * (total_tokens / max(total_bytes, 1.0))),
    }


def analyze_probe(
    model: tg.GPT,
    probe_tokens: np.ndarray,
    train_seq_len: int,
) -> dict[str, object]:
    total_seqs = (probe_tokens.size - 1) // train_seq_len
    layer_stats = {
        f"layer_{layer_idx}": {
            "all_queries": scalar_accumulator(),
            "token_queries": scalar_accumulator(),
            "register_queries": scalar_accumulator(),
        }
        for layer_idx in range(model.num_layers)
    }
    effect_stats = {
        "applied_not": scalar_accumulator(),
        "applied_and": scalar_accumulator(),
        "applied_or": scalar_accumulator(),
        "other_tokens": scalar_accumulator(),
        "registers": scalar_accumulator(),
    }
    negation_sequences = 0
    negation_tokens = 0
    and_tokens = 0
    or_tokens = 0

    for seq_idx in range(total_seqs):
        raw_start = seq_idx * train_seq_len
        chunk = probe_tokens[raw_start : raw_start + train_seq_len + 1]
        x_np = chunk[:-1].reshape(1, train_seq_len)
        x_ids = mx.array(x_np, dtype=mx.int32)
        raw_operator_codes_np = model.operator_codes_for_numpy(
            x_np,
            route_to_next=False,
            pad_registers_for_output=False,
        )
        if raw_operator_codes_np is None:
            continue
        raw_operator_codes = mx.array(raw_operator_codes_np, dtype=mx.int32)
        op_np = np.asarray(raw_operator_codes_np, dtype=np.int32)
        if not np.any(op_np == 1):
            continue
        negation_sequences += 1
        negation_tokens += int(np.sum(op_np == 1))
        and_tokens += int(np.sum(op_np == 2))
        or_tokens += int(np.sum(op_np == 3))

        x = model.embed_inputs(x_ids)
        x0 = x
        mask = resolve_attention_mask(model, int(x.shape[1]))
        raw_operator_codes = tg.pad_operator_codes(raw_operator_codes, model.num_registers)
        routed_operator_codes_np = model.operator_codes_for_numpy(x_np)
        routed_operator_codes = (
            None
            if routed_operator_codes_np is None
            else mx.array(routed_operator_codes_np, dtype=mx.int32)
        )
        skips: list[mx.array] = []
        layer_idx = 0

        for enc_idx in range(model.num_encoder_layers):
            block = model.block_for_step(layer_idx)
            mix = block.resid_mix.astype(x.dtype)
            x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            probs = attention_probs(block.attn, block.attn_norm(x_in), mask)
            neg_mass = (probs * mx.expand_dims(mx.expand_dims(raw_operator_codes == 1, axis=1), axis=1).astype(probs.dtype)).sum(axis=-1)
            neg_mass_np = np.asarray(neg_mass.astype(mx.float32), dtype=np.float32)
            update_scalar(layer_stats[f"layer_{layer_idx}"]["all_queries"], neg_mass_np)
            update_scalar(layer_stats[f"layer_{layer_idx}"]["token_queries"], neg_mass_np[:, :, model.num_registers :])
            if model.num_registers > 0:
                update_scalar(layer_stats[f"layer_{layer_idx}"]["register_queries"], neg_mass_np[:, :, : model.num_registers])
            x_block = block(x, x0, attn_mask=mask)
            x_logic = model.maybe_apply_logic_sidecar(x_block, routed_operator_codes, layer_idx)
            if model.logic_sidecar is not None and layer_idx == model.logic_layer_index and routed_operator_codes is not None:
                delta_norm = mx.sqrt(mx.sum(mx.square((x_logic - x_block).astype(mx.float32)), axis=-1))
                delta_norm_np = np.asarray(delta_norm, dtype=np.float32)
                routed_np = np.asarray(routed_operator_codes.astype(mx.int32), dtype=np.int32)
                real_token_mask = np.arange(routed_np.shape[1], dtype=np.int32)[None, :] >= model.num_registers
                register_mask = ~real_token_mask
                update_scalar(effect_stats["applied_not"], delta_norm_np[routed_np == 1])
                update_scalar(effect_stats["applied_and"], delta_norm_np[routed_np == 2])
                update_scalar(effect_stats["applied_or"], delta_norm_np[routed_np == 3])
                update_scalar(effect_stats["other_tokens"], delta_norm_np[(routed_np == 0) & real_token_mask])
                update_scalar(effect_stats["registers"], delta_norm_np[register_mask])
            x = x_logic
            skips.append(x)
            layer_idx += 1

        for dec_idx in range(model.num_decoder_layers):
            if skips:
                x = x + model.skip_weights[dec_idx].astype(x.dtype)[None, None, :] * skips.pop()
            block = model.block_for_step(layer_idx)
            mix = block.resid_mix.astype(x.dtype)
            x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            probs = attention_probs(block.attn, block.attn_norm(x_in), mask)
            neg_mass = (probs * mx.expand_dims(mx.expand_dims(raw_operator_codes == 1, axis=1), axis=1).astype(probs.dtype)).sum(axis=-1)
            neg_mass_np = np.asarray(neg_mass.astype(mx.float32), dtype=np.float32)
            update_scalar(layer_stats[f"layer_{layer_idx}"]["all_queries"], neg_mass_np)
            update_scalar(layer_stats[f"layer_{layer_idx}"]["token_queries"], neg_mass_np[:, :, model.num_registers :])
            if model.num_registers > 0:
                update_scalar(layer_stats[f"layer_{layer_idx}"]["register_queries"], neg_mass_np[:, :, : model.num_registers])
            x_block = block(x, x0, attn_mask=mask)
            x_logic = model.maybe_apply_logic_sidecar(x_block, routed_operator_codes, layer_idx)
            if model.logic_sidecar is not None and layer_idx == model.logic_layer_index and routed_operator_codes is not None:
                delta_norm = mx.sqrt(mx.sum(mx.square((x_logic - x_block).astype(mx.float32)), axis=-1))
                delta_norm_np = np.asarray(delta_norm, dtype=np.float32)
                routed_np = np.asarray(routed_operator_codes.astype(mx.int32), dtype=np.int32)
                real_token_mask = np.arange(routed_np.shape[1], dtype=np.int32)[None, :] >= model.num_registers
                register_mask = ~real_token_mask
                update_scalar(effect_stats["applied_not"], delta_norm_np[routed_np == 1])
                update_scalar(effect_stats["applied_and"], delta_norm_np[routed_np == 2])
                update_scalar(effect_stats["applied_or"], delta_norm_np[routed_np == 3])
                update_scalar(effect_stats["other_tokens"], delta_norm_np[(routed_np == 0) & real_token_mask])
                update_scalar(effect_stats["registers"], delta_norm_np[register_mask])
            x = x_logic
            layer_idx += 1

    gate_info = None
    if model.logic_sidecar is not None:
        gate_np = np.asarray(mx.tanh(model.logic_sidecar.gate).astype(mx.float32), dtype=np.float32)
        gate_info = {
            "mean_abs": float(np.abs(gate_np).mean(dtype=np.float64)),
            "rms": float(np.sqrt(np.square(gate_np, dtype=np.float64).mean())),
            "max_abs": float(np.max(np.abs(gate_np), initial=0.0)),
        }

    return {
        "probe_sequences_total": int(total_seqs),
        "probe_sequences_with_negation": int(negation_sequences),
        "probe_token_counts": {
            "not": int(negation_tokens),
            "and": int(and_tokens),
            "or": int(or_tokens),
        },
        "ntas": {
            "layers": {name: {metric: finalize_scalar(acc) for metric, acc in stats.items()} for name, stats in layer_stats.items()}
        },
        "logic_effect": {
            "layer_index": model.logic_layer_index,
            "gate": gate_info,
            "delta_norm": {name: finalize_scalar(acc) for name, acc in effect_stats.items()},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Negation-focused diagnostics for MLX logic-sidecar and register-token checkpoints.")
    parser.add_argument("checkpoint", type=Path, help="Raw MLX .npz checkpoint")
    parser.add_argument("--data-path", type=Path, required=True, help="Dataset directory containing fineweb_val_*.bin")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="SentencePiece tokenizer model")
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--probe-seqs", type=int, default=32)
    parser.add_argument("--val-max-seqs", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--logic-layer-index", type=int, help="Override side-car insertion layer for checkpoint reconstruction")
    parser.add_argument(
        "--no-logic-route-to-next-token",
        action="store_true",
        help="Disable the default operator routing shift when reconstructing the checkpointed architecture",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aq.load_flat_state(args.checkpoint)
    config = infer_logic_config(
        flat_state,
        logic_layer_index=args.logic_layer_index,
        logic_route_to_next_token=not args.no_logic_route_to_next_token,
    )
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    model = build_model(config, sp)
    model.update(tree_unflatten(list(flat_state.items())))
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()

    val_tokens = tg.limit_validation_tokens(
        tg.load_validation_tokens(str(args.data_path / "fineweb_val_*.bin"), args.train_seq_len),
        args.train_seq_len,
        args.val_max_seqs,
    )
    probe_tokens = tg.limit_validation_tokens(val_tokens, args.train_seq_len, args.probe_seqs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(sp, int(config["vocab_size"]))

    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    negation_seq_mask = np.zeros((total_seqs,), dtype=np.bool_)
    for seq_idx in range(total_seqs):
        raw_start = seq_idx * args.train_seq_len
        x_np = val_tokens[raw_start : raw_start + args.train_seq_len]
        raw_operator_codes = model.operator_codes_for_numpy(
            x_np.reshape(1, -1),
            route_to_next=False,
            pad_registers_for_output=False,
        )
        negation_seq_mask[seq_idx] = bool(
            raw_operator_codes is not None
            and np.any(np.asarray(raw_operator_codes, dtype=np.int32) == 1)
        )
    negation_seq_indices = np.flatnonzero(negation_seq_mask)
    affirmative_seq_indices = np.flatnonzero(~negation_seq_mask)

    results = {
        "checkpoint": str(args.checkpoint),
        "tokenizer_path": str(args.tokenizer_path),
        "data_path": str(args.data_path),
        "config": config,
        "probe": analyze_probe(model, probe_tokens, args.train_seq_len),
        "val_split": {
            "negation": subset_eval(
                model,
                val_tokens,
                args.train_seq_len,
                negation_seq_indices,
                args.val_batch_size,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            ),
            "affirmative": subset_eval(
                model,
                val_tokens,
                args.train_seq_len,
                affirmative_seq_indices,
                args.val_batch_size,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            ),
        },
    }
    output = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
