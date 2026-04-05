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
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import analyze_mlx_quant_export as aqe
import search_mlx_rope_gauge as srg
import train_gpt_mlx as tg


def parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(layers) if layers else None


def parse_gauge_result(path: Path | None) -> tuple[tuple[int, ...] | None, dict[int, int] | None, dict[tuple[int, int], int] | None]:
    if path is None:
        return None, None, None
    data = json.loads(path.read_text())
    final = data.get("final", {})
    layers_raw = final.get("layers", [])
    seed_by_layer_raw = final.get("seed_by_layer", {}) or {}
    seed_by_block_kv_raw = final.get("seed_by_block_kv", {}) or {}
    layers = tuple(int(x) for x in layers_raw) if layers_raw else None
    seed_by_layer = {int(k): int(v) for k, v in seed_by_layer_raw.items()} if seed_by_layer_raw else None
    seed_by_block_kv = None
    if seed_by_block_kv_raw:
        seed_by_block_kv = {}
        for key, value in seed_by_block_kv_raw.items():
            layer_idx, kv_head_idx = (int(part) for part in str(key).split(":", 1))
            seed_by_block_kv[(layer_idx, kv_head_idx)] = int(value)
    return layers, seed_by_layer, seed_by_block_kv


def build_model_from_state(flat_state: dict[str, mx.array]) -> tg.GPT:
    config = aqe.infer_model_config(flat_state)
    model = tg.GPT(
        vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_layers"]),
        num_layer_templates=int(config["num_layer_templates"]),
        dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        mlp_mult=int(config["mlp_mult"]),
        mlp_leaky_slope=float(config.get("mlp_leaky_slope", 0.5)),
        tie_embeddings=bool(config["tie_embeddings"]),
        logit_chunk_tokens=0,
        logit_softcap=float(config["logit_softcap"]),
        rope_base=float(config["rope_base"]),
        tied_embed_init_std=float(config["tied_embed_init_std"]),
        qk_gain_init=float(config["qk_gain_init"]),
    )
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()
    model.set_turbo_qat(False, 0.0)
    return model


def make_compare_state(
    flat_state: dict[str, mx.array],
    *,
    scheme: dict[str, object] | None,
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None,
    turbo_prod_patterns: tuple[str, ...] | None,
) -> tuple[dict[str, mx.array], dict[str, object] | None]:
    if scheme is None:
        return dict(flat_state), None
    if scheme["kind"] == "intn":
        quant_obj, stats = aqe.quantize_state_dict_intn(flat_state, int(scheme["bits"]))
        return aqe.dequantize_state_dict_intn(quant_obj), {"scheme": dict(scheme), "stats": stats}
    with aqe.turbo_config(
        flat_state,
        mse_bits=int(scheme["mse_bits"]),
        prod_bits=int(scheme["prod_bits"]),
        block_size=int(scheme["block_size"]),
        rot_seed=int(scheme["rot_seed"]),
        qjl_seed=int(scheme["qjl_seed"]),
        embed_export=turbo_embed_export,
        mse_patterns=turbo_mse_patterns,
        prod_patterns=turbo_prod_patterns,
    ):
        quant_obj, stats = tg.quantize_state_dict_turbo(flat_state)
        return tg.dequantize_state_dict(quant_obj), {"scheme": dict(scheme), "stats": stats}


def repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    shape = x.shape
    expanded = mx.broadcast_to(x[:, :, None, :, :], (shape[0], shape[1], repeats, shape[2], shape[3]))
    return expanded.reshape(shape[0], shape[1] * repeats, shape[2], shape[3])


def causal_softmax(logits: mx.array) -> mx.array:
    seq_len = int(logits.shape[-1])
    mask = np.triu(np.full((seq_len, seq_len), -1.0e9, dtype=np.float32), k=1)
    mask_mx = mx.array(mask, dtype=logits.dtype)
    return mx.softmax(logits + mask_mx[None, None, :, :], axis=-1)


def block_attn_observables(block: tg.Block, x_attn_in: mx.array) -> dict[str, mx.array]:
    attn = block.attn
    bsz, seqlen, dim = x_attn_in.shape
    q = attn.c_q(x_attn_in).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = attn.c_k(x_attn_in).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    v = attn.c_v(x_attn_in).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    q = attn.rope(tg.rms_norm(q).astype(tg.COMPUTE_DTYPE))
    k = attn.rope(tg.rms_norm(k).astype(tg.COMPUTE_DTYPE))
    q = q * attn.q_gain.astype(q.dtype)[None, :, None, None]
    repeat = attn.num_heads // attn.num_kv_heads
    k_rep = repeat_kv(k, repeat)
    v_rep = repeat_kv(v, repeat)
    logits = mx.matmul(q, k_rep.transpose(0, 1, 3, 2)) * attn.scale
    attn_probs = causal_softmax(logits.astype(mx.float32))
    msg = mx.matmul(attn_probs.astype(v_rep.dtype), v_rep)
    v_norm = mx.sqrt(mx.sum(v_rep.astype(mx.float32) * v_rep.astype(mx.float32), axis=-1) + 1e-12)
    denom = mx.matmul(attn_probs, v_norm[..., None]).squeeze(-1)
    msg_norm = mx.sqrt(mx.sum(msg.astype(mx.float32) * msg.astype(mx.float32), axis=-1) + 1e-12)
    coherence = msg_norm / mx.maximum(denom, mx.array(1.0e-6, dtype=msg_norm.dtype))
    return {
        "q": q.astype(mx.float32),
        "k_base": k.astype(mx.float32),
        "k_rep": k_rep.astype(mx.float32),
        "v_rep": v_rep.astype(mx.float32),
        "logits": logits.astype(mx.float32),
        "attn": attn_probs.astype(mx.float32),
        "coherence": coherence.astype(mx.float32),
    }


def init_block_accumulators(num_blocks: int, num_heads: int, num_kv_heads: int) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for _ in range(num_blocks):
        blocks.append(
            {
                "q_count": 0,
                "q_sum": np.zeros((num_heads,), dtype=np.float64),
                "q_sq_sum": np.zeros((num_heads,), dtype=np.float64),
                "k_count": 0,
                "k_sum": np.zeros((num_kv_heads,), dtype=np.float64),
                "k_sq_sum": np.zeros((num_kv_heads,), dtype=np.float64),
                "coh_count": 0,
                "coh_sum": np.zeros((num_heads,), dtype=np.float64),
                "coh_sq_sum": np.zeros((num_heads,), dtype=np.float64),
                "logit_count": 0,
                "logit_bias_sum": np.zeros((num_heads,), dtype=np.float64),
                "logit_sq_sum": np.zeros((num_heads,), dtype=np.float64),
                "attn_kl_sum": np.zeros((num_heads,), dtype=np.float64),
                "attn_kl_count": 0,
                "coh_delta_sum": np.zeros((num_heads,), dtype=np.float64),
            }
        )
    return {"blocks": blocks}


def update_norm_stats(acc: dict[str, object], layer_idx: int, obs_ref: dict[str, mx.array], obs_cmp: dict[str, mx.array]) -> None:
    block_acc = acc["blocks"][layer_idx]
    q_norm = np.asarray(mx.sqrt(mx.sum(obs_ref["q"] * obs_ref["q"], axis=-1)), dtype=np.float32)
    q_flat = q_norm.transpose(1, 0, 2).reshape(q_norm.shape[1], -1)
    block_acc["q_sum"] += q_flat.sum(axis=1, dtype=np.float64)
    block_acc["q_sq_sum"] += np.square(q_flat, dtype=np.float64).sum(axis=1)
    block_acc["q_count"] += int(q_flat.shape[1])

    k_norm = np.asarray(mx.sqrt(mx.sum(obs_ref["k_base"] * obs_ref["k_base"], axis=-1)), dtype=np.float32)
    k_flat = k_norm.transpose(1, 0, 2).reshape(k_norm.shape[1], -1)
    block_acc["k_sum"] += k_flat.sum(axis=1, dtype=np.float64)
    block_acc["k_sq_sum"] += np.square(k_flat, dtype=np.float64).sum(axis=1)
    block_acc["k_count"] += int(k_flat.shape[1])

    coh_ref = np.asarray(obs_ref["coherence"], dtype=np.float32).transpose(1, 0, 2).reshape(obs_ref["coherence"].shape[1], -1)
    coh_cmp = np.asarray(obs_cmp["coherence"], dtype=np.float32).transpose(1, 0, 2).reshape(obs_cmp["coherence"].shape[1], -1)
    block_acc["coh_sum"] += coh_ref.sum(axis=1, dtype=np.float64)
    block_acc["coh_sq_sum"] += np.square(coh_ref, dtype=np.float64).sum(axis=1)
    block_acc["coh_delta_sum"] += (coh_cmp - coh_ref).sum(axis=1, dtype=np.float64)
    block_acc["coh_count"] += int(coh_ref.shape[1])

    logits_ref = np.asarray(obs_ref["logits"], dtype=np.float32)
    logits_cmp = np.asarray(obs_cmp["logits"], dtype=np.float32)
    seq_len = logits_ref.shape[-1]
    valid = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    diff = (logits_cmp - logits_ref) * valid[None, None, :, :]
    diff_flat = diff.transpose(1, 0, 2, 3).reshape(diff.shape[1], -1)
    counts = valid.sum(dtype=np.float64) * diff.shape[0]
    block_acc["logit_bias_sum"] += diff_flat.sum(axis=1, dtype=np.float64)
    block_acc["logit_sq_sum"] += np.square(diff_flat, dtype=np.float64).sum(axis=1)
    block_acc["logit_count"] += int(counts)

    attn_ref = np.asarray(obs_ref["attn"], dtype=np.float32)
    attn_cmp = np.asarray(obs_cmp["attn"], dtype=np.float32)
    kl = attn_ref * (np.log(np.maximum(attn_ref, 1e-9)) - np.log(np.maximum(attn_cmp, 1e-9)))
    kl = (kl * valid[None, None, :, :]).sum(axis=-1)
    kl_flat = kl.transpose(1, 0, 2).reshape(kl.shape[1], -1)
    block_acc["attn_kl_sum"] += kl_flat.sum(axis=1, dtype=np.float64)
    block_acc["attn_kl_count"] += int(kl_flat.shape[1])


def finalize_block_metrics(acc: dict[str, object], num_heads: int, num_kv_heads: int) -> dict[str, object]:
    out_blocks: list[dict[str, object]] = []
    corr_pairs_x: list[float] = []
    corr_pairs_y: list[float] = []
    q_per_kv = num_heads // num_kv_heads
    for layer_idx, block_acc in enumerate(acc["blocks"]):
        q_mean = block_acc["q_sum"] / max(block_acc["q_count"], 1)
        q_var = np.maximum(block_acc["q_sq_sum"] / max(block_acc["q_count"], 1) - q_mean * q_mean, 0.0)
        q_std = np.sqrt(q_var)
        q_cv = q_std / np.maximum(q_mean, 1e-8)

        k_mean = block_acc["k_sum"] / max(block_acc["k_count"], 1)
        k_var = np.maximum(block_acc["k_sq_sum"] / max(block_acc["k_count"], 1) - k_mean * k_mean, 0.0)
        k_std = np.sqrt(k_var)
        k_cv = k_std / np.maximum(k_mean, 1e-8)

        coh_mean = block_acc["coh_sum"] / max(block_acc["coh_count"], 1)
        coh_var = np.maximum(block_acc["coh_sq_sum"] / max(block_acc["coh_count"], 1) - coh_mean * coh_mean, 0.0)
        coh_std = np.sqrt(coh_var)
        coh_delta_mean = block_acc["coh_delta_sum"] / max(block_acc["coh_count"], 1)

        logit_bias = block_acc["logit_bias_sum"] / max(block_acc["logit_count"], 1)
        logit_rmse = np.sqrt(block_acc["logit_sq_sum"] / max(block_acc["logit_count"], 1))
        attn_kl = block_acc["attn_kl_sum"] / max(block_acc["attn_kl_count"], 1)

        for q_head_idx in range(num_heads):
            corr_pairs_x.append(float(k_cv[q_head_idx // q_per_kv]))
            corr_pairs_y.append(float(logit_rmse[q_head_idx]))

        out_blocks.append(
            {
                "layer": layer_idx,
                "q_norm_mean": q_mean.tolist(),
                "q_norm_cv": q_cv.tolist(),
                "k_norm_mean": k_mean.tolist(),
                "k_norm_cv": k_cv.tolist(),
                "coherence_mean": coh_mean.tolist(),
                "coherence_std": coh_std.tolist(),
                "coherence_delta_mean": coh_delta_mean.tolist(),
                "logit_bias_mean": logit_bias.tolist(),
                "logit_rmse": logit_rmse.tolist(),
                "attn_kl_mean": attn_kl.tolist(),
            }
        )

    corr = 0.0
    if len(corr_pairs_x) >= 2:
        corr = float(np.corrcoef(np.asarray(corr_pairs_x, dtype=np.float64), np.asarray(corr_pairs_y, dtype=np.float64))[0, 1])
    return {
        "blocks": out_blocks,
        "summary": {
            "mean_k_norm_cv": float(np.mean([np.mean(block["k_norm_cv"], dtype=np.float64) for block in out_blocks])),
            "mean_q_norm_cv": float(np.mean([np.mean(block["q_norm_cv"], dtype=np.float64) for block in out_blocks])),
            "mean_coherence": float(np.mean([np.mean(block["coherence_mean"], dtype=np.float64) for block in out_blocks])),
            "mean_logit_rmse": float(np.mean([np.mean(block["logit_rmse"], dtype=np.float64) for block in out_blocks])),
            "mean_attn_kl": float(np.mean([np.mean(block["attn_kl_mean"], dtype=np.float64) for block in out_blocks])),
            "corr_k_cv_vs_logit_rmse": corr,
        },
    }


def iter_eval_batches(val_tokens: np.ndarray, seq_len: int, max_seqs: int, batch_seqs: int):
    total = min(max_seqs, len(val_tokens) // seq_len)
    trimmed = np.asarray(val_tokens[: total * seq_len], dtype=np.uint16).reshape(total, seq_len)
    for start in range(0, total, batch_seqs):
        yield trimmed[start : start + batch_seqs]


def analyze_geometry(
    ref_model: tg.GPT,
    cmp_model: tg.GPT,
    val_tokens: np.ndarray,
    seq_len: int,
    max_seqs: int,
    batch_seqs: int,
) -> dict[str, object]:
    num_blocks = ref_model.num_layers
    num_heads = ref_model.blocks[0].attn.num_heads
    num_kv_heads = ref_model.blocks[0].attn.num_kv_heads
    acc = init_block_accumulators(num_blocks, num_heads, num_kv_heads)

    for batch_np in iter_eval_batches(val_tokens, seq_len, max_seqs, batch_seqs):
        input_ids = mx.array(batch_np.astype(np.int32), dtype=mx.int32)
        operator_codes = ref_model.operator_codes_for_input(input_ids)
        x = ref_model.embed_inputs(input_ids)
        x0 = x
        skips: list[mx.array] = []
        layer_idx = 0

        for i in range(ref_model.num_encoder_layers):
            block_ref = ref_model.block_for_step(i)
            block_cmp = cmp_model.block_for_step(i)
            mix = block_ref.resid_mix.astype(x.dtype)
            x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            x_attn_in = block_ref.attn_norm(x_mixed)
            obs_ref = block_attn_observables(block_ref, x_attn_in)
            obs_cmp = block_attn_observables(block_cmp, x_attn_in)
            update_norm_stats(acc, layer_idx, obs_ref, obs_cmp)
            x = block_ref(x, x0, attn_mask=ref_model.attention_mask(x.shape[1]))
            x = ref_model.maybe_apply_logic_sidecar(x, operator_codes, layer_idx)
            skips.append(x)
            layer_idx += 1

        for i in range(ref_model.num_decoder_layers):
            if skips:
                x = x + ref_model.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            block_ref = ref_model.block_for_step(ref_model.num_encoder_layers + i)
            block_cmp = cmp_model.block_for_step(ref_model.num_encoder_layers + i)
            mix = block_ref.resid_mix.astype(x.dtype)
            x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            x_attn_in = block_ref.attn_norm(x_mixed)
            obs_ref = block_attn_observables(block_ref, x_attn_in)
            obs_cmp = block_attn_observables(block_cmp, x_attn_in)
            update_norm_stats(acc, layer_idx, obs_ref, obs_cmp)
            x = block_ref(x, x0, attn_mask=ref_model.attention_mask(x.shape[1]))
            x = ref_model.maybe_apply_logic_sidecar(x, operator_codes, layer_idx)
            layer_idx += 1

    return finalize_block_metrics(acc, num_heads, num_kv_heads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze attention-space geometry on real validation batches for an MLX checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--compare-scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-transform", choices=("none", "qk_only", "qkvo_full"), default="none")
    parser.add_argument("--gauge-parameterization", choices=("global_head_phase", "banded_phase", "full_pair_phase"), default="global_head_phase")
    parser.add_argument("--gauge-seed", type=int, default=0)
    parser.add_argument("--gauge-angle-scale", type=float, default=math.pi)
    parser.add_argument("--gauge-num-bands", type=int, default=4)
    parser.add_argument("--gauge-layers", help="Optional comma-separated block indices to transform")
    parser.add_argument("--gauge-result", type=Path, help="Optional greedy gauge-search JSON whose final map should be applied")
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--batch-seqs", type=int, default=2)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    gauge_meta = None
    gauge_layers = parse_layers(args.gauge_layers)
    gauge_result_layers, gauge_seed_by_layer, gauge_seed_by_block_kv = parse_gauge_result(args.gauge_result)
    if gauge_result_layers is not None:
        gauge_layers = gauge_result_layers
    if args.gauge_transform != "none":
        flat_state, gauge_meta = srg.apply_rope_gauge_transform(
            flat_state,
            seed=args.gauge_seed,
            angle_scale=args.gauge_angle_scale,
            transform=args.gauge_transform,
            parameterization=args.gauge_parameterization,
            num_bands=args.gauge_num_bands,
            layers=gauge_layers,
            seed_by_layer=gauge_seed_by_layer,
            seed_by_block_kv=gauge_seed_by_block_kv,
        )

    compare_scheme = None
    if args.compare_scheme.strip():
        schemes = aqe.parse_schemes(args.compare_scheme)
        if len(schemes) != 1:
            raise SystemExit("Expected exactly one compare scheme")
        compare_scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    compare_state, quant_meta = make_compare_state(
        flat_state,
        scheme=compare_scheme,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
    )

    ref_model = build_model_from_state(flat_state)
    cmp_model = build_model_from_state(compare_state)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    config = aqe.infer_model_config(flat_state)
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    val_tokens = tg.limit_validation_tokens(
        tg.load_validation_tokens(str(args.data_path / "fineweb_val_*.bin"), args.seq_len),
        args.seq_len,
        args.max_seqs,
    )
    geometry = analyze_geometry(ref_model, cmp_model, val_tokens, args.seq_len, args.max_seqs, args.batch_seqs)

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "compare_scheme": compare_scheme,
        "gauge": gauge_meta,
        "quant_meta": quant_meta,
        "seq_len": args.seq_len,
        "max_seqs": args.max_seqs,
        "batch_seqs": args.batch_seqs,
        "geometry": geometry,
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
