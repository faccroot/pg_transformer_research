#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as tg
import ternary_quant_mlx as trq
import turbo_quant_mlx as tq


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    flat = values.reshape(-1).astype(np.uint64, copy=False)
    planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8, copy=False)
    return np.packbits(planes.reshape(-1), bitorder="little")


def unpack_bits(blob: np.ndarray, count: int, bits: int) -> np.ndarray:
    bitstream = np.unpackbits(blob.astype(np.uint8, copy=False), bitorder="little")[: count * bits]
    planes = bitstream.reshape(count, bits).astype(np.uint64, copy=False)
    weights = (1 << np.arange(bits, dtype=np.uint64))[None, :]
    return (planes * weights).sum(axis=1).astype(np.int64, copy=False)


def parse_schemes(text: str) -> list[dict[str, object]]:
    schemes: list[dict[str, object]] = []
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        if part.startswith("int") and part[3:].isdigit():
            bits = int(part[3:])
            schemes.append({"kind": "intn", "bits": bits, "label": part})
            continue
        if part.startswith("turbo:"):
            fields = part.split(":")
            if len(fields) not in {4, 6}:
                raise ValueError(f"Bad turbo scheme {part}; expected turbo:mse_bits:prod_bits:block[:rot_seed:qjl_seed]")
            mse_bits, prod_bits, block_size = map(int, fields[1:4])
            rot_seed, qjl_seed = ((int(fields[4]), int(fields[5])) if len(fields) == 6 else (tg.TURBO_ROT_SEED, tg.TURBO_QJL_SEED))
            schemes.append(
                {
                    "kind": "turbo",
                    "mse_bits": mse_bits,
                    "prod_bits": prod_bits,
                    "block_size": block_size,
                    "rot_seed": rot_seed,
                    "qjl_seed": qjl_seed,
                    "label": part,
                }
            )
            continue
        raise ValueError(f"Unsupported scheme: {part}")
    if not schemes:
        raise ValueError("No schemes parsed")
    return schemes


def parse_pattern_list(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    patterns = tuple(part.strip() for part in text.split(",") if part.strip())
    return patterns or ()


def load_flat_state(path: Path) -> dict[str, mx.array]:
    return {name: value for name, value in mx.load(str(path)).items()}


def _natural_sort_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    out: list[object] = []
    for part in parts:
        if not part:
            continue
        out.append(int(part) if part.isdigit() else part)
    return tuple(out)


def discover_attention_blocks(flat_state: dict[str, mx.array]) -> list[dict[str, object]]:
    prefixes = sorted(
        {
            name[: -len(".attn.c_q.weight")]
            for name in flat_state
            if name.endswith(".attn.c_q.weight")
        },
        key=_natural_sort_key,
    )
    if not prefixes:
        raise ValueError("Could not infer attention block prefixes from checkpoint")

    blocks: list[dict[str, object]] = []
    for block_idx, prefix in enumerate(prefixes):
        entry = {
            "index": int(block_idx),
            "prefix": prefix,
            "q_name": f"{prefix}.attn.c_q.weight",
            "k_name": f"{prefix}.attn.c_k.weight",
            "v_name": f"{prefix}.attn.c_v.weight",
            "proj_name": f"{prefix}.attn.proj.weight",
            "q_gain_name": f"{prefix}.attn.q_gain",
            "mlp_fc_name": f"{prefix}.mlp.fc.weight",
        }
        required = ("q_name", "k_name", "q_gain_name", "mlp_fc_name")
        missing = [field for field in required if entry[field] not in flat_state]
        if missing:
            raise ValueError(
                f"Discovered attention block {prefix} is missing required tensors: "
                + ", ".join(missing)
            )
        blocks.append(entry)
    return blocks


def infer_model_config(flat_state: dict[str, mx.array]) -> dict[str, object]:
    tok_emb = flat_state["tok_emb.weight"]
    vocab_size, model_dim = map(int, tok_emb.shape)
    attention_blocks = discover_attention_blocks(flat_state)
    first_block = attention_blocks[0]
    num_layer_templates = len(attention_blocks)
    num_heads = int(flat_state[str(first_block["q_gain_name"])].size)
    head_dim = model_dim // num_heads
    num_kv_heads = int(flat_state[str(first_block["k_name"])].shape[0]) // head_dim
    mlp_mult = int(flat_state[str(first_block["mlp_fc_name"])].shape[0]) // model_dim
    tie_embeddings = "lm_head.weight" not in flat_state
    skip_weights = flat_state.get("skip_weights")
    num_layers = num_layer_templates
    if skip_weights is not None and skip_weights.ndim >= 1:
        num_layers = max(num_layers, int(skip_weights.shape[0]) * 2)
    return {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_mult": mlp_mult,
        "num_layer_templates": num_layer_templates,
        "num_layers": num_layers,
        "attention_blocks": attention_blocks,
        "tie_embeddings": tie_embeddings,
        "tied_embed_init_std": 0.005,
        "logit_softcap": 30.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.5,
    }


def empty_stats() -> dict[str, int]:
    return dict.fromkeys(
        (
            "param_count",
            "baseline_tensor_bytes",
            "payload_bytes",
            "passthrough_payload_bytes",
            "quantized_payload_bytes",
            "scale_bytes",
            "num_quantized_tensors",
            "num_passthrough_tensors",
        ),
        0,
    )


def quantize_float_array_intn(arr: mx.array, bits: int) -> tuple[np.ndarray, np.ndarray]:
    qmax = (1 << (bits - 1)) - 1
    f32 = tg._np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), tg.INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / max(qmax, 1), 1.0 / max(qmax, 1)).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -qmax, qmax).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(np.float16, copy=False))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), tg.INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / qmax if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -qmax, qmax).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_intn(flat_state: dict[str, mx.array], bits: int) -> tuple[dict[str, object], dict[str, int]]:
    qmax = (1 << (bits - 1)) - 1
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = empty_stats()
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["num_passthrough_tensors"] += 1
            stats["passthrough_payload_bytes"] += int(passthrough[name].nbytes)
            stats["payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if int(arr.size) <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = tg.keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_passthrough_tensors"] += 1
            stats["passthrough_payload_bytes"] += int(kept.nbytes)
            stats["payload_bytes"] += int(kept.nbytes)
            continue
        q, s = quantize_float_array_intn(arr, bits)
        offset = np.ascontiguousarray((q.astype(np.int16, copy=False) + qmax).astype(np.int64, copy=False))
        packed = np.ascontiguousarray(pack_bits(offset, bits))
        quantized[name] = packed
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        qmeta[name] = {"scheme": "per_row_packed" if q.ndim == 2 else "scalar_packed", "axis": 0, "bits": bits, "shape": tuple(q.shape), "count": int(q.size), "qmax": qmax}
        stats["num_quantized_tensors"] += 1
        stats["quantized_payload_bytes"] += int(packed.nbytes)
        stats["scale_bytes"] += int(s.nbytes)
        stats["payload_bytes"] += int(packed.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": f"int{bits}_clean_packed_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_intn(obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = obj["qmeta"]
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj["quantized"].items():
        meta = qmeta[name]
        bits = int(meta["bits"])
        shape = tuple(meta["shape"])
        count = int(meta["count"])
        qmax = int(meta["qmax"])
        unpacked = unpack_bits(np.asarray(packed), count, bits).reshape(shape).astype(np.int16, copy=False) - qmax
        q_np = unpacked.astype(np.float32, copy=False)
        scale = np.asarray(obj["scales"][name], dtype=np.float32)
        if meta["scheme"] == "per_row_packed" or scale.ndim > 0:
            out_arr = q_np * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np * float(scale)
        out[name] = mx.array(out_arr, dtype=tg.MX_DTYPE_FROM_NAME[obj["dtypes"][name]])
    for name, arr in obj["passthrough"].items():
        orig_dtype = passthrough_orig_dtypes.get(name)
        out[name] = mx.array(np.array(arr, copy=True), dtype=tg.MX_DTYPE_FROM_NAME[orig_dtype]) if isinstance(orig_dtype, str) else mx.array(np.array(arr, copy=True))
    return out


def resolve_turbo_patterns(
    flat_state: dict[str, mx.array],
    embed_export: bool,
    mse_patterns: tuple[str, ...] | None,
    prod_patterns: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if mse_patterns is None:
        mse_list = ["attn.c_q.weight", "attn.c_v.weight", "attn.proj.weight", "mlp.fc.weight", "mlp.proj.weight"]
    else:
        mse_list = list(mse_patterns)
    if prod_patterns is None:
        prod_list = ["attn.c_k.weight"]
        if "lm_head.weight" in flat_state:
            prod_list.append("lm_head.weight")
    else:
        prod_list = list(prod_patterns)
    if embed_export:
        if "lm_head.weight" in flat_state:
            if "tok_emb.weight" not in mse_list:
                mse_list.append("tok_emb.weight")
        elif "tok_emb.weight" not in prod_list:
            prod_list.append("tok_emb.weight")
    return tuple(mse_list), tuple(prod_list)


@contextmanager
def turbo_config(
    flat_state: dict[str, mx.array],
    mse_bits: int,
    prod_bits: int,
    block_size: int,
    rot_seed: int,
    qjl_seed: int,
    embed_export: bool,
    mse_patterns: tuple[str, ...] | None = None,
    prod_patterns: tuple[str, ...] | None = None,
):
    old = {
        "TURBO_BLOCK_SIZE": tg.TURBO_BLOCK_SIZE,
        "TURBO_MSE_BITS": tg.TURBO_MSE_BITS,
        "TURBO_PROD_BITS": tg.TURBO_PROD_BITS,
        "TURBO_ROT_SEED": tg.TURBO_ROT_SEED,
        "TURBO_QJL_SEED": tg.TURBO_QJL_SEED,
        "TURBO_MSE_NAME_PATTERNS": tg.TURBO_MSE_NAME_PATTERNS,
        "TURBO_PROD_NAME_PATTERNS": tg.TURBO_PROD_NAME_PATTERNS,
    }
    mse_patterns, prod_patterns = resolve_turbo_patterns(flat_state, embed_export, mse_patterns, prod_patterns)
    tg.configure_turbo_quant(
        block_size=block_size,
        mse_bits=mse_bits,
        prod_bits=prod_bits,
        rot_seed=rot_seed,
        qjl_seed=qjl_seed,
        mse_patterns=mse_patterns,
        prod_patterns=prod_patterns,
    )
    tg.TURBO_BLOCK_SIZE = block_size
    tg.TURBO_MSE_BITS = mse_bits
    tg.TURBO_PROD_BITS = prod_bits
    tg.TURBO_ROT_SEED = rot_seed
    tg.TURBO_QJL_SEED = qjl_seed
    tg.TURBO_MSE_NAME_PATTERNS = mse_patterns
    tg.TURBO_PROD_NAME_PATTERNS = prod_patterns
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(tg, key, value)
        tg.sync_turbo_quant_globals()


def accumulate_error(acc: dict[str, float], orig: np.ndarray, deq: np.ndarray) -> None:
    diff = deq - orig
    acc["count"] += float(orig.size)
    acc["sum_sq"] += float(np.square(diff, dtype=np.float64).sum())
    acc["sum_abs"] += float(np.abs(diff, dtype=np.float64).sum())
    acc["dot"] += float(np.multiply(orig, deq, dtype=np.float64).sum())
    acc["orig_sq"] += float(np.square(orig, dtype=np.float64).sum())
    acc["deq_sq"] += float(np.square(deq, dtype=np.float64).sum())


def finalize_error(acc: dict[str, float]) -> dict[str, float]:
    denom = max(acc["count"], 1.0)
    return {
        "rmse": math.sqrt(acc["sum_sq"] / denom),
        "mae": acc["sum_abs"] / denom,
        "cosine": acc["dot"] / max(math.sqrt(acc["orig_sq"] * acc["deq_sq"]), 1e-12),
    }


def group_name(name: str) -> str:
    if "attn.c_q.weight" in name or "attn.c_k.weight" in name:
        return "qk"
    if "tok_emb.weight" in name:
        return "embed"
    if "lm_head.weight" in name:
        return "head"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name:
        return "attn_other"
    return "other"


def compute_metrics(orig_state: dict[str, mx.array], deq_state: dict[str, mx.array]) -> dict[str, object]:
    grouped: dict[str, dict[str, float]] = {"all": dict(count=0.0, sum_sq=0.0, sum_abs=0.0, dot=0.0, orig_sq=0.0, deq_sq=0.0)}
    qk_gram: dict[str, dict[str, float]] = {}
    for name, orig in orig_state.items():
        if name not in deq_state or not mx.issubdtype(orig.dtype, mx.floating):
            continue
        orig_np = np.asarray(orig.astype(mx.float32), dtype=np.float32)
        deq_np = np.asarray(deq_state[name].astype(mx.float32), dtype=np.float32)
        grp = group_name(name)
        grouped.setdefault(grp, dict(count=0.0, sum_sq=0.0, sum_abs=0.0, dot=0.0, orig_sq=0.0, deq_sq=0.0))
        accumulate_error(grouped["all"], orig_np, deq_np)
        accumulate_error(grouped[grp], orig_np, deq_np)
        if grp == "qk" and orig_np.ndim == 2:
            diff = deq_np @ deq_np.T - orig_np @ orig_np.T
            qk_gram[name] = {
                "bias": float(diff.mean(dtype=np.float64)),
                "rmse": float(np.sqrt(np.square(diff, dtype=np.float64).mean())),
                "mae": float(np.abs(diff, dtype=np.float64).mean()),
            }
    return {
        "weight_error": {name: finalize_error(acc) for name, acc in grouped.items()},
        "qk_gram": qk_gram,
    }


def build_eval_context(
    config: dict[str, object],
    data_path: Path,
    tokenizer_path: Path,
    train_seq_len: int,
    val_max_seqs: int,
    val_batch_size: int,
    eval_seq_len: int,
    eval_stride: int,
    eval_batch_seqs: int,
) -> dict[str, object]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError("Tokenizer vocab mismatch")
    effective_eval_seq_len = eval_seq_len if eval_seq_len > 0 else train_seq_len
    val_tokens = tg.limit_validation_tokens(
        tg.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), max(train_seq_len, effective_eval_seq_len)),
        effective_eval_seq_len,
        val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(sp, int(config["vocab_size"]))
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
    effective_eval_batch_seqs = (
        eval_batch_seqs
        if eval_batch_seqs > 0
        else max(max(val_batch_size // 8, effective_eval_seq_len) // effective_eval_seq_len, 1)
    )
    args = SimpleNamespace(
        train_seq_len=train_seq_len,
        eval_seq_len=eval_seq_len,
        eval_stride=eval_stride,
        eval_batch_seqs=eval_batch_seqs,
        effective_eval_seq_len=effective_eval_seq_len,
        effective_eval_batch_seqs=effective_eval_batch_seqs,
        grad_accum_steps=8,
        val_batch_size=val_batch_size,
    )
    return {
        "args": args,
        "model": model,
        "val_tokens": val_tokens,
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space_lut": has_leading_space_lut,
        "is_boundary_token_lut": is_boundary_token_lut,
    }


def eval_state(eval_ctx: dict[str, object], flat_state: dict[str, mx.array]) -> tuple[float, float]:
    model = eval_ctx["model"]
    model.clear_turbo_cache()
    model.set_turbo_qat(False, 0.0)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()
    return tg.eval_val(
        eval_ctx["args"],
        model,
        lambda x, y, operator_codes=None: model.loss(x, y, operator_codes),
        lambda x, operator_codes=None: model.forward_logits(x, operator_codes),
        eval_ctx["val_tokens"],
        eval_ctx["base_bytes_lut"],
        eval_ctx["has_leading_space_lut"],
        eval_ctx["is_boundary_token_lut"],
    )


def analyze_scheme(
    flat_state: dict[str, mx.array],
    scheme: dict[str, object],
    eval_ctx: dict[str, object] | None,
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None = None,
    turbo_prod_patterns: tuple[str, ...] | None = None,
) -> dict[str, object]:
    quant_obj, stats, deq_state = realize_scheme(
        flat_state,
        scheme,
        turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    return summarize_realized_scheme(flat_state, scheme, quant_obj, stats, deq_state, eval_ctx)


def realize_scheme(
    flat_state: dict[str, mx.array],
    scheme: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns: tuple[str, ...] | None = None,
    turbo_prod_patterns: tuple[str, ...] | None = None,
) -> tuple[object, dict[str, object], dict[str, mx.array]]:
    if scheme["kind"] == "intn":
        quant_obj, stats = quantize_state_dict_intn(flat_state, int(scheme["bits"]))
        deq_state = dequantize_state_dict_intn(quant_obj)
    else:
        with turbo_config(
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
            deq_state = tg.dequantize_state_dict(quant_obj)
    return quant_obj, stats, deq_state


def summarize_realized_scheme(
    flat_state: dict[str, mx.array],
    scheme: dict[str, object],
    quant_obj: object,
    stats: dict[str, object],
    deq_state: dict[str, mx.array],
    eval_ctx: dict[str, object] | None,
) -> dict[str, object]:
    metrics = compute_metrics(flat_state, deq_state)
    result: dict[str, object] = {
        "scheme": dict(scheme),
        "bytes": summarize_quant_bytes(quant_obj, stats),
        "stats": stats,
        "metrics": metrics,
    }
    if eval_ctx is not None:
        val_loss, val_bpb = eval_state(eval_ctx, deq_state)
        result["eval"] = {"val_loss": float(val_loss), "val_bpb": float(val_bpb)}
    return result


def summarize_quant_bytes(quant_obj: object, stats: dict[str, object]) -> dict[str, object]:
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    payload_bytes = int(stats.get("payload_bytes", stats.get("int8_payload_bytes", 0)))
    param_count = max(int(stats.get("param_count", 1)), 1)
    return {
        "baseline_tensor_bytes": int(stats.get("baseline_tensor_bytes", 0)),
        "payload_bytes": payload_bytes,
        "raw_pickle_bytes": len(quant_raw),
        "zlib_bytes": len(quant_blob),
        "payload_bpc": float(8.0 * payload_bytes / param_count),
        "zlib_bpc": float(8.0 * len(quant_blob) / param_count),
    }


def turbo_meta_payload_breakdown(meta: dict[str, object]) -> dict[str, int]:
    scheme = str(meta.get("scheme", ""))
    if scheme == "turbo_sliced_rows_v1":
        total = {"payload_bytes": 0, "norm_bytes": 0, "qjl_bytes": 0}
        for part in meta.get("parts", []):
            child = turbo_meta_payload_breakdown(part)
            for key in total:
                total[key] += int(child[key])
        return total
    norm_bytes = int(np.asarray(meta["norms"]).nbytes)
    idx_bytes = int(np.asarray(meta["idx_packed"]).nbytes)
    qjl_bytes = 0
    if "residual_norms" in meta:
        norm_bytes += int(np.asarray(meta["residual_norms"]).nbytes)
    if "qjl_packed" in meta:
        qjl_bytes += int(np.asarray(meta["qjl_packed"]).nbytes)
    return {
        "payload_bytes": int(norm_bytes + idx_bytes + qjl_bytes),
        "norm_bytes": int(norm_bytes),
        "qjl_bytes": int(qjl_bytes),
    }


def dequantize_turbo_meta(
    meta: dict[str, object],
    codebook_override: np.ndarray | dict[tuple[str, int, int], np.ndarray] | None = None,
) -> mx.array:
    scheme = str(meta.get("scheme", ""))
    if scheme == "turbo_sliced_rows_v1":
        parts = [dequantize_turbo_meta(part_meta, codebook_override) for part_meta in meta.get("parts", [])]
        if not parts:
            shape = tuple(int(x) for x in meta.get("shape", (0, 0)))
            dtype_name = str(meta.get("dtype", "float32"))
            return mx.zeros(shape, dtype=tg.MX_DTYPE_FROM_NAME[dtype_name])
        return mx.concatenate(parts, axis=0).astype(tg.MX_DTYPE_FROM_NAME[str(meta.get("dtype", "float32"))])
    mode = str(meta["mode"])
    total_bits = int(meta["bits"])
    block_size = int(meta["block_size"])
    mse_bits = int(meta.get("mse_bits", total_bits if mode == "mse" else total_bits - 1))
    resolved_override = codebook_override
    if isinstance(codebook_override, dict):
        resolved_override = codebook_override.get((mode, mse_bits, block_size))
    if resolved_override is None:
        return tg.dequantize_turbo_tensor(meta)
    count = int(meta["count"])
    rot_seed = int(meta.get("rot_seed", tg.TURBO_ROT_SEED))
    qjl_seed = int(meta.get("qjl_seed", tg.TURBO_QJL_SEED))
    idx = unpack_bits(np.asarray(meta["idx_packed"]), count, mse_bits)
    codebook = np.asarray(resolved_override, dtype=np.float32)
    rotated = codebook[idx].reshape(-1, block_size)
    deq = np.asarray(tq.inverse_rotate_blocks_mx(mx.array(rotated), block_size, rot_seed).astype(mx.float32), dtype=np.float32)
    if mode == "prod":
        qjl_bits = unpack_bits(np.asarray(meta["qjl_packed"]), count, 1).reshape(-1, block_size)
        qjl = np.where(qjl_bits > 0, 1.0, -1.0).astype(np.float32, copy=False)
        gamma = np.asarray(meta["residual_norms"], dtype=np.float16).astype(np.float32, copy=False).reshape(-1, 1)
        deq = deq + math.sqrt(math.pi / 2.0) / block_size * gamma * (qjl @ tq._gaussian_np(block_size, qjl_seed))
    norms = np.asarray(meta["norms"], dtype=np.float16).astype(np.float32, copy=False).reshape(-1, 1)
    rows, row_dim = map(int, meta["shape"])
    pad = int(meta["pad"])
    out = np.ascontiguousarray((deq * norms).reshape(rows, row_dim + pad)[:, :row_dim])
    return mx.array(out, dtype=tg.MX_DTYPE_FROM_NAME[str(meta["dtype"])])


def dequantize_quant_obj(
    quant_obj: dict[str, object],
    codebook_overrides: dict[tuple[str, int, int], np.ndarray] | None = None,
    codebook_overrides_by_tensor: dict[str, dict[tuple[str, int, int], np.ndarray]] | None = None,
) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj.get("quantized", {}).items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=tg.MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj.get("passthrough", {}).items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=tg.MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    for name, meta in quant_obj.get("turbo", {}).items():
        mode = str(meta.get("mode", ""))
        total_bits = int(meta.get("bits", 0))
        block_size = int(meta.get("block_size", tg.TURBO_BLOCK_SIZE))
        mse_bits = int(meta.get("mse_bits", total_bits if mode == "mse" else total_bits - 1))
        override_key = (mode, mse_bits, block_size)
        override_map: dict[tuple[str, int, int], np.ndarray] = {}
        if codebook_overrides is not None:
            override_map.update(codebook_overrides)
        if codebook_overrides_by_tensor is not None:
            per_tensor = codebook_overrides_by_tensor.get(name)
            if per_tensor is not None:
                override_map.update(per_tensor)
        override = None if not override_map else override_map
        if override is not None and override_key not in override_map and str(meta.get("scheme", "")) != "turbo_sliced_rows_v1":
            override = None
        out[name] = dequantize_turbo_meta(meta, override)
    for name, meta in quant_obj.get("ternary", {}).items():
        out[name] = trq.dequantize_ternary_tensor(meta)
    return out


def quantize_turbo_tensor_row_slices(
    arr: mx.array,
    row_schemes: list[dict[str, object]],
    *,
    block_size: int,
) -> tuple[dict[str, object], mx.array, dict[str, int]]:
    parts: list[dict[str, object]] = []
    deq_parts: list[mx.array] = []
    payload_bytes = 0
    norm_bytes = 0
    qjl_bytes = 0
    for part in row_schemes:
        row_start = int(part["row_start"])
        row_end = int(part["row_end"])
        mode = str(part["mode"])
        bits = int(part["bits"])
        deq_np, meta = tg.turbo_quantize_dequantize_array(
            arr[row_start:row_end],
            mode=mode,
            total_bits=bits,
            block_size=block_size,
        )
        child = dict(meta)
        child["row_start"] = row_start
        child["row_end"] = row_end
        parts.append(child)
        deq_parts.append(mx.array(deq_np, dtype=arr.dtype))
        breakdown = turbo_meta_payload_breakdown(child)
        payload_bytes += int(breakdown["payload_bytes"])
        norm_bytes += int(breakdown["norm_bytes"])
        qjl_bytes += int(breakdown["qjl_bytes"])
    meta = {
        "scheme": "turbo_sliced_rows_v1",
        "axis": 0,
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype).split(".")[-1],
        "parts": parts,
    }
    deq = mx.concatenate(deq_parts, axis=0) if deq_parts else mx.zeros_like(arr)
    return meta, deq, {
        "payload_bytes": int(payload_bytes),
        "norm_bytes": int(norm_bytes),
        "qjl_bytes": int(qjl_bytes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline MLX quantization export analysis for an existing raw .npz checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Raw MLX .npz checkpoint")
    parser.add_argument("--schemes", default="int8,int6,turbo:2:3:256,turbo:3:4:256")
    parser.add_argument("--turbo-embed-export", action="store_true", help="For untied checkpoints, quantize tok_emb.weight with Turbo MSE")
    parser.add_argument("--turbo-mse-patterns", help="Optional comma-separated name substrings for Turbo MSE tensors")
    parser.add_argument("--turbo-prod-patterns", help="Optional comma-separated name substrings for Turbo prod tensors")
    parser.add_argument("--data-path", type=Path, help="Dataset root for optional capped eval")
    parser.add_argument("--tokenizer-path", type=Path, help="SentencePiece model for optional capped eval")
    parser.add_argument("--val-max-seqs", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = load_flat_state(args.checkpoint)
    config = infer_model_config(flat_state)
    schemes = parse_schemes(args.schemes)
    turbo_mse_patterns = parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = parse_pattern_list(args.turbo_prod_patterns)
    eval_ctx = None
    if args.data_path and args.tokenizer_path:
        eval_ctx = build_eval_context(
            config,
            args.data_path,
            args.tokenizer_path,
            args.train_seq_len,
            args.val_max_seqs,
            args.val_batch_size,
            args.eval_seq_len,
            args.eval_stride,
            args.eval_batch_seqs,
        )
    results = {
        "checkpoint": str(args.checkpoint),
        "config": config,
        "schemes": [
            analyze_scheme(
                flat_state,
                scheme,
                eval_ctx,
                args.turbo_embed_export,
                turbo_mse_patterns=turbo_mse_patterns,
                turbo_prod_patterns=turbo_prod_patterns,
            )
            for scheme in schemes
        ],
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
