#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt as tg


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_raw_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "__quant_format__" in obj:
        raise ValueError(f"{path} is already a quantized artifact; pass a raw .pt state_dict instead")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state_dict-like dict in {path}, got {type(obj).__name__}")

    out: dict[str, torch.Tensor] = {}
    for name, tensor in obj.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            continue
        clean_name = name.removeprefix("module.")
        out[clean_name] = tensor.detach().cpu().contiguous()
    if not out:
        raise ValueError(f"No tensors found in checkpoint: {path}")
    return out


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    tok_emb = state_dict["tok_emb.weight"]
    vocab_size, model_dim = (int(tok_emb.shape[0]), int(tok_emb.shape[1]))
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in state_dict
            if name.startswith("blocks.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
        }
    )
    if not block_ids:
        raise ValueError("Could not infer block structure from checkpoint")
    num_layer_templates = max(block_ids) + 1
    num_heads = int(state_dict["blocks.0.attn.q_gain"].numel())
    head_dim = model_dim // num_heads
    num_kv_heads = int(state_dict["blocks.0.attn.c_k.weight"].shape[0]) // head_dim
    mlp_mult = int(state_dict["blocks.0.mlp.fc.weight"].shape[0]) // model_dim
    tie_embeddings = "lm_head.weight" not in state_dict
    skip_weights = state_dict.get("skip_weights")
    min_num_layers = num_layer_templates
    num_layers_ambiguous = False
    if isinstance(skip_weights, torch.Tensor) and skip_weights.ndim >= 1:
        min_num_layers = max(min_num_layers, int(skip_weights.shape[0]) * 2)
        num_layers_ambiguous = num_layer_templates < (int(skip_weights.shape[0]) * 2 + 1)
    return {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_mult": mlp_mult,
        "num_layer_templates": num_layer_templates,
        "num_layers_min": min_num_layers,
        "num_layers_ambiguous": num_layers_ambiguous,
        "tie_embeddings": tie_embeddings,
    }


def resolve_model_config(args: argparse.Namespace, inferred: dict[str, object]) -> dict[str, object]:
    config = dict(inferred)
    for key in (
        "vocab_size",
        "model_dim",
        "num_heads",
        "num_kv_heads",
        "mlp_mult",
        "num_layer_templates",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["num_layers"] = inferred["num_layers_min"] if args.num_layers is None else args.num_layers
    config["tie_embeddings"] = inferred["tie_embeddings"] if args.tie_embeddings is None else bool(args.tie_embeddings)
    config["logit_softcap"] = args.logit_softcap
    config["rope_base"] = args.rope_base
    config["tied_embed_init_std"] = args.tied_embed_init_std
    config["qk_gain_init"] = args.qk_gain_init
    if int(config["num_layer_templates"]) > int(config["num_layers"]):
        raise ValueError(
            f"num_layer_templates={config['num_layer_templates']} exceeds num_layers={config['num_layers']}"
        )
    return config


@contextmanager
def quantization_config(
    quant_format: str,
    clip_percentile: float,
    keep_float_max_numel: int,
    keep_float_fp32_patterns: tuple[str, ...] | None,
    turbo_block_size: int,
    turbo_mse_bits: int,
    turbo_prod_bits: int,
):
    old_values = {
        "QUANT_FORMAT": tg.QUANT_FORMAT,
        "INT8_CLIP_PERCENTILE": tg.INT8_CLIP_PERCENTILE,
        "INT8_CLIP_Q": tg.INT8_CLIP_Q,
        "INT8_KEEP_FLOAT_MAX_NUMEL": tg.INT8_KEEP_FLOAT_MAX_NUMEL,
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS": tg.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
        "TURBO_BLOCK_SIZE": tg.TURBO_BLOCK_SIZE,
        "TURBO_MSE_BITS": tg.TURBO_MSE_BITS,
        "TURBO_PROD_BITS": tg.TURBO_PROD_BITS,
    }
    tg.QUANT_FORMAT = quant_format
    tg.INT8_CLIP_PERCENTILE = clip_percentile
    tg.INT8_CLIP_Q = clip_percentile / 100.0
    tg.INT8_KEEP_FLOAT_MAX_NUMEL = keep_float_max_numel
    tg.TURBO_BLOCK_SIZE = turbo_block_size
    tg.TURBO_MSE_BITS = turbo_mse_bits
    tg.TURBO_PROD_BITS = turbo_prod_bits
    if keep_float_fp32_patterns is not None:
        tg.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = keep_float_fp32_patterns
    tg.sync_turbo_quant_globals()
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(tg, key, value)
        tg.sync_turbo_quant_globals()


def build_eval_context(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    if args.data_path is None or args.tokenizer_path is None:
        return {}

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device_name = "cuda" if device_name == "auto" else device_name
    device = torch.device(device_name)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    if int(sp.vocab_size()) != int(config["vocab_size"]):
        raise ValueError(
            f"Checkpoint vocab_size={config['vocab_size']} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    val_tokens = tg.load_validation_tokens(str(Path(args.data_path) / "fineweb_val_*.bin"), args.train_seq_len)
    val_tokens = tg.limit_validation_tokens(val_tokens, args.train_seq_len, args.val_max_seqs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp,
        int(config["vocab_size"]),
        device,
    )
    model = tg.GPT(
        vocab_size=int(config["vocab_size"]),
        num_layers=int(config["num_layers"]),
        num_layer_templates=int(config["num_layer_templates"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        mlp_mult=int(config["mlp_mult"]),
        tie_embeddings=bool(config["tie_embeddings"]),
        tied_embed_init_std=float(config["tied_embed_init_std"]),
        logit_softcap=float(config["logit_softcap"]),
        rope_base=float(config["rope_base"]),
        qk_gain_init=float(config["qk_gain_init"]),
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, (tg.CastedLinear, tg.TurboLinear)):
            module.float()
    tg.restore_low_dim_params_to_fp32(model)
    eval_args = SimpleNamespace(train_seq_len=args.train_seq_len)
    return {
        "device": device,
        "model": model,
        "eval_args": eval_args,
        "val_tokens": val_tokens,
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space_lut": has_leading_space_lut,
        "is_boundary_token_lut": is_boundary_token_lut,
    }


def eval_state_dict(
    eval_ctx: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    batch_tokens: int,
) -> tuple[float, float]:
    model = eval_ctx["model"]
    model.load_state_dict(state_dict, strict=True)
    return tg.eval_val_single_process(
        eval_ctx["eval_args"],
        model,
        eval_ctx["device"],
        batch_tokens,
        eval_ctx["val_tokens"],
        eval_ctx["base_bytes_lut"],
        eval_ctx["has_leading_space_lut"],
        eval_ctx["is_boundary_token_lut"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep int8 quantization settings against a raw parameter-golf PyTorch checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a raw PyTorch state_dict checkpoint (.pt)")
    parser.add_argument("--out", type=Path, help="Optional JSONL output path")
    parser.add_argument("--clip-percentiles", default="99.999,99.99984,99.99995")
    parser.add_argument("--keep-float-max-numels", default="16384,65536,262144")
    parser.add_argument("--quant-format", choices=("int8_clean_per_row_v1", "turbo_block_v1"), default="int8_clean_per_row_v1")
    parser.add_argument(
        "--keep-float-fp32-patterns",
        default=",".join(tg.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS),
        help="Comma-separated tensor-name substrings that should stay floating-point",
    )
    parser.add_argument("--turbo-block-size", type=int, default=tg.TURBO_BLOCK_SIZE)
    parser.add_argument("--turbo-mse-bits", type=int, default=tg.TURBO_MSE_BITS)
    parser.add_argument("--turbo-prod-bits", type=int, default=tg.TURBO_PROD_BITS)
    parser.add_argument("--data-path", type=Path, help="Dataset root for optional validation evals")
    parser.add_argument("--tokenizer-path", type=Path, help="SentencePiece .model path for optional validation evals")
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-batch-tokens", type=int, default=131_072)
    parser.add_argument("--val-max-seqs", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-layer-templates", type=int)
    parser.add_argument("--model-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--mlp-mult", type=int)
    parser.add_argument("--tie-embeddings", type=int, choices=(0, 1))
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--tied-embed-init-std", type=float, default=0.005)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_state = load_raw_state_dict(args.checkpoint)
    inferred = infer_model_config(raw_state)
    config = resolve_model_config(args, inferred)
    patterns = tuple(part for part in args.keep_float_fp32_patterns.split(",") if part)
    clip_percentiles = parse_float_list(args.clip_percentiles)
    keep_float_max_numels = parse_int_list(args.keep_float_max_numels)
    if not clip_percentiles or not keep_float_max_numels:
        raise ValueError("At least one clip percentile and one keep-float threshold are required")

    if (args.data_path is not None or args.tokenizer_path is not None) and inferred["num_layers_ambiguous"] and args.num_layers is None:
        raise ValueError(
            "Checkpoint depth is ambiguous under shared templates; pass --num-layers for evaluation"
        )
    eval_ctx = build_eval_context(args, config)
    raw_eval = None
    if eval_ctx:
        raw_eval = eval_state_dict(eval_ctx, raw_state, args.val_batch_tokens)

    header = {
        "checkpoint": str(args.checkpoint.resolve()),
        "inferred": inferred,
        "resolved": config,
        "eval_enabled": bool(eval_ctx),
    }
    lines = [json.dumps(header, sort_keys=True)]

    for clip_percentile in clip_percentiles:
        for keep_float_max_numel in keep_float_max_numels:
            with quantization_config(
                args.quant_format,
                clip_percentile,
                keep_float_max_numel,
                patterns,
                args.turbo_block_size,
                args.turbo_mse_bits,
                args.turbo_prod_bits,
            ):
                quant_obj, quant_stats, quant_raw, quant_blob = tg.serialize_quantized_state_dict(raw_state)
                result = {
                    "clip_percentile": clip_percentile,
                    "keep_float_max_numel": keep_float_max_numel,
                    "quant_format": args.quant_format,
                    "turbo_block_size": args.turbo_block_size,
                    "turbo_mse_bits": args.turbo_mse_bits,
                    "turbo_prod_bits": args.turbo_prod_bits,
                    "int8_zlib_bytes": len(quant_blob),
                    "raw_torch_bytes": len(quant_raw),
                    "payload_bytes": int(quant_stats["int8_payload_bytes"]),
                    "baseline_tensor_bytes": int(quant_stats["baseline_tensor_bytes"]),
                    "payload_ratio": float(quant_stats["baseline_tensor_bytes"]) / max(
                        int(quant_stats["int8_payload_bytes"]),
                        1,
                    ),
                }
                if raw_eval is not None:
                    quant_eval = eval_state_dict(eval_ctx, tg.dequantize_state_dict(quant_obj), args.val_batch_tokens)
                    result.update(
                        {
                            "raw_val_loss": raw_eval[0],
                            "raw_val_bpb": raw_eval[1],
                            "quant_val_loss": quant_eval[0],
                            "quant_val_bpb": quant_eval[1],
                            "quant_gap_bpb": quant_eval[1] - raw_eval[1],
                            "val_max_seqs": args.val_max_seqs,
                        }
                    )
                lines.append(json.dumps(result, sort_keys=True))

    output = "\n".join(lines)
    print(output)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
