#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as tg


def parse_configs(text: str) -> list[dict[str, int | bool]]:
    configs: list[dict[str, int | bool]] = []
    for part in (item.strip() for item in text.split(",")):
        if not part:
            continue
        fields = part.split(":")
        if len(fields) not in {5, 6}:
            raise ValueError(
                f"Bad config {part!r}; expected layers:dim:heads:kv_heads:mlp_mult[:tie_embeddings]"
            )
        layers, dim, heads, kv_heads, mlp_mult = map(int, fields[:5])
        tie_embeddings = bool(int(fields[5])) if len(fields) == 6 else False
        configs.append(
            {
                "num_layers": layers,
                "num_layer_templates": layers,
                "model_dim": dim,
                "num_heads": heads,
                "num_kv_heads": kv_heads,
                "mlp_mult": mlp_mult,
                "tie_embeddings": tie_embeddings,
            }
        )
    if not configs:
        raise ValueError("No configs provided")
    return configs


def summarize_config(cfg: dict[str, int | bool]) -> str:
    return (
        f"L{cfg['num_layers']}_D{cfg['model_dim']}_H{cfg['num_heads']}"
        f"_KV{cfg['num_kv_heads']}_MLP{cfg['mlp_mult']}_T{int(bool(cfg['tie_embeddings']))}"
    )


def estimate_config(cfg: dict[str, int | bool]) -> dict[str, object]:
    model = tg.GPT(
        vocab_size=tg.Hyperparameters.vocab_size,
        num_layers=int(cfg["num_layers"]),
        num_layer_templates=int(cfg["num_layer_templates"]),
        dim=int(cfg["model_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        mlp_mult=int(cfg["mlp_mult"]),
        tie_embeddings=bool(cfg["tie_embeddings"]),
        logit_chunk_tokens=tg.Hyperparameters.logit_chunk_tokens,
        logit_softcap=tg.Hyperparameters.logit_softcap,
        rope_base=tg.Hyperparameters.rope_base,
        tied_embed_init_std=tg.Hyperparameters.tied_embed_init_std,
        qk_gain_init=tg.Hyperparameters.qk_gain_init,
    )
    flat_state = tg.exportable_flat_state(model)
    param_count = sum(int(arr.size) for arr in flat_state.values())
    baseline_tensor_bytes = sum(int(arr.nbytes) for arr in flat_state.values())
    quant_obj, stats, quant_raw, quant_blob = tg.serialize_quantized_state_dict(flat_state)
    payload_bytes = int(stats["int8_payload_bytes"])
    zlib_bytes = len(quant_blob)
    return {
        "label": summarize_config(cfg),
        "config": cfg,
        "param_count": param_count,
        "baseline_tensor_bytes": baseline_tensor_bytes,
        "quant_format": tg.QUANT_FORMAT,
        "turbo_config": quant_obj.get("turbo_config"),
        "bytes": {
            "payload_bytes": payload_bytes,
            "raw_pickle_bytes": len(quant_raw),
            "zlib_bytes": zlib_bytes,
            "payload_bpc": 8.0 * payload_bytes / max(param_count, 1),
            "zlib_bpc": 8.0 * zlib_bytes / max(param_count, 1),
            "artifact_headroom_16mib": 16 * 1024 * 1024 - zlib_bytes,
            "artifact_headroom_15mib": 15 * 1024 * 1024 - zlib_bytes,
        },
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        required=True,
        help="Comma-separated layers:dim:heads:kv_heads:mlp_mult[:tie_embeddings] entries",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "quant_format": tg.QUANT_FORMAT,
        "defaults": {
            "turbo_block_size": tg.TURBO_BLOCK_SIZE,
            "turbo_mse_bits": tg.TURBO_MSE_BITS,
            "turbo_prod_bits": tg.TURBO_PROD_BITS,
            "turbo_rot_seed": tg.TURBO_ROT_SEED,
            "turbo_qjl_seed": tg.TURBO_QJL_SEED,
            "turbo_mse_name_patterns": tg.TURBO_MSE_NAME_PATTERNS,
            "turbo_prod_name_patterns": tg.TURBO_PROD_NAME_PATTERNS,
            "turbo_embed_export": tg.TURBO_EMBED_EXPORT,
            "vocab_size": tg.Hyperparameters.vocab_size,
        },
        "configs": [estimate_config(cfg) for cfg in parse_configs(args.configs)],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
