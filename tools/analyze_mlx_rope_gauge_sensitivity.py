#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    try:
        sys.path.remove(str(root))
    except ValueError:
        pass
for root in (REPO_ROOT, TOOLS_ROOT):
    sys.path.insert(0, str(root))

import analyze_mlx_output_proxies as aop
import analyze_mlx_quant_export as aqe
import search_mlx_rope_gauge as srg


def parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def load_band_angles_from_result(
    flat_state,
    result_path: Path,
    *,
    init_angle_scale: float,
    init_parameterization: str,
    num_bands: int,
) -> tuple[np.ndarray, dict[str, object]]:
    data = json.loads(result_path.read_text())
    final = data.get("final", {})
    band_angles = final.get("band_angles")
    if band_angles is not None:
        return np.asarray(band_angles, dtype=np.float32), {"kind": "continuous"}
    seed_by_layer_raw = final.get("seed_by_layer", {}) or {}
    seed_by_block_kv_raw = final.get("seed_by_block_kv", {}) or {}
    layers_raw = final.get("layers", [])
    layers = tuple(int(x) for x in layers_raw) if layers_raw else None
    seed_by_layer = {int(k): int(v) for k, v in seed_by_layer_raw.items()}
    seed_by_block_kv: dict[tuple[int, int], int] = {}
    for key, value in seed_by_block_kv_raw.items():
        layer_idx, kv_head_idx = (int(part) for part in str(key).split(":", 1))
        seed_by_block_kv[(layer_idx, kv_head_idx)] = int(value)
    band_tensor = srg.band_angle_tensor_from_seed_assignments(
        flat_state,
        base_seed=0,
        angle_scale=init_angle_scale,
        parameterization=init_parameterization,
        num_bands=num_bands,
        layers=layers,
        seed_by_layer=seed_by_layer,
        seed_by_block_kv=seed_by_block_kv,
    )
    return band_tensor, {
        "kind": "discrete_seed_map",
        "layers": list(layers) if layers is not None else None,
    }


def summarize_candidate(
    flat_state,
    *,
    band_angles: np.ndarray,
    transform: str,
    layers: tuple[int, ...] | None,
    scheme: dict[str, object],
    proxy_ctx: dict[str, object],
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
) -> dict[str, object]:
    gauge_state, gauge_meta = srg.apply_rope_gauge_band_angles(
        flat_state,
        band_angles=band_angles,
        transform=transform,
        layers=layers,
    )
    quant_obj, stats, deq_state = aqe.realize_scheme(
        gauge_state,
        scheme,
        turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    bytes_info = aqe.summarize_quant_bytes(quant_obj, stats)
    proxy = aop.summarize_search_proxy(
        aop.analyze_compare_state_proxies(
            proxy_ctx["ref_model"],
            deq_state,
            proxy_ctx["val_tokens"],
            int(proxy_ctx["seq_len"]),
            int(proxy_ctx["max_seqs"]),
            int(proxy_ctx["batch_seqs"]),
        )
    )
    return {"gauge": gauge_meta, "bytes": bytes_info, "output_proxy": proxy}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze layer and KV-head sensitivity for an exact RoPE gauge result.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--gauge-result", type=Path, required=True)
    parser.add_argument("--transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument("--layers", help="Optional comma-separated layer subset to analyze")
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument(
        "--init-parameterization",
        choices=("global_head_phase", "banded_phase", "full_pair_phase"),
        default="banded_phase",
    )
    parser.add_argument("--init-angle-scale", type=float, default=np.pi)
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--proxy-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-max-seqs", type=int, default=8)
    parser.add_argument("--proxy-batch-seqs", type=int, default=1)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    config = srg.gauge_config(flat_state)
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1:
        raise SystemExit("Expected exactly one scheme")
    scheme = schemes[0]
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    layers = parse_layers(args.layers)
    band_angles, init_meta = load_band_angles_from_result(
        flat_state,
        args.gauge_result,
        init_angle_scale=args.init_angle_scale,
        init_parameterization=args.init_parameterization,
        num_bands=args.num_bands,
    )
    if layers is not None:
        layer_set = set(layers)
        trimmed = np.zeros_like(band_angles)
        for layer_idx in layers:
            trimmed[layer_idx] = band_angles[layer_idx]
        band_angles = trimmed
    else:
        layers = tuple(range(config["num_blocks"]))

    proxy_ctx = aop.build_proxy_context(
        flat_state,
        args.data_path,
        args.tokenizer_path,
        args.proxy_seq_len,
        args.proxy_max_seqs,
        args.proxy_batch_seqs,
    )

    full_result = summarize_candidate(
        flat_state,
        band_angles=band_angles,
        transform=args.transform,
        layers=layers,
        scheme=scheme,
        proxy_ctx=proxy_ctx,
        turbo_embed_export=args.turbo_embed_export,
        turbo_mse_patterns=turbo_mse_patterns,
        turbo_prod_patterns=turbo_prod_patterns,
    )

    per_layer: list[dict[str, object]] = []
    for layer_idx in layers:
        ablated = np.array(band_angles, copy=True)
        ablated[layer_idx, :, :] = 0.0
        candidate = summarize_candidate(
            flat_state,
            band_angles=ablated,
            transform=args.transform,
            layers=layers,
            scheme=scheme,
            proxy_ctx=proxy_ctx,
            turbo_embed_export=args.turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
        )
        per_layer.append(
            {
                "layer": int(layer_idx),
                "mean_ce_delta": float(candidate["output_proxy"]["mean_ce_delta"]),
                "mean_margin_delta": float(candidate["output_proxy"]["mean_margin_delta"]),
                "top1_true_rate": float(candidate["output_proxy"]["top1_true_rate"]),
                "zlib_bytes": int(candidate["bytes"]["zlib_bytes"]),
                "ce_loss_from_ablation": float(
                    candidate["output_proxy"]["mean_ce_delta"] - full_result["output_proxy"]["mean_ce_delta"]
                ),
                "margin_loss_from_ablation": float(
                    full_result["output_proxy"]["mean_margin_delta"] - candidate["output_proxy"]["mean_margin_delta"]
                ),
            }
        )

    per_block_kv: list[dict[str, object]] = []
    for layer_idx in layers:
        for kv_head_idx in range(config["num_kv_heads"]):
            if not np.any(np.abs(band_angles[layer_idx, kv_head_idx]) > 1.0e-9):
                continue
            ablated = np.array(band_angles, copy=True)
            ablated[layer_idx, kv_head_idx, :] = 0.0
            candidate = summarize_candidate(
                flat_state,
                band_angles=ablated,
                transform=args.transform,
                layers=layers,
                scheme=scheme,
                proxy_ctx=proxy_ctx,
                turbo_embed_export=args.turbo_embed_export,
                turbo_mse_patterns=turbo_mse_patterns,
                turbo_prod_patterns=turbo_prod_patterns,
            )
            per_block_kv.append(
                {
                    "layer": int(layer_idx),
                    "kv_head": int(kv_head_idx),
                    "mean_ce_delta": float(candidate["output_proxy"]["mean_ce_delta"]),
                    "mean_margin_delta": float(candidate["output_proxy"]["mean_margin_delta"]),
                    "top1_true_rate": float(candidate["output_proxy"]["top1_true_rate"]),
                    "zlib_bytes": int(candidate["bytes"]["zlib_bytes"]),
                    "ce_loss_from_ablation": float(
                        candidate["output_proxy"]["mean_ce_delta"] - full_result["output_proxy"]["mean_ce_delta"]
                    ),
                    "margin_loss_from_ablation": float(
                        full_result["output_proxy"]["mean_margin_delta"] - candidate["output_proxy"]["mean_margin_delta"]
                    ),
                }
            )

    per_layer.sort(key=lambda item: (-float(item["ce_loss_from_ablation"]), -float(item["margin_loss_from_ablation"])))
    per_block_kv.sort(key=lambda item: (-float(item["ce_loss_from_ablation"]), -float(item["margin_loss_from_ablation"])))

    result = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "gauge_result": str(args.gauge_result),
        "init": init_meta,
        "layers": list(layers),
        "full": full_result,
        "per_layer": per_layer,
        "per_block_kv": per_block_kv,
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
