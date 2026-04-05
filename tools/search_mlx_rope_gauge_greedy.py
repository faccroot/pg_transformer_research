#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    try:
        sys.path.remove(str(root))
    except ValueError:
        pass
for root in (REPO_ROOT, TOOLS_ROOT):
    sys.path.insert(0, str(root))

import analyze_mlx_quant_export as aqe
import analyze_mlx_output_proxies as aop
import search_mlx_rope_gauge as srg


def parse_int_list(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(values) if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy search for exact RoPE-compatible Q/K gauge transforms using real capped eval."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scheme", default="turbo:3:4:256:17:29")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices in greedy search order")
    parser.add_argument("--unit-scope", choices=("layer", "block_kv"), default="layer")
    parser.add_argument("--kv-heads", help="Optional comma-separated KV head ids for block_kv search")
    parser.add_argument("--init-result", type=Path, help="Optional prior greedy-search JSON to initialize incumbent seeds")
    parser.add_argument("--angle-scale", type=float, default=math.pi)
    parser.add_argument("--transform", choices=("qk_only", "qkvo_full"), default="qk_only")
    parser.add_argument(
        "--parameterization",
        choices=("global_head_phase", "banded_phase", "full_pair_phase"),
        default="banded_phase",
    )
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--refine-passes", type=int, default=1, help="Number of greedy refinement passes")
    parser.add_argument("--turbo-embed-export", action="store_true")
    parser.add_argument("--turbo-mse-patterns")
    parser.add_argument("--turbo-prod-patterns")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--val-max-seqs", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=262144)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--eval-batch-seqs", type=int, default=0)
    parser.add_argument("--selection-objective", choices=("eval_bpb", "output_proxy"), default="eval_bpb")
    parser.add_argument("--proxy-seq-len", type=int, default=1024)
    parser.add_argument("--proxy-max-seqs", type=int, default=8)
    parser.add_argument("--proxy-batch-seqs", type=int, default=1)
    parser.add_argument(
        "--raw-sanity",
        action="store_true",
        help="Run float eval on the final composed transform to verify exactness within eval noise",
    )
    parser.add_argument("--out", type=Path)
    return parser


def analyze_candidate(
    flat_state,
    *,
    layers: tuple[int, ...],
    seed_by_layer: dict[int, int],
    seed_by_block_kv: dict[tuple[int, int], int],
    base_seed: int,
    angle_scale: float,
    transform: str,
    parameterization: str,
    num_bands: int,
    scheme: dict[str, object],
    eval_ctx,
    output_proxy_ctx,
    turbo_embed_export: bool,
    turbo_mse_patterns,
    turbo_prod_patterns,
) -> dict[str, object]:
    candidate_state, meta = srg.apply_rope_gauge_transform(
        flat_state,
        seed=base_seed,
        angle_scale=angle_scale,
        transform=transform,
        parameterization=parameterization,
        num_bands=num_bands,
        layers=layers,
        seed_by_layer=seed_by_layer,
        seed_by_block_kv=seed_by_block_kv,
    )
    quant_obj, stats, deq_state = aqe.realize_scheme(
        candidate_state,
        scheme,
        turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    analysis = aqe.summarize_realized_scheme(candidate_state, scheme, quant_obj, stats, deq_state, eval_ctx)
    proxy = srg.summarize_proxy_metrics(analysis["metrics"])
    output_proxy = None
    if output_proxy_ctx is not None:
        proxy_metrics = aop.analyze_compare_state_proxies(
            output_proxy_ctx["ref_model"],
            deq_state,
            output_proxy_ctx["val_tokens"],
            int(output_proxy_ctx["seq_len"]),
            int(output_proxy_ctx["max_seqs"]),
            int(output_proxy_ctx["batch_seqs"]),
        )
        output_proxy = aop.summarize_search_proxy(proxy_metrics)
    return {
        "layers": list(layers),
        "seed_by_layer": {str(k): int(v) for k, v in sorted(seed_by_layer.items())},
        "seed_by_block_kv": {
            f"{layer_idx}:{kv_head_idx}": int(v)
            for (layer_idx, kv_head_idx), v in sorted(seed_by_block_kv.items())
        },
        "meta": meta,
        "analysis": analysis,
        "proxy": proxy,
        "output_proxy": output_proxy,
    }


def selection_key(candidate: dict[str, object], objective: str) -> tuple[float, ...]:
    analysis = candidate["analysis"]
    eval_info = analysis.get("eval", {})
    val_bpb = float(eval_info.get("val_bpb", math.inf))
    zlib_bytes = int(analysis["bytes"]["zlib_bytes"])
    if objective == "output_proxy":
        output_proxy = candidate.get("output_proxy") or {}
        return (
            float(output_proxy.get("mean_ce_delta", math.inf)),
            -float(output_proxy.get("mean_margin_delta", -math.inf)),
            -float(output_proxy.get("top1_true_rate", -math.inf)),
            val_bpb,
            zlib_bytes,
            float(output_proxy.get("score", math.inf)),
        )
    return (
        val_bpb,
        zlib_bytes,
        float(candidate["proxy"]["score"]),
    )


def is_better(candidate: dict[str, object], incumbent: dict[str, object], objective: str) -> bool:
    cand_eval = candidate["analysis"]["eval"]
    inc_eval = incumbent["analysis"]["eval"]
    _ = cand_eval, inc_eval
    return selection_key(candidate, objective) < selection_key(incumbent, objective)


def parse_init_result(path: Path | None) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    if path is None:
        return {}, {}
    data = json.loads(path.read_text())
    final = data.get("final", {})
    seed_by_layer_raw = final.get("seed_by_layer", {}) or {}
    seed_by_block_kv_raw = final.get("seed_by_block_kv", {}) or {}
    seed_by_layer = {int(k): int(v) for k, v in seed_by_layer_raw.items()}
    seed_by_block_kv: dict[tuple[int, int], int] = {}
    for key, value in seed_by_block_kv_raw.items():
        layer_idx, kv_head_idx = (int(part) for part in str(key).split(":", 1))
        seed_by_block_kv[(layer_idx, kv_head_idx)] = int(value)
    return seed_by_layer, seed_by_block_kv


def main() -> None:
    args = build_parser().parse_args()
    flat_state = aqe.load_flat_state(args.checkpoint)
    config = aqe.infer_model_config(flat_state)
    schemes = aqe.parse_schemes(args.scheme)
    if len(schemes) != 1:
        raise SystemExit("Expected exactly one scheme")
    scheme = schemes[0]
    seeds = srg.parse_seeds(args.seeds)
    search_layers = srg.parse_layers(args.layers)
    if not search_layers:
        raise SystemExit("Expected at least one search layer")
    turbo_mse_patterns = aqe.parse_pattern_list(args.turbo_mse_patterns)
    turbo_prod_patterns = aqe.parse_pattern_list(args.turbo_prod_patterns)
    kv_heads = parse_int_list(args.kv_heads)
    init_seed_by_layer, init_seed_by_block_kv = parse_init_result(args.init_result)
    eval_ctx = aqe.build_eval_context(
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

    baseline_raw_loss, baseline_raw_bpb = aqe.eval_state(eval_ctx, flat_state)
    baseline_quant = aqe.analyze_scheme(
        flat_state,
        scheme,
        eval_ctx,
        args.turbo_embed_export,
        turbo_mse_patterns,
        turbo_prod_patterns,
    )
    output_proxy_ctx = None
    baseline_output_proxy = None
    if args.selection_objective == "output_proxy":
        output_proxy_ctx = aop.build_proxy_context(
            flat_state,
            args.data_path,
            args.tokenizer_path,
            args.proxy_seq_len,
            args.proxy_max_seqs,
            args.proxy_batch_seqs,
        )
        baseline_compare_state, _baseline_quant_meta = aop.make_compare_state(
            flat_state,
            scheme=scheme,
            turbo_embed_export=args.turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
        )
        baseline_output_proxy = aop.summarize_search_proxy(
            aop.analyze_compare_state_proxies(
                output_proxy_ctx["ref_model"],
                baseline_compare_state,
                output_proxy_ctx["val_tokens"],
                int(output_proxy_ctx["seq_len"]),
                int(output_proxy_ctx["max_seqs"]),
                int(output_proxy_ctx["batch_seqs"]),
            )
        )
    incumbent = {
        "layers": [],
        "seed_by_layer": {},
        "seed_by_block_kv": {},
        "meta": None,
        "analysis": baseline_quant,
        "proxy": srg.summarize_proxy_metrics(baseline_quant["metrics"]),
        "output_proxy": baseline_output_proxy,
    }
    history: list[dict[str, object]] = []
    accepted_seed_by_layer: dict[int, int] = dict(init_seed_by_layer)
    accepted_seed_by_block_kv: dict[tuple[int, int], int] = dict(init_seed_by_block_kv)
    accepted_layers: list[int] = sorted(set(accepted_seed_by_layer) | {layer_idx for layer_idx, _kv_head_idx in accepted_seed_by_block_kv})
    if accepted_seed_by_layer or accepted_seed_by_block_kv:
        incumbent = analyze_candidate(
            flat_state,
            layers=tuple(accepted_layers),
            seed_by_layer=accepted_seed_by_layer,
            seed_by_block_kv=accepted_seed_by_block_kv,
            base_seed=0,
            angle_scale=args.angle_scale,
            transform=args.transform,
            parameterization=args.parameterization,
            num_bands=args.num_bands,
            scheme=scheme,
            eval_ctx=eval_ctx,
            output_proxy_ctx=output_proxy_ctx,
            turbo_embed_export=args.turbo_embed_export,
            turbo_mse_patterns=turbo_mse_patterns,
            turbo_prod_patterns=turbo_prod_patterns,
        )
        print(
            f"[init] loaded incumbent layers={accepted_layers} eval_bpb={incumbent['analysis']['eval']['val_bpb']:.8f}",
            flush=True,
        )
    if args.unit_scope == "layer":
        units: list[int | tuple[int, int]] = list(search_layers)
    else:
        num_kv_heads = int(config["num_kv_heads"])
        chosen_kv = kv_heads if kv_heads is not None else tuple(range(num_kv_heads))
        units = [(layer_idx, kv_head_idx) for layer_idx in search_layers for kv_head_idx in chosen_kv]

    for pass_idx in range(max(args.refine_passes, 1)):
        for unit in units:
            unit_label = (
                f"block_kv:{unit[0]}:{unit[1]}"
                if isinstance(unit, tuple)
                else f"layer:{unit}"
            )
            unit_candidates: list[dict[str, object]] = []
            noop = {
                "seed": None,
                "accepted_before": (
                    unit in accepted_seed_by_block_kv if isinstance(unit, tuple) else unit in accepted_seed_by_layer
                ),
                "analysis": {
                    "eval": incumbent["analysis"]["eval"],
                    "bytes": incumbent["analysis"]["bytes"],
                },
                "proxy": incumbent["proxy"],
                "delta_bpb_vs_incumbent": 0.0,
                "delta_zlib_vs_incumbent": 0,
                "would_accept": False,
            }
            unit_candidates.append(noop)

            for seed in seeds:
                candidate_seed_by_layer = dict(accepted_seed_by_layer)
                candidate_seed_by_block_kv = dict(accepted_seed_by_block_kv)
                if isinstance(unit, tuple):
                    candidate_seed_by_block_kv[unit] = seed
                else:
                    candidate_seed_by_layer[unit] = seed
                candidate_layers = tuple(
                    sorted(set(candidate_seed_by_layer) | {layer_idx for layer_idx, _kv_head_idx in candidate_seed_by_block_kv})
                )
                print(
                    f"[pass {pass_idx}] evaluating {unit_label} seed={seed} layers={candidate_layers}",
                    flush=True,
                )
                candidate = analyze_candidate(
                    flat_state,
                    layers=candidate_layers,
                    seed_by_layer=candidate_seed_by_layer,
                    seed_by_block_kv=candidate_seed_by_block_kv,
                    base_seed=0,
                    angle_scale=args.angle_scale,
                    transform=args.transform,
                    parameterization=args.parameterization,
                    num_bands=args.num_bands,
                    scheme=scheme,
                    eval_ctx=eval_ctx,
                    output_proxy_ctx=output_proxy_ctx,
                    turbo_embed_export=args.turbo_embed_export,
                    turbo_mse_patterns=turbo_mse_patterns,
                    turbo_prod_patterns=turbo_prod_patterns,
                )
                candidate["seed"] = seed
                candidate["delta_bpb_vs_incumbent"] = float(
                    candidate["analysis"]["eval"]["val_bpb"] - incumbent["analysis"]["eval"]["val_bpb"]
                )
                candidate["delta_zlib_vs_incumbent"] = int(candidate["analysis"]["bytes"]["zlib_bytes"]) - int(
                    incumbent["analysis"]["bytes"]["zlib_bytes"]
                )
                candidate["would_accept"] = is_better(candidate, incumbent, args.selection_objective)
                unit_candidates.append(candidate)

            unit_candidates.sort(
                key=lambda item: selection_key(item, args.selection_objective)
            )
            best = unit_candidates[0]
            accepted = best["seed"] is not None and is_better(best, incumbent, args.selection_objective)
            if accepted:
                if isinstance(unit, tuple):
                    accepted_seed_by_block_kv[unit] = int(best["seed"])
                else:
                    accepted_seed_by_layer[unit] = int(best["seed"])
                accepted_layers = sorted(set(accepted_seed_by_layer) | {layer_idx for layer_idx, _kv_head_idx in accepted_seed_by_block_kv})
                incumbent = {
                    "layers": accepted_layers,
                    "seed_by_layer": dict(accepted_seed_by_layer),
                    "seed_by_block_kv": dict(accepted_seed_by_block_kv),
                    "meta": best["meta"],
                    "analysis": best["analysis"],
                    "proxy": best["proxy"],
                    "output_proxy": best["output_proxy"],
                }
                print(
                    f"[pass {pass_idx}] accepted {unit_label} seed={best['seed']} delta_bpb={best['delta_bpb_vs_incumbent']:+.8f}",
                    flush=True,
                )
            else:
                print(f"[pass {pass_idx}] kept incumbent for {unit_label}", flush=True)

            history.append(
                {
                    "pass": pass_idx,
                    "unit": list(unit) if isinstance(unit, tuple) else unit,
                    "accepted": accepted,
                    "best_seed": best["seed"],
                    "best_eval": best["analysis"]["eval"],
                    "best_delta_bpb_vs_incumbent_before": best["delta_bpb_vs_incumbent"],
                    "best_delta_zlib_vs_incumbent_before": best["delta_zlib_vs_incumbent"],
                    "best_selection_key": list(selection_key(best, args.selection_objective)),
                    "incumbent_after": {
                        "layers": list(accepted_layers),
                        "seed_by_layer": {str(k): int(v) for k, v in sorted(accepted_seed_by_layer.items())},
                        "seed_by_block_kv": {
                            f"{layer_idx}:{kv_head_idx}": int(v)
                            for (layer_idx, kv_head_idx), v in sorted(accepted_seed_by_block_kv.items())
                        },
                        "eval": incumbent["analysis"]["eval"],
                        "output_proxy": incumbent.get("output_proxy"),
                    },
                    "candidates": unit_candidates,
                }
            )

    raw_sanity = None
    if args.raw_sanity and (accepted_seed_by_layer or accepted_seed_by_block_kv):
        final_state, final_meta = srg.apply_rope_gauge_transform(
            flat_state,
            seed=0,
            angle_scale=args.angle_scale,
            transform=args.transform,
            parameterization=args.parameterization,
            num_bands=args.num_bands,
            layers=tuple(sorted(set(accepted_seed_by_layer) | {layer_idx for layer_idx, _kv_head_idx in accepted_seed_by_block_kv})),
            seed_by_layer=accepted_seed_by_layer,
            seed_by_block_kv=accepted_seed_by_block_kv,
        )
        sanity_loss, sanity_bpb = aqe.eval_state(eval_ctx, final_state)
        raw_sanity = {
            "eval": {"val_loss": float(sanity_loss), "val_bpb": float(sanity_bpb)},
            "delta_bpb_vs_baseline_raw": float(sanity_bpb - baseline_raw_bpb),
            "delta_loss_vs_baseline_raw": float(sanity_loss - baseline_raw_loss),
            "meta": final_meta,
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "scheme": scheme,
        "search": {
            "transform": args.transform,
            "parameterization": args.parameterization,
            "num_bands": args.num_bands,
            "layers": list(search_layers),
            "unit_scope": args.unit_scope,
            "kv_heads": list(kv_heads) if kv_heads is not None else None,
            "angle_scale": args.angle_scale,
            "seeds": seeds,
            "refine_passes": args.refine_passes,
            "selection_objective": args.selection_objective,
            "proxy_seq_len": args.proxy_seq_len,
            "proxy_max_seqs": args.proxy_max_seqs,
            "proxy_batch_seqs": args.proxy_batch_seqs,
        },
        "baseline_raw": {"val_loss": float(baseline_raw_loss), "val_bpb": float(baseline_raw_bpb)},
        "baseline_quant": baseline_quant,
        "baseline_output_proxy": baseline_output_proxy,
        "final": {
            "layers": list(accepted_layers),
            "seed_by_layer": {str(k): int(v) for k, v in sorted(accepted_seed_by_layer.items())},
            "seed_by_block_kv": {
                f"{layer_idx}:{kv_head_idx}": int(v)
                for (layer_idx, kv_head_idx), v in sorted(accepted_seed_by_block_kv.items())
            },
            "analysis": incumbent["analysis"],
            "proxy": incumbent["proxy"],
            "output_proxy": incumbent["output_proxy"],
            "selection_key": list(selection_key(incumbent, args.selection_objective)),
            "delta_bpb_vs_baseline_quant": float(
                incumbent["analysis"]["eval"]["val_bpb"] - baseline_quant["eval"]["val_bpb"]
            ),
            "delta_zlib_vs_baseline_quant": int(incumbent["analysis"]["bytes"]["zlib_bytes"])
            - int(baseline_quant["bytes"]["zlib_bytes"]),
        },
        "raw_sanity": raw_sanity,
        "history": history,
    }
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
