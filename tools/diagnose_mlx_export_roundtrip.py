#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as base

try:
    import train_gpt_mlx_jepa_aux as jepa_aux
except ImportError:
    jepa_aux = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a raw MLX checkpoint, its in-memory quantized roundtrip, and its "
            "disk-style zlib roundtrip under the current export settings."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Raw MLX .npz checkpoint produced by training.")
    parser.add_argument(
        "support_files",
        nargs="*",
        help="Additional staged support files. Ignored at runtime; present only so dispatch.sh copies them.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--val-max-seqs", type=int, default=-1, help="Override VAL_MAX_SEQS for this analysis.")
    parser.add_argument("--tensor-topk", type=int, default=20, help="How many worst tensors to report.")
    return parser


def eval_model_state(
    hp: base.Hyperparameters,
    model: base.GPT,
    flat_state: dict[str, mx.array],
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    model.clear_turbo_cache()
    model.set_turbo_qat(False, 0.0)
    model.update(tree_unflatten(list(flat_state.items())))
    model.clear_turbo_cache()
    return base.eval_val(
        hp,
        model,
        lambda x, y, operator_codes=None: model.loss(x, y, operator_codes),
        lambda x, operator_codes=None: model.forward_logits(x, operator_codes),
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )


def tensor_error_report(
    flat_state: dict[str, mx.array],
    deq_state: dict[str, mx.array],
    topk: int,
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for name, orig in flat_state.items():
        deq = deq_state.get(name)
        if deq is None or not mx.issubdtype(orig.dtype, mx.floating):
            continue
        orig_np = np.asarray(orig.astype(mx.float32), dtype=np.float32)
        deq_np = np.asarray(deq.astype(mx.float32), dtype=np.float32)
        diff = deq_np - orig_np
        rmse = float(np.sqrt(np.square(diff, dtype=np.float64).mean()))
        mae = float(np.abs(diff, dtype=np.float64).mean())
        orig_rms = float(np.sqrt(np.square(orig_np, dtype=np.float64).mean()))
        deq_rms = float(np.sqrt(np.square(deq_np, dtype=np.float64).mean()))
        cosine = float(
            np.multiply(orig_np, deq_np, dtype=np.float64).sum()
            / max(
                math.sqrt(
                    float(np.square(orig_np, dtype=np.float64).sum())
                    * float(np.square(deq_np, dtype=np.float64).sum())
                ),
                1e-12,
            )
        )
        report.append(
            {
                "name": name,
                "shape": list(orig.shape),
                "dtype": str(orig.dtype),
                "numel": int(orig.size),
                "orig_rms": orig_rms,
                "deq_rms": deq_rms,
                "rmse": rmse,
                "relative_rmse": rmse / max(orig_rms, 1e-12),
                "mae": mae,
                "max_abs_diff": float(np.abs(diff, dtype=np.float64).max(initial=0.0)),
                "cosine": cosine,
            }
        )
    report.sort(key=lambda item: (item["relative_rmse"], item["rmse"]), reverse=True)
    return report[: max(int(topk), 0)]


def materialize_turbo_qat_state(flat_state: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for name, arr in flat_state.items():
        mode = base.infer_turbo_mode(name)
        if mode != "none" and arr.ndim == 2 and int(arr.shape[1]) >= base.TURBO_BLOCK_SIZE:
            deq_arr, _ = base.turbo_quantize_dequantize_array(
                arr,
                mode=mode,
                total_bits=base.TURBO_MSE_BITS if mode == "mse" else base.TURBO_PROD_BITS,
                block_size=base.TURBO_BLOCK_SIZE,
            )
            out[name] = mx.array(deq_arr, dtype=arr.dtype)
        else:
            out[name] = arr
    return out


def main() -> None:
    cli = build_parser().parse_args()
    hp = base.Hyperparameters()
    if cli.val_max_seqs >= 0:
        hp.val_max_seqs = cli.val_max_seqs
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    model = base.make_gpt(hp, sp)

    flat_state = {name: value for name, value in mx.load(str(cli.checkpoint)).items()}
    model_keys = set(base.exportable_flat_state(model))
    flat_keys = set(flat_state)
    export_schema = {
        "checkpoint_key_count": len(flat_keys),
        "model_key_count": len(model_keys),
        "missing_from_checkpoint": sorted(model_keys - flat_keys)[:25],
        "extra_in_checkpoint": sorted(flat_keys - model_keys)[:25],
    }

    val_tokens = base.limit_validation_tokens(
        base.load_validation_tokens(hp.val_files, max(hp.train_seq_len, hp.effective_eval_seq_len)),
        hp.effective_eval_seq_len,
        hp.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(sp, hp.vocab_size)

    raw_reload_loss, raw_reload_bpb = eval_model_state(
        hp,
        model,
        flat_state,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    jepa_class_eval: dict[str, float] | None = None
    if jepa_aux is not None:
        jepa_hp = jepa_aux.Hyperparameters()
        jepa_model = jepa_aux.GPTJEPAAux(
            **base.gpt_kwargs_from_args(jepa_hp, sp),
            jepa_chunk_size=jepa_hp.jepa_chunk_size,
            jepa_latent_dim=jepa_hp.jepa_latent_dim,
            jepa_pred_hidden=jepa_hp.jepa_pred_hidden,
            jepa_pred_weight=jepa_hp.jepa_pred_weight,
            jepa_sigreg_weight=jepa_hp.jepa_sigreg_weight,
            jepa_summary_mode=jepa_hp.jepa_summary_mode,
            jepa_pred_mode=jepa_hp.jepa_pred_mode,
            jepa_pred_init_std=jepa_hp.jepa_pred_init_std,
            jepa_sigreg_knots=jepa_hp.jepa_sigreg_knots,
            jepa_sigreg_num_proj=jepa_hp.jepa_sigreg_num_proj,
            jepa_sigreg_seed=jepa_hp.jepa_sigreg_seed,
        )
        jepa_reload_loss, jepa_reload_bpb = eval_model_state(
            hp,
            jepa_model,
            flat_state,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        jepa_class_eval = {"val_loss": float(jepa_reload_loss), "val_bpb": float(jepa_reload_bpb)}

    qat_materialized_state = materialize_turbo_qat_state(flat_state)
    qat_materialized_loss, qat_materialized_bpb = eval_model_state(
        hp,
        model,
        qat_materialized_state,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
    quant_deq_state = base.dequantize_state_dict(quant_obj)
    quant_loss, quant_bpb = eval_model_state(
        hp,
        model,
        quant_deq_state,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    disk_roundtrip = pickle.loads(zlib.decompress(quant_blob))
    disk_deq_state = base.dequantize_state_dict(disk_roundtrip)
    disk_loss, disk_bpb = eval_model_state(
        hp,
        model,
        disk_deq_state,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    qat_quant_obj, qat_quant_stats, qat_quant_raw, qat_quant_blob = base.serialize_quantized_state_dict(qat_materialized_state)
    qat_quant_deq_state = base.dequantize_state_dict(qat_quant_obj)
    qat_quant_loss, qat_quant_bpb = eval_model_state(
        hp,
        model,
        qat_quant_deq_state,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    results = {
        "checkpoint": str(cli.checkpoint),
        "config": {
            "run_id": hp.run_id,
            "quant_format": hp.quant_format,
            "num_layers": hp.num_layers,
            "num_layer_templates": hp.num_layer_templates,
            "model_dim": hp.model_dim,
            "num_heads": hp.num_heads,
            "num_kv_heads": hp.num_kv_heads,
            "mlp_mult": hp.mlp_mult,
            "mlp_leaky_slope": hp.mlp_leaky_slope,
            "tie_embeddings": hp.tie_embeddings,
            "num_registers": hp.num_registers,
            "logic_dim": hp.logic_dim,
            "logic_layer_index": base.resolve_logic_layer_index(hp.logic_layer_index, hp.num_layers) if hp.logic_dim > 0 else None,
            "logic_route_to_next_token": hp.logic_route_to_next_token,
            "val_max_seqs": hp.val_max_seqs,
            "effective_eval_seq_len": hp.effective_eval_seq_len,
        },
        "export_schema": export_schema,
        "eval": {
            "raw_reload": {"val_loss": float(raw_reload_loss), "val_bpb": float(raw_reload_bpb)},
            "qat_materialized_reload": {"val_loss": float(qat_materialized_loss), "val_bpb": float(qat_materialized_bpb)},
            "quant_in_memory": {"val_loss": float(quant_loss), "val_bpb": float(quant_bpb)},
            "quant_disk_roundtrip": {"val_loss": float(disk_loss), "val_bpb": float(disk_bpb)},
            "qat_materialized_quant_in_memory": {"val_loss": float(qat_quant_loss), "val_bpb": float(qat_quant_bpb)},
            "quantization_gap_bpb": float(disk_bpb - raw_reload_bpb),
            "qat_materialized_quantization_gap_bpb": float(qat_quant_bpb - qat_materialized_bpb),
        },
        "quant_bytes": {
            "payload_bytes": int(quant_stats.get("int8_payload_bytes", 0)),
            "raw_pickle_bytes": len(quant_raw),
            "zlib_bytes": len(quant_blob),
            "qat_materialized_payload_bytes": int(qat_quant_stats.get("int8_payload_bytes", 0)),
            "qat_materialized_raw_pickle_bytes": len(qat_quant_raw),
            "qat_materialized_zlib_bytes": len(qat_quant_blob),
        },
        "quant_stats": quant_stats,
        "qat_materialized_quant_stats": qat_quant_stats,
        "top_tensor_errors": tensor_error_report(flat_state, disk_deq_state, cli.tensor_topk),
        "top_qat_materialization_deltas": tensor_error_report(flat_state, qat_materialized_state, cli.tensor_topk),
    }
    if jepa_class_eval is not None:
        results["eval"]["jepa_class_raw_reload"] = jepa_class_eval
    text = json.dumps(results, indent=2, sort_keys=True)
    if cli.out:
        cli.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
