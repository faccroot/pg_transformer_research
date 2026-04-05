#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import sentencepiece as spm

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "tools" else SCRIPT_PATH.parent
TOOLS_ROOT = REPO_ROOT / "tools"
PATH_CANDIDATES = [
    REPO_ROOT,
    TOOLS_ROOT,
    Path.home() / "transformer_research" / "parameter-golf",
    Path.home() / "transformer_research" / "parameter-golf" / "tools",
]
for root in PATH_CANDIDATES:
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved MLX hardmax model under baseline and causal hardmax ablations."
    )
    p.add_argument("--artifact", required=True, help="Path to *_mlx_model.npz or *_int8zlib.pklz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument(
        "--ablations",
        default="baseline,zero_hardmax,zero_residual,zero_q_bias,zero_tau",
        help="Comma-separated ablation names. Supported: baseline, zero_hardmax, zero_residual, zero_q_bias, zero_tau",
    )
    p.add_argument("--result-json", default="", help="Optional output JSON path")
    p.add_argument("--label", default="", help="Optional run label")
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--data-path", default="", help="Optional override for DATA_PATH")
    p.add_argument("--val-max-seqs", type=int, default=-1, help="Optional override for VAL_MAX_SEQS")
    p.add_argument("--eval-seq-len", type=int, default=-1, help="Optional override for EVAL_SEQ_LEN")
    p.add_argument("--eval-stride", type=int, default=-1, help="Optional override for EVAL_STRIDE")
    p.add_argument("--eval-batch-seqs", type=int, default=-1, help="Optional override for EVAL_BATCH_SEQS")
    p.add_argument("--repo-bundle", default="", help="Optional tar/tgz of current repo Python modules for remote parity")
    return p.parse_args()


def maybe_stage_repo_bundle(bundle_path: str) -> Path | None:
    raw = bundle_path.strip()
    if not raw:
        return None
    bundle = Path(raw).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Repo bundle not found: {bundle}")
    temp_root = Path(tempfile.mkdtemp(prefix="hardmax_causal_bundle_"))
    with tarfile.open(bundle, "r:*") as tf:
        tf.extractall(temp_root)
    candidate_roots = [temp_root / "parameter-golf", temp_root]
    for root in candidate_roots:
        if root.exists():
            tools_root = root / "tools"
            for path in (root, tools_root):
                if path.exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return root
    return temp_root


def load_modules():
    import train_gpt_mlx as base_mod

    return importlib.reload(base_mod)


def load_config_env_payload(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", payload)
    if not isinstance(env_payload, dict):
        raise ValueError(f"Config env payload must be a JSON object: {config_path}")
    return env_payload


def apply_config_env(config_json: Path, args: argparse.Namespace) -> dict[str, object]:
    env = load_config_env_payload(config_json)
    for key, value in env.items():
        os.environ[str(key)] = str(value)
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = os.path.expanduser(args.tokenizer_path)
    if args.data_path:
        os.environ["DATA_PATH"] = os.path.expanduser(args.data_path)
    if args.val_max_seqs >= 0:
        os.environ["VAL_MAX_SEQS"] = str(int(args.val_max_seqs))
    if args.eval_seq_len > 0:
        os.environ["EVAL_SEQ_LEN"] = str(int(args.eval_seq_len))
    if args.eval_stride >= 0:
        os.environ["EVAL_STRIDE"] = str(int(args.eval_stride))
    if args.eval_batch_seqs > 0:
        os.environ["EVAL_BATCH_SEQS"] = str(int(args.eval_batch_seqs))
    os.environ["MLX_COMPILE"] = "0"
    os.environ["VAL_LOSS_EVERY"] = "0"
    os.environ["QUANT_EVAL_EVERY"] = "0"
    os.environ["QUANT_EVAL_MAX_SEQS"] = "0"
    return env


def load_flat_state(artifact_path: Path, base_mod) -> dict[str, object]:
    if artifact_path.suffix == ".pklz":
        return base_mod.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    if artifact_path.suffix == ".npz":
        return dict(base_mod.mx.load(str(artifact_path)).items())
    raise ValueError(f"Unsupported artifact type: {artifact_path}")


def parse_ablations(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("Provide at least one ablation name")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def evaluate_ablation(
    *,
    base_mod,
    hps,
    model,
    ablation: str,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> dict[str, object]:
    spec = base_mod.resolve_hardmax_eval_ablation(ablation)
    with model.hardmax_eval_ablation_scope(spec):
        model.clear_turbo_cache()
        uses_operator_codes = (
            getattr(model, "logic_sidecar", None) is not None
            or getattr(model, "hardmax_structural_controller", None) is not None
            or getattr(model, "polarity_detector", None) is not None
        )
        if uses_operator_codes:
            ce_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y, operator_codes)
            forward_logits = lambda x, operator_codes=None: model.forward_logits(x, operator_codes)
        else:
            ce_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y)
            forward_logits = lambda x, operator_codes=None: model.forward_logits(x)
        val_loss, val_bpb = base_mod.eval_val(
            hps,
            model,
            ce_loss,
            forward_logits,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=None,
        )
    return {
        "ablation": spec.name,
        "disable_controller": bool(spec.disable_controller),
        "disable_residual_write": bool(spec.disable_residual_write),
        "disable_residual_budget": bool(spec.disable_residual_budget),
        "disable_attn_q_bias": bool(spec.disable_attn_q_bias),
        "disable_attn_tau": bool(spec.disable_attn_tau),
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
    }


def main() -> None:
    args = parse_args()
    maybe_stage_repo_bundle(args.repo_bundle)
    artifact_path = Path(args.artifact).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve()
    env_payload = apply_config_env(config_path, args)
    base_mod = load_modules()
    hps = base_mod.Hyperparameters()
    tokenizer_path = os.path.expanduser(args.tokenizer_path or hps.tokenizer_path)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model = base_mod.make_gpt(hps, sp)

    checkpoint_state = load_flat_state(artifact_path, base_mod)
    selected_state, checkpoint_stats = base_mod.select_compatible_flat_state(model, checkpoint_state)
    base_mod.apply_flat_arrays(model, selected_state)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()

    val_seq_len = max(hps.train_seq_len, hps.effective_eval_seq_len)
    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, val_seq_len),
        hps.effective_eval_seq_len,
        hps.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base_mod.build_sentencepiece_luts(
        sp,
        hps.vocab_size,
    )

    ablation_names = parse_ablations(args.ablations)
    results = [
        evaluate_ablation(
            base_mod=base_mod,
            hps=hps,
            model=model,
            ablation=name,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        for name in ablation_names
    ]
    baseline = next((item for item in results if item["ablation"] == "baseline"), results[0])
    for item in results:
        item["delta_val_loss_vs_baseline"] = float(item["val_loss"]) - float(baseline["val_loss"])
        item["delta_val_bpb_vs_baseline"] = float(item["val_bpb"]) - float(baseline["val_bpb"])

    payload = {
        "label": args.label,
        "artifact": str(artifact_path),
        "config_json": str(config_path),
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "data_path": str(Path(os.path.expanduser(hps.data_path)).resolve()),
        "checkpoint_load_stats": checkpoint_stats,
        "env_overrides": {
            "val_max_seqs": int(hps.val_max_seqs),
            "eval_seq_len": int(hps.effective_eval_seq_len),
            "eval_stride": int(hps.eval_stride),
            "eval_batch_seqs": int(hps.effective_eval_batch_seqs),
        },
        "hardmax_runtime": {
            "has_controller": bool(getattr(model, "hardmax_structural_controller", None) is not None),
            "condition_mode": str(getattr(model, "hardmax_struct_condition_mode", "")),
            "route_residual_budget": bool(getattr(model, "hardmax_struct_route_residual_budget", False)),
            "has_attn_q_proj": bool(getattr(model, "hardmax_struct_attn_q_proj", None) is not None),
            "has_attn_tau_proj": bool(getattr(model, "hardmax_struct_attn_tau_proj", None) is not None),
            "fast_refinement_steps": int(getattr(model, "hardmax_struct_fast_refinement_steps", 0)),
        },
        "validation_tokens": int(val_tokens.size - 1),
        "config_env_keys": sorted(str(key) for key in env_payload.keys()),
        "results": results,
        "baseline": baseline,
        "best_val_bpb": min(results, key=lambda item: float(item["val_bpb"])),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.result_json:
        out_path = Path(args.result_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
