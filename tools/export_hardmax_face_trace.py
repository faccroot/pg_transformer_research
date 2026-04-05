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
from mlx.utils import tree_flatten, tree_unflatten

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
        description="Export a per-token hardmax/operator-state 'face trace' from a saved MLX artifact."
    )
    p.add_argument("--artifact", required=True, help="Path to *_mlx_model.npz or *_int8zlib.pklz")
    p.add_argument("--config-json", required=True, help="Training config JSON with an `env` block")
    p.add_argument("--text", default="", help="Inline prompt text to analyze")
    p.add_argument("--text-file", default="", help="Optional UTF-8 text file to analyze instead of --text")
    p.add_argument("--output-json", default="", help="Optional output path for the exported trace JSON")
    p.add_argument("--label", default="", help="Optional label stored in the trace payload")
    p.add_argument("--tokenizer-path", default="", help="Optional override for TOKENIZER_PATH")
    p.add_argument("--repo-bundle", default="", help="Optional tar/tgz of current repo Python modules for remote parity")
    p.add_argument("--top-k", type=int, default=5, help="How many next-token predictions to keep per position")
    p.add_argument("--max-soft-states", type=int, default=3, help="How many soft usage states to keep per position")
    p.add_argument("--truncate-tokens", type=int, default=0, help="If set, truncate the tokenized prompt to this length")
    return p.parse_args()


def load_modules():
    import train_gpt_mlx as base_mod

    return importlib.reload(base_mod)


def maybe_stage_repo_bundle(bundle_path: str) -> Path | None:
    raw = bundle_path.strip()
    if not raw:
        return None
    bundle = Path(raw).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Repo bundle not found: {bundle}")
    temp_root = Path(tempfile.mkdtemp(prefix="hardmax_face_bundle_"))
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


def flat_parameter_state(model) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def apply_flat_arrays(model, flat_state: dict[str, mx.array]) -> None:
    model.update(tree_unflatten(list(flat_state.items())))


def select_compatible_flat_state(
    model,
    checkpoint_state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, object]]:
    reference = flat_parameter_state(model)
    matched: dict[str, mx.array] = {}
    missing = 0
    mismatched = 0
    unexpected = 0
    matched_params = 0
    total_params = 0
    for name, ref_value in reference.items():
        numel = int(np.prod(ref_value.shape))
        total_params += numel
        checkpoint_value = checkpoint_state.get(name)
        if checkpoint_value is None:
            missing += 1
            continue
        if tuple(checkpoint_value.shape) != tuple(ref_value.shape):
            mismatched += 1
            continue
        matched[name] = checkpoint_value.astype(ref_value.dtype)
        matched_params += numel
    for name in checkpoint_state:
        if name not in reference:
            unexpected += 1
    return matched, {
        "matched_tensors": len(matched),
        "matched_params": matched_params,
        "total_tensors": len(reference),
        "total_params": total_params,
        "matched_param_fraction": matched_params / max(total_params, 1),
        "missing_tensors": missing,
        "mismatched_tensors": mismatched,
        "unexpected_tensors": unexpected,
    }


def resolve_text(args: argparse.Namespace) -> str:
    inline = args.text
    file_text = ""
    if args.text_file:
        file_text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
    text = file_text if file_text else inline
    if not text.strip():
        raise ValueError("Provide non-empty prompt text via --text or --text-file")
    return text


def to_numpy(x: mx.array | None) -> np.ndarray | None:
    if x is None:
        return None
    value = mx.stop_gradient(x)
    try:
        return np.asarray(value)
    except RuntimeError:
        dtype_text = str(getattr(value, "dtype", ""))
        if "float" in dtype_text or "bfloat" in dtype_text:
            return np.asarray(value.astype(mx.float32))
        if "int" in dtype_text or "uint" in dtype_text:
            return np.asarray(value.astype(mx.int32))
        raise


def strip_registers(model, x: mx.array | None) -> mx.array | None:
    if x is None:
        return None
    return model.strip_registers(x)


def top_state_summary(row: np.ndarray, max_states: int) -> list[dict[str, float | int]]:
    if row.ndim != 1 or row.size <= 0 or max_states <= 0:
        return []
    order = np.argsort(row)[::-1][: max_states]
    return [
        {
            "state": int(idx),
            "weight": float(row[idx]),
        }
        for idx in order
    ]


def prediction_summary(
    probs_row: np.ndarray,
    sp: spm.SentencePieceProcessor,
    *,
    top_k: int,
) -> list[dict[str, float | int | str]]:
    if probs_row.ndim != 1 or probs_row.size <= 0 or top_k <= 0:
        return []
    order = np.argsort(probs_row)[::-1][: top_k]
    return [
        {
            "token_id": int(idx),
            "piece": sp.id_to_piece(int(idx)),
            "prob": float(probs_row[idx]),
        }
        for idx in order
    ]


def build_trace_payload(
    *,
    artifact_path: Path,
    config_path: Path,
    label: str,
    text: str,
    token_ids: list[int],
    sp: spm.SentencePieceProcessor,
    checkpoint_load_stats: dict[str, object],
    operator_codes: np.ndarray | None,
    seed_polarity_scores: np.ndarray | None,
    polarity_scores: np.ndarray | None,
    confidence: np.ndarray | None,
    budget: np.ndarray | None,
    state_index: np.ndarray | None,
    soft_usage: np.ndarray | None,
    struct_state: np.ndarray | None,
    probs: np.ndarray,
    top_k: int,
    max_soft_states: int,
) -> dict[str, object]:
    seq_len = len(token_ids)
    rows: list[dict[str, object]] = []
    state_histogram: dict[str, int] = {}
    if state_index is not None:
        unique, counts = np.unique(state_index.reshape(-1), return_counts=True)
        state_histogram = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    for pos, token_id in enumerate(token_ids):
        row: dict[str, object] = {
            "position": int(pos),
            "token_id": int(token_id),
            "piece": sp.id_to_piece(int(token_id)),
        }
        if pos + 1 < seq_len:
            row["target_next_token_id"] = int(token_ids[pos + 1])
            row["target_next_piece"] = sp.id_to_piece(int(token_ids[pos + 1]))
        if operator_codes is not None:
            row["operator_code"] = int(operator_codes[0, pos])
        if seed_polarity_scores is not None:
            row["seed_polarity"] = float(seed_polarity_scores[0, pos])
        if polarity_scores is not None:
            row["polarity_score"] = float(polarity_scores[0, pos])
        if confidence is not None:
            row["hardmax_confidence"] = float(confidence[0, pos])
        if budget is not None:
            row["hardmax_budget"] = float(budget[0, pos])
        if state_index is not None:
            row["state_index"] = int(state_index[0, pos])
        if struct_state is not None:
            row["struct_state_norm"] = float(np.linalg.norm(struct_state[0, pos]))
        if soft_usage is not None:
            row["soft_top_states"] = top_state_summary(soft_usage[0, pos], max_soft_states)
        row["next_token_prediction_topk"] = prediction_summary(probs[0, pos], sp, top_k=top_k)
        rows.append(row)

    confidence_flat = None if confidence is None else confidence.reshape(-1).astype(np.float32)
    budget_flat = None if budget is None else budget.reshape(-1).astype(np.float32)
    operator_flat = None if operator_codes is None else operator_codes.reshape(-1).astype(np.int32)
    payload = {
        "label": label,
        "artifact": str(artifact_path),
        "config_json": str(config_path),
        "text": text,
        "decoded_text": sp.decode(token_ids),
        "num_tokens": int(seq_len),
        "checkpoint_load_stats": checkpoint_load_stats,
        "summary": {
            "has_hardmax_structural": confidence is not None,
            "confidence_mean": None if confidence_flat is None or confidence_flat.size <= 0 else float(confidence_flat.mean()),
            "confidence_std": None if confidence_flat is None or confidence_flat.size <= 0 else float(confidence_flat.std()),
            "budget_mean": None if budget_flat is None or budget_flat.size <= 0 else float(budget_flat.mean()),
            "budget_std": None if budget_flat is None or budget_flat.size <= 0 else float(budget_flat.std()),
            "state_histogram": state_histogram,
            "used_states": int(len(state_histogram)),
            "max_state_fraction": (
                0.0
                if not state_histogram
                else float(max(state_histogram.values()) / max(sum(state_histogram.values()), 1))
            ),
            "operator_histogram": (
                {}
                if operator_flat is None
                else {str(int(k)): int(v) for k, v in zip(*np.unique(operator_flat, return_counts=True))}
            ),
        },
        "tokens": rows,
    }
    return payload


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    text = resolve_text(args)
    maybe_stage_repo_bundle(args.repo_bundle)
    apply_config_env(config_path, args)
    base_mod = load_modules()
    hps = base_mod.Hyperparameters()
    if not hps.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {hps.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=hps.tokenizer_path)
    if int(sp.vocab_size()) != int(hps.vocab_size):
        raise ValueError(
            f"VOCAB_SIZE={hps.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    token_ids = sp.encode(text, out_type=int)
    if args.truncate_tokens > 0:
        token_ids = token_ids[: int(args.truncate_tokens)]
    if not token_ids:
        raise ValueError("Tokenized prompt is empty after truncation")

    model = base_mod.make_gpt(hps, sp)
    checkpoint_state = load_flat_state(artifact_path, base_mod)
    select_state_fn = getattr(base_mod, "select_compatible_flat_state", select_compatible_flat_state)
    apply_arrays_fn = getattr(base_mod, "apply_flat_arrays", apply_flat_arrays)
    compatible_state, load_stats = select_state_fn(model, checkpoint_state)
    if not compatible_state:
        raise ValueError(f"No compatible tensors found in {artifact_path}")
    apply_arrays_fn(model, compatible_state)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()

    input_ids = mx.array(np.asarray(token_ids, dtype=np.int32)[None, :], dtype=mx.int32)
    final_hidden, _captured, aux = model.forward_hidden_with_aux(input_ids)
    logits_proj = final_hidden @ model.tok_emb.weight.astype(final_hidden.dtype).T if model.tie_embeddings else model.lm_head(final_hidden)
    logits = model.softcap(logits_proj)
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)

    structural_aux = aux.get("hardmax_structural")
    operator_codes = strip_registers(model, aux.get("operator_codes"))
    seed_polarity_scores = strip_registers(model, aux.get("seed_polarity_scores"))
    polarity_scores = strip_registers(model, aux.get("polarity_scores"))
    confidence = strip_registers(model, structural_aux.get("confidence")) if isinstance(structural_aux, dict) else None
    budget = strip_registers(model, structural_aux.get("budget")) if isinstance(structural_aux, dict) else None
    state_index = strip_registers(model, structural_aux.get("state_index")) if isinstance(structural_aux, dict) else None
    soft_usage = strip_registers(model, structural_aux.get("soft_usage")) if isinstance(structural_aux, dict) else None
    struct_state = strip_registers(model, structural_aux.get("struct_state")) if isinstance(structural_aux, dict) else None

    budget_np = to_numpy(budget)
    if budget_np is not None and budget_np.ndim == 3 and budget_np.shape[-1] == 1:
        budget_np = budget_np[..., 0]
    payload = build_trace_payload(
        artifact_path=artifact_path,
        config_path=config_path,
        label=args.label,
        text=text,
        token_ids=token_ids,
        sp=sp,
        checkpoint_load_stats=load_stats,
        operator_codes=to_numpy(operator_codes),
        seed_polarity_scores=to_numpy(seed_polarity_scores),
        polarity_scores=to_numpy(polarity_scores),
        confidence=to_numpy(confidence),
        budget=budget_np,
        state_index=to_numpy(state_index),
        soft_usage=to_numpy(soft_usage),
        struct_state=to_numpy(struct_state),
        probs=to_numpy(probs),
        top_k=max(int(args.top_k), 1),
        max_soft_states=max(int(args.max_soft_states), 1),
    )

    rendered = json.dumps(payload, indent=2)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")
        print(output_json)
        return
    print(rendered)


if __name__ == "__main__":
    main()
