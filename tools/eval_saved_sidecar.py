#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import subprocess
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
from mlx.utils import tree_unflatten

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved JEPA sidecar artifact with reset and/or persistent eval.")
    p.add_argument("--artifact", required=True, help="Path to *_int8zlib.pklz or *_mlx_model.npz")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--val-max-seqs", type=int, default=0)
    p.add_argument("--train-seq-len", type=int, default=512)
    p.add_argument("--eval-seq-len", type=int, default=0)
    p.add_argument("--persistent", type=int, default=None, help="Legacy toggle: 0=reset, 1=persistent.")
    p.add_argument("--mode", choices=("reset", "persistent", "both"), default=None)
    p.add_argument("--persist-group-seqs", type=int, default=1)
    p.add_argument("--label", default="")
    p.add_argument("--result-json", default="")
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--tie-embeddings", type=int, default=0)
    p.add_argument("--logic-dim", type=int, default=0)
    p.add_argument("--logic-layer-index", type=int, default=4)
    p.add_argument("--logic-route-to-next-token", type=int, default=1)
    p.add_argument("--logic-operator-mode", default="not_only")
    p.add_argument("--polarity-detector-enabled", type=int, default=0)
    p.add_argument("--polarity-detector-layer-index", type=int, default=3)
    p.add_argument("--polarity-detector-hidden-dim", type=int, default=64)
    p.add_argument("--polarity-seed-blend", type=float, default=1.0)
    p.add_argument("--polarity-seed-weight", type=float, default=0.0)
    p.add_argument("--polarity-sparse-weight", type=float, default=0.0)
    p.add_argument("--polarity-smooth-weight", type=float, default=0.0)
    p.add_argument("--sidecar-polarity-write", type=int, default=0)
    p.add_argument("--sidecar-polarity-pool", default="max")
    p.add_argument("--sidecar-tap-layer", type=int, default=3)
    p.add_argument("--sidecar-pred-weight", type=float, default=0.05)
    p.add_argument("--sidecar-pred-offset", type=int, default=1)
    p.add_argument("--sidecar-sigreg-weight", type=float, default=0.01)
    p.add_argument("--sidecar-spherical-weight", type=float, default=0.01)
    p.add_argument("--sidecar-sigreg-mode", default="full")
    p.add_argument("--sidecar-weak-sigreg-dim", type=int, default=32)
    p.add_argument("--sidecar-read-rmsnorm", type=int, default=1)
    p.add_argument("--sidecar-summary-mode", default="query")
    p.add_argument("--sidecar-pred-target-mode", default="delta")
    p.add_argument("--sidecar-transition-reset-enabled", type=int, default=0)
    p.add_argument("--sidecar-transition-reset-cosine-threshold", type=float, default=0.80)
    p.add_argument("--sidecar-transition-reset-cosine-sharpness", type=float, default=12.0)
    p.add_argument("--sidecar-transition-reset-prior-weight", type=float, default=1.0)
    p.add_argument("--sidecar-transition-reset-learned-weight", type=float, default=0.0)
    p.add_argument("--sidecar-transition-reset-max-gate", type=float, default=1.0)
    p.add_argument("--sidecar-transition-reset-init-std", type=float, default=1e-3)
    p.add_argument("--sidecar-reset-on-bos", type=int, default=0)
    p.add_argument("--sidecar-reset-token-id", type=int, default=1)
    p.add_argument("--adapt-sidecar", type=int, default=0, help="If set, run sidecar-only retrospective self-distillation after scoring each eval chunk.")
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--adapt-beta1", type=float, default=0.9)
    p.add_argument("--adapt-beta2", type=float, default=0.999)
    p.add_argument("--adapt-recompute-carry", type=int, default=1, help="When persistent, recompute the carry state after each sidecar update.")
    p.add_argument("--adapt-rescore", type=int, default=1, help="If set, run a clean second exact re-score after sidecar adaptation.")
    p.add_argument(
        "--trainer-module",
        default="train_gpt_mlx_sidecar_canonical",
        help="Sidecar trainer module to instantiate, e.g. train_gpt_mlx_sidecar_canonical or train_gpt_mlx_jepa_sidecar.",
    )
    p.add_argument("support_files", nargs="*")
    return p.parse_args()


def set_env(args: argparse.Namespace) -> None:
    os.environ["TOKENIZER_PATH"] = os.path.expanduser(args.tokenizer_path)
    os.environ["DATA_PATH"] = os.path.expanduser(args.data_path)
    os.environ["VAL_MAX_SEQS"] = str(args.val_max_seqs)
    os.environ["TRAIN_SEQ_LEN"] = str(int(args.train_seq_len))
    os.environ["QUANT_EVAL_MAX_SEQS"] = "0"
    os.environ["VAL_LOSS_EVERY"] = "0"
    os.environ["QUANT_EVAL_EVERY"] = "0"
    os.environ["TRAIN_SHARDS"] = str(int(args.train_shards))
    os.environ["TRAIN_BATCH_TOKENS"] = "8192"
    os.environ["VAL_BATCH_SIZE"] = "8192"
    os.environ["GRAD_ACCUM_STEPS"] = "8"
    os.environ["MLX_COMPILE"] = "0"
    os.environ["TIE_EMBEDDINGS"] = str(int(args.tie_embeddings))
    if args.eval_seq_len > 0:
        os.environ["EVAL_SEQ_LEN"] = str(int(args.eval_seq_len))
    os.environ["LOGIC_DIM"] = str(int(args.logic_dim))
    os.environ["LOGIC_LAYER_INDEX"] = str(int(args.logic_layer_index))
    os.environ["LOGIC_ROUTE_TO_NEXT_TOKEN"] = str(int(args.logic_route_to_next_token))
    os.environ["LOGIC_OPERATOR_MODE"] = args.logic_operator_mode
    os.environ["POLARITY_DETECTOR_ENABLED"] = str(int(args.polarity_detector_enabled))
    os.environ["POLARITY_DETECTOR_LAYER_INDEX"] = str(int(args.polarity_detector_layer_index))
    os.environ["POLARITY_DETECTOR_HIDDEN_DIM"] = str(int(args.polarity_detector_hidden_dim))
    os.environ["POLARITY_SEED_BLEND"] = str(args.polarity_seed_blend)
    os.environ["POLARITY_SEED_WEIGHT"] = str(args.polarity_seed_weight)
    os.environ["POLARITY_SPARSE_WEIGHT"] = str(args.polarity_sparse_weight)
    os.environ["POLARITY_SMOOTH_WEIGHT"] = str(args.polarity_smooth_weight)
    os.environ["SIDECAR_POLARITY_WRITE"] = str(int(args.sidecar_polarity_write))
    os.environ["SIDECAR_POLARITY_POOL"] = args.sidecar_polarity_pool
    os.environ["SIDECAR_TAP_LAYER"] = str(int(args.sidecar_tap_layer))
    os.environ["SIDECAR_PRED_WEIGHT"] = str(args.sidecar_pred_weight)
    os.environ["SIDECAR_PRED_OFFSET"] = str(int(args.sidecar_pred_offset))
    os.environ["SIDECAR_SIGREG_WEIGHT"] = str(args.sidecar_sigreg_weight)
    os.environ["SIDECAR_SPHERICAL_WEIGHT"] = str(args.sidecar_spherical_weight)
    os.environ["SIDECAR_SIGREG_MODE"] = args.sidecar_sigreg_mode
    os.environ["SIDECAR_WEAK_SIGREG_DIM"] = str(int(args.sidecar_weak_sigreg_dim))
    os.environ["SIDECAR_READ_RMSNORM"] = str(int(args.sidecar_read_rmsnorm))
    os.environ["SIDECAR_SUMMARY_MODE"] = args.sidecar_summary_mode
    os.environ["SIDECAR_PRED_TARGET_MODE"] = args.sidecar_pred_target_mode
    os.environ["SIDECAR_TRANSITION_RESET_ENABLED"] = str(int(args.sidecar_transition_reset_enabled))
    os.environ["SIDECAR_TRANSITION_RESET_COSINE_THRESHOLD"] = str(args.sidecar_transition_reset_cosine_threshold)
    os.environ["SIDECAR_TRANSITION_RESET_COSINE_SHARPNESS"] = str(args.sidecar_transition_reset_cosine_sharpness)
    os.environ["SIDECAR_TRANSITION_RESET_PRIOR_WEIGHT"] = str(args.sidecar_transition_reset_prior_weight)
    os.environ["SIDECAR_TRANSITION_RESET_LEARNED_WEIGHT"] = str(args.sidecar_transition_reset_learned_weight)
    os.environ["SIDECAR_TRANSITION_RESET_MAX_GATE"] = str(args.sidecar_transition_reset_max_gate)
    os.environ["SIDECAR_TRANSITION_RESET_INIT_STD"] = str(args.sidecar_transition_reset_init_std)
    os.environ["SIDECAR_RESET_ON_BOS"] = str(int(args.sidecar_reset_on_bos))
    os.environ["SIDECAR_RESET_TOKEN_ID"] = str(int(args.sidecar_reset_token_id))
    persistent_flag = 0 if args.mode == "reset" else 1 if args.mode == "persistent" else int(args.persistent or 0)
    os.environ["SIDECAR_EVAL_PERSISTENT"] = str(persistent_flag)
    os.environ["SIDECAR_EVAL_PERSIST_GROUP_SEQS"] = str(int(args.persist_group_seqs))


def load_modules(sidecar_module_name: str = "train_gpt_mlx_jepa_sidecar"):
    import train_gpt_mlx as base_mod
    sidecar_mod = importlib.import_module(sidecar_module_name)
    return importlib.reload(base_mod), importlib.reload(sidecar_mod)


def load_flat_state(artifact_path: Path, base_mod) -> dict[str, object]:
    if artifact_path.suffix == ".pklz":
        return base_mod.dequantize_state_dict(pickle.loads(zlib.decompress(artifact_path.read_bytes())))
    if artifact_path.suffix == ".npz":
        import mlx.core as mx

        npz = mx.load(str(artifact_path))
        return dict(npz.items())
    raise ValueError(f"Unsupported artifact type: {artifact_path}")


def ensure_dataset_ready(args: argparse.Namespace) -> None:
    val_probe = Path(os.path.expanduser(args.data_path)) / "fineweb_val_000000.bin"
    if val_probe.exists():
        return
    local_cwd_script = Path.cwd() / "cached_challenge_fineweb.py"
    local_peer_script = Path(__file__).resolve().parent / "cached_challenge_fineweb.py"
    repo_script = Path(__file__).resolve().parents[1] / "data" / "cached_challenge_fineweb.py"
    if local_cwd_script.exists():
        cache_script = local_cwd_script
    elif local_peer_script.exists():
        cache_script = local_peer_script
    else:
        cache_script = repo_script
    subprocess.run(
        [
            "/opt/homebrew/bin/python3",
            str(cache_script),
            "--variant",
            args.cache_variant,
            "--train-shards",
            str(args.train_shards),
        ],
        check=True,
    )


def evaluate_mode(
    *,
    mode: str,
    base_mod,
    sidecar_mod,
    hps,
    model,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    label_prefix: str | None = None,
) -> dict[str, float | str]:
    model.clear_turbo_cache()
    if mode == "persistent":
        val_loss, val_bpb = sidecar_mod.eval_val_sidecar_persistent(
            hps,
            model,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=print,
        )
        label = label_prefix or "persistent_exact"
        print(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
        return {"mode": mode, "val_loss": float(val_loss), "val_bpb": float(val_bpb)}
    if mode != "reset":
        raise ValueError(f"Unsupported eval mode {mode!r}")
    uses_logic = (
        model.logic_sidecar is not None
        or model.polarity_detector is not None
        or model.sidecar_polarity_write
    )
    if uses_logic:
        ce_loss = lambda x, y, operator_codes=None: model.loss(x, y, operator_codes)
        forward_logits = lambda x, operator_codes=None: model.forward_logits(x, operator_codes)
    else:
        ce_loss = lambda x, y, operator_codes=None: model.loss(x, y)
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
        log_fn=print,
    )
    label = label_prefix or "reset_exact"
    print(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    return {"mode": mode, "val_loss": float(val_loss), "val_bpb": float(val_bpb)}


def _log_softmax(base_mod, logits):
    mx = base_mod.mx
    logits32 = logits.astype(mx.float32)
    return logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)


def _normalize_sidecar_state(base_mod, model, state, batch_size: int):
    mx = base_mod.mx
    if state is not None:
        return state.astype(base_mod.COMPUTE_DTYPE)
    return mx.zeros((batch_size, model.sidecar_state_dim), dtype=base_mod.COMPUTE_DTYPE)


def _sidecar_param_keys(base_mod, model) -> list[str]:
    return [name for name, _ in base_mod.tree_flatten(model.parameters()) if name.startswith("sidecar")]


def _apply_sidecar_grads(base_mod, model, optimizer, sidecar_keys: list[str], grads_tree) -> None:
    params = dict(base_mod.tree_flatten(model.parameters()))
    grads = dict(base_mod.tree_flatten(grads_tree))
    selected_keys = [k for k in sidecar_keys if k in params and k in grads]
    if not selected_keys:
        return
    selected_grads = {k: grads[k] for k in selected_keys}
    selected_params = {k: params[k] for k in selected_keys}
    updated = dict(params)
    updated_subset = optimizer.apply_gradients(selected_grads, selected_params)
    updated.update(updated_subset)
    model.update(base_mod.tree_unflatten(list(updated.items())))
    base_mod.mx.eval(*updated_subset.values())


def evaluate_mode_adaptive(
    *,
    mode: str,
    args: argparse.Namespace,
    base_mod,
    sidecar_mod,
    hps,
    model,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
) -> dict[str, float | str]:
    mx = base_mod.mx
    nn = base_mod.nn
    optim = base_mod.optim
    persistent = mode == "persistent"
    eval_seq_len = hps.effective_eval_seq_len
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    carry_state = None
    sidecar_keys = _sidecar_param_keys(base_mod, model)
    sdpo_loss_sum = 0.0
    adapt_steps = 0
    optimizer = optim.Adam(
        learning_rate=args.adapt_lr,
        betas=[args.adapt_beta1, args.adapt_beta2],
        eps=1e-8,
        bias_correction=True,
    )

    def sdpo_loss(x, operator_codes, start_state):
        student_logits, student_final_state, _aux = model.forward_logits_with_sidecar_state(
            x,
            operator_codes=operator_codes,
            initial_sidecar_state=start_state,
        )
        teacher_init = mx.stop_gradient(student_final_state)
        teacher_logits, _, _ = model.forward_logits_with_sidecar_state(
            x,
            operator_codes=operator_codes,
            initial_sidecar_state=teacher_init,
        )
        student_log_probs = _log_softmax(base_mod, student_logits)
        teacher_log_probs = mx.stop_gradient(_log_softmax(base_mod, teacher_logits))
        teacher_probs = mx.stop_gradient(mx.exp(teacher_log_probs))
        return mx.mean(mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1))

    loss_and_grad = nn.value_and_grad(model, sdpo_loss)

    for seq_idx in range(total_seqs):
        raw_start = seq_idx * eval_seq_len
        raw_end = raw_start + eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1][None, :]
        y_np = chunk[1:][None, :]
        x = mx.array(x_np, dtype=mx.int32)
        operator_codes = base_mod.operator_codes_mx_for_numpy_batch(model, x_np)
        start_state = _normalize_sidecar_state(base_mod, model, carry_state if persistent else None, batch_size=1)

        logits, scored_final_state, _aux = model.forward_logits_with_sidecar_state(
            x,
            operator_codes=operator_codes,
            initial_sidecar_state=start_state,
        )
        nll = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            mx.array(y_np, dtype=mx.int32),
            reduction="none",
        ).astype(mx.float32)
        nll_sum = mx.sum(nll)
        mx.eval(nll_sum, scored_final_state)
        total_loss_sum += float(nll_sum.item())
        chunk_token_count = float(y_np.size)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())

        sdpo_value, grads = loss_and_grad(x, operator_codes, start_state)
        _apply_sidecar_grads(base_mod, model, optimizer, sidecar_keys, grads)
        sdpo_loss_sum += float(sdpo_value.item())
        adapt_steps += 1

        if persistent:
            if args.adapt_recompute_carry:
                _, carry_state, _ = model.forward_logits_with_sidecar_state(
                    x,
                    operator_codes=operator_codes,
                    initial_sidecar_state=start_state,
                )
                carry_state = mx.stop_gradient(carry_state)
            else:
                carry_state = mx.stop_gradient(scored_final_state)
        if total_seqs > 1 and (seq_idx + 1 == 1 or seq_idx + 1 == total_seqs or (seq_idx + 1) % 25 == 0):
            print(
                f"adaptive_val_progress:{seq_idx + 1}/{total_seqs} "
                f"mode:{mode} sdpo_loss:{float(sdpo_value.item()):.6f}"
            )

    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    label = "adaptive_persistent_exact" if persistent else "adaptive_reset_exact"
    print(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    return {
        "mode": f"adaptive_{mode}",
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
        "adapt_steps": int(adapt_steps),
        "mean_sdpo_loss": float(sdpo_loss_sum / max(adapt_steps, 1)),
        "adaptive_tokens": int(total_tokens),
    }


def main() -> None:
    args = parse_args()
    ensure_dataset_ready(args)
    set_env(args)
    base_mod, sidecar_mod = load_modules(args.trainer_module)
    hps = sidecar_mod.Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=hps.tokenizer_path)
    flat_state = load_flat_state(Path(args.artifact), base_mod)

    def build_loaded_model():
        model = sidecar_mod.make_sidecar_gpt(hps, sp)
        model.update(tree_unflatten(list(flat_state.items())))
        return model

    val_tokens = base_mod.limit_validation_tokens(
        base_mod.load_validation_tokens(hps.val_files, hps.train_seq_len),
        hps.train_seq_len,
        hps.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base_mod.build_sentencepiece_luts(sp, hps.vocab_size)
    if args.mode is not None:
        modes = ["reset", "persistent"] if args.mode == "both" else [args.mode]
    else:
        modes = ["persistent" if int(args.persistent or 0) else "reset"]
    results = []
    for mode in modes:
        model = build_loaded_model()
        result = (
            evaluate_mode_adaptive(
                mode=mode,
                args=args,
                base_mod=base_mod,
                sidecar_mod=sidecar_mod,
                hps=hps,
                model=model,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            if int(args.adapt_sidecar)
            else evaluate_mode(
                mode=mode,
                base_mod=base_mod,
                sidecar_mod=sidecar_mod,
                hps=hps,
                model=model,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
        )
        if int(args.adapt_sidecar) and int(args.adapt_rescore):
            rescore = evaluate_mode(
                mode=mode,
                base_mod=base_mod,
                sidecar_mod=sidecar_mod,
                hps=hps,
                model=model,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                label_prefix=f"adaptive_{mode}_rescore_exact",
            )
            result["rescore_val_loss"] = float(rescore["val_loss"])
            result["rescore_val_bpb"] = float(rescore["val_bpb"])
            result["rescore_delta_bpb"] = float(rescore["val_bpb"] - result["val_bpb"])
            result["rescore_improved"] = bool(rescore["val_bpb"] < result["val_bpb"])
        results.append(result)
    if args.result_json:
        payload = {
            "label": args.label,
            "artifact": str(Path(args.artifact)),
            "tokenizer_path": hps.tokenizer_path,
            "data_path": hps.data_path,
            "val_max_seqs": int(hps.val_max_seqs),
            "eval_seq_len": int(hps.effective_eval_seq_len),
            "persist_group_seqs": int(hps.sidecar_eval_persist_group_seqs),
            "adapt_sidecar": int(args.adapt_sidecar),
            "adapt_rescore": int(args.adapt_rescore),
            "adapt_config": {
                "adapt_lr": float(args.adapt_lr),
                "adapt_beta1": float(args.adapt_beta1),
                "adapt_beta2": float(args.adapt_beta2),
                "adapt_recompute_carry": int(args.adapt_recompute_carry),
                "trainer_module": str(args.trainer_module),
                "sidecar_transition_reset_enabled": int(args.sidecar_transition_reset_enabled),
                "sidecar_transition_reset_cosine_threshold": float(args.sidecar_transition_reset_cosine_threshold),
                "sidecar_transition_reset_cosine_sharpness": float(args.sidecar_transition_reset_cosine_sharpness),
                "sidecar_transition_reset_prior_weight": float(args.sidecar_transition_reset_prior_weight),
                "sidecar_transition_reset_learned_weight": float(args.sidecar_transition_reset_learned_weight),
                "sidecar_transition_reset_max_gate": float(args.sidecar_transition_reset_max_gate),
            },
            "results": results,
        }
        out_path = Path(args.result_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote_result_json:{out_path}")


if __name__ == "__main__":
    main()
