from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("PG_PYTHON") or sys.executable or shutil.which("python3") or "python3"

TURBO_PROFILES: dict[str, dict[str, str]] = {
    "none": {},
    "56-b64": {
        "QUANT_FORMAT": "turbo_block_v1",
        "TURBO_BLOCK_SIZE": "64",
        "TURBO_MSE_BITS": "5",
        "TURBO_PROD_BITS": "6",
        "TURBO_QAT": "1",
        "TURBO_QAT_LAMBDA": "0.01",
        "TURBO_QAT_MUON_MOMENTUM": "0.92",
        "TURBO_QAT_MUON_MOMENTUM_WARMUP_START": "0.82",
        "TURBO_QAT_RAMP_FRAC": "0.25",
        "TURBO_QAT_START_FRAC": "0.5",
    },
    "45-b128": {
        "QUANT_FORMAT": "turbo_block_v1",
        "TURBO_BLOCK_SIZE": "128",
        "TURBO_MSE_BITS": "4",
        "TURBO_PROD_BITS": "5",
        "TURBO_QAT": "1",
        "TURBO_QAT_LAMBDA": "0.01",
        "TURBO_QAT_MUON_MOMENTUM": "0.92",
        "TURBO_QAT_MUON_MOMENTUM_WARMUP_START": "0.82",
        "TURBO_QAT_RAMP_FRAC": "0.25",
        "TURBO_QAT_START_FRAC": "0.5",
    },
    "34-b256": {
        "QUANT_FORMAT": "turbo_block_v1",
        "TURBO_BLOCK_SIZE": "256",
        "TURBO_MSE_BITS": "3",
        "TURBO_PROD_BITS": "4",
        "TURBO_QAT": "1",
        "TURBO_QAT_LAMBDA": "0.01",
        "TURBO_QAT_MUON_MOMENTUM": "0.92",
        "TURBO_QAT_MUON_MOMENTUM_WARMUP_START": "0.82",
        "TURBO_QAT_RAMP_FRAC": "0.25",
        "TURBO_QAT_START_FRAC": "0.5",
    },
}


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def common_env(duration_seconds: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DATA_PATH": str(ROOT / "datasets" / "fineweb10B_sp1024"),
            "TOKENIZER_PATH": str(ROOT / "tokenizers" / "fineweb_1024_bpe.model"),
            "VOCAB_SIZE": "1024",
            "ITERATIONS": "1000000",
            "VAL_LOSS_EVERY": "0",
            "QUANT_EVAL_EVERY": "0",
            "MAX_WALLCLOCK_SECONDS": str(duration_seconds),
            "TRAIN_SEQ_LEN": "512",
            "WARMUP_STEPS": "0",
            "MLX_COMPILE": "0",
        }
    )
    return env


def _suffix(run_suffix: str) -> str:
    return f"_{run_suffix}" if run_suffix else ""


def baseline_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"baseline_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "100",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
        }
    )
    return env


def slim_current_size_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"slim_control_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "TRAIN_SEQ_LEN": "1024",
            "TRAIN_LOG_EVERY": "100",
            "VAL_BATCH_SIZE": "524288",
            "VAL_LOSS_EVERY": "0",
            "QUANT_EVAL_EVERY": "0",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "NUM_LAYERS": "9",
            "NUM_LAYER_TEMPLATES": "9",
            "MLP_LEAKY_SLOPE": "0.5",
            "MLP_MULT": "2",
            "TIE_EMBEDDINGS": "0",
            "NUM_REGISTERS": "0",
            "LOGIC_DIM": "0",
            "POLARITY_DETECTOR_ENABLED": "0",
            "EMA_ENABLED": "0",
            "EMA_DECAY": "0.999",
            "EMA_START_FRAC": "0.5",
            "EMA_RESET_ON_PHASE_TRANSITION": "0",
            "EMA_RESET_ON_QAT_FULL": "0",
            "EMA_PHASE_RESET_MIN_PROGRESS": "0.75",
            "EMA_PHASE_RESET_COOLDOWN_STEPS": "500",
            "EMA_PHASE_RESET_REQUIRES_QAT_FULL": "1",
            "PHASE_TRANSITION_LOG": "1",
            "WARMDOWN_FRACTION": "0.18",
            "QUANT_FORMAT": "turbo_block_v1",
            "TURBO_BLOCK_SIZE": "256",
            "TURBO_MSE_BITS": "3",
            "TURBO_PROD_BITS": "4",
            "TURBO_MSE_NAME_PATTERNS": "attn.c_q.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight,lm_head.weight",
            "TURBO_PROD_NAME_PATTERNS": "attn.c_k.weight",
            "TURBO_QAT": "1",
            "TURBO_QAT_LAMBDA": "0.01",
            "TURBO_QAT_MUON_MOMENTUM": "0.92",
            "TURBO_QAT_MUON_MOMENTUM_WARMUP_START": "0.82",
            "TURBO_QAT_RAMP_FRAC": "0.25",
            "TURBO_QAT_START_FRAC": "0.5",
            "TURBO_QJL_SEED": "29",
            "TURBO_ROT_SEED": "17",
            "TURBO_EMBED_EXPORT": "1",
            "LONGCTX_NUM_STREAMS": "1",
        }
    )
    return env


def slim_curriculum_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = slim_current_size_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"slim_curriculum_mlx_{platform.node()}{_suffix(run_suffix)}"
    env["CURRICULUM_ENABLED"] = "1"
    env["CURRICULUM_FEATURES_PATH"] = str(
        ROOT / "research" / "iterations" / "generated" / "curriculum_train_shard0_features.npz"
    )
    env["CURRICULUM_PHASE_PLAN_PATH"] = str(
        ROOT / "research" / "iterations" / "templates" / "curriculum_phase_plan.example.json"
    )
    env["CURRICULUM_APPLY_QAT_PHASE_GATING"] = "1"
    env["CURRICULUM_APPLY_EMA_PHASE_GATING"] = "1"
    env["CURRICULUM_APPLY_FOCAL"] = "0"
    return env


def slim_curriculum_early_exit_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = slim_curriculum_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"slim_curriculum_early_exit_mlx_{platform.node()}{_suffix(run_suffix)}"
    env["EARLY_EXIT_LAYER_INDEX"] = "3"
    env["EARLY_EXIT_HORIZONS"] = "1,2,3"
    env["EARLY_EXIT_AUX_WEIGHT"] = "0.1"
    env["EARLY_EXIT_HEAD_INIT_STD"] = "0.005"
    return env


def chainrule_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = baseline_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"chainrule_mlx_{platform.node()}{_suffix(run_suffix)}"
    env.setdefault("CHAINRULE_LEVEL_SIZES", "32,128,256,512,1024")
    env.setdefault("CHAINRULE_HIDDEN_DIMS", "32,128,256,512")
    return env


def representation_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = baseline_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"representation_mlx_{platform.node()}{_suffix(run_suffix)}"
    env.setdefault("REP_LEARN_QK_INIT", "1")
    return env


def jepa_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"jepa_mps_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "4096",
            "VAL_BATCH_SIZE": "4096",
            "TRAIN_LOG_EVERY": "50",
            "NUM_LAYERS": "6",
            "NUM_LAYER_TEMPLATES": "6",
            "MODEL_DIM": "384",
            "NUM_HEADS": "6",
            "NUM_KV_HEADS": "3",
            "MLP_MULT": "2",
            "CHUNK_SIZE": "8",
            "LATENT_DIM": "192",
            "ENCODER_HIDDEN_DIM": "768",
            "DECODER_LAYERS": "2",
            "COMPILE_MODEL": "0",
            "PRED_LOSS_WEIGHT": "0.25",
            "SIGREG_WEIGHT": "0.05",
            "DECODER_TEACHER_WEIGHT": "0.1",
        }
    )
    return env


def state_jepa_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"state_jepa_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "4096",
            "VAL_BATCH_SIZE": "4096",
            "TRAIN_LOG_EVERY": "50",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "NUM_LAYERS": "6",
            "NUM_LAYER_TEMPLATES": "6",
            "MODEL_DIM": "384",
            "NUM_HEADS": "6",
            "NUM_KV_HEADS": "3",
            "MLP_MULT": "2",
            "CHUNK_SIZE": "8",
            "STATE_LATENT_DIM": "192",
            "STATE_PRED_HIDDEN": "384",
            "STATE_PRED_WEIGHT": "0.25",
            "SIGREG_WEIGHT": "0.05",
            "SIGREG_SEED": "17",
            "COMPILE_MODEL": "0",
        }
    )
    return env


def jepa_aux_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"jepa_aux_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "50",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "JEPA_CHUNK_SIZE": "8",
            "JEPA_GEOM_DIM": "96",
            "JEPA_DYN_DIM": "32",
            "JEPA_TAP_LAYER": "3",
            "JEPA_PRED_OFFSET": "1",
            "JEPA_PRED_HIDDEN": "64",
            "JEPA_PRED_WEIGHT": "0.05",
            "JEPA_SIGREG_WEIGHT": "0.01",
            "JEPA_SPHERICAL_WEIGHT": "0.01",
            "JEPA_DYN_SPHERICAL_WEIGHT": "0.00",
            "JEPA_DYN_COV_WEIGHT": "0.00",
            "JEPA_CROSS_WEIGHT": "0.00",
            "JEPA_SUMMARY_MODE": "query",
            "JEPA_PRED_MODE": "linear",
            "JEPA_PRED_TARGET_MODE": "next",
            "JEPA_PRED_INIT_STD": "1e-4",
            "JEPA_GRAD_SCRUB_NONFINITE": "1",
            "JEPA_SIGREG_SEED": "17",
            "JEPA_SIGREG_SAMPLE_MODE": "flatten",
            "JEPA_SIGREG_RESAMPLE_PROJ": "1",
        }
    )
    return env


def jepa_sidecar_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"jepa_sidecar_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "50",
            "GRAD_ACCUM_STEPS": "8",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "SIDECAR_CHUNK_SIZE": "8",
            "SIDECAR_TAP_LAYER": "3",
            "SIDECAR_STATE_DIM": "64",
            "SIDECAR_PRED_OFFSET": "1",
            "SIDECAR_PRED_WEIGHT": "0.05",
            "SIDECAR_SIGREG_WEIGHT": "0.01",
            "SIDECAR_SPHERICAL_WEIGHT": "0.01",
            "SIDECAR_SIGREG_MODE": "full",
            "SIDECAR_WEAK_SIGREG_DIM": "32",
            "SIDECAR_READ_RMSNORM": "1",
            "SIDECAR_POLARITY_WRITE": "0",
            "SIDECAR_POLARITY_POOL": "max",
            "SIDECAR_SUMMARY_MODE": "query",
            "SIDECAR_PRED_TARGET_MODE": "delta",
            "SIDECAR_READ_INIT_STD": "1e-3",
            "SIDECAR_PRED_INIT_STD": "1e-4",
            "SIDECAR_GRAD_SCRUB_NONFINITE": "1",
            "SIDECAR_SIGREG_SEED": "17",
            "SIDECAR_SIGREG_RESAMPLE_PROJ": "1",
            "LOGIC_DIM": "0",
            "LOGIC_LAYER_INDEX": "4",
            "LOGIC_ROUTE_TO_NEXT_TOKEN": "1",
            "LOGIC_OPERATOR_MODE": "not_only",
            "POLARITY_DETECTOR_ENABLED": "0",
            "POLARITY_DETECTOR_LAYER_INDEX": "3",
            "POLARITY_DETECTOR_HIDDEN_DIM": "64",
            "POLARITY_SEED_BLEND": "1.0",
            "POLARITY_SEED_WEIGHT": "0.0",
            "POLARITY_SPARSE_WEIGHT": "0.0",
            "POLARITY_SMOOTH_WEIGHT": "0.0",
        }
    )
    return env


def jepa_sidecar_ref_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = jepa_sidecar_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"jepa_sidecar_ref_mlx_{platform.node()}{_suffix(run_suffix)}"
    return env


def superlong_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"superlong_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "25",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "LONGCTX_NUM_STREAMS": "16",
            "LONGCTX_RESET_CARRY_EVERY_STEPS": "0",
            "SIDECAR_CHUNK_SIZE": "8",
            "SIDECAR_TAP_LAYER": "3",
            "SIDECAR_STATE_DIM": "64",
            "SIDECAR_PRED_OFFSET": "1",
            "SIDECAR_PRED_WEIGHT": "0.05",
            "SIDECAR_SIGREG_WEIGHT": "0.01",
            "SIDECAR_SPHERICAL_WEIGHT": "0.01",
            "SIDECAR_SIGREG_MODE": "full",
            "SIDECAR_WEAK_SIGREG_DIM": "32",
            "SIDECAR_READ_RMSNORM": "1",
            "SIDECAR_SUMMARY_MODE": "query",
            "SIDECAR_PRED_TARGET_MODE": "delta",
            "SIDECAR_READ_INIT_STD": "1e-3",
            "SIDECAR_PRED_INIT_STD": "1e-4",
            "LOGIC_DIM": "0",
            "POLARITY_DETECTOR_ENABLED": "0",
            "SIDECAR_POLARITY_WRITE": "0",
            "SIDECAR_EVAL_PERSISTENT": "1",
            "SIDECAR_EVAL_PERSIST_GROUP_SEQS": "1",
        }
    )
    return env


def segment_write_only_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"segment_write_only_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "25",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "LONGCTX_NUM_STREAMS": "16",
            "LONGCTX_RESET_CARRY_EVERY_STEPS": "0",
            "LONGCTX_PERSIST_TRAIN_CARRY": "1",
            "LONGCTX_ENABLE_SIDECAR_READ": "0",
            "LONGCTX_SEGMENT_LEN": "64",
            "LONGCTX_SEGMENT_PRED_OFFSETS": "1,4",
            "SIDECAR_CHUNK_SIZE": "64",
            "SIDECAR_LOSS_STRIDE": "1",
            "SIDECAR_TAP_LAYER": "3",
            "SIDECAR_STATE_DIM": "64",
            "SIDECAR_PRED_OFFSET": "1",
            "SIDECAR_PRED_WEIGHT": "0.01",
            "SIDECAR_SIGREG_WEIGHT": "0.0",
            "SIDECAR_SPHERICAL_WEIGHT": "0.0",
            "SIDECAR_SIGREG_MODE": "none",
            "SIDECAR_WEAK_SIGREG_DIM": "32",
            "SIDECAR_READ_RMSNORM": "1",
            "SIDECAR_SUMMARY_MODE": "mean",
            "SIDECAR_PRED_TARGET_MODE": "delta",
            "SIDECAR_READ_INIT_STD": "1e-3",
            "SIDECAR_PRED_INIT_STD": "1e-4",
            "LOGIC_DIM": "0",
            "POLARITY_DETECTOR_ENABLED": "0",
            "SIDECAR_POLARITY_WRITE": "0",
            "SIDECAR_EVAL_PERSISTENT": "0",
            "SIDECAR_EVAL_PERSIST_GROUP_SEQS": "1",
        }
    )
    return env


def segment_prev_read_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = segment_write_only_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"segment_prev_read_mlx_{platform.node()}{_suffix(run_suffix)}"
    env["LONGCTX_ENABLE_SIDECAR_READ"] = "1"
    return env


def harmonic_local_only_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = common_env(duration_seconds)
    env.update(
        {
            "RUN_ID": f"harmonic_local_only_mlx_{platform.node()}{_suffix(run_suffix)}",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "8192",
            "TRAIN_LOG_EVERY": "25",
            "GRAD_ACCUM_STEPS": "1",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "LONGCTX_NUM_STREAMS": "16",
            "HARM_PATCH_LEN": "8",
            "HARM_TAP_LAYER": "3",
            "HARM_CHORD_DIM": "64",
            "HARM_MIN_PATCHES_PER_CHORD": "1",
            "HARM_MAX_PATCHES_PER_CHORD": "8",
            "HARM_BOUNDARY_THRESHOLD": "0.12",
            "HARM_BANK_SIZE": "32",
            "HARM_READ_TOPK": "0",
            "HARM_ENABLE_READ": "0",
            "HARM_JEPA_WEIGHT": "0.0",
            "HARM_PRED_OFFSET": "1",
            "HARM_READ_INIT_STD": "1e-3",
            "HARM_PRED_INIT_STD": "1e-4",
            "LOGIC_DIM": "0",
            "POLARITY_DETECTOR_ENABLED": "0",
            "NUM_REGISTERS": "0",
            "MLX_COMPILE": "0",
        }
    )
    return env


def harmonic_prev_read_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = harmonic_local_only_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"harmonic_prev_read_mlx_{platform.node()}{_suffix(run_suffix)}"
    env["HARM_ENABLE_READ"] = "1"
    return env


def harmonic_prev_read_jepa_env(duration_seconds: int, run_suffix: str = "") -> dict[str, str]:
    env = harmonic_prev_read_env(duration_seconds, run_suffix)
    env["RUN_ID"] = f"harmonic_prev_read_jepa_mlx_{platform.node()}{_suffix(run_suffix)}"
    env["HARM_JEPA_WEIGHT"] = "0.01"
    return env


def apply_overrides(
    env: dict[str, str],
    *,
    turbo_profile: str,
    val_max_seqs: int,
    quant_eval_max_seqs: int,
    mlx_compile: str | None,
    jepa_geom_dim: int | None,
    jepa_dyn_dim: int | None,
    jepa_tap_layer: int | None,
    jepa_pred_weight: float | None,
    jepa_pred_offset: int | None,
    jepa_pred_end_weight: float | None,
    jepa_pred_decay_start_frac: float | None,
    jepa_pred_decay_end_frac: float | None,
    jepa_sigreg_weight: float | None,
    jepa_spherical_weight: float | None,
    jepa_dyn_spherical_weight: float | None,
    jepa_dyn_cov_weight: float | None,
    jepa_cross_weight: float | None,
    jepa_sigreg_sample_mode: str | None,
    jepa_sigreg_resample_proj: int | None,
    jepa_aux_start_frac: float | None,
    jepa_aux_ramp_frac: float | None,
    jepa_summary_mode: str | None,
    jepa_pred_mode: str | None,
    jepa_pred_target_mode: str | None,
    jepa_pred_init_std: float | None,
    jepa_grad_scrub_nonfinite: int | None,
) -> dict[str, str]:
    out = dict(env)
    if turbo_profile not in TURBO_PROFILES:
        raise ValueError(f"Unknown turbo profile: {turbo_profile}")
    out.update(TURBO_PROFILES[turbo_profile])
    if val_max_seqs > 0:
        out["VAL_MAX_SEQS"] = str(val_max_seqs)
    if quant_eval_max_seqs > 0:
        out["QUANT_EVAL_MAX_SEQS"] = str(quant_eval_max_seqs)
    if mlx_compile is not None:
        out["MLX_COMPILE"] = mlx_compile
    if jepa_geom_dim is not None:
        out["JEPA_GEOM_DIM"] = str(int(jepa_geom_dim))
    if jepa_dyn_dim is not None:
        out["JEPA_DYN_DIM"] = str(int(jepa_dyn_dim))
    if jepa_tap_layer is not None:
        out["JEPA_TAP_LAYER"] = str(int(jepa_tap_layer))
    if jepa_pred_weight is not None:
        out["JEPA_PRED_WEIGHT"] = str(jepa_pred_weight)
    if jepa_pred_offset is not None:
        out["JEPA_PRED_OFFSET"] = str(int(jepa_pred_offset))
    if jepa_pred_end_weight is not None:
        out["JEPA_PRED_END_WEIGHT"] = str(jepa_pred_end_weight)
    if jepa_pred_decay_start_frac is not None:
        out["JEPA_PRED_DECAY_START_FRAC"] = str(jepa_pred_decay_start_frac)
    if jepa_pred_decay_end_frac is not None:
        out["JEPA_PRED_DECAY_END_FRAC"] = str(jepa_pred_decay_end_frac)
    if jepa_sigreg_weight is not None:
        out["JEPA_SIGREG_WEIGHT"] = str(jepa_sigreg_weight)
    if jepa_spherical_weight is not None:
        out["JEPA_SPHERICAL_WEIGHT"] = str(jepa_spherical_weight)
    if jepa_dyn_spherical_weight is not None:
        out["JEPA_DYN_SPHERICAL_WEIGHT"] = str(jepa_dyn_spherical_weight)
    if jepa_dyn_cov_weight is not None:
        out["JEPA_DYN_COV_WEIGHT"] = str(jepa_dyn_cov_weight)
    if jepa_cross_weight is not None:
        out["JEPA_CROSS_WEIGHT"] = str(jepa_cross_weight)
    if jepa_sigreg_sample_mode is not None:
        out["JEPA_SIGREG_SAMPLE_MODE"] = jepa_sigreg_sample_mode
    if jepa_sigreg_resample_proj is not None:
        out["JEPA_SIGREG_RESAMPLE_PROJ"] = str(int(jepa_sigreg_resample_proj))
    if jepa_aux_start_frac is not None:
        out["JEPA_AUX_START_FRAC"] = str(jepa_aux_start_frac)
    if jepa_aux_ramp_frac is not None:
        out["JEPA_AUX_RAMP_FRAC"] = str(jepa_aux_ramp_frac)
    if jepa_summary_mode is not None:
        out["JEPA_SUMMARY_MODE"] = jepa_summary_mode
    if jepa_pred_mode is not None:
        out["JEPA_PRED_MODE"] = jepa_pred_mode
    if jepa_pred_target_mode is not None:
        out["JEPA_PRED_TARGET_MODE"] = jepa_pred_target_mode
    if jepa_pred_init_std is not None:
        out["JEPA_PRED_INIT_STD"] = str(jepa_pred_init_std)
    if jepa_grad_scrub_nonfinite is not None:
        out["JEPA_GRAD_SCRUB_NONFINITE"] = str(int(jepa_grad_scrub_nonfinite))
    return out


def apply_sidecar_alias_overrides(
    env: dict[str, str],
    *,
    jepa_tap_layer: int | None,
    jepa_pred_weight: float | None,
    jepa_pred_offset: int | None,
    jepa_sigreg_weight: float | None,
    jepa_spherical_weight: float | None,
    jepa_aux_start_frac: float | None,
    jepa_aux_ramp_frac: float | None,
    jepa_summary_mode: str | None,
    jepa_pred_target_mode: str | None,
    jepa_pred_init_std: float | None,
    jepa_dyn_dim: int | None,
    jepa_grad_scrub_nonfinite: int | None,
    jepa_sigreg_resample_proj: int | None,
    jepa_sigreg_mode: str | None,
    jepa_weak_sigreg_dim: int | None,
    jepa_read_rmsnorm: int | None,
    jepa_sidecar_eval_persistent: int | None,
    jepa_sidecar_eval_persist_group_seqs: int | None,
    jepa_sidecar_polarity_write: int | None,
    jepa_sidecar_polarity_pool: str | None,
    logic_dim: int | None,
    logic_layer_index: int | None,
    logic_route_to_next_token: int | None,
    logic_operator_mode: str | None,
    polarity_detector_enabled: int | None,
    polarity_detector_layer_index: int | None,
    polarity_detector_hidden_dim: int | None,
    polarity_seed_blend: float | None,
    polarity_seed_weight: float | None,
    polarity_sparse_weight: float | None,
    polarity_smooth_weight: float | None,
) -> dict[str, str]:
    out = dict(env)
    if jepa_tap_layer is not None:
        out["SIDECAR_TAP_LAYER"] = str(int(jepa_tap_layer))
    if jepa_pred_weight is not None:
        out["SIDECAR_PRED_WEIGHT"] = str(jepa_pred_weight)
    if jepa_pred_offset is not None:
        out["SIDECAR_PRED_OFFSET"] = str(int(jepa_pred_offset))
    if jepa_sigreg_weight is not None:
        out["SIDECAR_SIGREG_WEIGHT"] = str(jepa_sigreg_weight)
    if jepa_spherical_weight is not None:
        out["SIDECAR_SPHERICAL_WEIGHT"] = str(jepa_spherical_weight)
    if jepa_aux_start_frac is not None:
        out["SIDECAR_AUX_START_FRAC"] = str(jepa_aux_start_frac)
    if jepa_aux_ramp_frac is not None:
        out["SIDECAR_AUX_RAMP_FRAC"] = str(jepa_aux_ramp_frac)
    if jepa_summary_mode is not None:
        out["SIDECAR_SUMMARY_MODE"] = jepa_summary_mode
    if jepa_pred_target_mode is not None:
        out["SIDECAR_PRED_TARGET_MODE"] = jepa_pred_target_mode
    if jepa_pred_init_std is not None:
        out["SIDECAR_PRED_INIT_STD"] = str(jepa_pred_init_std)
    if jepa_dyn_dim is not None:
        out["SIDECAR_STATE_DIM"] = str(int(jepa_dyn_dim))
    if jepa_grad_scrub_nonfinite is not None:
        out["SIDECAR_GRAD_SCRUB_NONFINITE"] = str(int(jepa_grad_scrub_nonfinite))
    if jepa_sigreg_resample_proj is not None:
        out["SIDECAR_SIGREG_RESAMPLE_PROJ"] = str(int(jepa_sigreg_resample_proj))
    if jepa_sigreg_mode is not None:
        out["SIDECAR_SIGREG_MODE"] = jepa_sigreg_mode
    if jepa_weak_sigreg_dim is not None:
        out["SIDECAR_WEAK_SIGREG_DIM"] = str(int(jepa_weak_sigreg_dim))
    if jepa_read_rmsnorm is not None:
        out["SIDECAR_READ_RMSNORM"] = str(int(jepa_read_rmsnorm))
    if jepa_sidecar_eval_persistent is not None:
        out["SIDECAR_EVAL_PERSISTENT"] = str(int(jepa_sidecar_eval_persistent))
    if jepa_sidecar_eval_persist_group_seqs is not None:
        out["SIDECAR_EVAL_PERSIST_GROUP_SEQS"] = str(int(jepa_sidecar_eval_persist_group_seqs))
    if jepa_sidecar_polarity_write is not None:
        out["SIDECAR_POLARITY_WRITE"] = str(int(jepa_sidecar_polarity_write))
    if jepa_sidecar_polarity_pool is not None:
        out["SIDECAR_POLARITY_POOL"] = jepa_sidecar_polarity_pool
    if logic_dim is not None:
        out["LOGIC_DIM"] = str(int(logic_dim))
    if logic_layer_index is not None:
        out["LOGIC_LAYER_INDEX"] = str(int(logic_layer_index))
    if logic_route_to_next_token is not None:
        out["LOGIC_ROUTE_TO_NEXT_TOKEN"] = str(int(logic_route_to_next_token))
    if logic_operator_mode is not None:
        out["LOGIC_OPERATOR_MODE"] = logic_operator_mode
    if polarity_detector_enabled is not None:
        out["POLARITY_DETECTOR_ENABLED"] = str(int(polarity_detector_enabled))
    if polarity_detector_layer_index is not None:
        out["POLARITY_DETECTOR_LAYER_INDEX"] = str(int(polarity_detector_layer_index))
    if polarity_detector_hidden_dim is not None:
        out["POLARITY_DETECTOR_HIDDEN_DIM"] = str(int(polarity_detector_hidden_dim))
    if polarity_seed_blend is not None:
        out["POLARITY_SEED_BLEND"] = str(polarity_seed_blend)
    if polarity_seed_weight is not None:
        out["POLARITY_SEED_WEIGHT"] = str(polarity_seed_weight)
    if polarity_sparse_weight is not None:
        out["POLARITY_SPARSE_WEIGHT"] = str(polarity_sparse_weight)
    if polarity_smooth_weight is not None:
        out["POLARITY_SMOOTH_WEIGHT"] = str(polarity_smooth_weight)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a mini-cluster Parameter Golf run")
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "slim_control",
            "slim_curriculum",
            "slim_curriculum_early_exit",
            "chainrule",
            "representation",
            "jepa",
            "state_jepa",
            "jepa_aux",
            "jepa_sidecar",
            "jepa_sidecar_ref",
            "superlong",
            "segment_write_only",
            "segment_prev_read",
            "harmonic_local_only",
            "harmonic_prev_read",
            "harmonic_prev_read_jepa",
        ),
        required=True,
    )
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--turbo-profile", choices=tuple(TURBO_PROFILES), default="none")
    parser.add_argument("--val-max-seqs", type=int, default=0)
    parser.add_argument("--quant-eval-max-seqs", type=int, default=0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--mlx-compile", default=None)
    parser.add_argument("--jepa-geom-dim", type=int, default=None)
    parser.add_argument("--jepa-dyn-dim", type=int, default=None)
    parser.add_argument("--jepa-tap-layer", type=int, default=None)
    parser.add_argument("--jepa-pred-weight", type=float, default=None)
    parser.add_argument("--jepa-pred-offset", type=int, default=None)
    parser.add_argument("--jepa-pred-end-weight", type=float, default=None)
    parser.add_argument("--jepa-pred-decay-start-frac", type=float, default=None)
    parser.add_argument("--jepa-pred-decay-end-frac", type=float, default=None)
    parser.add_argument("--jepa-sigreg-weight", type=float, default=None)
    parser.add_argument("--jepa-spherical-weight", type=float, default=None)
    parser.add_argument("--jepa-dyn-spherical-weight", type=float, default=None)
    parser.add_argument("--jepa-dyn-cov-weight", type=float, default=None)
    parser.add_argument("--jepa-cross-weight", type=float, default=None)
    parser.add_argument("--jepa-sigreg-sample-mode", default=None)
    parser.add_argument("--jepa-sigreg-resample-proj", type=int, default=None)
    parser.add_argument("--jepa-sigreg-mode", default=None)
    parser.add_argument("--jepa-weak-sigreg-dim", type=int, default=None)
    parser.add_argument("--jepa-read-rmsnorm", type=int, default=None)
    parser.add_argument("--jepa-sidecar-eval-persistent", type=int, default=None)
    parser.add_argument("--jepa-sidecar-eval-persist-group-seqs", type=int, default=None)
    parser.add_argument("--jepa-sidecar-polarity-write", type=int, default=None)
    parser.add_argument("--jepa-sidecar-polarity-pool", default=None)
    parser.add_argument("--logic-dim", type=int, default=None)
    parser.add_argument("--logic-layer-index", type=int, default=None)
    parser.add_argument("--logic-route-to-next-token", type=int, default=None)
    parser.add_argument("--logic-operator-mode", default=None)
    parser.add_argument("--polarity-detector-enabled", type=int, default=None)
    parser.add_argument("--polarity-detector-layer-index", type=int, default=None)
    parser.add_argument("--polarity-detector-hidden-dim", type=int, default=None)
    parser.add_argument("--polarity-seed-blend", type=float, default=None)
    parser.add_argument("--polarity-seed-weight", type=float, default=None)
    parser.add_argument("--polarity-sparse-weight", type=float, default=None)
    parser.add_argument("--polarity-smooth-weight", type=float, default=None)
    parser.add_argument("--jepa-aux-start-frac", type=float, default=None)
    parser.add_argument("--jepa-aux-ramp-frac", type=float, default=None)
    parser.add_argument("--jepa-summary-mode", default=None)
    parser.add_argument("--jepa-pred-mode", default=None)
    parser.add_argument("--jepa-pred-target-mode", default=None)
    parser.add_argument("--jepa-pred-init-std", type=float, default=None)
    parser.add_argument("--jepa-grad-scrub-nonfinite", type=int, default=None)
    parser.add_argument("--rep-learn-priors-path", default=None)
    parser.add_argument("--rep-learn-init-strength", type=float, default=None)
    parser.add_argument("--rep-learn-qk-init", type=int, default=None)
    parser.add_argument("--rep-learn-init-targets", default=None)
    parser.add_argument("--rep-learn-adapter-mode", default=None)
    parser.add_argument("support_files", nargs="*")
    args = parser.parse_args()

    run(
        [
            PYTHON,
            str(ROOT / "data" / "cached_challenge_fineweb.py"),
            "--variant",
            "sp1024",
            "--train-shards",
            str(args.train_shards),
        ]
    )

    if args.mode == "baseline":
        run(
            [PYTHON, "train_gpt_mlx.py"],
            env=apply_overrides(
                baseline_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
        )
        return
    if args.mode == "slim_control":
        run(
            [PYTHON, "train_gpt_mlx_grouped_slim.py"],
            env=apply_overrides(
                slim_current_size_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    if args.mode == "slim_curriculum":
        run(
            [PYTHON, "train_gpt_mlx_grouped_slim.py"],
            env=apply_overrides(
                slim_curriculum_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    if args.mode == "slim_curriculum_early_exit":
        run(
            [PYTHON, "train_gpt_mlx_grouped_slim.py"],
            env=apply_overrides(
                slim_curriculum_early_exit_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    if args.mode == "chainrule":
        run(
            [PYTHON, "train_gpt_mlx_chainrule.py"],
            env=apply_overrides(
                chainrule_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
        )
        return
    if args.mode == "representation":
        if (args.rep_learn_qk_init is None or int(args.rep_learn_qk_init) != 0) and not args.rep_learn_priors_path:
            raise SystemExit("--mode representation requires --rep-learn-priors-path unless --rep-learn-qk-init=0")
        rep_env = apply_overrides(
            representation_env(args.duration_seconds, args.run_suffix),
            turbo_profile=args.turbo_profile,
            val_max_seqs=args.val_max_seqs,
            quant_eval_max_seqs=args.quant_eval_max_seqs,
            mlx_compile=args.mlx_compile,
            jepa_geom_dim=args.jepa_geom_dim,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_pred_end_weight=args.jepa_pred_end_weight,
            jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
            jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
            jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
            jepa_cross_weight=args.jepa_cross_weight,
            jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_mode=args.jepa_pred_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
        )
        if args.rep_learn_priors_path is not None:
            rep_env["REP_LEARN_PRIORS_PATH"] = args.rep_learn_priors_path
        if args.rep_learn_init_strength is not None:
            rep_env["REP_LEARN_INIT_STRENGTH"] = str(args.rep_learn_init_strength)
        if args.rep_learn_qk_init is not None:
            rep_env["REP_LEARN_QK_INIT"] = str(int(args.rep_learn_qk_init))
        if args.rep_learn_init_targets is not None:
            rep_env["REP_LEARN_INIT_TARGETS"] = args.rep_learn_init_targets
        if args.rep_learn_adapter_mode is not None:
            rep_env["REP_LEARN_ADAPTER_MODE"] = args.rep_learn_adapter_mode
        run(
            [PYTHON, "train_gpt_mlx_representation.py"],
            env=rep_env,
        )
        return
    if args.mode == "jepa":
        run(
            [PYTHON, "train_jepa.py"],
            env=apply_overrides(
                jepa_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
        )
        return
    if args.mode == "state_jepa":
        run(
            [PYTHON, "train_state_jepa_mlx.py"],
            env=apply_overrides(
                state_jepa_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
        )
        return
    if args.mode == "jepa_sidecar":
        sidecar_env = apply_sidecar_alias_overrides(
            apply_overrides(
                jepa_sidecar_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_sigreg_mode=args.jepa_sigreg_mode,
            jepa_weak_sigreg_dim=args.jepa_weak_sigreg_dim,
            jepa_read_rmsnorm=args.jepa_read_rmsnorm,
            jepa_sidecar_eval_persistent=args.jepa_sidecar_eval_persistent,
            jepa_sidecar_eval_persist_group_seqs=args.jepa_sidecar_eval_persist_group_seqs,
            jepa_sidecar_polarity_write=args.jepa_sidecar_polarity_write,
            jepa_sidecar_polarity_pool=args.jepa_sidecar_polarity_pool,
            logic_dim=args.logic_dim,
            logic_layer_index=args.logic_layer_index,
            logic_route_to_next_token=args.logic_route_to_next_token,
            logic_operator_mode=args.logic_operator_mode,
            polarity_detector_enabled=args.polarity_detector_enabled,
            polarity_detector_layer_index=args.polarity_detector_layer_index,
            polarity_detector_hidden_dim=args.polarity_detector_hidden_dim,
            polarity_seed_blend=args.polarity_seed_blend,
            polarity_seed_weight=args.polarity_seed_weight,
            polarity_sparse_weight=args.polarity_sparse_weight,
            polarity_smooth_weight=args.polarity_smooth_weight,
        )
        run(
            [PYTHON, "train_gpt_mlx_jepa_sidecar.py"],
            env=sidecar_env,
        )
        return
    if args.mode == "jepa_sidecar_ref":
        sidecar_ref_env = apply_sidecar_alias_overrides(
            apply_overrides(
                jepa_sidecar_ref_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_sigreg_mode=args.jepa_sigreg_mode,
            jepa_weak_sigreg_dim=args.jepa_weak_sigreg_dim,
            jepa_read_rmsnorm=args.jepa_read_rmsnorm,
            jepa_sidecar_eval_persistent=args.jepa_sidecar_eval_persistent,
            jepa_sidecar_eval_persist_group_seqs=args.jepa_sidecar_eval_persist_group_seqs,
            jepa_sidecar_polarity_write=args.jepa_sidecar_polarity_write,
            jepa_sidecar_polarity_pool=args.jepa_sidecar_polarity_pool,
            logic_dim=args.logic_dim,
            logic_layer_index=args.logic_layer_index,
            logic_route_to_next_token=args.logic_route_to_next_token,
            logic_operator_mode=args.logic_operator_mode,
            polarity_detector_enabled=args.polarity_detector_enabled,
            polarity_detector_layer_index=args.polarity_detector_layer_index,
            polarity_detector_hidden_dim=args.polarity_detector_hidden_dim,
            polarity_seed_blend=args.polarity_seed_blend,
            polarity_seed_weight=args.polarity_seed_weight,
            polarity_sparse_weight=args.polarity_sparse_weight,
            polarity_smooth_weight=args.polarity_smooth_weight,
        )
        run(
            [PYTHON, "train_gpt_mlx_jepa_sidecar_ref.py"],
            env=sidecar_ref_env,
        )
        return
    if args.mode == "superlong":
        long_env = apply_sidecar_alias_overrides(
            apply_overrides(
                superlong_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_sigreg_mode=args.jepa_sigreg_mode,
            jepa_weak_sigreg_dim=args.jepa_weak_sigreg_dim,
            jepa_read_rmsnorm=args.jepa_read_rmsnorm,
            jepa_sidecar_eval_persistent=args.jepa_sidecar_eval_persistent,
            jepa_sidecar_eval_persist_group_seqs=args.jepa_sidecar_eval_persist_group_seqs,
            jepa_sidecar_polarity_write=0,
            jepa_sidecar_polarity_pool=None,
            logic_dim=0,
            logic_layer_index=None,
            logic_route_to_next_token=None,
            logic_operator_mode=None,
            polarity_detector_enabled=0,
            polarity_detector_layer_index=None,
            polarity_detector_hidden_dim=None,
            polarity_seed_blend=None,
            polarity_seed_weight=0.0,
            polarity_sparse_weight=0.0,
            polarity_smooth_weight=0.0,
        )
        run(
            [PYTHON, "train_gpt_mlx_superlong.py"],
            env=long_env,
        )
        return
    if args.mode == "segment_write_only":
        segment_env = apply_sidecar_alias_overrides(
            apply_overrides(
                segment_write_only_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_sigreg_mode=args.jepa_sigreg_mode,
            jepa_weak_sigreg_dim=args.jepa_weak_sigreg_dim,
            jepa_read_rmsnorm=args.jepa_read_rmsnorm,
            jepa_sidecar_eval_persistent=args.jepa_sidecar_eval_persistent,
            jepa_sidecar_eval_persist_group_seqs=args.jepa_sidecar_eval_persist_group_seqs,
            jepa_sidecar_polarity_write=0,
            jepa_sidecar_polarity_pool=None,
            logic_dim=0,
            logic_layer_index=None,
            logic_route_to_next_token=None,
            logic_operator_mode=None,
            polarity_detector_enabled=0,
            polarity_detector_layer_index=None,
            polarity_detector_hidden_dim=None,
            polarity_seed_blend=None,
            polarity_seed_weight=0.0,
            polarity_sparse_weight=0.0,
            polarity_smooth_weight=0.0,
        )
        run(
            [PYTHON, "train_gpt_mlx_segmentlong.py"],
            env=segment_env,
        )
        return
    if args.mode == "segment_prev_read":
        segment_env = apply_sidecar_alias_overrides(
            apply_overrides(
                segment_prev_read_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=args.jepa_geom_dim,
                jepa_dyn_dim=args.jepa_dyn_dim,
                jepa_tap_layer=args.jepa_tap_layer,
                jepa_pred_weight=args.jepa_pred_weight,
                jepa_pred_offset=args.jepa_pred_offset,
                jepa_pred_end_weight=args.jepa_pred_end_weight,
                jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
                jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
                jepa_sigreg_weight=args.jepa_sigreg_weight,
                jepa_spherical_weight=args.jepa_spherical_weight,
                jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
                jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
                jepa_cross_weight=args.jepa_cross_weight,
                jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
                jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
                jepa_aux_start_frac=args.jepa_aux_start_frac,
                jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
                jepa_summary_mode=args.jepa_summary_mode,
                jepa_pred_mode=args.jepa_pred_mode,
                jepa_pred_target_mode=args.jepa_pred_target_mode,
                jepa_pred_init_std=args.jepa_pred_init_std,
                jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            ),
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_sigreg_mode=args.jepa_sigreg_mode,
            jepa_weak_sigreg_dim=args.jepa_weak_sigreg_dim,
            jepa_read_rmsnorm=args.jepa_read_rmsnorm,
            jepa_sidecar_eval_persistent=args.jepa_sidecar_eval_persistent,
            jepa_sidecar_eval_persist_group_seqs=args.jepa_sidecar_eval_persist_group_seqs,
            jepa_sidecar_polarity_write=0,
            jepa_sidecar_polarity_pool=None,
            logic_dim=0,
            logic_layer_index=None,
            logic_route_to_next_token=None,
            logic_operator_mode=None,
            polarity_detector_enabled=0,
            polarity_detector_layer_index=None,
            polarity_detector_hidden_dim=None,
            polarity_seed_blend=None,
            polarity_seed_weight=0.0,
            polarity_sparse_weight=0.0,
            polarity_smooth_weight=0.0,
        )
        run(
            [PYTHON, "train_gpt_mlx_segmentlong.py"],
            env=segment_env,
        )
        return
    if args.mode == "harmonic_local_only":
        run(
            [PYTHON, "train_gpt_mlx_harmonic.py"],
            env=apply_overrides(
                harmonic_local_only_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    if args.mode == "harmonic_prev_read":
        run(
            [PYTHON, "train_gpt_mlx_harmonic.py"],
            env=apply_overrides(
                harmonic_prev_read_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    if args.mode == "harmonic_prev_read_jepa":
        run(
            [PYTHON, "train_gpt_mlx_harmonic.py"],
            env=apply_overrides(
                harmonic_prev_read_jepa_env(args.duration_seconds, args.run_suffix),
                turbo_profile=args.turbo_profile,
                val_max_seqs=args.val_max_seqs,
                quant_eval_max_seqs=args.quant_eval_max_seqs,
                mlx_compile=args.mlx_compile,
                jepa_geom_dim=None,
                jepa_dyn_dim=None,
                jepa_tap_layer=None,
                jepa_pred_weight=None,
                jepa_pred_offset=None,
                jepa_pred_end_weight=None,
                jepa_pred_decay_start_frac=None,
                jepa_pred_decay_end_frac=None,
                jepa_sigreg_weight=None,
                jepa_spherical_weight=None,
                jepa_dyn_spherical_weight=None,
                jepa_dyn_cov_weight=None,
                jepa_cross_weight=None,
                jepa_sigreg_sample_mode=None,
                jepa_sigreg_resample_proj=None,
                jepa_aux_start_frac=None,
                jepa_aux_ramp_frac=None,
                jepa_summary_mode=None,
                jepa_pred_mode=None,
                jepa_pred_target_mode=None,
                jepa_pred_init_std=None,
                jepa_grad_scrub_nonfinite=None,
            ),
        )
        return
    run(
        [PYTHON, "train_gpt_mlx_jepa_aux.py"],
        env=apply_overrides(
            jepa_aux_env(args.duration_seconds, args.run_suffix),
            turbo_profile=args.turbo_profile,
            val_max_seqs=args.val_max_seqs,
            quant_eval_max_seqs=args.quant_eval_max_seqs,
            mlx_compile=args.mlx_compile,
            jepa_geom_dim=args.jepa_geom_dim,
            jepa_dyn_dim=args.jepa_dyn_dim,
            jepa_tap_layer=args.jepa_tap_layer,
            jepa_pred_weight=args.jepa_pred_weight,
            jepa_pred_offset=args.jepa_pred_offset,
            jepa_pred_end_weight=args.jepa_pred_end_weight,
            jepa_pred_decay_start_frac=args.jepa_pred_decay_start_frac,
            jepa_pred_decay_end_frac=args.jepa_pred_decay_end_frac,
            jepa_sigreg_weight=args.jepa_sigreg_weight,
            jepa_spherical_weight=args.jepa_spherical_weight,
            jepa_dyn_spherical_weight=args.jepa_dyn_spherical_weight,
            jepa_dyn_cov_weight=args.jepa_dyn_cov_weight,
            jepa_cross_weight=args.jepa_cross_weight,
            jepa_sigreg_sample_mode=args.jepa_sigreg_sample_mode,
            jepa_sigreg_resample_proj=args.jepa_sigreg_resample_proj,
            jepa_aux_start_frac=args.jepa_aux_start_frac,
            jepa_aux_ramp_frac=args.jepa_aux_ramp_frac,
            jepa_summary_mode=args.jepa_summary_mode,
            jepa_pred_mode=args.jepa_pred_mode,
            jepa_pred_target_mode=args.jepa_pred_target_mode,
            jepa_pred_init_std=args.jepa_pred_init_std,
            jepa_grad_scrub_nonfinite=args.jepa_grad_scrub_nonfinite,
        ),
    )


if __name__ == "__main__":
    main()
