#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import resource
import sys
import time
import uuid
import zlib
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from early_exit_aux import (
    EarlyExitBudgetController,
    derive_early_exit_aux_weight,
    horizon_shift,
    parse_horizons,
    select_contiguous_draft_horizons,
)
from token_category_weighting import (
    TokenCategoryLuts,
    TokenCategoryWeightingConfig,
    build_token_category_luts,
    compute_token_category_weights,
)
from token_context_weighting import ContextDeltaWeightingConfig
from sequential_data_filter import (
    SequentialCompressibilityFilterConfig,
    keep_chunk as keep_sequential_chunk,
)
from structural_branching import (
    adaptive_branch_length_from_divergence,
    StructuralBranchBudgetController,
    StructuralBranchBudgetSignals,
    StructuralBranchPoint,
    StructuralBranchingConfig,
    derive_structural_branching_config,
    select_structural_branch_points_np,
)
from text_prosody_features import (
    BOUNDARY_STRENGTH_NAMES,
    PROSODY_BINARY_FEATURE_NAMES,
    PUNCTUATION_ROLE_NAMES,
    PUNCTUATION_ROLE_TO_ID,
    TOKEN_CLASS_NAMES,
    TokenProsodyLuts,
    build_token_prosody_luts,
)
from replay_signal_runtime import FileReplayBuffer, ReplayExample, ReplayMixTokenLoader
from snapshot_signal_runtime import StudentSnapshotRuntime
from teacher_signal_runtime import FileTeacherHiddenCache, TeacherHiddenExample, teacher_window_key, window_tokens_from_xy

fallback_root = Path.home() / "transformer_research" / "parameter-golf"

try:
    from logic_register_mlx import (
        HardmaxStructuralController,
        LogicSideCar,
        OperatorRoutingSpec,
        PolarityDetector,
        RegisterTokens,
        StaticStructuralAdapter,
        build_operator_routing_spec,
        build_register_attention_mask,
        build_register_attention_mask_with_mode,
        detect_operator_codes_np,
        interleave_operator_codes,
        pad_operator_codes,
        register_position_mask,
        route_operator_codes,
        strip_register_positions,
    )
except ModuleNotFoundError as exc:
    if exc.name != "logic_register_mlx" or not (fallback_root / "logic_register_mlx.py").is_file():
        raise
    sys.path.insert(0, str(fallback_root))
    from logic_register_mlx import (  # type: ignore[no-redef]
        HardmaxStructuralController,
        LogicSideCar,
        OperatorRoutingSpec,
        PolarityDetector,
        RegisterTokens,
        StaticStructuralAdapter,
        build_operator_routing_spec,
        build_register_attention_mask,
        build_register_attention_mask_with_mode,
        detect_operator_codes_np,
        interleave_operator_codes,
        pad_operator_codes,
        register_position_mask,
        route_operator_codes,
        strip_register_positions,
    )

try:
    from turbo_quant_mlx import (
        TURBO_CODEBOOK_KIND,
        TURBO_QJL_KIND,
        TURBO_ROTATION_KIND,
        TURBO_SCHEME_VERSION,
        TurboLinear,
        configure as configure_turbo_quant,
        dequantize_turbo_tensor,
        infer_turbo_mode,
        turbo_quantize_dequantize_array,
    )
except ModuleNotFoundError as exc:
    if exc.name != "turbo_quant_mlx" or not (fallback_root / "turbo_quant_mlx.py").is_file():
        raise
    sys.path.insert(0, str(fallback_root))
    from turbo_quant_mlx import (  # type: ignore[no-redef]
        TURBO_CODEBOOK_KIND,
        TURBO_QJL_KIND,
        TURBO_ROTATION_KIND,
        TURBO_SCHEME_VERSION,
        TurboLinear,
        configure as configure_turbo_quant,
        dequantize_turbo_tensor,
        infer_turbo_mode,
        turbo_quantize_dequantize_array,
    )

try:
    from ternary_quant_mlx import dequantize_ternary_tensor
except ModuleNotFoundError as exc:
    if exc.name != "ternary_quant_mlx" or not (fallback_root / "ternary_quant_mlx.py").is_file():
        raise
    sys.path.insert(0, str(fallback_root))
    from ternary_quant_mlx import dequantize_ternary_tensor  # type: ignore[no-redef]

try:
    from apollo_mlx import ApolloMatrixOptimizer
except ModuleNotFoundError as exc:
    if exc.name != "apollo_mlx" or not (fallback_root / "apollo_mlx.py").is_file():
        raise
    sys.path.insert(0, str(fallback_root))
    from apollo_mlx import ApolloMatrixOptimizer  # type: ignore[no-redef]

def apply_env_overrides_from_config(argv: list[str]) -> None:
    config_path: str | None = None
    passthrough = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        passthrough.append(arg)
        i += 1

    if config_path is None:
        return

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", payload)
    if not isinstance(env_payload, dict):
        raise SystemExit(f"Config env payload must be a JSON object: {config_path}")
    for key, value in env_payload.items():
        if value is None:
            os.environ.pop(str(key), None)
        else:
            os.environ[str(key)] = str(value)
    sys.argv[:] = passthrough

apply_env_overrides_from_config(sys.argv)

COMPUTE_DTYPE = mx.bfloat16


@dataclass(frozen=True)
class HardmaxEvalAblationSpec:
    name: str = "baseline"
    disable_controller: bool = False
    disable_residual_write: bool = False
    disable_residual_budget: bool = False
    disable_attn_q_bias: bool = False
    disable_attn_tau: bool = False


_HARDMAX_EVAL_ABLATIONS: dict[str, HardmaxEvalAblationSpec] = {
    "baseline": HardmaxEvalAblationSpec(name="baseline"),
    "none": HardmaxEvalAblationSpec(name="baseline"),
    "off": HardmaxEvalAblationSpec(name="baseline"),
    "zero_hardmax": HardmaxEvalAblationSpec(
        name="zero_hardmax",
        disable_controller=True,
        disable_residual_write=True,
        disable_residual_budget=True,
        disable_attn_q_bias=True,
        disable_attn_tau=True,
    ),
    "zero_residual": HardmaxEvalAblationSpec(
        name="zero_residual",
        disable_residual_write=True,
        disable_residual_budget=True,
    ),
    "zero_q_bias": HardmaxEvalAblationSpec(
        name="zero_q_bias",
        disable_attn_q_bias=True,
    ),
    "zero_tau": HardmaxEvalAblationSpec(
        name="zero_tau",
        disable_attn_tau=True,
    ),
}


def resolve_hardmax_eval_ablation(spec: str | HardmaxEvalAblationSpec | None) -> HardmaxEvalAblationSpec:
    if isinstance(spec, HardmaxEvalAblationSpec):
        return spec
    normalized = str(spec or "baseline").strip().lower()
    if normalized not in _HARDMAX_EVAL_ABLATIONS:
        allowed = ", ".join(sorted(set(_HARDMAX_EVAL_ABLATIONS)))
        raise ValueError(f"Unsupported hardmax eval ablation {spec!r}; expected one of: {allowed}")
    return _HARDMAX_EVAL_ABLATIONS[normalized]

class Hyperparameters:
    data_path: str = os.path.expanduser(os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    tokenizer_path: str = os.path.expanduser(os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_seqs: int = int(os.environ.get("VAL_MAX_SEQS", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    quant_eval_every: int = int(os.environ.get("QUANT_EVAL_EVERY", 0))
    quant_eval_max_seqs: int = int(os.environ.get("QUANT_EVAL_MAX_SEQS", 0))
    tensor_activity_log_every: int = int(os.environ.get("TENSOR_ACTIVITY_LOG_EVERY", 0))
    tensor_activity_hot_threshold: float = float(os.environ.get("TENSOR_ACTIVITY_HOT_THRESHOLD", 1.0e-3))
    tensor_activity_warm_threshold: float = float(os.environ.get("TENSOR_ACTIVITY_WARM_THRESHOLD", 1.0e-4))
    tensor_activity_nonzero_threshold: float = float(os.environ.get("TENSOR_ACTIVITY_NONZERO_THRESHOLD", 1.0e-8))
    tensor_activity_topk: int = int(os.environ.get("TENSOR_ACTIVITY_TOPK", 4))
    step_audit_log_every: int = int(os.environ.get("STEP_AUDIT_LOG_EVERY", 0))
    step_audit_reset_peak: bool = bool(int(os.environ.get("STEP_AUDIT_RESET_PEAK", "1")))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    replay_queue_path: str = os.path.expanduser(os.environ.get("REPLAY_QUEUE_PATH", ""))
    replay_mix_fraction: float = float(os.environ.get("REPLAY_MIX_FRACTION", "0.0"))
    replay_refresh_every_steps: int = int(os.environ.get("REPLAY_REFRESH_EVERY_STEPS", "50"))
    replay_max_cached_examples: int = int(os.environ.get("REPLAY_MAX_CACHED_EXAMPLES", "2048"))
    replay_emit_every: int = int(os.environ.get("REPLAY_EMIT_EVERY", "0"))
    replay_emit_topk: int = int(os.environ.get("REPLAY_EMIT_TOPK", "2"))
    replay_emit_min_seq_loss: float = float(os.environ.get("REPLAY_EMIT_MIN_SEQ_LOSS", "0.0"))
    student_snapshot_dir: str = os.path.expanduser(os.environ.get("STUDENT_SNAPSHOT_DIR", ""))
    student_snapshot_every: int = int(os.environ.get("STUDENT_SNAPSHOT_EVERY", "0"))
    student_snapshot_keep_last: int = int(os.environ.get("STUDENT_SNAPSHOT_KEEP_LAST", "2"))
    student_snapshot_use_ema: bool = bool(int(os.environ.get("STUDENT_SNAPSHOT_USE_EMA", "1")))
    student_heartbeat_every: int = int(os.environ.get("STUDENT_HEARTBEAT_EVERY", "10"))
    external_controller_enabled: bool = bool(int(os.environ.get("EXTERNAL_CONTROLLER_ENABLED", "0")))
    external_controller_refresh_every: int = int(os.environ.get("EXTERNAL_CONTROLLER_REFRESH_EVERY", "10"))
    eval_seq_len: int = int(os.environ.get("EVAL_SEQ_LEN", "0"))
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", "0"))
    eval_batch_seqs: int = int(os.environ.get("EVAL_BATCH_SEQS", "0"))
    curriculum_enabled: bool = bool(int(os.environ.get("CURRICULUM_ENABLED", "0")))
    curriculum_features_path: str = os.path.expanduser(os.environ.get("CURRICULUM_FEATURES_PATH", ""))
    curriculum_phase_plan_path: str = os.path.expanduser(os.environ.get("CURRICULUM_PHASE_PLAN_PATH", ""))
    curriculum_min_compressibility: float = float(os.environ.get("CURRICULUM_MIN_COMPRESSIBILITY", "-1.0"))
    curriculum_apply_logic_phase_gating: bool = bool(int(os.environ.get("CURRICULUM_APPLY_LOGIC_PHASE_GATING", "1")))
    curriculum_apply_qat_phase_gating: bool = bool(int(os.environ.get("CURRICULUM_APPLY_QAT_PHASE_GATING", "1")))
    curriculum_apply_ema_phase_gating: bool = bool(int(os.environ.get("CURRICULUM_APPLY_EMA_PHASE_GATING", "1")))
    curriculum_apply_focal: bool = bool(int(os.environ.get("CURRICULUM_APPLY_FOCAL", "1")))
    curriculum_focal_gamma: float = float(os.environ.get("CURRICULUM_FOCAL_GAMMA", 2.0))
    curriculum_focal_max_multiplier: float = float(os.environ.get("CURRICULUM_FOCAL_MAX_MULTIPLIER", 4.0))
    curriculum_token_category_weighting: bool = bool(int(os.environ.get("CURRICULUM_TOKEN_CATEGORY_WEIGHTING", "0")))
    curriculum_url_token_weight: float = float(os.environ.get("CURRICULUM_URL_TOKEN_WEIGHT", 0.2))
    curriculum_identifier_token_weight: float = float(os.environ.get("CURRICULUM_IDENTIFIER_TOKEN_WEIGHT", 0.4))
    curriculum_repeat_token_weight: float = float(os.environ.get("CURRICULUM_REPEAT_TOKEN_WEIGHT", 2.0))
    curriculum_context_delta_weighting: bool = bool(int(os.environ.get("CURRICULUM_CONTEXT_DELTA_WEIGHTING", "0")))
    curriculum_context_delta_short_len: int = int(os.environ.get("CURRICULUM_CONTEXT_DELTA_SHORT_LEN", 128))
    curriculum_context_delta_max_multiplier: float = float(
        os.environ.get("CURRICULUM_CONTEXT_DELTA_MAX_MULTIPLIER", 4.0)
    )
    curriculum_context_delta_topk_fraction: float = float(
        os.environ.get("CURRICULUM_CONTEXT_DELTA_TOPK_FRACTION", 0.0)
    )
    curriculum_context_delta_power: float = float(os.environ.get("CURRICULUM_CONTEXT_DELTA_POWER", 1.0))
    curriculum_context_delta_use_abs: bool = bool(int(os.environ.get("CURRICULUM_CONTEXT_DELTA_USE_ABS", "1")))
    structural_branching_enabled: bool = bool(int(os.environ.get("STRUCTURAL_BRANCHING_ENABLED", "0")))
    structural_branching_start_frac: float = float(os.environ.get("STRUCTURAL_BRANCHING_START_FRAC", 0.6))
    structural_branching_weight: float = float(os.environ.get("STRUCTURAL_BRANCHING_WEIGHT", 0.1))
    structural_branching_branch_length: int = int(os.environ.get("STRUCTURAL_BRANCHING_BRANCH_LENGTH", 6))
    structural_branching_max_branches: int = int(os.environ.get("STRUCTURAL_BRANCHING_MAX_BRANCHES", 1))
    structural_branching_min_structural_miss: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_MIN_STRUCTURAL_MISS", 0.5)
    )
    structural_branching_max_top1_gap: float = float(os.environ.get("STRUCTURAL_BRANCHING_MAX_TOP1_GAP", 0.75))
    structural_branching_max_top12_cosine: float = float(os.environ.get("STRUCTURAL_BRANCHING_MAX_TOP12_COSINE", 1.0))
    structural_branching_min_branch_score: float = float(os.environ.get("STRUCTURAL_BRANCHING_MIN_BRANCH_SCORE", 0.0))
    structural_branching_min_top1_prob: float = float(os.environ.get("STRUCTURAL_BRANCHING_MIN_TOP1_PROB", 0.0))
    structural_branching_min_position_gap: int = int(os.environ.get("STRUCTURAL_BRANCHING_MIN_POSITION_GAP", 8))
    structural_branching_margin: float = float(os.environ.get("STRUCTURAL_BRANCHING_MARGIN", 0.1))
    structural_branching_state_divergence_weight: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_STATE_DIVERGENCE_WEIGHT", 0.0)
    )
    structural_branching_state_target_max_cosine: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_STATE_TARGET_MAX_COSINE", 0.25)
    )
    structural_branching_adaptive_depth_enabled: bool = bool(
        int(os.environ.get("STRUCTURAL_BRANCHING_ADAPTIVE_DEPTH_ENABLED", "1"))
    )
    structural_branching_adaptive_min_depth: int = int(os.environ.get("STRUCTURAL_BRANCHING_ADAPTIVE_MIN_DEPTH", 2))
    structural_branching_adaptive_plateau_tol: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_ADAPTIVE_PLATEAU_TOL", 0.02)
    )
    structural_branching_adaptive_converged_divergence: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_ADAPTIVE_CONVERGED_DIVERGENCE", 0.05)
    )
    structural_branching_dynamic_budget: bool = bool(int(os.environ.get("STRUCTURAL_BRANCHING_DYNAMIC_BUDGET", "0")))
    structural_branching_operator_density_high: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_OPERATOR_DENSITY_HIGH", 0.02)
    )
    structural_branching_operator_density_low: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_OPERATOR_DENSITY_LOW", 0.005)
    )
    structural_branching_high_human_compressibility: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_HIGH_HUMAN_COMPRESSIBILITY", 0.50)
    )
    structural_branching_low_human_compressibility: float = float(
        os.environ.get("STRUCTURAL_BRANCHING_LOW_HUMAN_COMPRESSIBILITY", 0.33)
    )
    early_exit_layer_index: int = int(os.environ.get("EARLY_EXIT_LAYER_INDEX", "-1"))
    early_exit_horizons: str = os.environ.get("EARLY_EXIT_HORIZONS", "1,2,3")
    early_exit_aux_weight: float = float(os.environ.get("EARLY_EXIT_AUX_WEIGHT", "0.0"))
    early_exit_head_init_std: float = float(os.environ.get("EARLY_EXIT_HEAD_INIT_STD", "0.005"))
    early_exit_cascaded_enabled: bool = bool(int(os.environ.get("EARLY_EXIT_CASCADED_ENABLED", "0")))
    early_exit_condition_init_std: float = float(os.environ.get("EARLY_EXIT_CONDITION_INIT_STD", "0.001"))
    early_exit_branch_draft_enabled: bool = bool(int(os.environ.get("EARLY_EXIT_BRANCH_DRAFT_ENABLED", "0")))
    early_exit_branch_conf_threshold: float = float(os.environ.get("EARLY_EXIT_BRANCH_CONF_THRESHOLD", "0.70"))
    early_exit_branch_max_draft_tokens: int = int(os.environ.get("EARLY_EXIT_BRANCH_MAX_DRAFT_TOKENS", "1"))
    early_exit_dynamic_budget: bool = bool(int(os.environ.get("EARLY_EXIT_DYNAMIC_BUDGET", "0")))
    early_exit_budget_min_scale: float = float(os.environ.get("EARLY_EXIT_BUDGET_MIN_SCALE", "0.50"))
    early_exit_budget_max_scale: float = float(os.environ.get("EARLY_EXIT_BUDGET_MAX_SCALE", "1.50"))
    early_exit_operator_density_high: float = float(os.environ.get("EARLY_EXIT_OPERATOR_DENSITY_HIGH", "0.02"))
    early_exit_operator_density_low: float = float(os.environ.get("EARLY_EXIT_OPERATOR_DENSITY_LOW", "0.005"))
    early_exit_high_human_compressibility: float = float(
        os.environ.get("EARLY_EXIT_HIGH_HUMAN_COMPRESSIBILITY", "0.50")
    )
    early_exit_low_human_compressibility: float = float(
        os.environ.get("EARLY_EXIT_LOW_HUMAN_COMPRESSIBILITY", "0.33")
    )
    prosody_type_embeddings_enabled: bool = bool(int(os.environ.get("PROSODY_TYPE_EMBEDDINGS_ENABLED", "0")))
    prosody_type_embedding_init_std: float = float(os.environ.get("PROSODY_TYPE_EMBEDDING_INIT_STD", "0.002"))
    prosody_extended_feature_set_enabled: bool = bool(int(os.environ.get("PROSODY_EXTENDED_FEATURE_SET_ENABLED", "0")))
    prosody_feature_embeddings_enabled: bool = bool(int(os.environ.get("PROSODY_FEATURE_EMBEDDINGS_ENABLED", "0")))
    prosody_feature_embedding_init_std: float = float(os.environ.get("PROSODY_FEATURE_EMBEDDING_INIT_STD", "0.002"))
    prosody_state_adapter_enabled: bool = bool(int(os.environ.get("PROSODY_STATE_ADAPTER_ENABLED", "0")))
    prosody_state_dim: int = int(os.environ.get("PROSODY_STATE_DIM", "64"))
    prosody_state_init_std: float = float(os.environ.get("PROSODY_STATE_INIT_STD", "0.005"))
    prosody_state_scale: float = float(os.environ.get("PROSODY_STATE_SCALE", "0.50"))
    prosody_state_reset_prior_weight: float = float(os.environ.get("PROSODY_STATE_RESET_PRIOR_WEIGHT", "1.00"))
    prosody_state_hierarchical_enabled: bool = bool(int(os.environ.get("PROSODY_STATE_HIERARCHICAL_ENABLED", "0")))
    prosody_state_slow_reset_scale: float = float(os.environ.get("PROSODY_STATE_SLOW_RESET_SCALE", "0.35"))
    wallclock_final_reserve_seconds: float = float(os.environ.get("WALLCLOCK_FINAL_RESERVE_SECONDS", "0.0"))
    prosody_aux_layer_index: int = int(os.environ.get("PROSODY_AUX_LAYER_INDEX", "-1"))
    prosody_aux_weight: float = float(os.environ.get("PROSODY_AUX_WEIGHT", "0.0"))
    prosody_aux_head_init_std: float = float(os.environ.get("PROSODY_AUX_HEAD_INIT_STD", "0.005"))
    prosody_aux_token_class_weight: float = float(os.environ.get("PROSODY_AUX_TOKEN_CLASS_WEIGHT", "1.0"))
    prosody_aux_boundary_weight: float = float(os.environ.get("PROSODY_AUX_BOUNDARY_WEIGHT", "1.0"))
    prosody_aux_quote_weight: float = float(os.environ.get("PROSODY_AUX_QUOTE_WEIGHT", "0.25"))
    prosody_aux_punctuation_weight: float = float(os.environ.get("PROSODY_AUX_PUNCTUATION_WEIGHT", "0.5"))
    sequential_min_compressibility: float = float(os.environ.get("SEQUENTIAL_MIN_COMPRESSIBILITY", "-1.0"))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmdown_fraction: float = float(os.environ.get("WARMDOWN_FRACTION", 0.18))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    num_layer_templates: int = int(os.environ.get("NUM_LAYER_TEMPLATES", os.environ.get("NUM_LAYERS", 9)))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))
    mlp_leaky_slope: float = float(os.environ.get("MLP_LEAKY_SLOPE", 0.5))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "0")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    num_registers: int = int(os.environ.get("NUM_REGISTERS", 0))
    register_layout: str = os.environ.get("REGISTER_LAYOUT", "prefix")
    register_stride: int = int(os.environ.get("REGISTER_STRIDE", 256))
    register_mask_mode: str = os.environ.get("REGISTER_MASK_MODE", "bidirectional")
    logic_dim: int = int(os.environ.get("LOGIC_DIM", 0))
    logic_layer_index: int = int(os.environ.get("LOGIC_LAYER_INDEX", -1))
    logic_route_to_next_token: bool = bool(int(os.environ.get("LOGIC_ROUTE_TO_NEXT_TOKEN", "1")))
    logic_operator_mode: str = os.environ.get("LOGIC_OPERATOR_MODE", "all")
    polarity_detector_enabled: bool = bool(int(os.environ.get("POLARITY_DETECTOR_ENABLED", "0")))
    polarity_detector_layer_index: int = int(os.environ.get("POLARITY_DETECTOR_LAYER_INDEX", -1))
    polarity_detector_hidden_dim: int = int(os.environ.get("POLARITY_DETECTOR_HIDDEN_DIM", 0))
    polarity_seed_blend: float = float(os.environ.get("POLARITY_SEED_BLEND", 1.0))
    polarity_seed_weight: float = float(os.environ.get("POLARITY_SEED_WEIGHT", 0.0))
    polarity_sparse_weight: float = float(os.environ.get("POLARITY_SPARSE_WEIGHT", 0.0))
    polarity_smooth_weight: float = float(os.environ.get("POLARITY_SMOOTH_WEIGHT", 0.0))
    hardmax_struct_num_states: int = int(os.environ.get("HARDMAX_STRUCT_NUM_STATES", 0))
    hardmax_struct_dim: int = int(os.environ.get("HARDMAX_STRUCT_DIM", 32))
    hardmax_struct_static_adapter: bool = bool(int(os.environ.get("HARDMAX_STRUCT_STATIC_ADAPTER", "0")))
    hardmax_struct_layer_index: int = int(os.environ.get("HARDMAX_STRUCT_LAYER_INDEX", -1))
    hardmax_struct_router_start_layer: int = int(os.environ.get("HARDMAX_STRUCT_ROUTER_START_LAYER", -1))
    hardmax_struct_temperature: float = float(os.environ.get("HARDMAX_STRUCT_TEMPERATURE", 1.0))
    hardmax_struct_temperature_start: float = float(
        os.environ.get("HARDMAX_STRUCT_TEMPERATURE_START", os.environ.get("HARDMAX_STRUCT_TEMPERATURE", "1.0"))
    )
    hardmax_struct_temperature_end: float = float(
        os.environ.get("HARDMAX_STRUCT_TEMPERATURE_END", os.environ.get("HARDMAX_STRUCT_TEMPERATURE", "1.0"))
    )
    hardmax_struct_temperature_anneal_frac: float = float(
        os.environ.get("HARDMAX_STRUCT_TEMPERATURE_ANNEAL_FRAC", "0.0")
    )
    hardmax_struct_fast_refinement_steps: int = int(os.environ.get("HARDMAX_STRUCT_FAST_REFINEMENT_STEPS", "1"))
    hardmax_struct_compute_min_scale: float = float(os.environ.get("HARDMAX_STRUCT_COMPUTE_MIN_SCALE", 0.35))
    hardmax_struct_compute_power: float = float(os.environ.get("HARDMAX_STRUCT_COMPUTE_POWER", 1.0))
    hardmax_struct_route_residual_budget: bool = bool(int(os.environ.get("HARDMAX_STRUCT_ROUTE_RESIDUAL_BUDGET", "1")))
    hardmax_struct_operator_prior_scale: float = float(os.environ.get("HARDMAX_STRUCT_OPERATOR_PRIOR_SCALE", 1.0))
    hardmax_struct_reset_prior_scale: float = float(os.environ.get("HARDMAX_STRUCT_RESET_PRIOR_SCALE", 1.0))
    hardmax_struct_usage_balance_weight: float = float(os.environ.get("HARDMAX_STRUCT_USAGE_BALANCE_WEIGHT", 0.0))
    hardmax_struct_diversity_weight: float = float(os.environ.get("HARDMAX_STRUCT_DIVERSITY_WEIGHT", 0.0))
    hardmax_struct_predict_weight: float = float(os.environ.get("HARDMAX_STRUCT_PREDICT_WEIGHT", 0.0))
    hardmax_struct_confidence_weight: float = float(os.environ.get("HARDMAX_STRUCT_CONFIDENCE_WEIGHT", 0.0))
    hardmax_struct_operator_weight: float = float(os.environ.get("HARDMAX_STRUCT_OPERATOR_WEIGHT", 1.0))
    hardmax_struct_token_class_weight: float = float(os.environ.get("HARDMAX_STRUCT_TOKEN_CLASS_WEIGHT", 0.5))
    hardmax_struct_boundary_weight: float = float(os.environ.get("HARDMAX_STRUCT_BOUNDARY_WEIGHT", 1.0))
    hardmax_struct_quote_weight: float = float(os.environ.get("HARDMAX_STRUCT_QUOTE_WEIGHT", 0.25))
    hardmax_struct_condition_mode: str = os.environ.get("HARDMAX_STRUCT_CONDITION_MODE", "residual").strip().lower()
    hardmax_struct_attn_q_scale: float = float(os.environ.get("HARDMAX_STRUCT_ATTN_Q_SCALE", 1.0))
    hardmax_struct_attn_tau_min: float = float(os.environ.get("HARDMAX_STRUCT_ATTN_TAU_MIN", 0.75))
    hardmax_struct_init_path: str = os.environ.get("HARDMAX_STRUCT_INIT_PATH", "").strip()
    hardmax_struct_init_mode: str = os.environ.get("HARDMAX_STRUCT_INIT_MODE", "").strip().lower()
    hardmax_struct_statebook_freeze_steps: int = int(
        os.environ.get("HARDMAX_STRUCT_STATEBOOK_FREEZE_STEPS", "0")
    )
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    matrix_optimizer: str = os.environ.get("MATRIX_OPTIMIZER", "muon")
    turbo_matrix_optimizer: str = os.environ.get("TURBO_MATRIX_OPTIMIZER", os.environ.get("MATRIX_OPTIMIZER", "muon"))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    apollo_rank: int = int(os.environ.get("APOLLO_RANK", 1))
    apollo_scale: float = float(os.environ.get("APOLLO_SCALE", 128.0))
    apollo_scale_type: str = os.environ.get("APOLLO_SCALE_TYPE", "tensor")
    apollo_proj_gap: int = int(os.environ.get("APOLLO_PROJ_GAP", 200))
    apollo_proj_type: str = os.environ.get("APOLLO_PROJ_TYPE", "std")
    apollo_scale_front: bool = bool(int(os.environ.get("APOLLO_SCALE_FRONT", "0")))
    apollo_disable_nl: bool = bool(int(os.environ.get("APOLLO_DISABLE_NL", "0")))
    apollo_beta1: float = float(os.environ.get("APOLLO_BETA1", os.environ.get("BETA1", 0.9)))
    apollo_beta2: float = float(os.environ.get("APOLLO_BETA2", 0.999))
    apollo_eps: float = float(os.environ.get("APOLLO_EPS", 1e-6))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    sanitize_nonfinite_grads: bool = bool(int(os.environ.get("SANITIZE_NONFINITE_GRADS", "1")))
    sanitize_nonfinite_grads_every: int = int(os.environ.get("SANITIZE_NONFINITE_GRADS_EVERY", 1))
    sanitize_nonfinite_grads_always_first_steps: int = int(os.environ.get("SANITIZE_NONFINITE_GRADS_ALWAYS_FIRST_STEPS", 10))
    nonfinite_grad_topk: int = int(os.environ.get("NONFINITE_GRAD_TOPK", 6))

    quant_format: str = os.environ.get("QUANT_FORMAT", "int8_clean_per_row_v1")
    turbo_block_size: int = int(os.environ.get("TURBO_BLOCK_SIZE", 64))
    turbo_mse_bits: int = int(os.environ.get("TURBO_MSE_BITS", 5))
    turbo_prod_bits: int = int(os.environ.get("TURBO_PROD_BITS", 6))
    turbo_rot_seed: int = int(os.environ.get("TURBO_ROT_SEED", 17))
    turbo_qjl_seed: int = int(os.environ.get("TURBO_QJL_SEED", 29))
    turbo_qat: bool = bool(int(os.environ.get("TURBO_QAT", "0")))
    turbo_qat_start_frac: float = float(os.environ.get("TURBO_QAT_START_FRAC", 0.5))
    turbo_qat_ramp_frac: float = float(os.environ.get("TURBO_QAT_RAMP_FRAC", 0.25))
    turbo_qat_lambda: float = float(os.environ.get("TURBO_QAT_LAMBDA", 0.0))
    turbo_qat_muon_momentum: float = float(os.environ.get("TURBO_QAT_MUON_MOMENTUM", os.environ.get("MUON_MOMENTUM", 0.95)))
    turbo_qat_muon_momentum_warmup_start: float = float(
        os.environ.get("TURBO_QAT_MUON_MOMENTUM_WARMUP_START", os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    )
    ema_enabled: bool = bool(int(os.environ.get("EMA_ENABLED", "0")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))
    ema_start_frac: float = float(os.environ.get("EMA_START_FRAC", 0.0))
    phase_transition_log: bool = bool(int(os.environ.get("PHASE_TRANSITION_LOG", "1")))
    ema_reset_on_phase_transition: bool = bool(int(os.environ.get("EMA_RESET_ON_PHASE_TRANSITION", "0")))
    ema_phase_reset_min_progress: float = float(os.environ.get("EMA_PHASE_RESET_MIN_PROGRESS", 0.5))
    ema_phase_reset_cooldown_steps: int = int(os.environ.get("EMA_PHASE_RESET_COOLDOWN_STEPS", 200))
    ema_phase_reset_requires_qat_full: bool = bool(int(os.environ.get("EMA_PHASE_RESET_REQUIRES_QAT_FULL", "1")))
    ema_adaptive_grad_ac: bool = bool(int(os.environ.get("EMA_ADAPTIVE_GRAD_AC", "0")))
    ema_grad_ac_beta: float = float(os.environ.get("EMA_GRAD_AC_BETA", 0.9))
    ema_grad_ac_start_threshold: float = float(os.environ.get("EMA_GRAD_AC_START_THRESHOLD", 0.3))
    ema_adaptive_min_decay: float = float(os.environ.get("EMA_ADAPTIVE_MIN_DECAY", 0.0))
    ema_adaptive_max_decay: float = float(os.environ.get("EMA_ADAPTIVE_MAX_DECAY", 0.99995))
    ema_aggressive_warmdown: bool = bool(int(os.environ.get("EMA_AGGRESSIVE_WARMDOWN", "1")))
    ema_reset_on_qat_full: bool = bool(int(os.environ.get("EMA_RESET_ON_QAT_FULL", "1")))
    ema_teacher_distill_enabled: bool = bool(int(os.environ.get("EMA_TEACHER_DISTILL_ENABLED", "0")))
    ema_teacher_decay: float = float(os.environ.get("EMA_TEACHER_DECAY", 0.999))
    ema_teacher_start_frac: float = float(os.environ.get("EMA_TEACHER_START_FRAC", 0.0))
    ema_teacher_distill_weight: float = float(os.environ.get("EMA_TEACHER_DISTILL_WEIGHT", 0.1))
    ema_teacher_temperature: float = float(os.environ.get("EMA_TEACHER_TEMPERATURE", 1.0))
    external_teacher_config_paths: str = os.path.expanduser(os.environ.get("EXTERNAL_TEACHER_CONFIG_PATHS", ""))
    external_teacher_checkpoint_paths: str = os.path.expanduser(os.environ.get("EXTERNAL_TEACHER_CHECKPOINT_PATHS", ""))
    external_teacher_start_frac: float = float(os.environ.get("EXTERNAL_TEACHER_START_FRAC", 0.0))
    external_teacher_distill_weight: float = float(os.environ.get("EXTERNAL_TEACHER_DISTILL_WEIGHT", 0.0))
    external_teacher_temperature: float = float(os.environ.get("EXTERNAL_TEACHER_TEMPERATURE", 1.0))
    external_teacher_hidden_distill_weight: float = float(os.environ.get("EXTERNAL_TEACHER_HIDDEN_DISTILL_WEIGHT", 0.0))
    external_teacher_hidden_layer: int = int(os.environ.get("EXTERNAL_TEACHER_HIDDEN_LAYER", -1))
    external_teacher_hidden_cache_dir: str = os.path.expanduser(os.environ.get("EXTERNAL_TEACHER_HIDDEN_CACHE_DIR", ""))
    external_teacher_hidden_cache_max_entries: int = int(os.environ.get("EXTERNAL_TEACHER_HIDDEN_CACHE_MAX_ENTRIES", 1024))
    external_teacher_hidden_cache_write: bool = bool(int(os.environ.get("EXTERNAL_TEACHER_HIDDEN_CACHE_WRITE", "1")))
    external_teacher_allow_partial_load: bool = bool(int(os.environ.get("EXTERNAL_TEACHER_ALLOW_PARTIAL_LOAD", "0")))
    external_teacher_min_param_fraction: float = float(os.environ.get("EXTERNAL_TEACHER_MIN_PARAM_FRACTION", 0.95))
    external_teacher_load_log_topk: int = int(os.environ.get("EXTERNAL_TEACHER_LOAD_LOG_TOPK", 8))
    adaptive_train_controller: bool = bool(int(os.environ.get("ADAPTIVE_TRAIN_CONTROLLER", "0")))
    adaptive_ctrl_log_every: int = int(os.environ.get("ADAPTIVE_CTRL_LOG_EVERY", 100))
    adaptive_ctrl_sanitize_min_every: int = int(os.environ.get("ADAPTIVE_CTRL_SANITIZE_MIN_EVERY", 1))
    adaptive_ctrl_sanitize_max_every: int = int(os.environ.get("ADAPTIVE_CTRL_SANITIZE_MAX_EVERY", 64))
    adaptive_ctrl_sanitize_stable_steps: int = int(os.environ.get("ADAPTIVE_CTRL_SANITIZE_STABLE_STEPS", 50))
    adaptive_ctrl_sanitize_recovery_steps: int = int(os.environ.get("ADAPTIVE_CTRL_SANITIZE_RECOVERY_STEPS", 25))
    adaptive_ctrl_distill_disagree_low: float = float(os.environ.get("ADAPTIVE_CTRL_DISTILL_DISAGREE_LOW", 0.02))
    adaptive_ctrl_distill_disagree_high: float = float(os.environ.get("ADAPTIVE_CTRL_DISTILL_DISAGREE_HIGH", 0.15))
    adaptive_ctrl_distill_min_mult: float = float(os.environ.get("ADAPTIVE_CTRL_DISTILL_MIN_MULT", 0.75))
    adaptive_ctrl_distill_max_mult: float = float(os.environ.get("ADAPTIVE_CTRL_DISTILL_MAX_MULT", 1.25))
    adaptive_ctrl_branch_disagree_low: float = float(os.environ.get("ADAPTIVE_CTRL_BRANCH_DISAGREE_LOW", 0.02))
    adaptive_ctrl_branch_disagree_high: float = float(os.environ.get("ADAPTIVE_CTRL_BRANCH_DISAGREE_HIGH", 0.15))
    adaptive_ctrl_branch_min_mult: float = float(os.environ.get("ADAPTIVE_CTRL_BRANCH_MIN_MULT", 0.50))
    adaptive_ctrl_branch_max_mult: float = float(os.environ.get("ADAPTIVE_CTRL_BRANCH_MAX_MULT", 1.50))
    adaptive_ctrl_branch_max_extra: int = int(os.environ.get("ADAPTIVE_CTRL_BRANCH_MAX_EXTRA", 1))
    ortho_probe_mode: str = os.environ.get("ORTHO_PROBE_MODE", "none")
    ortho_probe_targets: str = os.environ.get("ORTHO_PROBE_TARGETS", "")
    ortho_probe_eps: float = float(os.environ.get("ORTHO_PROBE_EPS", 1e-4))
    skip_final_export: bool = bool(int(os.environ.get("SKIP_FINAL_EXPORT", "0")))
    mlx_compile: str = os.environ.get("MLX_COMPILE", "auto")

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    @property
    def effective_eval_seq_len(self) -> int:
        return self.eval_seq_len if self.eval_seq_len > 0 else self.train_seq_len

    @property
    def effective_eval_batch_seqs(self) -> int:
        if self.eval_batch_seqs > 0:
            return self.eval_batch_seqs
        val_batch_tokens = max(self.val_batch_size // max(self.grad_accum_steps, 1), self.effective_eval_seq_len)
        return max(val_batch_tokens // self.effective_eval_seq_len, 1)

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            if self.warmdown_fraction <= 0.0 or self.max_wallclock_seconds <= 0:
                return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        if self.warmdown_fraction > 0.0:
            warmdown_ms = 1000.0 * self.max_wallclock_seconds * self.warmdown_fraction
            remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
            return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
TURBO_QAT_EXCLUDE_PATTERNS = tuple(
    pattern.strip()
    for pattern in os.environ.get("TURBO_QAT_EXCLUDE_PATTERNS", "").split(",")
    if pattern.strip()
)

def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks

def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def binary_cross_entropy_with_logits(logits: mx.array, targets: mx.array) -> mx.array:
    logits32 = logits.astype(mx.float32)
    targets32 = targets.astype(mx.float32)
    return mx.mean(mx.maximum(logits32, 0.0) - logits32 * targets32 + mx.log1p(mx.exp(-mx.abs(logits32))))


def token_cross_entropy_with_focal(
    logits: mx.array,
    targets: mx.array,
    *,
    focal_loss_weight: float = 0.0,
    focal_gamma: float = 2.0,
    focal_max_multiplier: float = 4.0,
    token_weights: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    nll = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none").astype(mx.float32)
    if focal_loss_weight > 0.0:
        pt = mx.exp(-nll)
        hard_weight = mx.power(mx.maximum(1.0 - pt, mx.array(1e-6, dtype=mx.float32)), focal_gamma).astype(mx.float32)
        hard_weight = hard_weight / mx.maximum(mx.mean(hard_weight), mx.array(1e-6, dtype=mx.float32))
        if focal_max_multiplier > 0.0:
            hard_weight = mx.minimum(hard_weight, mx.array(focal_max_multiplier, dtype=mx.float32))
        blend = (1.0 - focal_loss_weight) + focal_loss_weight * hard_weight
        nll = nll * blend.astype(mx.float32)
    if token_weights is not None:
        weights = token_weights.astype(mx.float32)
        weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-6, dtype=mx.float32))
        nll = nll * weights
    if reduction == "none":
        return nll
    if reduction == "sum":
        return mx.sum(nll)
    if reduction == "mean":
        return mx.mean(nll)
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def merge_token_weights(token_weights: mx.array | None, extra_token_weights: mx.array | None) -> mx.array | None:
    if token_weights is None:
        return extra_token_weights
    if extra_token_weights is None:
        return token_weights
    return token_weights.astype(mx.float32) * extra_token_weights.astype(mx.float32)


def build_local_causal_attention_mask(
    total_len: int,
    context_window: int,
    *,
    prefix_tokens: int = 0,
) -> mx.array:
    if context_window <= 0:
        raise ValueError(f"context_window must be positive, got {context_window}")
    positions = np.arange(total_len, dtype=np.int32)
    query = positions[:, None]
    key = positions[None, :]
    allowed = key <= query
    if prefix_tokens > 0:
        non_prefix = key >= prefix_tokens
        local_allowed = (query - key) < context_window
        allowed = np.where(non_prefix, allowed & local_allowed, allowed)
        allowed[:prefix_tokens, :] = True
    else:
        allowed &= (query - key) < context_window
    additive = np.where(allowed, 0.0, -1e9).astype(np.float32, copy=False)
    return mx.array(additive[None, None, :, :], dtype=mx.bfloat16)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


def token_category_weighting_config(args: Hyperparameters) -> TokenCategoryWeightingConfig:
    return TokenCategoryWeightingConfig(
        enabled=args.curriculum_token_category_weighting,
        url_like_weight=args.curriculum_url_token_weight,
        identifier_like_weight=args.curriculum_identifier_token_weight,
        repeat_content_weight=args.curriculum_repeat_token_weight,
    )


def context_delta_weighting_config(args: Hyperparameters) -> ContextDeltaWeightingConfig:
    return ContextDeltaWeightingConfig(
        enabled=args.curriculum_context_delta_weighting,
        short_context_len=args.curriculum_context_delta_short_len,
        max_multiplier=args.curriculum_context_delta_max_multiplier,
        topk_fraction=args.curriculum_context_delta_topk_fraction,
        score_power=args.curriculum_context_delta_power,
        use_absolute_delta=args.curriculum_context_delta_use_abs,
    )


def structural_branching_config(args: Hyperparameters) -> StructuralBranchingConfig:
    return StructuralBranchingConfig(
        enabled=args.structural_branching_enabled,
        start_frac=args.structural_branching_start_frac,
        weight=args.structural_branching_weight,
        branch_length=args.structural_branching_branch_length,
        max_branches=args.structural_branching_max_branches,
        min_structural_miss=args.structural_branching_min_structural_miss,
        max_top1_gap=args.structural_branching_max_top1_gap,
        max_top12_cosine=args.structural_branching_max_top12_cosine,
        min_branch_score=args.structural_branching_min_branch_score,
        min_top1_prob=args.structural_branching_min_top1_prob,
        min_position_gap=args.structural_branching_min_position_gap,
        margin=args.structural_branching_margin,
        state_divergence_weight=args.structural_branching_state_divergence_weight,
        state_target_max_cosine=args.structural_branching_state_target_max_cosine,
        adaptive_depth_enabled=args.structural_branching_adaptive_depth_enabled,
        adaptive_min_depth=args.structural_branching_adaptive_min_depth,
        adaptive_plateau_tol=args.structural_branching_adaptive_plateau_tol,
        adaptive_converged_divergence=args.structural_branching_adaptive_converged_divergence,
    )


def structural_branch_budget_controller(args: Hyperparameters) -> StructuralBranchBudgetController:
    return StructuralBranchBudgetController(
        enabled=args.structural_branching_dynamic_budget,
        operator_density_high=args.structural_branching_operator_density_high,
        operator_density_low=args.structural_branching_operator_density_low,
        high_human_compressibility=args.structural_branching_high_human_compressibility,
        low_human_compressibility=args.structural_branching_low_human_compressibility,
    )


def early_exit_budget_controller(args: Hyperparameters) -> EarlyExitBudgetController:
    return EarlyExitBudgetController(
        enabled=args.early_exit_dynamic_budget,
        min_scale=args.early_exit_budget_min_scale,
        max_scale=args.early_exit_budget_max_scale,
        operator_density_high=args.early_exit_operator_density_high,
        operator_density_low=args.early_exit_operator_density_low,
        high_human_compressibility=args.early_exit_high_human_compressibility,
        low_human_compressibility=args.early_exit_low_human_compressibility,
    )


@dataclass(frozen=True)
class AdaptiveTrainControllerConfig:
    enabled: bool = False
    log_every: int = 100
    sanitize_min_every: int = 1
    sanitize_max_every: int = 64
    sanitize_stable_steps: int = 50
    sanitize_recovery_steps: int = 25
    distill_disagree_low: float = 0.02
    distill_disagree_high: float = 0.15
    distill_min_mult: float = 0.75
    distill_max_mult: float = 1.25
    branch_disagree_low: float = 0.02
    branch_disagree_high: float = 0.15
    branch_min_mult: float = 0.50
    branch_max_mult: float = 1.50
    branch_max_extra: int = 1


@dataclass
class AdaptiveTrainControllerState:
    sanitize_every: int = 1
    sanitize_stable_steps: int = 0
    sanitize_recovery_steps_left: int = 0
    distill_weight_mult: float = 1.0
    branch_weight_mult: float = 1.0
    branch_extra_max_branches: int = 0
    last_teacher_disagree: float = 0.0
    last_action: str = "init"


def adaptive_train_controller_config(args: Hyperparameters) -> AdaptiveTrainControllerConfig:
    return AdaptiveTrainControllerConfig(
        enabled=args.adaptive_train_controller,
        log_every=args.adaptive_ctrl_log_every,
        sanitize_min_every=args.adaptive_ctrl_sanitize_min_every,
        sanitize_max_every=args.adaptive_ctrl_sanitize_max_every,
        sanitize_stable_steps=args.adaptive_ctrl_sanitize_stable_steps,
        sanitize_recovery_steps=args.adaptive_ctrl_sanitize_recovery_steps,
        distill_disagree_low=args.adaptive_ctrl_distill_disagree_low,
        distill_disagree_high=args.adaptive_ctrl_distill_disagree_high,
        distill_min_mult=args.adaptive_ctrl_distill_min_mult,
        distill_max_mult=args.adaptive_ctrl_distill_max_mult,
        branch_disagree_low=args.adaptive_ctrl_branch_disagree_low,
        branch_disagree_high=args.adaptive_ctrl_branch_disagree_high,
        branch_min_mult=args.adaptive_ctrl_branch_min_mult,
        branch_max_mult=args.adaptive_ctrl_branch_max_mult,
        branch_max_extra=args.adaptive_ctrl_branch_max_extra,
    )


def init_adaptive_train_controller_state(
    args: Hyperparameters,
    config: AdaptiveTrainControllerConfig,
) -> AdaptiveTrainControllerState:
    sanitize_every = max(
        min(int(args.sanitize_nonfinite_grads_every), max(int(config.sanitize_max_every), 1)),
        max(int(config.sanitize_min_every), 1),
    )
    return AdaptiveTrainControllerState(sanitize_every=sanitize_every)


def _controller_interp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def adaptive_train_controller_branch_config(
    base: StructuralBranchingConfig | None,
    controller_state: AdaptiveTrainControllerState,
) -> StructuralBranchingConfig | None:
    if base is None:
        return None
    weight = max(float(base.weight) * float(controller_state.branch_weight_mult), 0.0)
    max_branches = max(int(base.max_branches) + int(controller_state.branch_extra_max_branches), 0)
    enabled = bool(base.enabled and weight > 0.0 and max_branches > 0)
    return replace(base, enabled=enabled, weight=weight, max_branches=max_branches)


def update_adaptive_train_controller(
    config: AdaptiveTrainControllerConfig,
    state: AdaptiveTrainControllerState,
    *,
    train_loss_value: float,
    grad_nonfinite_tensors: int,
    teacher_count: float | None,
    teacher_disagree_frac: float | None,
) -> AdaptiveTrainControllerState:
    if not config.enabled:
        return state

    next_state = AdaptiveTrainControllerState(
        sanitize_every=int(state.sanitize_every),
        sanitize_stable_steps=int(state.sanitize_stable_steps),
        sanitize_recovery_steps_left=int(state.sanitize_recovery_steps_left),
        distill_weight_mult=float(state.distill_weight_mult),
        branch_weight_mult=float(state.branch_weight_mult),
        branch_extra_max_branches=int(state.branch_extra_max_branches),
        last_teacher_disagree=float(state.last_teacher_disagree),
        last_action="hold",
    )
    action_bits: list[str] = []
    stable = math.isfinite(train_loss_value) and int(grad_nonfinite_tensors) <= 0
    min_every = max(int(config.sanitize_min_every), 1)
    max_every = max(int(config.sanitize_max_every), min_every)
    if stable:
        next_state.sanitize_stable_steps += 1
        if next_state.sanitize_recovery_steps_left > 0:
            next_state.sanitize_recovery_steps_left -= 1
        elif next_state.sanitize_stable_steps >= max(int(config.sanitize_stable_steps), 1):
            widened = min(max(int(next_state.sanitize_every), min_every) * 2, max_every)
            if widened != next_state.sanitize_every:
                next_state.sanitize_every = widened
                next_state.sanitize_stable_steps = 0
                action_bits.append(f"sanitize_every->{widened}")
    else:
        if next_state.sanitize_every != min_every or next_state.sanitize_recovery_steps_left <= 0:
            action_bits.append(f"sanitize_recover->{min_every}")
        next_state.sanitize_every = min_every
        next_state.sanitize_stable_steps = 0
        next_state.sanitize_recovery_steps_left = max(int(config.sanitize_recovery_steps), 0)

    if teacher_count is not None and teacher_count > 1.0 and teacher_disagree_frac is not None:
        disagree = float(np.clip(float(teacher_disagree_frac), 0.0, 1.0))
        next_state.last_teacher_disagree = disagree
        distill_alpha = _controller_interp(disagree, config.distill_disagree_low, config.distill_disagree_high)
        branch_alpha = _controller_interp(disagree, config.branch_disagree_low, config.branch_disagree_high)
        next_state.distill_weight_mult = float(
            config.distill_max_mult - distill_alpha * (config.distill_max_mult - config.distill_min_mult)
        )
        next_state.branch_weight_mult = float(
            config.branch_min_mult + branch_alpha * (config.branch_max_mult - config.branch_min_mult)
        )
        next_state.branch_extra_max_branches = int(round(branch_alpha * max(int(config.branch_max_extra), 0)))
        action_bits.append(
            f"teacher_disagree:{disagree:.3f} distill_mult:{next_state.distill_weight_mult:.2f} "
            f"branch_mult:{next_state.branch_weight_mult:.2f} branch_extra:{next_state.branch_extra_max_branches}"
        )
    else:
        next_state.distill_weight_mult = 1.0
        next_state.branch_weight_mult = 1.0
        next_state.branch_extra_max_branches = 0

    if action_bits:
        next_state.last_action = ";".join(action_bits)
    return next_state


def resolve_early_exit_layer_index(requested_layer: int, num_layers: int) -> int:
    if requested_layer < 0:
        return max(num_layers // 2 - 1, 0)
    if requested_layer >= num_layers:
        raise ValueError(f"EARLY_EXIT_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


def resolve_prosody_aux_layer_index(requested_layer: int, num_layers: int) -> int:
    if requested_layer < 0:
        return max(num_layers // 2 - 1, 0)
    if requested_layer >= num_layers:
        raise ValueError(f"PROSODY_AUX_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


def token_category_weights_mx(
    args: Hyperparameters,
    luts: TokenCategoryLuts | None,
    x_np: np.ndarray,
    y_np: np.ndarray,
) -> mx.array | None:
    config = token_category_weighting_config(args)
    if not config.enabled or luts is None:
        return None
    weights_np = compute_token_category_weights(x_np, y_np, luts, config)
    return mx.array(weights_np, dtype=mx.float32)

class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def take_window(self, n: int, *, advance: int | None = None) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"window size must be positive, got {n}")
        if advance is None:
            advance = n
        if advance <= 0:
            raise ValueError(f"advance must be positive, got {advance}")
        start_file_idx = self.file_idx
        start_pos = self.pos
        start_tokens = self.tokens
        window = self.take(n)
        self.file_idx = start_file_idx
        self.pos = start_pos
        self.tokens = start_tokens
        _ = self.take(advance)
        return window

class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return x, y

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        x, y = self.next_batch_np(batch_tokens, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


class SequentialFilteredTokenLoader:
    def __init__(
        self,
        pattern: str,
        *,
        filter_config: SequentialCompressibilityFilterConfig,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)
        self.filter_config = filter_config
        self.seen_chunks = 0
        self.skipped_chunks = 0
        self.kept_chunks = 0

    def summary(self) -> dict[str, object]:
        return {
            "mode": "sequential_filter",
            "min_compressibility": float(self.filter_config.min_compressibility),
            "seen_chunks": int(self.seen_chunks),
            "kept_chunks": int(self.kept_chunks),
            "skipped_low_compressibility_chunks": int(self.skipped_chunks),
        }

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        while len(x_rows) < batch_seqs:
            chunk = self.stream.take_window(seq_len + 1, advance=seq_len)
            x_row = chunk[:-1]
            self.seen_chunks += 1
            if not keep_sequential_chunk(x_row, self.filter_config):
                self.skipped_chunks += 1
                continue
            self.kept_chunks += 1
            x_rows.append(x_row)
            y_rows.append(chunk[1:])
        x = np.stack(x_rows, axis=0).astype(np.int32, copy=False)
        y = np.stack(y_rows, axis=0).astype(np.int32, copy=False)
        return x, y

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        x, y = self.next_batch_np(batch_tokens, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def build_train_loader(
    args: Hyperparameters,
    log_fn: Callable[[str], None] | None = None,
    dataset_name: str = "",
):
    if args.curriculum_enabled and not args.curriculum_features_path:
        raise ValueError("CURRICULUM_ENABLED=1 requires CURRICULUM_FEATURES_PATH")
    if not args.curriculum_enabled:
        if args.sequential_min_compressibility >= 0.0:
            base_loader = SequentialFilteredTokenLoader(
                args.train_files,
                filter_config=SequentialCompressibilityFilterConfig(
                    enabled=True,
                    min_compressibility=args.sequential_min_compressibility,
                ),
                log_fn=log_fn,
                dataset_name=dataset_name,
            )
        else:
            base_loader = TokenLoader(args.train_files, log_fn=log_fn, dataset_name=dataset_name)
    else:
        from curriculum_runtime import CurriculumRuntimeConfig, CurriculumTokenLoader

        runtime_config = CurriculumRuntimeConfig(
            features_path=args.curriculum_features_path,
            phase_plan_path=args.curriculum_phase_plan_path,
            min_compressibility=args.curriculum_min_compressibility,
        )
        base_loader = CurriculumTokenLoader(
            args.train_files,
            seq_len=args.train_seq_len,
            total_train_tokens=args.iterations * args.train_batch_tokens,
            runtime_config=runtime_config,
            log_fn=log_fn,
            dataset_name=dataset_name,
        )
    replay_enabled = bool(args.replay_queue_path) and (
        args.replay_mix_fraction > 0.0 or args.replay_emit_every > 0
    )
    if not replay_enabled:
        return base_loader
    replay_buffer = FileReplayBuffer(
        args.replay_queue_path,
        seq_len=args.train_seq_len,
        refresh_every_steps=args.replay_refresh_every_steps,
        max_cached_examples=args.replay_max_cached_examples,
        seed=args.seed,
        log_fn=log_fn,
    )
    return ReplayMixTokenLoader(
        base_loader,
        replay_buffer,
        mix_fraction=args.replay_mix_fraction,
        seed=args.seed,
    )

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        return None

    def turbo_regularizer(self) -> mx.array:
        return mx.array(0.0, dtype=mx.float32)

    def clear_turbo_cache(self) -> None:
        return None


def _random_orthogonal_matrix(dim: int) -> mx.array:
    mat = np.random.standard_normal((dim, dim)).astype(np.float32)
    q, r = np.linalg.qr(mat)
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0).astype(np.float32)
    return mx.array(q * signs[None, :], dtype=mx.float32)


class CayleyDiagLinear(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.eps = eps
        lower_i, lower_j = np.tril_indices(dim, k=-1)
        self.rot_params = mx.zeros((lower_i.size,), dtype=mx.float32)
        self.log_scales = mx.zeros((dim,), dtype=mx.float32)
        self._lower_i = mx.array(lower_i, dtype=mx.int32)
        self._lower_j = mx.array(lower_j, dtype=mx.int32)
        self._eye = mx.eye(dim, dtype=mx.float32)
        base_weight = np.asarray(nn.Linear(dim, dim, bias=False).weight.astype(mx.float32), dtype=np.float32)
        row_scales = np.sqrt(np.sum(base_weight * base_weight, axis=1, dtype=np.float64) + 1e-8, dtype=np.float64).astype(np.float32)
        init_q = _random_orthogonal_matrix(dim)
        init_q_np = np.asarray(init_q, dtype=np.float32)
        eye_np = np.eye(dim, dtype=np.float32)
        init_a = np.linalg.solve(eye_np + init_q_np + self.eps * eye_np, eye_np - init_q_np)
        init_a = (0.5 * (init_a - init_a.T)).astype(np.float32)
        init_rot_params = mx.array(init_a[lower_i, lower_j], dtype=mx.float32)
        self.rot_params = init_rot_params.astype(mx.float32)
        self.log_scales = mx.log(mx.array(row_scales, dtype=mx.float32)).astype(mx.float32)

    def dense_weight(self) -> mx.array:
        skew = mx.zeros((self.dim, self.dim), dtype=mx.float32)
        skew = skew.at[self._lower_i, self._lower_j].add(self.rot_params)
        skew = skew.at[self._lower_j, self._lower_i].add(-self.rot_params)
        q = mx.linalg.solve(self._eye + skew + self.eps * self._eye, self._eye - skew, stream=mx.cpu)
        return mx.exp(self.log_scales)[:, None] * q

    @property
    def weight(self) -> mx.array:
        return self.dense_weight()

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.dense_weight().astype(x.dtype).T

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        return None

    def turbo_regularizer(self) -> mx.array:
        return mx.array(0.0, dtype=mx.float32)

    def clear_turbo_cache(self) -> None:
        return None


ORTHO_PROBE_MODE = os.environ.get("ORTHO_PROBE_MODE", "none").strip().lower()
ORTHO_PROBE_TARGET_PATTERNS = tuple(
    pattern.strip()
    for pattern in os.environ.get("ORTHO_PROBE_TARGETS", "").split(",")
    if pattern.strip()
)
ORTHO_PROBE_EPS = float(os.environ.get("ORTHO_PROBE_EPS", 1e-4))


def infer_ortho_probe_mode(name: str, in_dim: int, out_dim: int) -> str:
    if ORTHO_PROBE_MODE == "none" or in_dim != out_dim:
        return "none"
    if not ORTHO_PROBE_TARGET_PATTERNS:
        return "none"
    return ORTHO_PROBE_MODE if any(pattern in name for pattern in ORTHO_PROBE_TARGET_PATTERNS) else "none"


def infer_turbo_qat_excluded(name: str) -> bool:
    return any(pattern in name for pattern in TURBO_QAT_EXCLUDE_PATTERNS)


def make_turbo_linear(name: str, in_dim: int, out_dim: int) -> nn.Module:
    ortho_probe_mode = infer_ortho_probe_mode(name, in_dim, out_dim)
    if ortho_probe_mode == "cayley_diag":
        raise ValueError(
            "ORTHO_PROBE_MODE=cayley_diag is currently disabled on MLX because "
            "linalg solve/inverse VJP is not implemented. Use "
            "tools/eval_mlx_orthogonal_projection.py for offline diag×orthogonal tests."
        )
    turbo_mode = infer_turbo_mode(name)
    turbo_bits = 0
    turbo_block_size = 0
    if turbo_mode == "mse":
        turbo_bits = TURBO_MSE_BITS
        turbo_block_size = TURBO_BLOCK_SIZE
    elif turbo_mode == "prod":
        turbo_bits = TURBO_PROD_BITS
        turbo_block_size = TURBO_BLOCK_SIZE
    return TurboLinear(
        in_dim,
        out_dim,
        turbo_mode=turbo_mode,
        turbo_bits=turbo_bits,
        turbo_block_size=turbo_block_size,
        qat_excluded=infer_turbo_qat_excluded(name),
    )

class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class ProsodyStateAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        state_dim: int,
        num_features: int,
        init_std: float,
        output_scale: float,
        reset_prior_weight: float,
        hierarchical_enabled: bool,
        slow_reset_scale: float,
        feature_name_to_idx: dict[str, int] | None = None,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        self.state_dim = int(state_dim)
        self.output_scale = float(output_scale)
        self.reset_prior_weight = float(reset_prior_weight)
        self.hierarchical_enabled = bool(hierarchical_enabled and state_dim >= 2)
        self.slow_reset_scale = float(slow_reset_scale)
        self.feature_name_to_idx = dict(feature_name_to_idx or {})
        self.fast_dim = int(state_dim if not self.hierarchical_enabled else max(1, state_dim // 2))
        self.slow_dim = int(0 if not self.hierarchical_enabled else max(1, state_dim - self.fast_dim))
        if not self.hierarchical_enabled:
            self.feature_emb = (
                mx.random.normal((num_features, self.state_dim), dtype=mx.float32) * float(init_std)
            ).astype(mx.float32)
            self.input_proj = CastedLinear(dim, self.state_dim)
            self.keep_proj = CastedLinear(dim, self.state_dim)
            self.out_proj = CastedLinear(self.state_dim, dim)
            for layer in (self.input_proj, self.keep_proj, self.out_proj):
                layer.weight = (
                    mx.random.normal(layer.weight.shape, dtype=mx.float32) * float(init_std)
                ).astype(mx.float32)
            self.feature_emb_fast = None
            self.feature_emb_slow = None
            self.input_proj_fast = None
            self.input_proj_slow = None
            self.keep_proj_fast = None
            self.keep_proj_slow = None
            self.out_proj_fast = None
            self.out_proj_slow = None
        else:
            self.feature_emb = None
            self.input_proj = None
            self.keep_proj = None
            self.out_proj = None
            self.feature_emb_fast = (
                mx.random.normal((num_features, self.fast_dim), dtype=mx.float32) * float(init_std)
            ).astype(mx.float32)
            self.feature_emb_slow = (
                mx.random.normal((num_features, self.slow_dim), dtype=mx.float32) * float(init_std)
            ).astype(mx.float32)
            self.input_proj_fast = CastedLinear(dim, self.fast_dim)
            self.input_proj_slow = CastedLinear(dim, self.slow_dim)
            self.keep_proj_fast = CastedLinear(dim, self.fast_dim)
            self.keep_proj_slow = CastedLinear(dim, self.slow_dim)
            self.out_proj_fast = CastedLinear(self.fast_dim, dim)
            self.out_proj_slow = CastedLinear(self.slow_dim, dim)
            for layer in (
                self.input_proj_fast,
                self.input_proj_slow,
                self.keep_proj_fast,
                self.keep_proj_slow,
                self.out_proj_fast,
                self.out_proj_slow,
            ):
                layer.weight = (
                    mx.random.normal(layer.weight.shape, dtype=mx.float32) * float(init_std)
                ).astype(mx.float32)

    def compute_reset_priors(
        self,
        feature_stack: mx.array,
        reset_prior: mx.array,
    ) -> tuple[mx.array, mx.array]:
        fast_prior = reset_prior.astype(mx.float32)
        if not self.hierarchical_enabled:
            return fast_prior, mx.zeros_like(fast_prior)
        feats = feature_stack.astype(mx.float32)
        def feat(name: str) -> mx.array:
            idx = self.feature_name_to_idx.get(name)
            if idx is None:
                return mx.zeros(feats.shape[:-1], dtype=mx.float32)
            return feats[..., idx]
        slow_prior = (
            0.70 * feat("boundary_paragraph_plus")
            + 1.00 * feat("boundary_section")
            + 0.30 * feat("quote_like")
            + 0.25 * feat("heading_like")
            + 0.18 * feat("list_like")
            + 0.20 * feat("code_like")
            + 0.12 * feat("emphasis_like")
        )
        return fast_prior, mx.clip(slow_prior, 0.0, 1.0)

    def __call__(
        self,
        x: mx.array,
        feature_stack: mx.array,
        reset_prior: mx.array,
    ) -> mx.array:
        if int(x.shape[1]) <= 0:
            return x
        fast_prior, slow_prior = self.compute_reset_priors(feature_stack, reset_prior)
        if not self.hierarchical_enabled:
            feat_summary = feature_stack.astype(x.dtype) @ self.feature_emb.astype(x.dtype)
            candidate = mx.tanh(self.input_proj(x.astype(COMPUTE_DTYPE)).astype(x.dtype) + feat_summary)
            keep = mx.sigmoid(
                self.keep_proj(x.astype(COMPUTE_DTYPE)).astype(x.dtype)
                - self.reset_prior_weight * fast_prior.astype(x.dtype)[..., None]
            )
            state = mx.zeros((x.shape[0], self.state_dim), dtype=x.dtype)
            states: list[mx.array] = []
            for idx in range(int(x.shape[1])):
                state = keep[:, idx, :] * state + (1.0 - keep[:, idx, :]) * candidate[:, idx, :]
                states.append(state)
            stacked = mx.stack(states, axis=1)
            residual = self.out_proj(stacked.astype(COMPUTE_DTYPE)).astype(x.dtype)
            return x + self.output_scale * residual
        feat_summary_fast = feature_stack.astype(x.dtype) @ self.feature_emb_fast.astype(x.dtype)
        feat_summary_slow = feature_stack.astype(x.dtype) @ self.feature_emb_slow.astype(x.dtype)
        candidate_fast = mx.tanh(self.input_proj_fast(x.astype(COMPUTE_DTYPE)).astype(x.dtype) + feat_summary_fast)
        candidate_slow = mx.tanh(self.input_proj_slow(x.astype(COMPUTE_DTYPE)).astype(x.dtype) + feat_summary_slow)
        keep_fast = mx.sigmoid(
            self.keep_proj_fast(x.astype(COMPUTE_DTYPE)).astype(x.dtype)
            - self.reset_prior_weight * fast_prior.astype(x.dtype)[..., None]
        )
        keep_slow = mx.sigmoid(
            self.keep_proj_slow(x.astype(COMPUTE_DTYPE)).astype(x.dtype)
            - self.reset_prior_weight * self.slow_reset_scale * slow_prior.astype(x.dtype)[..., None]
        )
        state_fast = mx.zeros((x.shape[0], self.fast_dim), dtype=x.dtype)
        state_slow = mx.zeros((x.shape[0], self.slow_dim), dtype=x.dtype)
        states_fast: list[mx.array] = []
        states_slow: list[mx.array] = []
        for idx in range(int(x.shape[1])):
            state_fast = keep_fast[:, idx, :] * state_fast + (1.0 - keep_fast[:, idx, :]) * candidate_fast[:, idx, :]
            state_slow = keep_slow[:, idx, :] * state_slow + (1.0 - keep_slow[:, idx, :]) * candidate_slow[:, idx, :]
            states_fast.append(state_fast)
            states_slow.append(state_slow)
        stacked_fast = mx.stack(states_fast, axis=1)
        stacked_slow = mx.stack(states_slow, axis=1)
        residual = (
            self.out_proj_fast(stacked_fast.astype(COMPUTE_DTYPE)).astype(x.dtype)
            + self.out_proj_slow(stacked_slow.astype(COMPUTE_DTYPE)).astype(x.dtype)
        )
        return x + self.output_scale * residual

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        name_prefix: str,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = make_turbo_linear(f"{name_prefix}.attn.c_q.weight", dim, dim)
        self.c_k = make_turbo_linear(f"{name_prefix}.attn.c_k.weight", dim, kv_dim)
        self.c_v = make_turbo_linear(f"{name_prefix}.attn.c_v.weight", dim, kv_dim)
        self.proj = make_turbo_linear(f"{name_prefix}.attn.proj.weight", dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        self.c_q.set_turbo_qat(enabled, alpha)
        self.c_k.set_turbo_qat(enabled, alpha)
        self.c_v.set_turbo_qat(enabled, alpha)
        self.proj.set_turbo_qat(enabled, alpha)

    def turbo_regularizer(self) -> mx.array:
        return self.c_q.turbo_regularizer() + self.c_k.turbo_regularizer() + self.c_v.turbo_regularizer() + self.proj.turbo_regularizer()

    def clear_turbo_cache(self) -> None:
        self.c_q.clear_turbo_cache()
        self.c_k.clear_turbo_cache()
        self.c_v.clear_turbo_cache()
        self.proj.clear_turbo_cache()

    def __call__(
        self,
        x: mx.array,
        mask: str | mx.array | None = "causal",
        *,
        q_bias: mx.array | None = None,
        tau: mx.array | None = None,
    ) -> mx.array:
        bsz, seqlen, dim = x.shape
        q_in = self.c_q(x)
        if q_bias is not None:
            q_in = q_in + q_bias.astype(q_in.dtype)
        q = q_in.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        if tau is not None:
            tau_term = mx.maximum(tau.astype(q.dtype), mx.array(1.0e-4, dtype=q.dtype))
            tau_term = tau_term.transpose(0, 2, 1)[..., None]
            q = q / tau_term
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, leaky_slope: float, name_prefix: str):
        super().__init__()
        hidden = dim * mlp_mult
        self.leaky_slope = leaky_slope
        self.fc = make_turbo_linear(f"{name_prefix}.mlp.fc.weight", dim, hidden)
        self.proj = make_turbo_linear(f"{name_prefix}.mlp.proj.weight", hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        x = mx.maximum(x, 0) + self.leaky_slope * mx.minimum(x, 0)
        return self.proj(x * x)

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        self.fc.set_turbo_qat(enabled, alpha)
        self.proj.set_turbo_qat(enabled, alpha)

    def turbo_regularizer(self) -> mx.array:
        return self.fc.turbo_regularizer() + self.proj.turbo_regularizer()

    def clear_turbo_cache(self) -> None:
        self.fc.clear_turbo_cache()
        self.proj.clear_turbo_cache()

class Block(nn.Module):
    def __init__(
        self,
        block_index: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_leaky_slope: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        name_prefix = f"blocks.{block_index}"
        self.attn = CausalSelfAttention(dim, name_prefix, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, mlp_leaky_slope, name_prefix)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(
        self,
        x: mx.array,
        x0: mx.array,
        attn_mask: str | mx.array | None = "causal",
        residual_budget: mx.array | None = None,
        attn_q_bias: mx.array | None = None,
        attn_tau: mx.array | None = None,
    ) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), mask=attn_mask, q_bias=attn_q_bias, tau=attn_tau)
        budget = residual_budget.astype(x.dtype) if residual_budget is not None else 1.0
        x = x + budget * self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + budget * self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        self.attn.set_turbo_qat(enabled, alpha)
        self.mlp.set_turbo_qat(enabled, alpha)

    def turbo_regularizer(self) -> mx.array:
        return self.attn.turbo_regularizer() + self.mlp.turbo_regularizer()

    def clear_turbo_cache(self) -> None:
        self.attn.clear_turbo_cache()
        self.mlp.clear_turbo_cache()

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_layer_templates: int, dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, mlp_leaky_slope: float, tie_embeddings: bool, logit_chunk_tokens: int, logit_softcap: float,
                 rope_base: float, tied_embed_init_std: float, qk_gain_init: float, num_registers: int = 0,
                 register_layout: str = "prefix", register_stride: int = 256,
                 logic_dim: int = 0, logic_layer_index: int | None = None, logic_route_to_next_token: bool = True,
                 operator_routing: OperatorRoutingSpec | None = None, register_mask_mode: str = "bidirectional",
                 logic_operator_mode: str = "all", polarity_detector_enabled: bool = False,
                 polarity_detector_layer_index: int | None = None, polarity_detector_hidden_dim: int = 0,
                 polarity_seed_blend: float = 1.0, polarity_seed_weight: float = 0.0,
                 polarity_sparse_weight: float = 0.0, polarity_smooth_weight: float = 0.0,
                 hardmax_struct_num_states: int = 0, hardmax_struct_dim: int = 32,
                 hardmax_struct_static_adapter: bool = False,
                 hardmax_struct_layer_index: int | None = None,
                 hardmax_struct_router_start_layer: int | None = None,
                 hardmax_struct_temperature: float = 1.0,
                 hardmax_struct_fast_refinement_steps: int = 1,
                 hardmax_struct_compute_min_scale: float = 0.35,
                 hardmax_struct_compute_power: float = 1.0,
                 hardmax_struct_route_residual_budget: bool = True,
                 hardmax_struct_operator_prior_scale: float = 1.0,
                 hardmax_struct_reset_prior_scale: float = 1.0,
                 hardmax_struct_usage_balance_weight: float = 0.0,
                 hardmax_struct_diversity_weight: float = 0.0,
                 hardmax_struct_predict_weight: float = 0.0,
                 hardmax_struct_confidence_weight: float = 0.0,
                 hardmax_struct_operator_weight: float = 1.0,
                 hardmax_struct_token_class_weight: float = 0.5,
                 hardmax_struct_boundary_weight: float = 1.0,
                 hardmax_struct_quote_weight: float = 0.25,
                 hardmax_struct_condition_mode: str = "residual",
                 hardmax_struct_attn_q_scale: float = 1.0,
                 hardmax_struct_attn_tau_min: float = 0.75,
                 early_exit_layer_index: int = -1, early_exit_horizons: tuple[int, ...] = (),
                 early_exit_aux_weight: float = 0.0, early_exit_head_init_std: float = 0.005,
                 early_exit_cascaded_enabled: bool = False, early_exit_condition_init_std: float = 0.001,
                 early_exit_branch_draft_enabled: bool = False, early_exit_branch_conf_threshold: float = 0.70,
                 early_exit_branch_max_draft_tokens: int = 1,
                 prosody_type_embeddings_enabled: bool = False, prosody_type_embedding_init_std: float = 0.002,
                 prosody_extended_feature_set_enabled: bool = False,
                 prosody_feature_embeddings_enabled: bool = False, prosody_feature_embedding_init_std: float = 0.002,
                 prosody_state_adapter_enabled: bool = False, prosody_state_dim: int = 64,
                 prosody_state_init_std: float = 0.005, prosody_state_scale: float = 0.50,
                 prosody_state_reset_prior_weight: float = 1.0,
                 prosody_state_hierarchical_enabled: bool = False,
                 prosody_state_slow_reset_scale: float = 0.35,
                 prosody_aux_layer_index: int = -1, prosody_aux_weight: float = 0.0,
                 prosody_aux_head_init_std: float = 0.005,
                 prosody_aux_token_class_weight: float = 1.0,
                 prosody_aux_boundary_weight: float = 1.0,
                 prosody_aux_quote_weight: float = 0.25,
                 prosody_aux_punctuation_weight: float = 0.5,
                 token_prosody_luts: TokenProsodyLuts | None = None):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if num_layer_templates <= 0 or num_layer_templates > num_layers:
            raise ValueError(
                f"num_layer_templates must be in [1, num_layers], got {num_layer_templates} for num_layers={num_layers}"
            )
        if num_registers < 0:
            raise ValueError(f"num_registers must be non-negative, got {num_registers}")
        if register_layout not in {"prefix", "interleaved"}:
            raise ValueError(f"Unsupported register_layout {register_layout!r}")
        if register_stride <= 0:
            raise ValueError(f"register_stride must be positive, got {register_stride}")
        if logic_dim < 0:
            raise ValueError(f"logic_dim must be non-negative, got {logic_dim}")
        if logic_dim > 0 and logic_layer_index is None:
            raise ValueError("logic_layer_index must be resolved when logic_dim > 0")
        if logic_dim > 0 and operator_routing is None:
            raise ValueError("operator_routing is required when logic_dim > 0")
        if hardmax_struct_num_states < 0:
            raise ValueError(f"hardmax_struct_num_states must be non-negative, got {hardmax_struct_num_states}")
        if hardmax_struct_num_states > 0 and hardmax_struct_static_adapter:
            raise ValueError("HARDMAX_STRUCT_STATIC_ADAPTER is incompatible with HARDMAX_STRUCT_NUM_STATES > 0")
        if (hardmax_struct_num_states > 0 or hardmax_struct_static_adapter) and hardmax_struct_dim <= 0:
            raise ValueError(
                f"hardmax_struct_dim must be positive when a hardmax structural path is enabled, got {hardmax_struct_dim}"
            )
        if (hardmax_struct_num_states > 0 or hardmax_struct_static_adapter) and hardmax_struct_layer_index is None:
            raise ValueError("hardmax_struct_layer_index must be resolved when a hardmax structural path is enabled")
        if (hardmax_struct_num_states > 0 or hardmax_struct_static_adapter) and hardmax_struct_router_start_layer is None:
            raise ValueError("hardmax_struct_router_start_layer must be resolved when a hardmax structural path is enabled")
        if hardmax_struct_fast_refinement_steps <= 0:
            raise ValueError(
                f"hardmax_struct_fast_refinement_steps must be positive, got {hardmax_struct_fast_refinement_steps}"
            )
        condition_mode = hardmax_struct_condition_mode.strip().lower()
        if condition_mode not in {"residual", "q_bias", "q_bias_temp"}:
            raise ValueError(
                f"hardmax_struct_condition_mode must be one of residual/q_bias/q_bias_temp, got {hardmax_struct_condition_mode!r}"
            )
        if register_mask_mode not in {"bidirectional", "causal"}:
            raise ValueError(f"Unsupported register_mask_mode {register_mask_mode!r}")
        if logic_operator_mode not in {"all", "not_only"}:
            raise ValueError(f"Unsupported logic_operator_mode {logic_operator_mode!r}")
        if polarity_detector_enabled and logic_dim <= 0:
            raise ValueError("POLARITY_DETECTOR_ENABLED requires LOGIC_DIM > 0")
        if polarity_detector_enabled and polarity_detector_layer_index is None:
            raise ValueError("POLARITY_DETECTOR_ENABLED requires a resolved detector layer index")
        if polarity_detector_enabled and logic_layer_index is not None and polarity_detector_layer_index >= logic_layer_index:
            raise ValueError(
                f"POLARITY_DETECTOR_LAYER_INDEX must be earlier than LOGIC_LAYER_INDEX, got "
                f"{polarity_detector_layer_index} >= {logic_layer_index}"
            )
        if not (0.0 <= polarity_seed_blend <= 1.0):
            raise ValueError(f"POLARITY_SEED_BLEND must be in [0, 1], got {polarity_seed_blend}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_layer_templates = num_layer_templates
        self.tie_embeddings = tie_embeddings
        self.num_registers = num_registers
        self.register_layout = register_layout
        self.register_stride = register_stride
        self.register_mask_mode = register_mask_mode
        self.logic_layer_index = logic_layer_index
        self.logic_route_to_next_token = logic_route_to_next_token
        self.logic_operator_mode = logic_operator_mode
        self.polarity_detector_layer_index = polarity_detector_layer_index
        self.polarity_seed_blend = polarity_seed_blend
        self.polarity_seed_weight = polarity_seed_weight
        self.polarity_sparse_weight = polarity_sparse_weight
        self.polarity_smooth_weight = polarity_smooth_weight
        self.hardmax_struct_layer_index = hardmax_struct_layer_index
        self.hardmax_struct_router_start_layer = hardmax_struct_router_start_layer
        self.hardmax_struct_fast_refinement_steps = int(hardmax_struct_fast_refinement_steps)
        self.hardmax_struct_route_residual_budget = bool(hardmax_struct_route_residual_budget)
        self.hardmax_struct_static_adapter = bool(hardmax_struct_static_adapter)
        self.hardmax_struct_condition_mode = condition_mode
        self.hardmax_struct_attn_q_scale = float(hardmax_struct_attn_q_scale)
        self.hardmax_struct_attn_tau_min = max(float(hardmax_struct_attn_tau_min), 1.0e-3)
        self._hardmax_struct_usage_balance_weight = float(hardmax_struct_usage_balance_weight)
        self._hardmax_struct_diversity_weight = float(hardmax_struct_diversity_weight)
        self._hardmax_struct_predict_weight = float(hardmax_struct_predict_weight)
        self._hardmax_struct_confidence_weight = float(hardmax_struct_confidence_weight)
        self._hardmax_struct_operator_weight = float(hardmax_struct_operator_weight)
        self._hardmax_struct_token_class_weight = float(hardmax_struct_token_class_weight)
        self._hardmax_struct_boundary_weight = float(hardmax_struct_boundary_weight)
        self._hardmax_struct_quote_weight = float(hardmax_struct_quote_weight)
        self._early_exit_horizons = tuple(int(value) for value in early_exit_horizons)
        self._early_exit_aux_weight = float(early_exit_aux_weight)
        self._early_exit_cascaded_enabled = bool(early_exit_cascaded_enabled)
        self._early_exit_condition_init_std = float(early_exit_condition_init_std)
        self._early_exit_branch_draft_enabled = bool(early_exit_branch_draft_enabled)
        self._early_exit_branch_conf_threshold = float(early_exit_branch_conf_threshold)
        self._early_exit_branch_max_draft_tokens = max(int(early_exit_branch_max_draft_tokens), 1)
        self._early_exit_layer_index = resolve_early_exit_layer_index(int(early_exit_layer_index), num_layers)
        self._prosody_type_embeddings_enabled = bool(prosody_type_embeddings_enabled and token_prosody_luts is not None)
        self._prosody_extended_feature_set_enabled = bool(prosody_extended_feature_set_enabled)
        self._prosody_feature_embeddings_enabled = bool(prosody_feature_embeddings_enabled and token_prosody_luts is not None)
        self._prosody_state_adapter_enabled = bool(prosody_state_adapter_enabled and token_prosody_luts is not None)
        self._prosody_aux_weight = float(prosody_aux_weight)
        self._prosody_aux_layer_index = resolve_prosody_aux_layer_index(int(prosody_aux_layer_index), num_layers)
        self._prosody_aux_token_class_weight = float(prosody_aux_token_class_weight)
        self._prosody_aux_boundary_weight = float(prosody_aux_boundary_weight)
        self._prosody_aux_quote_weight = float(prosody_aux_quote_weight)
        self._prosody_aux_punctuation_weight = float(prosody_aux_punctuation_weight)

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.register_tokens = RegisterTokens(num_registers, dim) if num_registers > 0 else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(i, dim, num_heads, num_kv_heads, mlp_mult, mlp_leaky_slope, rope_base, qk_gain_init)
            for i in range(num_layer_templates)
        ]
        self.logic_sidecar = LogicSideCar(dim, logic_dim, operator_mode=logic_operator_mode) if logic_dim > 0 else None
        self.hardmax_structural_controller = None
        if hardmax_struct_static_adapter:
            self.hardmax_structural_controller = StaticStructuralAdapter(
                dim,
                hardmax_struct_dim,
                operator_prior_scale=float(hardmax_struct_operator_prior_scale),
                reset_prior_scale=float(hardmax_struct_reset_prior_scale),
            )
        elif hardmax_struct_num_states > 0:
            self.hardmax_structural_controller = HardmaxStructuralController(
                dim,
                hardmax_struct_dim,
                hardmax_struct_num_states,
                temperature=float(hardmax_struct_temperature),
                compute_min_scale=float(hardmax_struct_compute_min_scale),
                compute_power=float(hardmax_struct_compute_power),
                operator_prior_scale=float(hardmax_struct_operator_prior_scale),
                reset_prior_scale=float(hardmax_struct_reset_prior_scale),
            )
        self.hardmax_struct_operator_head: CastedLinear | None = None
        self.hardmax_struct_token_class_head: CastedLinear | None = None
        self.hardmax_struct_boundary_head: CastedLinear | None = None
        self.hardmax_struct_quote_head: CastedLinear | None = None
        self.hardmax_struct_attn_q_proj: CastedLinear | None = None
        self.hardmax_struct_attn_tau_proj: CastedLinear | None = None
        if self.hardmax_structural_controller is not None and self.hardmax_struct_condition_mode in {"q_bias", "q_bias_temp"}:
            self.hardmax_struct_attn_q_proj = CastedLinear(hardmax_struct_dim, dim)
            self.hardmax_struct_attn_q_proj.weight = mx.zeros_like(self.hardmax_struct_attn_q_proj.weight)
            if self.hardmax_struct_condition_mode == "q_bias_temp":
                self.hardmax_struct_attn_tau_proj = CastedLinear(hardmax_struct_dim, num_heads)
                self.hardmax_struct_attn_tau_proj.weight = mx.zeros_like(self.hardmax_struct_attn_tau_proj.weight)
        if self.hardmax_structural_controller is not None and (
            self._hardmax_struct_predict_weight > 0.0 or self._hardmax_struct_confidence_weight > 0.0
        ):
            self.hardmax_struct_operator_head = CastedLinear(hardmax_struct_dim, 4)
            self.hardmax_struct_token_class_head = CastedLinear(hardmax_struct_dim, len(TOKEN_CLASS_NAMES))
            self.hardmax_struct_boundary_head = CastedLinear(hardmax_struct_dim, len(BOUNDARY_STRENGTH_NAMES))
            self.hardmax_struct_quote_head = CastedLinear(hardmax_struct_dim, 2)
        self.polarity_detector = (
            PolarityDetector(dim, hidden_dim=polarity_detector_hidden_dim)
            if polarity_detector_enabled else None
        )
        self.final_norm = RMSNormNoWeight()
        self._register_attention_masks: dict[int, mx.array] = {}
        self._local_attention_masks: dict[tuple[int, int, int], mx.array] = {}
        self._operator_routing = operator_routing
        self._hardmax_eval_ablation = resolve_hardmax_eval_ablation(None)

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        self.prosody_token_class_emb: nn.Embedding | None = None
        self.prosody_boundary_emb: nn.Embedding | None = None
        self.prosody_quote_emb: nn.Embedding | None = None
        self.prosody_punctuation_emb: nn.Embedding | None = None
        self.prosody_feature_emb: mx.array | None = None
        self.prosody_state_adapter: ProsodyStateAdapter | None = None
        self._prosody_feature_names: tuple[str, ...] = ()
        self._prosody_feature_name_to_idx: dict[str, int] = {}
        self._prosody_token_class_lut = None
        self._prosody_boundary_strength_lut = None
        self._prosody_punctuation_role_lut = None
        self._prosody_quote_like_lut = None
        self._prosody_binary_feature_lut = None
        self._prosody_reset_prior_lut = None
        if token_prosody_luts is not None:
            self._prosody_feature_names = tuple(token_prosody_luts.binary_feature_names)
            self._prosody_feature_name_to_idx = {name: idx for idx, name in enumerate(self._prosody_feature_names)}
            self._prosody_token_class_lut = mx.array(token_prosody_luts.token_class_ids, dtype=mx.int32)
            self._prosody_boundary_strength_lut = mx.array(token_prosody_luts.boundary_strength_ids, dtype=mx.int32)
            self._prosody_punctuation_role_lut = mx.array(token_prosody_luts.punctuation_role_ids, dtype=mx.int32)
            self._prosody_quote_like_lut = mx.array(token_prosody_luts.quote_like_ids, dtype=mx.int32)
            self._prosody_binary_feature_lut = mx.array(token_prosody_luts.binary_feature_ids, dtype=mx.int32)
            self._prosody_reset_prior_lut = mx.array(token_prosody_luts.reset_prior_values, dtype=mx.float32)
        if self._prosody_type_embeddings_enabled:
            self.prosody_token_class_emb = nn.Embedding(len(TOKEN_CLASS_NAMES), dim)
            self.prosody_boundary_emb = nn.Embedding(len(BOUNDARY_STRENGTH_NAMES), dim)
            self.prosody_quote_emb = nn.Embedding(2, dim)
            emb_list = [self.prosody_token_class_emb, self.prosody_boundary_emb, self.prosody_quote_emb]
            if self._prosody_extended_feature_set_enabled:
                self.prosody_punctuation_emb = nn.Embedding(len(PUNCTUATION_ROLE_NAMES), dim)
                emb_list.append(self.prosody_punctuation_emb)
            for emb in emb_list:
                emb.weight = (
                    mx.random.normal(emb.weight.shape, dtype=mx.float32) * float(prosody_type_embedding_init_std)
                ).astype(COMPUTE_DTYPE)
        if self._prosody_feature_embeddings_enabled:
            self.prosody_feature_emb = (
                mx.random.normal((len(self._prosody_feature_names), dim), dtype=mx.float32)
                * float(prosody_feature_embedding_init_std)
            ).astype(COMPUTE_DTYPE)
        if self._prosody_state_adapter_enabled:
            self.prosody_state_adapter = ProsodyStateAdapter(
                dim,
                state_dim=int(prosody_state_dim),
                num_features=len(self._prosody_feature_names),
                init_std=float(prosody_state_init_std),
                output_scale=float(prosody_state_scale),
                reset_prior_weight=float(prosody_state_reset_prior_weight),
                hierarchical_enabled=bool(prosody_state_hierarchical_enabled),
                slow_reset_scale=float(prosody_state_slow_reset_scale),
                feature_name_to_idx=self._prosody_feature_name_to_idx,
            )
        if not tie_embeddings:
            self.lm_head = make_turbo_linear("lm_head.weight", dim, vocab_size)
            self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(mx.float32)
        self.early_exit_heads: list[CastedLinear] = []
        self.early_exit_condition_projs: list[CastedLinear] = []
        if self._early_exit_horizons:
            self.early_exit_heads = [CastedLinear(dim, vocab_size) for _ in self._early_exit_horizons]
            for head in self.early_exit_heads:
                head.weight = (
                    mx.random.normal(head.weight.shape, dtype=mx.float32) * early_exit_head_init_std
                ).astype(mx.float32)
            if self._early_exit_cascaded_enabled:
                self.early_exit_condition_projs = [CastedLinear(dim, dim) for _ in self._early_exit_horizons]
                for proj in self.early_exit_condition_projs:
                    proj.weight = (
                        mx.random.normal(proj.weight.shape, dtype=mx.float32) * self._early_exit_condition_init_std
                    ).astype(mx.float32)
        self.prosody_token_class_head: CastedLinear | None = None
        self.prosody_boundary_head: CastedLinear | None = None
        self.prosody_quote_head: CastedLinear | None = None
        self.prosody_punctuation_head: CastedLinear | None = None
        if token_prosody_luts is not None and self._prosody_aux_weight > 0.0:
            self.prosody_token_class_head = CastedLinear(dim, len(TOKEN_CLASS_NAMES))
            self.prosody_boundary_head = CastedLinear(dim, len(BOUNDARY_STRENGTH_NAMES))
            self.prosody_quote_head = CastedLinear(dim, 2)
            self.prosody_punctuation_head = CastedLinear(dim, len(PUNCTUATION_ROLE_NAMES))
            for head in (
                self.prosody_token_class_head,
                self.prosody_boundary_head,
                self.prosody_quote_head,
                self.prosody_punctuation_head,
            ):
                head.weight = (
                    mx.random.normal(head.weight.shape, dtype=mx.float32) * float(prosody_aux_head_init_std)
                ).astype(mx.float32)

    def has_early_exit_aux(self) -> bool:
        return bool(self.early_exit_heads) and self._early_exit_aux_weight > 0.0

    def has_early_exit_branch_drafter(self) -> bool:
        return bool(self.early_exit_heads) and self._early_exit_branch_draft_enabled

    def has_cascaded_early_exit(self) -> bool:
        return bool(self.early_exit_condition_projs) and self._early_exit_cascaded_enabled

    def has_prosody_type_embeddings(self) -> bool:
        base_ready = (
            self.prosody_token_class_emb is not None
            and self.prosody_boundary_emb is not None
            and self.prosody_quote_emb is not None
            and self._prosody_token_class_lut is not None
            and self._prosody_boundary_strength_lut is not None
            and self._prosody_quote_like_lut is not None
        )
        if not base_ready:
            return False
        if self._prosody_extended_feature_set_enabled:
            return self.prosody_punctuation_emb is not None and self._prosody_punctuation_role_lut is not None
        return True

    def has_prosody_feature_embeddings(self) -> bool:
        return self.prosody_feature_emb is not None and self._prosody_binary_feature_lut is not None

    def has_prosody_state_adapter(self) -> bool:
        return (
            self.prosody_state_adapter is not None
            and self._prosody_binary_feature_lut is not None
            and self._prosody_reset_prior_lut is not None
        )

    def has_prosody_aux(self) -> bool:
        return (
            self.prosody_token_class_head is not None
            and self.prosody_boundary_head is not None
            and self.prosody_quote_head is not None
            and self.prosody_punctuation_head is not None
            and self._prosody_aux_weight > 0.0
            and self._prosody_token_class_lut is not None
            and self._prosody_boundary_strength_lut is not None
            and self._prosody_punctuation_role_lut is not None
            and self._prosody_quote_like_lut is not None
        )

    def has_hardmax_structural_controller(self) -> bool:
        return self.hardmax_structural_controller is not None

    def current_hardmax_eval_ablation(self) -> HardmaxEvalAblationSpec:
        return self._hardmax_eval_ablation

    @contextmanager
    def hardmax_eval_ablation_scope(
        self,
        spec: str | HardmaxEvalAblationSpec | None,
    ):
        previous = self._hardmax_eval_ablation
        self._hardmax_eval_ablation = resolve_hardmax_eval_ablation(spec)
        try:
            yield self
        finally:
            self._hardmax_eval_ablation = previous

    def _hardmax_structural_layer_budget(
        self,
        structural_budget: mx.array | None,
        layer_idx: int,
    ) -> mx.array | None:
        ablation = self.current_hardmax_eval_ablation()
        if ablation.disable_controller or ablation.disable_residual_budget:
            return None
        if (
            structural_budget is None
            or not self.hardmax_struct_route_residual_budget
            or self.hardmax_struct_router_start_layer is None
            or layer_idx < self.hardmax_struct_router_start_layer
        ):
            return None
        return structural_budget

    def expand_float_feature_for_registers(self, values: mx.array) -> mx.array:
        if self.num_registers <= 0:
            return values
        if self.register_layout == "prefix":
            pad = mx.zeros((values.shape[0], self.num_registers), dtype=values.dtype)
            return mx.concatenate([pad, values], axis=1)
        parts: list[mx.array] = []
        zero_block = mx.zeros((values.shape[0], self.num_registers), dtype=values.dtype)
        token_len = int(values.shape[1])
        for start in range(0, token_len, self.register_stride):
            end = min(start + self.register_stride, token_len)
            parts.append(values[:, start:end])
            if end < token_len:
                parts.append(zero_block)
        return mx.concatenate(parts, axis=1) if parts else values[:, :0]

    def structural_reset_prior_for_input(self, input_ids: mx.array) -> mx.array | None:
        if self._prosody_reset_prior_lut is None:
            return None
        ids = input_ids.astype(mx.int32)
        reset_prior = self._prosody_reset_prior_lut[ids]
        return self.expand_float_feature_for_registers(reset_prior.astype(mx.float32))

    def _prosody_feature_stack_and_reset(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        if self._prosody_binary_feature_lut is None or self._prosody_reset_prior_lut is None:
            raise ValueError("Prosody feature stack requested without binary feature LUTs")
        ids = input_ids.astype(mx.int32)
        return (
            self._prosody_binary_feature_lut[ids],
            self._prosody_reset_prior_lut[ids],
        )

    def prosody_runtime_stats(self, input_ids: mx.array) -> dict[str, float]:
        if self._prosody_binary_feature_lut is None or self._prosody_reset_prior_lut is None:
            return {
                "prosody_feature_density": 0.0,
                "prosody_reset_prior": 0.0,
                "prosody_slow_reset_prior": 0.0,
                "prosody_punctuation_frac": 0.0,
                "prosody_quote_frac": 0.0,
                "prosody_sentence_boundary_frac": 0.0,
                "prosody_paragraph_boundary_frac": 0.0,
                "prosody_state_delta_rms": 0.0,
            }
        feature_stack, reset_prior = self._prosody_feature_stack_and_reset(input_ids)
        feature_np = np.asarray(feature_stack, dtype=np.float32)
        reset_np = np.asarray(reset_prior, dtype=np.float32)
        slow_reset_np = None
        if self.has_prosody_state_adapter():
            _fast_prior, slow_prior = self.prosody_state_adapter.compute_reset_priors(feature_stack, reset_prior)
            slow_reset_np = np.asarray(slow_prior, dtype=np.float32)
        def feature_mean(name: str) -> float:
            idx = self._prosody_feature_name_to_idx.get(name)
            if idx is None or feature_np.size <= 0:
                return 0.0
            return float(feature_np[..., idx].mean())
        stats = {
            "prosody_feature_density": float(feature_np.mean()) if feature_np.size > 0 else 0.0,
            "prosody_reset_prior": float(reset_np.mean()) if reset_np.size > 0 else 0.0,
            "prosody_slow_reset_prior": float(slow_reset_np.mean()) if slow_reset_np is not None and slow_reset_np.size > 0 else 0.0,
            "prosody_punctuation_frac": feature_mean("punctuation_like"),
            "prosody_quote_frac": feature_mean("quote_like"),
            "prosody_sentence_boundary_frac": feature_mean("boundary_sentence_plus"),
            "prosody_paragraph_boundary_frac": feature_mean("boundary_paragraph_plus"),
            "prosody_state_delta_rms": 0.0,
        }
        if self.has_prosody_state_adapter():
            base = self.tok_emb(input_ids.astype(mx.int32)).astype(COMPUTE_DTYPE)
            adapted = self.prosody_state_adapter(base, feature_stack, reset_prior)
            delta = adapted - base
            stats["prosody_state_delta_rms"] = float(
                mx.sqrt(mx.mean(mx.square(delta.astype(mx.float32)))).item()
            )
        return stats

    def primary_early_exit_head_idx(self) -> int:
        if not self.early_exit_heads:
            raise ValueError("No early-exit heads are available")
        if 1 in self._early_exit_horizons:
            return self._early_exit_horizons.index(1)
        return 0

    def ordered_early_exit_draft_heads(self) -> list[tuple[int, int]]:
        return sorted(
            [(int(horizon), int(head_idx)) for head_idx, horizon in enumerate(self._early_exit_horizons)],
            key=lambda item: item[0],
        )

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        for block in self.blocks:
            block.set_turbo_qat(enabled, alpha)
        if not self.tie_embeddings:
            self.lm_head.set_turbo_qat(enabled, alpha)

    def turbo_regularizer(self) -> mx.array:
        loss = mx.array(0.0, dtype=mx.float32)
        for block in self.blocks:
            loss = loss + block.turbo_regularizer()
        if not self.tie_embeddings:
            loss = loss + self.lm_head.turbo_regularizer()
        return loss

    def clear_turbo_cache(self) -> None:
        for block in self.blocks:
            block.clear_turbo_cache()
        if not self.tie_embeddings:
            self.lm_head.clear_turbo_cache()

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def block_for_step(self, step: int) -> Block:
        return self.blocks[step % self.num_layer_templates]

    def strip_registers(self, x: mx.array) -> mx.array:
        return (
            self.register_tokens.strip(x, layout=self.register_layout, register_stride=self.register_stride)
            if self.register_tokens is not None else x
        )

    def embed_inputs(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.has_prosody_type_embeddings():
            token_class_ids = self._prosody_token_class_lut[input_ids.astype(mx.int32)]
            boundary_ids = self._prosody_boundary_strength_lut[input_ids.astype(mx.int32)]
            quote_ids = self._prosody_quote_like_lut[input_ids.astype(mx.int32)]
            x = x + self.prosody_token_class_emb(token_class_ids).astype(COMPUTE_DTYPE)
            x = x + self.prosody_boundary_emb(boundary_ids).astype(COMPUTE_DTYPE)
            if self._prosody_extended_feature_set_enabled and self.prosody_punctuation_emb is not None and self._prosody_punctuation_role_lut is not None:
                punctuation_role_ids = self._prosody_punctuation_role_lut[input_ids.astype(mx.int32)]
                x = x + self.prosody_punctuation_emb(punctuation_role_ids).astype(COMPUTE_DTYPE)
            x = x + self.prosody_quote_emb(quote_ids).astype(COMPUTE_DTYPE)
        feature_stack: mx.array | None = None
        reset_prior: mx.array | None = None
        if self.has_prosody_feature_embeddings() or self.has_prosody_state_adapter():
            feature_stack, reset_prior = self._prosody_feature_stack_and_reset(input_ids)
        if self.has_prosody_feature_embeddings() and feature_stack is not None:
            x = x + feature_stack.astype(COMPUTE_DTYPE) @ self.prosody_feature_emb.astype(COMPUTE_DTYPE)
        if self.has_prosody_state_adapter() and feature_stack is not None and reset_prior is not None:
            x = self.prosody_state_adapter(x, feature_stack, reset_prior)
        if self.register_tokens is not None:
            x = self.register_tokens.inject(x, layout=self.register_layout, register_stride=self.register_stride)
        return rms_norm(x)

    def attention_mask(self, total_len: int, context_window: int | None = None) -> str | mx.array:
        if self.num_registers <= 0 or self.register_layout == "interleaved" or self.register_mask_mode == "causal":
            base_mask: str | mx.array = "causal"
        else:
            cached = self._register_attention_masks.get(total_len)
            if cached is None:
                cached = build_register_attention_mask_with_mode(total_len, self.num_registers, mode=self.register_mask_mode)
                self._register_attention_masks[total_len] = cached
            base_mask = cached

        if context_window is None or context_window <= 0:
            return base_mask

        prefix_tokens = self.num_registers if self.num_registers > 0 and self.register_layout == "prefix" else 0
        mask_key = (total_len, context_window, prefix_tokens)
        local_mask = self._local_attention_masks.get(mask_key)
        if local_mask is None:
            local_mask = build_local_causal_attention_mask(total_len, context_window, prefix_tokens=prefix_tokens)
            self._local_attention_masks[mask_key] = local_mask
        if isinstance(base_mask, str):
            return local_mask
        return mx.minimum(base_mask, local_mask)

    def operator_codes_for_numpy(
        self,
        input_ids: np.ndarray,
        *,
        route_to_next: bool | None = None,
        pad_registers_for_output: bool = True,
    ) -> np.ndarray | None:
        if self._operator_routing is None:
            return None
        operator_codes = detect_operator_codes_np(input_ids, self._operator_routing)
        if route_to_next is None:
            route_to_next = self.logic_route_to_next_token
        if route_to_next:
            operator_codes = route_operator_codes(operator_codes)
        if pad_registers_for_output:
            if self.register_layout == "prefix":
                operator_codes = pad_operator_codes(operator_codes, self.num_registers)
            else:
                operator_codes = interleave_operator_codes(
                    operator_codes,
                    self.num_registers,
                    layout=self.register_layout,
                    register_stride=self.register_stride,
                )
        return np.ascontiguousarray(operator_codes, dtype=np.int32)

    def operator_codes_for_input(
        self,
        input_ids: mx.array,
        *,
        route_to_next: bool | None = None,
        pad_registers_for_output: bool = True,
    ) -> mx.array | None:
        operator_codes = self.operator_codes_for_numpy(
            np.asarray(input_ids, dtype=np.int32),
            route_to_next=route_to_next,
            pad_registers_for_output=pad_registers_for_output,
        )
        if operator_codes is None:
            return None
        return mx.array(operator_codes, dtype=mx.int32)

    def seed_polarity_scores(self, operator_codes: mx.array | None) -> mx.array | None:
        if operator_codes is None:
            return None
        return (operator_codes == 1).astype(mx.float32)

    def polarity_detector_logits_and_scores(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self.polarity_detector is None:
            raise ValueError("polarity_detector_logits_and_scores called without a detector")
        logits = self.polarity_detector(x).astype(mx.float32)
        if self.num_registers > 0:
            reg_mask_np = register_position_mask(
                int(x.shape[1]),
                self.num_registers,
                layout=self.register_layout,
                register_stride=self.register_stride,
            )
            reg_mask = mx.array(reg_mask_np[None, :], dtype=mx.bool_)
            logits = mx.where(reg_mask, mx.full(logits.shape, -20.0, dtype=mx.float32), logits)
        return logits, mx.sigmoid(logits)

    def blend_polarity_scores(
        self,
        seed_scores: mx.array | None,
        detector_scores: mx.array | None,
    ) -> mx.array | None:
        if detector_scores is None:
            return seed_scores
        if seed_scores is None:
            return detector_scores
        alpha = self.polarity_seed_blend
        if alpha <= 0.0:
            return detector_scores
        if alpha >= 1.0:
            return seed_scores
        return alpha * seed_scores + (1.0 - alpha) * detector_scores

    def maybe_apply_logic_sidecar(
        self,
        x: mx.array,
        operator_codes: mx.array | None,
        layer_idx: int,
        polarity_scores: mx.array | None = None,
    ) -> mx.array:
        if self.logic_sidecar is None or operator_codes is None or layer_idx != self.logic_layer_index:
            if self.logic_sidecar is None or layer_idx != self.logic_layer_index:
                return x
            if operator_codes is None and polarity_scores is None:
                return x
        return self.logic_sidecar(x, operator_codes, polarity_scores=polarity_scores)

    def maybe_apply_hardmax_structural_controller(
        self,
        x: mx.array,
        operator_codes: mx.array | None,
        layer_idx: int,
        *,
        polarity_scores: mx.array | None = None,
        reset_prior: mx.array | None = None,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        ablation = self.current_hardmax_eval_ablation()
        if (
            self.hardmax_structural_controller is None
            or layer_idx != self.hardmax_struct_layer_index
            or ablation.disable_controller
        ):
            return x, None
        current = x
        controller_input = x
        aux: dict[str, mx.array] | None = None
        for _ in range(max(int(self.hardmax_struct_fast_refinement_steps), 1)):
            controller_output, aux = self.hardmax_structural_controller(
                controller_input,
                operator_codes,
                polarity_scores=polarity_scores,
                reset_prior=reset_prior,
            )
            controller_input = controller_output
            if self.hardmax_struct_condition_mode == "residual" and not ablation.disable_residual_write:
                current = controller_output
        return current, aux

    def hardmax_structural_attention_condition_from_aux(
        self,
        aux: dict[str, mx.array] | None,
        dtype: mx.Dtype,
    ) -> tuple[mx.array | None, mx.array | None]:
        ablation = self.current_hardmax_eval_ablation()
        if self.hardmax_structural_controller is None:
            return None, None
        if ablation.disable_controller:
            return None, None
        if self.hardmax_struct_condition_mode not in {"q_bias", "q_bias_temp"}:
            return None, None
        if not isinstance(aux, dict):
            return None, None
        struct_state = aux.get("struct_state")
        if struct_state is None or self.hardmax_struct_attn_q_proj is None:
            return None, None
        q_bias = self.hardmax_struct_attn_q_scale * self.hardmax_struct_attn_q_proj(
            struct_state.astype(COMPUTE_DTYPE)
        ).astype(dtype)
        tau = None
        if self.hardmax_struct_condition_mode == "q_bias_temp" and self.hardmax_struct_attn_tau_proj is not None:
            tau_raw = self.hardmax_struct_attn_tau_proj(struct_state.astype(COMPUTE_DTYPE)).astype(mx.float32)
            tau = 1.0 + nn.softplus(tau_raw) - float(np.log(2.0))
            tau = mx.maximum(tau, mx.array(self.hardmax_struct_attn_tau_min, dtype=mx.float32))
        if ablation.disable_attn_q_bias:
            q_bias = None
        if ablation.disable_attn_tau:
            tau = None
        return q_bias, tau

    def hardmax_structural_loss_terms_from_aux(
        self,
        aux: dict[str, mx.array | None],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        zero = mx.array(0.0, dtype=mx.float32)
        if self.hardmax_structural_controller is None:
            return zero, zero, zero, zero, zero
        structural_aux = aux.get("hardmax_structural")
        if not isinstance(structural_aux, dict):
            return zero, zero, zero, zero, zero
        balance_loss, diversity_loss, entropy = self.hardmax_structural_controller.regularization_losses(structural_aux)
        confidence = structural_aux.get("confidence")
        budget = structural_aux.get("budget")
        confidence_mean = (
            zero if confidence is None else mx.mean(confidence.astype(mx.float32))
        )
        budget_mean = zero if budget is None else mx.mean(budget.astype(mx.float32))
        return balance_loss, diversity_loss, confidence_mean, budget_mean, entropy

    def has_hardmax_structural_prediction_heads(self) -> bool:
        return (
            self.hardmax_struct_operator_head is not None
            and self.hardmax_struct_token_class_head is not None
            and self.hardmax_struct_boundary_head is not None
            and self.hardmax_struct_quote_head is not None
        )

    def _hardmax_structural_prediction_logits(
        self,
        struct_state: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if not self.has_hardmax_structural_prediction_heads():
            raise ValueError("Hardmax structural prediction heads are not available")
        op_logits = self.hardmax_struct_operator_head(struct_state)
        cls_logits = self.hardmax_struct_token_class_head(struct_state)
        bnd_logits = self.hardmax_struct_boundary_head(struct_state)
        quote_logits = self.hardmax_struct_quote_head(struct_state)
        return op_logits, cls_logits, bnd_logits, quote_logits

    def hardmax_structural_prediction_loss_terms(
        self,
        aux: dict[str, mx.array | None],
        target_ids: mx.array,
        *,
        token_weights: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        zero = mx.array(0.0, dtype=mx.float32)
        if not self.has_hardmax_structural_prediction_heads():
            return zero, zero, zero, zero, zero, zero
        structural_aux = aux.get("hardmax_structural")
        if not isinstance(structural_aux, dict):
            return zero, zero, zero, zero, zero, zero
        struct_state = structural_aux.get("struct_state")
        confidence = structural_aux.get("confidence")
        if struct_state is None or confidence is None:
            return zero, zero, zero, zero, zero, zero
        struct_state = strip_register_positions(
            struct_state,
            self.num_registers,
            layout=self.register_layout,
            register_stride=self.register_stride,
        )
        confidence = strip_register_positions(
            confidence.astype(mx.float32),
            self.num_registers,
            layout=self.register_layout,
            register_stride=self.register_stride,
        )
        if isinstance(struct_state, np.ndarray) or isinstance(confidence, np.ndarray):
            raise TypeError("hardmax_structural_prediction_loss_terms expects mlx arrays")
        op_logits, cls_logits, bnd_logits, quote_logits = self._hardmax_structural_prediction_logits(struct_state)
        target_operator = self.operator_codes_for_input(
            target_ids,
            route_to_next=False,
            pad_registers_for_output=False,
        )
        if target_operator is None:
            target_operator = mx.zeros(target_ids.shape, dtype=mx.int32)
        token_class_lut, boundary_lut, _punct_lut, quote_lut = self._prosody_label_luts()
        target_token_class = token_class_lut[target_ids.astype(mx.int32)]
        target_boundary = boundary_lut[target_ids.astype(mx.int32)]
        target_quote = quote_lut[target_ids.astype(mx.int32)]
        flat_weights = token_weights.reshape(-1) if token_weights is not None else None
        op_loss = token_cross_entropy_with_focal(
            op_logits.reshape(-1, op_logits.shape[-1]),
            target_operator.reshape(-1),
            token_weights=flat_weights,
        )
        cls_loss = token_cross_entropy_with_focal(
            cls_logits.reshape(-1, cls_logits.shape[-1]),
            target_token_class.reshape(-1),
            token_weights=flat_weights,
        )
        bnd_loss = token_cross_entropy_with_focal(
            bnd_logits.reshape(-1, bnd_logits.shape[-1]),
            target_boundary.reshape(-1),
            token_weights=flat_weights,
        )
        quote_loss = token_cross_entropy_with_focal(
            quote_logits.reshape(-1, quote_logits.shape[-1]),
            target_quote.reshape(-1),
            token_weights=flat_weights,
        )
        pred_total = (
            self._hardmax_struct_operator_weight * op_loss
            + self._hardmax_struct_token_class_weight * cls_loss
            + self._hardmax_struct_boundary_weight * bnd_loss
            + self._hardmax_struct_quote_weight * quote_loss
        )
        probs = [
            mx.softmax(op_logits.astype(mx.float32), axis=-1),
            mx.softmax(cls_logits.astype(mx.float32), axis=-1),
            mx.softmax(bnd_logits.astype(mx.float32), axis=-1),
            mx.softmax(quote_logits.astype(mx.float32), axis=-1),
        ]
        norm_entropies = []
        for prob in probs:
            entropy = -mx.sum(prob * mx.log(mx.maximum(prob, mx.array(1.0e-8, dtype=mx.float32))), axis=-1)
            norm = math.log(max(int(prob.shape[-1]), 2))
            norm_entropies.append(entropy / norm)
        mean_norm_entropy = sum(norm_entropies) / len(norm_entropies)
        target_confidence = mx.stop_gradient(1.0 - mean_norm_entropy.astype(mx.float32))
        conf_sq = mx.square(confidence.astype(mx.float32) - target_confidence)
        if token_weights is None:
            conf_loss = mx.mean(conf_sq)
        else:
            weights = token_weights.astype(mx.float32)
            conf_loss = mx.sum(conf_sq * weights) / mx.maximum(mx.sum(weights), mx.array(1.0e-6, dtype=mx.float32))
        return (
            pred_total.astype(mx.float32),
            op_loss.astype(mx.float32),
            cls_loss.astype(mx.float32),
            bnd_loss.astype(mx.float32),
            quote_loss.astype(mx.float32),
            conf_loss.astype(mx.float32),
        )

    def forward_hidden_with_aux(
        self,
        input_ids: mx.array,
        capture_layers: tuple[int, ...] = (),
        operator_codes: mx.array | None = None,
        attention_window: int | None = None,
    ) -> tuple[mx.array, dict[int, mx.array], dict[str, mx.array | None]]:
        x = self.embed_inputs(input_ids)
        x0 = x
        attn_mask = self.attention_mask(x.shape[1], context_window=attention_window)
        if operator_codes is None:
            operator_codes = self.operator_codes_for_input(input_ids)
        skips: list[mx.array] = []
        captured: dict[int, mx.array] = {}
        layer_idx = 0
        seed_polarity_scores = self.seed_polarity_scores(operator_codes)
        polarity_scores = seed_polarity_scores if (self.logic_operator_mode == "not_only" and self.polarity_detector is None) else None
        detector_logits: mx.array | None = None
        detector_scores: mx.array | None = None
        hardmax_structural_aux: dict[str, mx.array] | None = None
        structural_budget: mx.array | None = None
        structural_attn_q_bias: mx.array | None = None
        structural_attn_tau: mx.array | None = None
        structural_reset_prior = self.structural_reset_prior_for_input(input_ids) if self.has_hardmax_structural_controller() else None

        for i in range(self.num_encoder_layers):
            layer_budget = self._hardmax_structural_layer_budget(structural_budget, layer_idx)
            x = self.block_for_step(
                i
            )(x, x0, attn_mask=attn_mask, residual_budget=layer_budget, attn_q_bias=structural_attn_q_bias, attn_tau=structural_attn_tau)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            x, hardmax_structural_aux_step = self.maybe_apply_hardmax_structural_controller(
                x,
                operator_codes,
                layer_idx,
                polarity_scores=polarity_scores,
                reset_prior=structural_reset_prior,
            )
            if hardmax_structural_aux_step is not None:
                hardmax_structural_aux = hardmax_structural_aux_step
                structural_budget = hardmax_structural_aux_step["budget"].astype(x.dtype)
                structural_attn_q_bias, structural_attn_tau = self.hardmax_structural_attention_condition_from_aux(
                    hardmax_structural_aux_step,
                    x.dtype,
                )
            skips.append(x)
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            layer_idx += 1
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            layer_budget = self._hardmax_structural_layer_budget(structural_budget, layer_idx)
            x = self.block_for_step(
                self.num_encoder_layers + i
            )(x, x0, attn_mask=attn_mask, residual_budget=layer_budget, attn_q_bias=structural_attn_q_bias, attn_tau=structural_attn_tau)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            x, hardmax_structural_aux_step = self.maybe_apply_hardmax_structural_controller(
                x,
                operator_codes,
                layer_idx,
                polarity_scores=polarity_scores,
                reset_prior=structural_reset_prior,
            )
            if hardmax_structural_aux_step is not None:
                hardmax_structural_aux = hardmax_structural_aux_step
                structural_budget = hardmax_structural_aux_step["budget"].astype(x.dtype)
                structural_attn_q_bias, structural_attn_tau = self.hardmax_structural_attention_condition_from_aux(
                    hardmax_structural_aux_step,
                    x.dtype,
                )
            if layer_idx in capture_layers:
                captured[layer_idx] = self.strip_registers(x)
            layer_idx += 1
        return self.final_norm(self.strip_registers(x)), captured, {
            "operator_codes": operator_codes,
            "seed_polarity_scores": seed_polarity_scores,
            "polarity_detector_logits": detector_logits,
            "polarity_detector_scores": detector_scores,
            "polarity_scores": polarity_scores,
            "hardmax_structural": hardmax_structural_aux,
        }

    def forward_hidden(
        self,
        input_ids: mx.array,
        capture_layers: tuple[int, ...] = (),
        operator_codes: mx.array | None = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        final_hidden, captured, _aux = self.forward_hidden_with_aux(
            input_ids,
            capture_layers=capture_layers,
            operator_codes=operator_codes,
        )
        return final_hidden, captured

    def __call__(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        return self.forward_hidden(input_ids, operator_codes=operator_codes)[0]

    def forward_logits(self, input_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        x = self(input_ids, operator_codes=operator_codes)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x)
        return self.softcap(logits_proj)

    def forward_hidden_to_layer(
        self,
        input_ids: mx.array,
        stop_layer_idx: int,
        operator_codes: mx.array | None = None,
        attention_window: int | None = None,
    ) -> mx.array:
        if stop_layer_idx < 0 or stop_layer_idx >= self.num_layers:
            raise ValueError(f"stop_layer_idx must be in [0, {self.num_layers - 1}], got {stop_layer_idx}")
        x = self.embed_inputs(input_ids)
        x0 = x
        attn_mask = self.attention_mask(x.shape[1], context_window=attention_window)
        if operator_codes is None:
            operator_codes = self.operator_codes_for_input(input_ids)
        skips: list[mx.array] = []
        layer_idx = 0
        seed_polarity_scores = self.seed_polarity_scores(operator_codes)
        polarity_scores = seed_polarity_scores if (self.logic_operator_mode == "not_only" and self.polarity_detector is None) else None
        structural_budget: mx.array | None = None
        structural_attn_q_bias: mx.array | None = None
        structural_attn_tau: mx.array | None = None
        structural_reset_prior = self.structural_reset_prior_for_input(input_ids) if self.has_hardmax_structural_controller() else None

        for i in range(self.num_encoder_layers):
            layer_budget = self._hardmax_structural_layer_budget(structural_budget, layer_idx)
            x = self.block_for_step(
                i
            )(x, x0, attn_mask=attn_mask, residual_budget=layer_budget, attn_q_bias=structural_attn_q_bias, attn_tau=structural_attn_tau)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                _detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            x, hardmax_structural_aux = self.maybe_apply_hardmax_structural_controller(
                x,
                operator_codes,
                layer_idx,
                polarity_scores=polarity_scores,
                reset_prior=structural_reset_prior,
            )
            if hardmax_structural_aux is not None:
                structural_budget = hardmax_structural_aux["budget"].astype(x.dtype)
                structural_attn_q_bias, structural_attn_tau = self.hardmax_structural_attention_condition_from_aux(
                    hardmax_structural_aux,
                    x.dtype,
                )
            if layer_idx == stop_layer_idx:
                return self.strip_registers(x)
            skips.append(x)
            layer_idx += 1
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            layer_budget = self._hardmax_structural_layer_budget(structural_budget, layer_idx)
            x = self.block_for_step(
                self.num_encoder_layers + i
            )(x, x0, attn_mask=attn_mask, residual_budget=layer_budget, attn_q_bias=structural_attn_q_bias, attn_tau=structural_attn_tau)
            if self.polarity_detector is not None and layer_idx == self.polarity_detector_layer_index:
                _detector_logits, detector_scores = self.polarity_detector_logits_and_scores(x)
                polarity_scores = self.blend_polarity_scores(seed_polarity_scores, detector_scores)
            x = self.maybe_apply_logic_sidecar(x, operator_codes, layer_idx, polarity_scores=polarity_scores)
            x, hardmax_structural_aux = self.maybe_apply_hardmax_structural_controller(
                x,
                operator_codes,
                layer_idx,
                polarity_scores=polarity_scores,
                reset_prior=structural_reset_prior,
            )
            if hardmax_structural_aux is not None:
                structural_budget = hardmax_structural_aux["budget"].astype(x.dtype)
                structural_attn_q_bias, structural_attn_tau = self.hardmax_structural_attention_condition_from_aux(
                    hardmax_structural_aux,
                    x.dtype,
                )
            if layer_idx == stop_layer_idx:
                return self.strip_registers(x)
            layer_idx += 1
        raise RuntimeError(f"Failed to stop at layer {stop_layer_idx}")

    def _early_exit_logits_from_hidden(
        self,
        hidden: mx.array,
        head_idx: int,
    ) -> mx.array:
        x = self.final_norm(hidden.astype(COMPUTE_DTYPE)).reshape(-1, hidden.shape[-1])
        head = self.early_exit_heads[head_idx]
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            return self.softcap(head(x))
        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            parts.append(self.softcap(head(x[s:e])))
        return mx.concatenate(parts, axis=0)

    def _condition_early_exit_hidden(
        self,
        hidden: mx.array,
        condition_ids: tuple[mx.array, ...],
        head_idx: int,
    ) -> mx.array:
        if not self.has_cascaded_early_exit() or not condition_ids:
            return hidden
        cond_sum: mx.array | None = None
        for cond_ids in condition_ids:
            if int(cond_ids.size) == 0:
                continue
            cond_emb = mx.stop_gradient(self.tok_emb(cond_ids.astype(mx.int32))).astype(COMPUTE_DTYPE)
            cond_sum = cond_emb if cond_sum is None else (cond_sum + cond_emb)
        if cond_sum is None:
            return hidden
        proj = self.early_exit_condition_projs[head_idx]
        cond_delta = proj(cond_sum.reshape(-1, cond_sum.shape[-1])).reshape(cond_sum.shape)
        return hidden + cond_delta.astype(hidden.dtype)

    def _early_exit_nll_for_horizon(
        self,
        captured_hidden: mx.array,
        target_ids: mx.array,
        horizon: int,
        head_idx: int,
        *,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        shift = horizon_shift(horizon)
        if shift >= int(target_ids.shape[1]):
            return mx.zeros((target_ids.shape[0], 0), dtype=mx.float32)
        hidden_view = captured_hidden[:, : int(target_ids.shape[1]) - shift, :]
        target_view = target_ids[:, shift:]
        if shift > 0 and self.has_cascaded_early_exit():
            condition_ids = tuple(
                target_ids[:, prev_shift : prev_shift + int(target_view.shape[1])]
                for prev_shift in range(shift)
            )
            hidden_view = self._condition_early_exit_hidden(hidden_view, condition_ids, head_idx)
        weight_view = token_weights[:, shift:] if token_weights is not None else None
        logits = self._early_exit_logits_from_hidden(hidden_view, head_idx).reshape(
            target_view.shape[0],
            target_view.shape[1],
            -1,
        )
        return token_cross_entropy_with_focal(
            logits.reshape(-1, logits.shape[-1]),
            target_view.reshape(-1),
            focal_loss_weight=0.0,
            focal_gamma=2.0,
            focal_max_multiplier=0.0,
            token_weights=weight_view.reshape(-1) if weight_view is not None else None,
            reduction="none",
        ).reshape(target_view.shape)

    def early_exit_aux_loss(
        self,
        captured_hidden: mx.array,
        target_ids: mx.array,
        *,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        losses: list[mx.array] = []
        for head_idx, horizon in enumerate(self._early_exit_horizons):
            if horizon_shift(horizon) >= int(target_ids.shape[1]):
                continue
            horizon_nll = self._early_exit_nll_for_horizon(
                captured_hidden,
                target_ids,
                horizon,
                head_idx,
                token_weights=token_weights,
            )
            if int(horizon_nll.size) > 0:
                losses.append(mx.mean(horizon_nll.astype(mx.float32)))
        if not losses:
            return mx.array(0.0, dtype=mx.float32)
        return mx.mean(mx.stack(losses).astype(mx.float32))

    def _prosody_label_luts(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if (
            self._prosody_token_class_lut is None
            or self._prosody_boundary_strength_lut is None
            or self._prosody_punctuation_role_lut is None
            or self._prosody_quote_like_lut is None
        ):
            raise ValueError("Prosody label LUTs are not initialized")
        return (
            self._prosody_token_class_lut,
            self._prosody_boundary_strength_lut,
            self._prosody_punctuation_role_lut,
            self._prosody_quote_like_lut,
        )

    def _prosody_aux_logits_from_hidden(
        self,
        captured_hidden: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = self.final_norm(captured_hidden.astype(COMPUTE_DTYPE)).reshape(-1, captured_hidden.shape[-1])
        token_class_logits = self.prosody_token_class_head(hidden)
        boundary_logits = self.prosody_boundary_head(hidden)
        punctuation_logits = self.prosody_punctuation_head(hidden)
        quote_logits = self.prosody_quote_head(hidden)
        return (
            token_class_logits.reshape(captured_hidden.shape[0], captured_hidden.shape[1], -1),
            boundary_logits.reshape(captured_hidden.shape[0], captured_hidden.shape[1], -1),
            punctuation_logits.reshape(captured_hidden.shape[0], captured_hidden.shape[1], -1),
            quote_logits.reshape(captured_hidden.shape[0], captured_hidden.shape[1], -1),
        )

    def prosody_aux_loss_terms(
        self,
        captured_hidden: mx.array,
        target_ids: mx.array,
        *,
        token_weights: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        if not self.has_prosody_aux():
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, zero, zero, zero
        token_class_lut, boundary_lut, punctuation_lut, quote_lut = self._prosody_label_luts()
        token_class_logits, boundary_logits, punctuation_logits, quote_logits = self._prosody_aux_logits_from_hidden(captured_hidden)
        target_token_class = token_class_lut[target_ids.astype(mx.int32)]
        target_boundary = boundary_lut[target_ids.astype(mx.int32)]
        target_punctuation = punctuation_lut[target_ids.astype(mx.int32)]
        target_quote = quote_lut[target_ids.astype(mx.int32)]
        flat_weights = token_weights.reshape(-1) if token_weights is not None else None
        token_class_loss = token_cross_entropy_with_focal(
            token_class_logits.reshape(-1, token_class_logits.shape[-1]),
            target_token_class.reshape(-1),
            token_weights=flat_weights,
        )
        boundary_loss = token_cross_entropy_with_focal(
            boundary_logits.reshape(-1, boundary_logits.shape[-1]),
            target_boundary.reshape(-1),
            token_weights=flat_weights,
        )
        punctuation_mask = (target_punctuation.reshape(-1) != PUNCTUATION_ROLE_TO_ID["none"]).astype(mx.float32)
        punctuation_weights = punctuation_mask if flat_weights is None else flat_weights.astype(mx.float32) * punctuation_mask
        punctuation_loss = token_cross_entropy_with_focal(
            punctuation_logits.reshape(-1, punctuation_logits.shape[-1]),
            target_punctuation.reshape(-1),
            token_weights=punctuation_weights,
        )
        quote_loss = token_cross_entropy_with_focal(
            quote_logits.reshape(-1, quote_logits.shape[-1]),
            target_quote.reshape(-1),
            token_weights=flat_weights,
        )
        total = (
            self._prosody_aux_token_class_weight * token_class_loss
            + self._prosody_aux_boundary_weight * boundary_loss
            + self._prosody_aux_punctuation_weight * punctuation_loss
            + self._prosody_aux_quote_weight * quote_loss
        )
        return (
            total.astype(mx.float32),
            token_class_loss.astype(mx.float32),
            boundary_loss.astype(mx.float32),
            punctuation_loss.astype(mx.float32),
            quote_loss.astype(mx.float32),
        )

    def early_exit_next_token_logits(
        self,
        prefix_ids: mx.array,
        operator_codes: mx.array | None = None,
    ) -> mx.array:
        head_idx = self.primary_early_exit_head_idx()
        captured = self.forward_hidden_to_layer(
            prefix_ids[None, :].astype(mx.int32),
            self._early_exit_layer_index,
            operator_codes=operator_codes,
        )
        return self._early_exit_logits_from_hidden(captured[:, -1:, :], head_idx)[0]

    def early_exit_draft_tokens(
        self,
        prefix_ids: mx.array,
        *,
        max_tokens: int,
        operator_codes: mx.array | None = None,
    ) -> mx.array:
        limit = min(max(int(max_tokens), 0), self._early_exit_branch_max_draft_tokens)
        if limit <= 0 or not self.has_early_exit_branch_drafter():
            return mx.zeros((0,), dtype=mx.int32)
        captured = self.forward_hidden_to_layer(
            prefix_ids[None, :].astype(mx.int32),
            self._early_exit_layer_index,
            operator_codes=operator_codes,
        )
        candidate_tokens: dict[int, int] = {}
        candidate_confidences: dict[int, float] = {}
        draft_condition_ids: list[mx.array] = []
        for horizon, head_idx in self.ordered_early_exit_draft_heads():
            if horizon > limit:
                continue
            conditioned_hidden = self._condition_early_exit_hidden(
                captured[:, -1:, :],
                tuple(draft_condition_ids),
                head_idx,
            )
            logits = self._early_exit_logits_from_hidden(conditioned_hidden, head_idx)[0]
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            next_token = int(mx.argmax(logits).item())
            candidate_tokens[horizon] = next_token
            candidate_confidences[horizon] = float(mx.max(probs).item())
            draft_condition_ids.append(mx.array([[next_token]], dtype=mx.int32))
        accepted = select_contiguous_draft_horizons(
            candidate_tokens.keys(),
            (candidate_confidences[h] for h in candidate_tokens.keys()),
            threshold=self._early_exit_branch_conf_threshold,
            max_tokens=limit,
        )
        if not accepted:
            return mx.zeros((0,), dtype=mx.int32)
        return mx.array([candidate_tokens[horizon] for horizon in accepted], dtype=mx.int32)

    def token_ce_from_hidden(
        self,
        hidden: mx.array,
        target_ids: mx.array,
        *,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        per_token_nll = self.token_nll_from_hidden(
            hidden,
            target_ids,
            focal_loss_weight=focal_loss_weight,
            focal_gamma=focal_gamma,
            focal_max_multiplier=focal_max_multiplier,
            token_weights=token_weights,
        )
        return mx.mean(per_token_nll)

    def distill_kl_from_hidden(
        self,
        hidden: mx.array,
        teacher_logits: mx.array,
        *,
        temperature: float = 1.0,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        x = hidden.reshape(-1, self.tok_emb.weight.shape[1])
        teacher_flat = mx.stop_gradient(teacher_logits.reshape(-1, teacher_logits.shape[-1]).astype(mx.float32))
        flat_token_weights = token_weights.reshape(-1).astype(mx.float32) if token_weights is not None else None
        temp = max(float(temperature), 1e-6)
        temp_arr = mx.array(temp, dtype=mx.float32)
        scale = temp * temp

        def chunk_kl(student_logits_flat: mx.array, teacher_logits_flat: mx.array, weights_flat: mx.array | None) -> mx.array:
            student_scaled = student_logits_flat.astype(mx.float32) / temp_arr
            teacher_scaled = teacher_logits_flat.astype(mx.float32) / temp_arr
            student_log_probs = student_scaled - mx.logsumexp(student_scaled, axis=-1, keepdims=True)
            teacher_log_probs = teacher_scaled - mx.logsumexp(teacher_scaled, axis=-1, keepdims=True)
            teacher_probs = mx.exp(teacher_log_probs)
            per_token_kl = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1) * scale
            if weights_flat is None:
                return mx.mean(per_token_kl.astype(mx.float32))
            weight_sum = mx.sum(weights_flat.astype(mx.float32))
            return mx.sum(per_token_kl.astype(mx.float32) * weights_flat.astype(mx.float32)) / mx.maximum(
                weight_sum,
                mx.array(1e-6, dtype=mx.float32),
            )

        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x)
            logits = self.softcap(logits_proj)
            return chunk_kl(logits, teacher_flat, flat_token_weights)

        weighted_sum = mx.array(0.0, dtype=mx.float32)
        total_count = 0
        total_weight = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x[s:e])
            logits = self.softcap(logits_proj)
            chunk_weights = flat_token_weights[s:e] if flat_token_weights is not None else None
            chunk_value = chunk_kl(logits, teacher_flat[s:e], chunk_weights)
            if chunk_weights is None:
                chunk_count = e - s
                weighted_sum = weighted_sum + chunk_value.astype(mx.float32) * chunk_count
                total_count += chunk_count
            else:
                chunk_weight_sum = mx.sum(chunk_weights.astype(mx.float32))
                weighted_sum = weighted_sum + chunk_value.astype(mx.float32) * chunk_weight_sum
                total_weight = total_weight + chunk_weight_sum
        if flat_token_weights is None:
            return weighted_sum / max(total_count, 1)
        return weighted_sum / mx.maximum(total_weight, mx.array(1e-6, dtype=mx.float32))

    def hidden_distill_from_hidden(
        self,
        hidden: mx.array,
        teacher_hidden: mx.array,
        *,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        student = hidden.astype(mx.float32)
        teacher = mx.stop_gradient(teacher_hidden.astype(mx.float32))
        if student.shape != teacher.shape:
            raise ValueError(
                f"teacher_hidden shape mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )
        student_norm = student / mx.maximum(
            mx.linalg.norm(student, axis=-1, keepdims=True),
            mx.array(1e-6, dtype=mx.float32),
        )
        teacher_norm = teacher / mx.maximum(
            mx.linalg.norm(teacher, axis=-1, keepdims=True),
            mx.array(1e-6, dtype=mx.float32),
        )
        per_token = 1.0 - mx.sum(student_norm * teacher_norm, axis=-1).astype(mx.float32)
        if token_weights is None:
            return mx.mean(per_token)
        weights = token_weights.astype(mx.float32)
        return mx.sum(per_token * weights) / mx.maximum(mx.sum(weights), mx.array(1e-6, dtype=mx.float32))

    def token_nll_from_hidden(
        self,
        hidden: mx.array,
        target_ids: mx.array,
        *,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        x = hidden.reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        flat_token_weights = token_weights.reshape(-1) if token_weights is not None else None
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x)
            logits = self.softcap(logits_proj)
            return token_cross_entropy_with_focal(
                logits,
                y,
                focal_loss_weight=focal_loss_weight,
                focal_gamma=focal_gamma,
                focal_max_multiplier=focal_max_multiplier,
                token_weights=flat_token_weights,
                reduction="none",
            ).reshape(target_ids.shape)

        parts: list[mx.array] = []
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x[s:e])
            logits = self.softcap(logits_proj)
            parts.append(token_cross_entropy_with_focal(
                logits,
                y[s:e],
                focal_loss_weight=focal_loss_weight,
                focal_gamma=focal_gamma,
                focal_max_multiplier=focal_max_multiplier,
                token_weights=flat_token_weights[s:e] if flat_token_weights is not None else None,
                reduction="none",
            ))
        return mx.concatenate(parts, axis=0).reshape(target_ids.shape)

    def continuation_hidden_and_token_nll(
        self,
        prefix_ids: mx.array,
        continuation_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if int(continuation_ids.shape[0]) <= 0:
            hidden_dim = int(self.tok_emb.weight.shape[1])
            return mx.zeros((0, hidden_dim), dtype=mx.float32), mx.zeros((0,), dtype=mx.float32)
        full_tokens = mx.concatenate([prefix_ids.astype(mx.int32), continuation_ids.astype(mx.int32)], axis=0)
        scoring_input = full_tokens[:-1][None, :]
        scoring_target = full_tokens[1:][None, :]
        hidden, _captured, _aux = self.forward_hidden_with_aux(scoring_input)
        per_token_nll = self.token_nll_from_hidden(hidden, scoring_target)[0].astype(mx.float32)
        start = max(int(prefix_ids.shape[0]) - 1, 0)
        end = start + int(continuation_ids.shape[0])
        return hidden[0, start:end], per_token_nll[start:end]

    def continuation_mean_nll(
        self,
        prefix_ids: mx.array,
        continuation_ids: mx.array,
    ) -> mx.array:
        _hidden, per_token_nll = self.continuation_hidden_and_token_nll(prefix_ids, continuation_ids)
        if int(per_token_nll.shape[0]) <= 0:
            return mx.array(0.0, dtype=mx.float32)
        return mx.mean(per_token_nll.astype(mx.float32))

    def generate_committed_continuation(
        self,
        prefix_ids: mx.array,
        first_token_id: int,
        length: int,
    ) -> mx.array:
        if length <= 0:
            return mx.zeros((0,), dtype=mx.int32)
        next_token = mx.array([int(first_token_id)], dtype=mx.int32)
        generated: list[mx.array] = [next_token]
        current = mx.concatenate([prefix_ids.astype(mx.int32), next_token], axis=0)
        remaining = length - 1
        while remaining > 0:
            drafted = self.early_exit_draft_tokens(current, max_tokens=remaining)
            if int(drafted.shape[0]) > 0:
                for token_id in np.asarray(drafted, dtype=np.int32):
                    token = mx.array([int(token_id)], dtype=mx.int32)
                    generated.append(token)
                    current = mx.concatenate([current, token], axis=0)
                remaining -= int(drafted.shape[0])
                continue
            logits = self.forward_logits(current[None, :])[0, -1]
            next_token = mx.array([int(mx.argmax(logits).item())], dtype=mx.int32)
            generated.append(next_token)
            current = mx.concatenate([current, next_token], axis=0)
            remaining -= 1
        return mx.concatenate(generated, axis=0)

    def structural_branching_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        branch_plans: list[list[StructuralBranchPoint]] | None,
        *,
        config: StructuralBranchingConfig,
    ) -> tuple[mx.array, mx.array]:
        branch_length = int(config.branch_length)
        margin = float(config.margin)
        if not branch_plans or branch_length <= 0:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero
        ranking_total = mx.array(0.0, dtype=mx.float32)
        state_total = mx.array(0.0, dtype=mx.float32)
        count = 0
        margin_arr = mx.array(float(margin), dtype=mx.float32)
        target_max_cos_arr = mx.array(float(config.state_target_max_cosine), dtype=mx.float32)
        for row_idx, row_plans in enumerate(branch_plans):
            if not row_plans:
                continue
            row_input = input_ids[row_idx]
            row_target = target_ids[row_idx]
            row_limit = int(row_target.shape[0])
            for plan in row_plans:
                pos = int(plan.pos)
                available = row_limit - pos
                if available <= 0:
                    continue
                current_branch_len = min(branch_length, available)
                prefix = row_input[: pos + 1]
                real_continuation = row_target[pos : pos + current_branch_len]
                wrong_continuation = self.generate_committed_continuation(
                    prefix,
                    int(plan.predicted_token),
                    current_branch_len,
                )
                real_hidden, real_token_nll = self.continuation_hidden_and_token_nll(prefix, real_continuation)
                wrong_hidden, wrong_token_nll = self.continuation_hidden_and_token_nll(prefix, wrong_continuation)
                effective_branch_len = int(current_branch_len)
                if config.adaptive_depth_enabled and current_branch_len > 0:
                    real_hidden_np = np.asarray(mx.stop_gradient(real_hidden.astype(mx.float32)))
                    wrong_hidden_np = np.asarray(mx.stop_gradient(wrong_hidden.astype(mx.float32)))
                    if real_hidden_np.size > 0 and wrong_hidden_np.size > 0:
                        real_hidden_norm = real_hidden_np / np.maximum(
                            np.linalg.norm(real_hidden_np, axis=-1, keepdims=True),
                            1e-8,
                        )
                        wrong_hidden_norm = wrong_hidden_np / np.maximum(
                            np.linalg.norm(wrong_hidden_np, axis=-1, keepdims=True),
                            1e-8,
                        )
                        divergence = 1.0 - np.sum(real_hidden_norm * wrong_hidden_norm, axis=-1, dtype=np.float32)
                        effective_branch_len = adaptive_branch_length_from_divergence(
                            divergence,
                            min_depth=min(int(config.adaptive_min_depth), current_branch_len),
                            plateau_tol=float(config.adaptive_plateau_tol),
                            converged_divergence=float(config.adaptive_converged_divergence),
                        )
                effective_branch_len = max(1, min(int(effective_branch_len), current_branch_len))
                real_nll = mx.mean(real_token_nll[:effective_branch_len].astype(mx.float32))
                wrong_nll = mx.mean(wrong_token_nll[:effective_branch_len].astype(mx.float32))
                ranking_total = ranking_total + mx.maximum(
                    margin_arr + real_nll - wrong_nll,
                    mx.array(0.0, dtype=mx.float32),
                )
                if float(config.state_divergence_weight) > 0.0:
                    real_view = real_hidden[:effective_branch_len].astype(mx.float32)
                    wrong_view = wrong_hidden[:effective_branch_len].astype(mx.float32)
                    real_norm = real_view / mx.maximum(
                        mx.linalg.norm(real_view, axis=-1, keepdims=True),
                        mx.array(1e-8, dtype=mx.float32),
                    )
                    wrong_norm = wrong_view / mx.maximum(
                        mx.linalg.norm(wrong_view, axis=-1, keepdims=True),
                        mx.array(1e-8, dtype=mx.float32),
                    )
                    state_cos = mx.sum(real_norm * wrong_norm, axis=-1).astype(mx.float32)
                    state_total = state_total + mx.mean(
                        mx.maximum(
                            state_cos - target_max_cos_arr,
                            mx.array(0.0, dtype=mx.float32),
                        )
                    )
                count += 1
        if count <= 0:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero
        return ranking_total / count, state_total / count

    def context_delta_token_weights(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        long_hidden: mx.array,
        operator_codes: mx.array | None,
        config: ContextDeltaWeightingConfig | None,
    ) -> mx.array | None:
        if config is None or not config.enabled:
            return None
        prefix_tokens = self.num_registers if self.register_layout == "prefix" else 0
        available_context = int(input_ids.shape[1]) - prefix_tokens
        if config.short_context_len <= 0 or config.short_context_len >= available_context:
            return None

        long_nll = self.token_nll_from_hidden(long_hidden, target_ids)
        short_hidden, _captured, _aux = self.forward_hidden_with_aux(
            input_ids,
            operator_codes=operator_codes,
            attention_window=config.short_context_len,
        )
        short_nll = self.token_nll_from_hidden(short_hidden, target_ids)
        scores = short_nll - long_nll
        scores = mx.abs(scores) if config.use_absolute_delta else mx.maximum(scores, mx.array(0.0, dtype=mx.float32))
        if config.score_power != 1.0:
            scores = mx.power(scores, config.score_power)
        weights = 1.0 + scores.astype(mx.float32)
        if 0.0 < config.topk_fraction < 1.0:
            flat_scores = scores.reshape(-1)
            keep = max(1, int(math.ceil(float(flat_scores.shape[0]) * config.topk_fraction)))
            sorted_scores = mx.sort(flat_scores)
            threshold = sorted_scores[-keep]
            weights = mx.where(scores >= threshold, weights, mx.zeros_like(weights))
        if config.max_multiplier > 0.0:
            weights = mx.minimum(weights, mx.array(config.max_multiplier, dtype=mx.float32))
        return mx.stop_gradient(weights.astype(mx.float32))

    def polarity_loss_terms_from_aux(
        self,
        aux: dict[str, mx.array | None],
    ) -> tuple[mx.array, mx.array, mx.array]:
        logits = aux["polarity_detector_logits"]
        scores = aux["polarity_detector_scores"]
        seed_scores = aux["seed_polarity_scores"]
        zero = mx.array(0.0, dtype=mx.float32)
        if logits is None or scores is None or seed_scores is None:
            return zero, zero, zero
        logits = strip_register_positions(
            logits,
            self.num_registers,
            layout=self.register_layout,
            register_stride=self.register_stride,
        )
        scores = strip_register_positions(
            scores,
            self.num_registers,
            layout=self.register_layout,
            register_stride=self.register_stride,
        )
        seed_scores = strip_register_positions(
            seed_scores.astype(mx.float32),
            self.num_registers,
            layout=self.register_layout,
            register_stride=self.register_stride,
        )
        if isinstance(logits, np.ndarray) or isinstance(scores, np.ndarray) or isinstance(seed_scores, np.ndarray):
            raise TypeError("polarity_loss_terms_from_aux expects mlx arrays")
        if logits.shape[1] <= 0:
            return zero, zero, zero
        seed_loss = binary_cross_entropy_with_logits(logits.astype(mx.float32), seed_scores)
        sparse_loss = mx.mean(scores.astype(mx.float32))
        if scores.shape[1] <= 1:
            smooth_loss = zero
        else:
            smooth_loss = mx.mean(mx.abs(scores[:, 1:].astype(mx.float32) - scores[:, :-1].astype(mx.float32)))
        return seed_loss, sparse_loss, smooth_loss

    def ce_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
        *,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
        context_delta_config: ContextDeltaWeightingConfig | None = None,
    ) -> mx.array:
        final_hidden, _captured, _aux = self.forward_hidden_with_aux(input_ids, operator_codes=operator_codes)
        combined_token_weights = merge_token_weights(
            token_weights,
            self.context_delta_token_weights(
                input_ids,
                target_ids,
                final_hidden,
                operator_codes,
                context_delta_config,
            ),
        )
        return self.token_ce_from_hidden(
            final_hidden,
            target_ids,
            focal_loss_weight=focal_loss_weight,
            focal_gamma=focal_gamma,
            focal_max_multiplier=focal_max_multiplier,
            token_weights=combined_token_weights,
        )

    def loss_terms(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        operator_codes: mx.array | None = None,
        polarity_seed_weight: float | None = None,
        polarity_sparse_weight: float | None = None,
        polarity_smooth_weight: float | None = None,
        focal_loss_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_max_multiplier: float = 4.0,
        token_weights: mx.array | None = None,
        context_delta_config: ContextDeltaWeightingConfig | None = None,
        teacher_logits: mx.array | None = None,
        teacher_hidden: mx.array | None = None,
        ema_teacher_distill_weight: float = 0.0,
        teacher_hidden_distill_weight: float = 0.0,
        ema_teacher_temperature: float = 1.0,
        early_exit_aux_weight_override: float | None = None,
        structural_branching_cfg: StructuralBranchingConfig | None = None,
        branch_plans: list[list[StructuralBranchPoint]] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        effective_early_exit_weight = (
            self._early_exit_aux_weight
            if early_exit_aux_weight_override is None
            else float(early_exit_aux_weight_override)
        )
        capture_layers = tuple(sorted({
            self._early_exit_layer_index
            for _ in [0]
            if bool(self.early_exit_heads) and effective_early_exit_weight > 0.0
        } | {
            self._prosody_aux_layer_index
            for _ in [0]
            if self.has_prosody_aux()
        }))
        final_hidden, captured, aux = self.forward_hidden_with_aux(
            input_ids,
            capture_layers=capture_layers,
            operator_codes=operator_codes,
        )
        combined_token_weights = merge_token_weights(
            token_weights,
            self.context_delta_token_weights(
                input_ids,
                target_ids,
                final_hidden,
                operator_codes,
                context_delta_config,
            ),
        )
        ce_loss = self.token_ce_from_hidden(
            final_hidden,
            target_ids,
            focal_loss_weight=focal_loss_weight,
            focal_gamma=focal_gamma,
            focal_max_multiplier=focal_max_multiplier,
            token_weights=combined_token_weights,
        )
        seed_weight = self.polarity_seed_weight if polarity_seed_weight is None else polarity_seed_weight
        sparse_weight = self.polarity_sparse_weight if polarity_sparse_weight is None else polarity_sparse_weight
        smooth_weight = self.polarity_smooth_weight if polarity_smooth_weight is None else polarity_smooth_weight
        seed_loss, sparse_loss, smooth_loss = self.polarity_loss_terms_from_aux(aux)
        distill_loss = mx.array(0.0, dtype=mx.float32)
        hidden_distill_loss = mx.array(0.0, dtype=mx.float32)
        early_exit_loss = mx.array(0.0, dtype=mx.float32)
        prosody_aux_loss = mx.array(0.0, dtype=mx.float32)
        prosody_token_class_loss = mx.array(0.0, dtype=mx.float32)
        prosody_boundary_loss = mx.array(0.0, dtype=mx.float32)
        prosody_punctuation_loss = mx.array(0.0, dtype=mx.float32)
        prosody_quote_loss = mx.array(0.0, dtype=mx.float32)
        structural_branch_loss = mx.array(0.0, dtype=mx.float32)
        structural_branch_state_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_balance_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_diversity_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_confidence_mean = mx.array(0.0, dtype=mx.float32)
        hardmax_budget_mean = mx.array(0.0, dtype=mx.float32)
        hardmax_entropy = mx.array(0.0, dtype=mx.float32)
        hardmax_pred_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_operator_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_token_class_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_boundary_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_quote_loss = mx.array(0.0, dtype=mx.float32)
        hardmax_confidence_loss = mx.array(0.0, dtype=mx.float32)
        if bool(self.early_exit_heads) and effective_early_exit_weight > 0.0:
            early_exit_hidden = captured.get(self._early_exit_layer_index)
            if early_exit_hidden is not None:
                early_exit_loss = self.early_exit_aux_loss(
                    early_exit_hidden,
                    target_ids,
                    token_weights=combined_token_weights,
                )
        if self.has_prosody_aux():
            prosody_hidden = captured.get(self._prosody_aux_layer_index)
            if prosody_hidden is not None:
                (
                    prosody_aux_loss,
                    prosody_token_class_loss,
                    prosody_boundary_loss,
                    prosody_punctuation_loss,
                    prosody_quote_loss,
                ) = self.prosody_aux_loss_terms(
                    prosody_hidden,
                    target_ids,
                    token_weights=combined_token_weights,
                )
        if teacher_logits is not None and ema_teacher_distill_weight > 0.0:
            distill_loss = self.distill_kl_from_hidden(
                final_hidden,
                teacher_logits,
                temperature=ema_teacher_temperature,
                token_weights=combined_token_weights,
            )
        if teacher_hidden is not None and teacher_hidden_distill_weight > 0.0:
            hidden_distill_loss = self.hidden_distill_from_hidden(
                final_hidden,
                teacher_hidden,
                token_weights=combined_token_weights,
            )
        if self.has_hardmax_structural_controller():
            (
                hardmax_balance_loss,
                hardmax_diversity_loss,
                hardmax_confidence_mean,
                hardmax_budget_mean,
                hardmax_entropy,
            ) = self.hardmax_structural_loss_terms_from_aux(aux)
            if self._hardmax_struct_predict_weight > 0.0 or self._hardmax_struct_confidence_weight > 0.0:
                (
                    hardmax_pred_loss,
                    hardmax_operator_loss,
                    hardmax_token_class_loss,
                    hardmax_boundary_loss,
                    hardmax_quote_loss,
                    hardmax_confidence_loss,
                ) = self.hardmax_structural_prediction_loss_terms(
                    aux,
                    target_ids,
                    token_weights=combined_token_weights,
                )
        if (
            structural_branching_cfg is not None
            and structural_branching_cfg.enabled
            and structural_branching_cfg.weight > 0.0
            and branch_plans is not None
        ):
            structural_branch_loss, structural_branch_state_loss = self.structural_branching_loss(
                input_ids,
                target_ids,
                branch_plans,
                config=structural_branching_cfg,
            )
        total = (
            ce_loss
            + effective_early_exit_weight * early_exit_loss
            + seed_weight * seed_loss
            + sparse_weight * sparse_loss
            + smooth_weight * smooth_loss
            + self._prosody_aux_weight * prosody_aux_loss
            + self._hardmax_struct_usage_balance_weight * hardmax_balance_loss
            + self._hardmax_struct_diversity_weight * hardmax_diversity_loss
            + self._hardmax_struct_predict_weight * hardmax_pred_loss
            + self._hardmax_struct_confidence_weight * hardmax_confidence_loss
            + ema_teacher_distill_weight * distill_loss
            + teacher_hidden_distill_weight * hidden_distill_loss
            + (
                structural_branching_cfg.weight * structural_branch_loss
                + structural_branching_cfg.state_divergence_weight * structural_branch_state_loss
                if structural_branching_cfg is not None
                else 0.0
            )
        )
        return (
            total,
            ce_loss,
            seed_loss,
            sparse_loss,
            smooth_loss,
            early_exit_loss,
            prosody_aux_loss,
            prosody_token_class_loss,
            prosody_boundary_loss,
            prosody_punctuation_loss,
            prosody_quote_loss,
            distill_loss,
            hidden_distill_loss,
            structural_branch_loss,
            structural_branch_state_loss,
            hardmax_balance_loss,
            hardmax_diversity_loss,
            hardmax_confidence_mean,
            hardmax_budget_mean,
            hardmax_entropy,
            hardmax_pred_loss,
            hardmax_operator_loss,
            hardmax_token_class_loss,
            hardmax_boundary_loss,
            hardmax_quote_loss,
            hardmax_confidence_loss,
        )

    def loss(self, input_ids: mx.array, target_ids: mx.array, operator_codes: mx.array | None = None) -> mx.array:
        return self.ce_loss(input_ids, target_ids, operator_codes=operator_codes)

class Muon:
    def __init__(
        self,
        keys: list[str],
        params: dict[str, mx.array],
        args: Hyperparameters,
        *,
        momentum_target: float,
        momentum_warmup_start: float,
    ):
        self.keys = keys
        self.args = args
        self.momentum_target = momentum_target
        self.momentum_warmup_start = momentum_warmup_start
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(
        self,
        params: dict[str, mx.array],
        grads: dict[str, mx.array],
        step: int,
        lr_mul: float,
    ) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.momentum_warmup_start + t * self.momentum_target
        else:
            momentum = self.momentum_target
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        alignment = mx.array(0.0, dtype=mx.float32)
        grad_sq = mx.array(0.0, dtype=mx.float32)
        buf_sq = mx.array(0.0, dtype=mx.float32)
        for k in self.keys:
            p = params[k]
            g = grads[k]
            prev_buf = self.buffers[k].astype(mx.float32)
            g32 = g.astype(mx.float32)
            alignment = alignment + mx.sum(g32 * prev_buf)
            grad_sq = grad_sq + mx.sum(g32 * g32)
            buf_sq = buf_sq + mx.sum(prev_buf * prev_buf)
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out, alignment, grad_sq, buf_sq

class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.matrix_optimizer_name = str(args.matrix_optimizer).lower()
        self.turbo_matrix_optimizer_name = str(args.turbo_matrix_optimizer).lower()
        if self.matrix_optimizer_name not in {"muon", "apollo_mini"}:
            raise ValueError(f"Unsupported MATRIX_OPTIMIZER: {args.matrix_optimizer}")
        if self.turbo_matrix_optimizer_name not in {"muon", "apollo_mini"}:
            raise ValueError(f"Unsupported TURBO_MATRIX_OPTIMIZER: {args.turbo_matrix_optimizer}")
        self.embed_keys = ["tok_emb.weight"]
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and p.ndim == 2 and infer_turbo_mode(k) == "none" and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.turbo_matrix_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and p.ndim == 2 and infer_turbo_mode(k) != "none" and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and (k == "skip_weights" or p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        ]

        self.muon = None
        self.apollo = None
        apollo_seed_base = 0
        if self.matrix_keys:
            if self.matrix_optimizer_name == "muon":
                self.muon = Muon(
                    self.matrix_keys,
                    params,
                    args,
                    momentum_target=args.muon_momentum,
                    momentum_warmup_start=args.muon_momentum_warmup_start,
                )
            else:
                self.apollo = ApolloMatrixOptimizer(
                    self.matrix_keys,
                    params,
                    lr=args.matrix_lr,
                    beta1=args.apollo_beta1,
                    beta2=args.apollo_beta2,
                    eps=args.apollo_eps,
                    rank=args.apollo_rank,
                    scale=args.apollo_scale,
                    scale_type=args.apollo_scale_type,
                    update_proj_gap=args.apollo_proj_gap,
                    proj_type=args.apollo_proj_type,
                    seed_base=apollo_seed_base,
                    scale_front=args.apollo_scale_front,
                    disable_nl=args.apollo_disable_nl,
                )
                apollo_seed_base += len(self.matrix_keys)
        self.muon_turbo = None
        self.apollo_turbo = None
        if self.turbo_matrix_keys:
            if self.turbo_matrix_optimizer_name == "muon":
                self.muon_turbo = Muon(
                    self.turbo_matrix_keys,
                    params,
                    args,
                    momentum_target=args.turbo_qat_muon_momentum if args.turbo_qat else args.muon_momentum,
                    momentum_warmup_start=args.turbo_qat_muon_momentum_warmup_start if args.turbo_qat else args.muon_momentum_warmup_start,
                )
            else:
                self.apollo_turbo = ApolloMatrixOptimizer(
                    self.turbo_matrix_keys,
                    params,
                    lr=args.matrix_lr,
                    beta1=args.apollo_beta1,
                    beta2=args.apollo_beta2,
                    eps=args.apollo_eps,
                    rank=args.apollo_rank,
                    scale=args.apollo_scale,
                    scale_type=args.apollo_scale_type,
                    update_proj_gap=args.apollo_proj_gap,
                    proj_type=args.apollo_proj_type,
                    seed_base=apollo_seed_base,
                    scale_front=args.apollo_scale_front,
                    disable_nl=args.apollo_disable_nl,
                )
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> dict[str, float]:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        matrix_alignment = mx.array(0.0, dtype=mx.float32)
        matrix_grad_sq = mx.array(0.0, dtype=mx.float32)
        matrix_buf_sq = mx.array(0.0, dtype=mx.float32)
        turbo_alignment = mx.array(0.0, dtype=mx.float32)
        turbo_grad_sq = mx.array(0.0, dtype=mx.float32)
        turbo_buf_sq = mx.array(0.0, dtype=mx.float32)

        if self.muon is not None:
            matrix_updates, matrix_alignment, matrix_grad_sq, matrix_buf_sq = self.muon.step(params, grads, step=step, lr_mul=lr_mul)
            updated.update(matrix_updates)
        elif self.apollo is not None:
            matrix_updates, matrix_alignment, matrix_grad_sq, matrix_buf_sq = self.apollo.step(params, grads, lr_mul=lr_mul)
            updated.update(matrix_updates)
        if self.muon_turbo is not None:
            turbo_updates, turbo_alignment, turbo_grad_sq, turbo_buf_sq = self.muon_turbo.step(params, grads, step=step, lr_mul=lr_mul)
            updated.update(turbo_updates)
        elif self.apollo_turbo is not None:
            turbo_updates, turbo_alignment, turbo_grad_sq, turbo_buf_sq = self.apollo_turbo.step(params, grads, lr_mul=lr_mul)
            updated.update(turbo_updates)

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(self.adam_embed.apply_gradients({k: grads[k] for k in self.embed_keys}, {k: params[k] for k in self.embed_keys}))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))
        mx.eval(matrix_alignment, matrix_grad_sq, matrix_buf_sq, turbo_alignment, turbo_grad_sq, turbo_buf_sq)
        return {
            "matrix_alignment": float(matrix_alignment.item()),
            "matrix_grad_sq": float(matrix_grad_sq.item()),
            "matrix_buf_sq": float(matrix_buf_sq.item()),
            "turbo_alignment": float(turbo_alignment.item()),
            "turbo_grad_sq": float(turbo_grad_sq.item()),
            "turbo_buf_sq": float(turbo_buf_sq.item()),
        }

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
SERIALIZATION_SKIP_NAME_PATTERNS = ("._turbo_cache_",)
QUANT_FORMAT = os.environ.get("QUANT_FORMAT", "int8_clean_per_row_v1")
TURBO_BLOCK_SIZE = int(os.environ.get("TURBO_BLOCK_SIZE", 64))
TURBO_MSE_BITS = int(os.environ.get("TURBO_MSE_BITS", 5))
TURBO_PROD_BITS = int(os.environ.get("TURBO_PROD_BITS", 6))
TURBO_ROT_SEED = int(os.environ.get("TURBO_ROT_SEED", 17))
TURBO_QJL_SEED = int(os.environ.get("TURBO_QJL_SEED", 29))
TURBO_EMBED_EXPORT = bool(int(os.environ.get("TURBO_EMBED_EXPORT", "0")))
TIE_EMBEDDINGS_ENV = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
TURBO_MSE_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "TURBO_MSE_NAME_PATTERNS",
        "attn.c_q.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight",
    ).split(",")
    if pattern
)
TURBO_PROD_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "TURBO_PROD_NAME_PATTERNS",
        "attn.c_k.weight",
    ).split(",")
    if pattern
)
if TURBO_EMBED_EXPORT and "tok_emb.weight" not in TURBO_PROD_NAME_PATTERNS:
    if TIE_EMBEDDINGS_ENV:
        TURBO_PROD_NAME_PATTERNS = (*TURBO_PROD_NAME_PATTERNS, "tok_emb.weight")
    elif "tok_emb.weight" not in TURBO_MSE_NAME_PATTERNS:
        TURBO_MSE_NAME_PATTERNS = (*TURBO_MSE_NAME_PATTERNS, "tok_emb.weight")
if not TIE_EMBEDDINGS_ENV and "lm_head.weight" not in TURBO_PROD_NAME_PATTERNS:
    TURBO_PROD_NAME_PATTERNS = (*TURBO_PROD_NAME_PATTERNS, "lm_head.weight")

def sync_turbo_quant_globals() -> None:
    configure_turbo_quant(
        block_size=TURBO_BLOCK_SIZE,
        mse_bits=TURBO_MSE_BITS,
        prod_bits=TURBO_PROD_BITS,
        rot_seed=TURBO_ROT_SEED,
        qjl_seed=TURBO_QJL_SEED,
        mse_patterns=TURBO_MSE_NAME_PATTERNS,
        prod_patterns=TURBO_PROD_NAME_PATTERNS,
    )

sync_turbo_quant_globals()

def exportable_flat_state(model: nn.Module) -> dict[str, mx.array]:
    return {
        k: v
        for k, v in tree_flatten(model.state)
        if not any(part.startswith("_") for part in k.split("."))
        if not any(pattern in k for pattern in SERIALIZATION_SKIP_NAME_PATTERNS)
        if "early_exit_heads" not in k
        if "early_exit_condition_projs" not in k
        if "prosody_token_class_head" not in k
        if "prosody_boundary_head" not in k
        if "prosody_punctuation_head" not in k
        if "prosody_quote_head" not in k
    }


def cast_exportable_like(
    exportable_state: dict[str, mx.array],
    reference_state: dict[str, mx.array],
) -> dict[str, mx.array]:
    return {
        name: exportable_state[name].astype(reference_state[name].dtype)
        for name in exportable_state
        if name in reference_state
    }

def empty_quant_stats() -> dict[str, int]:
    return dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "num_passthrough_tensors",
            "num_fallback_int8_tensors",
            "num_turbo_mse_tensors",
            "num_turbo_prod_tensors",
            "passthrough_payload_bytes",
            "fallback_quantized_bytes",
            "fallback_scale_bytes",
            "turbo_mse_payload_bytes",
            "turbo_prod_payload_bytes",
            "turbo_norm_bytes",
            "turbo_qjl_bytes",
        ),
        0,
    )

def format_quant_stats(stats: dict[str, int]) -> str:
    param_count = max(int(stats["param_count"]), 1)
    payload_bpc = 8.0 * int(stats["int8_payload_bytes"]) / param_count
    norm_bpc = 8.0 * int(stats["turbo_norm_bytes"]) / param_count
    qjl_bpc = 8.0 * int(stats["turbo_qjl_bytes"]) / param_count
    return (
        "mix:"
        f" turbo_mse={stats['turbo_mse_payload_bytes']}"
        f" turbo_prod={stats['turbo_prod_payload_bytes']}"
        f" int8_q={stats['fallback_quantized_bytes']}"
        f" int8_scale={stats['fallback_scale_bytes']}"
        f" pass={stats['passthrough_payload_bytes']}"
        " tensors:"
        f" turbo_mse={stats['num_turbo_mse_tensors']}"
        f" turbo_prod={stats['num_turbo_prod_tensors']}"
        f" int8={stats['num_fallback_int8_tensors']}"
        f" pass={stats['num_passthrough_tensors']}"
        f" bpc:payload={payload_bpc:.3f}"
        f" norm={norm_bpc:.3f}"
        f" qjl={qjl_bpc:.3f}"
    )

def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)

def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))

def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale

def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = empty_quant_stats()
    for name, arr in flat_state.items():
        if any(pattern in name for pattern in SERIALIZATION_SKIP_NAME_PATTERNS):
            continue
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            stats["num_passthrough_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["passthrough_payload_bytes"] += int(passthrough[name].nbytes)
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_passthrough_tensors"] += 1
            stats["passthrough_payload_bytes"] += int(kept.nbytes)
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        stats["num_fallback_int8_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["fallback_quantized_bytes"] += int(q.nbytes)
        stats["fallback_scale_bytes"] += int(s.nbytes)
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def quantize_state_dict_turbo(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    turbo: dict[str, dict[str, object]] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = empty_quant_stats()
    for name, arr in flat_state.items():
        if any(pattern in name for pattern in SERIALIZATION_SKIP_NAME_PATTERNS):
            continue
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            stats["num_passthrough_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["passthrough_payload_bytes"] += int(passthrough[name].nbytes)
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        mode = infer_turbo_mode(name)
        if mode != "none" and arr.ndim == 2 and int(arr.shape[1]) >= TURBO_BLOCK_SIZE:
            stats["num_float_tensors"] += 1
            _, meta = turbo_quantize_dequantize_array(
                arr,
                mode=mode,
                total_bits=TURBO_MSE_BITS if mode == "mse" else TURBO_PROD_BITS,
                block_size=TURBO_BLOCK_SIZE,
            )
            turbo[name] = meta
            base_bytes = int(meta["norms"].nbytes + meta["idx_packed"].nbytes)
            stats["turbo_norm_bytes"] += int(meta["norms"].nbytes)
            if mode == "mse":
                stats["num_turbo_mse_tensors"] += 1
                stats["turbo_mse_payload_bytes"] += base_bytes
            else:
                stats["num_turbo_prod_tensors"] += 1
                stats["turbo_prod_payload_bytes"] += base_bytes
            stats["int8_payload_bytes"] += base_bytes
            if "residual_norms" in meta:
                qjl_bytes = int(meta["residual_norms"].nbytes + meta["qjl_packed"].nbytes)
                stats["turbo_norm_bytes"] += int(meta["residual_norms"].nbytes)
                stats["turbo_qjl_bytes"] += int(meta["qjl_packed"].nbytes)
                stats["turbo_prod_payload_bytes"] += qjl_bytes
                stats["int8_payload_bytes"] += qjl_bytes
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_passthrough_tensors"] += 1
            stats["passthrough_payload_bytes"] += int(kept.nbytes)
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        stats["num_fallback_int8_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["fallback_quantized_bytes"] += int(q.nbytes)
        stats["fallback_scale_bytes"] += int(s.nbytes)
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj: dict[str, object] = {
        "__quant_format__": "turbo_block_v1",
        "turbo_config": {
            "scheme_version": TURBO_SCHEME_VERSION,
            "rotation_kind": TURBO_ROTATION_KIND,
            "codebook_kind": TURBO_CODEBOOK_KIND,
            "qjl_kind": TURBO_QJL_KIND,
            "block_size": TURBO_BLOCK_SIZE,
            "mse_bits": TURBO_MSE_BITS,
            "prod_bits": TURBO_PROD_BITS,
            "rot_seed": TURBO_ROT_SEED,
            "qjl_seed": TURBO_QJL_SEED,
            "mse_name_patterns": TURBO_MSE_NAME_PATTERNS,
            "prod_name_patterns": TURBO_PROD_NAME_PATTERNS,
        },
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if turbo:
        obj["turbo"] = turbo
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def serialize_quantized_state_dict_int8(
    flat_state: dict[str, mx.array],
) -> tuple[dict[str, object], dict[str, int], bytes, bytes]:
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    return quant_obj, quant_stats, quant_raw, quant_blob

def serialize_quantized_state_dict(
    flat_state: dict[str, mx.array],
) -> tuple[dict[str, object], dict[str, int], bytes, bytes]:
    if QUANT_FORMAT == "turbo_block_v1":
        quant_obj, quant_stats = quantize_state_dict_turbo(flat_state)
        quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        quant_blob = zlib.compress(quant_raw, level=9)
        return quant_obj, quant_stats, quant_raw, quant_blob
    if QUANT_FORMAT != "int8_clean_per_row_v1":
        raise ValueError(f"Unsupported QUANT_FORMAT={QUANT_FORMAT}")
    return serialize_quantized_state_dict_int8(flat_state)

def dequantize_state_dict(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    codebook_overrides = quant_obj.get("turbo_codebook_overrides", {})
    tensor_codebook_overrides = quant_obj.get("turbo_codebook_overrides_by_tensor", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    for name, meta in quant_obj.get("turbo", {}).items():
        scheme = str(meta.get("scheme", ""))
        if scheme == "turbo_sliced_rows_v1":
            parts_out: list[mx.array] = []
            for child in meta.get("parts", ()):
                child_mode = str(child.get("mode", ""))
                child_total_bits = int(child.get("bits", 0))
                child_block_size = int(child.get("block_size", TURBO_BLOCK_SIZE))
                child_mse_bits = int(child.get("mse_bits", child_total_bits if child_mode == "mse" else child_total_bits - 1))
                child_override = None
                if isinstance(tensor_codebook_overrides, dict):
                    tensor_override_map = tensor_codebook_overrides.get(name)
                    if isinstance(tensor_override_map, dict):
                        child_override = tensor_override_map.get(f"{child_mode}:{child_mse_bits}:{child_block_size}")
                if isinstance(codebook_overrides, dict) and child_override is None:
                    child_override = codebook_overrides.get(f"{child_mode}:{child_mse_bits}:{child_block_size}")
                parts_out.append(
                    dequantize_turbo_tensor(
                        child,
                        None if child_override is None else np.asarray(child_override, dtype=np.float32),
                    )
                )
            axis = int(meta.get("axis", 0))
            out[name] = mx.concatenate(parts_out, axis=axis) if parts_out else mx.zeros(tuple(int(x) for x in meta.get("shape", ())), dtype=MX_DTYPE_FROM_NAME[str(meta.get("dtype", "float32"))])
            continue
        mode = str(meta.get("mode", ""))
        total_bits = int(meta.get("bits", 0))
        block_size = int(meta.get("block_size", TURBO_BLOCK_SIZE))
        mse_bits = int(meta.get("mse_bits", total_bits if mode == "mse" else total_bits - 1))
        override = None
        if isinstance(tensor_codebook_overrides, dict):
            tensor_override_map = tensor_codebook_overrides.get(name)
            if isinstance(tensor_override_map, dict):
                override = tensor_override_map.get(f"{mode}:{mse_bits}:{block_size}")
        if isinstance(codebook_overrides, dict):
            if override is None:
                override = codebook_overrides.get(f"{mode}:{mse_bits}:{block_size}")
        out[name] = dequantize_turbo_tensor(
            meta,
            None if override is None else np.asarray(override, dtype=np.float32),
        )
    for name, meta in quant_obj.get("ternary", {}).items():
        out[name] = dequantize_ternary_tensor(meta)
    return out

def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    return dequantize_state_dict(quant_obj)

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut

def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files

def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def limit_validation_tokens(val_tokens: np.ndarray, seq_len: int, max_seqs: int) -> np.ndarray:
    if max_seqs <= 0:
        return val_tokens
    usable_seqs = min(max_seqs, (val_tokens.size - 1) // seq_len)
    if usable_seqs <= 0:
        raise ValueError(f"QUANT_EVAL_MAX_SEQS must allow at least one sequence, got {max_seqs}")
    return val_tokens[: usable_seqs * seq_len + 1]


def build_sliding_eval_windows(total_tokens: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > seq_len:
        raise ValueError(f"stride must be <= seq_len, got stride={stride}, seq_len={seq_len}")
    windows: list[tuple[int, int, int, int]] = []
    next_score_token = 0
    for window_start in range(0, total_tokens, stride):
        window_end = min(window_start + seq_len, total_tokens)
        window_len = window_end - window_start
        if window_len < stride:
            continue
        default_score_start = window_start if window_start == 0 else window_end - stride
        score_start_abs = max(default_score_start, next_score_token)
        if score_start_abs >= window_end:
            continue
        score_start = score_start_abs - window_start
        windows.append((window_start, window_len, score_start, window_len))
        next_score_token = window_end
    if not windows:
        raise ValueError(
            f"Validation split is too short for sliding eval: total_tokens={total_tokens}, seq_len={seq_len}, stride={stride}"
        )
    return windows


def operator_codes_mx_for_numpy_batch(
    model: GPT,
    input_ids: np.ndarray,
    *,
    enabled: bool = True,
) -> mx.array | None:
    operator_codes = model.operator_codes_for_numpy(input_ids)
    if operator_codes is None:
        return None
    if not enabled:
        operator_codes = np.zeros_like(operator_codes, dtype=np.int32)
    return mx.array(operator_codes, dtype=mx.int32)


def split_csv_env_paths(raw: str) -> list[Path]:
    return [Path(os.path.expanduser(part.strip())) for part in raw.split(",") if part.strip()]


def load_config_env_payload(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    env_payload = payload.get("env", payload)
    if not isinstance(env_payload, dict):
        raise ValueError(f"Config env payload must be a JSON object: {config_path}")
    return env_payload


def _env_text(payload: dict[str, object], key: str, default: str) -> str:
    value = payload.get(key, default)
    return str(value)


def _env_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    return int(value)


def _env_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    return float(value)


def _env_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean-like value for {key}, got {value!r}")


def build_teacher_args_from_env(
    env_payload: dict[str, object],
    fallback: Hyperparameters,
) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=_env_int(env_payload, "VOCAB_SIZE", fallback.vocab_size),
        tokenizer_path=os.path.expanduser(_env_text(env_payload, "TOKENIZER_PATH", fallback.tokenizer_path)),
        num_layers=_env_int(env_payload, "NUM_LAYERS", fallback.num_layers),
        num_layer_templates=_env_int(env_payload, "NUM_LAYER_TEMPLATES", fallback.num_layer_templates),
        model_dim=_env_int(env_payload, "MODEL_DIM", fallback.model_dim),
        num_heads=_env_int(env_payload, "NUM_HEADS", fallback.num_heads),
        num_kv_heads=_env_int(env_payload, "NUM_KV_HEADS", fallback.num_kv_heads),
        mlp_mult=_env_int(env_payload, "MLP_MULT", fallback.mlp_mult),
        mlp_leaky_slope=_env_float(env_payload, "MLP_LEAKY_SLOPE", fallback.mlp_leaky_slope),
        tie_embeddings=_env_bool(env_payload, "TIE_EMBEDDINGS", fallback.tie_embeddings),
        logit_chunk_tokens=_env_int(env_payload, "LOGIT_CHUNK_TOKENS", fallback.logit_chunk_tokens),
        logit_softcap=_env_float(env_payload, "LOGIT_SOFTCAP", fallback.logit_softcap),
        rope_base=_env_float(env_payload, "ROPE_BASE", fallback.rope_base),
        tied_embed_init_std=_env_float(env_payload, "TIED_EMBED_INIT_STD", fallback.tied_embed_init_std),
        qk_gain_init=_env_float(env_payload, "QK_GAIN_INIT", fallback.qk_gain_init),
        num_registers=_env_int(env_payload, "NUM_REGISTERS", fallback.num_registers),
        register_layout=_env_text(env_payload, "REGISTER_LAYOUT", fallback.register_layout),
        register_stride=_env_int(env_payload, "REGISTER_STRIDE", fallback.register_stride),
        register_mask_mode=_env_text(env_payload, "REGISTER_MASK_MODE", fallback.register_mask_mode),
        logic_dim=_env_int(env_payload, "LOGIC_DIM", fallback.logic_dim),
        logic_layer_index=_env_int(env_payload, "LOGIC_LAYER_INDEX", fallback.logic_layer_index),
        logic_route_to_next_token=_env_bool(
            env_payload,
            "LOGIC_ROUTE_TO_NEXT_TOKEN",
            fallback.logic_route_to_next_token,
        ),
        logic_operator_mode=_env_text(env_payload, "LOGIC_OPERATOR_MODE", fallback.logic_operator_mode),
        polarity_detector_enabled=_env_bool(
            env_payload,
            "POLARITY_DETECTOR_ENABLED",
            fallback.polarity_detector_enabled,
        ),
        polarity_detector_layer_index=_env_int(
            env_payload,
            "POLARITY_DETECTOR_LAYER_INDEX",
            fallback.polarity_detector_layer_index,
        ),
        polarity_detector_hidden_dim=_env_int(
            env_payload,
            "POLARITY_DETECTOR_HIDDEN_DIM",
            fallback.polarity_detector_hidden_dim,
        ),
        polarity_seed_blend=_env_float(env_payload, "POLARITY_SEED_BLEND", fallback.polarity_seed_blend),
        polarity_seed_weight=_env_float(env_payload, "POLARITY_SEED_WEIGHT", fallback.polarity_seed_weight),
        polarity_sparse_weight=_env_float(env_payload, "POLARITY_SPARSE_WEIGHT", fallback.polarity_sparse_weight),
        polarity_smooth_weight=_env_float(env_payload, "POLARITY_SMOOTH_WEIGHT", fallback.polarity_smooth_weight),
        hardmax_struct_num_states=_env_int(env_payload, "HARDMAX_STRUCT_NUM_STATES", fallback.hardmax_struct_num_states),
        hardmax_struct_dim=_env_int(env_payload, "HARDMAX_STRUCT_DIM", fallback.hardmax_struct_dim),
        hardmax_struct_static_adapter=_env_bool(
            env_payload,
            "HARDMAX_STRUCT_STATIC_ADAPTER",
            fallback.hardmax_struct_static_adapter,
        ),
        hardmax_struct_layer_index=_env_int(env_payload, "HARDMAX_STRUCT_LAYER_INDEX", fallback.hardmax_struct_layer_index),
        hardmax_struct_router_start_layer=_env_int(
            env_payload,
            "HARDMAX_STRUCT_ROUTER_START_LAYER",
            fallback.hardmax_struct_router_start_layer,
        ),
        hardmax_struct_temperature=_env_float(env_payload, "HARDMAX_STRUCT_TEMPERATURE", fallback.hardmax_struct_temperature),
        hardmax_struct_temperature_start=_env_float(
            env_payload,
            "HARDMAX_STRUCT_TEMPERATURE_START",
            fallback.hardmax_struct_temperature_start,
        ),
        hardmax_struct_temperature_end=_env_float(
            env_payload,
            "HARDMAX_STRUCT_TEMPERATURE_END",
            fallback.hardmax_struct_temperature_end,
        ),
        hardmax_struct_temperature_anneal_frac=_env_float(
            env_payload,
            "HARDMAX_STRUCT_TEMPERATURE_ANNEAL_FRAC",
            fallback.hardmax_struct_temperature_anneal_frac,
        ),
        hardmax_struct_fast_refinement_steps=_env_int(
            env_payload,
            "HARDMAX_STRUCT_FAST_REFINEMENT_STEPS",
            fallback.hardmax_struct_fast_refinement_steps,
        ),
        hardmax_struct_compute_min_scale=_env_float(
            env_payload,
            "HARDMAX_STRUCT_COMPUTE_MIN_SCALE",
            fallback.hardmax_struct_compute_min_scale,
        ),
        hardmax_struct_compute_power=_env_float(
            env_payload,
            "HARDMAX_STRUCT_COMPUTE_POWER",
            fallback.hardmax_struct_compute_power,
        ),
        hardmax_struct_route_residual_budget=_env_bool(
            env_payload,
            "HARDMAX_STRUCT_ROUTE_RESIDUAL_BUDGET",
            fallback.hardmax_struct_route_residual_budget,
        ),
        hardmax_struct_operator_prior_scale=_env_float(
            env_payload,
            "HARDMAX_STRUCT_OPERATOR_PRIOR_SCALE",
            fallback.hardmax_struct_operator_prior_scale,
        ),
        hardmax_struct_reset_prior_scale=_env_float(
            env_payload,
            "HARDMAX_STRUCT_RESET_PRIOR_SCALE",
            fallback.hardmax_struct_reset_prior_scale,
        ),
        hardmax_struct_usage_balance_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_USAGE_BALANCE_WEIGHT",
            fallback.hardmax_struct_usage_balance_weight,
        ),
        hardmax_struct_diversity_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_DIVERSITY_WEIGHT",
            fallback.hardmax_struct_diversity_weight,
        ),
        hardmax_struct_predict_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_PREDICT_WEIGHT",
            fallback.hardmax_struct_predict_weight,
        ),
        hardmax_struct_confidence_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_CONFIDENCE_WEIGHT",
            fallback.hardmax_struct_confidence_weight,
        ),
        hardmax_struct_operator_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_OPERATOR_WEIGHT",
            fallback.hardmax_struct_operator_weight,
        ),
        hardmax_struct_token_class_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_TOKEN_CLASS_WEIGHT",
            fallback.hardmax_struct_token_class_weight,
        ),
        hardmax_struct_boundary_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_BOUNDARY_WEIGHT",
            fallback.hardmax_struct_boundary_weight,
        ),
        hardmax_struct_quote_weight=_env_float(
            env_payload,
            "HARDMAX_STRUCT_QUOTE_WEIGHT",
            fallback.hardmax_struct_quote_weight,
        ),
        hardmax_struct_condition_mode=_env_text(
            env_payload,
            "HARDMAX_STRUCT_CONDITION_MODE",
            fallback.hardmax_struct_condition_mode,
        ),
        hardmax_struct_attn_q_scale=_env_float(
            env_payload,
            "HARDMAX_STRUCT_ATTN_Q_SCALE",
            fallback.hardmax_struct_attn_q_scale,
        ),
        hardmax_struct_attn_tau_min=_env_float(
            env_payload,
            "HARDMAX_STRUCT_ATTN_TAU_MIN",
            fallback.hardmax_struct_attn_tau_min,
        ),
        early_exit_layer_index=_env_int(env_payload, "EARLY_EXIT_LAYER_INDEX", fallback.early_exit_layer_index),
        early_exit_horizons=_env_text(env_payload, "EARLY_EXIT_HORIZONS", fallback.early_exit_horizons),
        early_exit_aux_weight=_env_float(env_payload, "EARLY_EXIT_AUX_WEIGHT", fallback.early_exit_aux_weight),
        early_exit_head_init_std=_env_float(
            env_payload,
            "EARLY_EXIT_HEAD_INIT_STD",
            fallback.early_exit_head_init_std,
        ),
        early_exit_cascaded_enabled=_env_bool(
            env_payload,
            "EARLY_EXIT_CASCADED_ENABLED",
            fallback.early_exit_cascaded_enabled,
        ),
        early_exit_condition_init_std=_env_float(
            env_payload,
            "EARLY_EXIT_CONDITION_INIT_STD",
            fallback.early_exit_condition_init_std,
        ),
        early_exit_branch_draft_enabled=_env_bool(
            env_payload,
            "EARLY_EXIT_BRANCH_DRAFT_ENABLED",
            fallback.early_exit_branch_draft_enabled,
        ),
        early_exit_branch_conf_threshold=_env_float(
            env_payload,
            "EARLY_EXIT_BRANCH_CONF_THRESHOLD",
            fallback.early_exit_branch_conf_threshold,
        ),
        early_exit_branch_max_draft_tokens=_env_int(
            env_payload,
            "EARLY_EXIT_BRANCH_MAX_DRAFT_TOKENS",
            fallback.early_exit_branch_max_draft_tokens,
        ),
        prosody_type_embeddings_enabled=_env_bool(
            env_payload,
            "PROSODY_TYPE_EMBEDDINGS_ENABLED",
            fallback.prosody_type_embeddings_enabled,
        ),
        prosody_type_embedding_init_std=_env_float(
            env_payload,
            "PROSODY_TYPE_EMBEDDING_INIT_STD",
            fallback.prosody_type_embedding_init_std,
        ),
        prosody_extended_feature_set_enabled=_env_bool(
            env_payload,
            "PROSODY_EXTENDED_FEATURE_SET_ENABLED",
            fallback.prosody_extended_feature_set_enabled,
        ),
        prosody_feature_embeddings_enabled=_env_bool(
            env_payload,
            "PROSODY_FEATURE_EMBEDDINGS_ENABLED",
            fallback.prosody_feature_embeddings_enabled,
        ),
        prosody_feature_embedding_init_std=_env_float(
            env_payload,
            "PROSODY_FEATURE_EMBEDDING_INIT_STD",
            fallback.prosody_feature_embedding_init_std,
        ),
        prosody_state_adapter_enabled=_env_bool(
            env_payload,
            "PROSODY_STATE_ADAPTER_ENABLED",
            fallback.prosody_state_adapter_enabled,
        ),
        prosody_state_dim=_env_int(
            env_payload,
            "PROSODY_STATE_DIM",
            fallback.prosody_state_dim,
        ),
        prosody_state_init_std=_env_float(
            env_payload,
            "PROSODY_STATE_INIT_STD",
            fallback.prosody_state_init_std,
        ),
        prosody_state_scale=_env_float(
            env_payload,
            "PROSODY_STATE_SCALE",
            fallback.prosody_state_scale,
        ),
        prosody_state_reset_prior_weight=_env_float(
            env_payload,
            "PROSODY_STATE_RESET_PRIOR_WEIGHT",
            fallback.prosody_state_reset_prior_weight,
        ),
        prosody_state_hierarchical_enabled=_env_bool(
            env_payload,
            "PROSODY_STATE_HIERARCHICAL_ENABLED",
            fallback.prosody_state_hierarchical_enabled,
        ),
        prosody_state_slow_reset_scale=_env_float(
            env_payload,
            "PROSODY_STATE_SLOW_RESET_SCALE",
            fallback.prosody_state_slow_reset_scale,
        ),
        wallclock_final_reserve_seconds=_env_float(
            env_payload,
            "WALLCLOCK_FINAL_RESERVE_SECONDS",
            fallback.wallclock_final_reserve_seconds,
        ),
        prosody_aux_layer_index=_env_int(
            env_payload,
            "PROSODY_AUX_LAYER_INDEX",
            fallback.prosody_aux_layer_index,
        ),
        prosody_aux_weight=_env_float(
            env_payload,
            "PROSODY_AUX_WEIGHT",
            fallback.prosody_aux_weight,
        ),
        prosody_aux_head_init_std=_env_float(
            env_payload,
            "PROSODY_AUX_HEAD_INIT_STD",
            fallback.prosody_aux_head_init_std,
        ),
        prosody_aux_token_class_weight=_env_float(
            env_payload,
            "PROSODY_AUX_TOKEN_CLASS_WEIGHT",
            fallback.prosody_aux_token_class_weight,
        ),
        prosody_aux_boundary_weight=_env_float(
            env_payload,
            "PROSODY_AUX_BOUNDARY_WEIGHT",
            fallback.prosody_aux_boundary_weight,
        ),
        prosody_aux_quote_weight=_env_float(
            env_payload,
            "PROSODY_AUX_QUOTE_WEIGHT",
            fallback.prosody_aux_quote_weight,
        ),
        prosody_aux_punctuation_weight=_env_float(
            env_payload,
            "PROSODY_AUX_PUNCTUATION_WEIGHT",
            fallback.prosody_aux_punctuation_weight,
        ),
    )


def load_external_teacher_models(
    args: Hyperparameters,
    student_sp: spm.SentencePieceProcessor,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[GPT], list[dict[str, object]]]:
    config_paths = split_csv_env_paths(args.external_teacher_config_paths)
    checkpoint_paths = split_csv_env_paths(args.external_teacher_checkpoint_paths)
    if bool(config_paths) != bool(checkpoint_paths):
        raise ValueError(
            "EXTERNAL_TEACHER_CONFIG_PATHS and EXTERNAL_TEACHER_CHECKPOINT_PATHS must both be set or both be empty"
        )
    if len(config_paths) != len(checkpoint_paths):
        raise ValueError(
            "EXTERNAL_TEACHER_CONFIG_PATHS and EXTERNAL_TEACHER_CHECKPOINT_PATHS must have the same number of entries"
        )
    teacher_models: list[GPT] = []
    teacher_meta: list[dict[str, object]] = []
    student_tokenizer_resolved = Path(args.tokenizer_path).expanduser().resolve()
    for idx, (config_path, checkpoint_path) in enumerate(zip(config_paths, checkpoint_paths, strict=True), start=1):
        if not config_path.is_file():
            raise FileNotFoundError(f"External teacher config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"External teacher checkpoint not found: {checkpoint_path}")
        env_payload = load_config_env_payload(config_path)
        teacher_args = build_teacher_args_from_env(env_payload, args)
        teacher_tokenizer_resolved = Path(teacher_args.tokenizer_path).expanduser().resolve()
        if teacher_tokenizer_resolved != student_tokenizer_resolved:
            raise ValueError(
                "External teacher tokenizer must match the student tokenizer exactly. "
                f"student={student_tokenizer_resolved} teacher={teacher_tokenizer_resolved}"
            )
        if int(teacher_args.vocab_size) != args.vocab_size:
            raise ValueError(
                f"External teacher vocab_size mismatch: student={args.vocab_size} teacher={teacher_args.vocab_size}"
            )
        teacher_model = GPT(
            **gpt_kwargs_from_args(teacher_args, student_sp),
            early_exit_layer_index=teacher_args.early_exit_layer_index,
            early_exit_horizons=parse_horizons(teacher_args.early_exit_horizons),
            early_exit_aux_weight=teacher_args.early_exit_aux_weight,
            early_exit_head_init_std=teacher_args.early_exit_head_init_std,
            early_exit_cascaded_enabled=teacher_args.early_exit_cascaded_enabled,
            early_exit_condition_init_std=teacher_args.early_exit_condition_init_std,
            early_exit_branch_draft_enabled=teacher_args.early_exit_branch_draft_enabled,
            early_exit_branch_conf_threshold=teacher_args.early_exit_branch_conf_threshold,
            early_exit_branch_max_draft_tokens=teacher_args.early_exit_branch_max_draft_tokens,
        )
        teacher_model.set_turbo_qat(False, 0.0)
        checkpoint_state = dict(mx.load(str(checkpoint_path)))
        compatible_state, load_stats = select_compatible_flat_state(teacher_model, checkpoint_state)
        if args.external_teacher_allow_partial_load:
            if float(load_stats["matched_param_fraction"]) < args.external_teacher_min_param_fraction:
                raise ValueError(
                    "External teacher partial load coverage too low: "
                    f"{float(load_stats['matched_param_fraction']):.3f} < {args.external_teacher_min_param_fraction:.3f} "
                    f"for {checkpoint_path}"
                )
        else:
            if (
                int(load_stats["missing_tensors"]) > 0
                or int(load_stats["mismatched_tensors"]) > 0
                or int(load_stats["unexpected_tensors"]) > 0
            ):
                raise ValueError(
                    "External teacher checkpoint does not exactly match instantiated model. "
                    "Set EXTERNAL_TEACHER_ALLOW_PARTIAL_LOAD=1 to allow compatible intersection loading."
                )
        apply_flat_arrays(teacher_model, compatible_state)
        teacher_model.clear_turbo_cache()
        teacher_models.append(teacher_model)
        meta = {
            "index": idx,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "layers": int(teacher_args.num_layers),
            "dim": int(teacher_args.model_dim),
            "heads": int(teacher_args.num_heads),
            "logic_dim": int(teacher_args.logic_dim),
            "tie_embeddings": bool(teacher_args.tie_embeddings),
            "matched_param_fraction": float(load_stats["matched_param_fraction"]),
            "missing_tensors": int(load_stats["missing_tensors"]),
            "mismatched_tensors": int(load_stats["mismatched_tensors"]),
            "unexpected_tensors": int(load_stats["unexpected_tensors"]),
        }
        teacher_meta.append(meta)
        if log_fn is not None:
            log_fn(
                f"external_teacher_loaded:{idx}/{len(config_paths)} "
                f"layers:{teacher_args.num_layers} dim:{teacher_args.model_dim} heads:{teacher_args.num_heads} "
                f"logic_dim:{teacher_args.logic_dim} matched_param_frac:{float(load_stats['matched_param_fraction']):.3f} "
                f"missing_tensors:{int(load_stats['missing_tensors'])} "
                f"mismatched_tensors:{int(load_stats['mismatched_tensors'])} "
                f"unexpected_tensors:{int(load_stats['unexpected_tensors'])} "
                f"config:{config_path} checkpoint:{checkpoint_path}"
            )
            if args.external_teacher_allow_partial_load:
                topk = max(args.external_teacher_load_log_topk, 0)
                missing_top = summarize_name_numel(load_stats["missing_top"], topk=topk)
                mismatched_top = summarize_name_numel(load_stats["mismatched_top"], topk=topk)
                unexpected_top = summarize_name_numel(load_stats["unexpected_top"], topk=topk)
                if missing_top:
                    log_fn(f"external_teacher_missing_top:{idx} {missing_top}")
                if mismatched_top:
                    log_fn(f"external_teacher_mismatched_top:{idx} {mismatched_top}")
                if unexpected_top:
                    log_fn(f"external_teacher_unexpected_top:{idx} {unexpected_top}")
    return teacher_models, teacher_meta


def ensemble_teacher_logits_for_batch(
    teacher_models: list[GPT] | None,
    x: mx.array,
    x_np: np.ndarray,
    *,
    logic_phase_enabled: bool,
    collect_metrics: bool = False,
) -> tuple[mx.array | None, dict[str, float] | None]:
    if not teacher_models:
        return None, None
    logits_list: list[mx.array] = []
    for teacher_model in teacher_models:
        teacher_operator_codes = operator_codes_mx_for_numpy_batch(
            teacher_model,
            x_np,
            enabled=logic_phase_enabled,
        )
        logits_list.append(mx.stop_gradient(teacher_model.forward_logits(x, teacher_operator_codes).astype(mx.float32)))
    if len(logits_list) == 1:
        return logits_list[0], {
            "teacher_count": 1.0,
            "teacher_disagree_frac": 0.0,
            "teacher_unique_top1": 1.0,
            "teacher_pairwise_agree": 1.0,
        }
    ensemble_logits = mx.mean(mx.stack(logits_list, axis=0), axis=0)
    if not collect_metrics:
        return ensemble_logits, {
            "teacher_count": float(len(logits_list)),
            "teacher_disagree_frac": 0.0,
            "teacher_unique_top1": 1.0,
            "teacher_pairwise_agree": 1.0,
        }
    top1 = np.stack(
        [np.asarray(mx.argmax(logits, axis=-1), dtype=np.int32) for logits in logits_list],
        axis=0,
    )
    unique_top1 = np.ones_like(top1[0], dtype=np.int32)
    for idx in range(1, top1.shape[0]):
        unique_top1 += np.all(top1[idx : idx + 1] != top1[:idx], axis=0)
    disagree_frac = float(np.mean(np.any(top1 != top1[:1], axis=0), dtype=np.float64))
    pairwise_agree = 1.0
    pair_count = 0
    pair_sum = 0.0
    for left in range(top1.shape[0]):
        for right in range(left + 1, top1.shape[0]):
            pair_sum += float(np.mean(top1[left] == top1[right], dtype=np.float64))
            pair_count += 1
    if pair_count > 0:
        pairwise_agree = pair_sum / pair_count
    return ensemble_logits, {
        "teacher_count": float(top1.shape[0]),
        "teacher_disagree_frac": disagree_frac,
        "teacher_unique_top1": float(np.mean(unique_top1, dtype=np.float64)),
        "teacher_pairwise_agree": float(pairwise_agree),
    }


def _teacher_hidden_for_model_batch(
    teacher_model: GPT,
    x: mx.array,
    x_np: np.ndarray,
    *,
    logic_phase_enabled: bool,
    hidden_layer: int,
) -> mx.array:
    teacher_operator_codes = operator_codes_mx_for_numpy_batch(
        teacher_model,
        x_np,
        enabled=logic_phase_enabled,
    )
    if hidden_layer < 0:
        hidden, _captured = teacher_model.forward_hidden(x, operator_codes=teacher_operator_codes)
        return mx.stop_gradient(hidden.astype(mx.float32))
    hidden, captured = teacher_model.forward_hidden(
        x,
        capture_layers=(int(hidden_layer),),
        operator_codes=teacher_operator_codes,
    )
    captured_hidden = captured.get(int(hidden_layer))
    if captured_hidden is None:
        raise ValueError(f"Teacher hidden layer {hidden_layer} was not captured")
    _ = hidden
    return mx.stop_gradient(captured_hidden.astype(mx.float32))


def ensemble_teacher_hidden_for_batch(
    teacher_models: list[GPT] | None,
    x: mx.array,
    x_np: np.ndarray,
    *,
    logic_phase_enabled: bool,
    hidden_layer: int,
) -> mx.array | None:
    if not teacher_models:
        return None
    hidden_list: list[mx.array] = []
    for teacher_model in teacher_models:
        hidden_list.append(
            _teacher_hidden_for_model_batch(
                teacher_model,
                x,
                x_np,
                logic_phase_enabled=logic_phase_enabled,
                hidden_layer=hidden_layer,
            )
        )
    if len(hidden_list) == 1:
        return hidden_list[0]
    return mx.mean(mx.stack(hidden_list, axis=0), axis=0)


def lookup_teacher_hidden_cache_for_batch(
    teacher_hidden_cache: FileTeacherHiddenCache | None,
    x_np: np.ndarray,
    y_np: np.ndarray,
) -> tuple[list[np.ndarray | None] | None, dict[str, float] | None]:
    if teacher_hidden_cache is None:
        return None, None
    return teacher_hidden_cache.lookup_rows(x_np, y_np)


def merge_teacher_hidden_rows(
    cached_rows: list[np.ndarray | None] | None,
    live_hidden: mx.array | None,
) -> tuple[mx.array | None, int]:
    if cached_rows is None:
        return live_hidden, 0
    miss_count = int(sum(row is None for row in cached_rows))
    if live_hidden is None and miss_count > 0:
        return None, miss_count
    if live_hidden is None:
        stacked = np.stack([np.asarray(row, dtype=np.float32) for row in cached_rows], axis=0)
        return mx.array(stacked, dtype=mx.float32), 0
    live_np = np.asarray(mx.stop_gradient(live_hidden.astype(mx.float32)))
    merged_rows: list[np.ndarray] = []
    for row_idx, cached in enumerate(cached_rows):
        merged_rows.append(np.asarray(cached, dtype=np.float32) if cached is not None else live_np[row_idx])
    return mx.array(np.stack(merged_rows, axis=0), dtype=mx.float32), miss_count


def append_teacher_hidden_cache_for_batch(
    teacher_hidden_cache: FileTeacherHiddenCache | None,
    x_np: np.ndarray,
    y_np: np.ndarray,
    teacher_hidden: mx.array | None,
    *,
    step: int,
    source_run_id: str,
    cached_rows: list[np.ndarray | None] | None,
) -> int:
    if teacher_hidden_cache is None or teacher_hidden is None:
        return 0
    hidden_np = np.asarray(mx.stop_gradient(teacher_hidden.astype(mx.float32)))
    examples: list[TeacherHiddenExample] = []
    for row_idx, (x_row, y_row) in enumerate(zip(x_np, y_np, strict=True)):
        if cached_rows is not None and row_idx < len(cached_rows) and cached_rows[row_idx] is not None:
            continue
        tokens = window_tokens_from_xy(x_row, y_row)
        key = teacher_window_key(
            tokens,
            layer_index=teacher_hidden_cache.layer_index,
            hidden_dim=teacher_hidden_cache.hidden_dim,
        )
        examples.append(
            TeacherHiddenExample(
                key=key,
                tokens=tokens,
                hidden=hidden_np[row_idx],
                layer_index=teacher_hidden_cache.layer_index,
                step=step,
                source_run_id=source_run_id,
            )
        )
    return teacher_hidden_cache.append_examples(examples)


def process_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def metal_memory_snapshot() -> dict[str, float]:
    snapshot = {
        "metal_active_bytes": float("nan"),
        "metal_peak_bytes": float("nan"),
        "metal_cache_bytes": float("nan"),
        "host_rss_bytes": float(process_rss_bytes()),
    }
    for key, attrs in (
        ("metal_active_bytes", ("get_active_memory",)),
        ("metal_peak_bytes", ("get_peak_memory",)),
        ("metal_cache_bytes", ("get_cache_memory",)),
    ):
        fn = next((getattr(mx, attr, None) for attr in attrs if getattr(mx, attr, None) is not None), None)
        if fn is None:
            metal = getattr(mx, "metal", None)
            fn = next((getattr(metal, attr, None) for attr in attrs if metal is not None and getattr(metal, attr, None) is not None), None)
        if fn is None:
            continue
        try:
            snapshot[key] = float(fn())
        except Exception:
            continue
    return snapshot


def reset_metal_peak_memory() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, "reset_peak_memory", None) if metal is not None else None
    if fn is None:
        return
    try:
        fn()
    except Exception:
        return


def snapshot_exportable_state(
    model: GPT,
    ema_state: dict[str, mx.array] | None,
    *,
    use_ema: bool,
) -> dict[str, mx.array]:
    exportable = exportable_flat_state(model)
    if not use_ema or ema_state is None:
        return exportable
    return cast_exportable_like(ema_state, exportable)


def maybe_write_student_heartbeat(
    runtime: StudentSnapshotRuntime | None,
    args: Hyperparameters,
    *,
    step: int,
    train_time_ms: float,
    step_avg_ms: float,
    train_loss_value: float,
    tok_s: float,
    qat_scale: float,
    distill_weight: float,
    replay_buffer: FileReplayBuffer | None,
    controller_metrics: dict[str, float] | None = None,
) -> None:
    if runtime is None or args.student_heartbeat_every <= 0:
        return
    if step <= 0 or step % args.student_heartbeat_every != 0:
        return
    payload: dict[str, object] = {
        "run_id": args.run_id,
        "step": int(step),
        "train_time_ms": float(train_time_ms),
        "step_avg_ms": float(step_avg_ms),
        "train_loss": float(train_loss_value),
        "tok_s": float(tok_s),
        "turbo_qat_scale": float(qat_scale),
        "distill_weight": float(distill_weight),
    }
    if replay_buffer is not None:
        payload.update(replay_buffer.summary())
    if controller_metrics is not None:
        payload["controller_metrics"] = {key: float(value) for key, value in controller_metrics.items()}
    runtime.write_heartbeat(payload)


def maybe_write_student_snapshot(
    runtime: StudentSnapshotRuntime | None,
    args: Hyperparameters,
    model: GPT,
    ema_state: dict[str, mx.array] | None,
    *,
    step: int,
    train_time_ms: float,
    train_loss_value: float,
    tok_s: float,
    val_bpb: float | None,
    quant_gap_bpb: float | None,
    replay_buffer: FileReplayBuffer | None,
) -> Path | None:
    if runtime is None or args.student_snapshot_every <= 0:
        return None
    if step <= 0 or step % args.student_snapshot_every != 0:
        return None
    snapshot_state = snapshot_exportable_state(
        model,
        ema_state,
        use_ema=args.student_snapshot_use_ema,
    )
    snapshot_path = runtime.snapshot_npz_path(step)
    mx.savez(str(snapshot_path), **snapshot_state)
    meta: dict[str, object] = {
        "run_id": args.run_id,
        "step": int(step),
        "snapshot_path": str(snapshot_path),
        "train_time_ms": float(train_time_ms),
        "train_loss": float(train_loss_value),
        "tok_s": float(tok_s),
        "used_ema": bool(args.student_snapshot_use_ema and ema_state is not None),
        "param_tensors": int(len(snapshot_state)),
        "bytes": int(snapshot_path.stat().st_size),
    }
    if val_bpb is not None:
        meta["val_bpb"] = float(val_bpb)
    if quant_gap_bpb is not None:
        meta["quant_gap_bpb"] = float(quant_gap_bpb)
    if replay_buffer is not None:
        meta.update(replay_buffer.summary())
    runtime.write_snapshot_metadata(step, meta)
    return snapshot_path


def maybe_load_external_controller_decision(
    runtime: StudentSnapshotRuntime | None,
    args: Hyperparameters,
    *,
    step: int,
    last_mtime_ns: int,
) -> tuple[dict[str, object] | None, int]:
    if runtime is None or not args.external_controller_enabled:
        return None, last_mtime_ns
    refresh_every = max(int(args.external_controller_refresh_every), 1)
    if step <= 0 or step % refresh_every != 0:
        return None, last_mtime_ns
    decision_path = runtime.controller_decision_path
    if not decision_path.is_file():
        return None, last_mtime_ns
    try:
        stat = decision_path.stat()
    except OSError:
        return None, last_mtime_ns
    if stat.st_mtime_ns == last_mtime_ns:
        return None, last_mtime_ns
    payload = runtime.read_controller_decision()
    if payload is None:
        return None, last_mtime_ns
    return payload, int(stat.st_mtime_ns)


def apply_external_controller_decision(
    decision: dict[str, object],
    *,
    controller_state: AdaptiveTrainControllerState,
    train_loader,
    model: GPT | None = None,
) -> dict[str, object]:
    applied: dict[str, object] = {}
    min_step = int(decision.get("min_step", 0))
    max_step = int(decision.get("max_step", 0))
    applied["min_step"] = min_step
    applied["max_step"] = max_step
    sanitize_every = decision.get("sanitize_every")
    if sanitize_every is not None:
        controller_state.sanitize_every = max(int(sanitize_every), 1)
        controller_state.sanitize_stable_steps = 0
        applied["sanitize_every"] = int(controller_state.sanitize_every)
    distill_mult = decision.get("distill_weight_mult")
    if distill_mult is not None:
        controller_state.distill_weight_mult = max(float(distill_mult), 0.0)
        applied["distill_weight_mult"] = float(controller_state.distill_weight_mult)
    branch_mult = decision.get("branch_weight_mult")
    if branch_mult is not None:
        controller_state.branch_weight_mult = max(float(branch_mult), 0.0)
        applied["branch_weight_mult"] = float(controller_state.branch_weight_mult)
    branch_extra = decision.get("branch_extra_max_branches")
    if branch_extra is not None:
        controller_state.branch_extra_max_branches = max(int(branch_extra), 0)
        applied["branch_extra_max_branches"] = int(controller_state.branch_extra_max_branches)
    replay_mix_fraction = decision.get("replay_mix_fraction")
    if replay_mix_fraction is not None and hasattr(train_loader, "set_mix_fraction"):
        train_loader.set_mix_fraction(float(replay_mix_fraction))
        applied["replay_mix_fraction"] = float(train_loader.mix_fraction)
    hardmax_micro_steps = decision.get("hardmax_micro_steps")
    if hardmax_micro_steps is not None and model is not None and hasattr(model, "hardmax_struct_fast_refinement_steps"):
        model.hardmax_struct_fast_refinement_steps = max(int(hardmax_micro_steps), 1)
        applied["hardmax_micro_steps"] = int(model.hardmax_struct_fast_refinement_steps)
    decision_id = decision.get("decision_id")
    if decision_id is not None:
        applied["decision_id"] = str(decision_id)
    source = decision.get("source")
    if source is not None:
        applied["source"] = str(source)
    note = decision.get("note")
    if note is not None:
        applied["note"] = str(note)
    action_bits = [f"external:{applied.get('decision_id', 'anon')}"]
    if "sanitize_every" in applied:
        action_bits.append(f"sanitize->{applied['sanitize_every']}")
    if "replay_mix_fraction" in applied:
        action_bits.append(f"replay->{applied['replay_mix_fraction']:.3f}")
    if "hardmax_micro_steps" in applied:
        action_bits.append(f"hardmax_micro->{applied['hardmax_micro_steps']}")
    controller_state.last_action = ";".join(action_bits)
    return applied


def emit_replay_examples_for_batch(
    args: Hyperparameters,
    model: GPT,
    replay_buffer: FileReplayBuffer | None,
    x: mx.array,
    y: mx.array,
    x_np: np.ndarray,
    y_np: np.ndarray,
    operator_codes: mx.array | None,
    *,
    step: int,
    batch_metrics: dict[str, float | int] | None = None,
) -> int:
    if replay_buffer is None or args.replay_emit_topk <= 0:
        return 0
    logits = mx.stop_gradient(model.forward_logits(x, operator_codes).astype(mx.float32))
    nll = nn.losses.cross_entropy(
        logits,
        y,
        reduction="none",
    ).astype(mx.float32)
    mx.eval(nll)
    row_losses = np.mean(np.array(nll, dtype=np.float32, copy=False), axis=1, dtype=np.float64)
    if row_losses.size <= 0:
        return 0
    topk = min(max(args.replay_emit_topk, 0), int(row_losses.shape[0]))
    if topk <= 0:
        return 0
    candidate_rows = np.argsort(row_losses)[-topk:][::-1]
    examples: list[ReplayExample] = []
    safe_batch_metrics = batch_metrics or {}
    for row_idx in candidate_rows.tolist():
        row_loss = float(row_losses[row_idx])
        if row_loss < args.replay_emit_min_seq_loss:
            continue
        tokens = np.concatenate([x_np[row_idx], y_np[row_idx, -1:]], axis=0).astype(np.int32, copy=False)
        metadata: dict[str, float | int | str] = {"row_loss": row_loss}
        for key in (
            "mean_difficulty",
            "mean_operator_density",
            "mean_human_compressibility",
            "mean_compressibility",
            "mean_learnability",
            "mean_quality",
            "replay_frac",
        ):
            value = safe_batch_metrics.get(key)
            if value is None:
                continue
            metadata[key] = int(value) if isinstance(value, int) else float(value)
        examples.append(
            ReplayExample(
                tokens=tuple(int(token) for token in tokens.tolist()),
                score=row_loss,
                seq_len=args.train_seq_len,
                source_run_id=args.run_id,
                step=step,
                kind="student_row_nll",
                metadata=metadata,
            )
        )
    replay_buffer.append_examples(examples)
    return len(examples)

def loss_and_grad_chunked(
    args: Hyperparameters,
    model: GPT,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    *,
    logic_phase_enabled: bool = True,
    token_category_luts: TokenCategoryLuts | None = None,
    teacher_models: list[GPT] | None = None,
    structural_branching_cfg: StructuralBranchingConfig | None = None,
    structural_branch_budget_controller: StructuralBranchBudgetController | None = None,
    early_exit_budget_controller: EarlyExitBudgetController | None = None,
    structural_branching_active: bool = False,
    curriculum_phase_focus: str | None = None,
    polarity_seed_weight: float | None = None,
    polarity_sparse_weight: float | None = None,
    polarity_smooth_weight: float | None = None,
    focal_loss_weight: float = 0.0,
    focal_gamma: float = 2.0,
    focal_max_multiplier: float = 4.0,
    context_delta_config: ContextDeltaWeightingConfig | None = None,
    distill_weight: float = 0.0,
    teacher_hidden_distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
    collect_loss_components: bool = False,
    collect_timing: bool = False,
    replay_buffer: FileReplayBuffer | None = None,
    replay_emit_step: int | None = None,
    teacher_hidden_cache: FileTeacherHiddenCache | None = None,
    teacher_hidden_cache_write: bool = True,
    teacher_hidden_layer: int = -1,
) -> tuple[mx.array, dict, dict[str, float] | None, dict[str, float] | None, dict[str, float]]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    controller_totals = {
        "teacher_count": 0.0,
        "teacher_disagree_frac": 0.0,
        "teacher_unique_top1": 0.0,
        "teacher_pairwise_agree": 0.0,
        "branch_points": 0.0,
        "replay_emitted_examples": 0.0,
        "teacher_hidden_cache_hits": 0.0,
        "teacher_hidden_cache_rows": 0.0,
        "teacher_hidden_cache_hit_frac": 0.0,
        "teacher_hidden_cache_written": 0.0,
    }
    component_totals = {
        "ce_loss": 0.0,
        "seed_loss": 0.0,
        "sparse_loss": 0.0,
        "smooth_loss": 0.0,
        "aux_loss": 0.0,
        "early_exit_loss": 0.0,
        "prosody_aux_loss": 0.0,
        "prosody_token_class_loss": 0.0,
        "prosody_boundary_loss": 0.0,
        "prosody_punctuation_loss": 0.0,
        "prosody_quote_loss": 0.0,
        "distill_loss": 0.0,
        "hidden_distill_loss": 0.0,
        "branch_aux_loss": 0.0,
        "branch_rank_loss": 0.0,
        "branch_state_loss": 0.0,
        "hardmax_balance_loss": 0.0,
        "hardmax_diversity_loss": 0.0,
        "hardmax_confidence": 0.0,
        "hardmax_budget": 0.0,
        "hardmax_entropy": 0.0,
        "hardmax_pred_loss": 0.0,
        "hardmax_operator_loss": 0.0,
        "hardmax_token_class_loss": 0.0,
        "hardmax_boundary_loss": 0.0,
        "hardmax_quote_loss": 0.0,
        "hardmax_confidence_loss": 0.0,
        "early_exit_weight": 0.0,
        "branch_points": 0.0,
        "branch_budget": 0.0,
        "branch_min_miss": 0.0,
        "branch_max_gap": 0.0,
        "branch_top12_cos": 0.0,
        "branch_score": 0.0,
        "branch_enabled": 0.0,
        "teacher_count": 0.0,
        "teacher_disagree_frac": 0.0,
        "teacher_unique_top1": 0.0,
        "teacher_pairwise_agree": 0.0,
        "teacher_hidden_cache_hit_frac": 0.0,
        "teacher_hidden_cache_written": 0.0,
        "prosody_feature_density": 0.0,
        "prosody_reset_prior": 0.0,
        "prosody_slow_reset_prior": 0.0,
        "prosody_punctuation_frac": 0.0,
        "prosody_quote_frac": 0.0,
        "prosody_sentence_boundary_frac": 0.0,
        "prosody_paragraph_boundary_frac": 0.0,
        "prosody_state_delta_rms": 0.0,
    } if collect_loss_components else None
    timing_totals = {
        "batch_ms": 0.0,
        "teacher_ms": 0.0,
        "branch_ms": 0.0,
        "lossgrad_ms": 0.0,
        "component_ms": 0.0,
    } if collect_timing else None
    embedding_table_np = None
    if structural_branching_cfg is not None and structural_branching_active and structural_branching_cfg.enabled:
        embedding_table_np = np.asarray(mx.stop_gradient(model.tok_emb.weight.astype(mx.float32)))
    for chunk_tokens in chunk_sizes:
        batch_t0 = time.perf_counter()
        x_np, y_np = train_loader.next_batch_np(chunk_tokens, args.train_seq_len)
        operator_codes = operator_codes_mx_for_numpy_batch(
            model,
            x_np,
            enabled=(logic_phase_enabled or model.hardmax_structural_controller is not None),
        )
        token_weights = token_category_weights_mx(args, token_category_luts, x_np, y_np)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        if timing_totals is not None:
            timing_totals["batch_ms"] += 1000.0 * (time.perf_counter() - batch_t0)
        teacher_logits = None
        teacher_hidden = None
        teacher_metrics = None
        teacher_hidden_cache_rows = None
        teacher_hidden_cache_metrics = None
        teacher_hidden_cache_written = 0
        branch_plans = None
        batch_branch_cfg = structural_branching_cfg
        batch_early_exit_weight = args.early_exit_aux_weight if args.early_exit_aux_weight > 0.0 else None
        if teacher_models is not None:
            teacher_t0 = time.perf_counter()
            teacher_logits, teacher_metrics = ensemble_teacher_logits_for_batch(
                teacher_models,
                x,
                x_np,
                logic_phase_enabled=logic_phase_enabled,
                collect_metrics=collect_loss_components,
            )
            if timing_totals is not None:
                timing_totals["teacher_ms"] += 1000.0 * (time.perf_counter() - teacher_t0)
        if teacher_hidden_distill_weight > 0.0 and teacher_hidden_cache is not None:
            teacher_hidden_cache_rows, teacher_hidden_cache_metrics = lookup_teacher_hidden_cache_for_batch(
                teacher_hidden_cache,
                x_np,
                y_np,
            )
        if embedding_table_np is not None and structural_branching_cfg is not None:
            branch_t0 = time.perf_counter()
            batch_metrics = (
                train_loader.last_batch_metrics()
                if hasattr(train_loader, "last_batch_metrics")
                else {}
            )
            if structural_branch_budget_controller is not None and structural_branch_budget_controller.enabled:
                batch_branch_cfg = derive_structural_branching_config(
                    structural_branching_cfg,
                    StructuralBranchBudgetSignals(
                        phase_focus=curriculum_phase_focus,
                        mean_operator_density=(
                            float(batch_metrics["mean_operator_density"])
                            if "mean_operator_density" in batch_metrics
                            else None
                        ),
                        mean_human_compressibility=(
                            float(batch_metrics["mean_human_compressibility"])
                            if "mean_human_compressibility" in batch_metrics
                            else None
                        ),
                        mean_compressibility=(
                            float(batch_metrics["mean_compressibility"])
                            if "mean_compressibility" in batch_metrics
                            else None
                        ),
                    ),
                    structural_branch_budget_controller,
                )
            if early_exit_budget_controller is not None and early_exit_budget_controller.enabled and batch_early_exit_weight is not None:
                batch_early_exit_weight = derive_early_exit_aux_weight(
                    args.early_exit_aux_weight,
                    phase_focus=curriculum_phase_focus,
                    controller=early_exit_budget_controller,
                    mean_operator_density=(
                        float(batch_metrics["mean_operator_density"])
                        if "mean_operator_density" in batch_metrics
                        else None
                    ),
                    mean_human_compressibility=(
                        float(batch_metrics["mean_human_compressibility"])
                        if "mean_human_compressibility" in batch_metrics
                        else None
                    ),
                    mean_compressibility=(
                        float(batch_metrics["mean_compressibility"])
                        if "mean_compressibility" in batch_metrics
                        else None
                    ),
                )
            if component_totals is not None and batch_branch_cfg is not None:
                component_totals["branch_budget"] += float(batch_branch_cfg.max_branches)
                component_totals["branch_min_miss"] += float(batch_branch_cfg.min_structural_miss)
                component_totals["branch_max_gap"] += float(batch_branch_cfg.max_top1_gap)
                component_totals["branch_enabled"] += float(int(batch_branch_cfg.enabled))
            if timing_totals is not None:
                timing_totals["branch_ms"] += 1000.0 * (time.perf_counter() - branch_t0)
        elif early_exit_budget_controller is not None and early_exit_budget_controller.enabled and batch_early_exit_weight is not None:
            batch_metrics = (
                train_loader.last_batch_metrics()
                if hasattr(train_loader, "last_batch_metrics")
                else {}
            )
            batch_early_exit_weight = derive_early_exit_aux_weight(
                args.early_exit_aux_weight,
                phase_focus=curriculum_phase_focus,
                controller=early_exit_budget_controller,
                mean_operator_density=(
                    float(batch_metrics["mean_operator_density"])
                    if "mean_operator_density" in batch_metrics
                    else None
                ),
                mean_human_compressibility=(
                    float(batch_metrics["mean_human_compressibility"])
                    if "mean_human_compressibility" in batch_metrics
                    else None
                ),
                mean_compressibility=(
                    float(batch_metrics["mean_compressibility"])
                    if "mean_compressibility" in batch_metrics
                    else None
                ),
            )
        if component_totals is not None and batch_early_exit_weight is not None:
            component_totals["early_exit_weight"] += float(batch_early_exit_weight)
        if embedding_table_np is not None and batch_branch_cfg is not None and batch_branch_cfg.enabled:
            branch_logits = mx.stop_gradient(model.forward_logits(x, operator_codes).astype(mx.float32))
            branch_plans = select_structural_branch_points_np(
                np.asarray(branch_logits),
                y_np,
                embedding_table_np,
                batch_branch_cfg,
            )
        scale = float(y.size) / total_tokens
        if component_totals is not None and branch_plans is not None:
            flat_branch_plans = [plan for row in branch_plans for plan in row]
            component_totals["branch_points"] += float(len(flat_branch_plans))
            if flat_branch_plans:
                component_totals["branch_top12_cos"] += float(
                    sum(float(plan.top12_cosine) for plan in flat_branch_plans) / len(flat_branch_plans)
                )
                component_totals["branch_score"] += float(
                    sum(float(plan.score) for plan in flat_branch_plans) / len(flat_branch_plans)
                )
        if component_totals is not None and teacher_metrics is not None:
            component_totals["teacher_count"] += float(teacher_metrics["teacher_count"]) * scale
            component_totals["teacher_disagree_frac"] += float(teacher_metrics["teacher_disagree_frac"]) * scale
            component_totals["teacher_unique_top1"] += float(teacher_metrics["teacher_unique_top1"]) * scale
            component_totals["teacher_pairwise_agree"] += float(teacher_metrics["teacher_pairwise_agree"]) * scale
        if component_totals is not None:
            prosody_stats = model.prosody_runtime_stats(x)
            component_totals["prosody_feature_density"] += float(prosody_stats["prosody_feature_density"]) * scale
            component_totals["prosody_reset_prior"] += float(prosody_stats["prosody_reset_prior"]) * scale
            component_totals["prosody_slow_reset_prior"] += float(prosody_stats["prosody_slow_reset_prior"]) * scale
            component_totals["prosody_punctuation_frac"] += float(prosody_stats["prosody_punctuation_frac"]) * scale
            component_totals["prosody_quote_frac"] += float(prosody_stats["prosody_quote_frac"]) * scale
            component_totals["prosody_sentence_boundary_frac"] += float(prosody_stats["prosody_sentence_boundary_frac"]) * scale
            component_totals["prosody_paragraph_boundary_frac"] += float(prosody_stats["prosody_paragraph_boundary_frac"]) * scale
            component_totals["prosody_state_delta_rms"] += float(prosody_stats["prosody_state_delta_rms"]) * scale
        if teacher_metrics is not None:
            controller_totals["teacher_count"] += float(teacher_metrics["teacher_count"]) * scale
            controller_totals["teacher_disagree_frac"] += float(teacher_metrics["teacher_disagree_frac"]) * scale
            controller_totals["teacher_unique_top1"] += float(teacher_metrics["teacher_unique_top1"]) * scale
            controller_totals["teacher_pairwise_agree"] += float(teacher_metrics["teacher_pairwise_agree"]) * scale
        if teacher_hidden_cache_metrics is not None:
            controller_totals["teacher_hidden_cache_hits"] += float(teacher_hidden_cache_metrics["teacher_hidden_cache_hits"])
            controller_totals["teacher_hidden_cache_rows"] += float(teacher_hidden_cache_metrics["teacher_hidden_cache_rows"])
            controller_totals["teacher_hidden_cache_hit_frac"] += float(teacher_hidden_cache_metrics["teacher_hidden_cache_hit_frac"]) * scale
        if branch_plans is not None:
            controller_totals["branch_points"] += float(sum(len(plan) for plan in branch_plans)) * scale
        if replay_buffer is not None and replay_emit_step is not None:
            batch_metrics = (
                train_loader.last_batch_metrics()
                if hasattr(train_loader, "last_batch_metrics")
                else {}
            )
            emitted = emit_replay_examples_for_batch(
                args,
                model,
                replay_buffer,
                x,
                y,
                x_np,
                y_np,
                operator_codes,
                step=replay_emit_step,
                batch_metrics=batch_metrics,
            )
            controller_totals["replay_emitted_examples"] = controller_totals.get("replay_emitted_examples", 0.0) + float(emitted)
        if teacher_hidden_distill_weight > 0.0:
            needs_live_teacher_hidden = (
                teacher_models is not None
                and (
                    teacher_hidden_cache_rows is None
                    or any(row is None for row in teacher_hidden_cache_rows)
                )
            )
            live_teacher_hidden = None
            if needs_live_teacher_hidden:
                teacher_hidden_t0 = time.perf_counter()
                live_teacher_hidden = ensemble_teacher_hidden_for_batch(
                    teacher_models,
                    x,
                    x_np,
                    logic_phase_enabled=logic_phase_enabled,
                    hidden_layer=teacher_hidden_layer,
                )
                if timing_totals is not None:
                    timing_totals["teacher_ms"] += 1000.0 * (time.perf_counter() - teacher_hidden_t0)
            teacher_hidden, _hidden_miss_count = merge_teacher_hidden_rows(teacher_hidden_cache_rows, live_teacher_hidden)
            if teacher_hidden_cache is not None and teacher_hidden_cache_write and live_teacher_hidden is not None:
                written = append_teacher_hidden_cache_for_batch(
                    teacher_hidden_cache,
                    x_np,
                    y_np,
                    live_teacher_hidden,
                    step=replay_emit_step if replay_emit_step is not None else 0,
                    source_run_id=args.run_id,
                    cached_rows=teacher_hidden_cache_rows,
                )
                teacher_hidden_cache_written = int(written)
                controller_totals["teacher_hidden_cache_written"] += float(written)
        lossgrad_t0 = time.perf_counter()
        loss, grads = compiled_loss_and_grad(
            x,
            y,
            operator_codes,
            token_weights,
            teacher_logits,
            teacher_hidden,
            branch_plans,
            batch_branch_cfg,
            batch_early_exit_weight,
        )
        if timing_totals is not None:
            timing_totals["lossgrad_ms"] += 1000.0 * (time.perf_counter() - lossgrad_t0)
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if component_totals is not None:
            component_t0 = time.perf_counter()
            (
                _total_term,
                ce_term,
                seed_term,
                sparse_term,
                smooth_term,
                early_exit_term,
                prosody_aux_term,
                prosody_token_class_term,
                prosody_boundary_term,
                prosody_punctuation_term,
                prosody_quote_term,
                distill_term,
                hidden_distill_term,
                branch_rank_term,
                branch_state_term,
                hardmax_balance_term,
                hardmax_diversity_term,
                hardmax_confidence_term,
                hardmax_budget_term,
                hardmax_entropy_term,
                hardmax_pred_term,
                hardmax_operator_term,
                hardmax_token_class_term,
                hardmax_boundary_term,
                hardmax_quote_term,
                hardmax_confidence_loss_term,
            ) = model.loss_terms(
                x,
                y,
                operator_codes=operator_codes,
                polarity_seed_weight=polarity_seed_weight,
                polarity_sparse_weight=polarity_sparse_weight,
                polarity_smooth_weight=polarity_smooth_weight,
                focal_loss_weight=focal_loss_weight,
                focal_gamma=focal_gamma,
                focal_max_multiplier=focal_max_multiplier,
                token_weights=token_weights,
                context_delta_config=context_delta_config,
                teacher_logits=teacher_logits,
                teacher_hidden=teacher_hidden,
                ema_teacher_distill_weight=distill_weight,
                teacher_hidden_distill_weight=teacher_hidden_distill_weight,
                ema_teacher_temperature=distill_temperature,
                early_exit_aux_weight_override=batch_early_exit_weight,
                structural_branching_cfg=batch_branch_cfg if structural_branching_active else None,
                branch_plans=branch_plans,
            )
            mx.eval(
                ce_term,
                seed_term,
                sparse_term,
                smooth_term,
                early_exit_term,
                prosody_aux_term,
                prosody_token_class_term,
                prosody_boundary_term,
                prosody_punctuation_term,
                prosody_quote_term,
                distill_term,
                hidden_distill_term,
                branch_rank_term,
                branch_state_term,
                hardmax_balance_term,
                hardmax_diversity_term,
                hardmax_confidence_term,
                hardmax_budget_term,
                hardmax_entropy_term,
                hardmax_pred_term,
                hardmax_operator_term,
                hardmax_token_class_term,
                hardmax_boundary_term,
                hardmax_quote_term,
                hardmax_confidence_loss_term,
            )
            component_totals["ce_loss"] += float(ce_term.item()) * scale
            component_totals["seed_loss"] += float(seed_term.item()) * scale
            component_totals["sparse_loss"] += float(sparse_term.item()) * scale
            component_totals["smooth_loss"] += float(smooth_term.item()) * scale
            component_totals["early_exit_loss"] += float(early_exit_term.item()) * scale
            component_totals["prosody_aux_loss"] += float(prosody_aux_term.item()) * scale
            component_totals["prosody_token_class_loss"] += float(prosody_token_class_term.item()) * scale
            component_totals["prosody_boundary_loss"] += float(prosody_boundary_term.item()) * scale
            component_totals["prosody_punctuation_loss"] += float(prosody_punctuation_term.item()) * scale
            component_totals["prosody_quote_loss"] += float(prosody_quote_term.item()) * scale
            component_totals["distill_loss"] += float(distill_term.item()) * scale
            component_totals["hidden_distill_loss"] += float(hidden_distill_term.item()) * scale
            component_totals["branch_rank_loss"] += float(branch_rank_term.item()) * scale
            component_totals["branch_state_loss"] += float(branch_state_term.item()) * scale
            component_totals["hardmax_balance_loss"] += float(hardmax_balance_term.item()) * scale
            component_totals["hardmax_diversity_loss"] += float(hardmax_diversity_term.item()) * scale
            component_totals["hardmax_confidence"] += float(hardmax_confidence_term.item()) * scale
            component_totals["hardmax_budget"] += float(hardmax_budget_term.item()) * scale
            component_totals["hardmax_entropy"] += float(hardmax_entropy_term.item()) * scale
            component_totals["hardmax_pred_loss"] += float(hardmax_pred_term.item()) * scale
            component_totals["hardmax_operator_loss"] += float(hardmax_operator_term.item()) * scale
            component_totals["hardmax_token_class_loss"] += float(hardmax_token_class_term.item()) * scale
            component_totals["hardmax_boundary_loss"] += float(hardmax_boundary_term.item()) * scale
            component_totals["hardmax_quote_loss"] += float(hardmax_quote_term.item()) * scale
            component_totals["hardmax_confidence_loss"] += float(hardmax_confidence_loss_term.item()) * scale
            component_totals["branch_aux_loss"] += (
                float(branch_rank_term.item()) + float(branch_state_term.item())
            ) * scale
            component_totals["aux_loss"] += (
                float(early_exit_term.item())
                + float(prosody_aux_term.item())
                + float(hidden_distill_term.item())
                + float(branch_rank_term.item())
                + float(branch_state_term.item())
                + float(hardmax_balance_term.item())
                + float(hardmax_diversity_term.item())
                + float(hardmax_pred_term.item())
                + float(hardmax_confidence_loss_term.item())
            ) * scale
            if teacher_hidden_cache_metrics is not None:
                component_totals["teacher_hidden_cache_hit_frac"] += float(teacher_hidden_cache_metrics["teacher_hidden_cache_hit_frac"]) * scale
            component_totals["teacher_hidden_cache_written"] += float(teacher_hidden_cache_written)
            if timing_totals is not None:
                timing_totals["component_ms"] += 1000.0 * (time.perf_counter() - component_t0)
    if component_totals is not None and chunk_sizes:
        denom = float(len(chunk_sizes))
        component_totals["branch_points"] /= denom
        component_totals["branch_budget"] /= denom
        component_totals["branch_min_miss"] /= denom
        component_totals["branch_max_gap"] /= denom
        component_totals["branch_top12_cos"] /= denom
        component_totals["branch_score"] /= denom
        component_totals["branch_enabled"] /= denom
        component_totals["early_exit_weight"] /= denom
    return loss_value, tree_unflatten(list(grad_accum.items())), component_totals, timing_totals, controller_totals

def eval_val_non_overlapping(
    args: Hyperparameters,
    model: GPT,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    eval_seq_len = args.effective_eval_seq_len
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < eval_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one eval sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"EVAL_SEQ_LEN={eval_seq_len}"
        )
    val_batch_seqs = max(val_batch_tokens // eval_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // eval_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * eval_seq_len
        raw_end = batch_seq_end * eval_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, eval_seq_len)
        y_np = chunk[1:].reshape(-1, eval_seq_len)
        operator_codes = operator_codes_mx_for_numpy_batch(model, x_np)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y, operator_codes).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val_sliding(
    args: Hyperparameters,
    model: GPT,
    compiled_forward_logits,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    eval_seq_len = args.effective_eval_seq_len
    windows = build_sliding_eval_windows(val_tokens.size - 1, eval_seq_len, args.eval_stride)
    batch_seqs = args.effective_eval_batch_seqs
    total_batches = max((len(windows) + batch_seqs - 1) // batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    for batch_idx, batch_start in enumerate(range(0, len(windows), batch_seqs), start=1):
        batch_windows = windows[batch_start:batch_start + batch_seqs]
        x_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
        y_np = np.zeros((len(batch_windows), eval_seq_len), dtype=np.int32)
        for i, (window_start, window_len, _, _) in enumerate(batch_windows):
            chunk = val_tokens[window_start:window_start + window_len + 1]
            x_np[i, :window_len] = chunk[:-1]
            y_np[i, :window_len] = chunk[1:]

        operator_codes = operator_codes_mx_for_numpy_batch(model, x_np)
        logits = compiled_forward_logits(mx.array(x_np, dtype=mx.int32), operator_codes)
        nll = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            mx.array(y_np, dtype=mx.int32),
            reduction="none",
        ).astype(mx.float32)
        mx.eval(nll)
        nll_np = np.array(nll, dtype=np.float32, copy=False)

        for i, (_, _, score_start, score_end) in enumerate(batch_windows):
            scored_nll = nll_np[i, score_start:score_end]
            total_loss_sum += float(scored_nll.astype(np.float64).sum())
            tgt_ids = y_np[i, score_start:score_end]
            prev_ids = x_np[i, score_start:score_end]
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            total_tokens += float(score_end - score_start)
            total_bytes += float(bytes_np.astype(np.float64).sum())

        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"sliding_val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val(
    args: Hyperparameters,
    model: GPT,
    compiled_loss,
    compiled_forward_logits,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    if args.eval_stride > 0:
        if compiled_forward_logits is None:
            raise ValueError("compiled_forward_logits is required when eval_stride > 0")
        return eval_val_sliding(
            args,
            model,
            compiled_forward_logits,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log_fn,
        )
    return eval_val_non_overlapping(
        args,
        model,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log_fn,
    )

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        grad_np = np.nan_to_num(_np_float32(grad), nan=0.0, posinf=0.0, neginf=0.0)
        total_sq += float(np.sum(np.square(grad_np), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def empty_nonfinite_grad_summary() -> dict[str, object]:
    return {
        "tensor_count": 0,
        "param_count": 0,
        "nonfinite_tensors": 0,
        "nonfinite_params": 0,
        "top": [],
    }


def sanitize_grad_tree(grads_tree: dict, *, topk: int) -> tuple[dict, dict[str, object]]:
    flat = dict(tree_flatten(grads_tree))
    if not flat:
        return grads_tree, empty_nonfinite_grad_summary()
    summary = empty_nonfinite_grad_summary()
    sanitized: list[tuple[str, mx.array]] = []
    ranked: list[tuple[int, str]] = []
    any_nonfinite = False
    for name, grad in flat.items():
        grad_np = _np_float32(grad)
        finite_mask = np.isfinite(grad_np)
        nonfinite_count = int(grad_np.size - int(finite_mask.sum()))
        summary["tensor_count"] += 1
        summary["param_count"] += int(grad_np.size)
        if nonfinite_count > 0:
            any_nonfinite = True
            summary["nonfinite_tensors"] += 1
            summary["nonfinite_params"] += nonfinite_count
            ranked.append((nonfinite_count, name))
            clean_np = np.where(finite_mask, grad_np, 0.0).astype(grad_np.dtype, copy=False)
            sanitized.append((name, mx.array(clean_np, dtype=grad.dtype)))
        else:
            sanitized.append((name, grad))
    if not any_nonfinite:
        return grads_tree, summary
    ranked.sort(key=lambda item: item[0], reverse=True)
    summary["top"] = ranked[: max(int(topk), 0)]
    return tree_unflatten(sanitized), summary


def realize_grad_tree(grads_tree: dict) -> None:
    flat = dict(tree_flatten(grads_tree))
    if not flat:
        return
    mx.eval(*flat.values())


def maybe_mask_hardmax_structural_statebook_grads(
    flat_grads: dict[str, mx.array],
    args: Hyperparameters,
    step: int,
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    freeze_steps = max(int(getattr(args, "hardmax_struct_statebook_freeze_steps", 0)), 0)
    if freeze_steps <= 0 or step > freeze_steps:
        return flat_grads, ()
    target_name = "hardmax_structural_controller.state_book"
    grad = flat_grads.get(target_name)
    if grad is None:
        return flat_grads, ()
    masked = dict(flat_grads)
    masked[target_name] = mx.zeros(grad.shape, dtype=grad.dtype)
    return masked, (target_name,)


def should_sanitize_nonfinite_grads(
    args: Hyperparameters,
    step: int,
    train_loss_value: float,
    interval_override: int | None = None,
) -> bool:
    if not args.sanitize_nonfinite_grads:
        return False
    if not math.isfinite(train_loss_value):
        return True
    if step < max(args.sanitize_nonfinite_grads_always_first_steps, 0):
        return True
    interval = max(int(interval_override if interval_override is not None else args.sanitize_nonfinite_grads_every), 1)
    if interval <= 1:
        return True
    return (step + 1) % interval == 0


def resolve_logic_layer_index(requested_layer: int, num_layers: int) -> int | None:
    if requested_layer < 0:
        return num_layers // 3
    if requested_layer >= num_layers:
        raise ValueError(f"LOGIC_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


def resolve_polarity_detector_layer_index(requested_layer: int, num_layers: int) -> int | None:
    if requested_layer < 0:
        return num_layers // 2
    if requested_layer >= num_layers:
        raise ValueError(f"POLARITY_DETECTOR_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


def resolve_hardmax_structural_layer_index(requested_layer: int, num_layers: int) -> int | None:
    if requested_layer < 0:
        return num_layers // 3
    if requested_layer >= num_layers:
        raise ValueError(f"HARDMAX_STRUCT_LAYER_INDEX must be in [0, {num_layers - 1}], got {requested_layer}")
    return requested_layer


def resolve_hardmax_structural_router_start_layer(
    requested_layer: int,
    num_layers: int,
    controller_layer: int | None,
) -> int | None:
    if controller_layer is None:
        return None
    if requested_layer < 0:
        return min(num_layers - 1, controller_layer + 1)
    if requested_layer >= num_layers:
        raise ValueError(
            f"HARDMAX_STRUCT_ROUTER_START_LAYER must be in [0, {num_layers - 1}], got {requested_layer}"
        )
    if requested_layer < controller_layer:
        raise ValueError(
            f"HARDMAX_STRUCT_ROUTER_START_LAYER must be >= HARDMAX_STRUCT_LAYER_INDEX, got "
            f"{requested_layer} < {controller_layer}"
        )
    return requested_layer


def gpt_kwargs_from_args(
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None = None,
) -> dict[str, object]:
    needs_operator_routing = args.logic_dim > 0 or args.hardmax_struct_num_states > 0 or args.hardmax_struct_static_adapter
    operator_routing = build_operator_routing_spec(sp, args.vocab_size) if needs_operator_routing and sp is not None else None
    if needs_operator_routing and operator_routing is None:
        raise ValueError("logic or hardmax structural control requires tokenizer-backed operator routing")
    hardmax_layer_index = (
        resolve_hardmax_structural_layer_index(args.hardmax_struct_layer_index, args.num_layers)
        if (args.hardmax_struct_num_states > 0 or args.hardmax_struct_static_adapter) else None
    )
    return {
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "num_layer_templates": args.num_layer_templates,
        "dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "mlp_leaky_slope": args.mlp_leaky_slope,
        "tie_embeddings": args.tie_embeddings,
        "logit_chunk_tokens": args.logit_chunk_tokens,
        "logit_softcap": args.logit_softcap,
        "rope_base": args.rope_base,
        "tied_embed_init_std": args.tied_embed_init_std,
        "qk_gain_init": args.qk_gain_init,
        "num_registers": args.num_registers,
        "register_layout": args.register_layout,
        "register_stride": args.register_stride,
        "register_mask_mode": args.register_mask_mode,
        "logic_dim": args.logic_dim,
        "logic_layer_index": resolve_logic_layer_index(args.logic_layer_index, args.num_layers) if args.logic_dim > 0 else None,
        "logic_route_to_next_token": args.logic_route_to_next_token,
        "logic_operator_mode": args.logic_operator_mode,
        "polarity_detector_enabled": args.polarity_detector_enabled,
        "polarity_detector_layer_index": (
            resolve_polarity_detector_layer_index(args.polarity_detector_layer_index, args.num_layers)
            if args.polarity_detector_enabled else None
        ),
        "polarity_detector_hidden_dim": args.polarity_detector_hidden_dim,
        "polarity_seed_blend": args.polarity_seed_blend,
        "polarity_seed_weight": args.polarity_seed_weight,
        "polarity_sparse_weight": args.polarity_sparse_weight,
        "polarity_smooth_weight": args.polarity_smooth_weight,
        "hardmax_struct_num_states": args.hardmax_struct_num_states,
        "hardmax_struct_dim": args.hardmax_struct_dim,
        "hardmax_struct_static_adapter": args.hardmax_struct_static_adapter,
        "hardmax_struct_layer_index": hardmax_layer_index,
        "hardmax_struct_router_start_layer": (
            resolve_hardmax_structural_router_start_layer(
                args.hardmax_struct_router_start_layer,
                args.num_layers,
                hardmax_layer_index,
            )
            if (args.hardmax_struct_num_states > 0 or args.hardmax_struct_static_adapter) else None
        ),
        "hardmax_struct_temperature": args.hardmax_struct_temperature_start,
        "hardmax_struct_fast_refinement_steps": args.hardmax_struct_fast_refinement_steps,
        "hardmax_struct_compute_min_scale": args.hardmax_struct_compute_min_scale,
        "hardmax_struct_compute_power": args.hardmax_struct_compute_power,
        "hardmax_struct_route_residual_budget": args.hardmax_struct_route_residual_budget,
        "hardmax_struct_operator_prior_scale": args.hardmax_struct_operator_prior_scale,
        "hardmax_struct_reset_prior_scale": args.hardmax_struct_reset_prior_scale,
        "hardmax_struct_usage_balance_weight": args.hardmax_struct_usage_balance_weight,
        "hardmax_struct_diversity_weight": args.hardmax_struct_diversity_weight,
        "hardmax_struct_predict_weight": args.hardmax_struct_predict_weight,
        "hardmax_struct_confidence_weight": args.hardmax_struct_confidence_weight,
        "hardmax_struct_operator_weight": args.hardmax_struct_operator_weight,
        "hardmax_struct_token_class_weight": args.hardmax_struct_token_class_weight,
        "hardmax_struct_boundary_weight": args.hardmax_struct_boundary_weight,
        "hardmax_struct_quote_weight": args.hardmax_struct_quote_weight,
        "hardmax_struct_condition_mode": args.hardmax_struct_condition_mode,
        "hardmax_struct_attn_q_scale": args.hardmax_struct_attn_q_scale,
        "hardmax_struct_attn_tau_min": args.hardmax_struct_attn_tau_min,
        "prosody_type_embeddings_enabled": args.prosody_type_embeddings_enabled,
        "prosody_type_embedding_init_std": args.prosody_type_embedding_init_std,
        "prosody_extended_feature_set_enabled": args.prosody_extended_feature_set_enabled,
        "prosody_feature_embeddings_enabled": args.prosody_feature_embeddings_enabled,
        "prosody_feature_embedding_init_std": args.prosody_feature_embedding_init_std,
        "prosody_state_adapter_enabled": args.prosody_state_adapter_enabled,
        "prosody_state_dim": args.prosody_state_dim,
        "prosody_state_init_std": args.prosody_state_init_std,
        "prosody_state_scale": args.prosody_state_scale,
        "prosody_state_reset_prior_weight": args.prosody_state_reset_prior_weight,
        "prosody_state_hierarchical_enabled": args.prosody_state_hierarchical_enabled,
        "prosody_state_slow_reset_scale": args.prosody_state_slow_reset_scale,
        "prosody_aux_layer_index": args.prosody_aux_layer_index,
        "prosody_aux_weight": args.prosody_aux_weight,
        "prosody_aux_head_init_std": args.prosody_aux_head_init_std,
        "prosody_aux_token_class_weight": args.prosody_aux_token_class_weight,
        "prosody_aux_boundary_weight": args.prosody_aux_boundary_weight,
        "prosody_aux_quote_weight": args.prosody_aux_quote_weight,
        "prosody_aux_punctuation_weight": args.prosody_aux_punctuation_weight,
        "token_prosody_luts": None if sp is None else build_token_prosody_luts(
            sp,
            extended_binary_features=args.prosody_extended_feature_set_enabled,
        ),
        "operator_routing": operator_routing,
    }


def make_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> "GPT":
    return GPT(
        **gpt_kwargs_from_args(args, sp),
        early_exit_layer_index=args.early_exit_layer_index,
        early_exit_horizons=parse_horizons(args.early_exit_horizons),
        early_exit_aux_weight=args.early_exit_aux_weight,
        early_exit_head_init_std=args.early_exit_head_init_std,
        early_exit_cascaded_enabled=args.early_exit_cascaded_enabled,
        early_exit_condition_init_std=args.early_exit_condition_init_std,
        early_exit_branch_draft_enabled=args.early_exit_branch_draft_enabled,
        early_exit_branch_conf_threshold=args.early_exit_branch_conf_threshold,
        early_exit_branch_max_draft_tokens=args.early_exit_branch_max_draft_tokens,
    )


def flat_parameter_state(model: nn.Module) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def apply_flat_arrays(model: nn.Module, flat_state: dict[str, mx.array]) -> None:
    model.update(tree_unflatten(list(flat_state.items())))


def summarize_name_numel(items: list[tuple[str, int]], *, topk: int) -> str:
    if not items or topk <= 0:
        return ""
    ranked = sorted(items, key=lambda item: item[1], reverse=True)[:topk]
    return ",".join(f"{name}:{numel}" for name, numel in ranked)


def select_compatible_flat_state(
    model: nn.Module,
    checkpoint_state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, object]]:
    reference = flat_parameter_state(model)
    matched: dict[str, mx.array] = {}
    missing: list[tuple[str, int]] = []
    mismatched: list[tuple[str, int]] = []
    unexpected: list[tuple[str, int]] = []
    matched_params = 0
    total_params = 0
    for name, ref_value in reference.items():
        numel = int(np.prod(ref_value.shape))
        total_params += numel
        checkpoint_value = checkpoint_state.get(name)
        if checkpoint_value is None:
            missing.append((name, numel))
            continue
        if tuple(checkpoint_value.shape) != tuple(ref_value.shape):
            mismatched.append((name, numel))
            continue
        matched[name] = checkpoint_value.astype(ref_value.dtype)
        matched_params += numel
    for name, value in checkpoint_state.items():
        if name in reference:
            continue
        unexpected.append((name, int(np.prod(value.shape))))
    stats = {
        "matched_tensors": len(matched),
        "matched_params": matched_params,
        "total_tensors": len(reference),
        "total_params": total_params,
        "matched_param_fraction": matched_params / max(total_params, 1),
        "missing_tensors": len(missing),
        "mismatched_tensors": len(mismatched),
        "unexpected_tensors": len(unexpected),
        "missing_top": missing,
        "mismatched_top": mismatched,
        "unexpected_top": unexpected,
    }
    return matched, stats


def load_flat_npz_state(path: Path) -> dict[str, mx.array]:
    return {name: value for name, value in mx.load(str(path)).items()}


def select_hardmax_structural_init_state(
    model: nn.Module,
    checkpoint_state: dict[str, mx.array],
    *,
    mode: str,
) -> tuple[dict[str, mx.array], dict[str, object]]:
    reference = flat_parameter_state(model)
    prefix = "hardmax_structural_controller."
    target_state = {name: value for name, value in reference.items() if name.startswith(prefix)}
    checkpoint_controller = {name: value for name, value in checkpoint_state.items() if name.startswith(prefix)}
    if not target_state:
        raise ValueError("Model does not expose hardmax_structural_controller parameters for initialization.")

    normalized_mode = mode.strip().lower() or "core"
    allowed_suffixes: set[str] | None
    if normalized_mode == "full":
        allowed_suffixes = None
    elif normalized_mode == "state_book":
        allowed_suffixes = {"state_book"}
    elif normalized_mode == "core":
        allowed_suffixes = {
            "recur.weight",
            "state_logits.weight",
            "state_book",
        }
    else:
        raise ValueError(f"Unsupported HARDMAX_STRUCT_INIT_MODE {mode!r}; expected one of: full, state_book, core.")

    matched: dict[str, mx.array] = {}
    missing: list[tuple[str, int]] = []
    mismatched: list[tuple[str, int]] = []
    unexpected: list[tuple[str, int]] = []
    matched_params = 0
    total_target_params = 0

    for name, ref_value in target_state.items():
        suffix = name[len(prefix) :]
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        numel = int(np.prod(ref_value.shape))
        total_target_params += numel
        checkpoint_value = checkpoint_controller.get(name)
        if checkpoint_value is None:
            missing.append((name, numel))
            continue
        if tuple(checkpoint_value.shape) != tuple(ref_value.shape):
            mismatched.append((name, numel))
            continue
        matched[name] = checkpoint_value.astype(ref_value.dtype)
        matched_params += numel

    for name, value in checkpoint_controller.items():
        suffix = name[len(prefix) :]
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        if name in matched or name in target_state:
            continue
        unexpected.append((name, int(np.prod(value.shape))))

    stats = {
        "mode": normalized_mode,
        "matched_tensors": len(matched),
        "matched_params": matched_params,
        "total_target_tensors": (
            len(target_state)
            if allowed_suffixes is None
            else sum(1 for name in target_state if name[len(prefix) :] in allowed_suffixes)
        ),
        "total_target_params": total_target_params,
        "coverage_frac": matched_params / max(total_target_params, 1),
        "missing_names": [name for name, _ in missing],
        "mismatched_names": [name for name, _ in mismatched],
        "unexpected_names": [name for name, _ in unexpected],
        "missing_top": missing,
        "mismatched_top": mismatched,
        "unexpected_top": unexpected,
    }
    return matched, stats


def maybe_initialize_hardmax_structural_controller(
    model: nn.Module,
    args: Hyperparameters,
    *,
    log_fn=print,
) -> dict[str, object] | None:
    init_path_raw = args.hardmax_struct_init_path.strip()
    if not init_path_raw:
        return None
    controller = getattr(model, "hardmax_structural_controller", None)
    if controller is None:
        raise ValueError(
            "HARDMAX_STRUCT_INIT_PATH was provided, but the instantiated model has no hardmax_structural_controller."
        )
    init_path = Path(init_path_raw).expanduser()
    if not init_path.is_absolute():
        init_path = (Path.cwd() / init_path).resolve()
    if init_path.suffix.lower() != ".npz":
        raise ValueError(f"HARDMAX_STRUCT_INIT_PATH must point to an .npz file, got {init_path}")
    if not init_path.is_file():
        raise FileNotFoundError(f"Hardmax structural init artifact not found: {init_path}")
    checkpoint_state = load_flat_npz_state(init_path)
    selected_state, stats = select_hardmax_structural_init_state(
        model,
        checkpoint_state,
        mode=args.hardmax_struct_init_mode,
    )
    if not selected_state:
        raise ValueError(
            "Hardmax structural init matched zero tensors. "
            f"path={init_path} mode={stats['mode']} missing={len(stats['missing_names'])} "
            f"mismatched={len(stats['mismatched_names'])}"
        )
    apply_flat_arrays(model, selected_state)
    if hasattr(model, "clear_turbo_cache"):
        model.clear_turbo_cache()
    log_fn(
        "hardmax_struct_init:"
        f"applied path:{init_path} mode:{stats['mode']} "
        f"matched_tensors:{stats['matched_tensors']} "
        f"coverage:{float(stats['coverage_frac']):.3f} "
        f"missing:{len(stats['missing_names'])} mismatched:{len(stats['mismatched_names'])}"
    )
    return {
        "path": str(init_path),
        **stats,
    }


def init_ema_state(model: nn.Module) -> dict[str, mx.array]:
    return {name: value.astype(mx.float32) for name, value in flat_parameter_state(model).items()}


def update_ema_state(ema_state: dict[str, mx.array], model: nn.Module, decay: float) -> None:
    current_state = flat_parameter_state(model)
    one_minus_decay = 1.0 - decay
    updated = {
        name: ema_state[name] * decay + current_state[name].astype(mx.float32) * one_minus_decay
        for name in ema_state
    }
    mx.eval(*updated.values())
    ema_state.update(updated)


def build_ema_group_map(opt: "SplitOptimizers", model: nn.Module) -> dict[str, str]:
    group_map: dict[str, str] = {}
    for name in opt.matrix_keys:
        group_map[name] = "matrix"
    for name in opt.turbo_matrix_keys:
        group_map[name] = "turbo"
    for name in opt.scalar_keys:
        group_map[name] = "scalar"
    for name in flat_parameter_state(model):
        group_map.setdefault(name, "scalar")
    return group_map


def update_ema_state_grouped(
    ema_state: dict[str, mx.array],
    model: nn.Module,
    decay_by_group: dict[str, float],
    group_by_name: dict[str, str],
    default_decay: float,
) -> None:
    current_state = flat_parameter_state(model)
    updated: dict[str, mx.array] = {}
    for name in ema_state:
        decay = decay_by_group.get(group_by_name.get(name, "scalar"), default_decay)
        one_minus_decay = 1.0 - decay
        updated[name] = ema_state[name] * decay + current_state[name].astype(mx.float32) * one_minus_decay
    mx.eval(*updated.values())
    ema_state.update(updated)


def cast_ema_like(ema_state: dict[str, mx.array], reference_state: dict[str, mx.array]) -> dict[str, mx.array]:
    return {name: ema_state[name].astype(reference_state[name].dtype) for name in ema_state}


def ema_active_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> bool:
    if not args.ema_enabled:
        return False
    frac = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    return frac >= args.ema_start_frac


def ema_teacher_active_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> bool:
    if not args.ema_teacher_distill_enabled or args.ema_teacher_distill_weight <= 0.0:
        return False
    frac = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    return frac >= args.ema_teacher_start_frac


def external_teacher_active_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> bool:
    if args.external_teacher_distill_weight <= 0.0:
        return False
    if not args.external_teacher_config_paths or not args.external_teacher_checkpoint_paths:
        return False
    frac = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    return frac >= args.external_teacher_start_frac


def structural_branching_active_for_progress(
    config: StructuralBranchingConfig,
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> bool:
    if not config.enabled or config.weight <= 0.0:
        return False
    frac = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    return frac >= config.start_frac


def ema_gap_metrics(ema_state: dict[str, mx.array] | None, model: nn.Module) -> dict[str, float]:
    if not ema_state:
        return {"rmse": 0.0, "mae": 0.0}
    current_state = flat_parameter_state(model)
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0.0
    for name, ema_value in ema_state.items():
        current_np = np.asarray(current_state[name].astype(mx.float32), dtype=np.float32)
        ema_np = np.asarray(ema_value.astype(mx.float32), dtype=np.float32)
        diff = current_np - ema_np
        sum_sq += float(np.square(diff, dtype=np.float64).sum())
        sum_abs += float(np.abs(diff, dtype=np.float64).sum())
        count += float(diff.size)
    denom = max(count, 1.0)
    return {
        "rmse": math.sqrt(sum_sq / denom),
        "mae": sum_abs / denom,
    }


def gradient_group_cosines(
    flat_grads: dict[str, mx.array],
    prev_flat_grads: dict[str, mx.array] | None,
    opt: "SplitOptimizers",
) -> dict[str, float]:
    group_names = {
        "matrix": opt.matrix_keys,
        "turbo": opt.turbo_matrix_keys,
        "scalar": opt.scalar_keys,
    }
    cosines = {name: 0.0 for name in group_names}
    if prev_flat_grads is None:
        return cosines
    for group_name, keys in group_names.items():
        dot = 0.0
        grad_sq = 0.0
        prev_sq = 0.0
        for name in keys:
            grad = flat_grads.get(name)
            prev = prev_flat_grads.get(name)
            if grad is None or prev is None:
                continue
            grad32 = grad.astype(mx.float32)
            prev32 = prev.astype(mx.float32)
            dot += float((grad32 * prev32).sum().item())
            grad_sq += float((grad32 * grad32).sum().item())
            prev_sq += float((prev32 * prev32).sum().item())
        if grad_sq > 0.0 and prev_sq > 0.0:
            cosines[group_name] = dot / max(math.sqrt(grad_sq * prev_sq), 1e-12)
    return cosines


def tensor_activity_snapshot(
    flat_grads: dict[str, mx.array],
    *,
    hot_threshold: float,
    warm_threshold: float,
    nonzero_threshold: float,
    topk: int,
) -> dict[str, object]:
    summary = {
        "tensor_count": 0,
        "param_count": 0,
        "hot_tensors": 0,
        "warm_tensors": 0,
        "cold_tensors": 0,
        "hot_params": 0,
        "warm_params": 0,
        "cold_params": 0,
        "top": [],
    }
    ranked: list[tuple[float, float, int, str, str]] = []
    for name, grad in flat_grads.items():
        grad32 = grad.astype(mx.float32)
        abs_grad = mx.abs(grad32)
        mean_abs = float(mx.mean(abs_grad).item())
        nonzero_frac = float(mx.mean((abs_grad > nonzero_threshold).astype(mx.float32)).item())
        numel = int(grad.size)
        if mean_abs >= hot_threshold:
            bucket = "hot"
        elif mean_abs >= warm_threshold:
            bucket = "warm"
        else:
            bucket = "cold"
        summary["tensor_count"] += 1
        summary["param_count"] += numel
        summary[f"{bucket}_tensors"] += 1
        summary[f"{bucket}_params"] += numel
        ranked.append((mean_abs, nonzero_frac, numel, name, bucket))
    ranked.sort(key=lambda item: item[0], reverse=True)
    summary["top"] = ranked[: max(int(topk), 0)]
    return summary


def logic_gate_metrics(model: nn.Module) -> dict[str, float]:
    logic_sidecar = getattr(model, "logic_sidecar", None)
    if logic_sidecar is None:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0}
    gate_np = np.asarray(mx.tanh(logic_sidecar.gate).astype(mx.float32), dtype=np.float32)
    if gate_np.size == 0:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0}
    abs_gate = np.abs(gate_np, dtype=np.float32)
    return {
        "mean_abs": float(abs_gate.mean(dtype=np.float64)),
        "rms": float(np.sqrt(np.square(gate_np, dtype=np.float64).mean())),
        "max_abs": float(abs_gate.max(initial=0.0)),
    }


def hardmax_structural_metrics(model: nn.Module) -> dict[str, float]:
    controller = getattr(model, "hardmax_structural_controller", None)
    if controller is None:
        return {"gate_rms": 0.0, "gate_max_abs": 0.0, "book_offdiag_cos": 0.0, "temperature": 0.0}
    gate_np = np.asarray(mx.tanh(controller.gate).astype(mx.float32), dtype=np.float32)
    book_np = np.asarray(controller.state_book.astype(mx.float32), dtype=np.float32)
    if book_np.size <= 0:
        offdiag_cos = 0.0
    else:
        norms = np.linalg.norm(book_np, axis=-1, keepdims=True)
        norm_book = book_np / np.maximum(norms, 1.0e-6)
        cosine = norm_book @ norm_book.T
        mask = np.ones_like(cosine, dtype=np.float32) - np.eye(cosine.shape[0], dtype=np.float32)
        denom = max(float(mask.sum(dtype=np.float64)), 1.0)
        offdiag_cos = float(np.sum(np.abs(cosine * mask), dtype=np.float64) / denom)
    return {
        "gate_rms": float(np.sqrt(np.square(gate_np, dtype=np.float64).mean())) if gate_np.size > 0 else 0.0,
        "gate_max_abs": float(np.abs(gate_np, dtype=np.float32).max(initial=0.0)) if gate_np.size > 0 else 0.0,
        "book_offdiag_cos": offdiag_cos,
        "temperature": float(getattr(controller, "temperature", 0.0)),
    }


def resolve_mlx_compile(compile_mode: str, turbo_qat: bool) -> bool:
    mode = compile_mode.strip().lower()
    if mode == "auto":
        return not turbo_qat
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"MLX_COMPILE must be auto/0/1/true/false, got {compile_mode!r}")


def training_progress_fraction(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    step_frac = step_idx / max(args.iterations, 1) if args.iterations > 0 else 0.0
    if max_wallclock_ms is None or max_wallclock_ms <= 0:
        return min(max(step_frac, 0.0), 1.0)
    time_frac = train_time_ms / max(max_wallclock_ms, 1.0)
    return min(max(step_frac, time_frac, 0.0), 1.0)


def hardmax_structural_temperature_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    start = float(args.hardmax_struct_temperature_start)
    end = float(args.hardmax_struct_temperature_end)
    anneal_frac = max(float(args.hardmax_struct_temperature_anneal_frac), 0.0)
    if anneal_frac <= 0.0:
        return end
    progress = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    ramp = min(max(progress / max(anneal_frac, 1.0e-8), 0.0), 1.0)
    return start + (end - start) * ramp


def turbo_qat_scale_for_progress(
    args: Hyperparameters,
    step_idx: int,
    train_time_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    if not args.turbo_qat:
        return 0.0
    frac = training_progress_fraction(args, step_idx, train_time_ms, max_wallclock_ms)
    if frac < args.turbo_qat_start_frac:
        return 0.0
    if args.turbo_qat_ramp_frac <= 0:
        return 1.0
    return min((frac - args.turbo_qat_start_frac) / args.turbo_qat_ramp_frac, 1.0)

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    token_category_luts = build_token_category_luts(sp) if args.curriculum_token_category_weighting else None
    context_delta_cfg = context_delta_weighting_config(args)
    structural_branch_cfg = structural_branching_config(args)
    structural_branch_budget_ctrl = structural_branch_budget_controller(args)
    early_exit_budget_ctrl = early_exit_budget_controller(args)
    adaptive_ctrl_cfg = adaptive_train_controller_config(args)
    adaptive_ctrl_state = init_adaptive_train_controller_state(args, adaptive_ctrl_cfg)
    if (
        args.ema_teacher_distill_enabled
        and args.ema_teacher_distill_weight > 0.0
        and args.external_teacher_distill_weight > 0.0
    ):
        raise ValueError("EMA teacher distillation and external teacher distillation cannot both be enabled yet")
    if args.register_layout == "interleaved" and context_delta_cfg.enabled:
        raise ValueError("CURRICULUM_CONTEXT_DELTA_WEIGHTING does not support REGISTER_LAYOUT=interleaved")
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_seq_len = max(args.train_seq_len, args.effective_eval_seq_len)
    val_tokens = limit_validation_tokens(
        load_validation_tokens(args.val_files, val_seq_len),
        args.effective_eval_seq_len,
        args.val_max_seqs,
    )
    quant_eval_tokens = limit_validation_tokens(val_tokens, args.effective_eval_seq_len, args.quant_eval_max_seqs)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    mx.random.seed(args.seed)

    train_loader = build_train_loader(args, log_fn=log, dataset_name=dataset_name)
    replay_buffer = getattr(train_loader, "replay_buffer", None)
    snapshot_runtime = (
        StudentSnapshotRuntime(
            Path(args.student_snapshot_dir),
            run_id=args.run_id,
            keep_last=args.student_snapshot_keep_last,
        )
        if args.student_snapshot_dir
        else None
    )
    external_controller_mtime_ns = -1

    model = make_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    hardmax_init_stats = maybe_initialize_hardmax_structural_controller(model, args, log_fn=log)
    opt = SplitOptimizers(model, args)
    ema_group_map = build_ema_group_map(opt, model)
    ema_state = init_ema_state(model) if args.ema_enabled and args.ema_start_frac <= 0.0 else None
    ema_teacher_state = init_ema_state(model) if args.ema_teacher_distill_enabled else None
    ema_teacher_model: GPT | None = make_gpt(args, sp) if args.ema_teacher_distill_enabled else None
    if ema_teacher_model is not None:
        ema_teacher_model.set_turbo_qat(False, 0.0)
        if ema_teacher_state is not None:
            apply_flat_arrays(ema_teacher_model, cast_ema_like(ema_teacher_state, flat_parameter_state(ema_teacher_model)))
            ema_teacher_model.clear_turbo_cache()
    external_teacher_models: list[GPT] = []
    external_teacher_meta: list[dict[str, object]] = []
    if args.external_teacher_config_paths or args.external_teacher_checkpoint_paths:
        external_teacher_models, external_teacher_meta = load_external_teacher_models(args, sp, log_fn=log)
    if args.external_teacher_hidden_distill_weight > 0.0:
        for meta in external_teacher_meta:
            if int(meta["dim"]) != int(args.model_dim):
                raise ValueError(
                    "EXTERNAL_TEACHER_HIDDEN_DISTILL_WEIGHT requires teacher MODEL_DIM to match the student MODEL_DIM. "
                    f"student={args.model_dim} teacher={meta['dim']}"
                )
    teacher_hidden_cache = (
        FileTeacherHiddenCache(
            Path(args.external_teacher_hidden_cache_dir),
            layer_index=args.external_teacher_hidden_layer,
            hidden_dim=args.model_dim,
            max_entries=args.external_teacher_hidden_cache_max_entries,
            log_fn=log,
        )
        if args.external_teacher_hidden_cache_dir
        else None
    )
    quant_eval_model: GPT | None = None
    compiled_quant_loss = None
    compiled_quant_forward_logits = None

    compiled = resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    uses_logic = model.logic_sidecar is not None or model.hardmax_structural_controller is not None
    if compiled and model.hardmax_structural_controller is not None:
        log("mlx_compile:disable reason:hardmax_structural_controller_python_loop")
        compiled = False
    if compiled:
        if uses_logic:
            compiled_loss_impl = mx.compile(
                lambda x, y, operator_codes: model.ce_loss(x, y, operator_codes),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_forward_logits_impl = mx.compile(
                lambda x, operator_codes: model.forward_logits(x, operator_codes),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y, operator_codes: model.loss_terms(x, y, operator_codes)[0]),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss = lambda x, y, operator_codes=None: compiled_loss_impl(x, y, operator_codes)
            compiled_forward_logits = lambda x, operator_codes=None: compiled_forward_logits_impl(x, operator_codes)
            compiled_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: compiled_loss_and_grad_impl(x, y, operator_codes)
        else:
            compiled_loss_impl = mx.compile(lambda x, y: model.ce_loss(x, y), inputs=model.state, outputs=model.state)
            compiled_forward_logits_impl = mx.compile(lambda x: model.forward_logits(x), inputs=model.state, outputs=model.state)
            compiled_loss_and_grad_impl = mx.compile(
                nn.value_and_grad(model, lambda x, y: model.loss_terms(x, y)[0]),
                inputs=model.state,
                outputs=model.state,
            )
            compiled_loss = lambda x, y, operator_codes=None: compiled_loss_impl(x, y)
            compiled_forward_logits = lambda x, operator_codes=None: compiled_forward_logits_impl(x)
            compiled_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: compiled_loss_and_grad_impl(x, y)
    else:
        if uses_logic:
            compiled_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y, operator_codes)
            compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits(x, operator_codes)
            compiled_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y, operator_codes: model.loss_terms(x, y, operator_codes)[0])
            compiled_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: compiled_loss_and_grad_impl(x, y, operator_codes)
        else:
            compiled_loss = lambda x, y, operator_codes=None: model.ce_loss(x, y)
            compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits(x)
            compiled_loss_and_grad_impl = nn.value_and_grad(model, lambda x, y: model.loss_terms(x, y)[0])
            compiled_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: compiled_loss_and_grad_impl(x, y)

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"layer_templates:{args.num_layer_templates} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} eval_seq_len:{args.effective_eval_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"logic_registers:registers:{args.num_registers} layout:{args.register_layout} stride:{args.register_stride} "
        f"logic_dim:{args.logic_dim} "
        f"logic_layer:{model.logic_layer_index if model.logic_sidecar is not None else 'off'} "
        f"logic_route_to_next:{args.logic_route_to_next_token} "
        f"register_mask_mode:{args.register_mask_mode} logic_operator_mode:{args.logic_operator_mode}"
    )
    log(
        f"polarity_detector:enabled:{int(args.polarity_detector_enabled)} "
        f"layer:{model.polarity_detector_layer_index if model.polarity_detector is not None else 'off'} "
        f"hidden_dim:{args.polarity_detector_hidden_dim} seed_blend:{args.polarity_seed_blend:.3f} "
        f"seed_weight:{args.polarity_seed_weight:.6f} sparse_weight:{args.polarity_sparse_weight:.6f} "
        f"smooth_weight:{args.polarity_smooth_weight:.6f}"
    )
    log(
        f"hardmax_struct:states:{args.hardmax_struct_num_states} dim:{args.hardmax_struct_dim} "
        f"static_adapter:{int(args.hardmax_struct_static_adapter)} "
        f"layer:{model.hardmax_struct_layer_index if model.hardmax_structural_controller is not None else 'off'} "
        f"router_start:{model.hardmax_struct_router_start_layer if model.hardmax_structural_controller is not None else 'off'} "
        f"route_budget:{int(bool(args.hardmax_struct_route_residual_budget))} "
        f"condition_mode:{args.hardmax_struct_condition_mode} "
        f"attn_q_scale:{args.hardmax_struct_attn_q_scale:.3f} "
        f"attn_tau_min:{args.hardmax_struct_attn_tau_min:.3f} "
        f"temp_start:{args.hardmax_struct_temperature_start:.3f} "
        f"temp_end:{args.hardmax_struct_temperature_end:.3f} "
        f"temp_anneal:{args.hardmax_struct_temperature_anneal_frac:.3f} "
        f"micro_steps:{args.hardmax_struct_fast_refinement_steps} "
        f"compute_min:{args.hardmax_struct_compute_min_scale:.3f} "
        f"compute_power:{args.hardmax_struct_compute_power:.3f} "
        f"init_mode:{args.hardmax_struct_init_mode or 'off'} "
        f"init_path:{args.hardmax_struct_init_path or 'off'} "
        f"statebook_freeze:{args.hardmax_struct_statebook_freeze_steps} "
        f"usage_w:{args.hardmax_struct_usage_balance_weight:.6f} div_w:{args.hardmax_struct_diversity_weight:.6f} "
        f"pred_w:{args.hardmax_struct_predict_weight:.6f} conf_w:{args.hardmax_struct_confidence_weight:.6f}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} val_seqs:{(val_tokens.size - 1) // args.effective_eval_seq_len} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f} "
        f"warmdown_iters:{args.warmdown_iters} warmdown_fraction:{args.warmdown_fraction:.3f}"
    )
    if args.curriculum_enabled:
        summary = train_loader.summary() if hasattr(train_loader, "summary") else {}
        log(
            f"curriculum:enabled features:{args.curriculum_features_path} "
            f"phase_plan:{args.curriculum_phase_plan_path or 'default'} "
            f"apply_logic_gate:{int(args.curriculum_apply_logic_phase_gating)} "
            f"apply_qat_gate:{int(args.curriculum_apply_qat_phase_gating)} "
            f"apply_ema_gate:{int(args.curriculum_apply_ema_phase_gating)} "
            f"apply_focal:{int(args.curriculum_apply_focal)} "
            f"focal_gamma:{args.curriculum_focal_gamma:.3f} "
            f"focal_max_multiplier:{args.curriculum_focal_max_multiplier:.3f} "
            f"min_compressibility:{args.curriculum_min_compressibility:.3f} "
            f"num_chunks:{summary.get('num_chunks', 'unknown')} "
            f"repeatable_chunks:{summary.get('repeatable_chunks', 'unknown')} "
            f"skipped_chunks:{summary.get('skipped_chunks', 'unknown')} "
            f"filtered_low_compressibility_chunks:{summary.get('filtered_low_compressibility_chunks', 'unknown')}"
        )
    elif args.sequential_min_compressibility >= 0.0:
        summary = train_loader.summary() if hasattr(train_loader, "summary") else {}
        log(
            f"sequential_filter:enabled min_compressibility:{args.sequential_min_compressibility:.3f} "
            f"seen_chunks:{summary.get('seen_chunks', 0)} "
            f"kept_chunks:{summary.get('kept_chunks', 0)} "
            f"skipped_low_compressibility_chunks:{summary.get('skipped_low_compressibility_chunks', 0)}"
        )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:{opt.matrix_optimizer_name}+adam matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps} "
        f"apollo_rank:{args.apollo_rank} apollo_scale:{args.apollo_scale} "
        f"apollo_scale_type:{args.apollo_scale_type} apollo_proj_gap:{args.apollo_proj_gap} "
        f"apollo_beta1:{args.apollo_beta1} apollo_beta2:{args.apollo_beta2} apollo_eps:{args.apollo_eps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    if args.eval_stride > 0:
        log(
            f"eval_mode:sliding_window seq_len:{args.effective_eval_seq_len} stride:{args.eval_stride} "
            f"batch_seqs:{args.effective_eval_batch_seqs}"
        )
    else:
        log(f"eval_mode:non_overlapping seq_len:{args.effective_eval_seq_len}")
    log(
        f"compute_dtype:{COMPUTE_DTYPE} compile:{compiled} compile_mode:{args.mlx_compile} quant_format:{args.quant_format} "
        f"turbo_qat:{args.turbo_qat} turbo_block:{args.turbo_block_size} "
        f"turbo_bits_mse:{args.turbo_mse_bits} turbo_bits_prod:{args.turbo_prod_bits}"
    )
    if TURBO_QAT_EXCLUDE_PATTERNS:
        log(f"turbo_qat_exclude_patterns:{','.join(TURBO_QAT_EXCLUDE_PATTERNS)}")
    log(
        f"ortho_probe:mode:{args.ortho_probe_mode} targets:{args.ortho_probe_targets or 'off'} "
        f"eps:{args.ortho_probe_eps:.1e} skip_final_export:{int(args.skip_final_export)}"
    )
    log(
        f"optimizer_turbo:{opt.turbo_matrix_optimizer_name} turbo_matrix_params:{len(opt.turbo_matrix_keys)} "
        f"turbo_muon_momentum:{args.turbo_qat_muon_momentum if args.turbo_qat else args.muon_momentum}"
    )
    if args.curriculum_token_category_weighting:
        log(
            f"curriculum_token_weighting:enabled url:{args.curriculum_url_token_weight:.3f} "
            f"identifier:{args.curriculum_identifier_token_weight:.3f} "
            f"repeat:{args.curriculum_repeat_token_weight:.3f}"
        )
    if context_delta_cfg.enabled:
        log(
            f"curriculum_context_delta:enabled short_len:{context_delta_cfg.short_context_len} "
            f"topk_fraction:{context_delta_cfg.topk_fraction:.3f} "
            f"max_multiplier:{context_delta_cfg.max_multiplier:.3f} "
            f"power:{context_delta_cfg.score_power:.3f} "
            f"use_abs:{int(context_delta_cfg.use_absolute_delta)}"
        )
    if structural_branch_cfg.enabled:
        log(
            f"structural_branching:enabled start_frac:{structural_branch_cfg.start_frac:.3f} "
            f"weight:{structural_branch_cfg.weight:.3f} branch_length:{structural_branch_cfg.branch_length} "
            f"max_branches:{structural_branch_cfg.max_branches} min_miss:{structural_branch_cfg.min_structural_miss:.3f} "
            f"max_gap:{structural_branch_cfg.max_top1_gap:.3f} max_top12_cos:{structural_branch_cfg.max_top12_cosine:.3f} "
            f"min_score:{structural_branch_cfg.min_branch_score:.3f} min_prob:{structural_branch_cfg.min_top1_prob:.3f} "
            f"min_pos_gap:{structural_branch_cfg.min_position_gap} margin:{structural_branch_cfg.margin:.3f} "
            f"state_w:{structural_branch_cfg.state_divergence_weight:.3f} "
            f"state_target_max_cos:{structural_branch_cfg.state_target_max_cosine:.3f} "
            f"adaptive_depth:{int(structural_branch_cfg.adaptive_depth_enabled)} "
            f"adaptive_min_depth:{structural_branch_cfg.adaptive_min_depth} "
            f"adaptive_plateau_tol:{structural_branch_cfg.adaptive_plateau_tol:.3f} "
            f"adaptive_converged_div:{structural_branch_cfg.adaptive_converged_divergence:.3f}"
        )
    if args.early_exit_aux_weight > 0.0 or args.early_exit_branch_draft_enabled:
        log(
            f"early_exit:layer:{resolve_early_exit_layer_index(args.early_exit_layer_index, args.num_layers)} "
            f"horizons:{args.early_exit_horizons} aux_weight:{args.early_exit_aux_weight:.3f} "
            f"head_init_std:{args.early_exit_head_init_std:.4f} "
            f"cascaded:{int(args.early_exit_cascaded_enabled)} "
            f"condition_init_std:{args.early_exit_condition_init_std:.4f} "
            f"branch_draft:{int(args.early_exit_branch_draft_enabled)} "
            f"branch_conf_threshold:{args.early_exit_branch_conf_threshold:.3f} "
            f"branch_max_draft_tokens:{args.early_exit_branch_max_draft_tokens}"
        )
    if early_exit_budget_ctrl.enabled:
        log(
            f"early_exit_dynamic_budget:enabled min_scale:{early_exit_budget_ctrl.min_scale:.3f} "
            f"max_scale:{early_exit_budget_ctrl.max_scale:.3f} "
            f"op_density_high:{early_exit_budget_ctrl.operator_density_high:.4f} "
            f"op_density_low:{early_exit_budget_ctrl.operator_density_low:.4f} "
            f"high_human_compressibility:{early_exit_budget_ctrl.high_human_compressibility:.3f} "
            f"low_human_compressibility:{early_exit_budget_ctrl.low_human_compressibility:.3f}"
        )
    log(
        f"ema:enabled:{args.ema_enabled} decay:{args.ema_decay:.6f} start_frac:{args.ema_start_frac:.3f} "
        f"reset_on_phase:{int(args.ema_reset_on_phase_transition)} "
        f"reset_min_progress:{args.ema_phase_reset_min_progress:.3f} "
        f"reset_cooldown_steps:{args.ema_phase_reset_cooldown_steps} "
        f"reset_requires_qat_full:{int(args.ema_phase_reset_requires_qat_full)} "
        f"adaptive_grad_ac:{int(args.ema_adaptive_grad_ac)} "
        f"grad_ac_beta:{args.ema_grad_ac_beta:.3f} "
        f"grad_ac_start_threshold:{args.ema_grad_ac_start_threshold:.3f} "
        f"adaptive_min_decay:{args.ema_adaptive_min_decay:.6f} "
        f"adaptive_max_decay:{args.ema_adaptive_max_decay:.6f} "
        f"aggressive_warmdown:{int(args.ema_aggressive_warmdown)} "
        f"reset_on_qat_full:{int(args.ema_reset_on_qat_full)}"
    )
    log(
        f"ema_teacher:enabled:{int(args.ema_teacher_distill_enabled)} "
        f"decay:{args.ema_teacher_decay:.6f} start_frac:{args.ema_teacher_start_frac:.3f} "
        f"distill_weight:{args.ema_teacher_distill_weight:.4f} "
        f"temperature:{args.ema_teacher_temperature:.3f}"
    )
    log(
        f"external_teacher:enabled:{int(bool(external_teacher_models))} "
        f"count:{len(external_teacher_models)} start_frac:{args.external_teacher_start_frac:.3f} "
        f"distill_weight:{args.external_teacher_distill_weight:.4f} "
        f"temperature:{args.external_teacher_temperature:.3f}"
    )
    if args.external_teacher_hidden_distill_weight > 0.0 or teacher_hidden_cache is not None:
        log(
            f"external_teacher_hidden:weight:{args.external_teacher_hidden_distill_weight:.4f} "
            f"layer:{args.external_teacher_hidden_layer} "
            f"cache_dir:{args.external_teacher_hidden_cache_dir or '-'} "
            f"cache_max_entries:{args.external_teacher_hidden_cache_max_entries} "
            f"cache_write:{int(args.external_teacher_hidden_cache_write)}"
        )
    log(
        f"step_audit:enabled:{int(args.step_audit_log_every > 0)} "
        f"log_every:{args.step_audit_log_every} "
        f"reset_peak:{int(args.step_audit_reset_peak)}"
    )
    log(
        f"sanitize_nonfinite:enabled:{int(args.sanitize_nonfinite_grads)} "
        f"every:{args.sanitize_nonfinite_grads_every} "
        f"always_first_steps:{args.sanitize_nonfinite_grads_always_first_steps} "
        f"topk:{args.nonfinite_grad_topk}"
    )
    if replay_buffer is not None:
        replay_summary = replay_buffer.summary()
        log(
            f"replay_queue:enabled path:{replay_summary['replay_queue_path']} "
            f"mix_fraction:{args.replay_mix_fraction:.3f} "
            f"refresh_every_steps:{args.replay_refresh_every_steps} "
            f"max_cached_examples:{args.replay_max_cached_examples} "
            f"emit_every:{args.replay_emit_every} emit_topk:{args.replay_emit_topk} "
            f"emit_min_seq_loss:{args.replay_emit_min_seq_loss:.4f} "
            f"cached_examples:{replay_summary['replay_cached_examples']}"
        )
    if snapshot_runtime is not None:
        log(
            f"student_snapshot:enabled dir:{snapshot_runtime.root_dir} "
            f"every:{args.student_snapshot_every} keep_last:{args.student_snapshot_keep_last} "
            f"use_ema:{int(args.student_snapshot_use_ema)} heartbeat_every:{args.student_heartbeat_every}"
        )
    if args.external_controller_enabled:
        log(
            f"external_controller:enabled refresh_every:{args.external_controller_refresh_every} "
            f"decision_path:{snapshot_runtime.controller_decision_path if snapshot_runtime is not None else '-'}"
        )
    if adaptive_ctrl_cfg.enabled:
        log(
            f"adaptive_controller:enabled log_every:{adaptive_ctrl_cfg.log_every} "
            f"sanitize_min_every:{adaptive_ctrl_cfg.sanitize_min_every} "
            f"sanitize_max_every:{adaptive_ctrl_cfg.sanitize_max_every} "
            f"sanitize_stable_steps:{adaptive_ctrl_cfg.sanitize_stable_steps} "
            f"sanitize_recovery_steps:{adaptive_ctrl_cfg.sanitize_recovery_steps} "
            f"distill_disagree:[{adaptive_ctrl_cfg.distill_disagree_low:.3f},{adaptive_ctrl_cfg.distill_disagree_high:.3f}] "
            f"distill_mult:[{adaptive_ctrl_cfg.distill_min_mult:.2f},{adaptive_ctrl_cfg.distill_max_mult:.2f}] "
            f"branch_disagree:[{adaptive_ctrl_cfg.branch_disagree_low:.3f},{adaptive_ctrl_cfg.branch_disagree_high:.3f}] "
            f"branch_mult:[{adaptive_ctrl_cfg.branch_min_mult:.2f},{adaptive_ctrl_cfg.branch_max_mult:.2f}] "
            f"branch_max_extra:{adaptive_ctrl_cfg.branch_max_extra}"
        )
    for meta in external_teacher_meta:
        log(
            f"external_teacher_spec:{meta['index']} "
            f"layers:{meta['layers']} dim:{meta['dim']} heads:{meta['heads']} "
            f"logic_dim:{meta['logic_dim']} tie_embeddings:{int(bool(meta['tie_embeddings']))} "
            f"matched_param_frac:{float(meta['matched_param_fraction']):.3f} "
            f"missing_tensors:{int(meta['missing_tensors'])} "
            f"mismatched_tensors:{int(meta['mismatched_tensors'])} "
            f"unexpected_tensors:{int(meta['unexpected_tensors'])} "
            f"config:{meta['config_path']} checkpoint:{meta['checkpoint_path']}"
        )
    if args.quant_eval_every > 0:
        log(
            f"quant_eval:enabled every:{args.quant_eval_every} seqs:{(quant_eval_tokens.size - 1) // args.effective_eval_seq_len}"
        )
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
        f"{f' lm_head:{model.lm_head.weight.dtype}' if not args.tie_embeddings else ''}"
    )
    qat_full_frac = args.turbo_qat_start_frac + max(args.turbo_qat_ramp_frac, 0.0)
    if args.turbo_qat and args.warmdown_fraction > 0.0:
        warmdown_start_frac = 1.0 - args.warmdown_fraction
        if qat_full_frac > warmdown_start_frac:
            log(
                f"WARNING: qat_full_frac:{qat_full_frac:.3f} exceeds warmdown_start_frac:{warmdown_start_frac:.3f}; "
                "full-strength QAT will start after warmdown begins"
            )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, _, _, _ = loss_and_grad_chunked(
                    args,
                    model,
                    train_loader,
                    compiled_loss_and_grad,
                    token_category_luts=token_category_luts,
                    early_exit_budget_controller=early_exit_budget_ctrl,
                )
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        eval_seq_len = args.effective_eval_seq_len
        if args.eval_stride > 0:
            warm_windows = build_sliding_eval_windows(val_tokens.size - 1, eval_seq_len, args.eval_stride)
            warm_batch = warm_windows[: args.effective_eval_batch_seqs]
            x_np = np.zeros((len(warm_batch), eval_seq_len), dtype=np.int32)
            for i, (window_start, window_len, _, _) in enumerate(warm_batch):
                chunk = val_tokens[window_start:window_start + window_len + 1]
                x_np[i, :window_len] = chunk[:-1]
            warm_operator_codes = operator_codes_mx_for_numpy_batch(model, x_np)
            warm_val_logits = compiled_forward_logits(mx.array(x_np, dtype=mx.int32), warm_operator_codes)
            mx.eval(warm_val_logits)
        else:
            val_batch_tokens = args.val_batch_size // args.grad_accum_steps
            if val_batch_tokens < eval_seq_len:
                raise ValueError(
                    "VAL_BATCH_SIZE must provide at least one eval sequence; "
                    f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                    f"EVAL_SEQ_LEN={eval_seq_len}"
                )
            warm_val_seqs = min(val_batch_tokens // eval_seq_len, (val_tokens.size - 1) // eval_seq_len)
            warm_chunk = val_tokens[: warm_val_seqs * eval_seq_len + 1]
            x_val_np = warm_chunk[:-1].reshape(-1, eval_seq_len)
            x_val = mx.array(x_val_np, dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(-1, eval_seq_len), dtype=mx.int32)
            warm_operator_codes = operator_codes_mx_for_numpy_batch(model, x_val_np)
            warm_val_loss = compiled_loss(x_val, y_val, warm_operator_codes)
            mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = build_train_loader(args, log_fn=log, dataset_name=dataset_name)
        replay_buffer = getattr(train_loader, "replay_buffer", None)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    wallclock_final_reserve_ms = max(1000.0 * args.wallclock_final_reserve_seconds, 0.0)
    stop_after_step: int | None = None
    prev_transition_alignment: float | None = None
    last_transition_step: int | None = None
    last_ema_reset_step: int | None = None
    prev_flat_grads: dict[str, mx.array] | None = None
    grad_ac_state: dict[str, float] = {"matrix": 0.0, "turbo": 0.0, "scalar": 0.0}
    prev_qat_full = False
    last_val_bpb: float | None = None
    last_quant_gap_bpb: float | None = None
    run_start_t0 = time.perf_counter()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_quant_eval = args.quant_eval_every > 0 and (last_step or step % args.quant_eval_every == 0)
        if should_validate or should_quant_eval:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            restore_params: dict[str, mx.array] | None = None
            if ema_state is not None:
                restore_params = flat_parameter_state(model)
                apply_flat_arrays(model, cast_ema_like(ema_state, restore_params))
                model.clear_turbo_cache()
            if should_validate:
                val_loss, val_bpb = eval_val(
                    args,
                    model,
                    compiled_loss,
                    compiled_forward_logits,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                )
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
                last_val_bpb = float(val_bpb)
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = make_gpt(args, sp)
                    if compiled:
                        if uses_logic:
                            compiled_quant_loss_impl = mx.compile(
                                lambda x, y, operator_codes: quant_eval_model.ce_loss(x, y, operator_codes),
                                inputs=quant_eval_model.state,
                                outputs=quant_eval_model.state,
                            )
                            compiled_quant_forward_logits_impl = mx.compile(
                                lambda x, operator_codes: quant_eval_model.forward_logits(x, operator_codes),
                                inputs=quant_eval_model.state,
                                outputs=quant_eval_model.state,
                            )
                            compiled_quant_loss = lambda x, y, operator_codes=None: compiled_quant_loss_impl(x, y, operator_codes)
                            compiled_quant_forward_logits = lambda x, operator_codes=None: compiled_quant_forward_logits_impl(x, operator_codes)
                        else:
                            compiled_quant_loss_impl = mx.compile(
                                lambda x, y: quant_eval_model.ce_loss(x, y),
                                inputs=quant_eval_model.state,
                                outputs=quant_eval_model.state,
                            )
                            compiled_quant_forward_logits_impl = mx.compile(
                                lambda x: quant_eval_model.forward_logits(x),
                                inputs=quant_eval_model.state,
                                outputs=quant_eval_model.state,
                            )
                            compiled_quant_loss = lambda x, y, operator_codes=None: compiled_quant_loss_impl(x, y)
                            compiled_quant_forward_logits = lambda x, operator_codes=None: compiled_quant_forward_logits_impl(x)
                    else:
                        if uses_logic:
                            compiled_quant_loss = lambda x, y, operator_codes=None: quant_eval_model.ce_loss(x, y, operator_codes)
                            compiled_quant_forward_logits = lambda x, operator_codes=None: quant_eval_model.forward_logits(x, operator_codes)
                        else:
                            compiled_quant_loss = lambda x, y, operator_codes=None: quant_eval_model.ce_loss(x, y)
                            compiled_quant_forward_logits = lambda x, operator_codes=None: quant_eval_model.forward_logits(x)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val(
                    args,
                    model,
                    compiled_loss,
                    compiled_forward_logits,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = eval_val(
                    args,
                    quant_eval_model,
                    compiled_quant_loss,
                    compiled_quant_forward_logits,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                log(
                    f"step:{step}/{args.iterations} quant_diag_seqs:{(quant_eval_tokens.size - 1) // args.effective_eval_seq_len} "
                    f"raw_val_loss:{raw_q_val_loss:.4f} raw_val_bpb:{raw_q_val_bpb:.4f} "
                    f"quant_val_loss:{q_val_loss:.4f} quant_val_bpb:{q_val_bpb:.4f} "
                    f"quant_gap_bpb:{q_val_bpb - raw_q_val_bpb:+.4f} int8_zlib_bytes:{len(quant_blob)} "
                    f"payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)} "
                    f"{format_quant_stats(quant_stats)} "
                    f"eval_time:{1000.0 * (time.perf_counter() - q_t0):.0f}ms"
                )
                last_quant_gap_bpb = float(q_val_bpb - raw_q_val_bpb)
            if restore_params is not None:
                apply_flat_arrays(model, restore_params)
                model.clear_turbo_cache()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms "
                    f"wallclock_elapsed:{1000.0 * (time.perf_counter() - run_start_t0):.0f}ms "
                    f"reserve:{wallclock_final_reserve_ms:.0f}ms step:{step}/{args.iterations}"
                )
            break

        wallclock_elapsed_ms = 1000.0 * (time.perf_counter() - run_start_t0)
        progress_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        if model.has_hardmax_structural_controller():
            scheduled_temp = hardmax_structural_temperature_for_progress(
                args,
                step,
                progress_ms,
                max_wallclock_ms,
            )
            model.hardmax_structural_controller.set_temperature(scheduled_temp)
        if hasattr(train_loader, "maybe_refresh"):
            train_loader.maybe_refresh(step)
        controller_runtime_enabled = bool(adaptive_ctrl_cfg.enabled or args.external_controller_enabled)
        if args.external_controller_enabled:
            external_decision, external_controller_mtime_ns = maybe_load_external_controller_decision(
                snapshot_runtime,
                args,
                step=step,
                last_mtime_ns=external_controller_mtime_ns,
            )
            if external_decision is not None:
                min_step = int(external_decision.get("min_step", 0))
                max_step = int(external_decision.get("max_step", 0))
                if step >= min_step and (max_step <= 0 or step <= max_step):
                    applied = apply_external_controller_decision(
                        external_decision,
                        controller_state=adaptive_ctrl_state,
                        train_loader=train_loader,
                        model=model,
                    )
                    if applied:
                        decision_note = str(applied.get("note", "-"))
                        log(
                            f"external_controller:applied step:{step}/{args.iterations} "
                            f"decision_id:{applied.get('decision_id', '-')} "
                            f"source:{applied.get('source', '-') } "
                            f"sanitize_every:{applied.get('sanitize_every', adaptive_ctrl_state.sanitize_every)} "
                            f"distill_mult:{applied.get('distill_weight_mult', adaptive_ctrl_state.distill_weight_mult):.2f} "
                            f"branch_mult:{applied.get('branch_weight_mult', adaptive_ctrl_state.branch_weight_mult):.2f} "
                            f"branch_extra:{applied.get('branch_extra_max_branches', adaptive_ctrl_state.branch_extra_max_branches)} "
                            f"replay_mix:{applied.get('replay_mix_fraction', getattr(train_loader, 'mix_fraction', 0.0)):.3f} "
                            f"hardmax_micro:{applied.get('hardmax_micro_steps', getattr(model, 'hardmax_struct_fast_refinement_steps', 1))} "
                            f"note:{decision_note}"
                        )
        lr_mul = args.lr_mul(step, progress_ms)
        if args.curriculum_enabled and hasattr(train_loader, "begin_step"):
            curriculum_phase = train_loader.begin_step()
        else:
            curriculum_phase = train_loader.current_phase() if args.curriculum_enabled and hasattr(train_loader, "current_phase") else None
        curriculum_logic_enabled = (
            curriculum_phase.enable_logic_sidecar
            if curriculum_phase is not None and args.curriculum_apply_logic_phase_gating
            else True
        )
        logic_seed_weight = args.polarity_seed_weight if curriculum_logic_enabled else 0.0
        logic_sparse_weight = args.polarity_sparse_weight if curriculum_logic_enabled else 0.0
        logic_smooth_weight = args.polarity_smooth_weight if curriculum_logic_enabled else 0.0
        curriculum_focal_weight = (
            curriculum_phase.focal_loss_weight
            if curriculum_phase is not None and args.curriculum_apply_focal
            else 0.0
        )
        qat_scale = turbo_qat_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        if args.curriculum_apply_qat_phase_gating and curriculum_phase is not None and not curriculum_phase.enable_qat:
            qat_scale = 0.0
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        model.set_turbo_qat(qat_active, qat_scale)
        needs_dynamic_logic = uses_logic and args.curriculum_apply_logic_phase_gating and curriculum_phase is not None
        needs_token_category_weighting = args.curriculum_token_category_weighting and token_category_luts is not None
        needs_context_delta_weighting = context_delta_cfg.enabled
        runtime_branch_cfg = (
            adaptive_train_controller_branch_config(structural_branch_cfg, adaptive_ctrl_state)
            if controller_runtime_enabled
            else structural_branch_cfg
        )
        structural_branching_active = structural_branching_active_for_progress(
            runtime_branch_cfg,
            args,
            step,
            progress_ms,
            max_wallclock_ms,
        )
        ema_teacher_active = ema_teacher_active_for_progress(args, step, progress_ms, max_wallclock_ms)
        external_teacher_active = external_teacher_active_for_progress(args, step, progress_ms, max_wallclock_ms)
        teacher_distill_active = ema_teacher_active or external_teacher_active
        teacher_models = (
            [ema_teacher_model] if ema_teacher_active and ema_teacher_model is not None
            else external_teacher_models if external_teacher_active else None
        )
        distill_weight = (
            args.ema_teacher_distill_weight if ema_teacher_active else
            args.external_teacher_distill_weight if external_teacher_active else
            0.0
        )
        if controller_runtime_enabled:
            distill_weight *= adaptive_ctrl_state.distill_weight_mult
        distill_temperature = (
            args.ema_teacher_temperature if ema_teacher_active else
            args.external_teacher_temperature if external_teacher_active else
            1.0
        )
        teacher_hidden_distill_weight = (
            args.external_teacher_hidden_distill_weight if external_teacher_active else 0.0
        )
        if ema_teacher_model is not None:
            ema_teacher_model.set_turbo_qat(qat_active, qat_scale)
            ema_teacher_model.clear_turbo_cache()
        if (
            qat_active
            or curriculum_focal_weight > 0.0
            or needs_dynamic_logic
            or needs_token_category_weighting
            or needs_context_delta_weighting
            or structural_branching_active
            or teacher_distill_active
            or teacher_hidden_distill_weight > 0.0
        ):
            if uses_logic:
                step_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner, operator_codes_inner, token_weights_inner, teacher_logits_inner, teacher_hidden_inner, early_exit_aux_weight_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        operator_codes_inner,
                        polarity_seed_weight=logic_seed_weight,
                        polarity_sparse_weight=logic_sparse_weight,
                        polarity_smooth_weight=logic_smooth_weight,
                        focal_loss_weight=curriculum_focal_weight,
                        focal_gamma=args.curriculum_focal_gamma,
                        focal_max_multiplier=args.curriculum_focal_max_multiplier,
                        token_weights=token_weights_inner,
                        context_delta_config=context_delta_cfg,
                        teacher_logits=teacher_logits_inner,
                        teacher_hidden=teacher_hidden_inner,
                        ema_teacher_distill_weight=distill_weight,
                        teacher_hidden_distill_weight=teacher_hidden_distill_weight,
                        ema_teacher_temperature=distill_temperature,
                        early_exit_aux_weight_override=early_exit_aux_weight_inner,
                        structural_branching_cfg=structural_branching_cfg_override if structural_branching_active else None,
                        branch_plans=branch_plans,
                    )[0] + (qat_lambda * model.turbo_regularizer() if qat_active else 0.0),
                )(x, y, operator_codes, token_weights, teacher_logits, teacher_hidden, early_exit_aux_weight_override)
            else:
                step_loss_and_grad = lambda x, y, operator_codes=None, token_weights=None, teacher_logits=None, teacher_hidden=None, branch_plans=None, structural_branching_cfg_override=None, early_exit_aux_weight_override=None: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner, token_weights_inner, teacher_logits_inner, teacher_hidden_inner, early_exit_aux_weight_inner: model.loss_terms(
                        x_inner,
                        y_inner,
                        focal_loss_weight=curriculum_focal_weight,
                        focal_gamma=args.curriculum_focal_gamma,
                        focal_max_multiplier=args.curriculum_focal_max_multiplier,
                        token_weights=token_weights_inner,
                        context_delta_config=context_delta_cfg,
                        teacher_logits=teacher_logits_inner,
                        teacher_hidden=teacher_hidden_inner,
                        ema_teacher_distill_weight=distill_weight,
                        teacher_hidden_distill_weight=teacher_hidden_distill_weight,
                        ema_teacher_temperature=distill_temperature,
                        early_exit_aux_weight_override=early_exit_aux_weight_inner,
                        structural_branching_cfg=structural_branching_cfg_override if structural_branching_active else None,
                        branch_plans=branch_plans,
                    )[0] + (qat_lambda * model.turbo_regularizer() if qat_active else 0.0),
                )(x, y, token_weights, teacher_logits, teacher_hidden, early_exit_aux_weight_override)
        else:
            step_loss_and_grad = compiled_loss_and_grad
        step_t0 = time.perf_counter()
        audit_this_step = (
            args.step_audit_log_every > 0
            and (step <= 10 or step % args.step_audit_log_every == 0)
        )
        if audit_this_step and args.step_audit_reset_peak:
            reset_metal_peak_memory()
        step_mem_before = metal_memory_snapshot() if audit_this_step else None

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        component_metrics: dict[str, float] | None = None
        audit_timings: dict[str, float] | None = None
        controller_metrics = {
            "teacher_count": 0.0,
            "teacher_disagree_frac": 0.0,
            "teacher_unique_top1": 0.0,
            "teacher_pairwise_agree": 0.0,
            "branch_points": 0.0,
            "replay_emitted_examples": 0.0,
        }
        grad_scale = 1.0 / args.grad_accum_steps
        collect_loss_components = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        for _ in range(args.grad_accum_steps):
            replay_emit_step = (
                step
                if replay_buffer is not None
                and args.replay_emit_every > 0
                and step > 0
                and step % args.replay_emit_every == 0
                else None
            )
            loss, grads, chunk_component_metrics, chunk_timing_metrics, chunk_controller_metrics = loss_and_grad_chunked(
                args,
                model,
                train_loader,
                step_loss_and_grad,
                logic_phase_enabled=curriculum_logic_enabled,
                token_category_luts=token_category_luts,
                teacher_models=teacher_models,
                structural_branching_cfg=runtime_branch_cfg,
                structural_branch_budget_controller=structural_branch_budget_ctrl,
                early_exit_budget_controller=early_exit_budget_ctrl,
                structural_branching_active=structural_branching_active,
                curriculum_phase_focus=curriculum_phase.focus if curriculum_phase is not None else None,
                polarity_seed_weight=logic_seed_weight,
                polarity_sparse_weight=logic_sparse_weight,
                polarity_smooth_weight=logic_smooth_weight,
                focal_loss_weight=curriculum_focal_weight,
                focal_gamma=args.curriculum_focal_gamma,
                focal_max_multiplier=args.curriculum_focal_max_multiplier,
                context_delta_config=context_delta_cfg,
                distill_weight=distill_weight,
                teacher_hidden_distill_weight=teacher_hidden_distill_weight,
                distill_temperature=distill_temperature,
                collect_loss_components=collect_loss_components,
                collect_timing=audit_this_step,
                replay_buffer=replay_buffer,
                replay_emit_step=replay_emit_step,
                teacher_hidden_cache=teacher_hidden_cache,
                teacher_hidden_cache_write=args.external_teacher_hidden_cache_write,
                teacher_hidden_layer=args.external_teacher_hidden_layer,
            )
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if chunk_component_metrics is not None:
                if component_metrics is None:
                    component_metrics = {key: 0.0 for key in chunk_component_metrics}
                for key, value in chunk_component_metrics.items():
                    component_metrics[key] += float(value) * grad_scale
            if chunk_timing_metrics is not None:
                if audit_timings is None:
                    audit_timings = {key: 0.0 for key in chunk_timing_metrics}
                for key, value in chunk_timing_metrics.items():
                    audit_timings[key] += float(value)
            for key, value in chunk_controller_metrics.items():
                controller_metrics[key] = controller_metrics.get(key, 0.0) + float(value) * grad_scale
        if args.curriculum_enabled and hasattr(train_loader, "end_step"):
            train_loader.end_step()

        grads = tree_unflatten(list(accum.items()))
        train_loss_value = float(train_loss.item())
        grad_eval_ms = 0.0
        if audit_this_step:
            grad_eval_t0 = time.perf_counter()
            realize_grad_tree(grads)
            grad_eval_ms = 1000.0 * (time.perf_counter() - grad_eval_t0)
        sanitize_this_step = should_sanitize_nonfinite_grads(
            args,
            step,
            train_loss_value,
            interval_override=adaptive_ctrl_state.sanitize_every if controller_runtime_enabled else None,
        )
        sanitize_t0 = time.perf_counter()
        if sanitize_this_step:
            grads, grad_nonfinite = sanitize_grad_tree(grads, topk=args.nonfinite_grad_topk)
        else:
            grad_nonfinite = empty_nonfinite_grad_summary()
        sanitize_ms = 1000.0 * (time.perf_counter() - sanitize_t0)
        clip_t0 = time.perf_counter()
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        clip_ms = 1000.0 * (time.perf_counter() - clip_t0)
        flat_clipped_grads = dict(tree_flatten(grads))
        flat_clipped_grads, masked_grad_names = maybe_mask_hardmax_structural_statebook_grads(
            flat_clipped_grads,
            args,
            step,
        )
        if masked_grad_names:
            grads = tree_unflatten(list(flat_clipped_grads.items()))
        grad_group_cos = gradient_group_cosines(flat_clipped_grads, prev_flat_grads, opt)
        beta = min(max(args.ema_grad_ac_beta, 0.0), 0.9999)
        one_minus_beta = 1.0 - beta
        for group_name, cosine in grad_group_cos.items():
            grad_ac_state[group_name] = beta * grad_ac_state[group_name] + one_minus_beta * cosine
        opt_t0 = time.perf_counter()
        opt_stats = opt.step(model, grads, step=step, lr_mul=lr_mul)
        prev_flat_grads = {name: value.astype(mx.float32) for name, value in flat_clipped_grads.items()}
        model.clear_turbo_cache()
        mx.synchronize()
        opt_ms = 1000.0 * (time.perf_counter() - opt_t0)
        if adaptive_ctrl_cfg.enabled:
            adaptive_ctrl_state = update_adaptive_train_controller(
                adaptive_ctrl_cfg,
                adaptive_ctrl_state,
                train_loss_value=train_loss_value,
                grad_nonfinite_tensors=int(grad_nonfinite["nonfinite_tensors"]),
                teacher_count=controller_metrics.get("teacher_count"),
                teacher_disagree_frac=controller_metrics.get("teacher_disagree_frac"),
            )

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        wallclock_elapsed_ms = 1000.0 * (time.perf_counter() - run_start_t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        warmdown_active = lr_mul < 0.999999
        min_decay = min(max(args.ema_adaptive_min_decay, 0.0), args.ema_decay)
        max_decay = max(args.ema_decay, args.ema_adaptive_max_decay)
        start_threshold = max(args.ema_grad_ac_start_threshold, 1e-9)

        def adaptive_ema_decay(group_name: str) -> float:
            if not args.ema_adaptive_grad_ac:
                return args.ema_decay
            if args.ema_aggressive_warmdown and warmdown_active:
                return max_decay
            grad_ac = grad_ac_state[group_name]
            if grad_ac >= args.ema_grad_ac_start_threshold:
                return min_decay
            settled = min(max((args.ema_grad_ac_start_threshold - grad_ac) / start_threshold, 0.0), 1.0)
            return min_decay + (args.ema_decay - min_decay) * settled

        ema_active = ema_active_for_progress(args, step, approx_train_time_ms, max_wallclock_ms)
        if args.curriculum_apply_ema_phase_gating and curriculum_phase is not None:
            ema_active = ema_active and curriculum_phase.enable_ema
        if ema_active and ema_state is None:
            ema_state = init_ema_state(model)
            log(f"ema:start step:{step}/{args.iterations} progress:{training_progress_fraction(args, step, approx_train_time_ms, max_wallclock_ms):.3f}")
        just_hit_qat_full = args.ema_reset_on_qat_full and args.turbo_qat and not prev_qat_full and qat_scale >= 1.0
        if just_hit_qat_full and ema_state is not None:
            ema_state = init_ema_state(model)
            last_ema_reset_step = step
            log(f"ema:reset_on_qat_full step:{step}/{args.iterations} progress:{training_progress_fraction(args, step, approx_train_time_ms, max_wallclock_ms):.3f}")
        ema_t0 = time.perf_counter()
        if ema_state is not None:
            if args.ema_adaptive_grad_ac:
                decay_by_group = {
                    group_name: adaptive_ema_decay(group_name)
                    for group_name in grad_ac_state
                }
                update_ema_state_grouped(ema_state, model, decay_by_group, ema_group_map, args.ema_decay)
            else:
                update_ema_state(ema_state, model, args.ema_decay)
        if ema_teacher_state is not None:
            update_ema_state(ema_teacher_state, model, args.ema_teacher_decay)
            if ema_teacher_model is not None:
                apply_flat_arrays(
                    ema_teacher_model,
                    cast_ema_like(ema_teacher_state, flat_parameter_state(ema_teacher_model)),
                )
                ema_teacher_model.clear_turbo_cache()
        ema_ms = 1000.0 * (time.perf_counter() - ema_t0)
        step_mem_after = metal_memory_snapshot() if audit_this_step else None

        matrix_cos = opt_stats["matrix_alignment"] / max(math.sqrt(opt_stats["matrix_grad_sq"] * opt_stats["matrix_buf_sq"]), 1e-12)
        turbo_cos = opt_stats["turbo_alignment"] / max(math.sqrt(opt_stats["turbo_grad_sq"] * opt_stats["turbo_buf_sq"]), 1e-12)
        transition_alignment = opt_stats["turbo_alignment"] if opt_stats["turbo_buf_sq"] > 0.0 else opt_stats["matrix_alignment"]
        transition_cos = turbo_cos if opt_stats["turbo_buf_sq"] > 0.0 else matrix_cos
        ema_decay_matrix = adaptive_ema_decay("matrix")
        ema_decay_turbo = adaptive_ema_decay("turbo")
        is_transition = prev_transition_alignment is not None and prev_transition_alignment >= 0.0 and transition_alignment < 0.0
        if is_transition:
            last_transition_step = step
            progress_frac = training_progress_fraction(args, step, approx_train_time_ms, max_wallclock_ms)
            ema_gap = ema_gap_metrics(ema_state, model)
            if args.phase_transition_log:
                log(
                    f"phase_transition:step:{step}/{args.iterations} align:{transition_alignment:.6e} "
                    f"cos:{transition_cos:.6f} matrix_align:{opt_stats['matrix_alignment']:.6e} "
                    f"turbo_align:{opt_stats['turbo_alignment']:.6e} lr_mul:{lr_mul:.6f} "
                    f"turbo_qat_scale:{qat_scale:.3f} progress:{progress_frac:.3f} ema_active:{int(ema_state is not None)} "
                    f"ema_teacher_active:{int(ema_teacher_active)} "
                    f"external_teacher_active:{int(external_teacher_active)} "
                    f"ema_gap_rmse:{ema_gap['rmse']:.6e} ema_gap_mae:{ema_gap['mae']:.6e} "
                    f"grad_ac_matrix:{grad_ac_state['matrix']:.6f} grad_ac_turbo:{grad_ac_state['turbo']:.6f}"
                )
            reset_allowed = (
                args.ema_reset_on_phase_transition
                and ema_state is not None
                and progress_frac >= args.ema_phase_reset_min_progress
                and (not args.ema_phase_reset_requires_qat_full or qat_scale >= 1.0)
                and (
                    last_ema_reset_step is None
                    or step - last_ema_reset_step >= args.ema_phase_reset_cooldown_steps
                )
            )
            if reset_allowed:
                ema_state = init_ema_state(model)
                last_ema_reset_step = step
                log(
                    f"ema:reset_on_phase step:{step}/{args.iterations} progress:{progress_frac:.3f} "
                    f"turbo_qat_scale:{qat_scale:.3f} cooldown:{args.ema_phase_reset_cooldown_steps}"
                )
        prev_transition_alignment = transition_alignment
        should_log_train = (
            (args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0))
            or (
                adaptive_ctrl_cfg.enabled
                and adaptive_ctrl_cfg.log_every > 0
                and (step <= 10 or step % adaptive_ctrl_cfg.log_every == 0)
            )
            or stop_after_step is not None
        )
        if should_log_train:
            ema_gap = ema_gap_metrics(ema_state, model) if ema_state is not None else {"rmse": 0.0, "mae": 0.0}
            gate_metrics = logic_gate_metrics(model)
            hardmax_metrics = hardmax_structural_metrics(model)
            curriculum_step_metrics = (
                train_loader.last_step_metrics()
                if args.curriculum_enabled and hasattr(train_loader, "last_step_metrics")
                else {}
            )
            curriculum_metrics_str = ""
            if curriculum_step_metrics:
                metric_order = (
                    "chunks",
                    "unique_chunk_frac",
                    "repeat_bucket_frac",
                    "once_bucket_frac",
                    "repeat_reuse_frac",
                    "mean_difficulty",
                    "mean_operator_density",
                    "mean_compressibility",
                    "mean_learnability",
                    "mean_quality",
                )
                parts: list[str] = []
                for key in metric_order:
                    value = curriculum_step_metrics.get(key)
                    if value is None:
                        continue
                    if isinstance(value, int):
                        parts.append(f"{key}:{value}")
                    else:
                        parts.append(f"{key}:{float(value):.3f}")
                if parts:
                    curriculum_metrics_str = " " + " ".join(parts)
            replay_metrics_str = ""
            replay_batch_metrics = (
                train_loader.last_batch_metrics()
                if hasattr(train_loader, "last_batch_metrics")
                else {}
            )
            if replay_buffer is not None:
                replay_summary = replay_buffer.summary()
                replay_metrics_str = (
                    f" replay_rows:{int(replay_batch_metrics.get('replay_rows', 0))}"
                    f" replay_frac:{float(replay_batch_metrics.get('replay_frac', 0.0)):.3f}"
                    f" replay_emitted_step:{int(round(float(controller_metrics.get('replay_emitted_examples', 0.0))))}"
                    f" replay_cache_entries:{int(replay_summary.get('replay_cached_examples', 0))}"
                    f" replay_total_emitted:{int(replay_summary.get('replay_total_emitted', 0))}"
                    f" replay_total_sampled:{int(replay_summary.get('replay_total_sampled', 0))}"
                )
            tensor_activity_str = ""
            should_log_tensor_activity = (
                args.tensor_activity_log_every > 0
                and (step <= 10 or step % args.tensor_activity_log_every == 0 or stop_after_step is not None)
            )
            if should_log_tensor_activity:
                activity = tensor_activity_snapshot(
                    flat_clipped_grads,
                    hot_threshold=args.tensor_activity_hot_threshold,
                    warm_threshold=args.tensor_activity_warm_threshold,
                    nonzero_threshold=args.tensor_activity_nonzero_threshold,
                    topk=args.tensor_activity_topk,
                )
                total_params = max(int(activity["param_count"]), 1)
                top_items = activity["top"]
                top_str = ",".join(
                    f"{name}:{bucket}:{mean_abs:.2e}:{nonzero_frac:.2f}"
                    for mean_abs, nonzero_frac, _numel, name, bucket in top_items
                )
                tensor_activity_str = (
                    f" grad_hot_tensors:{activity['hot_tensors']}/{activity['tensor_count']}"
                    f" grad_warm_tensors:{activity['warm_tensors']}/{activity['tensor_count']}"
                    f" grad_cold_tensors:{activity['cold_tensors']}/{activity['tensor_count']}"
                    f" grad_hot_param_frac:{int(activity['hot_params']) / total_params:.3f}"
                    f" grad_warm_param_frac:{int(activity['warm_params']) / total_params:.3f}"
                    f" grad_cold_param_frac:{int(activity['cold_params']) / total_params:.3f}"
                )
                if top_str:
                    tensor_activity_str += f" grad_top:{top_str}"
            grad_nonfinite_str = ""
            if int(grad_nonfinite["nonfinite_tensors"]) > 0:
                total_params = max(int(grad_nonfinite["param_count"]), 1)
                top_nonfinite = ",".join(
                    f"{name}:{count}"
                    for count, name in grad_nonfinite["top"]
                )
                grad_nonfinite_str = (
                    f" grad_nonfinite_tensors:{grad_nonfinite['nonfinite_tensors']}/{grad_nonfinite['tensor_count']}"
                    f" grad_nonfinite_param_frac:{int(grad_nonfinite['nonfinite_params']) / total_params:.3f}"
                )
                if top_nonfinite:
                    grad_nonfinite_str += f" grad_nonfinite_top:{top_nonfinite}"
            component_metrics_str = ""
            if component_metrics is not None:
                component_metrics_str = (
                    f" train_ce:{component_metrics['ce_loss']:.4f}"
                    f" train_seed:{component_metrics['seed_loss']:.4f}"
                    f" train_sparse:{component_metrics['sparse_loss']:.4f}"
                    f" train_smooth:{component_metrics['smooth_loss']:.4f}"
                    f" train_aux:{component_metrics['aux_loss']:.4f}"
                    f" train_earlyexit:{component_metrics['early_exit_loss']:.4f}"
                    f" train_prosody:{component_metrics['prosody_aux_loss']:.4f}"
                    f" train_prosody_cls:{component_metrics['prosody_token_class_loss']:.4f}"
                    f" train_prosody_bnd:{component_metrics['prosody_boundary_loss']:.4f}"
                    f" train_prosody_punct:{component_metrics['prosody_punctuation_loss']:.4f}"
                    f" train_prosody_q:{component_metrics['prosody_quote_loss']:.4f}"
                    f" train_distill:{component_metrics['distill_loss']:.4f}"
                    f" train_hidden_distill:{component_metrics['hidden_distill_loss']:.4f}"
                    f" earlyexit_w:{component_metrics['early_exit_weight']:.4f}"
                    f" train_branch_aux:{component_metrics['branch_aux_loss']:.4f}"
                    f" train_branch_rank:{component_metrics['branch_rank_loss']:.4f}"
                    f" train_branch_state:{component_metrics['branch_state_loss']:.4f}"
                    f" train_hardmax_bal:{component_metrics['hardmax_balance_loss']:.4f}"
                    f" train_hardmax_div:{component_metrics['hardmax_diversity_loss']:.4f}"
                    f" train_hardmax_pred:{component_metrics['hardmax_pred_loss']:.4f}"
                    f" train_hardmax_conf_loss:{component_metrics['hardmax_confidence_loss']:.4f}"
                    f" train_hardmax_op:{component_metrics['hardmax_operator_loss']:.4f}"
                    f" train_hardmax_cls:{component_metrics['hardmax_token_class_loss']:.4f}"
                    f" train_hardmax_bnd:{component_metrics['hardmax_boundary_loss']:.4f}"
                    f" train_hardmax_q:{component_metrics['hardmax_quote_loss']:.4f}"
                    f" hardmax_conf:{component_metrics['hardmax_confidence']:.3f}"
                    f" hardmax_budget:{component_metrics['hardmax_budget']:.3f}"
                    f" hardmax_H:{component_metrics['hardmax_entropy']:.3f}"
                    f" branch_enabled:{component_metrics['branch_enabled']:.2f}"
                    f" branch_budget:{component_metrics['branch_budget']:.2f}"
                    f" branch_points:{component_metrics['branch_points']:.2f}"
                    f" branch_miss:{component_metrics['branch_min_miss']:.3f}"
                    f" branch_gap:{component_metrics['branch_max_gap']:.3f}"
                    f" branch_cos:{component_metrics['branch_top12_cos']:.3f}"
                    f" branch_score:{component_metrics['branch_score']:.3f}"
                    f" teacher_count:{component_metrics['teacher_count']:.2f}"
                    f" teacher_disagree:{component_metrics['teacher_disagree_frac']:.3f}"
                    f" teacher_unique_top1:{component_metrics['teacher_unique_top1']:.3f}"
                    f" teacher_pairwise_agree:{component_metrics['teacher_pairwise_agree']:.3f}"
                    f" teacher_hidden_cache_hit:{component_metrics['teacher_hidden_cache_hit_frac']:.3f}"
                    f" teacher_hidden_cache_written:{component_metrics['teacher_hidden_cache_written']:.1f}"
                    f" prosody_feat_den:{component_metrics['prosody_feature_density']:.3f}"
                    f" prosody_reset:{component_metrics['prosody_reset_prior']:.3f}"
                    f" prosody_slow_reset:{component_metrics['prosody_slow_reset_prior']:.3f}"
                    f" prosody_punct_frac:{component_metrics['prosody_punctuation_frac']:.3f}"
                    f" prosody_quote_frac:{component_metrics['prosody_quote_frac']:.3f}"
                    f" prosody_sent_frac:{component_metrics['prosody_sentence_boundary_frac']:.3f}"
                    f" prosody_para_frac:{component_metrics['prosody_paragraph_boundary_frac']:.3f}"
                    f" prosody_state_rms:{component_metrics['prosody_state_delta_rms']:.5f}"
                )
            controller_metrics_str = ""
            if controller_runtime_enabled:
                controller_metrics_str = (
                    f" ctrl_sanitize_every:{adaptive_ctrl_state.sanitize_every}"
                    f" ctrl_distill_mult:{adaptive_ctrl_state.distill_weight_mult:.2f}"
                    f" ctrl_branch_mult:{adaptive_ctrl_state.branch_weight_mult:.2f}"
                    f" ctrl_branch_extra:{adaptive_ctrl_state.branch_extra_max_branches}"
                    f" ctrl_teacher_disagree:{adaptive_ctrl_state.last_teacher_disagree:.3f}"
                    f" ctrl_action:{adaptive_ctrl_state.last_action}"
                )
            grad_mask_str = ""
            if masked_grad_names:
                grad_mask_str = f" grad_masked:{','.join(masked_grad_names)}"
            audit_metrics_str = ""
            if audit_this_step and audit_timings is not None and step_mem_after is not None:
                accounted_ms = (
                    audit_timings["batch_ms"]
                    + audit_timings["teacher_ms"]
                    + audit_timings["branch_ms"]
                    + audit_timings["lossgrad_ms"]
                    + audit_timings["component_ms"]
                    + sanitize_ms
                    + clip_ms
                    + opt_ms
                    + ema_ms
                )
                other_ms = max(step_ms - accounted_ms, 0.0)
                active_delta_mb = float("nan")
                if step_mem_before is not None:
                    active_delta_mb = (
                        step_mem_after["metal_active_bytes"] - step_mem_before["metal_active_bytes"]
                    ) / (1024.0 * 1024.0)
                audit_metrics_str = (
                    f" audit_batch_ms:{audit_timings['batch_ms']:.1f}"
                    f" audit_teacher_ms:{audit_timings['teacher_ms']:.1f}"
                    f" audit_branch_ms:{audit_timings['branch_ms']:.1f}"
                    f" audit_lossgrad_ms:{audit_timings['lossgrad_ms']:.1f}"
                    f" audit_component_ms:{audit_timings['component_ms']:.1f}"
                    f" audit_grad_eval_ms:{grad_eval_ms:.1f}"
                    f" audit_sanitize_ms:{sanitize_ms:.1f}"
                    f" audit_clip_ms:{clip_ms:.1f}"
                    f" audit_opt_ms:{opt_ms:.1f}"
                    f" audit_ema_ms:{ema_ms:.1f}"
                    f" audit_other_ms:{other_ms:.1f}"
                    f" audit_metal_active_gb:{step_mem_after['metal_active_bytes'] / (1024.0**3):.3f}"
                    f" audit_metal_peak_gb:{step_mem_after['metal_peak_bytes'] / (1024.0**3):.3f}"
                    f" audit_metal_cache_gb:{step_mem_after['metal_cache_bytes'] / (1024.0**3):.3f}"
                    f" audit_host_rss_gb:{step_mem_after['host_rss_bytes'] / (1024.0**3):.3f}"
                )
                if not math.isnan(active_delta_mb):
                    audit_metrics_str += f" audit_metal_active_delta_mb:{active_delta_mb:.1f}"
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"wallclock_elapsed:{wallclock_elapsed_ms:.0f}ms "
                f"curriculum_phase:{curriculum_phase.name if curriculum_phase is not None else 'off'} "
                f"curriculum_logic:{int(curriculum_logic_enabled)} "
                f"curriculum_focal:{curriculum_focal_weight:.3f} "
                f"branch_dynamic:{int(structural_branch_budget_ctrl.enabled)} "
                f"earlyexit_dynamic:{int(early_exit_budget_ctrl.enabled)} "
                f"turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f} "
                f"align:{transition_alignment:.6e} align_cos:{transition_cos:.6f} "
                f"ema_active:{int(ema_state is not None)} ema_gap_rmse:{ema_gap['rmse']:.6e} "
                f"ema_teacher_active:{int(ema_teacher_active)} "
                f"external_teacher_active:{int(external_teacher_active)} "
                f"grad_ac_matrix:{grad_ac_state['matrix']:.6f} grad_ac_turbo:{grad_ac_state['turbo']:.6f} "
                f"ema_decay_matrix:{ema_decay_matrix:.6f} ema_decay_turbo:{ema_decay_turbo:.6f} "
                f"logic_gate_mean_abs:{gate_metrics['mean_abs']:.6e} "
                f"logic_gate_rms:{gate_metrics['rms']:.6e} "
                f"logic_gate_max_abs:{gate_metrics['max_abs']:.6e} "
                f"hardmax_gate_rms:{hardmax_metrics['gate_rms']:.6e} "
                f"hardmax_gate_max_abs:{hardmax_metrics['gate_max_abs']:.6e} "
                f"hardmax_book_cos:{hardmax_metrics['book_offdiag_cos']:.6f} "
                f"hardmax_temp:{hardmax_metrics['temperature']:.4f}"
                f"{grad_mask_str}"
                f"{grad_nonfinite_str}"
                f"{tensor_activity_str}"
                f"{component_metrics_str}"
                f"{controller_metrics_str}"
                f"{audit_metrics_str}"
                f"{replay_metrics_str}"
                f"{curriculum_metrics_str}"
            )
        maybe_write_student_heartbeat(
            snapshot_runtime,
            args,
            step=step,
            train_time_ms=approx_train_time_ms,
            step_avg_ms=approx_train_time_ms / max(step, 1),
            train_loss_value=train_loss_value,
            tok_s=tok_s,
            qat_scale=qat_scale,
            distill_weight=distill_weight,
            replay_buffer=replay_buffer,
            controller_metrics=controller_metrics,
        )
        snapshot_path = maybe_write_student_snapshot(
            snapshot_runtime,
            args,
            model,
            ema_state,
            step=step,
            train_time_ms=approx_train_time_ms,
            train_loss_value=train_loss_value,
            tok_s=tok_s,
            val_bpb=last_val_bpb,
            quant_gap_bpb=last_quant_gap_bpb,
            replay_buffer=replay_buffer,
        )
        if snapshot_path is not None:
            log(
                f"student_snapshot:step:{step}/{args.iterations} path:{snapshot_path} "
                f"bytes:{snapshot_path.stat().st_size}"
            )
        prev_qat_full = qat_scale >= 1.0
        if (
            max_wallclock_ms is not None
            and stop_after_step is None
            and wallclock_elapsed_ms >= max(max_wallclock_ms - wallclock_final_reserve_ms, 0.0)
        ):
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    if last_transition_step is not None:
        log(f"phase_transition:last_step:{last_transition_step}")
    if last_ema_reset_step is not None:
        log(f"ema:last_reset_step:{last_ema_reset_step}")
    if ema_state is not None:
        log("ema:applying weights for export")
        apply_flat_arrays(model, cast_ema_like(ema_state, flat_parameter_state(model)))
    if args.skip_final_export:
        log("final_export:skipped")
        return
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    raw_t0 = time.perf_counter()
    raw_val_loss, raw_val_bpb = eval_val(
        args,
        model,
        compiled_loss,
        compiled_forward_logits,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    raw_eval_ms = 1000.0 * (time.perf_counter() - raw_t0)
    log(f"final_raw_export_ready val_loss:{raw_val_loss:.4f} val_bpb:{raw_val_bpb:.4f} eval_time:{raw_eval_ms:.0f}ms")
    log(f"final_raw_export_ready_exact val_loss:{raw_val_loss:.8f} val_bpb:{raw_val_bpb:.8f}")
    log(f"final_wallclock_elapsed:{1000.0 * (time.perf_counter() - run_start_t0):.0f}ms")
    flat_state = exportable_flat_state(model)
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats, quant_raw, quant_blob = serialize_quantized_state_dict(flat_state)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x "
        f"{format_quant_stats(quant_stats)})"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.clear_turbo_cache()
    model.update(tree_unflatten(list(quant_flat.items())))
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        compiled_loss,
        compiled_forward_logits,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

if __name__ == "__main__":
    main()
