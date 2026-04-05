#!/usr/bin/env python3
"""
Slim segment-clock trainer for the current-size TurboQuant lane.

This keeps the lightweight grouped-slim training loop, but swaps in the
segment-clock memory model from `train_gpt_mlx_segmentlong.py`. The intended
promotion target is the proven no-carry `segment-prev-read` branch on top of
the new `slim-curriculum` control.

Scope:
- supports grouped streaming when curriculum is off
- supports the standard curriculum loader when curriculum is on
- supports no-carry segment memory (`LONGCTX_PERSIST_TRAIN_CARRY=0`)
- supports static early-exit aux and Turbo QAT

It intentionally rejects heavier dynamic-control features from the full base
trainer and does not support persistent carry streams.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_unflatten

import train_gpt_mlx as base
import train_gpt_mlx_segmentlong as segmentlong


class Hyperparameters(segmentlong.Hyperparameters):
    pass


def loss_and_grad_batch(
    args: Hyperparameters,
    train_loader,
    compiled_loss_and_grad,
    *,
    aux_scale: float,
    pred_weight: float,
    sigreg_weight: float,
    spherical_weight: float,
) -> tuple[mx.array, dict, tuple[mx.array, mx.array] | None]:
    x_np, y_np = train_loader.next_batch_np(args.microbatch_tokens, args.train_seq_len)
    chunk_sizes = base.token_chunks(
        args.microbatch_tokens,
        args.train_seq_len,
        args.mlx_max_microbatch_tokens,
    )
    rows_per_chunk = [chunk_tokens // args.train_seq_len for chunk_tokens in chunk_sizes]
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    last_batch: tuple[mx.array, mx.array] | None = None
    row_start = 0
    for rows in rows_per_chunk:
        row_end = row_start + rows
        x = mx.array(x_np[row_start:row_end], dtype=mx.int32)
        y = mx.array(y_np[row_start:row_end], dtype=mx.int32)
        last_batch = (x, y)
        loss, grads = compiled_loss_and_grad(
            x,
            y,
            aux_scale,
            pred_weight,
            sigreg_weight,
            spherical_weight,
        )
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
        row_start = row_end
    assert grad_accum is not None
    return loss_value, tree_unflatten(list(grad_accum.items())), last_batch


def validate_supported_features(args: Hyperparameters) -> None:
    unsupported: list[str] = []
    if args.curriculum_enabled and args.curriculum_apply_focal:
        unsupported.append("CURRICULUM_APPLY_FOCAL=1")
    if args.curriculum_token_category_weighting:
        unsupported.append("CURRICULUM_TOKEN_CATEGORY_WEIGHTING=1")
    if args.curriculum_context_delta_weighting:
        unsupported.append("CURRICULUM_CONTEXT_DELTA_WEIGHTING=1")
    if args.structural_branching_enabled:
        unsupported.append("STRUCTURAL_BRANCHING_ENABLED=1")
    if args.ema_enabled:
        unsupported.append("EMA_ENABLED=1")
    if args.ema_teacher_distill_enabled:
        unsupported.append("EMA_TEACHER_DISTILL_ENABLED=1")
    if args.external_teacher_distill_weight > 0.0:
        unsupported.append("EXTERNAL_TEACHER_DISTILL_WEIGHT>0")
    if args.external_teacher_hidden_distill_weight > 0.0:
        unsupported.append("EXTERNAL_TEACHER_HIDDEN_DISTILL_WEIGHT>0")
    if args.adaptive_train_controller:
        unsupported.append("ADAPTIVE_TRAIN_CONTROLLER=1")
    if args.longctx_persist_train_carry:
        unsupported.append("LONGCTX_PERSIST_TRAIN_CARRY=1")
    if args.sidecar_eval_persistent:
        unsupported.append("SIDECAR_EVAL_PERSISTENT=1")
    if args.num_registers != 0:
        unsupported.append("NUM_REGISTERS!=0")
    if args.logic_dim != 0:
        unsupported.append("LOGIC_DIM!=0")
    if args.polarity_detector_enabled:
        unsupported.append("POLARITY_DETECTOR_ENABLED=1")
    if args.sidecar_polarity_write:
        unsupported.append("SIDECAR_POLARITY_WRITE=1")
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "train_gpt_mlx_segmentlong_grouped_slim.py only supports the no-carry "
            "segment-memory/Turbo/current-size lane plus static early-exit and curriculum gating. "
            f"Unsupported: {joined}"
        )


def total_objective_matches_ce(args: Hyperparameters) -> bool:
    return (
        args.sidecar_pred_weight == 0.0
        and args.sidecar_sigreg_weight == 0.0
        and args.sidecar_spherical_weight == 0.0
        and args.early_exit_aux_weight == 0.0
        and not args.early_exit_branch_draft_enabled
    )


def main() -> None:
    args = Hyperparameters()
    validate_supported_features(args)
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
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_name, actual_train_files, expected_train_files = base.validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = base.limit_validation_tokens(
        base.load_validation_tokens(args.val_files, args.train_seq_len),
        args.train_seq_len,
        args.val_max_seqs,
    )
    quant_eval_tokens = base.limit_validation_tokens(val_tokens, args.train_seq_len, args.quant_eval_max_seqs)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(sp, args.vocab_size)

    batch_seqs = args.microbatch_tokens // args.train_seq_len
    num_streams = args.longctx_num_streams if args.longctx_num_streams > 0 else batch_seqs
    if num_streams != batch_seqs and not args.curriculum_enabled:
        raise ValueError(
            f"LONGCTX_NUM_STREAMS must match microbatch batch size for grouped slim segmentlong, got "
            f"{num_streams} vs {batch_seqs}"
        )

    mx.random.seed(args.seed)
    if args.curriculum_enabled:
        train_loader = base.build_train_loader(args, log_fn=log, dataset_name=dataset_name)
    else:
        train_loader = segmentlong.GroupedStreamingTokenLoader(
            args.train_files,
            num_streams=num_streams,
            seq_len=args.train_seq_len,
            log_fn=log,
            dataset_name=dataset_name,
        )
    model = segmentlong.make_segmentlong_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: segmentlong.GPTSegmentLong | None = None
    objective_matches_ce = total_objective_matches_ce(args)

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    if compiled:
        log("mlx_compile:disable reason:segmentlong_grouped_slim_not_compile_hardened")
        compiled = False

    objective_name = "ce_loss_streaming_fastpath" if objective_matches_ce else "loss_terms_streaming_total"
    if objective_matches_ce:
        compiled_loss_and_grad_impl = nn.value_and_grad(
            model,
            lambda x, y, aux_scale, pred_w, sig_w, sph_w: model.ce_loss_streaming(x, y),
        )
    else:
        compiled_loss_and_grad_impl = nn.value_and_grad(
            model,
            lambda x, y, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
                x,
                y,
                initial_sidecar_state=None,
                aux_scale=aux_scale,
                pred_weight=pred_w,
                sigreg_weight=sig_w,
                spherical_weight=sph_w,
            )[0],
        )
    compiled_loss_and_grad = (
        lambda x, y, aux_scale, pred_w, sig_w, sph_w:
        compiled_loss_and_grad_impl(x, y, aux_scale, pred_w, sig_w, sph_w)
    )
    compiled_ce_loss = lambda x, y, operator_codes=None: model.ce_loss_streaming(x, y)
    compiled_forward_logits = lambda x, operator_codes=None: model.forward_logits_with_sidecar_state(x)[0]
    compiled_loss_components = (
        lambda x, y, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
            x,
            y,
            initial_sidecar_state=None,
            aux_scale=aux_scale,
            pred_weight=pred_w,
            sigreg_weight=sig_w,
            spherical_weight=sph_w,
        )
    )

    n_params = sum(int(np.prod(param.shape)) for _, param in base.tree_flatten(model.trainable_parameters()))
    pred_offsets = segmentlong.parse_positive_int_tuple(
        args.longctx_segment_pred_offsets,
        (max(int(args.sidecar_pred_offset), 1),),
    )
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
        f"layer_templates:{args.num_layer_templates} dim:{args.model_dim} heads:{args.num_heads} "
        f"kv_heads:{args.num_kv_heads} seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"grouped_slim_segment:streams:{num_streams} grouped_active:{int(not args.curriculum_enabled)} "
        f"train_objective:{objective_name}"
    )
    log(
        f"segmentlong:no_carry_only=1 segment_len:{args.longctx_segment_len} pred_offsets:{pred_offsets} "
        f"enable_sidecar_read:{int(args.longctx_enable_sidecar_read)} tap_layer:{args.sidecar_tap_layer} "
        f"state_dim:{args.sidecar_state_dim} loss_stride:{args.sidecar_loss_stride}"
    )
    log(
        f"loss_weights:ce=1.0 sidecar_pred:{args.sidecar_pred_weight} "
        f"sigreg:{args.sidecar_sigreg_weight} spherical:{args.sidecar_spherical_weight}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{batch_seqs} "
        f"val_batch_size:{args.val_batch_size} val_seqs:{(val_tokens.size - 1) // args.train_seq_len} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:{opt.matrix_optimizer_name}+adam matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log(
        f"compute_dtype:{base.COMPUTE_DTYPE} compile:{compiled} compile_mode:{args.mlx_compile} "
        f"quant_format:{args.quant_format}"
    )
    if args.curriculum_enabled and hasattr(train_loader, "summary"):
        summary = train_loader.summary()
        summary_parts = [f"{key}:{value}" for key, value in summary.items()]
        if summary_parts:
            log("curriculum_loader:" + " ".join(summary_parts))
        log("grouped_slim_segment:curriculum_enabled grouped_streaming_bypassed=1")
    if args.curriculum_enabled:
        log(
            f"curriculum:enabled features:{args.curriculum_features_path or '-'} "
            f"phase_plan:{args.curriculum_phase_plan_path or 'default'} "
            f"apply_jepa_gate:{int(args.curriculum_apply_jepa_phase_gating)} "
            f"apply_qat_gate:{int(args.curriculum_apply_qat_phase_gating)} "
            f"apply_ema_gate:{int(args.curriculum_apply_ema_phase_gating)} "
            f"apply_focal:{int(args.curriculum_apply_focal)}"
        )
    if args.early_exit_aux_weight > 0.0 or args.early_exit_branch_draft_enabled:
        log(
            f"early_exit:layer:{base.resolve_early_exit_layer_index(args.early_exit_layer_index, args.num_layers)} "
            f"horizons:{args.early_exit_horizons} aux_weight:{args.early_exit_aux_weight:.3f} "
            f"head_init_std:{args.early_exit_head_init_std:.4f} "
            f"branch_draft:{int(args.early_exit_branch_draft_enabled)} "
            f"branch_conf_threshold:{args.early_exit_branch_conf_threshold:.3f} "
            f"branch_max_draft_tokens:{args.early_exit_branch_max_draft_tokens}"
        )

    def eval_val_for_model(
        eval_model: segmentlong.GPTSegmentLong,
        eval_tokens: np.ndarray,
        *,
        log_eval_progress: bool = False,
    ) -> tuple[float, float]:
        return base.eval_val(
            args,
            eval_model,
            lambda x, y, operator_codes=None: eval_model.ce_loss_streaming(x, y),
            lambda x, operator_codes=None: eval_model.forward_logits_with_sidecar_state(x)[0],
            eval_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log if log_eval_progress else None,
        )

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_quant_eval = args.quant_eval_every > 0 and (last_step or step % args.quant_eval_every == 0)
        if should_validate or should_quant_eval:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if should_validate:
                val_loss, val_bpb = eval_val_for_model(model, val_tokens, log_eval_progress=True)
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = segmentlong.make_segmentlong_gpt(args, sp)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val_for_model(model, quant_eval_tokens)
                model.clear_turbo_cache()
                flat_state = segmentlong.exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = eval_val_for_model(quant_eval_model, quant_eval_tokens)
                log(
                    f"step:{step}/{args.iterations} quant_diag_seqs:{(quant_eval_tokens.size - 1) // args.train_seq_len} "
                    f"raw_val_loss:{raw_q_val_loss:.4f} raw_val_bpb:{raw_q_val_bpb:.4f} "
                    f"quant_val_loss:{q_val_loss:.4f} quant_val_bpb:{q_val_bpb:.4f} "
                    f"quant_gap_bpb:{q_val_bpb - raw_q_val_bpb:+.4f} int8_zlib_bytes:{len(quant_blob)} "
                    f"payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)} "
                    f"{base.format_quant_stats(quant_stats)} "
                    f"eval_time:{1000.0 * (time.perf_counter() - q_t0):.0f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        progress_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        lr_mul = args.lr_mul(step, progress_ms)
        if args.curriculum_enabled and hasattr(train_loader, "begin_step"):
            curriculum_phase = train_loader.begin_step()
        else:
            curriculum_phase = (
                train_loader.current_phase()
                if args.curriculum_enabled and hasattr(train_loader, "current_phase")
                else None
            )
        curriculum_jepa_enabled = (
            curriculum_phase.enable_jepa
            if curriculum_phase is not None and args.curriculum_apply_jepa_phase_gating
            else True
        )
        qat_scale = base.turbo_qat_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        if args.curriculum_apply_qat_phase_gating and curriculum_phase is not None and not curriculum_phase.enable_qat:
            qat_scale = 0.0
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        aux_scale = segmentlong.sidecar.sidecar_aux_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        if args.curriculum_apply_jepa_phase_gating and curriculum_phase is not None and not curriculum_phase.enable_jepa:
            aux_scale = 0.0
        pred_weight = args.sidecar_pred_weight
        sigreg_weight = args.sidecar_sigreg_weight
        spherical_weight = args.sidecar_spherical_weight
        model.set_turbo_qat(qat_active, qat_scale)
        if qat_active:
            if objective_matches_ce:
                step_loss_and_grad = (
                    lambda x, y, aux_scale_inner, pred_w, sig_w, sph_w: nn.value_and_grad(
                        model,
                        lambda x_inner, y_inner, aux_scale_param, pred_param, sig_param, sph_param:
                        model.ce_loss_streaming(x_inner, y_inner) + qat_lambda * model.turbo_regularizer(),
                    )(x, y, aux_scale_inner, pred_w, sig_w, sph_w)
                )
            else:
                step_loss_and_grad = (
                    lambda x, y, aux_scale_inner, pred_w, sig_w, sph_w: nn.value_and_grad(
                        model,
                        lambda x_inner, y_inner, aux_scale_param, pred_param, sig_param, sph_param:
                        model.loss_terms_streaming(
                            x_inner,
                            y_inner,
                            initial_sidecar_state=None,
                            aux_scale=aux_scale_param,
                            pred_weight=pred_param,
                            sigreg_weight=sig_param,
                            spherical_weight=sph_param,
                        )[0] + qat_lambda * model.turbo_regularizer(),
                    )(x, y, aux_scale_inner, pred_w, sig_w, sph_w)
                )
        else:
            step_loss_and_grad = compiled_loss_and_grad

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        last_batch: tuple[mx.array, mx.array] | None = None
        grad_scale = 1.0 / args.grad_accum_steps
        step_t0 = time.perf_counter()
        for _ in range(args.grad_accum_steps):
            train_loss, grads, last_batch = loss_and_grad_batch(
                args,
                train_loader,
                step_loss_and_grad,
                aux_scale=aux_scale,
                pred_weight=pred_weight,
                sigreg_weight=sigreg_weight,
                spherical_weight=spherical_weight,
            )
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
        if args.curriculum_enabled and hasattr(train_loader, "end_step"):
            train_loader.end_step()
        grads_tree = tree_unflatten(list(accum.items()))
        train_loss_value = float(train_loss.item())
        sanitize_this_step = base.should_sanitize_nonfinite_grads(args, step, train_loss_value)
        if sanitize_this_step:
            grads_tree, grad_nonfinite = base.sanitize_grad_tree(grads_tree, topk=args.nonfinite_grad_topk)
        else:
            grad_nonfinite = base.empty_nonfinite_grad_summary()
        grads_tree = base.clip_grad_tree(grads_tree, args.grad_clip_norm)
        flat_clipped_grads = dict(base.tree_flatten(grads_tree))
        opt.step(model, grads_tree, step=step, lr_mul=lr_mul)
        model.clear_turbo_cache()
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
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
            tensor_activity_str = ""
            should_log_tensor_activity = (
                args.tensor_activity_log_every > 0
                and (step <= 10 or step % args.tensor_activity_log_every == 0 or stop_after_step is not None)
            )
            if should_log_tensor_activity:
                activity = base.tensor_activity_snapshot(
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
                top_nonfinite = ",".join(f"{name}:{count}" for count, name in grad_nonfinite["top"])
                grad_nonfinite_str = (
                    f" grad_nonfinite_tensors:{grad_nonfinite['nonfinite_tensors']}/{grad_nonfinite['tensor_count']}"
                    f" grad_nonfinite_param_frac:{int(grad_nonfinite['nonfinite_params']) / total_params:.3f}"
                )
                if top_nonfinite:
                    grad_nonfinite_str += f" grad_nonfinite_top:{top_nonfinite}"
            extra = ""
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                metrics = compiled_loss_components(
                    last_batch[0],
                    last_batch[1],
                    aux_scale,
                    pred_weight,
                    sigreg_weight,
                    spherical_weight,
                )
                mx.eval(*metrics)
                _, ce_metric, pred_metric, sig_metric, sph_metric, early_exit_metric, _ = metrics
                extra = (
                    f" ce:{float(ce_metric.item()):.4f}"
                    f" sidecar_pred:{float(pred_metric.item()):.4f}"
                    f" sidecar_sigreg:{float(sig_metric.item()):.4f}"
                    f" sidecar_spherical:{float(sph_metric.item()):.4f}"
                    f" early_exit:{float(early_exit_metric.item()):.4f}"
                )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.0f} "
                f"curriculum_phase:{curriculum_phase.name if curriculum_phase is not None else 'off'} "
                f"curriculum_jepa:{int(curriculum_jepa_enabled)} "
                f"sidecar_aux_scale:{aux_scale:.3f} "
                f"sidecar_pred_weight:{pred_weight:.4f} "
                f"sidecar_sigreg_weight:{sigreg_weight:.4f} "
                f"sidecar_spherical_weight:{spherical_weight:.4f} "
                f"turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f}"
                f"{grad_nonfinite_str}{tensor_activity_str}{extra}"
                f"{curriculum_metrics_str}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    flat_state = segmentlong.exportable_flat_state(model)
    mx.savez(str(out_path), **flat_state)
    model_bytes = out_path.stat().st_size
    code_bytes = len(code.encode("utf-8"))
    log(f"Serialized model: {model_bytes} bytes")
    log(f"Code size: {code_bytes} bytes")
    log(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.pkl.z"
    quant_path.write_bytes(quant_blob)
    log(
        f"Serialized model int8+zlib: {len(quant_blob)} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)})"
    )
    log(f"Total submission size int8+zlib: {len(quant_blob) + code_bytes} bytes")

    exact_val_loss, exact_val_bpb = eval_val_for_model(model, val_tokens, log_eval_progress=True)
    log(f"Final exact val loss: {exact_val_loss:.8f}")
    log(f"Final exact val bpb: {exact_val_bpb:.8f}")

    roundtrip_model = segmentlong.make_segmentlong_gpt(args, sp)
    roundtrip_model.clear_turbo_cache()
    roundtrip_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
    roundtrip_val_loss, roundtrip_val_bpb = eval_val_for_model(roundtrip_model, val_tokens, log_eval_progress=True)
    log(f"Final exact roundtrip int8+zlib val loss: {roundtrip_val_loss:.8f}")
    log(f"Final exact roundtrip int8+zlib val bpb: {roundtrip_val_bpb:.8f}")


if __name__ == "__main__":
    main()
