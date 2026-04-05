#!/usr/bin/env python3
"""
Slim grouped-streaming plain-GPT control for harmonic probes.

This keeps the lightweight training loop and grouped streaming schedule from the
harmonic prototype, but removes the harmonic model path entirely. It exists to
separate:

- grouped streaming schedule / slim trainer loop effects
- from the GPTHarmonic model-class and inactive harmonic branch
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_unflatten

import train_gpt_mlx as base


class Hyperparameters(base.Hyperparameters):
    longctx_num_streams: int = int(os.environ.get("LONGCTX_NUM_STREAMS", "0"))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class GroupedStreamingTokenLoader:
    """Persistent stream lanes for contiguous local-window training."""

    def __init__(
        self,
        pattern: str,
        *,
        num_streams: int,
        seq_len: int,
        log_fn=None,
        dataset_name: str = "",
    ):
        if num_streams <= 0:
            raise ValueError(f"LONGCTX_NUM_STREAMS must be > 0, got {num_streams}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.stream = base.TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)
        self.num_streams = int(num_streams)
        self.seq_len = int(seq_len)
        self.windows = [self.stream.take(self.seq_len + 1) for _ in range(self.num_streams)]

    def summary(self) -> dict[str, object]:
        return {
            "mode": "grouped_streaming",
            "num_streams": int(self.num_streams),
            "seq_len": int(self.seq_len),
        }

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        if seq_len != self.seq_len:
            raise ValueError(f"loader seq_len={self.seq_len} does not match request seq_len={seq_len}")
        if batch_seqs != self.num_streams:
            raise ValueError(
                f"grouped streaming expects batch_seqs == num_streams, got batch_seqs={batch_seqs} "
                f"num_streams={self.num_streams}"
            )
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for lane_idx in range(self.num_streams):
            window = self.windows[lane_idx]
            x_rows.append(window[:-1])
            y_rows.append(window[1:])
            next_tokens = self.stream.take(self.seq_len)
            self.windows[lane_idx] = np.concatenate((window[-1:], next_tokens), axis=0)
        x = np.stack(x_rows, axis=0).astype(np.int32, copy=False)
        y = np.stack(y_rows, axis=0).astype(np.int32, copy=False)
        return x, y


def loss_and_grad_batch(
    args: Hyperparameters,
    model: base.GPT,
    train_loader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict, tuple[mx.array, mx.array] | None]:
    x_np, y_np = train_loader.next_batch_np(args.microbatch_tokens, args.train_seq_len)
    chunk_sizes = base.token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
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
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
        row_start = row_end
    assert grad_accum is not None
    return loss_value, tree_unflatten(list(grad_accum.items())), last_batch


def main() -> None:
    args = Hyperparameters()
    if args.num_registers != 0 or args.logic_dim != 0 or args.polarity_detector_enabled:
        raise ValueError(
            "train_gpt_mlx_harmonic_slim_control.py is a clean grouped plain-GPT control; "
            "disable registers/logic/polarity"
        )
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
            f"LONGCTX_NUM_STREAMS must match microbatch batch size for grouped control, got "
            f"{num_streams} vs {batch_seqs}"
        )

    mx.random.seed(args.seed)
    if args.curriculum_enabled:
        train_loader = base.build_train_loader(args, log_fn=log, dataset_name=dataset_name)
    else:
        train_loader = GroupedStreamingTokenLoader(
            args.train_files,
            num_streams=num_streams,
            seq_len=args.train_seq_len,
            log_fn=log,
            dataset_name=dataset_name,
        )
    model = base.make_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: base.GPT | None = None

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    if compiled:
        compiled_ce_loss = mx.compile(
            lambda x, y: model.ce_loss(x, y),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_and_grad_impl = mx.compile(
            nn.value_and_grad(model, lambda x, y: model.ce_loss(x, y)),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_and_grad = lambda x, y: compiled_loss_and_grad_impl(x, y)
    else:
        compiled_ce_loss = lambda x, y: model.ce_loss(x, y)
        compiled_loss_and_grad = nn.value_and_grad(model, lambda x, y: model.ce_loss(x, y))

    n_params = sum(int(np.prod(param.shape)) for _, param in base.tree_flatten(model.trainable_parameters()))
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
    log(f"grouped_control:streams:{num_streams}")
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

    def eval_val_for_model(
        eval_model: base.GPT,
        eval_tokens: np.ndarray,
        *,
        log_eval_progress: bool = False,
    ) -> tuple[float, float]:
        return base.eval_val(
            args,
            eval_model,
            lambda x, y, operator_codes=None: eval_model.ce_loss(x, y),
            lambda x, operator_codes=None: eval_model.forward_logits(x),
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
                    quant_eval_model = base.make_gpt(args, sp)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val_for_model(model, quant_eval_tokens)
                model.clear_turbo_cache()
                flat_state = base.exportable_flat_state(model)
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
        qat_scale = base.turbo_qat_scale_for_progress(args, step, progress_ms, max_wallclock_ms)
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        model.set_turbo_qat(qat_active, qat_scale)
        step_loss_and_grad = (
            (
                lambda x, y: nn.value_and_grad(
                    model,
                    lambda x_inner, y_inner: model.ce_loss(x_inner, y_inner) + qat_lambda * model.turbo_regularizer(),
                )(x, y)
            )
            if qat_active
            else compiled_loss_and_grad
        )
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        last_batch: tuple[mx.array, mx.array] | None = None
        grad_scale = 1.0 / args.grad_accum_steps
        step_t0 = time.perf_counter()
        for _ in range(args.grad_accum_steps):
            train_loss, grads, last_batch = loss_and_grad_batch(
                args,
                model,
                train_loader,
                step_loss_and_grad,
            )
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
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
                ce_metric = compiled_ce_loss(last_batch[0], last_batch[1])
                mx.eval(ce_metric)
                extra = f" ce:{float(ce_metric.item()):.4f}"
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.0f} turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f}"
                f"{grad_nonfinite_str}{tensor_activity_str}{extra}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    flat_state = base.exportable_flat_state(model)
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

    quant_roundtrip = base.make_gpt(args, sp)
    quant_roundtrip.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
    quant_roundtrip.clear_turbo_cache()
    raw_val_loss, raw_val_bpb = eval_val_for_model(model, val_tokens)
    q_val_loss, q_val_bpb = eval_val_for_model(quant_roundtrip, val_tokens)
    log(f"final_raw_exact_val_loss:{raw_val_loss:.8f} final_raw_exact_val_bpb:{raw_val_bpb:.8f}")
    log(f"final_int8_zlib_roundtrip_exact_val_loss:{q_val_loss:.8f} final_int8_zlib_roundtrip_exact_val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
