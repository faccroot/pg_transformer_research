#!/usr/bin/env python3
"""
Super-long-context research trainer for Parameter Golf.

This is a V1 prototype for the "local exact tokens + long predictive state"
architecture:

- the transformer still processes exact local token order inside TRAIN_SEQ_LEN
- a token-level causal sidecar state is carried across training windows
- the sidecar is trained with JEPA-style future-state prediction on a subsampled
  stride, not next-token reconstruction
- training data is consumed as persistent stream lanes, so each batch row is the
  continuation of the same lane from the previous step

This version intentionally keeps the scope narrow:
- no logic/register path
- no polarity detector
- no eval-time adaptation
- no sparse retrieval yet

The goal is to test whether persistent predictive state across windows is a real
long-context lever, not to solve every memory mechanism at once.
"""
from __future__ import annotations

import math
import os
import sys
import time
import zlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_unflatten

import train_gpt_mlx as base
import train_gpt_mlx_jepa_sidecar as sidecar


class Hyperparameters(sidecar.Hyperparameters):
    longctx_num_streams: int = int(os.environ.get("LONGCTX_NUM_STREAMS", "0"))
    longctx_reset_carry_every_steps: int = int(os.environ.get("LONGCTX_RESET_CARRY_EVERY_STEPS", "0"))
    longctx_persist_train_carry: bool = bool(int(os.environ.get("LONGCTX_PERSIST_TRAIN_CARRY", "1")))
    longctx_enable_sidecar_read: bool = bool(int(os.environ.get("LONGCTX_ENABLE_SIDECAR_READ", "1")))
    sidecar_loss_stride: int = int(os.environ.get("SIDECAR_LOSS_STRIDE", os.environ.get("SIDECAR_CHUNK_SIZE", "8")))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class GroupedStreamingTokenLoader:
    """Persistent stream lanes for truncated BPTT-style training.

    Each lane emits contiguous TRAIN_SEQ_LEN windows with a one-token overlap
    between successive windows so the next window starts with the previous
    target token.
    """

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

    def reset_windows(self) -> None:
        self.windows = [self.stream.take(self.seq_len + 1) for _ in range(self.num_streams)]

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


class GPTSuperLong(sidecar.GPTJEPASidecar):
    def __init__(self, *args, longctx_enable_sidecar_read: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.longctx_enable_sidecar_read = bool(longctx_enable_sidecar_read)

    def sidecar_token_condition(
        self,
        side_states: mx.array,
        token_len: int,
        initial_state: mx.array | None = None,
        reset_mask: mx.array | None = None,
    ) -> mx.array:
        if not self.longctx_enable_sidecar_read:
            return mx.zeros((side_states.shape[0], token_len, self.tok_emb.weight.shape[1]), dtype=base.COMPUTE_DTYPE)
        return super().sidecar_token_condition(
            side_states,
            token_len,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )

    def loss_terms_streaming(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        *,
        initial_sidecar_state: mx.array | None = None,
        aux_scale: float = 1.0,
        pred_weight: float | None = None,
        sigreg_weight: float | None = None,
        spherical_weight: float | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        pred_weight = self.sidecar_pred_weight if pred_weight is None else pred_weight
        sigreg_weight = self.sidecar_sigreg_weight if sigreg_weight is None else sigreg_weight
        spherical_weight = self.sidecar_spherical_weight if spherical_weight is None else spherical_weight
        final_hidden, _tap_hidden, aux = self.forward_with_sidecar_hidden_aux(
            input_ids,
            initial_sidecar_state=initial_sidecar_state,
        )
        ce_loss = self.token_ce_from_hidden(final_hidden, target_ids)
        zero = mx.array(0.0, dtype=mx.float32)
        if aux_scale <= 0.0:
            pred_loss, sigreg_loss, spherical_loss = zero, zero, zero
        else:
            side_states = aux["sidecar_states"]
            assert side_states is not None
            pred_loss, sigreg_loss, spherical_loss = self.sidecar_terms_from_states(side_states)
        total = ce_loss + mx.array(aux_scale, dtype=mx.float32) * (
            pred_weight * pred_loss + sigreg_weight * sigreg_loss + spherical_weight * spherical_loss
        )
        final_state = aux["final_sidecar_state"]
        assert final_state is not None
        return total, ce_loss, pred_loss, sigreg_loss, spherical_loss, mx.stop_gradient(final_state)

    def ce_loss_streaming(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        *,
        initial_sidecar_state: mx.array | None = None,
    ) -> mx.array:
        final_hidden, _tap_hidden, _aux = self.forward_with_sidecar_hidden_aux(
            input_ids,
            initial_sidecar_state=initial_sidecar_state,
        )
        return self.token_ce_from_hidden(final_hidden, target_ids)

    def final_sidecar_state(
        self,
        input_ids: mx.array,
        *,
        initial_sidecar_state: mx.array | None = None,
    ) -> mx.array:
        _logits, final_state, _aux = self.forward_logits_with_sidecar_state(
            input_ids,
            initial_sidecar_state=initial_sidecar_state,
        )
        return mx.stop_gradient(final_state)


def make_superlong_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTSuperLong:
    model = GPTSuperLong(
        **base.gpt_kwargs_from_args(args, sp),
        longctx_enable_sidecar_read=args.longctx_enable_sidecar_read,
        sidecar_chunk_size=args.sidecar_chunk_size,
        sidecar_tap_layer=args.sidecar_tap_layer,
        sidecar_state_dim=args.sidecar_state_dim,
        sidecar_pred_hidden=args.sidecar_pred_hidden,
        sidecar_pred_offset=args.sidecar_pred_offset,
        sidecar_pred_weight=args.sidecar_pred_weight,
        sidecar_sigreg_weight=args.sidecar_sigreg_weight,
        sidecar_spherical_weight=args.sidecar_spherical_weight,
        sidecar_summary_mode=args.sidecar_summary_mode,
        sidecar_pred_target_mode=args.sidecar_pred_target_mode,
        sidecar_read_init_std=args.sidecar_read_init_std,
        sidecar_pred_init_std=args.sidecar_pred_init_std,
        sidecar_sigreg_knots=args.sidecar_sigreg_knots,
        sidecar_sigreg_num_proj=args.sidecar_sigreg_num_proj,
        sidecar_sigreg_seed=args.sidecar_sigreg_seed,
        sidecar_sigreg_resample_proj=args.sidecar_sigreg_resample_proj,
        sidecar_sigreg_mode=args.sidecar_sigreg_mode,
        sidecar_weak_sigreg_dim=args.sidecar_weak_sigreg_dim,
        sidecar_read_rmsnorm=args.sidecar_read_rmsnorm,
        sidecar_polarity_write=args.sidecar_polarity_write,
        sidecar_polarity_pool=args.sidecar_polarity_pool,
        sidecar_reset_on_bos=args.sidecar_reset_on_bos,
        sidecar_reset_token_id=args.sidecar_reset_token_id,
    )
    model.sidecar_loss_stride = max(int(args.sidecar_loss_stride), 1)
    return model


def make_sidecar_gpt(args: Hyperparameters, sp: spm.SentencePieceProcessor | None = None) -> GPTSuperLong:
    return make_superlong_gpt(args, sp)


def exportable_flat_state(model: GPTSuperLong) -> dict[str, mx.array]:
    return sidecar.exportable_flat_state(model)


def normalize_carry_state(
    carry_state: mx.array | None,
    batch_seqs: int,
    state_dim: int,
) -> mx.array:
    if carry_state is None:
        return mx.zeros((batch_seqs, state_dim), dtype=base.COMPUTE_DTYPE)
    if int(carry_state.shape[0]) != batch_seqs or int(carry_state.shape[1]) != state_dim:
        raise ValueError(
            f"carry_state shape {tuple(carry_state.shape)} does not match "
            f"(batch_seqs={batch_seqs}, state_dim={state_dim})"
        )
    return carry_state.astype(base.COMPUTE_DTYPE)


def loss_and_grad_streaming(
    args: Hyperparameters,
    model: GPTSuperLong,
    train_loader: GroupedStreamingTokenLoader,
    compiled_loss_and_grad,
    compiled_final_state,
    carry_state: mx.array | None,
    *,
    aux_scale: float,
    pred_weight: float,
    sigreg_weight: float,
    spherical_weight: float,
) -> tuple[mx.array, dict, mx.array, tuple[mx.array, mx.array, mx.array] | None]:
    x_np, y_np = train_loader.next_batch_np(args.microbatch_tokens, args.train_seq_len)
    batch_seqs = int(x_np.shape[0])
    carry_state = normalize_carry_state(carry_state, batch_seqs, model.sidecar_state_dim)
    chunk_sizes = base.token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    rows_per_chunk = [chunk_tokens // args.train_seq_len for chunk_tokens in chunk_sizes]
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    final_state_slices: list[mx.array] = []
    last_batch: tuple[mx.array, mx.array, mx.array] | None = None
    row_start = 0
    for rows in rows_per_chunk:
        row_end = row_start + rows
        x = mx.array(x_np[row_start:row_end], dtype=mx.int32)
        y = mx.array(y_np[row_start:row_end], dtype=mx.int32)
        init_state = carry_state[row_start:row_end]
        last_batch = (x, y, init_state)
        loss, grads = compiled_loss_and_grad(
            x,
            y,
            init_state,
            aux_scale,
            pred_weight,
            sigreg_weight,
            spherical_weight,
        )
        final_state = compiled_final_state(x, init_state)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
        final_state_slices.append(final_state)
        row_start = row_end
    new_carry_state = mx.concatenate(final_state_slices, axis=0) if final_state_slices else carry_state
    return loss_value, tree_unflatten(list(grad_accum.items())), new_carry_state, last_batch


def main() -> None:
    args = Hyperparameters()
    if args.num_registers != 0 or args.logic_dim != 0 or args.polarity_detector_enabled or args.sidecar_polarity_write:
        raise ValueError("train_gpt_mlx_superlong.py is a clean long-context prototype; disable registers/logic/polarity")
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
    if num_streams != batch_seqs:
        raise ValueError(
            f"LONGCTX_NUM_STREAMS must match microbatch_batch_size for V1, got {num_streams} vs {batch_seqs}"
        )

    mx.random.seed(args.seed)
    train_loader = GroupedStreamingTokenLoader(
        args.train_files,
        num_streams=num_streams,
        seq_len=args.train_seq_len,
        log_fn=log,
        dataset_name=dataset_name,
    )
    model = make_superlong_gpt(args, sp)
    model.set_turbo_qat(False, 0.0)
    opt = base.SplitOptimizers(model, args)
    quant_eval_model: GPTSuperLong | None = None

    compiled = base.resolve_mlx_compile(args.mlx_compile, args.turbo_qat)
    if compiled:
        compiled_ce_loss = mx.compile(
            lambda x, y: model.ce_loss(x, y),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_forward_logits = mx.compile(
            lambda x: model.forward_logits(x),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_components = mx.compile(
            lambda x, y, init, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
                x,
                y,
                initial_sidecar_state=init,
                aux_scale=aux_scale,
                pred_weight=pred_w,
                sigreg_weight=sig_w,
                spherical_weight=sph_w,
            ),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_loss_and_grad = mx.compile(
            nn.value_and_grad(
                model,
                lambda x, y, init, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
                    x,
                    y,
                    initial_sidecar_state=init,
                    aux_scale=aux_scale,
                    pred_weight=pred_w,
                    sigreg_weight=sig_w,
                    spherical_weight=sph_w,
                )[0],
            ),
            inputs=model.state,
            outputs=model.state,
        )
        compiled_final_state = mx.compile(
            lambda x, init: model.final_sidecar_state(x, initial_sidecar_state=init),
            inputs=model.state,
            outputs=model.state,
        )
    else:
        compiled_ce_loss = lambda x, y: model.ce_loss(x, y)
        compiled_forward_logits = lambda x: model.forward_logits(x)
        compiled_loss_components = lambda x, y, init, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
            x,
            y,
            initial_sidecar_state=init,
            aux_scale=aux_scale,
            pred_weight=pred_w,
            sigreg_weight=sig_w,
            spherical_weight=sph_w,
        )
        compiled_loss_and_grad = nn.value_and_grad(
            model,
            lambda x, y, init, aux_scale, pred_w, sig_w, sph_w: model.loss_terms_streaming(
                x,
                y,
                initial_sidecar_state=init,
                aux_scale=aux_scale,
                pred_weight=pred_w,
                sigreg_weight=sig_w,
                spherical_weight=sph_w,
            )[0],
        )
        compiled_final_state = lambda x, init: model.final_sidecar_state(x, initial_sidecar_state=init)

    n_params = sum(int(np.prod(param.shape)) for _, param in tree_flatten(model.trainable_parameters()))
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
        f"superlong:streams:{num_streams} reset_carry_every_steps:{args.longctx_reset_carry_every_steps} "
        f"persist_train_carry:{int(args.longctx_persist_train_carry)} "
        f"enable_sidecar_read:{int(args.longctx_enable_sidecar_read)} "
        f"tap_layer:{args.sidecar_tap_layer} state_dim:{args.sidecar_state_dim} "
        f"loss_stride:{args.sidecar_loss_stride} pred_offset:{args.sidecar_pred_offset}"
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

    if args.warmup_steps > 0:
        carry_state: mx.array | None = None
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, carry_state, _ = loss_and_grad_streaming(
                    args,
                    model,
                    train_loader,
                    compiled_loss_and_grad,
                    compiled_final_state,
                    carry_state,
                    aux_scale=1.0,
                    pred_weight=args.sidecar_pred_weight,
                    sigreg_weight=args.sidecar_sigreg_weight,
                    spherical_weight=args.sidecar_spherical_weight,
                )
                accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum, carry_state)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        train_loader = GroupedStreamingTokenLoader(
            args.train_files,
            num_streams=num_streams,
            seq_len=args.train_seq_len,
            log_fn=log,
            dataset_name=dataset_name,
        )

    def eval_val_for_model(eval_model: GPTSuperLong, eval_tokens: np.ndarray, *, log_eval_progress: bool = False) -> tuple[float, float]:
        if args.sidecar_eval_persistent:
            return sidecar.eval_val_sidecar_persistent(
                args,
                eval_model,
                eval_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log if log_eval_progress else None,
            )
        return base.eval_val(
            args,
            eval_model,
            lambda x, y, operator_codes=None: compiled_ce_loss(x, y),
            lambda x, operator_codes=None: compiled_forward_logits(x),
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
    carry_state: mx.array | None = None

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
                    quant_eval_model = make_superlong_gpt(args, sp)
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = eval_val_for_model(model, quant_eval_tokens)
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
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

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        aux_scale = sidecar.sidecar_aux_scale_for_progress(args, step, train_time_ms, max_wallclock_ms)
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        last_batch: tuple[mx.array, mx.array, mx.array] | None = None
        grad_scale = 1.0 / args.grad_accum_steps
        step_t0 = time.perf_counter()
        for _ in range(args.grad_accum_steps):
            active_carry = carry_state if args.longctx_persist_train_carry else None
            train_loss, grads, carry_state, last_batch = loss_and_grad_streaming(
                args,
                model,
                train_loader,
                compiled_loss_and_grad,
                compiled_final_state,
                active_carry,
                aux_scale=aux_scale,
                pred_weight=args.sidecar_pred_weight,
                sigreg_weight=args.sidecar_sigreg_weight,
                spherical_weight=args.sidecar_spherical_weight,
            )
            if not args.longctx_persist_train_carry:
                carry_state = None
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
        grads_tree = tree_unflatten(list(accum.items()))
        grads_tree = base.clip_grad_tree(grads_tree, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads_tree, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.longctx_reset_carry_every_steps > 0 and step % args.longctx_reset_carry_every_steps == 0:
            carry_state = None
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            extra = ""
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                metrics = compiled_loss_components(
                    last_batch[0],
                    last_batch[1],
                    last_batch[2],
                    aux_scale,
                    args.sidecar_pred_weight,
                    args.sidecar_sigreg_weight,
                    args.sidecar_spherical_weight,
                )
                mx.eval(*metrics)
                _, ce_metric, pred_metric, sig_metric, sph_metric, _ = metrics
                extra = (
                    f" ce:{float(ce_metric.item()):.4f}"
                    f" sidecar_pred:{float(pred_metric.item()):.4f}"
                    f" sidecar_sigreg:{float(sig_metric.item()):.4f}"
                    f" sidecar_spherical:{float(sph_metric.item()):.4f}"
                )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.0f} aux_scale:{aux_scale:.3f}{extra}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    model.set_turbo_qat(False, 0.0)
    model.clear_turbo_cache()
    flat_state = exportable_flat_state(model)
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

    quant_roundtrip = make_superlong_gpt(args, sp)
    quant_roundtrip.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
    quant_roundtrip.clear_turbo_cache()
    raw_val_loss, raw_val_bpb = eval_val_for_model(model, val_tokens)
    q_val_loss, q_val_bpb = eval_val_for_model(quant_roundtrip, val_tokens)
    log(f"final_raw_exact_val_loss:{raw_val_loss:.8f} final_raw_exact_val_bpb:{raw_val_bpb:.8f}")
    log(f"final_int8_zlib_roundtrip_exact_val_loss:{q_val_loss:.8f} final_int8_zlib_roundtrip_exact_val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
