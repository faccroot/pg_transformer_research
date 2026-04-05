#!/usr/bin/env python3
"""
State-token JEPA training path for Parameter Golf on MLX.

This keeps the autoregressive token interface for exact BPB scoring, but makes latent
state a first-class part of the sequence:

- every chunk gets a prior state token before the chunk tokens
- every chunk gets a posterior state token after the chunk tokens
- the transformer processes the mixed sequence jointly
- token CE trains the observation model
- latent prediction + SIGReg train the state dynamics
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import sentencepiece as spm
from mlx.utils import tree_flatten, tree_unflatten

import train_gpt_mlx as base


_STATE_LATENT_DEFAULT = int(os.environ.get("STATE_LATENT_DIM", 192))


class Hyperparameters(base.Hyperparameters):
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 8))
    state_latent_dim: int = _STATE_LATENT_DEFAULT
    state_pred_hidden: int = int(os.environ.get("STATE_PRED_HIDDEN", _STATE_LATENT_DEFAULT * 2))
    state_pred_weight: float = float(os.environ.get("STATE_PRED_WEIGHT", 0.25))
    sigreg_weight: float = float(os.environ.get("SIGREG_WEIGHT", 0.05))
    sigreg_knots: int = int(os.environ.get("SIGREG_KNOTS", 17))
    sigreg_num_proj: int = int(os.environ.get("SIGREG_NUM_PROJ", 256))
    sigreg_seed: int = int(os.environ.get("SIGREG_SEED", 17))
    compile_model: bool = bool(int(os.environ.get("COMPILE_MODEL", "1")))
    out_dir: str = os.environ.get("OUT_DIR", "logs")


class SIGReg(nn.Module):
    def __init__(
        self,
        input_dim: int,
        knots: int = 17,
        num_proj: int = 256,
        seed: int = 17,
    ):
        super().__init__()
        self.num_proj = num_proj
        self.input_dim = input_dim
        t = np.linspace(0.0, 3.0, knots, dtype=np.float32)
        dt = 3.0 / max(knots - 1, 1)
        weights = np.full((knots,), 2.0 * dt, dtype=np.float32)
        if knots > 1:
            weights[0] = dt
            weights[-1] = dt
        window = np.exp(-(t * t) / 2.0, dtype=np.float32)
        rng = np.random.default_rng(seed)
        proj_matrix = rng.standard_normal((input_dim, num_proj), dtype=np.float32)
        proj_matrix = proj_matrix / np.sqrt(np.sum(proj_matrix * proj_matrix, axis=0, keepdims=True) + 1e-6)
        self.t = mx.array(t, dtype=mx.float32)
        self.phi = mx.array(window, dtype=mx.float32)
        self.weights = mx.array(weights * window, dtype=mx.float32)
        # Keep the projection basis fixed so the compiled MLX loss does not depend on runtime RNG.
        self.proj_matrix = mx.array(proj_matrix, dtype=mx.float32)
        self.freeze(recurse=False)

    def __call__(self, proj: mx.array) -> mx.array:
        # proj: (T_chunks, B, D)
        proj32 = proj.astype(mx.float32)
        x_t = mx.expand_dims(proj32 @ self.proj_matrix, axis=-1) * self.t
        cos_mean = mx.mean(mx.cos(x_t), axis=1)
        sin_mean = mx.mean(mx.sin(x_t), axis=1)
        err = mx.square(cos_mean - self.phi) + mx.square(sin_mean)
        statistic = mx.sum(err * self.weights, axis=-1) * float(proj.shape[1])
        return mx.mean(statistic)

class StateSpaceGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        num_layer_templates: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
        chunk_size: int,
        state_latent_dim: int,
        state_pred_hidden: int,
        state_pred_weight: float,
        sigreg_weight: float,
        sigreg_knots: int,
        sigreg_num_proj: int,
        sigreg_seed: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if num_layer_templates <= 0 or num_layer_templates > num_layers:
            raise ValueError(
                f"num_layer_templates must be in [1, num_layers], got {num_layer_templates} for num_layers={num_layers}"
            )
        if chunk_size <= 1:
            raise ValueError(f"chunk_size must be > 1, got {chunk_size}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_layer_templates = num_layer_templates
        self.chunk_size = chunk_size
        self.state_pred_weight = state_pred_weight
        self.sigreg_weight = sigreg_weight
        self.state_latent_dim = state_latent_dim
        self.expanded_token_factor = chunk_size + 2

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            base.Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layer_templates)
        ]
        self.final_norm = base.RMSNormNoWeight()
        self.prior_token = mx.zeros((dim,), dtype=mx.float32)
        self.post_token = mx.zeros((dim,), dtype=mx.float32)
        self.state_proj = base.CastedLinear(dim, state_latent_dim)
        self.state_pred_fc = base.CastedLinear(state_latent_dim, state_pred_hidden)
        self.state_pred_out = base.CastedLinear(state_pred_hidden, state_latent_dim)
        self.state_pred_out.weight = mx.zeros_like(self.state_pred_out.weight)
        self.sigreg = SIGReg(
            input_dim=state_latent_dim,
            knots=sigreg_knots,
            num_proj=sigreg_num_proj,
            seed=sigreg_seed,
        )

        for block in self.blocks:
            block.attn.proj.weight = mx.zeros_like(block.attn.proj.weight)
            block.mlp.proj.weight = mx.zeros_like(block.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(base.COMPUTE_DTYPE)
        self._position_cache: dict[int, tuple[mx.array, mx.array, mx.array]] = {}

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        for block in self.blocks:
            block.set_turbo_qat(enabled, alpha)

    def turbo_regularizer(self) -> mx.array:
        loss = mx.array(0.0, dtype=mx.float32)
        for block in self.blocks:
            loss = loss + block.turbo_regularizer()
        return loss

    def clear_turbo_cache(self) -> None:
        for block in self.blocks:
            block.clear_turbo_cache()

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def block_for_step(self, step: int) -> base.Block:
        return self.blocks[step % self.num_layer_templates]

    def positions_for_seq(self, seq_len: int) -> tuple[mx.array, mx.array, mx.array]:
        cached = self._position_cache.get(seq_len)
        if cached is not None:
            return cached
        if seq_len % self.chunk_size != 0:
            raise ValueError(f"TRAIN_SEQ_LEN={seq_len} must be divisible by CHUNK_SIZE={self.chunk_size}")
        num_chunks = seq_len // self.chunk_size
        base_idx = np.arange(num_chunks, dtype=np.int32) * self.expanded_token_factor
        prior_positions = base_idx
        token_positions = (base_idx[:, None] + 1 + np.arange(self.chunk_size, dtype=np.int32)[None, :]).reshape(-1)
        post_positions = base_idx + self.chunk_size + 1
        out = (
            mx.array(prior_positions, dtype=mx.int32),
            mx.array(token_positions, dtype=mx.int32),
            mx.array(post_positions, dtype=mx.int32),
        )
        self._position_cache[seq_len] = out
        return out

    def augmented_seq_len(self, seq_len: int) -> int:
        if seq_len % self.chunk_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by chunk_size={self.chunk_size}")
        return (seq_len // self.chunk_size) * self.expanded_token_factor

    def build_augmented_sequence(self, input_ids: mx.array) -> mx.array:
        bsz, seq_len = input_ids.shape
        if seq_len % self.chunk_size != 0:
            raise ValueError(f"input_ids length {seq_len} must be divisible by chunk_size={self.chunk_size}")
        num_chunks = seq_len // self.chunk_size
        tok = self.tok_emb(input_ids).astype(base.COMPUTE_DTYPE).reshape(bsz, num_chunks, self.chunk_size, -1)
        prior = mx.broadcast_to(self.prior_token.astype(tok.dtype), (bsz, num_chunks, 1, tok.shape[-1]))
        post = mx.broadcast_to(self.post_token.astype(tok.dtype), (bsz, num_chunks, 1, tok.shape[-1]))
        x = mx.concatenate((prior, tok, post), axis=2).reshape(bsz, num_chunks * self.expanded_token_factor, -1)
        return base.rms_norm(x)

    def hidden_views(self, input_ids: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        bsz, seq_len = input_ids.shape
        x = self.build_augmented_sequence(input_ids)
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.block_for_step(i)(x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.block_for_step(self.num_encoder_layers + i)(x, x0)
        x = self.final_norm(x)
        prior_positions, token_positions, post_positions = self.positions_for_seq(seq_len)
        return (
            mx.take(x, token_positions, axis=1),
            mx.take(x, prior_positions, axis=1),
            mx.take(x, post_positions, axis=1),
        )

    def token_ce(self, token_h: mx.array, target_ids: mx.array) -> mx.array:
        x = token_h.reshape(-1, token_h.shape[-1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

    def state_latents(self, prior_h: mx.array, post_h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        prior_z = base.rms_norm(self.state_proj(prior_h).astype(base.COMPUTE_DTYPE))
        post_z = base.rms_norm(self.state_proj(post_h).astype(base.COMPUTE_DTYPE))
        pred_z = self.state_pred_out(nn.silu(self.state_pred_fc(prior_z))).astype(base.COMPUTE_DTYPE)
        pred_z = base.rms_norm(pred_z)
        return prior_z, post_z, pred_z

    def losses(self, input_ids: mx.array, target_ids: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        token_h, prior_h, post_h = self.hidden_views(input_ids)
        ce_loss = self.token_ce(token_h, target_ids)
        _, post_z, pred_z = self.state_latents(prior_h, post_h)
        pred_loss = mx.mean(mx.square(pred_z.astype(mx.float32) - post_z.astype(mx.float32)))
        transposed_post_z = mx.transpose(post_z, (1, 0, 2))
        sigreg_loss = self.sigreg(transposed_post_z)
        total = ce_loss + self.state_pred_weight * pred_loss + self.sigreg_weight * sigreg_loss
        return total, ce_loss, pred_loss, sigreg_loss


class SplitOptimizers:
    def __init__(self, model: StateSpaceGPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.trainable_parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys: list[str] = []
        self.turbo_matrix_keys: list[str] = []
        self.scalar_keys: list[str] = []
        for key, param in params.items():
            if key == self.embed_key:
                continue
            is_control = param.ndim < 2 or any(pattern in key for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
            if param.ndim == 2 and not is_control:
                if base.infer_turbo_mode(key) != "none":
                    self.turbo_matrix_keys.append(key)
                else:
                    self.matrix_keys.append(key)
            else:
                self.scalar_keys.append(key)
        self.muon = (
            base.Muon(
                self.matrix_keys,
                params,
                args,
                momentum_target=args.muon_momentum,
                momentum_warmup_start=args.muon_momentum_warmup_start,
            )
            if self.matrix_keys
            else None
        )
        self.muon_turbo = (
            base.Muon(
                self.turbo_matrix_keys,
                params,
                args,
                momentum_target=args.turbo_qat_muon_momentum if args.turbo_qat else args.muon_momentum,
                momentum_warmup_start=args.turbo_qat_muon_momentum_warmup_start if args.turbo_qat else args.muon_momentum_warmup_start,
            )
            if self.turbo_matrix_keys
            else None
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

    def step(self, model: StateSpaceGPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.trainable_parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        if self.muon is not None:
            updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        if self.muon_turbo is not None:
            updated.update(self.muon_turbo.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


def exportable_flat_state(model: StateSpaceGPT) -> dict[str, mx.array]:
    return {
        k: v
        for k, v in tree_flatten(model.state)
        if not k.startswith("sigreg.") and not any(pattern in k for pattern in base.SERIALIZATION_SKIP_NAME_PATTERNS)
    }


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: base.TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict, tuple[mx.array, mx.array] | None]:
    chunk_sizes = base.token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    last_batch: tuple[mx.array, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        last_batch = (x, y)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = base.accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items())), last_batch


def main() -> None:
    args = Hyperparameters()
    if not args.tie_embeddings:
        raise NotImplementedError("train_state_jepa_mlx.py only supports tied embeddings")
    if args.train_seq_len % args.chunk_size != 0:
        raise ValueError(f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by CHUNK_SIZE={args.chunk_size}")

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

    mx.random.seed(args.seed)
    train_loader = base.TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    model = StateSpaceGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_layer_templates=args.num_layer_templates,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
        state_latent_dim=args.state_latent_dim,
        state_pred_hidden=args.state_pred_hidden,
        state_pred_weight=args.state_pred_weight,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_seed=args.sigreg_seed,
    )
    model.set_turbo_qat(False, 0.0)
    opt = SplitOptimizers(model, args)
    quant_eval_model: StateSpaceGPT | None = None
    compiled_quant_ce_loss = None
    compiled = args.compile_model and not args.turbo_qat
    if compiled:
        compiled_ce_loss = mx.compile(lambda x, y: model.losses(x, y)[1], inputs=model.state, outputs=model.state)
        compiled_loss_components = mx.compile(lambda x, y: model.losses(x, y), inputs=model.state, outputs=model.state)
        compiled_loss_and_grad = mx.compile(
            nn.value_and_grad(model, lambda x, y: model.losses(x, y)[0]),
            inputs=model.state,
            outputs=model.state,
        )
    else:
        compiled_ce_loss = lambda x, y: model.losses(x, y)[1]
        compiled_loss_components = lambda x, y: model.losses(x, y)
        compiled_loss_and_grad = nn.value_and_grad(model, lambda x, y: model.losses(x, y)[0])

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
        f"state_jepa:chunk_size:{args.chunk_size} augmented_seq_len:{model.augmented_seq_len(args.train_seq_len)} "
        f"state_latent_dim:{args.state_latent_dim} state_pred_hidden:{args.state_pred_hidden}"
    )
    log(
        f"loss_weights:ce=1.0 state_pred:{args.state_pred_weight} sigreg:{args.sigreg_weight} "
        f"sigreg_projs:{args.sigreg_num_proj} sigreg_seed:{args.sigreg_seed}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} val_seqs:{(val_tokens.size - 1) // args.train_seq_len} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(
        f"compute_dtype:{base.COMPUTE_DTYPE} compile:{compiled} quant_format:{args.quant_format} "
        f"turbo_qat:{args.turbo_qat} turbo_block:{args.turbo_block_size} "
        f"turbo_bits_mse:{args.turbo_mse_bits} turbo_bits_prod:{args.turbo_prod_bits}"
    )
    if args.quant_eval_every > 0:
        log(
            f"quant_eval:enabled every:{args.quant_eval_every} seqs:{(quant_eval_tokens.size - 1) // args.train_seq_len}"
        )
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} prior_token:{model.prior_token.dtype} "
        f"state_proj:{model.state_proj.weight.dtype} skip_weights:{model.skip_weights.dtype}"
    )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, _ = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            mx.eval(warmup_loss, grads)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_ce_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()
        train_loader = base.TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0

    def turbo_qat_scale(step_idx: int) -> float:
        if not args.turbo_qat or args.iterations <= 0:
            return 0.0
        frac = step_idx / max(args.iterations, 1)
        if frac < args.turbo_qat_start_frac:
            return 0.0
        if args.turbo_qat_ramp_frac <= 0:
            return 1.0
        return min((frac - args.turbo_qat_start_frac) / args.turbo_qat_ramp_frac, 1.0)

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_quant_eval = args.quant_eval_every > 0 and (last_step or step % args.quant_eval_every == 0)
        if should_validate or should_quant_eval:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if should_validate:
                val_loss, val_bpb = base.eval_val(
                    args,
                    model,
                    (lambda x, y, operator_codes=None: compiled_ce_loss(x, y)),
                    None,
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
            if should_quant_eval:
                if quant_eval_model is None:
                    quant_eval_model = StateSpaceGPT(
                        vocab_size=args.vocab_size,
                        num_layers=args.num_layers,
                        num_layer_templates=args.num_layer_templates,
                        dim=args.model_dim,
                        num_heads=args.num_heads,
                        num_kv_heads=args.num_kv_heads,
                        mlp_mult=args.mlp_mult,
                        logit_chunk_tokens=args.logit_chunk_tokens,
                        logit_softcap=args.logit_softcap,
                        rope_base=args.rope_base,
                        tied_embed_init_std=args.tied_embed_init_std,
                        qk_gain_init=args.qk_gain_init,
                        chunk_size=args.chunk_size,
                        state_latent_dim=args.state_latent_dim,
                        state_pred_hidden=args.state_pred_hidden,
                        state_pred_weight=args.state_pred_weight,
                        sigreg_weight=args.sigreg_weight,
                        sigreg_knots=args.sigreg_knots,
                        sigreg_num_proj=args.sigreg_num_proj,
                        sigreg_seed=args.sigreg_seed,
                    )
                    compiled_quant_ce_loss = mx.compile(
                        lambda x, y: quant_eval_model.losses(x, y)[1],
                        inputs=quant_eval_model.state,
                        outputs=quant_eval_model.state,
                    )
                q_t0 = time.perf_counter()
                raw_q_val_loss, raw_q_val_bpb = base.eval_val(
                    args,
                    model,
                    (lambda x, y, operator_codes=None: compiled_ce_loss(x, y)),
                    None,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                model.clear_turbo_cache()
                flat_state = exportable_flat_state(model)
                quant_obj, quant_stats, quant_raw, quant_blob = base.serialize_quantized_state_dict(flat_state)
                quant_eval_model.clear_turbo_cache()
                quant_eval_model.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
                q_val_loss, q_val_bpb = base.eval_val(
                    args,
                    quant_eval_model,
                    (lambda x, y, operator_codes=None: compiled_quant_ce_loss(x, y)),
                    None,
                    quant_eval_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                log(
                    f"step:{step}/{args.iterations} quant_diag_seqs:{(quant_eval_tokens.size - 1) // args.train_seq_len} "
                    f"raw_val_loss:{raw_q_val_loss:.4f} raw_val_bpb:{raw_q_val_bpb:.4f} "
                    f"quant_val_loss:{q_val_loss:.4f} quant_val_bpb:{q_val_bpb:.4f} "
                    f"quant_gap_bpb:{q_val_bpb - raw_q_val_bpb:+.4f} int8_zlib_bytes:{len(quant_blob)} "
                    f"payload:{quant_stats['int8_payload_bytes']} raw_pickle:{len(quant_raw)} "
                    f"eval_time:{1000.0 * (time.perf_counter() - q_t0):.0f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        qat_scale = turbo_qat_scale(step)
        qat_lambda = args.turbo_qat_lambda * qat_scale
        qat_active = args.turbo_qat and qat_scale > 0.0
        model.set_turbo_qat(qat_active, qat_scale)
        step_loss_and_grad = (
            nn.value_and_grad(model, lambda x, y: model.losses(x, y)[0] + qat_lambda * model.turbo_regularizer())
            if qat_active
            else compiled_loss_and_grad
        )
        step_t0 = time.perf_counter()
        train_loss, grads, last_batch = loss_and_grad_chunked(args, train_loader, step_loss_and_grad)
        grads = base.clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            extra = ""
            if last_batch is not None and (step <= 10 or step % max(args.train_log_every, 50) == 0):
                metrics = compiled_loss_components(*last_batch)
                mx.eval(*metrics)
                _, ce_metric, pred_metric, sig_metric = metrics
                extra = (
                    f" ce:{float(ce_metric.item()):.4f}"
                    f" state_pred:{float(pred_metric.item()):.4f}"
                    f" sigreg:{float(sig_metric.item()):.4f}"
                )
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"turbo_qat_scale:{qat_scale:.3f} turbo_qat_lambda:{qat_lambda:.6f}{extra}"
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

    quant_roundtrip = StateSpaceGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_layer_templates=args.num_layer_templates,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size,
        state_latent_dim=args.state_latent_dim,
        state_pred_hidden=args.state_pred_hidden,
        state_pred_weight=args.state_pred_weight,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_seed=args.sigreg_seed,
    )
    quant_roundtrip.update(tree_unflatten(list(base.dequantize_state_dict(quant_obj).items())))
    compiled_roundtrip_ce = mx.compile(
        lambda x, y: quant_roundtrip.losses(x, y)[1],
        inputs=quant_roundtrip.state,
        outputs=quant_roundtrip.state,
    )
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        quant_roundtrip,
        (lambda x, y, operator_codes=None: compiled_roundtrip_ce(x, y)),
        None,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    log(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
