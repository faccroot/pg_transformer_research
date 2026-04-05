#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time

import mlx.core as mx
import numpy as np


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def build_inputs() -> dict[str, mx.array]:
    batch = getenv_int("BATCH", 16)
    seq = getenv_int("SEQ", 512)
    dim = getenv_int("DIM", 512)
    patch = getenv_int("PATCH", 8)
    chord_dim = getenv_int("CHORD_DIM", 64)
    bank = getenv_int("BANK", 32)
    seed = getenv_int("SEED", 17)
    mx.random.seed(seed)
    hidden = mx.random.normal((batch, seq, dim), dtype=mx.float32)
    write_w = mx.random.normal((dim, chord_dim), dtype=mx.float32) * (dim ** -0.5)
    query_w = mx.random.normal((dim, chord_dim), dtype=mx.float32) * (dim ** -0.5)
    read_w = mx.random.normal((chord_dim, dim), dtype=mx.float32) * 1e-3
    return {
        "hidden": hidden,
        "write_w": write_w,
        "query_w": query_w,
        "read_w": read_w,
        "patch": mx.array(patch, dtype=mx.int32),
        "bank": mx.array(bank, dtype=mx.int32),
    }


def rms_norm(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def patch_pool(hidden: mx.array, patch_len: int) -> mx.array:
    batch, seq, dim = hidden.shape
    patch_count = seq // patch_len
    trimmed = hidden[:, : patch_count * patch_len, :]
    return mx.mean(trimmed.reshape(batch, patch_count, patch_len, dim), axis=2)


def l2_normalize(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.sum(mx.square(x), axis=-1, keepdims=True) + 1e-6)


def current_harmonic_numpy(hidden: mx.array, write_w: mx.array, query_w: mx.array, read_w: mx.array, *, patch_len: int, bank_size: int, boundary_threshold: float, max_patches_per_chord: int) -> mx.array:
    write_patches = rms_norm(patch_pool(hidden, patch_len)) @ write_w
    query_patches = rms_norm(patch_pool(hidden, patch_len)) @ query_w
    batch = int(write_patches.shape[0])
    patch_count = int(write_patches.shape[1])
    dim = int(write_patches.shape[2])
    zero = mx.zeros((dim,), dtype=mx.float32)
    patch_reads_batches: list[mx.array] = []
    for batch_idx in range(batch):
        write_b = write_patches[batch_idx]
        query_b = query_patches[batch_idx]
        write_np = np.asarray(write_b, dtype=np.float32)
        flux = np.zeros((patch_count,), dtype=np.float32)
        if patch_count > 1:
            cur = write_np[1:]
            prev = write_np[:-1]
            denom = np.linalg.norm(cur, axis=-1) * np.linalg.norm(prev, axis=-1)
            denom = np.maximum(denom, 1e-6)
            flux[1:] = 1.0 - np.clip(np.sum(cur * prev, axis=-1) / denom, -1.0, 1.0)
        bank: list[mx.array] = []
        patch_reads: list[mx.array] = []
        chord_start = 0
        for patch_idx in range(patch_count):
            if bank:
                bank_tensor = mx.stack(bank, axis=0)
                bank_keys = l2_normalize(bank_tensor)
                query = l2_normalize(query_b[patch_idx : patch_idx + 1])
                scores = (query @ bank_keys.T) * 8.0
                weights = mx.softmax(scores, axis=-1)
                read = mx.squeeze(weights @ bank_tensor, axis=0)
            else:
                read = zero
            patch_reads.append(read)
            if patch_idx <= 0:
                continue
            current_len = patch_idx - chord_start
            should_split = current_len >= max_patches_per_chord or (
                current_len >= 1 and float(flux[patch_idx]) >= boundary_threshold
            )
            if should_split:
                chord = mx.mean(write_b[chord_start:patch_idx, :], axis=0)
                bank.append(chord)
                if len(bank) > bank_size:
                    bank = bank[-bank_size:]
                chord_start = patch_idx
        patch_reads_batches.append(mx.stack(patch_reads, axis=0))
    patch_reads = mx.stack(patch_reads_batches, axis=0)
    return patch_reads @ read_w


def vectorized_flux_segments(hidden: mx.array, write_w: mx.array, query_w: mx.array, read_w: mx.array, *, patch_len: int, bank_size: int, boundary_threshold: float, max_patches_per_chord: int) -> mx.array:
    write_patches = rms_norm(patch_pool(hidden, patch_len)) @ write_w
    query_patches = rms_norm(patch_pool(hidden, patch_len)) @ query_w
    batch, patch_count, chord_dim = write_patches.shape
    write_unit = l2_normalize(write_patches)
    flux = mx.zeros((batch, patch_count), dtype=mx.float32)
    if patch_count > 1:
        flux_delta = 1.0 - mx.sum(write_unit[:, 1:, :].astype(mx.float32) * write_unit[:, :-1, :].astype(mx.float32), axis=-1)
        flux = mx.concatenate([mx.zeros((batch, 1), dtype=mx.float32), flux_delta.astype(mx.float32)], axis=1)
    patch_positions = mx.arange(patch_count, dtype=mx.int32)
    periodic_boundary = ((patch_positions % max_patches_per_chord) == 0)[None, :]
    threshold_boundary = flux >= mx.array(boundary_threshold, dtype=mx.float32)
    if patch_count > 2:
        prev_flux = mx.concatenate([mx.full((batch, 1), -1e9, dtype=mx.float32), flux[:, :-1]], axis=1)
        next_flux = mx.concatenate([flux[:, 1:], mx.full((batch, 1), -1e9, dtype=mx.float32)], axis=1)
        local_peak = mx.logical_and(flux >= prev_flux, flux > next_flux)
        threshold_boundary = mx.logical_and(threshold_boundary, local_peak)
    first_patch = mx.concatenate(
        [
            mx.ones((batch, 1), dtype=mx.bool_),
            mx.zeros((batch, max(patch_count - 1, 0)), dtype=mx.bool_),
        ],
        axis=1,
    )
    boundary_flags = mx.logical_or(mx.logical_or(periodic_boundary, threshold_boundary), first_patch)
    segment_ids = mx.cumsum(boundary_flags.astype(mx.int32), axis=1) - 1
    chord_slots = mx.arange(patch_count, dtype=mx.int32)
    segment_onehot = (segment_ids[:, :, None] == chord_slots[None, None, :]).astype(mx.float32)
    chord_counts = mx.sum(segment_onehot, axis=1)
    chord_mask = chord_counts > 0.0
    chord_sums = mx.matmul(mx.swapaxes(segment_onehot, 1, 2), write_patches)
    chords = chord_sums / mx.maximum(chord_counts[:, :, None], 1.0)
    if bank_size > 0 and patch_count > bank_size:
        chord_count_per_batch = mx.sum(chord_mask.astype(mx.int32), axis=1)
        keep_start = mx.maximum(chord_count_per_batch - bank_size, 0)
        recent_keep = chord_slots[None, :] >= keep_start[:, None]
        chord_mask = mx.logical_and(chord_mask, recent_keep)
    q = l2_normalize(query_patches)
    k = l2_normalize(chords)
    scores = mx.matmul(q, mx.swapaxes(k, 1, 2)) * 8.0
    prev_mask = mx.logical_and(chord_slots[None, None, :] < segment_ids[:, :, None], chord_mask[:, None, :])
    prev_mask_f = prev_mask.astype(mx.float32)
    masked = scores + (prev_mask_f - 1.0) * 1e9
    weights = mx.softmax(masked, axis=-1) * prev_mask_f
    weights_sum = mx.sum(weights, axis=-1, keepdims=True)
    weights = mx.where(weights_sum > 0.0, weights / mx.maximum(weights_sum, 1e-6), 0.0)
    patch_reads = mx.matmul(weights, chords)
    return patch_reads @ read_w


def vectorized_dense_fixed(hidden: mx.array, write_w: mx.array, query_w: mx.array, read_w: mx.array, *, patch_len: int, bank_size: int, chord_span: int) -> mx.array:
    write_patches = rms_norm(patch_pool(hidden, patch_len)) @ write_w
    query_patches = rms_norm(patch_pool(hidden, patch_len)) @ query_w
    batch, patch_count, chord_dim = write_patches.shape
    chord_count = max(1, math.ceil(patch_count / chord_span))
    pad = chord_count * chord_span - patch_count
    if pad > 0:
        write_patches = mx.concatenate([write_patches, mx.zeros((batch, pad, chord_dim), dtype=mx.float32)], axis=1)
    chords = mx.mean(write_patches.reshape(batch, chord_count, chord_span, chord_dim), axis=2)
    if chord_count > bank_size:
        chords = chords[:, -bank_size:, :]
        chord_count = bank_size
    patch_chord_idx_np = np.minimum(np.arange(patch_count) // chord_span, chord_count - 1).astype(np.int32)
    prev_mask_np = (np.arange(chord_count)[None, :] < patch_chord_idx_np[:, None]).astype(np.float32)
    prev_mask = mx.array(prev_mask_np[None, :, :], dtype=mx.float32)
    q = l2_normalize(query_patches)
    k = l2_normalize(chords)
    scores = mx.matmul(q, mx.swapaxes(k, 1, 2)) * 8.0
    masked = scores + (prev_mask - 1.0) * 1e9
    weights = mx.softmax(masked, axis=-1) * prev_mask
    weights_sum = mx.sum(weights, axis=-1, keepdims=True)
    weights = mx.where(weights_sum > 0.0, weights / mx.maximum(weights_sum, 1e-6), 0.0)
    patch_reads = mx.matmul(weights, chords)
    return patch_reads @ read_w


def benchmark(name: str, fn, *, warmup: int, iters: int) -> None:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
        mx.eval(last)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    per_iter_ms = 1000.0 * elapsed / max(iters, 1)
    print(f"{name}: total_ms={elapsed * 1000.0:.2f} per_iter_ms={per_iter_ms:.3f}")
    if last is not None:
        print(f"{name}: out_shape={tuple(last.shape)} out_mean={float(mx.mean(last).item()):.6f}")


def maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    return mx.compile(fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile harmonic-path kernels on MLX.")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--chord-dim", type=int, default=None)
    parser.add_argument("--bank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=None)
    parser.add_argument("--chord-span", type=int, default=None)
    parser.add_argument("--max-patches-per-chord", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warmup = args.warmup if args.warmup is not None else getenv_int("WARMUP", 5)
    iters = args.iters if args.iters is not None else getenv_int("ITERS", 20)
    patch_len = args.patch if args.patch is not None else getenv_int("PATCH", 8)
    bank_size = args.bank if args.bank is not None else getenv_int("BANK", 32)
    boundary_threshold = (
        args.boundary_threshold if args.boundary_threshold is not None else getenv_float("BOUNDARY_THRESHOLD", 0.12)
    )
    chord_span = args.chord_span if args.chord_span is not None else getenv_int("CHORD_SPAN", 4)
    max_patches_per_chord = (
        args.max_patches_per_chord
        if args.max_patches_per_chord is not None
        else getenv_int("MAX_PATCHES_PER_CHORD", 8)
    )
    enable_compile = args.compile or (getenv_int("COMPILE", 0) != 0)
    if args.batch is not None:
        os.environ["BATCH"] = str(args.batch)
    if args.seq is not None:
        os.environ["SEQ"] = str(args.seq)
    if args.dim is not None:
        os.environ["DIM"] = str(args.dim)
    if args.patch is not None:
        os.environ["PATCH"] = str(args.patch)
    if args.chord_dim is not None:
        os.environ["CHORD_DIM"] = str(args.chord_dim)
    if args.bank is not None:
        os.environ["BANK"] = str(args.bank)
    if args.seed is not None:
        os.environ["SEED"] = str(args.seed)
    tensors = build_inputs()
    hidden = tensors["hidden"]
    write_w = tensors["write_w"]
    query_w = tensors["query_w"]
    read_w = tensors["read_w"]
    batch, seq, dim = hidden.shape
    print(
        "config:",
        {
            "batch": int(batch),
            "seq": int(seq),
            "dim": int(dim),
            "patch_len": patch_len,
            "patches": int(seq) // patch_len,
            "chord_dim": int(write_w.shape[1]),
            "bank_size": bank_size,
            "compile": enable_compile,
            "warmup": warmup,
            "iters": iters,
        },
    )
    patch_pool_proj_only = lambda: rms_norm(patch_pool(hidden, patch_len)) @ write_w
    harmonic_current_numpy_fn = lambda: current_harmonic_numpy(
        hidden,
        write_w,
        query_w,
        read_w,
        patch_len=patch_len,
        bank_size=bank_size,
        boundary_threshold=boundary_threshold,
        max_patches_per_chord=max_patches_per_chord,
    )
    harmonic_vectorized_flux_fn = lambda: vectorized_flux_segments(
        hidden,
        write_w,
        query_w,
        read_w,
        patch_len=patch_len,
        bank_size=bank_size,
        boundary_threshold=boundary_threshold,
        max_patches_per_chord=max_patches_per_chord,
    )
    harmonic_vectorized_fixed_fn = lambda: vectorized_dense_fixed(
        hidden,
        write_w,
        query_w,
        read_w,
        patch_len=patch_len,
        bank_size=bank_size,
        chord_span=chord_span,
    )

    benchmark(
        "patch_pool_proj_only",
        patch_pool_proj_only,
        warmup=warmup,
        iters=iters,
    )
    benchmark(
        "harmonic_current_numpy",
        harmonic_current_numpy_fn,
        warmup=max(2, warmup // 2),
        iters=max(5, iters // 5),
    )
    benchmark(
        "harmonic_vectorized_flux",
        harmonic_vectorized_flux_fn,
        warmup=warmup,
        iters=iters,
    )
    benchmark(
        "harmonic_vectorized_fixed",
        harmonic_vectorized_fixed_fn,
        warmup=warmup,
        iters=iters,
    )
    if enable_compile:
        benchmark(
            "patch_pool_proj_only_compiled",
            maybe_compile(patch_pool_proj_only, True),
            warmup=warmup,
            iters=iters,
        )
        benchmark(
            "harmonic_vectorized_flux_compiled",
            maybe_compile(harmonic_vectorized_flux_fn, True),
            warmup=warmup,
            iters=iters,
        )
        benchmark(
            "harmonic_vectorized_fixed_compiled",
            maybe_compile(harmonic_vectorized_fixed_fn, True),
            warmup=warmup,
            iters=iters,
        )


if __name__ == "__main__":
    main()
