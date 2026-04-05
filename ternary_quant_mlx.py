from __future__ import annotations

import math

import numpy as np

import mlx.core as mx

import turbo_quant_mlx as tq

TERNARY_SCHEME_VERSION = 1
TERNARY_KIND = "gaussian_3level_v1"

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def ternary_gaussian_levels() -> tuple[float, float]:
    centroid = math.sqrt(2.0 / math.pi)
    for _ in range(32):
        threshold = 0.5 * centroid
        tail_prob = 0.5 * math.erfc(threshold / math.sqrt(2.0))
        pdf = math.exp(-0.5 * threshold * threshold) / math.sqrt(2.0 * math.pi)
        centroid = pdf / max(tail_prob, 1.0e-12)
    threshold = 0.5 * centroid
    return threshold, centroid


TERNARY_THRESHOLD_STD, TERNARY_CENTROID_STD = ternary_gaussian_levels()


def _ternary_levels(block_size: int) -> tuple[np.float32, np.float32]:
    scale = np.float32(1.0 / math.sqrt(block_size))
    return (
        np.float32(TERNARY_THRESHOLD_STD) * scale,
        np.float32(TERNARY_CENTROID_STD) * scale,
    )


def ternary_quantize_dequantize_array(
    arr: mx.array,
    *,
    block_size: int,
    rotate: bool,
    rot_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    f32 = np.asarray(arr.astype(mx.float32), dtype=np.float32)
    if f32.ndim != 2 or block_size <= 0 or (block_size & (block_size - 1)):
        raise ValueError(
            f"TernaryQuant requires 2D tensors and power-of-two block size, got shape={tuple(f32.shape)} block_size={block_size}"
        )
    rows, row_dim = f32.shape
    pad = (-row_dim) % block_size
    flat = np.pad(f32, ((0, 0), (0, pad))) if pad else f32
    blocks = flat.reshape(-1, block_size)
    norms = np.linalg.norm(blocks, axis=-1).astype(np.float32, copy=False)
    unit = np.divide(
        blocks,
        np.maximum(norms[:, None], 1.0e-8),
        out=np.zeros_like(blocks),
        where=norms[:, None] > 0,
    )
    if rotate:
        rotated = np.asarray(
            tq.rotate_blocks_mx(mx.array(unit), block_size, rot_seed).astype(mx.float32),
            dtype=np.float32,
        )
    else:
        rotated = unit
    threshold, centroid = _ternary_levels(block_size)
    states = np.where(rotated > threshold, 2, np.where(rotated < -threshold, 0, 1)).astype(
        np.int64,
        copy=False,
    )
    values = np.asarray(
        np.take(np.array([-centroid, 0.0, centroid, 0.0], dtype=np.float32), states, axis=0),
        dtype=np.float32,
    )
    if rotate:
        deq_unit = np.asarray(
            tq.inverse_rotate_blocks_mx(mx.array(values), block_size, rot_seed).astype(mx.float32),
            dtype=np.float32,
        )
    else:
        deq_unit = values
    deq = np.ascontiguousarray((deq_unit * norms[:, None]).reshape(rows, row_dim + pad)[:, :row_dim])
    meta: dict[str, object] = {
        "scheme": "ternary_block_v1",
        "scheme_version": TERNARY_SCHEME_VERSION,
        "ternary_kind": TERNARY_KIND,
        "shape": tuple(f32.shape),
        "row_dim": row_dim,
        "pad": pad,
        "dtype": str(arr.dtype).split(".")[-1],
        "block_size": int(block_size),
        "rotate": bool(rotate),
        "rot_seed": int(rot_seed),
        "rotation_kind": tq.TURBO_ROTATION_KIND if rotate else "identity",
        "count": int(states.size),
        "norms": np.ascontiguousarray(norms.astype(np.float16, copy=False)),
        "state_packed": np.ascontiguousarray(tq.pack_bits(states, 2)),
    }
    return deq.astype(f32.dtype, copy=False), meta


def ternary_payload_breakdown(meta: dict[str, object]) -> dict[str, int]:
    return {
        "payload_bytes": int(np.asarray(meta["norms"]).nbytes + np.asarray(meta["state_packed"]).nbytes),
        "norm_bytes": int(np.asarray(meta["norms"]).nbytes),
        "state_bytes": int(np.asarray(meta["state_packed"]).nbytes),
    }


def dequantize_ternary_tensor(meta: dict[str, object]) -> mx.array:
    block_size = int(meta["block_size"])
    count = int(meta["count"])
    rotate = bool(meta.get("rotate", False))
    rot_seed = int(meta.get("rot_seed", tq.TURBO_ROT_SEED))
    row_dim = int(meta["row_dim"])
    pad = int(meta["pad"])
    rows, _ = map(int, meta["shape"])
    _, centroid = _ternary_levels(block_size)
    states = tq.unpack_bits(np.asarray(meta["state_packed"]), count, 2).reshape(-1, block_size)
    table = np.array([-centroid, 0.0, centroid, 0.0], dtype=np.float32)
    values = np.take(table, states, axis=0)
    if rotate:
        deq = np.asarray(
            tq.inverse_rotate_blocks_mx(mx.array(values), block_size, rot_seed).astype(mx.float32),
            dtype=np.float32,
        )
    else:
        deq = values.astype(np.float32, copy=False)
    norms = np.asarray(meta["norms"], dtype=np.float16).astype(np.float32, copy=False).reshape(-1, 1)
    out = np.ascontiguousarray((deq * norms).reshape(rows, row_dim + pad)[:, :row_dim])
    return mx.array(out, dtype=MX_DTYPE_FROM_NAME[str(meta["dtype"])])
