from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

import mlx.core as mx
import mlx.nn as nn

TURBO_BLOCK_SIZE = 64
TURBO_MSE_BITS = 5
TURBO_PROD_BITS = 6
TURBO_ROT_SEED = 17
TURBO_QJL_SEED = 29
TURBO_SCHEME_VERSION = 2
TURBO_ROTATION_KIND = "signed_perm_fwht_v1"
TURBO_CODEBOOK_KIND = "gaussian_lloyd_max_v1"
TURBO_QJL_KIND = "gaussian_dense_sign_v1"
TURBO_MSE_NAME_PATTERNS: tuple[str, ...] = ("attn.c_q.weight", "attn.c_v.weight", "attn.proj.weight", "mlp.fc.weight", "mlp.proj.weight")
TURBO_PROD_NAME_PATTERNS: tuple[str, ...] = ("attn.c_k.weight",)

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

_NORMAL = NormalDist()
_HADAMARD_NP_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_HADAMARD_MX_CACHE: dict[tuple[int, int], tuple[mx.array, mx.array, mx.array]] = {}
_GAUSSIAN_NP_CACHE: dict[tuple[int, int], np.ndarray] = {}
_GAUSSIAN_MX_CACHE: dict[tuple[int, int], mx.array] = {}
_CODEBOOK_NP_CACHE: dict[tuple[int, int], np.ndarray] = {}
_CODEBOOK_MX_CACHE: dict[tuple[int, int], mx.array] = {}


def configure(
    *,
    block_size: int,
    mse_bits: int,
    prod_bits: int,
    rot_seed: int,
    qjl_seed: int,
    mse_patterns: tuple[str, ...],
    prod_patterns: tuple[str, ...],
) -> None:
    global TURBO_BLOCK_SIZE, TURBO_MSE_BITS, TURBO_PROD_BITS, TURBO_ROT_SEED, TURBO_QJL_SEED
    global TURBO_MSE_NAME_PATTERNS, TURBO_PROD_NAME_PATTERNS
    TURBO_BLOCK_SIZE = block_size
    TURBO_MSE_BITS = mse_bits
    TURBO_PROD_BITS = prod_bits
    TURBO_ROT_SEED = rot_seed
    TURBO_QJL_SEED = qjl_seed
    TURBO_MSE_NAME_PATTERNS = mse_patterns
    TURBO_PROD_NAME_PATTERNS = prod_patterns


def infer_turbo_mode(name: str) -> str:
    if any(pattern in name for pattern in TURBO_PROD_NAME_PATTERNS):
        return "prod"
    if any(pattern in name for pattern in TURBO_MSE_NAME_PATTERNS):
        return "mse"
    return "none"


def _hadamard_np(block_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (block_size, seed)
    cached = _HADAMARD_NP_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed + block_size * 104729)
    perm = rng.permutation(block_size).astype(np.int64, copy=False)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(block_size, dtype=np.int64)
    signs = np.where(rng.integers(0, 2, size=(block_size,), dtype=np.int64) > 0, 1.0, -1.0).astype(np.float32, copy=False)
    out = (perm, inv_perm, signs)
    _HADAMARD_NP_CACHE[key] = out
    return out


def hadamard_mx(block_size: int, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    key = (block_size, seed)
    cached = _HADAMARD_MX_CACHE.get(key)
    if cached is not None:
        return cached
    perm, inv_perm, signs = _hadamard_np(block_size, seed)
    out = (mx.array(perm), mx.array(inv_perm), mx.array(signs))
    _HADAMARD_MX_CACHE[key] = out
    return out


def _gaussian_np(block_size: int, seed: int) -> np.ndarray:
    key = (block_size, seed)
    cached = _GAUSSIAN_NP_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed + block_size * 130363)
    mat = rng.standard_normal((block_size, block_size), dtype=np.float32)
    _GAUSSIAN_NP_CACHE[key] = mat
    return mat


def gaussian_mx(block_size: int, seed: int) -> mx.array:
    key = (block_size, seed)
    cached = _GAUSSIAN_MX_CACHE.get(key)
    if cached is not None:
        return cached
    out = mx.array(_gaussian_np(block_size, seed))
    _GAUSSIAN_MX_CACHE[key] = out
    return out


def _codebook_np(bits: int, block_size: int) -> np.ndarray:
    key = (bits, block_size)
    cached = _CODEBOOK_NP_CACHE.get(key)
    if cached is not None:
        return cached
    levels = 1 << bits
    probs = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = np.array([_NORMAL.inv_cdf(float(p)) for p in probs], dtype=np.float64)
    for _ in range(24):
        mids = 0.5 * (centroids[:-1] + centroids[1:])
        bounds = np.concatenate(([-np.inf], mids, [np.inf]))
        updated = np.empty_like(centroids)
        for i in range(levels):
            lo, hi = float(bounds[i]), float(bounds[i + 1])
            cdf_lo = 0.0 if math.isinf(lo) and lo < 0 else _NORMAL.cdf(lo)
            cdf_hi = 1.0 if math.isinf(hi) and hi > 0 else _NORMAL.cdf(hi)
            pdf_lo = 0.0 if math.isinf(lo) else math.exp(-0.5 * lo * lo) / math.sqrt(2.0 * math.pi)
            pdf_hi = 0.0 if math.isinf(hi) else math.exp(-0.5 * hi * hi) / math.sqrt(2.0 * math.pi)
            updated[i] = (pdf_lo - pdf_hi) / max(cdf_hi - cdf_lo, 1e-12)
        centroids = updated
    cached = np.asarray(centroids / math.sqrt(block_size), dtype=np.float32)
    _CODEBOOK_NP_CACHE[key] = cached
    return cached


def codebook_mx(bits: int, block_size: int) -> mx.array:
    key = (bits, block_size)
    cached = _CODEBOOK_MX_CACHE.get(key)
    if cached is not None:
        return cached
    out = mx.array(_codebook_np(bits, block_size))
    _CODEBOOK_MX_CACHE[key] = out
    return out


def fwht_mx(x: mx.array) -> mx.array:
    y = x.reshape((-1, x.shape[-1]))
    h = 1
    while h < y.shape[-1]:
        y = y.reshape((y.shape[0], -1, 2, h))
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = mx.concatenate(((a + b)[:, :, None, :], (a - b)[:, :, None, :]), axis=2).reshape((y.shape[0], -1))
        h *= 2
    return y.reshape(x.shape) / math.sqrt(x.shape[-1])


def rotate_blocks_mx(x: mx.array, block_size: int, seed: int) -> mx.array:
    perm, _, signs = hadamard_mx(block_size, seed)
    return fwht_mx(mx.take(x, perm, axis=-1) * signs)


def inverse_rotate_blocks_mx(x: mx.array, block_size: int, seed: int) -> mx.array:
    _, inv_perm, signs = hadamard_mx(block_size, seed)
    return mx.take(fwht_mx(x) * signs, inv_perm, axis=-1)


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    flat = values.reshape(-1).astype(np.uint64, copy=False)
    planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8, copy=False)
    return np.packbits(planes.reshape(-1), bitorder="little")


def unpack_bits(blob: np.ndarray, count: int, bits: int) -> np.ndarray:
    bitstream = np.unpackbits(blob.astype(np.uint8, copy=False), bitorder="little")[: count * bits]
    planes = bitstream.reshape(count, bits).astype(np.uint64, copy=False)
    weights = (1 << np.arange(bits, dtype=np.uint64))[None, :]
    return (planes * weights).sum(axis=1).astype(np.int64, copy=False)


def bucketize_mx(values: mx.array, bounds: mx.array) -> mx.array:
    # MLX 0.31.x on the Minis does not expose searchsorted, so we emulate the same
    # right-side insertion rule with a compare-and-count reduction.
    return mx.sum(values[..., None] >= bounds.reshape((1, 1, -1)), axis=-1).astype(mx.int32)


def turbo_quantize_dequantize_weight(weight: mx.array, mode: str, total_bits: int, block_size: int) -> mx.array:
    if weight.ndim != 2 or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"TurboQuant requires 2D tensors and power-of-two block size, got shape={tuple(weight.shape)} block_size={block_size}")
    mse_bits = total_bits if mode == "mse" else total_bits - 1
    if mse_bits <= 0:
        raise ValueError(f"Turbo {mode} quantization requires positive MSE bits, got total_bits={total_bits}")
    rows, row_dim = weight.shape
    pad = (-row_dim) % block_size
    t32 = weight.astype(mx.float32)
    flat = mx.pad(t32, [(0, 0), (0, pad)]) if pad else t32
    blocks = flat.reshape((-1, block_size))
    norms = mx.sqrt(mx.sum(blocks * blocks, axis=-1))
    unit = blocks / mx.maximum(norms[:, None], 1e-8)
    unit = mx.where(norms[:, None] > 0, unit, mx.zeros_like(unit))
    codebook = codebook_mx(mse_bits, block_size)
    bounds = 0.5 * (codebook[:-1] + codebook[1:])
    rotated = rotate_blocks_mx(unit, block_size, TURBO_ROT_SEED)
    idx = bucketize_mx(rotated, bounds)
    mse = inverse_rotate_blocks_mx(mx.take(codebook, idx, axis=0), block_size, TURBO_ROT_SEED)
    out = mse
    if mode == "prod":
        residual = unit - mse
        gamma = mx.sqrt(mx.sum(residual * residual, axis=-1, keepdims=True))
        sketch_mat = gaussian_mx(block_size, TURBO_QJL_SEED)
        qjl = mx.where(residual @ sketch_mat.T >= 0, 1.0, -1.0)
        out = out + math.sqrt(math.pi / 2.0) / block_size * gamma * (qjl @ sketch_mat)
    deq = (out * norms[:, None]).reshape((rows, row_dim + pad))
    return deq[:, :row_dim].astype(weight.dtype)


def turbo_quantize_dequantize_array(arr: mx.array, mode: str, total_bits: int, block_size: int) -> tuple[np.ndarray, dict[str, object]]:
    f32 = np.asarray(arr.astype(mx.float32), dtype=np.float32)
    if f32.ndim != 2 or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"TurboQuant requires 2D tensors and power-of-two block size, got shape={tuple(f32.shape)} block_size={block_size}")
    mse_bits = total_bits if mode == "mse" else total_bits - 1
    if mse_bits <= 0:
        raise ValueError(f"Turbo {mode} quantization requires positive MSE bits, got total_bits={total_bits}")
    rows, row_dim = f32.shape
    pad = (-row_dim) % block_size
    flat = np.pad(f32, ((0, 0), (0, pad))) if pad else f32
    blocks = flat.reshape(-1, block_size)
    norms = np.linalg.norm(blocks, axis=-1).astype(np.float32, copy=False)
    unit = np.divide(blocks, np.maximum(norms[:, None], 1e-8), out=np.zeros_like(blocks), where=norms[:, None] > 0)
    codebook = _codebook_np(mse_bits, block_size)
    bounds = 0.5 * (codebook[:-1] + codebook[1:])
    perm, inv_perm, signs = _hadamard_np(block_size, TURBO_ROT_SEED)
    rotated = np.asarray(fwht_mx(mx.array(unit[:, perm] * signs)).astype(mx.float32), dtype=np.float32)
    idx = np.searchsorted(bounds, rotated, side="right").astype(np.int64, copy=False)
    mse = np.asarray(
        inverse_rotate_blocks_mx(mx.array(codebook[idx]), block_size, TURBO_ROT_SEED).astype(mx.float32),
        dtype=np.float32,
    )
    out = mse
    meta: dict[str, object] = {
        "scheme": "turbo_block",
        "scheme_version": TURBO_SCHEME_VERSION,
        "mode": mode,
        "bits": total_bits,
        "mse_bits": mse_bits,
        "block_size": block_size,
        "shape": tuple(f32.shape),
        "row_dim": row_dim,
        "pad": pad,
        "dtype": str(arr.dtype).split(".")[-1],
        "rot_seed": TURBO_ROT_SEED,
        "qjl_seed": TURBO_QJL_SEED,
        "rotation_kind": TURBO_ROTATION_KIND,
        "codebook_kind": TURBO_CODEBOOK_KIND,
        "qjl_kind": TURBO_QJL_KIND,
        "norms": np.ascontiguousarray(norms.astype(np.float16, copy=False)),
        "idx_packed": np.ascontiguousarray(pack_bits(idx, mse_bits)),
        "count": int(idx.size),
    }
    if mode == "prod":
        residual = unit - mse
        gamma = np.linalg.norm(residual, axis=-1).astype(np.float32, copy=False)
        sketch = _gaussian_np(block_size, TURBO_QJL_SEED)
        qjl = np.where(residual @ sketch.T >= 0, 1.0, -1.0).astype(np.float32, copy=False)
        out = out + math.sqrt(math.pi / 2.0) / block_size * gamma[:, None] * (qjl @ sketch)
        meta["residual_norms"] = np.ascontiguousarray(gamma.astype(np.float16, copy=False))
        meta["qjl_packed"] = np.ascontiguousarray(pack_bits((qjl > 0).astype(np.int64, copy=False), 1))
    deq = np.ascontiguousarray((out * norms[:, None]).reshape(rows, row_dim + pad)[:, :row_dim].astype(f32.dtype, copy=False))
    return deq, meta


def dequantize_turbo_tensor(meta: dict[str, object], codebook_override: np.ndarray | None = None) -> mx.array:
    block_size = int(meta["block_size"])
    mode = str(meta["mode"])
    scheme_version = int(meta.get("scheme_version", 1))
    if scheme_version > TURBO_SCHEME_VERSION:
        raise ValueError(f"Unsupported Turbo scheme_version={scheme_version}, max supported={TURBO_SCHEME_VERSION}")
    total_bits = int(meta["bits"])
    count = int(meta["count"])
    rot_seed = int(meta.get("rot_seed", TURBO_ROT_SEED))
    qjl_seed = int(meta.get("qjl_seed", TURBO_QJL_SEED))
    mse_bits = int(meta.get("mse_bits", total_bits if mode == "mse" else total_bits - 1))
    idx = unpack_bits(np.asarray(meta["idx_packed"]), count, mse_bits)
    codebook = _codebook_np(mse_bits, block_size) if codebook_override is None else np.asarray(codebook_override, dtype=np.float32)
    rotated = codebook[idx].reshape(-1, block_size)
    deq = np.asarray(inverse_rotate_blocks_mx(mx.array(rotated), block_size, rot_seed).astype(mx.float32), dtype=np.float32)
    if mode == "prod":
        qjl_bits = unpack_bits(np.asarray(meta["qjl_packed"]), count, 1).reshape(-1, block_size)
        qjl = np.where(qjl_bits > 0, 1.0, -1.0).astype(np.float32, copy=False)
        gamma = np.asarray(meta["residual_norms"], dtype=np.float16).astype(np.float32, copy=False).reshape(-1, 1)
        deq = deq + math.sqrt(math.pi / 2.0) / block_size * gamma * (qjl @ _gaussian_np(block_size, qjl_seed))
    norms = np.asarray(meta["norms"], dtype=np.float16).astype(np.float32, copy=False).reshape(-1, 1)
    rows, row_dim = map(int, meta["shape"])
    pad = int(meta["pad"])
    out = np.ascontiguousarray((deq * norms).reshape(rows, row_dim + pad)[:, :row_dim])
    return mx.array(out, dtype=MX_DTYPE_FROM_NAME[str(meta["dtype"])])


class TurboLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        turbo_mode: str = "none",
        turbo_bits: int = 0,
        turbo_block_size: int = 0,
        qat_excluded: bool = False,
    ):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
        self.turbo_mode = turbo_mode
        self.turbo_bits = turbo_bits
        self.turbo_block_size = turbo_block_size
        self.turbo_qat_excluded = qat_excluded
        self.turbo_qat_enabled = False
        self.turbo_qat_alpha = 0.0
        self._turbo_cache_weight: mx.array | None = None

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        self.turbo_qat_enabled = enabled and not self.turbo_qat_excluded
        self.turbo_qat_alpha = alpha

    def clear_turbo_cache(self) -> None:
        self._turbo_cache_weight = None

    def turbo_dequantized_weight(self) -> mx.array | None:
        if self.turbo_mode == "none" or self.turbo_bits <= 0 or self.turbo_block_size <= 0:
            return None
        if self._turbo_cache_weight is None:
            self._turbo_cache_weight = turbo_quantize_dequantize_weight(
                self.weight,
                mode=self.turbo_mode,
                total_bits=self.turbo_bits,
                block_size=self.turbo_block_size,
            ).astype(self.weight.dtype)
        return self._turbo_cache_weight

    def turbo_regularizer(self) -> mx.array:
        if self.turbo_qat_excluded:
            return mx.array(0.0, dtype=mx.float32)
        deq = self.turbo_dequantized_weight()
        if deq is None:
            return mx.array(0.0, dtype=mx.float32)
        diff = self.weight.astype(mx.float32) - deq.astype(mx.float32)
        return mx.mean(diff * diff)

    @staticmethod
    def turbo_qat_blend(weight: mx.array, deq: mx.array, alpha: float) -> mx.array:
        alpha = float(max(0.0, min(alpha, 1.0)))
        if alpha <= 0.0:
            return weight
        # Use a numerically safer STE form than stop_gradient(deq - weight).
        # Forward: (1 - alpha) * weight + alpha * deq
        # Backward: identity wrt weight.
        qhat = mx.stop_gradient(deq) + weight - mx.stop_gradient(weight)
        if alpha >= 1.0:
            return qhat
        return (1.0 - alpha) * weight + alpha * qhat

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight
        if self.turbo_qat_enabled and self.turbo_qat_alpha > 0.0:
            deq = self.turbo_dequantized_weight()
            if deq is not None:
                weight = self.turbo_qat_blend(weight, deq, self.turbo_qat_alpha)
        return x @ weight.astype(x.dtype).T
