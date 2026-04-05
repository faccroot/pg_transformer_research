from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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

_HADAMARD_CACHE: dict[tuple[int, int, str], tuple[Tensor, Tensor, Tensor]] = {}
_GAUSSIAN_CACHE: dict[tuple[int, int, str], Tensor] = {}
_CODEBOOK_CACHE: dict[tuple[int, int], Tensor] = {}


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


def fwht(x: Tensor) -> Tensor:
    y = x.reshape(-1, x.shape[-1])
    h = 1
    while h < y.shape[-1]:
        y = y.view(y.shape[0], -1, 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(y.shape[0], -1)
        h *= 2
    return y.reshape_as(x) / math.sqrt(x.shape[-1])


def hadamard_params(block_size: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    key = (block_size, seed, str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + block_size * 104729)
    perm = torch.randperm(block_size, generator=gen)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(block_size, dtype=torch.int64)
    signs = torch.where(torch.randint(0, 2, (block_size,), generator=gen, dtype=torch.int64) > 0, 1.0, -1.0)
    out = (
        perm.to(device=device),
        inv_perm.to(device=device),
        signs.to(device=device, dtype=torch.float32),
    )
    _HADAMARD_CACHE[key] = out
    return out


def rotate_blocks(x: Tensor, block_size: int, seed: int) -> Tensor:
    perm, _, signs = hadamard_params(block_size, seed, x.device)
    return fwht(x.index_select(-1, perm) * signs)


def inverse_rotate_blocks(x: Tensor, block_size: int, seed: int) -> Tensor:
    _, inv_perm, signs = hadamard_params(block_size, seed, x.device)
    return (fwht(x) * signs).index_select(-1, inv_perm)


def gaussian_matrix(block_size: int, seed: int, device: torch.device) -> Tensor:
    key = (block_size, seed, str(device))
    cached = _GAUSSIAN_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + block_size * 130363)
    mat = torch.randn((block_size, block_size), generator=gen, dtype=torch.float32).to(device=device)
    _GAUSSIAN_CACHE[key] = mat
    return mat


def turbo_codebook(bits: int, block_size: int, device: torch.device) -> Tensor:
    key = (bits, block_size)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is None:
        levels = 1 << bits
        probs = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
        centroids = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)
        for _ in range(24):
            mids = (centroids[:-1] + centroids[1:]) * 0.5
            bounds = [-float("inf"), *mids.tolist(), float("inf")]
            updated = []
            for i in range(levels):
                lo, hi = bounds[i], bounds[i + 1]
                cdf_lo = 0.0 if math.isinf(lo) and lo < 0 else 0.5 * (1.0 + math.erf(lo / math.sqrt(2.0)))
                cdf_hi = 1.0 if math.isinf(hi) and hi > 0 else 0.5 * (1.0 + math.erf(hi / math.sqrt(2.0)))
                pdf_lo = 0.0 if math.isinf(lo) else math.exp(-0.5 * lo * lo) / math.sqrt(2.0 * math.pi)
                pdf_hi = 0.0 if math.isinf(hi) else math.exp(-0.5 * hi * hi) / math.sqrt(2.0 * math.pi)
                updated.append((pdf_lo - pdf_hi) / max(cdf_hi - cdf_lo, 1e-12))
            centroids = torch.tensor(updated, dtype=torch.float64)
        cached = (centroids / math.sqrt(block_size)).to(torch.float32)
        _CODEBOOK_CACHE[key] = cached
    return cached.to(device=device)


def pack_bits(values: Tensor, bits: int) -> Tensor:
    flat = values.reshape(-1).to(dtype=torch.int64, device="cpu").numpy().astype(np.uint64, copy=False)
    planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8, copy=False)
    return torch.from_numpy(np.packbits(planes.reshape(-1), bitorder="little"))


def unpack_bits(blob: Tensor, count: int, bits: int) -> Tensor:
    bitstream = np.unpackbits(blob.to(device="cpu").numpy(), bitorder="little")[: count * bits]
    planes = bitstream.reshape(count, bits).astype(np.uint64, copy=False)
    weights = (1 << np.arange(bits, dtype=np.uint64))[None, :]
    return torch.from_numpy((planes * weights).sum(axis=1).astype(np.int64, copy=False))


def quantize_unit_blocks(blocks: Tensor, mode: str, total_bits: int, block_size: int) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    mse_bits = total_bits if mode == "mse" else total_bits - 1
    if mse_bits <= 0:
        raise ValueError(f"Turbo {mode} quantization requires positive MSE bits, got total_bits={total_bits}")
    codebook = turbo_codebook(mse_bits, block_size, blocks.device)
    bounds = (codebook[:-1] + codebook[1:]) * 0.5
    rotated = rotate_blocks(blocks, block_size, TURBO_ROT_SEED)
    idx = torch.bucketize(rotated, bounds)
    mse = inverse_rotate_blocks(codebook[idx], block_size, TURBO_ROT_SEED)
    if mode != "prod":
        return mse, idx, None, None
    residual = blocks - mse
    gamma = residual.norm(dim=-1, keepdim=True)
    sketch_mat = gaussian_matrix(block_size, TURBO_QJL_SEED, blocks.device)
    qjl = torch.where(residual @ sketch_mat.T >= 0, 1.0, -1.0)
    correction = math.sqrt(math.pi / 2.0) / block_size * gamma * (qjl @ sketch_mat)
    return mse + correction, idx, gamma.squeeze(-1), qjl


def turbo_quantize_dequantize_matrix(t: Tensor, mode: str, total_bits: int, block_size: int) -> tuple[Tensor, dict[str, object]]:
    if t.ndim != 2 or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"TurboQuant requires 2D tensors and power-of-two block size, got shape={tuple(t.shape)} block_size={block_size}")
    t32 = t.float()
    rows, row_dim = t32.shape
    pad = (-row_dim) % block_size
    flat = F.pad(t32, (0, pad)) if pad else t32
    blocks = flat.view(-1, block_size)
    norms = blocks.norm(dim=-1)
    unit = blocks / norms.clamp_min(1e-8)[:, None]
    unit = torch.where(norms[:, None] > 0, unit, torch.zeros_like(unit))
    deq_unit, idx, residual_norms, qjl = quantize_unit_blocks(unit, mode, total_bits, block_size)
    deq = (deq_unit * norms[:, None]).view(rows, row_dim + pad)[:, :row_dim].to(dtype=t.dtype).contiguous()
    meta: dict[str, object] = {
        "scheme": "turbo_block",
        "scheme_version": TURBO_SCHEME_VERSION,
        "mode": mode,
        "bits": total_bits,
        "mse_bits": total_bits if mode == "mse" else total_bits - 1,
        "block_size": block_size,
        "shape": tuple(t.shape),
        "row_dim": row_dim,
        "pad": pad,
        "dtype": str(t.dtype).removeprefix("torch."),
        "rot_seed": TURBO_ROT_SEED,
        "qjl_seed": TURBO_QJL_SEED,
        "rotation_kind": TURBO_ROTATION_KIND,
        "codebook_kind": TURBO_CODEBOOK_KIND,
        "qjl_kind": TURBO_QJL_KIND,
        "norms": norms.to(device="cpu", dtype=torch.float16).contiguous(),
        "idx_packed": pack_bits(idx.to(device="cpu"), total_bits if mode == "mse" else total_bits - 1),
        "count": int(idx.numel()),
    }
    if residual_norms is not None and qjl is not None:
        meta["residual_norms"] = residual_norms.to(device="cpu", dtype=torch.float16).contiguous()
        meta["qjl_packed"] = pack_bits((qjl > 0).to(device="cpu", dtype=torch.int64), 1)
    return deq, meta


def dequantize_turbo_tensor(meta: dict[str, object]) -> Tensor:
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
    idx = unpack_bits(meta["idx_packed"], count, mse_bits).to(dtype=torch.long)
    rotated = turbo_codebook(mse_bits, block_size, torch.device("cpu"))[idx].view(-1, block_size)
    deq = inverse_rotate_blocks(rotated, block_size, rot_seed)
    if mode == "prod":
        qjl_bits = unpack_bits(meta["qjl_packed"], count, 1).view(-1, block_size)
        qjl = torch.where(qjl_bits > 0, 1.0, -1.0)
        gamma = meta["residual_norms"].float().view(-1, 1)
        sketch_mat = gaussian_matrix(block_size, qjl_seed, torch.device("cpu"))
        deq = deq + math.sqrt(math.pi / 2.0) / block_size * gamma * (qjl @ sketch_mat)
    norms = meta["norms"].float().view(-1, 1)
    rows, row_dim = map(int, meta["shape"])
    pad = int(meta["pad"])
    out = (deq * norms).view(rows, row_dim + pad)[:, :row_dim]
    return out.to(dtype=getattr(torch, str(meta["dtype"]))).contiguous()


class TurboLinear(nn.Linear):
    def __init__(self, *args, turbo_mode: str = "none", turbo_bits: int = 0, turbo_block_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.turbo_mode = turbo_mode
        self.turbo_bits = turbo_bits
        self.turbo_block_size = turbo_block_size
        self.turbo_qat_enabled = False
        self.turbo_qat_alpha = 0.0
        self._turbo_cache_version = -1
        self._turbo_cache_weight: Tensor | None = None

    def set_turbo_qat(self, enabled: bool, alpha: float) -> None:
        self.turbo_qat_enabled = enabled
        self.turbo_qat_alpha = alpha

    def turbo_dequantized_weight(self) -> Tensor | None:
        if self.turbo_mode == "none" or self.turbo_bits <= 0 or self.turbo_block_size <= 0:
            return None
        version = int(self.weight._version)
        if self._turbo_cache_weight is None or self._turbo_cache_version != version:
            deq, _ = turbo_quantize_dequantize_matrix(
                self.weight.detach(),
                mode=self.turbo_mode,
                total_bits=self.turbo_bits,
                block_size=self.turbo_block_size,
            )
            self._turbo_cache_weight = deq.to(device=self.weight.device, dtype=self.weight.dtype)
            self._turbo_cache_version = version
        return self._turbo_cache_weight

    def turbo_regularizer(self) -> Tensor:
        deq = self.turbo_dequantized_weight()
        if deq is None:
            return self.weight.new_zeros(())
        return (self.weight.float() - deq.float()).square().mean()

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self.training and self.turbo_qat_enabled and self.turbo_qat_alpha > 0.0:
            deq = self.turbo_dequantized_weight()
            if deq is not None:
                weight = weight + self.turbo_qat_alpha * (deq.to(dtype=weight.dtype) - weight).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(x.dtype), bias)
