"""
Minimal Gram Newton-Schulz fallback derived from Dao-AILab's MIT-licensed
`gram-newton-schulz` reference implementation.

This path uses only PyTorch matmuls and is intended as a local drop-in when
the external package and symmetric Hopper kernels are not installed.
"""

from __future__ import annotations

import torch
from torch import Tensor


_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (a / _SAFETY_FACTOR, b / (_SAFETY_FACTOR ** 3), c / (_SAFETY_FACTOR ** 5))
    for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS
)


def gram_newton_schulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    restart_iterations: tuple[int, ...] = (2,),
) -> Tensor:
    if G.ndim != 2:
        raise ValueError(f"Gram Newton-Schulz expects a 2D tensor, got shape {tuple(G.shape)}")
    if steps != len(POLAR_EXPRESS_COEFFICIENTS):
        raise ValueError(
            f"Local Gram Newton-Schulz fallback only supports {len(POLAR_EXPRESS_COEFFICIENTS)} steps, got {steps}"
        )

    dtype = G.dtype
    X = G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT

    X /= X.norm() + eps
    X = X.to(torch.float16)

    if X.size(0) == X.size(1):
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return (X.mT if transposed else X).to(dtype)

    R = X @ X.mT
    I = torch.eye(R.size(0), device=X.device, dtype=X.dtype)
    Q = None
    reset_points = set(restart_iterations)

    for i, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
        if i in reset_points and i != 0:
            X = Q @ X
            R = X @ X.mT
            Q = None

        Z = b * R + c * (R @ R)
        if i != 0 and i not in reset_points:
            Q = Q @ Z + a * Q
        else:
            Q = Z + a * I

        if i < steps - 1 and i + 1 not in reset_points:
            RZ = R @ Z + a * R
            R = Z @ RZ + a * RZ

    X = Q @ X
    if transposed:
        X = X.mT
    return X.to(dtype)
