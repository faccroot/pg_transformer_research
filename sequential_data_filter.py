from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np


@dataclass(frozen=True)
class SequentialCompressibilityFilterConfig:
    enabled: bool = False
    min_compressibility: float = -1.0
    zlib_level: int = 6


def token_bytes_compressibility(tokens: np.ndarray, *, level: int = 6) -> float:
    flat = np.asarray(tokens, dtype=np.int32).reshape(-1)
    raw = np.asarray(flat, dtype="<u2").tobytes()
    return 1.0 - (len(zlib.compress(raw, level=level)) / max(len(raw), 1))


def keep_chunk(tokens: np.ndarray, config: SequentialCompressibilityFilterConfig) -> bool:
    if not config.enabled or config.min_compressibility < 0.0:
        return True
    return token_bytes_compressibility(tokens, level=config.zlib_level) >= float(config.min_compressibility)

