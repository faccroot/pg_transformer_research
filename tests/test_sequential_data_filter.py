from __future__ import annotations

import unittest

import numpy as np

from sequential_data_filter import (
    SequentialCompressibilityFilterConfig,
    keep_chunk,
    token_bytes_compressibility,
)


class SequentialDataFilterTests(unittest.TestCase):
    def test_repetitive_tokens_are_more_compressible_than_random_tokens(self) -> None:
        repetitive = np.full((1024,), 7, dtype=np.int32)
        randomish = np.arange(1024, dtype=np.int32) % 1024
        comp_repetitive = token_bytes_compressibility(repetitive)
        comp_randomish = token_bytes_compressibility(randomish)
        self.assertGreater(comp_repetitive, comp_randomish)

    def test_keep_chunk_respects_threshold(self) -> None:
        repetitive = np.full((1024,), 7, dtype=np.int32)
        randomish = np.arange(1024, dtype=np.int32) % 1024
        cfg = SequentialCompressibilityFilterConfig(enabled=True, min_compressibility=0.3)
        self.assertTrue(keep_chunk(repetitive, cfg))
        self.assertFalse(keep_chunk(randomish, cfg))


if __name__ == "__main__":
    unittest.main()

