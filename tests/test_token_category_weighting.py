from __future__ import annotations

import unittest

import numpy as np

from token_category_weighting import (
    TokenCategoryWeightingConfig,
    build_token_category_luts,
    classify_piece,
    compute_token_category_weights,
)


class FakeSentencePiece:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def vocab_size(self) -> int:
        return len(self._pieces)

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]


class TokenCategoryWeightingTests(unittest.TestCase):
    def test_classify_piece_flags_urls_identifiers_and_repeat_candidates(self) -> None:
        self.assertEqual(classify_piece("▁http"), (True, False, False))
        self.assertEqual(classify_piece("abc123"), (False, True, True))
        self.assertEqual(classify_piece("▁bank"), (False, False, True))
        self.assertEqual(classify_piece("▁the"), (False, False, False))

    def test_build_luts_from_sentencepiece(self) -> None:
        sp = FakeSentencePiece(["▁the", "▁bank", "abc123", "▁http", "▁www.example.com"])
        luts = build_token_category_luts(sp)
        np.testing.assert_array_equal(luts.url_like.astype(np.int32), np.array([0, 0, 0, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(luts.identifier_like.astype(np.int32), np.array([0, 0, 1, 0, 0], dtype=np.int32))
        np.testing.assert_array_equal(luts.repeat_candidate.astype(np.int32), np.array([0, 1, 1, 0, 0], dtype=np.int32))

    def test_compute_weights_downweights_urls_and_identifiers_and_upweights_rerefs(self) -> None:
        sp = FakeSentencePiece(["▁the", "▁bank", "abc123", "▁http", "▁corp"])
        luts = build_token_category_luts(sp)
        cfg = TokenCategoryWeightingConfig(
            enabled=True,
            url_like_weight=0.2,
            identifier_like_weight=0.4,
            repeat_content_weight=2.0,
        )
        x = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
        y = np.array([[1, 2, 1, 3, 1]], dtype=np.int32)
        weights = compute_token_category_weights(x, y, luts, cfg)
        expected = np.array([[1.0, 0.4, 2.0, 0.2, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(weights, expected)

    def test_disabled_config_returns_unit_weights(self) -> None:
        sp = FakeSentencePiece(["▁bank", "abc123"])
        luts = build_token_category_luts(sp)
        cfg = TokenCategoryWeightingConfig(enabled=False)
        x = np.array([[0, 1]], dtype=np.int32)
        y = np.array([[0, 1]], dtype=np.int32)
        weights = compute_token_category_weights(x, y, luts, cfg)
        np.testing.assert_allclose(weights, np.ones_like(y, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
