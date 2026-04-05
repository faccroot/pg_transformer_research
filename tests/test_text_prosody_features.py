import unittest

import numpy as np

from text_prosody_features import (
    BOUNDARY_STRENGTH_TO_ID,
    PUNCTUATION_ROLE_TO_ID,
    PROSODY_BINARY_FEATURE_NAMES,
    TOKEN_CLASS_TO_ID,
    build_token_prosody_luts,
    build_binary_feature_stack,
    build_reset_prior_values,
    bucketize_distances,
    classify_piece,
    distance_to_next,
    extract_text_prosody_features_from_pieces,
    is_code_like_piece,
    is_emoji_like_piece,
    is_heading_like_piece,
    is_interruption_like_piece,
    is_list_like_piece,
    is_markup_delimiter_piece,
    is_url_like_piece,
    punctuation_role_for_piece,
)


class TextProsodyFeatureTests(unittest.TestCase):
    class _FakeSentencePiece:
        def __init__(self, pieces):
            self._pieces = list(pieces)

        def get_piece_size(self) -> int:
            return len(self._pieces)

        def id_to_piece(self, idx: int) -> str:
            return self._pieces[idx]

    def test_classify_piece_separates_basic_types(self) -> None:
        self.assertEqual(classify_piece("▁hello"), "content")
        self.assertEqual(classify_piece(","), "punctuation")
        self.assertEqual(classify_piece("▁"), "whitespace")
        self.assertEqual(classify_piece("<div>"), "markup")
        self.assertEqual(classify_piece('"'), "quote")

    def test_url_emoji_and_markup_helpers(self) -> None:
        self.assertTrue(is_url_like_piece("https://example.com"))
        self.assertTrue(is_emoji_like_piece("🙂"))
        self.assertTrue(is_markup_delimiter_piece("**"))
        self.assertFalse(is_markup_delimiter_piece("word"))
        self.assertTrue(is_heading_like_piece("##"))
        self.assertTrue(is_list_like_piece("- "))
        self.assertTrue(is_code_like_piece("```python"))
        self.assertTrue(is_interruption_like_piece("—"))

    def test_punctuation_roles_capture_mixed_structure(self) -> None:
        self.assertEqual(punctuation_role_for_piece(","), PUNCTUATION_ROLE_TO_ID["comma_clause"])
        self.assertEqual(punctuation_role_for_piece("—"), PUNCTUATION_ROLE_TO_ID["dash_interrupt"])
        self.assertEqual(punctuation_role_for_piece("..."), PUNCTUATION_ROLE_TO_ID["ellipsis"])
        self.assertEqual(punctuation_role_for_piece('"'), PUNCTUATION_ROLE_TO_ID["quote_mark"])
        self.assertEqual(punctuation_role_for_piece("**"), PUNCTUATION_ROLE_TO_ID["markup_delim"])

    def test_boundary_strength_detects_sentence_and_paragraph(self) -> None:
        feats = extract_text_prosody_features_from_pieces(["hello", ".", "\n\n", "world"])
        self.assertEqual(int(feats.boundary_strength_ids[1]), BOUNDARY_STRENGTH_TO_ID["sentence"])
        self.assertEqual(int(feats.boundary_strength_ids[2]), BOUNDARY_STRENGTH_TO_ID["paragraph"])

    def test_quote_state_toggles_on_double_quotes(self) -> None:
        feats = extract_text_prosody_features_from_pieces(['"', "hello", '"', "world"])
        np.testing.assert_array_equal(feats.quote_state, np.array([0, 1, 1, 0], dtype=np.int32))

    def test_distance_to_next_marks_future_boundary(self) -> None:
        mask = np.array([False, False, True, False, False], dtype=np.bool_)
        got = distance_to_next(mask)
        np.testing.assert_array_equal(got, np.array([2, 1, 0, 5, 5], dtype=np.int32))

    def test_bucketize_distances(self) -> None:
        got = bucketize_distances(np.array([0, 1, 5, 20], dtype=np.int32), (0, 2, 8))
        np.testing.assert_array_equal(got, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_recent_density_tracks_noncontent(self) -> None:
        feats = extract_text_prosody_features_from_pieces(["word", ",", "word", "\n\n"], density_window=2)
        self.assertEqual(int(feats.token_class_ids[0]), TOKEN_CLASS_TO_ID["content"])
        self.assertGreaterEqual(float(feats.recent_noncontent_density[-1]), 0.5)

    def test_build_token_prosody_luts(self) -> None:
        sp = self._FakeSentencePiece(["▁word", "\n\n", '"', "https://x.y", "🙂", "—", "##", "```"])
        luts = build_token_prosody_luts(sp, extended_binary_features=True)
        self.assertEqual(luts.token_class_ids.shape[0], 8)
        self.assertEqual(luts.binary_feature_ids.shape, (8, len(PROSODY_BINARY_FEATURE_NAMES)))
        self.assertEqual(luts.reset_prior_values.shape[0], 8)
        self.assertEqual(int(luts.token_class_ids[0]), TOKEN_CLASS_TO_ID["content"])
        self.assertEqual(int(luts.boundary_strength_ids[1]), BOUNDARY_STRENGTH_TO_ID["paragraph"])
        self.assertEqual(int(luts.quote_like_ids[2]), 1)
        self.assertEqual(int(luts.url_like_ids[3]), 1)
        self.assertEqual(int(luts.emoji_like_ids[4]), 1)
        self.assertEqual(int(luts.punctuation_role_ids[5]), PUNCTUATION_ROLE_TO_ID["dash_interrupt"])
        self.assertEqual(int(luts.binary_feature_ids[6, 9]), 1)
        self.assertEqual(int(luts.binary_feature_ids[7, 10]), 1)

    def test_binary_feature_stack_and_reset_priors(self) -> None:
        token_class_ids = np.array(
            [
                TOKEN_CLASS_TO_ID["content"],
                TOKEN_CLASS_TO_ID["punctuation"],
                TOKEN_CLASS_TO_ID["whitespace"],
            ],
            dtype=np.int32,
        )
        boundary_strength_ids = np.array(
            [
                BOUNDARY_STRENGTH_TO_ID["none"],
                BOUNDARY_STRENGTH_TO_ID["sentence"],
                BOUNDARY_STRENGTH_TO_ID["paragraph"],
            ],
            dtype=np.int32,
        )
        features = build_binary_feature_stack(
            token_class_ids,
            boundary_strength_ids,
            np.array([0, 0, 1], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
        )
        self.assertEqual(features.shape, (3, len(PROSODY_BINARY_FEATURE_NAMES)))
        priors = build_reset_prior_values(features)
        self.assertGreater(float(priors[2]), float(priors[1]))
        self.assertGreater(float(priors[1]), float(priors[0]))


if __name__ == "__main__":
    unittest.main()
