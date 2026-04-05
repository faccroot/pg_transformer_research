import unittest

from tools.audit_tokenizer_fragmentation import (
    construct_kind_for_whitespace,
    scan_constructs,
    summarize_construct_records,
)


class TokenizerFragmentationAuditTests(unittest.TestCase):
    def test_construct_kind_for_whitespace(self) -> None:
        self.assertEqual(construct_kind_for_whitespace("\n"), "single_newline")
        self.assertEqual(construct_kind_for_whitespace("\n  "), "newline_plus_indent")
        self.assertEqual(construct_kind_for_whitespace("\n\n"), "paragraph_break")
        self.assertEqual(construct_kind_for_whitespace("   "), "space_or_tab_run")

    def test_scan_constructs_finds_mixed_structural_spans(self) -> None:
        pieces = ["▁hello", "\n\n", "https://x", ".y", "🙂", '"', "**", "▁world"]
        piece_texts = [" hello", "\n\n", "https://x", ".y", "🙂", '"', "**", " world"]
        got = scan_constructs(piece_texts, pieces)
        self.assertEqual(len(got["paragraph_break"]), 1)
        self.assertEqual(int(got["paragraph_break"][0]["token_count"]), 1)
        self.assertEqual(len(got["url_span"]), 1)
        self.assertEqual(int(got["url_span"][0]["token_count"]), 2)
        self.assertEqual(len(got["emoji_span"]), 1)
        self.assertEqual(len(got["quote_span"]), 1)
        self.assertEqual(len(got["markup_delim_span"]), 1)

    def test_summarize_construct_records_reports_tax(self) -> None:
        records = [
            {"token_count": 1, "char_count": 2, "text": "\\n\\n"},
            {"token_count": 3, "char_count": 18, "text": "https://example"},
        ]
        summary = summarize_construct_records(records, sample_tokens=16, top_k=4)
        self.assertEqual(int(summary["count"]), 2)
        self.assertEqual(int(summary["extra_token_tax"]), 2)
        self.assertGreater(float(summary["extra_token_tax_share"]), 0.0)
        self.assertEqual(len(summary["examples"]), 2)


if __name__ == "__main__":
    unittest.main()
