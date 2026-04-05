import unittest

import numpy as np

from text_prosody_features import build_token_prosody_luts


class ProsodyAuxTests(unittest.TestCase):
    class _FakeSentencePiece:
        def __init__(self, pieces):
            self._pieces = list(pieces)

        def get_piece_size(self) -> int:
            return len(self._pieces)

        def id_to_piece(self, idx: int) -> str:
            return self._pieces[idx]

    def test_base_model_supports_prosody_embeddings_and_aux_export_strip(self) -> None:
        try:
            import mlx.core as mx
            import train_gpt_mlx as base
        except ModuleNotFoundError as exc:
            self.skipTest(f"trainer import unavailable in unit environment: {exc}")
        sp = self._FakeSentencePiece(["▁word", ",", "\n\n", '"', "https://x.y", "🙂", "<div>", "▁"])
        luts = build_token_prosody_luts(sp, extended_binary_features=True)
        model = base.GPT(
            vocab_size=8,
            num_layers=4,
            num_layer_templates=4,
            dim=16,
            num_heads=4,
            num_kv_heads=4,
            mlp_mult=2,
            mlp_leaky_slope=0.5,
            tie_embeddings=False,
            logit_chunk_tokens=0,
            logit_softcap=30.0,
            rope_base=10000.0,
            tied_embed_init_std=0.02,
            qk_gain_init=1.0,
            prosody_type_embeddings_enabled=True,
            prosody_type_embedding_init_std=0.002,
            prosody_extended_feature_set_enabled=True,
            prosody_feature_embeddings_enabled=True,
            prosody_feature_embedding_init_std=0.002,
            prosody_state_adapter_enabled=True,
            prosody_state_dim=8,
            prosody_state_init_std=0.005,
            prosody_state_scale=0.5,
            prosody_state_reset_prior_weight=1.0,
            prosody_state_hierarchical_enabled=True,
            prosody_state_slow_reset_scale=0.35,
            prosody_aux_layer_index=1,
            prosody_aux_weight=0.05,
            prosody_aux_head_init_std=0.005,
            prosody_aux_punctuation_weight=0.5,
            token_prosody_luts=luts,
        )
        self.assertTrue(model.has_prosody_type_embeddings())
        self.assertTrue(model.has_prosody_feature_embeddings())
        self.assertTrue(model.has_prosody_state_adapter())
        self.assertTrue(model.has_prosody_aux())
        embedded = model.embed_inputs(mx.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=mx.int32))
        self.assertEqual(tuple(embedded.shape), (1, 4, 16))
        flat = base.exportable_flat_state(model)
        self.assertTrue(any("prosody_token_class_emb" in key for key in flat))
        self.assertTrue(any("prosody_punctuation_emb" in key for key in flat))
        self.assertTrue(any("prosody_feature_emb" in key for key in flat))
        self.assertTrue(any("prosody_state_adapter" in key for key in flat))
        self.assertFalse(any("prosody_token_class_head" in key for key in flat))
        self.assertFalse(any("prosody_boundary_head" in key for key in flat))
        self.assertFalse(any("prosody_punctuation_head" in key for key in flat))
        self.assertFalse(any("prosody_quote_head" in key for key in flat))


if __name__ == "__main__":
    unittest.main()
