import unittest


class HardmaxStructuralAnticollapseTests(unittest.TestCase):
    def test_simvq_effective_state_book_uses_shared_projection(self) -> None:
        try:
            import mlx.core as mx
            from logic_register_mlx import HardmaxStructuralController
        except ModuleNotFoundError as exc:
            self.skipTest(f"hardmax controller unavailable in unit environment: {exc}")
        controller = HardmaxStructuralController(
            model_dim=8,
            state_dim=3,
            num_states=2,
            simvq_enabled=True,
        )
        controller.state_book = mx.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=mx.float32,
        )
        controller.state_book_proj.weight = 2.0 * mx.eye(3, dtype=mx.float32)
        effective = controller.effective_state_book().astype(mx.float32).tolist()
        self.assertEqual(effective[0], [2.0, 0.0, 0.0])
        self.assertEqual(effective[1], [0.0, 2.0, 0.0])

    def test_gpt_loss_terms_accept_hardmax_simvq_nextlat(self) -> None:
        try:
            import mlx.core as mx
            import train_gpt_mlx as base
        except ModuleNotFoundError as exc:
            self.skipTest(f"trainer import unavailable in unit environment: {exc}")
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
            hardmax_struct_num_states=4,
            hardmax_struct_dim=8,
            hardmax_struct_layer_index=1,
            hardmax_struct_router_start_layer=2,
            hardmax_struct_simvq_enabled=True,
            hardmax_struct_nextlat_weight=0.1,
        )
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
        y = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        terms = model.loss_terms(x, y)
        self.assertEqual(len(terms), 38)
        try:
            nextlat_loss = float(terms[32].item())
            nextlat_acc = float(terms[33].item())
        except RuntimeError as exc:
            if "Compile::eval_cpu" in str(exc):
                self.skipTest(f"mlx cpu eval unavailable on this host: {exc}")
            raise
        self.assertGreaterEqual(nextlat_loss, 0.0)
        self.assertGreaterEqual(nextlat_acc, 0.0)
        self.assertLessEqual(nextlat_acc, 1.0)


if __name__ == "__main__":
    unittest.main()
