import unittest


class ResidualFeedbackTests(unittest.TestCase):
    def test_argmax_residual_novelty_weights_downweight_repeated_errors(self) -> None:
        try:
            import mlx.core as mx
            from residual_feedback import argmax_residual_novelty_weights_from_ids
        except ModuleNotFoundError as exc:
            self.skipTest(f"mlx residual feedback unavailable in unit environment: {exc}")
        embedding = mx.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=mx.float32,
        )
        predicted = mx.array([[0, 0, 1]], dtype=mx.int32)
        target = mx.array([[1, 1, 0]], dtype=mx.int32)
        weights, mean_similarity, mean_novelty, mean_weight, valid_fraction = argmax_residual_novelty_weights_from_ids(
            predicted,
            target,
            embedding,
            min_scale=0.5,
            max_scale=1.5,
        )
        weights_np = weights.astype(mx.float32).tolist()[0]
        self.assertAlmostEqual(float(weights_np[0]), 1.0, places=6)
        self.assertLess(float(weights_np[1]), 0.6)
        self.assertGreater(float(weights_np[2]), 1.4)
        self.assertAlmostEqual(float(mean_similarity.item()), 0.0, places=5)
        self.assertAlmostEqual(float(mean_novelty.item()), 0.5, places=5)
        self.assertAlmostEqual(float(valid_fraction.item()), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(mean_weight.item()), sum(weights_np) / len(weights_np), places=5)

    def test_residual_prediction_alignment_loss_prefers_aligned_direction(self) -> None:
        try:
            import mlx.core as mx
            from residual_feedback import residual_prediction_alignment_loss
        except ModuleNotFoundError as exc:
            self.skipTest(f"mlx residual feedback unavailable in unit environment: {exc}")
        target = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
        aligned = mx.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=mx.float32)
        misaligned = mx.array([[[-0.9, -0.1], [-0.1, -0.9]]], dtype=mx.float32)
        aligned_total, aligned_mse, aligned_cos = residual_prediction_alignment_loss(aligned, target, cosine_weight=0.5)
        misaligned_total, misaligned_mse, misaligned_cos = residual_prediction_alignment_loss(
            misaligned,
            target,
            cosine_weight=0.5,
        )
        self.assertLess(float(aligned_total.item()), float(misaligned_total.item()))
        self.assertLess(float(aligned_mse.item()), float(misaligned_mse.item()))
        self.assertGreater(float(aligned_cos.item()), float(misaligned_cos.item()))

    def test_argmax_residual_novelty_weights_support_ema_history(self) -> None:
        try:
            import mlx.core as mx
            from residual_feedback import argmax_residual_novelty_weights_from_ids
        except ModuleNotFoundError as exc:
            self.skipTest(f"mlx residual feedback unavailable in unit environment: {exc}")
        embedding = mx.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=mx.float32,
        )
        predicted = mx.array([[0, 0, 0, 1]], dtype=mx.int32)
        target = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
        weights, mean_similarity, mean_novelty, _mean_weight, valid_fraction = argmax_residual_novelty_weights_from_ids(
            predicted,
            target,
            embedding,
            min_scale=0.5,
            max_scale=1.5,
            ema_decay=0.5,
        )
        weights_np = weights.astype(mx.float32).tolist()[0]
        self.assertAlmostEqual(float(weights_np[0]), 1.0, places=6)
        self.assertLess(float(weights_np[1]), 0.6)
        self.assertLess(float(weights_np[2]), 0.6)
        self.assertGreater(float(weights_np[3]), 1.4)
        self.assertAlmostEqual(float(valid_fraction.item()), 3.0 / 4.0, places=5)
        self.assertLess(float(mean_similarity.item()), 0.5)
        self.assertGreater(float(mean_novelty.item()), 0.25)

    def test_base_trainer_loss_terms_accept_residual_novelty_config(self) -> None:
        try:
            import mlx.core as mx
            import train_gpt_mlx as base
            from residual_feedback import ResidualNoveltyWeightingConfig
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
        )
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
        y = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        terms = model.loss_terms(
            x,
            y,
            residual_novelty_config=ResidualNoveltyWeightingConfig(
                enabled=True,
                min_scale=0.75,
                max_scale=1.25,
                ema_decay=0.5,
            ),
        )
        self.assertEqual(len(terms), 38)
        self.assertTrue(float(terms[-2].item()) > 0.0)

    def test_base_trainer_loss_terms_accept_residual_error_prior(self) -> None:
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
            residual_error_prior_enabled=True,
            residual_error_prior_weight=0.1,
            residual_error_prior_bottleneck_dim=8,
            residual_error_prior_init_std=0.02,
            residual_error_prior_cosine_weight=0.5,
            residual_error_prior_norm_epsilon=1.0e-6,
            residual_error_prior_target_mode="expected",
        )
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
        y = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        terms = model.loss_terms(x, y)
        self.assertEqual(len(terms), 38)
        self.assertTrue(float(terms[11].item()) >= 0.0)
        self.assertTrue(float(terms[12].item()) >= 0.0)


if __name__ == "__main__":
    unittest.main()
