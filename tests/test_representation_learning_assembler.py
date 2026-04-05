from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.representation_learning.assemble_reasoning_core import ReasoningCoreAssembler
from tools.representation_learning.schemas import LayerGeometry, PlatonicGeometry


class RepresentationLearningAssemblerTests(unittest.TestCase):
    def test_assembler_is_deterministic_for_same_seed(self) -> None:
        geometry = PlatonicGeometry(
            canonical_dim=4,
            layer_geometries={
                1: LayerGeometry(
                    relative_depth=0.5,
                    directions=np.array(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    scales=np.array([2.0, 1.0], dtype=np.float32),
                    coactivation=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32),
                )
            },
        )
        assembler = ReasoningCoreAssembler(geometry)
        model_a = assembler.assemble(
            target_hidden_dim=8,
            target_num_layers=2,
            target_vocab_size=32,
            num_heads=2,
            max_seq_len=16,
            seed=17,
        )
        model_b = assembler.assemble(
            target_hidden_dim=8,
            target_num_layers=2,
            target_vocab_size=32,
            num_heads=2,
            max_seq_len=16,
            seed=17,
        )
        self.assertTrue(torch.allclose(model_a.blocks[0].attn.q_proj.weight, model_b.blocks[0].attn.q_proj.weight))
        input_ids = torch.randint(0, 32, (2, 16))
        logits = model_a(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 16, 32))
        loss = assembler.evaluate_zero_shot(model_a, input_ids)
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
