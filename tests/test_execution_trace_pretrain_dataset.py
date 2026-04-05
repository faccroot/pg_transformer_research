from __future__ import annotations

import unittest

from execution_trace_dataset import ExampleBuilder, GenerationConfig, build_example
from execution_trace_pretrain_dataset import (
    NONE_TOKEN,
    build_trace_pretrain_vocab,
    encode_trace_example,
    encode_trace_examples,
    pad_encoded_batch,
)


class ExecutionTracePretrainDatasetTests(unittest.TestCase):
    def test_encode_example_has_shifted_targets(self) -> None:
        builder = ExampleBuilder(rng=__import__("random").Random(5), config=GenerationConfig(seed=5, task_family="loop"))
        example = build_example(builder)
        vocab = build_trace_pretrain_vocab([example])
        encoded = encode_trace_example(example, vocab)

        self.assertEqual(encoded.length, len(example["trace"]))
        self.assertEqual(float(encoded.valid_mask[-1]), 0.0)
        self.assertEqual(
            int(encoded.target_next_opcode_ids[0]),
            vocab.opcode_to_id[str(example["trace"][1]["opcode"])],
        )
        self.assertEqual(
            int(encoded.target_next_step_type_ids[0]),
            vocab.step_type_to_id[str(example["trace"][1]["step_type"])],
        )
        self.assertEqual(
            int(encoded.target_next_stack_depth_ids[0]),
            min(len(example["trace"][1].get("stack_after", [])), vocab.max_stack_bucket - 1),
        )
        self.assertEqual(
            int(encoded.target_next_env_size_ids[0]),
            min(len(example["trace"][1].get("env_after", {})), vocab.max_env_bucket - 1),
        )

    def test_encode_example_tracks_memory_vars(self) -> None:
        builder = ExampleBuilder(rng=__import__("random").Random(9), config=GenerationConfig(seed=9, task_family="loop"))
        example = build_example(builder)
        vocab = build_trace_pretrain_vocab([example])
        encoded = encode_trace_example(example, vocab)

        none_var_id = vocab.variable_to_id[NONE_TOKEN]
        self.assertTrue(any(int(value) != none_var_id for value in encoded.read_var_ids))
        self.assertTrue(any(int(value) != none_var_id for value in encoded.write_var_ids))
        self.assertTrue(any(int(value) > 0 for value in encoded.read_count_ids))
        self.assertTrue(any(int(value) > 0 for value in encoded.write_count_ids))

    def test_pad_batch_shapes(self) -> None:
        builder = ExampleBuilder(rng=__import__("random").Random(13), config=GenerationConfig(seed=13))
        examples = [build_example(builder) for _ in range(3)]
        vocab = build_trace_pretrain_vocab(examples)
        encoded = encode_trace_examples(examples, vocab)
        batch = pad_encoded_batch(encoded, vocab)

        self.assertEqual(batch["opcode_ids"].shape[0], 3)
        self.assertEqual(batch["valid_mask"].shape, batch["opcode_ids"].shape)
        self.assertEqual(batch["target_next_write_var_ids"].shape, batch["opcode_ids"].shape)
        self.assertEqual(batch["target_next_read_count_ids"].shape, batch["opcode_ids"].shape)
        self.assertEqual(batch["target_next_write_count_ids"].shape, batch["opcode_ids"].shape)
        self.assertEqual(batch["target_next_stack_depth_ids"].shape, batch["opcode_ids"].shape)
        self.assertEqual(batch["target_next_env_size_ids"].shape, batch["opcode_ids"].shape)


if __name__ == "__main__":
    unittest.main()
