import unittest

import numpy as np

from execution_trace_pretrain_dataset import TracePretrainVocab
from execution_trace_verifier import (
    apply_teacher_forced_input_ablation,
    build_mixed_rollout_input_batch,
    build_rollout_input_batch,
    build_rollout_next_input_row,
    predicted_fields_exact_for_step,
    summarize_rollout_failures,
)


class ExecutionTraceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = TracePretrainVocab(
            opcode_to_id={"<pad>": 0, "LOAD": 1, "STORE": 2},
            step_type_to_id={"<pad>": 0, "read": 1, "write": 2},
            variable_to_id={"<pad>": 0, "<none>": 1, "a": 2, "b": 3},
            branch_to_id={"none": 0, "false": 1, "true": 2},
            max_stack_bucket=8,
            max_env_bucket=8,
            max_delta_bucket=8,
            max_memop_bucket=8,
        )
        self.batch = {
            "opcode_ids": np.array([[1, 2]], dtype=np.int32),
            "step_type_ids": np.array([[1, 2]], dtype=np.int32),
            "read_var_ids": np.array([[2, 3]], dtype=np.int32),
            "write_var_ids": np.array([[3, 2]], dtype=np.int32),
            "read_count_ids": np.array([[1, 2]], dtype=np.int32),
            "write_count_ids": np.array([[0, 1]], dtype=np.int32),
            "branch_ids": np.array([[2, 1]], dtype=np.int32),
            "stack_depth_ids": np.array([[4, 5]], dtype=np.int32),
            "env_size_ids": np.array([[6, 7]], dtype=np.int32),
            "env_delta_size_ids": np.array([[2, 3]], dtype=np.int32),
            "output_flag_ids": np.array([[1, 0]], dtype=np.int32),
            "target_next_opcode_ids": np.array([[2, 0]], dtype=np.int32),
            "target_next_step_type_ids": np.array([[2, 0]], dtype=np.int32),
            "target_next_read_var_ids": np.array([[3, 0]], dtype=np.int32),
            "target_next_write_var_ids": np.array([[2, 0]], dtype=np.int32),
            "target_next_read_count_ids": np.array([[2, 0]], dtype=np.int32),
            "target_next_write_count_ids": np.array([[1, 0]], dtype=np.int32),
            "target_next_branch_ids": np.array([[1, 0]], dtype=np.int32),
            "target_next_stack_depth_ids": np.array([[5, 0]], dtype=np.int32),
            "target_next_env_size_ids": np.array([[7, 0]], dtype=np.int32),
            "target_next_env_delta_size_ids": np.array([[3, 0]], dtype=np.int32),
            "target_next_output_flag_ids": np.array([[0, 0]], dtype=np.int32),
            "valid_mask": np.array([[1.0, 0.0]], dtype=np.float32),
        }

    def test_opcode_step_plus_sizes_ablation_keeps_sizes_but_drops_trace_rich_inputs(self) -> None:
        out = apply_teacher_forced_input_ablation(self.batch, self.vocab, "opcode_step_plus_sizes")
        np.testing.assert_array_equal(out["opcode_ids"], self.batch["opcode_ids"])
        np.testing.assert_array_equal(out["step_type_ids"], self.batch["step_type_ids"])
        np.testing.assert_array_equal(out["stack_depth_ids"], self.batch["stack_depth_ids"])
        np.testing.assert_array_equal(out["env_size_ids"], self.batch["env_size_ids"])
        np.testing.assert_array_equal(out["read_var_ids"], np.full_like(self.batch["read_var_ids"], self.vocab.none_variable_id))
        np.testing.assert_array_equal(out["write_var_ids"], np.full_like(self.batch["write_var_ids"], self.vocab.none_variable_id))
        np.testing.assert_array_equal(out["branch_ids"], np.zeros_like(self.batch["branch_ids"]))
        np.testing.assert_array_equal(out["env_delta_size_ids"], np.zeros_like(self.batch["env_delta_size_ids"]))
        np.testing.assert_array_equal(out["output_flag_ids"], np.zeros_like(self.batch["output_flag_ids"]))

    def test_build_rollout_next_input_row_uses_predictions_with_oracle_sizes(self) -> None:
        predicted = {
            "opcode_ids": 2,
            "step_type_ids": 2,
            "read_var_ids": 3,
            "write_var_ids": 2,
            "read_count_ids": 2,
            "write_count_ids": 1,
            "branch_ids": 1,
            "stack_depth_ids": 5,
            "env_size_ids": 7,
            "env_delta_size_ids": 3,
            "output_flag_ids": 0,
        }
        actual_1d = {
            key: value[0]
            for key, value in self.batch.items()
        }
        row = build_rollout_next_input_row(
            actual_1d,
            1,
            predicted,
            self.vocab,
            "predicted_all_oracle_sizes",
        )
        self.assertEqual(row["opcode_ids"], 2)
        self.assertEqual(row["read_var_ids"], 3)
        self.assertEqual(row["stack_depth_ids"], 5)
        self.assertEqual(row["env_size_ids"], 7)

    def test_build_rollout_next_input_row_predicted_all_uses_predicted_sizes(self) -> None:
        predicted = {
            "opcode_ids": 2,
            "step_type_ids": 2,
            "read_var_ids": 3,
            "write_var_ids": 2,
            "read_count_ids": 2,
            "write_count_ids": 1,
            "branch_ids": 1,
            "stack_depth_ids": 3,
            "env_size_ids": 4,
            "env_delta_size_ids": 3,
            "output_flag_ids": 0,
        }
        actual_1d = {
            key: value[0]
            for key, value in self.batch.items()
        }
        row = build_rollout_next_input_row(
            actual_1d,
            1,
            predicted,
            self.vocab,
            "predicted_all",
        )
        self.assertEqual(row["stack_depth_ids"], 3)
        self.assertEqual(row["env_size_ids"], 4)

    def test_build_rollout_input_batch_updates_future_rows_from_predictions(self) -> None:
        predicted = {
            "opcode_ids": np.array([[2, 0]], dtype=np.int32),
            "step_type_ids": np.array([[2, 0]], dtype=np.int32),
            "read_var_ids": np.array([[3, 0]], dtype=np.int32),
            "write_var_ids": np.array([[2, 0]], dtype=np.int32),
            "read_count_ids": np.array([[2, 0]], dtype=np.int32),
            "write_count_ids": np.array([[1, 0]], dtype=np.int32),
            "branch_ids": np.array([[1, 0]], dtype=np.int32),
            "stack_depth_ids": np.array([[5, 0]], dtype=np.int32),
            "env_size_ids": np.array([[7, 0]], dtype=np.int32),
            "env_delta_size_ids": np.array([[3, 0]], dtype=np.int32),
            "output_flag_ids": np.array([[0, 0]], dtype=np.int32),
        }
        out = build_rollout_input_batch(self.batch, predicted, self.vocab, "predicted_all")
        self.assertEqual(int(out["opcode_ids"][0, 0]), int(self.batch["opcode_ids"][0, 0]))
        self.assertEqual(int(out["opcode_ids"][0, 1]), 2)
        self.assertEqual(int(out["step_type_ids"][0, 1]), 2)
        self.assertEqual(int(out["stack_depth_ids"][0, 1]), 5)
        self.assertEqual(int(out["env_size_ids"][0, 1]), 7)

    def test_build_mixed_rollout_input_batch_replaces_future_rows_when_prob_one(self) -> None:
        predicted = {
            "opcode_ids": np.array([[2, 0]], dtype=np.int32),
            "step_type_ids": np.array([[2, 0]], dtype=np.int32),
            "read_var_ids": np.array([[3, 0]], dtype=np.int32),
            "write_var_ids": np.array([[2, 0]], dtype=np.int32),
            "read_count_ids": np.array([[2, 0]], dtype=np.int32),
            "write_count_ids": np.array([[1, 0]], dtype=np.int32),
            "branch_ids": np.array([[1, 0]], dtype=np.int32),
            "stack_depth_ids": np.array([[5, 0]], dtype=np.int32),
            "env_size_ids": np.array([[7, 0]], dtype=np.int32),
            "env_delta_size_ids": np.array([[3, 0]], dtype=np.int32),
            "output_flag_ids": np.array([[0, 0]], dtype=np.int32),
        }
        mixed, mask = build_mixed_rollout_input_batch(
            self.batch,
            predicted,
            self.vocab,
            "predicted_all",
            replace_prob=1.0,
            rng=np.random.default_rng(0),
            min_teacher_prefix=1,
        )
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 1]))
        self.assertEqual(int(mixed["opcode_ids"][0, 0]), int(self.batch["opcode_ids"][0, 0]))
        self.assertEqual(int(mixed["opcode_ids"][0, 1]), 2)
        self.assertEqual(int(mixed["env_size_ids"][0, 1]), 7)

    def test_build_mixed_rollout_input_batch_respects_prefix_keep(self) -> None:
        predicted = {
            "opcode_ids": np.array([[2, 0]], dtype=np.int32),
            "step_type_ids": np.array([[2, 0]], dtype=np.int32),
            "read_var_ids": np.array([[3, 0]], dtype=np.int32),
            "write_var_ids": np.array([[2, 0]], dtype=np.int32),
            "read_count_ids": np.array([[2, 0]], dtype=np.int32),
            "write_count_ids": np.array([[1, 0]], dtype=np.int32),
            "branch_ids": np.array([[1, 0]], dtype=np.int32),
            "stack_depth_ids": np.array([[5, 0]], dtype=np.int32),
            "env_size_ids": np.array([[7, 0]], dtype=np.int32),
            "env_delta_size_ids": np.array([[3, 0]], dtype=np.int32),
            "output_flag_ids": np.array([[0, 0]], dtype=np.int32),
        }
        mixed, mask = build_mixed_rollout_input_batch(
            self.batch,
            predicted,
            self.vocab,
            "predicted_all",
            replace_prob=1.0,
            rng=np.random.default_rng(0),
            min_teacher_prefix=2,
        )
        self.assertFalse(bool(mask[0, 0]))
        self.assertFalse(bool(mask[0, 1]))
        np.testing.assert_array_equal(mixed["opcode_ids"], self.batch["opcode_ids"])

    def test_predicted_fields_exact_for_step(self) -> None:
        predicted = {
            "opcode_ids": 2,
            "step_type_ids": 2,
            "read_var_ids": 3,
            "write_var_ids": 2,
            "read_count_ids": 2,
            "write_count_ids": 1,
            "branch_ids": 1,
            "stack_depth_ids": 5,
            "env_size_ids": 7,
            "env_delta_size_ids": 3,
            "output_flag_ids": 0,
        }
        actual_1d = {
            key: value[0]
            for key, value in self.batch.items()
        }
        self.assertTrue(predicted_fields_exact_for_step(predicted, actual_1d, 0))
        predicted["output_flag_ids"] = 1
        self.assertFalse(predicted_fields_exact_for_step(predicted, actual_1d, 0))

    def test_rollout_failure_summary(self) -> None:
        summary = summarize_rollout_failures([True, True, False, False])
        self.assertFalse(summary.full_trace_exact)
        self.assertEqual(summary.first_failure_step, 3)
        exact = summarize_rollout_failures([True, True])
        self.assertTrue(exact.full_trace_exact)
        self.assertIsNone(exact.first_failure_step)


if __name__ == "__main__":
    unittest.main()
