from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from execution_trace_dataset import (
    AssignStmt,
    BinOpExpr,
    CompareExpr,
    ConstExpr,
    EmitStmt,
    Program,
    VarExpr,
    WhileStmt,
    build_example,
    compile_program,
    execute_program,
    ExampleBuilder,
    GenerationConfig,
)


class ExecutionTraceDatasetTests(unittest.TestCase):
    def test_loop_program_produces_memory_deltas(self) -> None:
        program = Program(
            (
                AssignStmt("x", ConstExpr(3)),
                AssignStmt("acc", ConstExpr(0)),
                WhileStmt(
                    CompareExpr(">", VarExpr("x"), ConstExpr(0)),
                    (
                        AssignStmt("acc", BinOpExpr("+", VarExpr("acc"), VarExpr("x"))),
                        AssignStmt("x", BinOpExpr("-", VarExpr("x"), ConstExpr(1))),
                    ),
                ),
                EmitStmt(VarExpr("acc")),
            )
        )
        instructions = compile_program(program)
        trace, final_env, output, halt_reason = execute_program(instructions, max_steps=64)

        self.assertEqual(halt_reason, "halt")
        self.assertEqual(output, [6])
        self.assertEqual(final_env["x"], 0)
        self.assertEqual(final_env["acc"], 6)
        self.assertTrue(any(event.memory_writes for event in trace))
        self.assertTrue(any(event.memory_reads for event in trace))
        self.assertTrue(any(event.branch_taken is not None for event in trace))

    def test_generator_emits_aligned_views(self) -> None:
        builder = ExampleBuilder(rng=__import__("random").Random(7), config=GenerationConfig(seed=7))
        example = build_example(builder)

        self.assertEqual(example["schema_version"], "execution_trace_v1")
        self.assertIn("source", example)
        self.assertIn("ir", example)
        self.assertIn("trace", example)
        self.assertIn("final", example)
        self.assertIn("python_runtime", example)
        self.assertTrue(example["trace"])
        self.assertEqual(len(example["trace"]), len(example["views"]["trace_text"]))
        self.assertEqual(
            example["python_runtime"]["output"],
            example["final"]["output"],
        )
        self.assertEqual(
            example["python_runtime"]["final_locals"],
            example["final"]["env"],
        )
        self.assertEqual(
            len(example["python_runtime"]["events"]),
            len(example["views"]["python_runtime_trace_text"]),
        )
        self.assertEqual(example["final"]["halt_reason"], "halt")

    def test_nested_family_emits_nested_ast(self) -> None:
        builder = ExampleBuilder(
            rng=__import__("random").Random(17),
            config=GenerationConfig(seed=17, task_family="nested"),
        )
        example = build_example(builder)

        self.assertEqual(example["task_family"], "nested")
        ast_payload = example["source"]["ast"]
        body = ast_payload["body"]
        self.assertTrue(any(stmt["type"] == "while" for stmt in body))
        outer_while = next(stmt for stmt in body if stmt["type"] == "while")
        self.assertTrue(any(child["type"] == "if" for child in outer_while["body"]))
        self.assertTrue(any(child["type"] == "while" for child in outer_while["body"]))
        self.assertEqual(example["final"]["halt_reason"], "halt")

    def test_mixed_family_can_be_weighted_toward_nested(self) -> None:
        builder = ExampleBuilder(
            rng=__import__("random").Random(5),
            config=GenerationConfig(
                seed=5,
                task_family="mixed",
                task_family_mixture=(("nested", 1.0),),
            ),
        )
        example = build_example(builder)

        self.assertEqual(example["task_family"], "nested")
        self.assertEqual(example["requested_task_family"], "mixed")

    def test_cli_writes_jsonl(self) -> None:
        root = Path("/home/zaytor/transformer_research/parameter-golf")
        script = root / "tools" / "generate_execution_trace_corpus.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace.jsonl"
            rc = __import__("subprocess").run(
                [
                    "python3",
                    str(script),
                    "--num-examples",
                    "3",
                    "--seed",
                    "11",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rc.returncode, 0, msg=rc.stderr)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["schema_version"] == "execution_trace_v1" for row in rows))


if __name__ == "__main__":
    unittest.main()
