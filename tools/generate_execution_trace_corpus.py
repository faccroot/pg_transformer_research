#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_trace_dataset import build_example, example_to_jsonl, ExampleBuilder, GenerationConfig  # noqa: E402


def parse_task_family_mixture(spec: str) -> tuple[tuple[str, float], ...]:
    spec = spec.strip()
    if not spec:
        return ()
    pairs: list[tuple[str, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mixture item {item!r}; expected family:weight")
        name, weight = item.split(":", 1)
        pairs.append((name.strip(), float(weight)))
    return tuple(pairs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic execution-trace corpus for hardmax/controller training.")
    p.add_argument("--num-examples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-statements", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--max-trace-steps", type=int, default=128)
    p.add_argument("--task-family", choices=["straight", "branch", "loop", "nested", "mixed"], default="mixed")
    p.add_argument("--task-family-mixture", default="", help="Optional weighted mixture for mixed-family sampling, e.g. 'straight:1,branch:1,loop:1,nested:2'")
    p.add_argument("--output", default="")
    p.add_argument("--pretty-first", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        seed=args.seed,
        max_statements=args.max_statements,
        max_depth=args.max_depth,
        max_trace_steps=args.max_trace_steps,
        task_family=args.task_family,
        task_family_mixture=parse_task_family_mixture(args.task_family_mixture),
    )
    builder = ExampleBuilder(rng=random.Random(args.seed), config=config)
    examples = [build_example(builder) for _ in range(args.num_examples)]

    if args.pretty_first and examples:
        import json

        print(json.dumps(examples[0], indent=2, sort_keys=True), file=sys.stderr)

    lines = "\n".join(example_to_jsonl(example) for example in examples) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lines, encoding="utf-8")
    else:
        sys.stdout.write(lines)


if __name__ == "__main__":
    main()
