# `pg_transformer_research`

Public research fork for our Parameter Golf work.

This is not the official challenge repo. It is our working research fork built
on top of OpenAI's Parameter Golf starter code, with additional trainers,
diagnostics, research notes, and training-system experiments.

- Official challenge repo: <https://github.com/openai/parameter-golf>
- Public fork guide: [PUBLIC_FORK_GUIDE.md](./PUBLIC_FORK_GUIDE.md)

## What This Repo Is

We are treating Parameter Golf as both:

- a **tiny-artifact language modeling** problem
- and a **compute allocation / training systems** problem

The basic thesis is:

- keep the final artifact tiny
- allow training-time structure to be much richer
- use training-time-only scaffolding when it buys BPB
- prove signs-of-life first, then optimize systems later

That has led us to a portfolio of lanes:

- Turbo/QAT mainline compression path
- curriculum and early-exit supervision
- JEPA / long-context / slow-memory branches
- hardmax structural-controller transfer
- latent-state / prosody / branching work
- representation-learning / geometry-prior work
- helper-worker / manager / replay / teacher-cache training systems

## Current Best Public-Facing Mainline

The strongest trusted competition-facing path right now is:

1. current-size Turbo/QAT base
2. `+` phase-gated curriculum
3. `+` early-exit auxiliary supervision
4. `+` `segment-prev-read` as the first structural challenger

Key trusted reference:

- exact roundtrip BPB: `1.43412685`
- artifact size: `6,925,213` bytes
- promotion memo: [20260401_turboquant_transfer_handoff.md](./research/project_wide/20260401_turboquant_transfer_handoff.md)

Best clean additive wins on the main path:

- phase-gated curriculum: `-0.02495940` BPB
- early-exit aux: `-0.01501839` BPB
- segment-prev-read: about `-0.0155` / `-0.0135` on exact probes

## Wild Results We Think Matter

Some of the most interesting results here are not the mainline leaderboard
stack, but they are exactly the kind of weird signals we think are worth
pursuing.

### 1. Execution-Trace Structural Transfer Improves FineWeb BPB

We pretrained a tiny hardmax structural controller on synthetic execution
traces, transferred its learned `state_book` into a language model, and saw a
real FineWeb BPB improvement.

Matched results:

- plain control: `1.88888776`
- matched random structural baseline: `1.85942869`
- trace `state_book` transfer: `1.83989451`
- trace `state_book` + `freeze300`: `1.82215938`

That is a strange transfer path on its face: execution-trace supervision
helping natural-language modeling. It is one of the strongest "weird but real"
results in the repo.

Primary notes:

- [execution_trace_hardmax_lane_20260403.md](./research/project_wide/execution_trace_hardmax_lane_20260403.md)
- [hardmax_structural_controller_lane_20260403.md](./research/project_wide/hardmax_structural_controller_lane_20260403.md)

### 2. Segment-Level Previous-Summary Read Is a Real Long-Context Win

The cleanest JEPA / long-context architectural result so far is not persistent
token-rate carry, but a slower-clock segment memory with previous-summary read.

- exact baseline: `2.10011343`
- `segment-prev-read`, `nocarry`, `pred_weight=0.01`: `2.08463778`

This suggests the useful memory object is a compressed slower-clock summary,
not token-rate recurrent carry.

Primary note:

- [20260329_turboquant_handoff_segmentlong.md](./research/project_wide/20260329_turboquant_handoff_segmentlong.md)

### 3. Training As A Managed System, Not Just One Step Loop

We also have a working `student + manager + helper worker` loop:

- replay queues
- teacher caches
- hidden-state helper workers
- external controller decisions over live training

This is not yet a promoted leaderboard path, but it is an important systems
direction for using training-time compute to improve one final tiny artifact.

## Breakthrough-Area Alignment

The challenge organizers explicitly asked for weird ideas. This repo already
fits several of those buckets:

- `ternary quantization`
  - direct fit via [ternary_quant_mlx.py](./ternary_quant_mlx.py)
- `JEPA`
  - direct fit via sidecar, segment-clock, and harmonic trainers
- `super long context`
  - direct fit via `superlong`, `segmentlong`, and prefix-compiler work
- `parameter tying / recurrence`
  - partial direct fit via reused layer templates and slower-clock latent state
- `test-time compute / adaptive training`
  - partial fit via branching, compiler ideas, replay, teacher caches, and the
    manager/helper-worker control plane

Areas we are adjacent to, but have not fully converted yet:

- `1-bit quantization`
- `H-net tokenization`
- `text diffusion`
- `learning adapters on random linear maps`
- `state-space / E2E TTT`
- `megakernels`

We think this is a strength, not a weakness: a lot of the repo is about
building the right abstractions so those bridges become short.

## Repo Map

- [train_gpt.py](./train_gpt.py): PyTorch benchmark / H100 path
- [train_gpt_mlx.py](./train_gpt_mlx.py): MLX main trainer
- [research/project_wide](./research/project_wide): research memos, lane
  summaries, promotion notes
- [research/iterations/templates](./research/iterations/templates): reusable
  experiment specs
- [tools](./tools): sweep prep, diagnostics, export tooling, analysis, cluster
  helpers
- [tests](./tests): research-tooling and runtime tests

## What Is Intentionally Not In The Public Fork

This public repo excludes:

- downloaded datasets and tokenizer binaries
- generated run trees and archived run artifacts
- local theory dumps
- local cluster access / auth notes
- mirrors of other competitors' submissions
- large representation-learning artifact blobs

That boundary is documented in:

- [PUBLIC_FORK_GUIDE.md](./PUBLIC_FORK_GUIDE.md)

## How To Read This Repo

If you want the shortest path:

1. read [PUBLIC_FORK_GUIDE.md](./PUBLIC_FORK_GUIDE.md)
2. read [20260401_turboquant_transfer_handoff.md](./research/project_wide/20260401_turboquant_transfer_handoff.md)
3. read [20260329_turboquant_handoff_segmentlong.md](./research/project_wide/20260329_turboquant_handoff_segmentlong.md)
4. read [execution_trace_hardmax_lane_20260403.md](./research/project_wide/execution_trace_hardmax_lane_20260403.md)

If you want the official challenge rules, leaderboard, and starter context, use
the upstream repo:

- <https://github.com/openai/parameter-golf>
