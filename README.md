# `pg_transformer_research`

Public research fork for our Parameter Golf work.

This is not the official challenge repo. It is our working research fork built
on top of OpenAI's Parameter Golf starter code, with additional trainers,
diagnostics, research notes, and training-system experiments.

- Official challenge repo: <https://github.com/openai/parameter-golf>
- Public fork guide: [PUBLIC_FORK_GUIDE.md](./PUBLIC_FORK_GUIDE.md)

We trained a tiny controller to execute programs on a synthetic VM. Then we
transferred part of it into a language model. FineWeb BPB improved. We do not
fully understand why execution-trace structure helps natural-language
compression, but it does:

- plain control: `1.88888776`
- matched random structural baseline: `1.85942869`
- trace `state_book` transfer: `1.83989451`
- trace `state_book` + `freeze300`: `1.82215938`

That is the tone of this repo: not "safe tweaks to a baseline," but repeated
attempts to find compression-significant structure that should not obviously
work, then test it cleanly enough that the result survives skepticism.

## Results That Shouldn't Work But Do

### 1. Execution-Trace Transfer Helps FineWeb

The cleanest weird result in the repo is the hardmax/controller lane:

- pretrain a tiny structural controller on synthetic execution traces
- transfer its learned `state_book` into the LM
- briefly freeze that transferred vocabulary so the trunk learns to use it
- FineWeb exact BPB improves materially

This is not just "add a small side path." The matched random structural
baseline is already strong, and the trace-pretrained transfer still beats it.

Read:

- [execution_trace_hardmax_lane_20260403.md](./research/project_wide/execution_trace_hardmax_lane_20260403.md)
- [hardmax_structural_controller_lane_20260403.md](./research/project_wide/hardmax_structural_controller_lane_20260403.md)

### 2. Memory Helps Only When It Runs On A Slower Clock

In the long-context lane, fast token-rate persistent carry was negative. The
first real architectural win came from a slower-clock memory object: segment
summaries with previous-segment read.

- exact baseline: `2.10011343`
- `segment-prev-read`, `nocarry`, `pred_weight=0.01`: `2.08463778`

The important claim is not just "this one config won." It is:

> Memory helped only when it ran on a slower clock than token rate.

That is a stronger idea than this one repo.

Read:

- [20260329_turboquant_handoff_segmentlong.md](./research/project_wide/20260329_turboquant_handoff_segmentlong.md)

### 3. Our Best Mainline Is Strong Because It Is Small

The strongest trusted competition-facing foundation is the Turbo/QAT current-size
stack. Its interesting property is not only score. It is score **plus**
headroom:

- trusted exact roundtrip BPB: `1.43412685`
- compressed artifact size: `6,925,213` bytes
- remaining room under the `16,000,000` byte cap: about `9.07 MB`

That matters because this is not a near-maxed-out artifact. It is a strong
under-cap platform with room to add structure.

The best clean validated levers on top of that path are baseline-relative:

- phase-gated curriculum: `1.52461262 -> 1.49965322`
- early-exit aux: `1.59840419 -> 1.58338580`
- segment-prev-read: `2.10011343 -> 2.08463778`

These numbers come from different matched ablations and runtimes, so they
should be read as **within-lane deltas**, not as one flat leaderboard.

Read:

- [20260401_turboquant_transfer_handoff.md](./research/project_wide/20260401_turboquant_transfer_handoff.md)

## Thesis

Small models are bottlenecked less by raw parameter count than by supervision.
Standard next-token prediction gives very little direct signal per position to
shape a large internal state space. Much of the model lives in the null space
of the loss.

We attack that gap by adding extra structure where it buys delivered BPB:

- auxiliary supervision
- slower-clock memory objects
- structural state vocabularies
- curriculum and phase-gating
- hidden-state teacher signals
- managed helper lanes and replay/teacher caches

Then we try to strip the result back to a tiny exportable artifact.

## What We Killed And Why

This repo is not just wins. A lot of the value is in what we ruled out.

- `token-rate persistent carry`: dead. The carry object itself was bad, not
  just expensive.
- `budget routing / compute gating`: dead in the hardmax lane. Always-on
  conditioning beat gating.
- `order-only curriculum`: dead. The real lever was phase gating, not ordering
  by itself.
- `current harmonic read + JEPA stack`: dead as a modeling win so far, even
  though the tensorized systems rewrite was a major engineering speedup.

The pruning matters. It means the positive branches are surviving a real search
process, not one lucky idea.

## Why This Naturally Lands In The Organizers' Buckets

We did not retroactively fit our work into the organizers' list. Several lanes
landed there independently:

- `JEPA / super long context`
  - arrived at through segment-clock memory and slower summaries
- `parameter tying / recurrence`
  - arrived at through reused layer templates and explicit state/controller
    paths
- `test-time compute / adaptive training`
  - arrived at through profiling, then helper workers, replay queues, teacher
    caches, and an external manager
- `ternary quantization`
  - directly implemented in [ternary_quant_mlx.py](./ternary_quant_mlx.py)

Other requested areas are clear bridges from the current codebase:

- `1-bit quantization`
- `H-net tokenization`
- `text diffusion`
- `learning adapters on random linear maps`
- `state-space / E2E TTT`
- `megakernels`

## If You Want To Read One Thing

- If you want the result that surprised us most:
  [execution_trace_hardmax_lane_20260403.md](./research/project_wide/execution_trace_hardmax_lane_20260403.md)
- If you want the cleanest validated competition-facing win:
  [20260401_turboquant_transfer_handoff.md](./research/project_wide/20260401_turboquant_transfer_handoff.md)
- If you want the strongest long-context architectural result:
  [20260329_turboquant_handoff_segmentlong.md](./research/project_wide/20260329_turboquant_handoff_segmentlong.md)
- If you want to see what a managed training system looks like:
  [student_manager_worker.py](./student_manager_worker.py),
  [teacher_hidden_cache_worker.py](./teacher_hidden_cache_worker.py), and
  [snapshot_signal_runtime.py](./snapshot_signal_runtime.py)

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

Most of this was built on a 14-node Mac Mini M4 Pro cluster. No H100 access
yet.
