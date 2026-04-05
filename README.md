# `pg_transformer_research`

Public research fork for our Parameter Golf work.

This is not the official challenge repo. It is our working research fork built
on top of OpenAI's Parameter Golf starter code, with additional trainers,
diagnostics, research notes, and training-system experiments.

- Official challenge repo: <https://github.com/openai/parameter-golf>
- Public fork guide: [PUBLIC_FORK_GUIDE.md](./PUBLIC_FORK_GUIDE.md)

This repo is not one clever trick. It is a portfolio of direct hits and live
bridges into the exact "weird breakthrough" areas the challenge is asking for:

- direct hits: `ternary/low-bit`, `JEPA/latent prediction`, `slow memory / long context`
- strong bridges: `depth recurrence`, `H-net / hierarchical tokenization`, `TTT / adaptive compute`, `random-map adapters`

We trained a tiny controller to execute self-generated programs on a synthetic
VM. Then we transferred part of it into a language model. FineWeb BPB
improved. This is not yet the model training on its own internal trace; it is
self-generated algorithmic supervision transferred into language modeling. We
do not fully understand why execution-trace structure helps natural-language
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

The useful framing is:

> the hardmax path acts like a low-bandwidth structural compiler pass for the
> softmax LM

Today that claim is still modest. We are not showing that the LM trains on its
own internal trace or that an inner optimizer has emerged. We are showing that
self-generated execution-state supervision can learn a reusable structural
state vocabulary that transfers into language modeling.

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

### 3. Training-Loop Structure Alone Can Move BPB

One of the most useful recent results is not architectural at all. A slimmer
grouped/current-size trainer loop beat the base curriculum trainer on the same
host:

- `current-size-curriculum`: `2.23956102`
- `slim-curriculum`: `2.20450494`
- same host: `mini10`
- delta: `-0.03505608`

That matters because it says some of the remaining headroom is in trainer shape
and order-of-operations, not only in model architecture. We take that as a
precursor to later kernel/fusion work, not a replacement for it.

Read:

- [20260404_mlx-grouped-slim-transfer-ab6-180s/results_summary.md](./research/iterations/generated/20260404_mlx-grouped-slim-transfer-ab6-180s/results_summary.md)

### 4. Our Best Mainline Is Strong Because It Is Small

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

## Multi-Agent Research Orchestration

We treat research coordination itself as a systems problem. This repo includes
a persisted branch-memory control plane: machine-readable node bundles, sweep
manifests, observed results, and queue/runtime snapshots that together define
an MCTS-like research tree. Agents operate on local branches of that tree,
while an orchestrator expands promising nodes, reprioritizes frontier work, and
prunes dead branches. It is not a literal Monte Carlo search engine yet, but
it already behaves like one in practice: structured search state persists
across sessions, parallel agents avoid duplicate work, and experiment outcomes
feed back into the next branch decisions.

The goal is to make research state first-class and machine-readable, so agents
can search the experiment tree directly instead of coordinating through ad hoc
chat and linear task lists.

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

## This Already Lands In The Organizers' Buckets

We did not back-fit our work into the organizer list after the fact. Several
lanes landed there on their own:

- direct hits:
  - `ternary / low-bit` via Turbo/QAT and [ternary_quant_mlx.py](./ternary_quant_mlx.py)
  - `JEPA / latent prediction` via sidecar, segment memory, SIGReg, and hidden-state objectives
  - `super long context / slow memory` via `segment-prev-read`, prefix/compiler memory, and helper caches
- strong bridges:
  - `universal transformer / depth recurrence` via layer-template reuse and controller/state paths
  - `H-net tokenization` via segment/harmonic/prosody/boundary work
  - `state-space / E2E TTT / adaptive compute` via replay, hidden-state caches, helper workers, and the manager bus
  - `random-linear-map adapters` via the geometry-prior / representation-transfer lane

The cleaner way to say it is: this repo keeps rediscovering the categories the
challenge cares about from first principles, then testing whether they actually
move BPB.

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
