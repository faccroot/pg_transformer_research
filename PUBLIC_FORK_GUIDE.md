# Public Fork Guide

This repository is being prepared as a public research fork of OpenAI's
Parameter Golf starter repo. The goal is to publish the code, tests, tools,
and research notes that explain the work, while keeping bulky local artifacts,
private scratch space, and downloaded datasets out of the public history.

## What This Fork Contains

- training code for the main Turbo/QAT path
- experimental trainers for JEPA, long-context, curriculum, branching,
  hardmax, latent-state, and representation-learning lanes
- tooling for Mini-cluster sweeps, diagnostics, and helper-lane orchestration
- project-wide research notes that explain the hypotheses, ablations, and
  transfer decisions
- tests for the research tooling and runtime substrate

## What Stays Out Of The Public Fork

- downloaded datasets and tokenizer binaries under `data/`
- generated run artifacts under `research/iterations/generated/`
- archived run directories under `research/iterations/archive/`
- run indexes and local experiment ledgers that embed host/path metadata
- mirrors of other participants' submissions under
  `research/competition_prs/`
- large representation-learning artifacts under
  `research/representation_learning/{calibration,generated,priors,reports,runs}/`
- the local `theory/` dump
- cluster access notes with host/auth details
- local virtualenvs, logs, scratch dirs, and temporary files

## Research Alignment With The Challenge

This fork already overlaps several of the "breakthrough" directions the
challenge organizers explicitly asked to see:

- `ternary quantization`
  - direct fit via [ternary_quant_mlx.py](./ternary_quant_mlx.py)
- `JEPA`
  - direct fit via the sidecar, segment-clock, and harmonic trainers
- `super long context`
  - direct fit via `superlong`, `segmentlong`, and prefix-compiler work
- `parameter tying / recurrence`
  - partial fit via layer-template reuse and slower-clock memory/state objects
- `test-time compute / adaptive training`
  - partial fit via branching, helper workers, replay queues, teacher caches,
    and the external manager/control-plane work

Other requested areas are not direct hits yet, but are natural bridges from the
current codebase:

- `1-bit quantization`
  - adjacent through Turbo/QAT, ternary work, and discrete-state lanes
- `H-net tokenization`
  - adjacent through harmonic segmentation, prosody features, and hierarchical
    units
- `text diffusion`
  - adjacent through iterative compiler/branch/refinement ideas
- `learning adapters on random linear maps`
  - adjacent through the representation-learning and geometry-prior lane

## Wild Results Worth Highlighting

Some of the most interesting results in this repo are exactly the kind of
"weird but real" signs-of-life the challenge asks for:

- A tiny hardmax structural controller pretrained on synthetic execution traces
  improves FineWeb BPB after transfer into a language model.
  - plain control: `1.88888776`
  - matched random structural baseline: `1.85942869`
  - trace `state_book` transfer: `1.83989451`
  - trace `state_book` + `freeze300`: `1.82215938`
- A JEPA-derived segment-level previous-summary read path produced a clean exact
  improvement over baseline in the long-context lane.
- The compute-aware `student + manager + helper worker` loop now runs end to
  end, which gives us a path toward adaptive training and asynchronous signal
  generation instead of one monolithic sequential run.

These branches are not all immediate leaderboard candidates, but they are the
kind of nonstandard ideas we want the public repo to showcase.

## Current Best Public-Facing Mainline

If someone wants the shortest route to the current strongest trusted path, it
is:

1. trusted current-size Turbo/QAT base
2. `+` phase-gated curriculum
3. `+` early-exit auxiliary supervision
4. `+` `segment-prev-read` as the first structural challenger

The clearest promotion memo for that stack is in
[research/project_wide/20260401_turboquant_transfer_handoff.md](./research/project_wide/20260401_turboquant_transfer_handoff.md).

## Public Push Workflow

Before pushing a public branch:

1. review `git status` and confirm generated/data dirs are ignored
2. keep code, tests, templates, and the publishable
   `research/project_wide` notes
3. do not add `theory/`, local envs, logs, or dataset artifacts
4. do not add mirrors of other participants' submissions
5. do not add cluster access notes or raw run indexes with host/path metadata
6. prefer including concise markdown summaries over raw generated run trees
