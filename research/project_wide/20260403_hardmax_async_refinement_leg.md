# Hardmax Async Refinement Leg (2026-04-03)

## Purpose

Add a new MCTS branch for always-on hardmax refinement that does not depend on the failed confidence-to-budget router.

The question is not "should the trunk be skipped?"

The question is:

- can a small hardmax controller perform multiple cheap refinement updates
- on a compressed structural view of the trunk/input state
- and feed a better annotation back into the same fixed-budget softmax trunk

This is the runtime-efficiency leg for the hardmax lane.

## Why this branch exists

The current hardmax evidence points in one direction:

- structural conditioning helps
- routing hurts
- execution-trace supervision keeps the controller alive

That suggests the hardmax should be treated as an always-on fast conditioning path, not as a discrete router.

The architecture target is closer to:

- fast structural micro-updates
- slower trunk updates
- concurrent or staggered conditioning

than to:

- confidence -> skip compute

## Current implementation seam

The first version of this branch is deliberately conservative.

The repo already has two useful seams:

1. The controller call site inside the LM trunk
   - [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
   - `maybe_apply_hardmax_structural_controller(...)`

2. A real helper/manager architecture for off-critical-path work
   - [run_train_with_manager.py](/home/zaytor/transformer_research/parameter-golf/run_train_with_manager.py)
   - [teacher_hidden_cache_worker.py](/home/zaytor/transformer_research/parameter-golf/teacher_hidden_cache_worker.py)
   - [snapshot_signal_runtime.py](/home/zaytor/transformer_research/parameter-golf/snapshot_signal_runtime.py)

What we do not have yet is same-device kernel overlap scheduling inside MLX.

So the branch is staged in this order:

## Branch structure

### G1: Sequential Microstep Baseline

Implemented via:

- `HARDMAX_STRUCT_FAST_REFINEMENT_STEPS`

This runs the existing hardmax controller multiple times per application site before returning the structural annotation to the trunk.

This is the cleanest first test because it isolates the modeling question:

- do repeated fast hardmax updates help?

without mixing in runtime/helper complexity.

### G2: Next-Step Async Helper Annotations

Planned next if `G1` is positive.

Use the existing manager/worker pattern to compute hardmax refinement off the critical path and feed the result into the next step or next refresh window.

This is the first real "fill the gaps" implementation, but at process granularity rather than device-stream granularity.

### G3: Helper-Tuned Microstep Controller

Planned.

Allow helper/controller decisions to tune:

- `hardmax_micro_steps`
- maybe later structural temperature or compression mode

without changing the main training loop structure.

### G4: Top-K / Compressed-State Refinement

Planned.

Restrict refinement to:

- a compressed projection of the trunk state
- or a top-K subset of structurally salient positions

This is the first step toward a genuinely cheap fast path rather than just "run the same controller more times."

### G5: Same-Device Overlap / Tail-Filling

Deferred.

This is the strongest systems version of the idea:

- use tail/slack phases in the training step
- run hardmax work on the same device in otherwise underused windows

That is not the first experiment because the current MLX trainer is still fundamentally synchronous around gradient realization and `mx.synchronize()`.

## Current code hook

The first hook is now in:

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

New knob:

- `HARDMAX_STRUCT_FAST_REFINEMENT_STEPS`

Current behavior:

- `1` means the old behavior
- `2` or `4` means multiple controller updates before the annotation is returned

The external controller path can also now adjust `hardmax_micro_steps` through the existing decision bus, but that is future-facing and not required for the first smoke.

## First smoke

Template:

- [mlx_hardmax_trace_transfer_microsteps_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_transfer_microsteps_smoke.example.json)

Generated sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_050137_mlx-hardmax-trace-transfer-microsteps-smoke/manifest.json)

Queue session:

- `31470`

First comparison:

- `control`
- `d32` random-init `structonly`
- `d32` trace-core with `micro_steps=1`
- `d32` trace-core with `micro_steps=2`
- `d32` trace-core with `micro_steps=4`

This keeps the comparison focused on the new branch:

- same trunk
- same controller size
- same transfer init
- only the number of refinement steps changes

## Success criteria

`G1` counts as positive if it improves one of the transfer arms without killing controller liveness.

Primary:

- better exact val BPB than `trace-core micro_steps=1`

Secondary:

- hard-state peak usage stays non-degenerate
- confidence variance does not collapse
- transfer diagnostics remain healthier than the old shallow-label lane

Runtime:

- throughput cost stays materially below the cost of a larger trunk ablation that buys the same gain

## Prune rule

Prune the runtime-heavy versions of this branch if the sequential baseline is flat or negative.

That means:

- do not build helper overlap
- do not build top-K compression machinery
- do not attempt same-device tail-filling

unless `G1` first shows a real modeling benefit.

## Early read

The first finished arm from the microsteps sweep was not positive:

- `structonly-step800-n8-d32-trace-core-micro1`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.89123827`

That is worse than:

- the matched `d32` random structural baseline
- the winning `trace-statebook` transfer branch

So the current branch state is:

- `trace-core` transfer is weak in this runtime leg
- `micro4` is still worth observing before pruning
- do not expand to helper/runtime overlap unless a higher-microstep arm actually beats the single-step baseline
