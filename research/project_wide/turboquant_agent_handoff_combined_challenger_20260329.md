# TurboQuant Agent Handoff: Combined Challenger Promotion Path

Purpose: hand off the highest-signal findings from the multi-lane session to the TurboQuant agent, with a concrete promotion path for a combined "challenger" stack and a clear list of features that should stay out unless they are revalidated.

Related note for the new branching / early-exit lane:

- [branching_and_early_exit_ablation_lane.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/branching_and_early_exit_ablation_lane.md)

## Executive Summary

The strongest promotable challenger from this session is:

- `9x512x2`
- untied embeddings
- no EMA
- `LeakyReLU^2` via `MLP_LEAKY_SLOPE=0.5`
- Turbo QAT on from the winning current-size lane
- `turbo_block_v1`
- block size `256`
- `K` in Turbo prod
- `Q/V/O + MLP + lm_head` in Turbo MSE
- clean JEPA sidecar ref
- phase-gated curriculum
- QAT phase gating
- plain Turbo export

Best end-to-end result from this stack:

- `1.34762541` BPB
- artifact `7,108,079` bytes
- source: [20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md)

Best non-sidecar/non-curriculum strong control:

- `1.43412685` BPB
- artifact `6,925,213` bytes
- source: [mini12_taskaware_export_20260326/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/mini12_taskaware_export_20260326/README.md)

Net composed gain:

- `-0.08650144` BPB
- `+182,866` bytes

This is the main promotion result. The combined challenger should start here, not from the plain 4-hour control.

## Promotion Table

| Feature / Decision | Status | Best Evidence | Read |
| --- | --- | --- | --- |
| Current-size combined control: untied + no EMA + `LeakyReLU^2` + Turbo QAT `3/4@256` | Promote now as base control | `1.43412685` BPB, `6,925,213` bytes | Best validated non-sidecar compression lane. |
| Clean JEPA sidecar ref | Promote now | Full composed winner `1.34762541` BPB; sidecar probe ref `2.00630008` vs baseline `2.10114580` | Positive at probe scale and inside the winning 1h/4h composed stack. |
| Phase-gated curriculum | Promote now | Plain 1h: `1.52461262 -> 1.49965322`; combined 1h: `1.52204793 -> 1.50848396` | Real lever is gating, not ordering. |
| Plain Turbo export | Promote now | Strong control and full-stack winner both use this path | Export default should stay simple. |
| Detector-written sidecar | Conditional branch only | 180s probe best: `1.99294778`, delta `-0.10819802` vs baseline | Strong short-probe signal, but not yet the sidecar variant used in the long-run champion. |
| Scaled architecture `12x512x3` | Conditional H100-only branch | 4h scaled run `1.43932836`, artifact `11,449,908` bytes | Under Mini compute it lost to current-size; do not promote as default. |
| Codebook-only export micro-tune, no gauge | Conditional micro-export check only | Earlier mini12 export-harness note showed tiny gains without gauge | Too small and too fragile to block the mainline. |
| Task-aware export stack with gauge / per-tensor codebooks / bitalloc | Do not promote | Full-val standalone export: `1.49196427` baseline vs `1.49466154` best task-aware | Worse BPB for `-150` bytes. |
| Compressibility filter | Do not promote | 1h plain curriculum small positive, but 4h strong control `1.43519626 -> 1.45832297` | Hurts once the model is data-saturated. |
| Heuristic token-category weighting | Do not promote | `1.52665040 -> 1.53971556` | Negative. |
| Short-vs-long context token weighting | Do not promote | `1.52621112 -> 1.56916658` | Negative and slower. |
| EMA-teacher self-distillation | Do not promote | `1.56782029 -> 1.58544478`, about `18.2%` fewer steps | KL did not pay for itself. |
| BOS-reset or persistent sidecar carryover heuristics | Do not promote | Eval persistence and BOS-reset probes both negative | Not a useful near-term lever. |

## Most Important Findings

### 1. The winning composed stack is real, not a probe artifact

- 1h combined sidecar control: `1.52204793`
- 1h combined sidecar + curriculum: `1.50848396`
- 4h combined sidecar + curriculum: `1.34762541`

Sources:

- [20260328_011954_mlx-combined-sidecar-curriculum-ab2-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260328_011954_mlx-combined-sidecar-curriculum-ab2-1h/README.md)
- [20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md)

Interpretation:

- sidecar and curriculum remain positive when composed
- the gain survives a longer wallclock
- this is the strongest end-to-end signal from the session

### 2. Curriculum works because of phase gating, not reordering

Plain 1h curriculum result:

- sequential baseline: `1.52461262`
- order-only: `1.53198915`
- order + phase gating: `1.49965322`

Source:

- [20260326_060710_mlx-curriculum-ab3-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_060710_mlx-curriculum-ab3-1h/results_summary.md)

Interpretation:

- ordering-only is a regression
- phase gating is the actual promotable feature
- any "challenger" curriculum should preserve gating and not spend time refining global ordering by itself

### 3. Sequence-shape analysis says the model should learn spines, not templates

Most important sequence-shape findings:

- exact shapes seen in both train and val: `16.08%`
- shape families seen in both: `33.58%`
- structural spines seen in both: `51.38%`
- operator traffic:
  - `AND`: about `55%`
  - `IF`: about `42%`
  - `NOT`: about `2.76%`

Sources:

- [sequence_shape_findings.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/sequence_shape_deep_dive_20260326_v1/sequence_shape_findings.md)
- [sequence_shape_deep_dive.html](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/sequence_shape_deep_dive_20260326_v1/sequence_shape_deep_dive.html)

Interpretation:

- Phase 2 should target coordination- and conditional-heavy text, not narrow negation-heavy text
- the sidecar story is broader clause scaffolding, not just polarity
- curriculum features and future sidecar probes should be built around spines and clause-linkage programs

### 4. Train and validation are structurally close enough that curriculum transfers

Key train-vs-val deltas:

- compressibility p50: train `0.4934`, val `0.4920`
- operator density p50: both `0.0166`
- val has slightly more contact/account and URL boilerplate

Source:

- [train_vs_val_comparison_20260326.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/train_vs_val_comparison_20260326.md)

Interpretation:

- the current curriculum thesis transfers
- warmdown can bias slightly toward boilerplate/account-ish web text
- there is no evidence for a major train/val structural mismatch

### 5. Structure transfers; memorization mostly does not

Approximate overlap results:

- title-case multi-word spans, rough proper-noun proxy:
  - unique overlap `13.67%`
  - `70.03%` of validation occurrences are val-only
- URLs:
  - unique overlap `1.32%`
  - `97.97%` of validation URL occurrences are val-only

Source:

- [memorization_overlap_20260326.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/memorization_overlap_20260326.md)

Interpretation:

- the structural-over-content thesis is empirically supported
- however, the tested token-weighting implementations were still negative
- use this result to guide curriculum and sidecar design, not to justify current token-weighting code

## Potential Promotion Path

Promotion path for the TurboQuant agent's combined challenger should be:

1. Start from the validated current-size combined control.
2. Add the clean JEPA sidecar ref, not the detector-written sidecar, for the mainline challenger.
3. Add phase-gated curriculum on top of that stack.
4. Keep the export path plain Turbo.
5. Treat this as `Challenger-A`.
6. Run best-of ablations around `Challenger-A`, not around the plain control.

Recommended ablation order:

1. `Challenger-A`
   - current best composed stack
2. `Challenger-A minus curriculum`
   - verifies the curriculum contribution survives on the current checkpoint family
3. `Challenger-A minus sidecar`
   - verifies the sidecar contribution survives on the current checkpoint family
4. `Challenger-A plus detector-sidecar swap`
   - only if there is budget for a side-branch
5. `Challenger-A scaled`
   - H100-only branch, not default Mini promotion

## What Should Stay Out Of The Combined Challenger

Keep these out unless they are independently revalidated with a clean BPB win:

- task-aware export with gauge/per-tensor/bitalloc
- compressibility filtering
- heuristic token-category weighting
- short-vs-long context token weighting
- EMA-teacher distillation
- naive eval-time sidecar persistence
- BOS-reset persistence heuristics
- synthetic-data / self-distillation / adaptive-depth ideas without direct measured BPB gains
- representation-learning / geometry-prior lanes that still lack a direct validated BPB result

## Export Caveat

The standalone export benchmark is still useful for relative ranking inside the export stack, but it does not numerically reproduce the trainer's own final exact export number for the strong 4h checkpoints.

Relevant note:

- [taskaware_export_combined_control_4h_20260327_v1/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/taskaware_export_combined_control_4h_20260327_v1/results_summary.md)

Interpretation:

- use trainer-log exact roundtrip numbers as the primary run record
- use the standalone exporter only for relative export-stack comparisons

## Suggested Starting Config For The TurboQuant Agent

Start from the exact recipe documented here:

- [20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260328_115723_mlx-combined-sidecar-curriculum-4h/README.md)

If a second branch is needed, fork from the same recipe and change only one of:

- sidecar variant
- model scale
- export micro-policy

Do not combine multiple speculative changes in the first challenger pass. The session's strongest lesson is that the big wins came from a few clear training-time choices, while most extra tricks regressed once the recipe was strong.
