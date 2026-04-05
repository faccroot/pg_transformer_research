# Autonomous Architecture Execution Plan

Date: 2026-03-30

Purpose: provide a structured, long-running execution plan for the outstanding architecture work, result analysis, and code improvement backlog. This is intended to support autonomous progress without re-deriving priorities every turn.

## Mission

Improve the strongest current `Parameter Golf` stack by:

- converting the validated short-run signals into durable architecture improvements
- cleaning up and unifying overlapping implementation lanes
- analyzing results quickly and acting on them with explicit stop/go rules
- keeping speculative work separated from promotable challenger work

## Current Base Truth

The current architecture center of gravity is:

- strong current-size Turbo/QAT control as the anchor
- phase-gated curriculum as a real training-time lever
- early-exit mid-stack supervision as the cleanest new positive signal
- structural branching as the highest-upside, still-underbuilt lane
- causal sidecar as the main long-horizon latent-state lane, but only in clean causal form

The current code state is ahead of the earlier notes:

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py) now has:
  - structural branching
  - cosine gating
  - adaptive branch depth
  - dynamic branch budgeting
  - base-trainer early-exit heads
  - early-exit branch drafting
  - curriculum-routed early-exit weighting
- [structural_branching.py](/home/zaytor/transformer_research/parameter-golf/structural_branching.py) contains the selector and budget controller logic
- [early_exit_aux.py](/home/zaytor/transformer_research/parameter-golf/early_exit_aux.py) now contains both horizon helpers and early-exit budget control

## In-Flight Result Gates

These runs are the near-term decision points:

### Gate 1. Early-exit 4-hour promotion

Run:

- [20260330_035736_mlx-early-exit-aux-promotion-ab2-4h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_035736_mlx-early-exit-aux-promotion-ab2-4h/README.md)

Question:

- does the `-0.015` 1-hour early-exit signal survive longer training?

Action rule:

- if positive at 4 hours, early-exit becomes a promotable component candidate
- if flat or negative, keep it as a branch enabler but not a default promoted feature

### Gate 2. Same-host structural branching rerun

Run:

- [20260330_035736_mlx-structural-branching-samehost-ab2-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_035736_mlx-structural-branching-samehost-ab2-1h/README.md)

Question:

- does the first `-0.035` signal survive after removing the throughput confound?

Action rule:

- if positive on matched host, branching is a real challenger lane
- if it disappears, branching stays a research lane until the integrated cheap-draft version is tested

## Workstreams

## Workstream A. Result Analysis And Decision Hygiene

### Goal

Turn every meaningful run into a fast decision with minimal ambiguity.

### Tasks

1. Maintain `results_summary.md` for every run that reaches a meaningful endpoint.
2. Record:
   - final exact BPB
   - pre-export checkpoint eval
   - step count
   - average step time
   - known confounds
3. Compare not just final BPB, but:
   - same-host fairness
   - step count
   - whether the auxiliary is active over enough of training
4. Update:
   - [architecture_backpass_20260330.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/architecture_backpass_20260330.md)
   - [branching_and_early_exit_ablation_lane.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/branching_and_early_exit_ablation_lane.md)
   when a result changes the promotion logic

### Success Gate

- every architectural lane has an up-to-date durable read
- every new run can be classified as `promote`, `branch`, `hold`, or `drop`

## Workstream B. Mainline Trainer Composition

### Goal

Make the base trainer the home for the best near-term architecture stack:

- `CE`
- `curriculum`
- `early-exit`
- `bounded branching`

### Current Status

Implemented in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py):

- early-exit aux
- early-exit branch drafter
- cosine-gated branching
- adaptive branch depth
- dynamic branching budget
- dynamic early-exit budget

### Remaining Tasks

1. Stage a clean combined bounded A/B from the base trainer:
   - `EARLY_EXIT_AUX_WEIGHT>0`
   - `EARLY_EXIT_BRANCH_DRAFT_ENABLED=1`
   - `STRUCTURAL_BRANCHING_ENABLED=1`
   - `STRUCTURAL_BRANCHING_DYNAMIC_BUDGET=1`
   - `EARLY_EXIT_DYNAMIC_BUDGET=1`
2. Make sure logs expose enough signal to debug:
   - `train_earlyexit`
   - `train_branch_aux`
   - `earlyexit_w`
   - `branch_budget`
   - `branch_points`
   - `branch_score`
3. If the combined run wins:
   - create a dedicated promotion run
4. If the combined run loses:
   - compare against the two single-feature runs before modifying the architecture

### Success Gate

- one clean combined 1-hour A/B with trustworthy result quality

## Workstream C. Branch Objective Refinement

### Goal

Improve the branching lane without overcomplicating it.

### Current Status

Implemented:

- branch selection
- branch ranking
- adaptive scoring horizon
- cheap next-token drafting
- bounded multi-token draft acceptance from existing early-exit horizon heads
- first bounded conditioned/cascaded early-exit draft chain
- explicit branch hidden-state divergence loss

Not implemented:

- Medusa-style or deeper speculative draft trees
- recursive branch trees

### Ordered Tasks

1. Wait for the combined bounded branching result.
2. If positive, add `branch-state divergence` behind a separate flag.
3. Keep the first divergence version minimal:
   - no trees
   - no additional sampling policies
   - no eval adaptation in the same patch
4. Only after that, consider:
   - multi-token cheap draft heads
   - richer future-divergence scoring

### Success Gate

- branching remains stable while the richer loss increases signal

## Workstream D. Curriculum As Compute Router

### Goal

Use curriculum metadata to route *all* expensive structural compute, not just data order and branch budget.

### Current Status

Already routed:

- branch budget
- early-exit aux weight
- sidecar auxiliary scale in the canonical chunk-causal sidecar runner

Still not routed:

- possible fixed recurrence depth
- later eval-time adaptation budgets

### Ordered Tasks

1. Keep the current routing simple and inspect logs.
2. If the current routing is stable, add sidecar aux routing in the canonical sidecar runner.
3. Do not route more than one new mechanism at a time.

### Operating Rule

- `easy` / `diverse`: reduce structured auxiliary spend
- `operator_dense`: spend aggressively on structural auxiliaries
- `hard`: keep selective
- `late training`: allow structured auxiliaries to reopen as corpus freshness declines

### Success Gate

- all structural compute is intentionally placed rather than globally constant

## Workstream E. Canonical Causal Sidecar Lane

### Goal

Reduce the sidecar family to one clean, promotable causal path.

### Current Problem

The sidecar logic is fragmented across:

- [train_gpt_mlx_jepa_sidecar_chunkcausal.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_jepa_sidecar_chunkcausal.py)
- [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py)
- [train_gpt_mlx_superlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_superlong.py)
- older sidecar reference families

### Ordered Tasks

1. Keep the canonical entrypoint fixed at [train_gpt_mlx_sidecar_canonical.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_sidecar_canonical.py).
2. Use [sidecar_canonicalization_plan_20260330.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/sidecar_canonicalization_plan_20260330.md) as the durable routing note.
3. Run:
   - canonical sidecar vs strong control
   - canonical sidecar + curriculum
   - canonical sidecar + early-exit, if the early-exit lane remains positive

### Success Gate

- one sidecar path is clearly the “real” causal sidecar lane

## Workstream F. Separate Eval-Time Adaptation Surface

### Goal

Keep eval adaptation real, but out of the record-path trainer.

### Scope

Future evaluator should handle:

- score-first branch TTT
- sidecar-only adaptation
- fixed extra shared-template recurrence
- later, manifold-address cache or latent preface retrieval

### Ordered Tasks

1. Do not modify the base evaluator further for now.
2. Use the separate saved-artifact structural evaluator at [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py) as the first non-record analysis surface.
3. Keep the structural path bounded and separate from the trainer in [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py); use its emitted branch counts, adaptation counts, and re-score deltas as the comparison contract.
4. Use [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py) for sidecar-only adaptation and re-score on saved sidecar artifacts; use its adaptation counts and re-score deltas the same way.
5. Next evaluator work should be richer non-record adaptation modes, not more trainer flags.

### Success Gate

- evaluation-time adaptation can be tested without destabilizing the training runner

## Workstream G. Next New Architecture

### Goal

Prototype the next major architecture only after the existing stack stabilizes.

### Candidate

- `latent document preface`

Meaning:

- a small latent prefix or prefix-token bank
- inferred from the revealed early document prefix
- refreshed sparsely, not every token
- acts as a slow estimate of document generative regime

### Rule

Do *not* start this before:

1. combined bounded early-exit + branching is measured
2. causal sidecar lane is canonicalized
3. eval-adaptation surface exists separately

## Autonomous Operating Rules

Use these rules to keep progress structured:

1. One primary architectural change per A/B.
   Exception:
   - composition checkpoints are allowed when both components already have isolated positive evidence.

2. Promotion requires:
   - a clean win
   - no obvious host or throughput confound
   - exact exported BPB, not just live train loss

3. If a run is confounded:
   - rerun on matched host before claiming promotion

4. If a lane is positive but small:
   - prefer composing it with the mainline only after one promotion-scale rerun

5. If a lane is speculative and expensive:
   - keep it in a dedicated runner or dedicated evaluator, not in the base path

6. Keep the CE anchor.
   - do not zero it out in branching experiments

7. Keep export fair.
   - any training-only aux heads must be stripped from export

8. Update durable notes whenever a result changes the decision tree.

## Concrete Next Actions

### Immediate

1. Monitor the in-flight 4-hour early-exit promotion run.
2. Monitor the same-host structural branching rerun.
3. Stage the first fully composed bounded run from the base trainer once host capacity is available.

### After The Current Gates Resolve

1. If early-exit and same-host branching are both positive:
   - promote the combined bounded run to highest priority
2. If only early-exit is positive:
   - keep branching as a research lane and move to sidecar cleanup
3. If only branching is positive:
   - keep early-exit as a branch drafter but not a promoted standalone feature
4. If both weaken:
   - prioritize sidecar canonicalization and stop adding branch complexity

## Deliverables

The plan is complete when the repo has:

- one canonical strong mainline trainer
- one canonical causal sidecar trainer
- one separate evaluator for adaptation
- one durable result note per active lane
- one clear promotion tree for what enters the combined challenger and what stays in research
