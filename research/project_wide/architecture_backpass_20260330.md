# Architecture Backpass: What The Results Actually Say

Date: 2026-03-30

Purpose: review the validated results, convert them into architecture conclusions, and prioritize the next refinement work while the current promotion runs are still in flight.

## Bottom Line

The stack is currently telling a fairly clean story:

- plain training-time structure beats clever export tricks
- phase gating is real and should control more of the compute budget
- small fixed-size latent structure is promising, but only the causal versions should be trusted
- mid-stack supervision is real
- branching is high-upside, but still underbuilt

The safest current mental model is:

- `CE anchor` for calibration
- `curriculum` for when to spend compute
- `early-exit heads` for cheap structured forecasting
- `branching` for rich counterfactual supervision
- `causal sidecar` for compact belief-state structure over longer horizons

## What Is Actually Validated

### 1. Strong mainline control is still the anchor

Use the current-size combined Turbo/QAT recipe as the base control, not the plain baseline.

This remains the correct starting point for every new challenger.

### 2. Phase-gated curriculum is a real architecture-adjacent win

This is no longer just a data-ordering trick. The result says:

- order alone is not the point
- phase gating is the point
- curriculum metadata should route auxiliary compute, not just chunk sampling

Practical implication:

- curriculum should control branch budget
- curriculum should control sidecar auxiliary scale
- curriculum should likely control early-exit auxiliary scale as well

### 3. Mid-stack early-exit supervision is the cleanest new positive lane

Validated 1-hour result:

- early-exit aux improved exact BPB by about `-0.015`

Interpretation:

- the model benefits from structured supervision at an intermediate layer
- the auxiliary is cheap enough to be a good permanent component candidate
- the same heads should be reused for branch drafting rather than living as a separate experiment

### 4. Structural branching is promising, but the first measured win is not yet promotable

Measured 1-hour result:

- branching improved exact BPB by about `-0.035`

But:

- the first run was badly throughput-confounded

Interpretation:

- the mechanism is stable enough to train
- the signal is strong enough to justify more work
- the result is still not trustworthy enough to promote directly

### 5. Sidecar remains important, but only the causal family should count

The old sidecar family gave strong headline numbers, but causality issues force a harder rule:

- use only chunk-causal or otherwise clean causal sidecar paths as architecture evidence
- treat earlier flashy sidecar wins as research signals, not promotion records

This means the sidecar lane is still alive, but it needs cleanup more than more variants.

## What To Stop Spending Time On

These lanes should stay deprioritized unless new evidence appears:

- export-stack complexity as a primary improvement path
- per-token weighting heuristics
- compressibility filtering as a default training feature
- EMA-teacher distillation
- naive eval-time persistence or reset heuristics

None of these currently looks like the shortest path to a better challenger.

## The Main Architecture Thesis Now

The architecture should separate four roles:

1. `Literal evidence`
   - recent token context and ordinary autoregressive CE
2. `Cheap structural forecast`
   - mid-stack early-exit heads
3. `Explicit hypothesis testing`
   - bounded structural branching
4. `Compact longer-horizon belief state`
   - causal sidecar / segment-long state

That is the clean decomposition the results are pushing toward.

## Priority Work While Runs Are In Flight

### Priority A. Unify early-exit and branching

This is the highest-value immediate refinement.

The right architecture is not:

- one lane for early-exit aux
- a separate lane for full-stack branching

It is:

- early-exit heads provide the cheap branch drafter
- branching provides richer supervision that should improve those heads

The current code now reaches the first bounded version of that idea:

- early-exit heads can draft a short contiguous multi-token prefix for branch rollouts
- branching can add both rank loss and optional branch-state separation loss

Required work:

- finish the integrated `CE + early-exit + branching` path in the base trainer
- run one clean bounded A/B with:
  - `EARLY_EXIT_AUX_WEIGHT > 0`
  - `EARLY_EXIT_BRANCH_DRAFT_ENABLED=1`
  - cosine-gated branching
  - adaptive branch depth
  - dynamic branch budget

Expected value:

- highest near-term upside
- directly composes two already-positive or promising signals

### Priority B. Make curriculum the general compute router

This is probably the most underexploited thing already in the repo.

We now have:

- phase metadata
- operator density
- compressibility proxies

Those should route more than branch count.

Next routing targets:

- early-exit auxiliary weight by phase
- sidecar auxiliary scale by phase
- maybe fixed recurrence / extra shared-template depth by phase

Practical principle:

- `easy/diverse`: low auxiliary compute
- `operator_dense`: aggressive branching and stronger structural auxiliaries
- `hard`: selective branching only
- `late training`: reopen richer auxiliaries as raw data gets stale

### Priority C. Canonicalize the causal sidecar lane

The sidecar code is split across too many runners:

- chunk-causal JEPA sidecar
- segment-long
- superlong carryover
- older ref families

What is needed now is not more sidecar creativity. It is one clean, promotable lane:

- one canonical causal sidecar trainer
- one canonical long-context evaluation path
- one canonical curriculum-composed experiment

This is now partially addressed by:

- [train_gpt_mlx_sidecar_canonical.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_sidecar_canonical.py)
- [sidecar_canonicalization_plan_20260330.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/sidecar_canonicalization_plan_20260330.md)

Questions to answer:

- does the clean chunk-causal family still give a real gain at 1 hour?
- does it compose with the current strong control?
- does it compose with early-exit?
- can transition-aware reset preserve the within-regime gain while reducing
  cross-regime stale-belief carry?

The residual-autocorrelation read now makes the next sidecar refinement very
specific:

- sidecar is acting like a regime-conditional state estimator
- it helps inside stable regimes
- it is too sticky across regime boundaries

So the next sidecar architecture should not just be `more sidecar`. It should
be:

- sidecar plus transition detector
- sidecar plus partial reset
- sidecar plus reset-gate discipline
  - sparse by default
  - weakly aligned to transition evidence, not hard-forced
- and then residual-ACF remeasurement, especially `cross_regime`

### Priority D. Add branch-state divergence only after the combined lane is stable

Branch ranking is already implemented in a minimal form.

The next meaningful upgrade is:

- hidden-state divergence between right and wrong branches

But this should come after the early-exit-drafter integration is validated, not before.

Reason:

- it is the clean next increase in signal richness
- but it adds another degree of freedom and should not be stacked on top of an unstable branch walker

### Priority E. Build the separate eval/adaptation surface

Do not keep adding evaluation-time adaptation features to the main trainer.

Make a separate evaluator for:

- score-first branch TTT
- sidecar-only adaptation
- fixed extra shared-template recurrence
- later, manifold-address or cached latent-state experiments

The first evaluator surfaces now exist at:

- [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py)
- [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py)

It now also contains the first bounded score-first branch adaptation loop for
saved base artifacts, including a bounded two-pass adapt-plus-rescore path. The
sidecar evaluator now also contains sidecar-only retrospective adaptation for
saved sidecar artifacts, with a bounded post-adapt re-score path. The next step
there is not more trainer integration; it is richer evaluator-only adaptations
inside the separate surfaces.

This remains a real lane, but it should stay separate from record-path training.

## The Next New Architecture After The Current Stack

The next genuinely new architecture idea worth building is:

- `latent document preface`

Meaning:

- a small set of latent prefix tokens or vectors
- inferred causally from the first observed tokens
- updated sparsely, not every step
- acts as a slow estimate of "what generative process produced this document"

But this is not the next patch.

It should come only after:

1. early-exit and branching are unified cleanly
2. the causal sidecar lane is canonicalized
3. the separate eval/adaptation surface exists

Otherwise too many ideas are competing at once.

## Suggested Build Order

### Immediate

1. Finish and test `early-exit as branch drafter`.
2. Run combined bounded `early-exit + branching`.
3. Let curriculum control those budgets.

### Next

1. Canonicalize and rerun the clean chunk-causal sidecar.
2. Test `sidecar + curriculum`.
3. Test `sidecar + early-exit`.

### After That

1. Add branch-state divergence.
2. Add a dedicated eval adaptation wrapper.
3. Prototype latent document preface in a separate runner.

## Practical Operating Rule

If a change does not strengthen one of these three things, it is probably not the best use of time right now:

- better structured supervision per step
- better routing of compute to structurally valuable data
- better causal compression of long-horizon context

That is the current architectural center of gravity.
