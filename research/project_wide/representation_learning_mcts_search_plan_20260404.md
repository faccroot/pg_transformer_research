# Representation Learning MCTS Search Plan

Date: 2026-04-04

## Purpose

This document turns the representation-learning lane into an explicit search problem.

The immediate problem is no longer "what ideas do we have?" We have too many ideas.
The problem is:

- which branches deserve more compute
- which branches should be pruned quickly
- which branches are diagnostics versus product paths
- which experiments should be run in parallel versus sequentially

The right frame is a Monte Carlo Tree Search style policy over implementation branches.

The analogy is:

- a `state` is the current partial implementation plus the evidence collected so far
- an `action` is a new implementation change or experiment
- a `rollout` is a staged evaluation from cheap local diagnostics to expensive Mini or real-target benchmarks
- `backpropagation` is updating our belief about which branches deserve more depth

This is not literal AlphaZero. It is an MCTS-style research and engineering controller.

## Root Objective

Build one deployable student or initialized target model that is better than any single source model because it uses the best transferable structure across the source cohort.

That means the true root win condition is not:

- better cosine to the ecology teacher
- better GCCA residual
- better routing accuracy in isolation

It is:

- better BPB / held-out loss / downstream quality than the strongest single-source baseline under matched budget

Everything else is a latent variable or a verifier for reaching that root goal.

## Current Board State

The current frontier is already much narrower than it was a week ago.

### What is already established

- shared external structure exists:
  - chunk-loss correlations across `Qwen`, `Phi`, `OLMo` are very high
- internal strategies differ:
  - subspace overlaps are much lower
- first direct merge signal exists:
  - `Qwen+OLMo` activation merge is positive on the toy evaluation
- shared-space diagnostics are real:
  - GCCA shared geometry exists across the cohort
- direct functional interchangeability is weak:
  - continuation-mode stitching improved, but is still not strong enough
- routing is learnable:
  - ecology teacher can predict verified winners very well with oracle-like information
- first student lane is real:
  - the kernel-student path now trains end-to-end on Minis

### Current best student frontier

The best current student recipe is:

- direct teacher supervision
- pure distillation
- unfrozen backbone
- `mean_last` readout
- `128x2`
- `8 epochs`
- `1e-3`

Best report:

- `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_capacity_v1/comparison_v1.json`

Best numbers:

- `val_loss_total = 0.734414183625988`
- `val_mean_teacher_cosine = 0.26558581613052906`

### What the last branches already proved

- `mean_last` beats `mean`, `last` is close, learned attention pooling is worse
- modest width/depth growth regresses
- naive disagreement-factor MSE supervision fails badly

Implication:

- the current bottleneck is not "student too small"
- the current bottleneck is not "pooling too dumb"
- the next likely gains are in target formulation and evaluation, not brute-force model scaling

## Search Objects

There are five major subtrees under the root.

### A. Teacher subtree

Question:

- what teacher signal is worth imitating?

Candidate branches:

- current ecology best-of-N teacher
- single-source `Qwen` teacher
- single-source `OLMo` teacher
- static hybrid/moonshot merged teacher
- GCCA / stitched / translated teacher later

Why this subtree matters:

- if the ecology teacher is not better than simpler teachers on downstream BPB, the whole lane is over-engineered

### B. Student-target subtree

Question:

- what target should the student learn?

Candidate branches:

- direct cleared embedding
- normalized factor targets
- factor cosine / correlation targets
- mixture weight prediction plus reconstruction
- contrastive teacher matching

This subtree is now higher priority than capacity search.

### C. Readout/head subtree

Question:

- how should the student expose the representation it is trying to match?

Current evidence:

- simple structural readouts win

Candidate branches:

- `mean_last` baseline
- residual factor head over `mean_last`
- direct head plus low-weight factor regularizer
- multi-head factor prediction with shared trunk

### D. Backbone/capacity subtree

Question:

- does the student need more backbone capacity?

Current evidence:

- modest scaling hurt

This subtree should now have low priority unless a better target/head clearly saturates.

### E. Evaluation subtree

Question:

- which experiments answer product-level value rather than proxy value?

Candidate branches:

- teacher-fit loss only
- student BPB on held-out corpus
- student versus single-source baseline
- student versus static-merge baseline
- student as target initializer on real `9x512`

This subtree is high priority because it is the only one that can tell us whether the student path produces value.

## State Representation

Each state in the tree should be represented by:

- `teacher_type`
- `student_target_type`
- `readout_type`
- `projection_type`
- `backbone_shape`
- `optimizer/budget`
- `evaluation_level`
- `observed_metrics`

Concretely, a state might be:

- teacher = `ecology_best_of_n_v1`
- target = `direct_embedding`
- readout = `mean_last`
- projection = `direct`
- backbone = `128x2`
- budget = `8 epochs @ 1e-3`
- evaluation = `teacher_fit_only`
- metrics = `loss_total=0.7344`, `cos=0.2656`

The next action then changes one coordinate or runs one higher-fidelity rollout.

## Rollout Levels

Use staged rollouts rather than sending every branch straight to a full Mini budget.

### L0. Static plausibility

Cost: free

Checks:

- does the target formulation make sense dimensionally?
- does it duplicate an already-failed branch?
- does it conflict with current evidence?

Use:

- prune obviously bad branches before code

### L1. Local smoke

Cost: cheap

Checks:

- does the code run?
- does the new loss stay numerically sane?
- do the metrics move in the right direction on toy/small runs?

Use:

- implementation validation
- scale sanity checks

### L2. Mini teacher-fit run

Cost: moderate

Checks:

- does the branch beat the current local best on teacher-fit metrics?
- is the trend stable across epochs?
- does it generalize on validation rather than only training?

Use:

- choosing student target / head / optimizer branches

### L3. End-to-end downstream comparison

Cost: high

Checks:

- does the student actually beat single-source or static-merge baselines on BPB / held-out loss?

Use:

- selecting which teacher/target branch is worth keeping alive

### L4. Real target-model benchmark

Cost: highest

Checks:

- does the student or teacher artifact improve the real `9x512` target path?

Use:

- root-level value

## Reward Function

Use a cost-aware reward, not a raw metric.

At low rollout levels:

- reward should emphasize information value and sanity

At high rollout levels:

- reward should emphasize product value

A practical reward for a branch can be:

`Reward = ProductScore + 0.5 * TeacherFitScore + 0.25 * DiagnosticSupport - CostPenalty - FragilityPenalty`

Where:

- `ProductScore`
  - downstream BPB or matched-budget win against baseline
- `TeacherFitScore`
  - normalized distill loss / cosine score
- `DiagnosticSupport`
  - supporting evidence from routing, stitching, GCCA, etc.
- `CostPenalty`
  - Mini time / implementation complexity
- `FragilityPenalty`
  - unstable training, extreme sensitivity, or obvious overfitting

Important rule:

- a branch that only improves teacher-fit but has no plausible path to downstream win should not keep expanding

## Priors

The search should not be uniform. It should use current evidence as priors.

### High prior branches

- `teacher_target = direct_embedding`
  - because it is the current winner
- `readout = mean_last` or `last`
  - because they beat learned attention pooling
- `backbone = 128x2`
  - because modest scaling regressed
- `teacher = ecology_best_of_n`
  - because this is the current central hypothesis to test

### Medium prior branches

- `factorized target`, but only with normalized/cosine-style losses
  - because the theory is good but the first formulation failed
- `single-source teacher baselines`
  - because they are necessary for end-to-end value tests
- `static merged teacher baselines`
  - same reason

### Low prior branches

- more naive width/depth scaling
- more learned pooling variants
- more raw factor-MSE runs

## Progressive Widening Rules

Do not expand every child at once.

Use these rules:

- only widen a subtree after the parent has a stable result
- only add 1-2 sibling branches at a time
- do not open a larger-capacity subtree until the target formulation is stable
- do not open more factorized variants until the normalization problem is fixed

This avoids the common failure mode:

- too many expensive runs on branches whose parent node is still unresolved

## Backpropagation Rules

Every completed branch should update belief at three levels:

### 1. Local branch

Did that exact implementation help?

### 2. Family belief

What did it imply about the family?

Examples:

- `attention readout worse than mean_last`
  - update family belief: learned pooling is low prior
- `160x2 worse than 128x2`
  - update family belief: current bottleneck is not width
- `factorized_k8` collapses because factor loss is huge
  - update family belief: factorization still plausible, raw coefficient MSE is bad

### 3. Root objective

Did it make the path to a better-than-baseline student more or less plausible?

## Principal Variation Right Now

Given the current evidence, the principal variation is:

1. `Teacher branch`
   - keep current ecology teacher as the working teacher

2. `Student target branch`
   - keep direct embedding supervision as the current baseline
   - next child: normalized factor supervision, not raw factor MSE

3. `Head branch`
   - keep `mean_last`
   - next child: direct head + auxiliary factor head

4. `Evaluation branch`
   - compare ecology teacher student against:
     - best single-source teacher student
     - best static-merge teacher student

5. `Root-value branch`
   - whichever teacher branch wins downstream BPB gets promoted into real-target transfer tests

That is the best current line through the tree.

## Immediate High-Value Expansions

These are the next expansions with the best expected value.

### Node 1. Teacher baseline comparison

Question:

- is the ecology teacher actually better than simpler teachers?

Run:

- train the same best student recipe against:
  - current ecology teacher
  - single-source `Qwen` teacher
  - single-source `OLMo` teacher
  - best static merged teacher

Why it matters:

- this is the first end-to-end value test for the ecology path

### Node 2. Auxiliary factor head

Question:

- can factor information help when it is regularized, rather than dominating the objective?

Run:

- keep direct embedding target as primary
- add a low-weight auxiliary normalized factor head

Why it matters:

- this is the correct successor to the failed raw factor-MSE branch

### Node 3. Downstream evaluation on the student itself

Question:

- does lower teacher-fit loss translate into better held-out BPB?

Run:

- evaluate best student checkpoints directly on held-out text

Why it matters:

- otherwise we are still optimizing a proxy with no proof of product value

### Node 4. Student as real-target initializer

Question:

- can the best student signal improve the real target path?

Run:

- use the winning student artifact or its projection head as part of target initialization / guidance

Why it matters:

- this closes the loop between routing-distillation and the real merger product

## Pruning Rules

Hard-prune branches when:

- they are strictly worse than the current best on the same budget and differ only by naive capacity
- they fail for scaling reasons already understood
- they improve only a proxy that we already know is not aligned

Concretely, prune for now:

- more attention-pooling variants
- more naive width/depth increases
- more raw factor-MSE branches

## Logging Discipline

Each run should update:

- the subtree
- the parent-family belief
- whether the root posterior went up or down

So every comparison report should include:

- branch family
- baseline branch
- direct result
- interpretation
- whether to `expand`, `hold`, or `prune`

## Bottom Line

The search policy is now:

- exploit the current strong branch:
  - direct teacher supervision
  - pure distillation
  - `mean_last`
  - small student
- explore only high-value neighboring branches:
  - auxiliary normalized factor heads
  - teacher baseline comparisons
  - downstream BPB evaluation

Do not spend more frontier budget on:

- naive capacity growth
- more learned pooling
- more raw factor-coefficient losses

The best next experiment is not "another clever student architecture."
It is:

- proving whether the ecology teacher is actually worth distilling compared to simpler teachers,

while introducing factor information only through normalized auxiliary targets rather than as the main loss.
