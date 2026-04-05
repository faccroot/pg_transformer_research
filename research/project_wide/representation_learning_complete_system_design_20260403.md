# Representation Learning Complete-System Design

Date: 2026-04-03

## Purpose

This document resets the representation-learning lane around the strongest full system we actually want to build.

The immediate problem has not been lack of ideas. It has been a repeated engineering pattern:

- implement a cheap proxy for a strong idea
- run a quick ablation
- use the quick ablation to steer the architecture

That pattern is useful for pruning obviously bad branches, but it has repeatedly under-tested the strongest concepts:

- continuous routing instead of hard winner classification
- multi-view shared geometry instead of pairwise similarity diagnostics
- functional verification instead of only winner labels from held-out loss
- factor-aware clearing instead of scalar summaries

This document treats the literature-backed full system as the default target, and treats quick local ablations as diagnostics rather than architecture setters.

## Goal

Build a continuous cross-model clearing system over heterogeneous language models.

The system should:

- ingest `N` source models of different architectures and sizes
- extract typed high-dimensional representational events from each model
- learn a shared multi-view geometry over those representations
- verify functional interchangeability where possible
- predict per-factor mixture weights rather than only a single hard winner
- distill the cleared representation into one deployable student model

The product is not a static interpretability report.

The product is:

- a learned routing / clearing kernel over model representations
- a compressed student initialized or trained from that kernel

## What The Current Results Already Prove

The current local and prior-cluster results justify continuing the lane, but they do not yet validate the full system.

What is already real:

- different source models do own different parts of the representational surface
- shared chunk difficulty is highly correlated across models while internal geometry differs materially
- a small activation merge already gives a positive toy result
- a learned ecology model can predict verified winners very well when it has oracle-like loss features
- non-oracle structure-only routing is possible, but still materially below the oracle ceiling

The practical implication is:

- cross-model transfer looks feasible
- the remaining problem is architecture quality, not basic existence of signal

## Failure Mode To Avoid

The main failure mode is the scalar trap.

We saw this directly:

- repaired per-chunk geometry helped
- scalar forward summaries mostly hurt
- disagreement factors helped in some lanes, but only partially

That means the next system should default to:

- multi-view geometry
- factor-aware structure
- functional verification

not to more scalar feature engineering.

## External Methods Map

The external literature suggests a clean separation of roles.

### 1. Representation comparison diagnostics

Use these to compare or monitor representations, not as the merger itself.

- SVCCA
  - [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability)
- CKA
  - [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html)
- contextual representation comparison
  - [A Similarity Analysis of Contextual Word Representation Models](https://aclanthology.org/2020.acl-main.422/)

How they map to our system:

- layer diagnostics
- sanity checks for shared geometry
- monitoring whether a new alignment branch is behaving better or worse

How they do **not** map:

- they are not the routing kernel
- they are not the transfer operator

### 2. Shared multi-view latent geometry

This is the closest direct external match to our “many models, one latent manifold” problem.

- DGCCA
  - [Deep Generalized Canonical Correlation Analysis](https://arxiv.org/abs/1702.02519)

Why it matters:

- pairwise CCA is the wrong scaling primitive for a true model zoo
- DGCCA explicitly learns a shared latent representation from arbitrarily many views
- our source models are naturally “views” of the same underlying language structure

How it maps to our system:

- `shared-space learner`
- input: typed activation events or factor banks from each source model
- output: one multi-view latent geometry plus per-model projection heads

### 3. Functional interchangeability verification

This should become one of the core verification operators.

- model stitching
  - [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)

Why it matters:

- winner labels from held-out loss are weak
- stitching directly tests whether one model’s internal representation can be consumed by another model’s later computation
- that is much closer to the actual “is this transferable?” question

How it maps to our system:

- `functional verifier`
- candidate source representation survives if it can be stitched / consumed with low degradation

### 4. Same-architecture or near-isomorphic weight synthesis

These methods are real, but they belong to a separate branch.

- Git Re-Basin
  - [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
- Fisher-weighted averaging
  - [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832)
- optimal transport fusion
  - [Model Fusion via Optimal Transport](https://proceedings.neurips.cc/paper/2020/hash/fb2697869f56484404c8ceee2985b01-Abstract.html)

Why they matter:

- they are strong references for checkpoint synthesis and same-arch merges
- they can become the downstream synthesis layer after the routing/kernel layer is mature

Why they are not the universal answer:

- they assume much stronger neuron or parameter correspondence than we have across arbitrary model families
- they are better thought of as `late-stage synthesis operators`, not as the universal kernel

## Complete System Architecture

There should be six long-lived modules.

### A. Typed Activation Event Builder

Inputs:

- source checkpoints
- calibration / probe corpus
- optional generated probes

Outputs:

- `ActivationEventDataset`

Each event should eventually include:

- model id
- layer id
- hidden-state embedding
- per-factor coordinates
- per-factor topology against other models
- token distribution object or compressed surrogate
- optional verifier-side signals

The important design decision:

- event objects should stay high-dimensional
- avoid reducing them to scalar summaries unless the scalar is itself a routing target

### B. Multi-View Shared-Space Learner

Method family:

- DGCCA-style shared latent geometry

Outputs:

- one shared latent geometry
- one projection head per source model

This replaces the current implicit mixture of:

- pairwise similarity diagnostics
- ad hoc factor PCA
- local feature unions

### C. Functional Verifier

Verification should become explicitly layered.

Verification operators:

- held-out chunk loss
- model stitching
- downstream task correctness
- code execution / factual lookup where possible

The key rule:

- held-out loss is the cheap bootstrap verifier
- stitching is the preferred transfer verifier

### D. Continuous Clearing Kernel

This is the central architecture.

It should not only predict:

- “which model wins?”

It should predict:

- per-factor mixture weights
- optionally per-factor source-layer preferences

This is the shift from classifier to clearing engine.

Output shape should eventually look like:

- `chunk x factor x source_model`

rather than:

- `chunk -> source_model`

### E. Student Compiler

The student should be trained against the cleared representation, not against one teacher.

Targets:

- cleared factor mixtures
- cleared routed hidden states
- optionally routed logits as auxiliary loss

This is the real model-merger product.

### F. Same-Arch Synthesis Branch

Once the kernel is mature, use:

- re-basin
- Fisher merging
- OT-based fusion

to synthesize explicit checkpoints where the architecture assumptions make that defensible.

This branch is downstream of the kernel, not upstream of it.

## Implementation Order

This is the recommended high-bar implementation sequence.

### Phase 1. Stabilize the data model

No more ad hoc feature sprawl.

Finalize the interfaces for:

- `ActivationEventDataset`
- `EcologyTrainingSet`
- `VerificationTable`
- `RoutingKernel`
- future `SharedLatentGeometryArtifact`

Exit criteria:

- all new branches consume the same event schema
- new feature families land as typed blocks, not one-off scalars

### Phase 2. Build the DGCCA branch

Implement a real multi-view shared latent learner over current source cohorts.

Deliverables:

- trainable DGCCA-style shared space
- per-model encoder/projection heads
- artifacts saved per layer or layer-band

Exit criteria:

- shared-space branch beats current ad hoc factor PCA on downstream structural routing quality

### Phase 3. Build the stitching verifier

Implement a proper stitching-based verification harness.

At minimum:

- same-layer or nearby-layer stitching checks
- degradation score per source pair and region

Exit criteria:

- we can label not only “winner model” but “transferable representation region”

### Phase 4. Replace hard winner routing with per-factor clearing

Implement the first continuous mixture target.

Start with:

- per-factor mixture weights over source models

Then extend to:

- per-factor source-layer routing

Exit criteria:

- mixture routing beats hard routing on held-out verification

### Phase 5. Distill the student

Train a student on:

- cleared hidden-state targets
- auxiliary logits
- compression-friendly regularization

Exit criteria:

- student beats best single-source prior under fixed training budget

### Phase 6. Downstream checkpoint synthesis

Only after the kernel is working should we invest heavily in:

- same-arch weight synthesis
- checkpoint fusion operators

## What We Should Stop Doing By Default

- stop using scalar forward features as the main new branch
- stop letting quick local ablations choose the architecture
- stop treating pairwise similarity measures as if they are the merger
- stop using hard winner classification as the conceptual endpoint

Quick ablations are still useful, but only for:

- debugging
- regression testing
- local ranking inside an already-justified branch

## Immediate Next Tranche

The next serious implementation tranche should be:

1. `DGCCA shared-space branch`
2. `stitching verifier`
3. `factor-topology mixture router`

Recommended order:

1. build the DGCCA artifact and training harness
2. build stitching verification on the current `Qwen/Phi/OLMo` cohort
3. use DGCCA outputs plus stitching labels to train a per-factor mixture router

Only after those exist should we run the next major comparison sweep.

## Success Criteria For The Next Tranche

The next tranche is successful if any of these happens:

- DGCCA-based structural routing beats the current repaired-`Phi` structural baseline
- stitching labels provide a better transfer target than held-out-loss winner labels
- per-factor mixture routing beats hard winner routing

If none of those happen, that is still useful:

- it would mean the current problem is not “insufficient implementation of the strong idea”
- it would mean the strong idea itself needs revision

Right now we are not at that point. We have not yet tested the strongest version of the system.
