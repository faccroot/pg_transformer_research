# Hardmax Hypothesis-Driven MCTS (2026-04-04)

This note reframes the hardmax search tree around theory tests rather than local exploit tuning.

The current lane has now produced a real optimization trajectory:

- control
  - `1.88888776`
- old `d64` anti-collapse
  - `1.86368503`
- `d32` random structural
  - `1.85942869`
- `state_book` transfer
  - `1.83989451`
- `state_book + freeze200`
  - `1.83088557`
- `state_book + freeze300`
  - `1.82215938`

That matters, but the next search phase should not be "keep tuning freeze length."

The more important question is:

- which theoretical story about the hardmax path is actually true?

## Empirical anchors we should treat as fixed

These are strong enough to guide the next tree.

1. Structural conditioning is real.
   - hardmax-related side conditioning can improve LM BPB

2. Budget routing is not the right direction.
   - the routed branch lost
   - the architecture is pointing away from discrete compute routing

3. Execution-trace supervision is real.
   - it is the first supervision family that kept the controller alive in controller-only training

4. `state_book` transfer is real.
   - transferring the discrete embedding prior helps LM BPB

5. Warm-start freezing is real.
   - preserving the transferred `state_book` early improves further

6. The hard state machine inside the LM is still not healthy.
   - the best transfer results so far look more like
     - strong transferred state vocabulary
     - plus a useful soft control surface
   - not a revived multi-state hard controller

That means the next phase should test theory, not just exploit the current local optimum.

## Root objective

Determine which hardmax theory is correct enough to justify the next architecture.

Not:

- "what is the best freeze step?"

But:

- "what role should the hardmax path play in the model?"

## Hypothesis tree

### H1: Supervision Hypothesis

The bottleneck is supervision, not architecture.

Claim:

- hardmax needs machine-state-like supervision to become useful
- shallow structural labels are too weak

Current evidence:

- supported

Next tests:

1. trace pretrain vs teacher-distill pretrain
2. trace pretrain vs self-distill / agreement-derived supervision
3. mixed multitask controller:
   - execution state
   - certainty floor

Decision rule:

- if multiple strong supervision families keep the controller alive, the bottleneck was supervision
- if only execution traces work, the theory narrows toward explicit machine state

### H2: Conditioning Hypothesis

The hardmax path should be an always-on conditioner, not a router.

Claim:

- the controller should shape the trunk's attention/state geometry
- the trunk should stay fully on

Current evidence:

- supported indirectly
  - structural conditioning helps
  - routing hurts

Immediate tests:

1. residual conditioning
2. `q_bias`
3. `q_bias_temp`
4. later:
   - repeated conditioning across layers
   - per-layer controller refresh

Decision rule:

- if always-on attention shaping beats residual-only transfer, the main architectural successor is clear

### H3: Transfer Object Hypothesis

The useful transferred object is the discrete state vocabulary, not the controller dynamics.

Claim:

- `state_book` matters more than the recurrent core right now

Current evidence:

- strongly supported
  - `state_book` transfer wins
  - `trace-core` has not shown a win

Next tests:

1. stronger `state_book` exports from the best memops trace family
2. `state_book` under new conditioning loci
3. later:
   - full controller transfer only after the new locus is validated

Decision rule:

- if `state_book` keeps winning across architectures, the current reusable object is a learned structural vocabulary

### H4: Concurrent Pathways Hypothesis

Hardmax and softmax should run concurrently, with hardmax shaping softmax attention rather than gating compute.

Claim:

- there is no discrete router
- there is an always-on fast path and an always-on slow path
- the interaction is through state/attention, not skip decisions

Current evidence:

- consistent with all strong results so far

Next tests:

1. single-layer always-on attention shaping
2. repeated conditioning across multiple layers
3. only later:
   - microsteps on the attention-shaped path

Decision rule:

- if repeated conditioning improves over single injection, the concurrent-pathway story gets stronger

### H5: Compressed Structural State Hypothesis

The hardmax path should operate on a compressed structural subspace, not compete with the full trunk dimensionality.

Claim:

- a low-dimensional controller can still carry the right structure if the supervision is right

Current evidence:

- supported
  - `d32` trace-transferred controller beats older `d64` hardmax baselines

Next tests:

1. current `d32` attention-shaping branch
2. later:
   - top-K / compressed-state refinement
   - repeated microsteps only if the new locus is positive

Decision rule:

- if low-dimensional conditioning keeps winning, the theory should favor compressed structural subspaces rather than larger hardmax width

### H6: Teacher-Certainty Hypothesis

The hardmax path can learn the softmax model's mechanical floor from the teacher's certainty surface.

Claim:

- teacher-distill supervision is a second valid path to a useful controller

Current evidence:

- not yet resolved
  - lane is wired
  - blocked mostly by cluster capacity

Next tests:

1. complete the teacher-distill smoke
2. compare with trace supervision
3. if both work, test multitask pretraining

Decision rule:

- if teacher-distill works, we gain a second supervision family and a route toward latent lock/fork conditioning

### H7: Externalized Workspace Hypothesis

The right long-run training object is not next-token over a flat sequence but next-state over a persistent typed workspace.

Claim:

- visible output should be surface only
- hidden structural / reasoning / alternative state should persist behind it

Current evidence:

- conceptual only

This is not a near-term trunk ablation.

Nearest tractable version:

1. define a minimal external workspace format for code/doc generation
2. represent:
   - surface span
   - structural state
   - confidence / lock-fork status
   - optional hidden scratch
3. train on workspace edits rather than only final surface strings

Decision rule:

- keep this as a separate research-design branch, not the next cluster sweep

### H8: Self-Generated Objective Hypothesis

The model should eventually choose its own highest-value prediction targets from its uncertainty surface.

Claim:

- the optimal supervision target is latent and model-dependent

Current evidence:

- conceptual plus indirect support from self-distillation work

Immediate research version:

1. use self-sampled agreement/divergence to define locks vs forks
2. use that signal as a controller target

This should remain downstream of H2 and H6.

## Reprioritized branch budget

The search budget should now shift from exploit tuning to theory tests.

Suggested allocation:

- 25% to H2: conditioning architecture
- 20% to H1: supervision family comparison
- 15% to H3: transfer object identity
- 15% to H4: concurrent pathway / repeated conditioning
- 10% to H6: teacher-certainty lane
- 10% to H5: compressed structural state
- 5% to H7/H8 design work

What should shrink:

- freeze-window tuning after the current sweep
- more residual-budget routing work
- more nested-only curriculum search

## Concrete next experiment matrix

This is the smallest matrix that tests theory rather than just tuning.

### Matrix A: Conditioning Hypothesis

Use the best current transfer object:

- `state_book + freeze300` residual
- `state_book + freeze300` `q_bias`
- `state_book + freeze300` `q_bias_temp`
- matched random-init `q_bias`

Question:

- does always-on attention shaping beat residual conditioning?

### Matrix B: Transfer Object Hypothesis

After memops export is available:

- old trace `state_book`
- memops `state_book`
- each under the best conditioning locus from Matrix A

Question:

- is the main object a better structural vocabulary?

### Matrix C: Supervision Family Hypothesis

When teacher-distill is runnable:

- trace-pretrained controller
- teacher-distilled controller
- if both are positive, a joint multitask controller

Question:

- what family of supervision actually creates useful structural state?

### Matrix D: Concurrent Pathways Hypothesis

Only after Matrix A:

- best single-injection attention-shaping config
- same config with two conditioning loci
- same config with four conditioning loci

Question:

- is one structural update enough, or does the theory require repeated concurrent exchange?

## Diagnostics that matter for theory

Primary:

- exact BPB / NLL
- hard state peak fraction
- state usage entropy
- soft usage perplexity
- confidence std

For H2 / H4:

- attention entropy by hardmax-confidence quantile
- attention entropy by boundary regime
- causal eval ablation:
  - zero hardmax state at eval and measure NLL delta

For H3:

- state-book cosine structure
- transfer sensitivity to `state_book` freeze vs core freeze

For H1 / H6:

- controller liveness under each supervision family
- transfer survival into LM

Secondary:

- residual ACF
- boundary-binned NLL
- confidence vs NLL correlation

## Practical search policy

Stop spending major budget on:

- "what is the best freeze length?"

after the current result is clear enough.

Start spending major budget on:

- "what is the correct role of the hardmax path?"

That means:

1. finish the current exploit readout
2. move immediately to attention-shaping
3. compare supervision families
4. only then widen into repeated conditioning or richer workspace ideas

## One-line policy

Use the current best transfer result as the launch point for theory-testing ablations, not as the start of a long local hill-climb on freeze schedules.
