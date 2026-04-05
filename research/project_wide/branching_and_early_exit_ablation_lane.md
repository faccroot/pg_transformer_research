# Branching And Early-Exit Ablation Lane

This note captures the current read on causal latent-state training, mid-stack speculative heads, and structural branching.

The shortest correct summary is:

- tokens are evidence
- the sidecar is the causal latent interpretation of the revealed prefix
- branches are explicit competing hypotheses about what world the prefix implies next

That split matters because it gives a clean causal boundary:

- `training`: branching is a richer pretraining objective built on known future tokens
- `eval`: branching is only valid as score-first retrospective adaptation on already-scored chunks

## Core interpretation

The useful framing is not "the context window stores the data." It is "the context window should approximate an address into the model's generative manifold."

In a causally valid system:

- literal tokens remain the observed evidence
- a fixed-size latent state tracks the model's belief about the local generative process
- branch rollouts represent alternative futures implied by unresolved uncertainty

This is exactly why the sidecar remains important even after the correction on eval leakage:

- the sidecar is the valid online manifold-address estimator
- it only updates from the revealed prefix
- it compresses long history into a fixed-size causal belief state

What is *not* valid for leaderboard scoring is optimizing a synthetic prefix on the same tokens being scored. That remains a research idea for training or for score-first future-chunk adaptation, not for same-token eval.

## Training vs Eval

### Training

During pretraining, future tokens are already in the batch. Branching is therefore a supervision trick, not a leakage issue.

The right interpretation is:

- teacher forcing tells the model which token was right
- branching tells the model which world it should have believed

The branch auxiliary is strongest when it converts one real continuation into:

- one positive path
- one or more model-generated counterfactual paths

The critical mechanism is forced commitment. A weak wrong generation usually collapses into noise after a few steps. A forced committed branch instead produces a coherent wrong world, which is much more informative as a negative.

### Eval

At eval, the same mechanism is only valid in score-first form:

1. score chunk `n`
2. branch retrospectively on the chunk's revealed structural misses
3. optionally adapt from those misses
4. only let that help chunk `n+1` onward

So:

- `training`: branching is a pretraining objective
- `eval`: branching is a retrospective adaptation objective

They should not be conflated.

## Minimum viable branch objective

The first safe version should stay deliberately small.

Keep:

- a nonzero CE anchor for the full run
- burn-in before any branch auxiliary turns on
- depth-1 branching only
- at most `B=1..3` branch points per sequence
- `K=4..8` rollout tokens
- cheap draft generation from a mid-stack head where possible

Do *not* start with:

- recursive branch trees
- pure entropy triggers
- zeroing out CE late in training
- eval-time synthetic context optimization on the same scored chunk

The practical branch trigger should be selective. The useful template is:

`branch_score_t = realized_loss_t * top2_ambiguity_t * semantic_distance_t * future_divergence_t`

Where:

- `realized_loss`: this position actually hurt under teacher forcing
- `top2_ambiguity`: top-1 and top-2 are close
- `semantic_distance`: the competing tokens are meaningfully different
- `future_divergence`: cheap rollout says they lead to different continuations

This prevents branching on cheap misses like `the` vs `a`, and spends compute on real structural forks like `approve` vs `reject` or `however` vs `therefore`.

## Loss shape

The clean target objective is:

`L = L_CE + λ_mtp L_MTP + λ_rank L_branch_rank + λ_state L_branch_state + λ_sem L_semantic_weight`

Where:

- `L_CE`: standard next-token CE anchor
- `L_MTP`: mid-stack multi-horizon auxiliary, e.g. token `+1/+2/+3`
- `L_branch_rank`: real continuation should outrank the wrong committed branch
- `L_branch_state`: right and wrong branches should occupy different hidden-state trajectories
- `L_semantic_weight`: structurally large misses matter more than synonym-ish misses

This is the clean composition target. It is broader than the current prototype, but it is the correct ladder.

## Promotion path

The safest implementation ladder is:

1. `CE anchor + MTP only`
   - validate that a cheap mid-stack head learns useful forward structure
   - this is the cheapest entry point and doubles as the future branch drafter
2. `CE anchor + bounded branch rank`
   - depth-1 only
   - top `B` structural misses only
   - no recursive trees
3. `CE + MTP + bounded branch rank`
   - use the mid-stack heads to cheapen rollout cost
   - only promote to full-stack rollout when the cheap path stays ambiguous
4. `Add hidden-state divergence`
   - only after rank loss alone proves stable
   - this is the cleanest "wrong world vs right world" representation-sharpening term
5. `Eval-side score-first TTT`
   - only after the training objective is positive
   - use already-scored chunks only
6. `Manifold-address cache`
   - only after the causal sidecar / branch stack proves value
   - treat as a future evaluator optimization, not the current near-term lane

## Current live experiments

Two active runs already map to the first two steps of that ladder.

### 1. Bounded structural branching prototype

Run: [20260330_030543_mlx-structural-branching-ab2-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_030543_mlx-structural-branching-ab2-1h/README.md)

What it tests:

- detached teacher-forced prepass to find branch points
- structurally large misses only
- narrow-gap branch candidates only
- one committed wrong-path rollout
- real continuation should outrank the wrong path

What it does *not* test yet:

- explicit hidden-state divergence regularization
- multi-branch trees
- cheap mid-stack drafting inside the branch walker

This is the correct first branch ablation.

### 2. Mid-stack multi-horizon early-exit auxiliary

Run: [20260330_031324_mlx-early-exit-aux-ab2-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_031324_mlx-early-exit-aux-ab2-1h/README.md)

What it tests:

- a captured mid-stack hidden state
- training-only heads for horizons `1,2,3`
- stripped export so the artifact remains fair

Why it matters:

- if positive, this becomes the obvious cheap branch drafter
- if negative, the branch lane should probably keep using the full stack and stay very selective

## Current implementation status

The codebase is now slightly ahead of the original note.

What is now implemented in the base trainer path:

- cosine-gated branch selection
- adaptive branch depth from inter-branch hidden divergence
- curriculum-controlled dynamic branch budgeting
- opt-in early-exit heads in the base trainer
- export-time stripping of early-exit heads
- early-exit next-token drafting inside committed branch rollouts
- bounded multi-token draft acceptance from existing early-exit horizon heads
- first teacher-forced conditioned/cascaded early-exit draft chain
- curriculum-controlled dynamic early-exit auxiliary weighting
- hidden-state divergence as an explicit optional branch loss term

What is now implemented in the separate eval surface:

- saved-artifact structural analysis in [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py)
- first score-first branch adaptation loop in [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py)
- bounded two-pass branch adaptation plus exact re-score in [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py)

What is still *not* implemented:

- Medusa-style or deeper speculative draft trees
- recursive branch trees
- latent document preface or manifold-address cache

So the current code status is:

- the `CE + early-exit + bounded branching + dynamic budget` path now exists
- the next question is no longer implementation feasibility
- the next question is whether the fully composed bounded version wins cleanly in A/Bs

## Decision rules

If `early-exit aux` is positive:

- promote it as the cheap-draft component for branching
- next run should be `CE + MTP + bounded branching`

If `branching` is positive:

- keep the CE anchor
- add hidden-state divergence only as the next single change

If both are positive:

- combine them in a clean 1-hour composition run before any 4-hour promotion

If both are negative:

- keep the sidecar as the main causal latent-state lane
- treat full branching as a research branch, not a challenger promotion path

## What Should Stay Out For Now

Do not promote these into the current challenger stack without a clean win:

- recursive branching trees
- raw-entropy branch triggers
- same-token eval-time synthetic context optimization
- turning off CE and relying on branches alone
- assuming "thinking at eval" is the main source of value

The strongest near-term claim is narrower:

- the main value of branching is pretraining signal
- eval-side branching is secondary and should be score-first only
- the sidecar remains the cleanest currently valid causal latent-state mechanism

## Practical summary for other agents

If another agent needs the short operational instruction, use this:

1. Keep the current strong control as the anchor.
2. Treat early-exit multi-horizon heads as the cheapest first branch-enabler.
3. Treat structural branching as a pretraining auxiliary, not an eval hack.
4. Keep branching depth-1 and selective.
5. Never remove the CE anchor.
6. Only move to eval-side retrospective branching after the training lane wins.
