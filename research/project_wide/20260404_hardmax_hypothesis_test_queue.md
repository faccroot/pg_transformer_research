# Hardmax Hypothesis Test Queue (2026-04-04)

This note converts the current hypothesis roundup into an engineering queue.

The point is not to keep hill-climbing the current best transfer result.

The point is to use the current best result:

- `state_book + freeze300 = 1.82215938`

as the launch point for theory-testing ablations.

Integrated umbrella note:

- [20260404_hardmax_integrated_architecture.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_integrated_architecture.md)

## Current anchor facts

These are the empirical anchors that the queue should assume:

1. Structural conditioning is real.
2. Budget routing is the wrong branch.
3. Execution-trace supervision is the first supervision family that kept the controller alive.
4. `state_book` transfer is stronger than core transfer.
5. Early freezing improves transfer.
6. The hard controller inside the LM is still mostly collapsed.

That means the queue should prefer:

- theory tests about supervision, conditioning, and transfer objects

over:

- more local tuning of the same residual path.

## Shared Instrumentation

Two instrumentation seams now exist for future winners:

- saved-artifact causal ablations:
  - [eval_saved_hardmax_causal_ablation.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_hardmax_causal_ablation.py)
  - supports `baseline`, `zero_hardmax`, `zero_residual`, `zero_q_bias`, `zero_tau`
- face/mirror trace export:
  - [export_hardmax_face_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/export_hardmax_face_trace.py)
- Mini-first saved-artifact wrapper:
  - [run_remote_saved_hardmax_analysis.py](/home/zaytor/transformer_research/parameter-golf/tools/run_remote_saved_hardmax_analysis.py)
  - supports `controller`, `causal`, `factors`, and `face`
- iteration-level hardmax diagnostics wrapper:
  - [run_iteration_hardmax_transfer_diagnostics.py](/home/zaytor/transformer_research/parameter-golf/tools/run_iteration_hardmax_transfer_diagnostics.py)
  - now supports:
    - `--skip-residual`
    - `--remote-analyzers`
    - `--include-causal-ablation`
    - `--include-logit-factors`

The causal evaluator uses a model-scoped hardmax ablation context added in
[train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py), so H2/H12/H13 checkpoints can be scored for mechanism deltas without changing training configs.

Practical note:

- local Ubuntu CPU-MLX import is now available, but actual MLX CPU execution here is still brittle because of the host JIT/compiler path
- so Mini-first remote execution is the preferred path for saved-artifact controller / causal / factor analysis

## Hypothesis status matrix

### H1. Softmax logit distillation to hardmax

Claim:

- the controller can learn the trunk's own certainty surface directly
- this may transfer better than execution traces because there is no domain mismatch

Current status:

- partially built already via the teacher-distill lane
- current implementation is:
  - [train_hardmax_teacher_distill.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_teacher_distill.py)
  - [teacher_distill_hardmax_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/teacher_distill_hardmax_lane_20260403.md)
- but it is still framed as an external teacher-draft lane, not yet the exact "winning trunk distills itself" version

Engineering delta:

1. point the teacher-distill lane at the current best local LM checkpoint family
2. add richer teacher targets:
   - top-1 token
   - top-1 probability
   - entropy or top-1 / top-2 margin
3. optionally log top-k later

Minimal test:

- teacher-distill `state_book` transfer vs trace-pretrained `state_book` transfer

Priority:

- very high

### H2. Attention temperature modulation instead of budget routing

Claim:

- the controller should modulate attention shape, not compute budget

Current status:

- partially implemented now in trunk
- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py) supports:
  - `HARDMAX_STRUCT_CONDITION_MODE=residual`
  - `HARDMAX_STRUCT_CONDITION_MODE=q_bias`
  - `HARDMAX_STRUCT_CONDITION_MODE=q_bias_temp`
- branch note:
  - [20260404_hardmax_attention_shaping_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_attention_shaping_leg.md)

Important nuance:

- the current `q_bias_temp` path derives temperature from structural state, not from the scalar confidence directly
- so this is already a good test of the broad theory
- but not yet a pure "confidence -> tau" ablation

Minimal tests:

1. `freeze300` residual baseline
2. `freeze300` `q_bias`
3. `freeze300` `q_bias_temp`
4. matched random-init `q_bias`

Staged canonical run:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211052_mlx-hardmax-trace-transfer-attn-conditioning-smoke/manifest.json)

Priority:

- highest

### H3. Multiple hardmax refinement cycles per trunk layer

Claim:

- multiple fast controller updates per conditioning site may produce a more useful structural state

Current status:

- infra already exists:
  - `HARDMAX_STRUCT_FAST_REFINEMENT_STEPS`
- branch note:
  - [20260403_hardmax_async_refinement_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260403_hardmax_async_refinement_leg.md)
- but existing tests were mostly on weaker transfer slices like `trace-core`, not the winning `state_book + freeze` line

Engineering delta:

1. retarget the microstep matrix to the winning transfer object
2. preferably test it under the best attention-shaping locus, not only residual mode

Minimal test:

- best conditioning locus with microsteps `1, 2, 4`

Priority:

- high, but after H2

### H4. Bidirectional conditioning

Claim:

- the controller should read trunk state mid-forward-pass and revise its own structural state

Current status:

- not implemented

Engineering delta:

1. add controller refresh points at later layers
2. allow controller update from a stopgrad trunk readout
3. feed revised structural state forward to the next conditioning locus

Minimal test:

- single injection vs two-locus refresh with stopgrad trunk feedback

Priority:

- medium-high

### H5. Per-factor confidence decomposition

Claim:

- scalar confidence is too blunt
- the controller should expose multiple uncertainty channels

Current status:

- factor-extraction scaffold now exists:
  - [analyze_saved_logit_factors.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_saved_logit_factors.py)
- current extractor supports:
  - PCA over raw logits
  - PCA over probability residuals (`softmax(logits) - one_hot(target)`)
  - optional hardmax eval ablations via the saved-artifact causal seam
  - factor summaries with:
    - explained variance
    - token-loading tables
    - correlation with NLL and entropy
    - simple activation-binned difficulty / top-1 accuracy

Engineering delta:

1. define factor axes
   - simplest first version: PCA of output-logit variance or disagreement directions
2. replace scalar confidence head with vector confidence head
3. add factor-specific diagnostics

Minimal test:

- scalar confidence vs 4D confidence vector on:
  - calibration
  - correlation with factor-specific difficulty
  - downstream transfer utility

Priority:

- medium

### H6. Emotion vectors as hardmax training signal

Claim:

- low-dimensional behavioral-affect state could be a useful hardmax target

Current status:

- conceptual only

Priority:

- low for now

### H7. Certainty-ordered generation

Claim:

- generation should fill easy structural positions before uncertain content positions

Current status:

- conceptual only

Nearest tractable version:

- iterative refinement over low-confidence spans rather than a full new decoding paradigm

Priority:

- low for now

### H8. Apple self-distillation with hardmax verification

Claim:

- self-generated data improves quality
- hardmax consistency can act as the filter/verifier

Current status:

- not implemented
- but it connects directly to the current trace-verifier line

Engineering delta:

1. generate coding samples
2. score them by hardmax consistency
3. split:
   - random samples
   - hardmax-consistent samples
   - hardmax-inconsistent samples
4. fine-tune on each split

Minimal test:

- does hardmax-consistent self-distill beat unfiltered self-distill?

Priority:

- medium-high

### H9. Cross-model state_book transfer

Claim:

- the learned computational states may be portable across models

Current status:

- not implemented

Dependencies:

- need a stable reusable `state_book` object first
- ideally after H2 so the conditioning locus is not confounded

Priority:

- medium-low

### H10. Hardmax as KV cache compactor

Claim:

- controller state can predict stale vs active memory

Current status:

- not implemented

Priority:

- low until the controller is more alive inside the LM

### H11. Face / Mirror operator-state trace

Claim:

- the model should not only emit surface text
- it should also emit a low-bandwidth trace of its own operator/state configuration
- that trace should be readable both by itself later (`mirror`) and by other agents (`face`)

This is not just "emotion conditioning."

It is a more general operator-state hypothesis:

- negation scope
- polarity flips
- confidence / uncertainty
- active structural scopes
- later, emotion-like valence / arousal channels

Two distinct tasks should be separated:

1. forward self-read
   - the model reads its own stored state trace later
   - privileged self-access

2. backward other-read
   - another model or agent infers state from text alone
   - theory-of-mind style reconstruction

Current status:

- conceptual only
- but strongly supported by the current interpretability framing
- first exporter seam now exists:
  - [export_hardmax_face_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/export_hardmax_face_trace.py)
  - [20260404_hardmax_face_mirror_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_face_mirror_leg.md)
  - [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223900_hardmax-face-trace-smoke/README.md)
- first real structural export now exists via the repo-bundle workaround
- current read is the same as the LM diagnostics:
  - soft confidence/budget surface is alive
  - hard state remains collapsed (`used_states=1` in the first `freeze200` prompt trace)

Nearest tractable test:

1. export a minimal per-token face trace during generation:
   - controller confidence
   - state index / soft usage
   - polarity / negation scope features
2. run a review agent on:
   - text only
   - text + face trace
3. measure whether the trace improves:
   - uncertainty localization
   - bug / inconsistency finding
   - revision quality

Priority:

- medium
- after H2, before very long-horizon generation-format changes

### H12. Compiled-state supervision density

Claim:

- next-token targets under-supervise the latent computation
- the context window is source code, but the model actually computes over compiled internal state
- useful extra supervision should target the process and compiled state, not only the product token

This branch treats:

- multi-horizon token prediction
- rollout residuals
- internal state summaries
- KV relevance / memory utility

as extra supervision channels.

Branch note:

- [20260404_compiled_state_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_compiled_state_supervision_leg.md)

Nearest tractable tests:

1. train-only multi-horizon auxiliary heads
2. short rollout residual targets
3. KV relevance prediction heads

Priority:

- medium-high
- after H2/H1, before long-horizon generation-format changes

### H13. Distribution-shape supervision

Claim:

- next-token CE supervises only one realized target token
- the model already computes a much richer output distribution
- richer distribution targets should add supervision bits without major architecture change

This branch treats:

- EMA-teacher KL self-distill
- later temperature ladders
- later external-teacher KL

as target-rich supervision on the same artifact format.

Branch note:

- [20260404_distribution_shape_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_distribution_shape_supervision_leg.md)

Nearest tractable tests:

1. EMA-teacher KL on the `state_book + freeze300` anchor
2. matched random-init EMA-KL separator
3. KL weight ladder before pruning the branch

Priority:

- medium-high
- adjacent to H12, because both test supervision density with minimal new machinery

Current status:

- first H13 EMA-KL smoke is staged and launched:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/manifest.json)
  - [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/README.md)
- the key separator run is `random-ema-kl010`

## Queue order

### Queue A: Immediate

These are the next theory tests to engineer and queue.

1. H2 canonical attention-shaping sweep
   - already staged
   - use the `freeze300` anchor

2. H1 self-distill / trunk-logit supervision refresh
   - adapt the teacher-distill lane to the current local best LM family

3. H3 microsteps rerun on the winning transfer object
   - not on `trace-core`
   - preferably on the best H2 conditioning locus

### Queue B: Next

4. H4 bidirectional conditioning
5. H8 self-distillation + hardmax verification
6. H5 per-factor confidence
7. H12 compiled-state supervision density
8. H13 distribution-shape supervision

### Queue C: Later

9. H9 cross-model `state_book` transfer
10. H10 KV cache compaction
11. H11 face / mirror operator-state trace
12. H6 emotion vectors
13. H7 certainty-ordered generation

## Concrete engineering tasks

### Task 1: Launch and read H2

Use:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211052_mlx-hardmax-trace-transfer-attn-conditioning-smoke/manifest.json)

Question:

- does always-on attention shaping beat residual conditioning?

### Task 2: Convert teacher-distill into trunk-certainty distill

Use:

- [train_hardmax_teacher_distill.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_teacher_distill.py)

Add:

- clean teacher selection from the current best local LM checkpoints
- top-1 probability / entropy targets if not already present in the selected run

Question:

- can the controller learn the trunk's own certainty floor?

### Task 3: Retarget microsteps

Do not spend more budget on `trace-core` microsteps.

Retarget to:

- winning transfer slice
- best conditioning locus from H2

Question:

- does controller refinement help once the conditioning locus is correct?

### Task 4: Stage compiled-state supervision tests

Use:

- [20260404_compiled_state_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_compiled_state_supervision_leg.md)

Start with:

- train-only multi-horizon token heads
- then short rollout residual targets

Question:

- can richer supervision on the compiled process outperform plain next-token supervision at matched export format?

Current status:

- first H12 multi-horizon smoke is staged and launched:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/manifest.json)
  - [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/README.md)
- pre-staged follow-on aux-weight ladder:
  - [mlx_h12_multihorizon_auxweight_ladder.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_h12_multihorizon_auxweight_ladder.example.json)

Interpretation to watch:

- `random-h123-aux005` is the key separator in the live H12 smoke
- if random-init gains nearly as much as transferred `state_book`, then supervision density is doing most of the work
- if transferred `state_book` gains substantially more, then the vocabulary transfer is real structure that richer supervision can exploit
- if `0.05` is only weakly positive, do not prune H12 yet; escalate the auxiliary weight ladder to `0.1` and `0.2`

### Task 5: Stage distribution-shape supervision tests

Use:

- [20260404_distribution_shape_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_distribution_shape_supervision_leg.md)

Start with:

- EMA-teacher KL on the current `state_book + freeze300` anchor
- matched random-init separator
- KL weight ladder before pruning

Question:

- does richer target-distribution supervision help more than one-hot next-token CE at matched architecture and export format?

## Decision rules

### If H2 wins

- expand repeated conditioning across multiple layers
- then test H3 on top of the H2 winner

### If H2 is flat but H1 wins

- supervision is more important than conditioning locus
- push harder on self/trunk distillation

### If H2 and H1 both work

- build a multi-task controller:
  - trace state
  - certainty floor

### If H2 fails and H1 fails

- revisit the controller parameterization before expanding broader theory branches

## One-line policy

Use the current best transfer result as a stable anchor, and spend the next branch budget on theory tests that distinguish supervision role, conditioning role, and transfer object identity.
