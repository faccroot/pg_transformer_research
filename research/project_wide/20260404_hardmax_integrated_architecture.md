# Hardmax Integrated Architecture

This note compresses the current design direction into one architecture from input to loss.

It is not a claim that every component should be built immediately.

It is the umbrella design that the current branches are approximating.

## One correction up front

Standard LM next-token training is not MSE on logits.

It is cross-entropy over the vocabulary distribution:

- model produces logits
- logits are softmaxed into a distribution
- loss is `-log p(target_token)`

So the default supervision signal is the next observed token, passed through cross-entropy.

The core argument here is still right:

- the latent computation is much richer than the single observed next-token target
- we want more supervision bits per position than plain next-token CE provides

## Core framing

The context window is the source program.

- raw document / conversation / code
- human-readable
- not optimized for transformer execution

The KV cache is the compiled runtime state.

- the model's internal representation of the source program
- what later attention actually computes over

The weights are the compiler.

That means standard LM training mostly supervises the product of the compiled program, not the
compilation process itself.

The hardmax program is moving toward richer supervision on the compiled process.

## Four-stream forward pass

### Stream 1. Softmax trunk

This is the standard LM path.

- token embeddings
- full transformer trunk
- final logits over vocabulary
- standard next-token cross-entropy

The trunk is the expensive probabilistic path.

### Stream 2. Hardmax structural controller

Small, recurrent, and stateful.

Role:

- track low-bandwidth structural state
- condition the trunk continuously
- expose a readable structural trace

Inputs:

- compressed trunk state
- token-structural priors
- operator / polarity / boundary-like signals

Outputs:

- structural embedding
- discrete state identity
- confidence / budget / usage summaries

What exists already:

- hardmax structural controller in [logic_register_mlx.py](/home/zaytor/transformer_research/parameter-golf/logic_register_mlx.py)
- `state_book` transfer
- freeze warm-start
- attention conditioning modes in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

What is still missing:

- healthy multi-state usage inside the LM
- richer per-factor confidence instead of a mostly scalar soft surface
- real recurrent trunk feedback within one forward pass

### Stream 3. Operator / emotion tracker

This should be understood broadly as operator-state tracking, not only "emotion labels."

Target channels:

- negation / polarity scope
- uncertainty / confidence
- scope / boundary regime
- later, emotion-like latent directions

Role:

- produce a low-bandwidth face trace
- later support mirror-style self-read

What exists already:

- operator codes and polarity seeds in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
- prompt-level face trace exporter in [export_hardmax_face_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/export_hardmax_face_trace.py)

What is still missing:

- learned emotion/operator vectors beyond current polarity/operator features
- stored metadata inside a production KV path
- downstream reviewer / mirror experiments using the trace

### Stream 4. Multi-horizon prediction heads

Train-time supervision heads that predict future tokens beyond `t+1`.

Role:

- add supervision bits on the compiled process
- train discourse / planning structure in the hidden state
- provide the first practical H12 branch

What exists already:

- early-exit / multi-horizon auxiliary heads in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
- dedicated wrapper in [train_gpt_mlx_earlyexit.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_earlyexit.py)
- first H12 sweep at [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/manifest.json)

What is still missing:

- factor-decomposed horizon supervision
- rollout residual supervision
- stronger supervision-weight ladder results

## Loss stack

### Loss 1. Standard next-token CE

Exists.

This remains the primary product loss.

### Loss 2. Multi-horizon CE

Partially exists.

Current seam:

- early-exit auxiliary horizons

Current H12 branch:

- [20260404_compiled_state_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_compiled_state_supervision_leg.md)
- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/manifest.json)

### Loss 2b. Distribution-shape KL

Partially exists.

Current seam:

- EMA-teacher and external-teacher logit KL in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

Current H13 branch:

- [20260404_distribution_shape_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_distribution_shape_supervision_leg.md)

Role:

- supervise more of the already-computed vocabulary distribution
- test whether the bottleneck is target richness more than forward-pass compute

### Loss 3. Per-factor decomposed loss

Not implemented.

This is the right future H5 direction:

- decompose difficulty by factor
- train confidence and supervision against factor-specific uncertainty

Current note:

- [20260404_hardmax_hypothesis_test_queue.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_hypothesis_test_queue.md)

### Loss 4. Hardmax structural prediction loss

Exists.

This is already one of the key reasons the controller stopped totally dying.

Current seam:

- hardmax structural prediction heads in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

### Loss 5. Confidence calibration loss

Partially exists, but only in a narrow form.

Current system:

- scalar-like soft confidence surface
- useful but insufficient

Missing:

- per-factor calibration against per-factor difficulty

This is the concrete upgrade path for H5.

### Loss 6. Temporal-difference rollout loss

Not implemented.

This is the strongest "predict the compiled process, not only the product token" extension.

The intended role:

- compare earlier future predictions against realized futures
- train temporal consistency of the model's own planning process

This should be staged only after the simpler multi-horizon branch is read.

## Attention shaping

The current architectural lesson is:

- not budget routing
- yes always-on conditioning
- yes attention shaping

Current state:

- `residual`
- `q_bias`
- `q_bias_temp`

already exist in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

The likely future refinement is:

- factor-aware or head-aware confidence-driven shaping
- not global budget gating

## KV and face/mirror read

The face/mirror idea is now concrete enough to say this:

- the face trace exporter works
- the first real trace says the soft surface is alive
- the hard state is still collapsed in the `freeze200` winner

Artifact:

- [face_trace_freeze200_negation.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223900_hardmax-face-trace-smoke/face_trace_freeze200_negation.json)

Interpretation:

- the metadata path is real
- but its value is currently limited by controller richness

So we should not over-build face/mirror consumers yet.

First make the controller use more than one state.

## What this architecture implies for build order

### Stage 0. Current real anchor

What already works:

- `state_book + freeze300`
- attention-shaping branch live
- first real face trace exported
- H12 multi-horizon smoke launched

### Stage 1. Supervision-density first

Do now:

- read H12
- if directionally positive, escalate aux weight
- keep random-init separator in the matrix

Why:

- H11 already says the architecture is alive enough to expose a soft trace
- the remaining bottleneck still looks like supervision pressure

### Stage 2. Factorization

Do next:

- replace scalar confidence with vector confidence
- connect confidence to factor-specific difficulty

Why:

- this is the cleanest way to convert "alive soft surface" into a more structured control signal

### Stage 3. Better conditioning, not more routers

Do after H12/H5:

- multi-locus conditioning
- then trunk feedback / bidirectional conditioning

Why:

- only worth it once supervision is strong enough to maintain richer state

### Stage 4. Rollout residuals

Do after multi-horizon basic signal is established.

Why:

- this is a larger jump in complexity
- but it is the clearest route toward learning from compiled-process errors rather than only token targets

### Stage 5. Real face/mirror consumers

Only after the controller has richer state usage.

Why:

- a face trace with one dominant state is still diagnostically useful
- but not yet the multi-agent or mirror substrate we actually want

## Immediate decisions

1. H11 result means:
   - tooling is solved
   - controller richness is not

2. H12 result should decide:
   - whether richer supervision bits can force more useful latent structure

3. The critical separator is still:
   - transferred `state_book` vs random-init under the same richer supervision

If random-init benefits almost as much, supervision density is doing most of the work.

If transferred `state_book` benefits much more, the transferred vocabulary is real structure that
becomes useful once the model is pressured to use it.
