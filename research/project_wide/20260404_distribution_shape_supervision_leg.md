# Distribution-Shape Supervision Leg

## Core distinction

The cost bottleneck is not producing the full next-token distribution.

- the model already computes full logits over the vocabulary
- standard next-token CE supervises only the realized target token
- the remaining distribution shape is mostly computed and then discarded

That makes this a target problem more than a forward-pass cost problem.

## Hypothesis

Language models should benefit from supervision on the shape of the output
distribution, not only on the one-hot realized token.

The practical claim is:

- richer distribution-shape targets can add supervision bits without changing the export format
- this may help force cleaner latent structure before we add more bespoke internal-state heads

## Why this matters here

The hardmax lane already shows:

- transferred `state_book` structure helps
- the soft control surface is alive
- the hard discrete state still collapses under plain next-token pressure

If that collapse is partly a supervision-thinness problem, then a fuller target
distribution is the cheapest next thing to test.

## First concrete tests

### D1. EMA-teacher KL on the current best anchor

Use the existing EMA-teacher KL seam in
[train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py).

Question:

- does lagged self-distillation on the full token distribution improve BPB on top of `state_book + freeze300`?

First smoke template:

- [mlx_h13_distribution_shape_statebook_freeze300_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_h13_distribution_shape_statebook_freeze300_smoke.example.json)

Generated run:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/README.md)

Current status:

- the first H13 smoke is launched
- initial queue dispatch started with:
  - `statebook-freeze300-baseline`
  - `statebook-freeze300-ema-kl005`
- the critical separator run is `random-ema-kl010`

Important reframing:

- plain sharp KL is no longer the preferred long-term H13 form
- the next useful H13 work should be calibrated uncertainty distillation, not only "more EMA-KL at temperature 1"

Decision separator:

- matched random-init EMA-KL control

### D2. EMA-teacher temperature ladder

If EMA-KL is positive, then test whether a softer teacher provides more useful
"dark knowledge" than temperature `1.0`.

Question:

- is the gain coming from generic smoothing, or from a genuinely richer target shape?

### D3. External-teacher KL on the best recovered hardmax family

After the local teacher artifact story is less brittle, test KL against the
best recovered hardmax LM family instead of only a lagged self-teacher.

Question:

- does a stronger structured teacher distribution improve over EMA self-distill?

### D4. Uncertainty-aware distillation

New principal variation for H13:

- temperature-scaled teacher KL
- entropy-gated KL on high-uncertainty positions
- later, variance-aware or checkpoint-ensemble teacher targets

Intended read:

- preserve dark knowledge that plain sharp KL may wash out
- test whether the soft control surface benefits more from calibrated targets than from overconfident ones

## Existing engineering seam

[train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
already supports:

- `EMA_TEACHER_DISTILL_ENABLED`
- `EMA_TEACHER_DISTILL_WEIGHT`
- `EMA_TEACHER_TEMPERATURE`
- `EXTERNAL_TEACHER_*` logit-KL paths
- chunked KL from full teacher logits via `distill_kl_from_hidden`

So this leg does not need new trainer plumbing to start.

## Decision rule

Advance this branch if:

- BPB improves at matched artifact format
- random-init does not explain away the whole gain
- the result holds at a reasonable KL weight rather than only at one fragile setting

Current queue interpretation:

- keep the live EMA-KL smoke as H13a
- but promote calibrated uncertainty-aware KL to H13b as the next true follow-on if H13 remains worth pursuing
