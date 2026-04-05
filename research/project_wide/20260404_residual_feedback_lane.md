# Residual Feedback Lane

Date: 2026-04-04

Purpose: turn residual autocorrelation from a pure diagnostic into a bounded
training signal.

Current branch state:

- H15b `SimVQ + NextLat` on the `freeze300` anchor is now a negative result
- this lane is therefore promoted from supporting diagnostic work to the next
  bounded hardmax intervention

## Thesis

The model does not only make large errors. It often makes the same error
repeatedly across adjacent positions and sometimes across long stable spans.

Standard next-token loss treats:

- error at `t`
- error at `t+1`
- error at `t+2`

as independent supervision events.

Residual autocorrelation says they are often one event spread over many
positions:

- the model is missing a slow latent variable
- the residual direction is the shadow of that missing latent
- the persistence length tells us how slow the missing variable is

That gives two concrete opportunities:

1. mechanistic validation
   - which error directions are persistent?
   - which architectural changes reduce that persistence?
2. cheap training intervention
   - bias learning toward novel error directions rather than redundant repeats

## Current Instrumentation

Numerical core:

- [residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/residual_autocorrelation.py)
- [residual_feedback.py](/home/zaytor/transformer_research/parameter-golf/residual_feedback.py)

Saved-artifact tooling:

- [analyze_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_residual_autocorrelation.py)
- [compare_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/compare_residual_autocorrelation.py)

Trainer path:

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

Focused tests:

- [test_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tests/test_residual_autocorrelation.py)
- [test_residual_feedback.py](/home/zaytor/transformer_research/parameter-golf/tests/test_residual_feedback.py)

## What Is Implemented

### 1. Factorized residual ACF

The analyzer can now PCA-factorize residual vectors and report per-factor:

- explained variance
- nearest-token direction
- scalar ACF over factor scores
- within-regime vs cross-regime persistence
- optional transition-window score summaries

This is the direct mechanism test for the hardmax lane:

- if trace/state-book transfer reduces persistence on a narrow subset of
  structural factors, it is acting like structural state support
- if it reduces persistence uniformly, it is acting more like a general adapter

### 2. Residual-novelty token weighting

`train_gpt_mlx.py` now supports a cheap loss-level intervention:

- compute argmax residual direction from predicted token vs actual token
- compare each residual to the previous residual with cosine similarity
- downweight repeated residual directions
- upweight novel residual directions

The weighting is applied through the existing token-weight merge path rather
than a second objective stack.

Env knobs:

- `RESIDUAL_NOVELTY_WEIGHTING_ENABLED`
- `RESIDUAL_NOVELTY_MIN_SCALE`
- `RESIDUAL_NOVELTY_MAX_SCALE`
- `RESIDUAL_NOVELTY_NORM_EPSILON`

Logged metrics:

- `residnov_sim`
- `residnov_mean`
- `residnov_w`
- `residnov_valid`

Current caveat:

- MLX compile is disabled when residual-novelty weighting is active

## Immediate Decision Surface

Current live bundle:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_164756_mlx-hardmax-residual-novelty-freeze300-smoke/manifest.json)
- first claimed hosts:
  - `freeze300-baseline` on `mini04`
  - `freeze300-residnov-090-110` on `mini06`

### E1. Mechanistic read on recovered hardmax artifacts

Run factorized residual ACF on:

- matched `d32 random`
- trace `state_book`
- `freeze100`
- `freeze200`

Answer:

- which factors lose persistence?
- is the reduction mostly within-regime or cross-regime?
- do the improved BPB arms reduce structural persistence specifically?

### E2. Cheap training ablation

Use the current hardmax anchor and test whether novelty-weighted loss helps the
same lane that already benefits from transferred structural priors.

The first bounded sweep is:

- baseline
- light novelty weighting
- standard novelty weighting
- strong novelty weighting
- novelty weighting crossed with the best anti-collapse objective if the latter
  reads positive

### E3. Next bounded extension

If novelty weighting looks useful, the next extension is a residual mirror:

- maintain a tiny running summary of recent residual directions
- inject that summary as a small conditioning object
- test whether residual ACF drops further

That should stay behind the cheap loss-level ablation until the surface is
clear.

## Keep / Kill Rules

Keep if either:

- BPB improves by at least `0.003`
- BPB is flat within `0.002` and residual ACF on leading factors drops
  materially

Kill if:

- BPB regresses
- novelty weighting changes token weights but leaves factor persistence flat
- the apparent gain is just uniform scaling with no factor-selective effect

## Why This Matters

This lane is a direct test of a broader hypothesis:

the model's own residual stream is already a compressed memory of what it is
missing.

If that is true, then residual persistence is not just a diagnostic. It is a
supervision channel.
