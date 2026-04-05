# Residual Feedback Lane

Date: 2026-04-04

Purpose: turn the residual-autocorrelation work into both:

- a stronger mechanistic diagnostic
- a cheap trainable intervention

## What Landed

Residual factorization now exists in the analyzer stack:

- [residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/residual_autocorrelation.py)
- [analyze_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_residual_autocorrelation.py)

The analyzer can now:

- PCA-factorize each residual family
- report per-factor ACF
- split that ACF into `all / within_regime / cross_regime`
- attach nearest-token interpretations to each factor direction

This is the first concrete path for testing:

- which error factors are persistent?
- which interventions reduce persistence on specific factors instead of only
  the aggregate residual?

The first cheap training intervention also landed in:

- [residual_feedback.py](/home/zaytor/transformer_research/parameter-golf/residual_feedback.py)
- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

This is `residual novelty weighting`:

- repeated residual direction -> lower token weight
- novel residual direction -> higher token weight

It is intentionally cheap:

- no new model parameters
- no new external labels
- reuses the existing token-weight merge path

New env surface:

- `RESIDUAL_NOVELTY_WEIGHTING_ENABLED`
- `RESIDUAL_NOVELTY_MIN_SCALE`
- `RESIDUAL_NOVELTY_MAX_SCALE`
- `RESIDUAL_NOVELTY_NORM_EPSILON`
- `RESIDUAL_NOVELTY_EMA_DECAY`

Training logs now surface:

- `residnov_sim`
- `residnov_mean`
- `residnov_w`
- `residnov_valid`

The next bounded prototype also landed in the main trainer:

- a small `ResidualErrorPrior` module that predicts the model's own residual
  direction from final hidden state
- training-only auxiliary terms for:
  - residual prediction loss
  - residual prediction MSE
  - residual prediction cosine alignment

New env surface:

- `RESIDUAL_ERROR_PRIOR_ENABLED`
- `RESIDUAL_ERROR_PRIOR_WEIGHT`
- `RESIDUAL_ERROR_PRIOR_BOTTLENECK_DIM`
- `RESIDUAL_ERROR_PRIOR_INIT_STD`
- `RESIDUAL_ERROR_PRIOR_COSINE_WEIGHT`
- `RESIDUAL_ERROR_PRIOR_NORM_EPSILON`
- `RESIDUAL_ERROR_PRIOR_TARGET_MODE`

Training logs now also surface:

- `train_errprior`
- `train_errprior_mse`
- `errprior_cos`

## Why This Is The Right First Step

The stronger mirror-state idea is interesting, but the first bounded test should
not add memory, KV surfaces, or new trainable modules yet.

The immediate questions are:

1. does persistent residual structure cluster into a few dominant factors?
2. can the optimizer use a novelty-weighted error stream better than a
   repetition-weighted one?
3. can a tiny global error-prior head learn a useful map from
   `context -> likely failure direction`?

If the answer to either is yes, that justifies the next stage:

- local error-memory entries
- global error-prior prediction
- or hardmax-conditioned error-state tracking

If the answer is no, the mirror-state idea is less likely to be the right next
mainline investment.

## Immediate Test Queue

1. Run factorized residual ACF on the recovered hardmax freeze family.
   Goal: see whether `state_book` transfer reduces persistence on a small set of
   structural error factors.

2. Run a clean base A/B with residual novelty weighting using:
   - [mlx_residual_novelty_weighting_ab2_1h.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_residual_novelty_weighting_ab2_1h.example.json)

3. Run a clean base A/B with the global error-prior head using:
   - [mlx_residual_error_prior_ab2_1h.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_residual_error_prior_ab2_1h.example.json)

4. Only if one of the cheap residual-feedback lanes helps:
   - try a bounded learned error prior
   - then a local mirror-state / extra-KV prototype

## Current Constraint

The exact `freeze300` hardmax anchor is not yet synced locally as a saved
artifact, so the first mechanistic factor run is being targeted at the best
recovered predecessor (`freeze200`) while waiting for fuller artifact recovery.

Local Linux CPU MLX artifact analysis also needed a toolchain workaround:

- the saved-artifact analyzers now seed permissive Linux C++ compiler settings
  before importing MLX
- the first residual-novelty queue attempt was operationally blocked by cluster
  availability, not by the residual code path itself

## First Mechanistic Read

The first matched remote comparison is now in:

- [results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_residual_factor_hardmax_freeze200/results_summary.md)

Short read:

- `freeze200` improved mean NLL vs random init
- scalar NLL ACF did not improve
- expected embedding residual persistence improved slightly
- argmax embedding residual persistence improved materially
- the strongest improvements concentrate in a small number of dominant factors,
  especially factor `0` and `1/2` depending on residual family

So the mechanistic picture is now sharper:

- hardmax trace/statebook transfer is reducing persistent **directional** error
- it is not yet solving transition-sticky scalar surprise
- that is consistent with a structural controller that helps committed decisions
  more than full calibration

## Operational Progress Since Landing

The saved-artifact residual analyzer and the remote hardmax-analysis launcher now
support staged repo bundles cleanly, so remote Mini analysis no longer depends on
the local repo layout:

- [analyze_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_residual_autocorrelation.py)
- [run_remote_saved_hardmax_analysis.py](/home/zaytor/transformer_research/parameter-golf/tools/run_remote_saved_hardmax_analysis.py)

This mattered in practice: the first remote residual-factor run initially failed
because the analyzer could not import bundled repo modules. That path is now fixed.

The one-host execution gap is also now closed with:

- [run_iteration_serial_host.py](/home/zaytor/transformer_research/parameter-golf/tools/run_iteration_serial_host.py)

Purpose:

- run a generated iteration serially on one Mini
- append durable `dispatch.out` lines
- recover cluster artifacts after each run
- refresh `observed_results.json` automatically

This is the correct fallback when cluster availability is too thin for the
default `queue_parallelism=2` launchers.

Residual novelty weighting also has a bounded V2 now:

- with `RESIDUAL_NOVELTY_EMA_DECAY=0`, weighting compares each residual
  direction to the immediately previous one
- with `RESIDUAL_NOVELTY_EMA_DECAY>0`, weighting compares to an EMA history of
  prior residual directions instead

That makes the cheap intervention match the real hypothesis better:

- not just “was I wrong like the last token?”
- but “am I still making the same mistake over a longer stretch?”

## Current Live Test Queue

Residual novelty weighting:

- generated sweep: [20260405_022150_mlx-residual-novelty-weighting-ab2-1h](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_022150_mlx-residual-novelty-weighting-ab2-1h)
- control arm is live on `mini09` via manual same-host dispatch
- a local follow-up chain is now active and will:
  - wait for the manual control process on `mini09`
  - launch the novelty challenger on the same host
  - then reuse the same host for the staged error-prior A/B
  - [20260405_residual_feedback_followup_chain.sh](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_residual_feedback_followup_chain.sh)

Residual error prior:

- generated spec: [20260405_024700_mlx-residual-error-prior-ab2-1h.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_024700_mlx-residual-error-prior-ab2-1h.json)
- prepared bundle: [20260405_024449_mlx-residual-error-prior-ab2-1h](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_024449_mlx-residual-error-prior-ab2-1h)

So the lane is now in the right operational shape:

- mechanistic factor read: done
- cheap novelty-weighting A/B: running on a clean same-host path
- global error-prior A/B: staged and ready next
