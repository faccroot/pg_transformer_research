# Residual Autocorrelation Framework

Date: 2026-03-30

Purpose: provide a concrete saved-artifact analysis path for measuring whether a
language model's prediction residuals are temporally correlated, and whether
that correlation is concentrated within stable regimes or around regime
transitions.

## Core Idea

At each scored position, the model is not only `wrong by some amount`; it is
wrong in a direction.

This framework treats the residual as an embedding-space error vector:

- expected residual:
  - `embedding(actual_token) - E_p[embedding(token)]`
- argmax residual:
  - `embedding(actual_token) - embedding(argmax_prediction)`

Then it measures whether those residual directions are autocorrelated over
token lag.

If the residual direction remains similar over many positions, the model is
systematically misreading some slow-moving latent variable:

- style/register
- document stance
- long-range factual state
- topic or argument regime

That is the language analogue of a persistent residual in market prediction.

## Implementation

Numerical core:

- [residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/residual_autocorrelation.py)

Saved-artifact analyzer:

- [analyze_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_residual_autocorrelation.py)

Comparator:

- [compare_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/compare_residual_autocorrelation.py)

Context sweep wrapper:

- [sweep_residual_context.py](/home/zaytor/transformer_research/parameter-golf/tools/sweep_residual_context.py)

Tests:

- [test_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tests/test_residual_autocorrelation.py)

## What The Tool Produces

For a saved artifact plus training config, the analyzer emits:

- residual ACF over lag for:
  - expected-embedding residuals
  - argmax-embedding residuals
- scalar ACF for token NLL
- regime segmentation from consecutive hidden-state cosine drops
- residual ACF split into:
  - all pairs
  - within-regime pairs
  - cross-regime pairs
- transition examples with decoded text previews
- nearest token directions for the mean residual vector
  - overall
  - immediately after regime transitions
- optional transition-window summaries
  - first `N` positions after detected regime changes
- optional layerwise regime segmentation summaries
  - same residual/NLL persistence viewed through different hidden layers

The analyzer now also supports non-base trainer modules, so it can be run on:

- plain base checkpoints
- sidecar checkpoints via `--trainer-module train_gpt_mlx_jepa_sidecar_ref`
- canonical sidecar checkpoints via `--trainer-module train_gpt_mlx_sidecar_canonical`

This gives a concrete answer to:

- are residuals persistent at all?
- on what lag scale?
- mostly within regimes or at regime boundaries?
- in what semantic direction does the model stay wrong?

It now also emits a first text-native prosody surface for FineWeb:

- token-class loss decomposition
- boundary-conditioned loss summaries
- quote-conditioned loss summaries
- prosody-conditioned loss correlations
- lightweight hidden-state probes for quote state and distance-to-boundary

It now also supports optional residual-factor decomposition:

- PCA factorization of each residual family
- per-factor scalar ACF over lag
- within-regime vs cross-regime per-factor persistence
- nearest-token interpretation for each principal error direction

This is intended to answer the next mechanistic question:

- which error directions are actually persistent?
- which interventions reduce the persistence of specific factors rather than
  only the aggregate residual?

That lane is documented separately in:

- [fineweb_prosody_diagnostics_lane_20260331.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/fineweb_prosody_diagnostics_lane_20260331.md)

## Recommended Read Of The Outputs

If `acf_within_regime` is high:

- the model is missing a persistent local regime variable
- likely style, stance, topic framing, or unresolved latent belief

If `acf_cross_regime` is high:

- the model is carrying stale beliefs across transitions
- likely a slow adaptation failure after topic/style/stance changes

If residual ACF stays near zero but scalar NLL ACF is high:

- the model is repeatedly wrong in magnitude, but not in a stable semantic
  direction
- this looks more like noisy difficulty than a coherent latent miss

If the post-transition mean residual points toward a coherent token family:

- the tool has found a directionally interpretable stale prior
- e.g. formal-vs-informal, positive-vs-negative stance, one discourse marker
  family vs another

## Example Invocation

```bash
python3 tools/analyze_residual_autocorrelation.py \
  --artifact /path/to/model.int8zlib.pklz \
  --config-json /path/to/run_config.json \
  --result-json /tmp/residual_autocorr.json \
  --analysis-max-batches 32 \
  --max-lag 64 \
  --residual-mode both \
  --regime-layer -1 \
  --regime-cosine-quantile 0.05 \
  --regime-min-segment-length 16
```

Sidecar example:

```bash
python3 tools/analyze_residual_autocorrelation.py \
  --artifact /path/to/sidecar_model.npz \
  --config-json /path/to/sidecar_run_config.json \
  --trainer-module train_gpt_mlx_jepa_sidecar_ref \
  --result-json /tmp/sidecar_residual_autocorr.json
```

Comparator example:

```bash
python3 tools/compare_residual_autocorrelation.py \
  --left /tmp/control_residual_autocorr.json \
  --right /tmp/challenger_residual_autocorr.json \
  --result-json /tmp/residual_compare.json
```

Factorized example:

```bash
python3 tools/analyze_residual_autocorrelation.py \
  --artifact /path/to/model.npz \
  --config-json /path/to/run_config.json \
  --factor-top-k 4 \
  --transition-window 32 \
  --result-json /tmp/residual_factorized.json
```

Context sweep example:

```bash
python3 tools/sweep_residual_context.py \
  --artifact /path/to/model.npz \
  --config-json /path/to/run_config.json \
  --eval-seq-lens 128,256,512,1024 \
  --transition-window 32 \
  --layerwise-layers 0,2,4,-1 \
  --result-json /tmp/residual_context_sweep.json
```

## Immediate Uses

1. Compare current strong control vs sidecar vs branching artifacts.
2. See whether sidecar reduces long-lag within-regime residual ACF.
3. See whether branching reduces cross-regime stale-belief carry.
4. Identify which failure modes are actually persistent enough to justify
   architectural memory or latent-preface work.
5. Run regime-conditioned context sweeps to test whether persistent residuals
   collapse mainly with more retrieval horizon or with better latent state.
6. Test whether a training intervention is reducing aggregate error or
   specifically collapsing the persistence of a small number of dominant error
   factors.

## Why This Matters

Per-token BPB tells us `how much` error remains.

Residual autocorrelation tells us whether the remaining error is:

- independent noise
- or a structured, persistent miss of a latent state

That distinction is exactly what decides whether the next win should come from:

- more local capacity
- better adaptive compute
- better regime/state tracking
- or a document-level latent preface

## Training Follow-On

The first cheap training intervention built on top of this framework is
residual-novelty weighting in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py).

Instead of treating every repeated error equally, it derives an additional
token-weight multiplier from lag-1 residual novelty:

- repeated error direction -> lower weight
- novel error direction -> higher weight

This is intentionally cheap and bounded:

- no new model parameters
- no new supervision source
- reuses the existing token-weight merge path

The purpose is to test whether the model learns faster when it is told, in
effect, which errors are redundant repetitions and which errors are new
information about a different failure mode.
