# FineWeb Prosody Feature/State Lane

Date: 2026-04-01

## Why This Lane Exists

The current FineWeb prosody results say three things at once:

- the prosodic/boundary channel is real
- separating that channel helps `content` compression, not just formatting tokens
- a hard `content vs prosody` split is too crude, especially for punctuation

That implies the next serious architecture should do two things:

1. use factorized token features instead of a single hard prosody class
2. maintain a small persistent prosody/voice state instead of only adding static embeddings

The newer refinement from the text-prosody literature is:

3. do not force punctuation into a pure prosody bucket
4. distinguish fast clause/sentence cadence from slower paragraph/voice state

## Implemented Architecture

The base trainer now supports a stronger prosody lane behind flags in
`train_gpt_mlx.py`.

### 1. Factorized Binary Prosody Features

`text_prosody_features.py` now builds binary per-piece/token features in the
tokenizer LUT:

- `punctuation_like`
- `whitespace_like`
- `quote_like`
- `markup_like`
- `url_like`
- `emoji_like`
- `boundary_clause_plus`
- `boundary_sentence_plus`
- `boundary_paragraph_plus`
- `boundary_section`

These are intentionally multi-label. A token can now be:

- punctuation-like and boundary-like
- quote-like and punctuation-like
- markup-like and boundary-like

instead of being forced into a single prosody bucket.

There is now also an opt-in extended feature set behind
`PROSODY_EXTENDED_FEATURE_SET_ENABLED` that adds:

- `interruption_like`
- `ellipsis_like`
- `list_like`
- `heading_like`
- `code_like`
- `emphasis_like`

This is gated so older saved prosody artifacts still load with the earlier
10-feature width.

### 1b. Punctuation Role Path

The lane now treats punctuation as a mixed channel rather than a pure prosody
token.

`text_prosody_features.py` now builds punctuation-role ids such as:

- comma/clause
- colon/semicolon
- dash/interruption
- ellipsis
- terminal/question/exclamation
- bracket
- quote mark
- markup delimiter

The base trainer now supports:

- punctuation-role embeddings when `PROSODY_TYPE_EMBEDDINGS_ENABLED=1` and
  `PROSODY_EXTENDED_FEATURE_SET_ENABLED=1`
- a punctuation-only auxiliary head with
  `PROSODY_AUX_PUNCTUATION_WEIGHT`

This is the bounded implementation of the “dual-route punctuation” fix.

### 2. Exportable Prosody State Adapter

`train_gpt_mlx.py` now has a small `ProsodyStateAdapter`:

- input: token hidden state plus factorized prosody features
- state: low-dimensional recurrent latent
- update: keep gate plus candidate update
- reset pressure: token-local boundary-driven prior
- output: additive residual into the token stream

The intent is:

- persist cadence/voice state inside a regime
- partially reset at stronger boundaries
- stay cheap enough to test on the current-size MLX stack

There is now also an optional hierarchical mode:

- `PROSODY_STATE_HIERARCHICAL_ENABLED`
- `PROSODY_STATE_SLOW_RESET_SCALE`

In hierarchical mode the adapter splits into fast and slow latent state:

- fast: clause/sentence cadence, stronger reset pressure
- slow: paragraph/section/voice state, weaker reset pressure and stronger
  dependence on paragraph/section/quote/list/heading/code features

This is not a training-only head. It is an exportable inference-time component.

## New Flags

Feature embeddings:

- `PROSODY_FEATURE_EMBEDDINGS_ENABLED`
- `PROSODY_FEATURE_EMBEDDING_INIT_STD`

State adapter:

- `PROSODY_STATE_ADAPTER_ENABLED`
- `PROSODY_STATE_DIM`
- `PROSODY_STATE_INIT_STD`
- `PROSODY_STATE_SCALE`
- `PROSODY_STATE_RESET_PRIOR_WEIGHT`

The earlier lane remains available:

- `PROSODY_TYPE_EMBEDDINGS_ENABLED`
- `PROSODY_AUX_*`

New refinement flags:

- `PROSODY_EXTENDED_FEATURE_SET_ENABLED`
- `PROSODY_AUX_PUNCTUATION_WEIGHT`
- `PROSODY_STATE_HIERARCHICAL_ENABLED`
- `PROSODY_STATE_SLOW_RESET_SCALE`

### Operational/Analysis Tooling Added

To make the short ablation ladder easier to trust and compare, the lane now also
has:

- `tools/check_mlx_sweep_status.py`
  - compact live/completed status for generated queue sweeps
  - optional remote process check via `--check-remote`
- `tools/run_iteration_saved_diagnostics.py`
  - discovers locally synced artifacts for a generated sweep
  - runs the saved-artifact residual/prosody analyzer per arm
  - emits per-run JSONs, control-vs-challenger comparison JSONs, and a compact
    markdown summary

The current 1-hour prosody templates also now set:

- `WALLCLOCK_FINAL_RESERVE_SECONDS=300`

so short sweeps stop the training loop early enough to leave budget for final
export/eval instead of spilling unpredictably past the nominal wallclock cap.

## Current Ablation Logic

We now have two linked ablation ladders.

First ladder:

- `control`
- `type embeddings + prosody aux`

Second ladder:

- `control`
- `type + extended features + punctuation-aware prosody aux`
- `type + extended features + hierarchical prosody state + punctuation-aware prosody aux`

The point of the second ladder is to answer two distinct questions:

1. does factorized token metadata outperform the earlier hard class split?
2. does persistent prosody state add value beyond static features?
3. does explicit punctuation handling remove the main regression from the
   earlier hard split?

## Acceptance Criteria

For this lane to deserve promotion to longer runs, the short-run ablations should
show at least one of:

- lower overall `val_bpb`
- lower `content` bits in the saved-artifact prosody diagnostics
- lower whitespace/boundary bits without catastrophic punctuation regression
- better residual persistence around transitions once the analyzer is rerun

The state adapter specifically should be judged on whether it improves:

- punctuation calibration relative to the hard-class lane
- transition behavior relative to static feature embeddings alone

## Current Risk

The main risk is not numerical instability. The main risk is ontology error:

- if punctuation remains strongly mixed-channel, even factorized token features
  may still be too coarse
- if so, the next move would be either a richer punctuation-specific head or a
  tokenizer-aware structural-piece audit that can justify tokenizer surgery

## Related Artifacts

- `research/iterations/templates/mlx_prosody_type_aux_ab2_1h.example.json`
- `research/iterations/templates/mlx_prosody_feature_state_ab3_1h.example.json`
- `research/iterations/generated/20260401_prosody_diag_larger/results_summary.md`
- `research/iterations/generated/20260401_tokenizer_fragmentation_audit/results_summary.md`
- `tools/check_mlx_sweep_status.py`
- `tools/run_iteration_saved_diagnostics.py`
