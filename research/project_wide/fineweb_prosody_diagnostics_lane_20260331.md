# FineWeb Prosody Diagnostics Lane

Date: 2026-03-31

Purpose: measure whether FineWeb text contains a usable prosodic / boundary /
voice channel, and whether current language-model checkpoints already represent
it implicitly.

This is the first implementation step for the text-side version of:

- pause / silence
- sentence and paragraph rhythm
- quote / speaker state
- content-vs-prosody channel separation

## What Is Implemented

Reusable feature extraction:

- [text_prosody_features.py](/home/zaytor/transformer_research/parameter-golf/text_prosody_features.py)

Focused tests:

- [test_text_prosody_features.py](/home/zaytor/transformer_research/parameter-golf/tests/test_text_prosody_features.py)

Saved-artifact analyzer, extended:

- [analyze_residual_autocorrelation.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_residual_autocorrelation.py)

The analyzer now emits:

- token-class loss decomposition
  - `content`
  - `punctuation`
  - `whitespace`
  - `markup`
  - `quote`
- boundary-conditioned loss summaries
  - exact previous-boundary class
  - cumulative `after_sentence` / `after_paragraph` / `after_section`
- quote-conditioned loss summaries
- prosody-conditioned NLL correlations
  - previous boundary strength
  - distance to next sentence / paragraph break
  - recent punctuation density
  - recent non-content density
- lightweight linear probes from hidden states for:
  - inside-vs-outside quote
  - next sentence-boundary distance bucket
  - next paragraph-boundary distance bucket
  - content-vs-noncontent token identity

## Why This Matters

This is the low-cost first pass for two questions:

1. Is the prosodic channel large enough to matter for compression?
2. Does the current model already represent that channel internally?

If the answer to both is yes, then the next architecture interventions become
worth testing:

- token type embeddings
- prosody auxiliary heads
- transition/reset priors from paragraph and quote boundaries
- later, explicit prosody sidecars or tokenizer changes

## Current Heuristic Channel Definition

The current feature extractor is intentionally weak-label and text-native.

It uses SentencePiece piece strings to assign:

- token class
- boundary strength
- quote state
- distance to next sentence / paragraph / section break
- recent punctuation / non-content density

This is not meant to be the final ontology. It is the cheapest way to measure
whether the signal is real before changing the model.

## Example Invocation

```bash
python3 tools/analyze_residual_autocorrelation.py \
  --artifact /path/to/model.npz \
  --config-json /path/to/run_config.json \
  --result-json /tmp/prosody_diag.json \
  --analysis-max-batches 32 \
  --max-lag 64 \
  --transition-window 32 \
  --layerwise-layers 0,2,4,-1 \
  --probe-layers -1
```

The new JSON surface should be read first at:

- `token_class_loss`
- `boundary_conditioned`
- `quote_conditioned`
- `prosody_correlations`
- `prosody_probes`

## Expected Reads

If `punctuation` / `whitespace` / `quote` carry a nontrivial share of total
NLL, the prosodic channel is not negligible.

If `after_paragraph` or `after_sentence` NLL is materially higher than baseline,
boundaries behave like transition points and should feed reset logic.

If quote-state or distance-to-boundary probes decode well from hidden states,
the model already learned an implicit prosody subsystem.

If these probes improve strongly with layer depth, that tells us where the
stack starts resolving rhythm / boundary / voice state.

## Not Yet Implemented

- direct attention-head auditing for punctuation/newline specialization
- transcript-subset or dialogue-subset comparisons
- tokenizer changes such as typed-token BPE or graded silence tokens
- prosody auxiliary training heads

Those should come after this measurement lane confirms the signal is worth
separating architecturally.
