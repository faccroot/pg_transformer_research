# TurboQuant Handoff: Segment-Clock JEPA Findings

This note is the promotion-oriented handoff from the JEPA / long-context lane to the TurboQuant / combined-challenger lane.

## Bottom Line

The only JEPA-derived feature that currently looks worth promoting is:

- `segment-prev-read`
- `LONGCTX_PERSIST_TRAIN_CARRY=0`
- `LONGCTX_ENABLE_SIDECAR_READ=1`
- `LONGCTX_SEGMENT_LEN=64`
- `SIDECAR_STATE_DIM=64`
- `SIDECAR_TAP_LAYER=3`
- `SIDECAR_PRED_WEIGHT=0.01` or possibly `0.0`
- `SIDECAR_SIGREG_WEIGHT=0.0`
- `SIDECAR_SPHERICAL_WEIGHT=0.0`

## Current update

The modeling read has shifted slightly since this handoff:

- `segment-prev-read` still looks like the right architectural object
- token-rate persistent carry is still dead
- but the next useful branch is no longer "more shared JEPA latent work"

The next branch should be:

- detached boundary routing
- boundary enrichment / confidence alignment metrics
- boundary placement driven by predictive difficulty or uncertainty

That means the follow-on is:

- keep the slow segment-clock read path
- stop sharing one latent between "what to write" and "where to cut"
- use uncertainty or surprisal to decide where extra segment structure is worth paying for

This is implemented in [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py).

The clean exact result matrix is packaged in:

- [followup_matrix_20260329.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260329_segmentlong_probe_180s/followup_matrix_20260329.json)
- [followup_matrix_20260329.csv](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260329_segmentlong_probe_180s/followup_matrix_20260329.csv)

## Exact Results

Reference baseline from the original segment-clock suite:

- baseline exact BPB: `2.10011343`
- baseline int8+zlib artifact: `15,602,801` bytes

Segment previous-read follow-ups on a clean staged rerun:

- `persist_train_carry=1`, `pred_weight=0.01`: `2.11128816`
- `persist_train_carry=1`, `pred_weight=0.0`: `2.11317911`
- `persist_train_carry=0`, `pred_weight=0.01`: `2.08463778`
- `persist_train_carry=0`, `pred_weight=0.0`: `2.08661388`

Key deltas versus baseline:

- `nocarry + pred=0.01`: `-0.01547565 BPB`
- `nocarry + pred=0.0`: `-0.01349955 BPB`

Interpretation:

- The improvement comes mainly from the **segment previous-read architecture**.
- The JEPA prediction term is only **mildly positive**:
  - `2.08463778` vs `2.08661388`
  - gain from prediction loss is about `0.00198 BPB`

That distinction matters for compression:

- if bytes are tight, the `sidecar_pred` head is a plausible thing to simplify, compress harder, or remove in a follow-on implementation
- the architectural benefit likely survives most of that change

## Byte Situation

Important: the winning `nocarry` variant is **not** under the current 16 MB cap on the plain `int8+zlib` path.

For `persist_train_carry=0`, `pred_weight=0.01`:

- model int8+zlib blob: `16,056,423` bytes
- code size: `37,718` bytes
- total: `16,094,141` bytes

So the promotion path is:

- use this variant as a **modeling candidate**
- let TurboQuant / export work recover the missing ~`94 KB` plus any safety margin

## What To Promote

### 1. Main JEPA Promotion Candidate

Promote exactly one JEPA-style feature into the combined challenger:

- segment-clock previous-read conditioning
- **no** training-time cross-window carry
- reset eval remains mainline

This should be treated as a **slow-clock architectural prior**, not as a persistent streaming memory system.

### 2. Optional JEPA Head Choice

There are two defensible challenger variants:

1. `nocarry + pred_weight=0.01`
   - best exact BPB
   - more faithful to the JEPA framing

2. `nocarry + pred_weight=0.0`
   - slightly worse BPB
   - useful if the TurboQuant agent wants to delete or radically simplify the predictive head

If the combined challenger is byte-constrained, variant 2 is a legitimate fallback.

## What Not To Promote

### Dead / Negative

Do not promote the token-rate persistent-carry line from the earlier `superlong` prototype.

Verified negative:

- baseline: `2.11278213`
- token-rate superlong: `2.15356879`

See:

- [20260329_superlong_probe_180s/results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260329_superlong_probe_180s/results.json)

Failure decomposition showed the problem was the carry/read object itself, not just framework overhead. The null-overhead control was only slightly worse than baseline; the carry path was the real damage.

Do not promote:

- token-rate persistent carry
- persistent eval as a mainline feature
- write-only segment JEPA with no read path

### Contaminated / Invalid

Do not promote the earlier plain sidecar ref or detector-conditioned sidecar lines as submission-clean JEPA wins.

These branches had either:

- within-chunk causality contamination, or
- ambiguous leak history that makes them unsafe as the basis for a combined challenger

In particular, the flashy 1-hour sidecar ref numbers are not the safe promotion path for the TurboQuant agent.

## Secondary Candidate

There is one small, clean, non-sidecar structural result worth keeping as a shadow candidate:

- 2-level clustered chain-rule token head

Exact 180s result:

- baseline: `2.10119820`
- clustered head: `2.10056559`
- gain: `-0.00063261 BPB`
- byte delta: `+52,262`

See:

- [20260328_structured_heads_probe_180s/results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260328_structured_heads_probe_180s/results.json)

This is too small to be the main promotion, but it is a legitimate low-amplitude shadow feature if the TurboQuant agent wants one more cheap structural ablation.

## Promotion Path

The JEPA recommendation for the combined challenger is:

1. Start from the strongest non-JEPA combined control stack the TurboQuant agent already trusts.
2. Add **only** the segment previous-read mechanism from [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py).
3. Keep:
   - `LONGCTX_PERSIST_TRAIN_CARRY=0`
   - `LONGCTX_ENABLE_SIDECAR_READ=1`
   - `LONGCTX_SEGMENT_LEN=64`
4. Use `SIDECAR_PRED_WEIGHT=0.01` as the first challenger.
5. Keep a byte-recovery fallback with `SIDECAR_PRED_WEIGHT=0.0` or a physically removed prediction head if compression pressure demands it.

In other words:

- promote the **segment-clock read architecture**
- do **not** promote persistent recurrent memory
- treat the JEPA loss as optional garnish, not the core value

## TurboQuant-Specific Advice

If TurboQuant is choosing where to spend compression effort, prioritize:

- `sidecar_read_proj`
- `sidecar_read_scale`
- `sidecar_cell` weights
- any segment-side tensors added by [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py)

If bytes remain tight, the most disposable JEPA-specific component is likely:

- `sidecar_pred` and anything only needed for the auxiliary prediction loss

Reason:

- prediction loss helps only about `0.002 BPB`
- architecture-without-prediction is still better than baseline

## Implementation Notes

- The segmentlong export bug has already been fixed in [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py).
- The bug was caused by non-tensor hyperparameter ints leaking into `model.state`; export now filters to tensor-valued state before `mx.savez`.
- The current file is safe to use as the promotion starting point.

## Recommended Combined-Challenger Order

1. Non-JEPA combined control stack
2. Add `segment-prev-read`
3. Keep `LONGCTX_PERSIST_TRAIN_CARRY=0`
4. First try `pred_weight=0.01`
5. If bytes are too tight, try `pred_weight=0.0` or a no-pred variant
6. Only then consider the tiny clustered-head shadow addition

That is the cleanest promotion path out of this JEPA session.
