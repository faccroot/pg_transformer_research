# Sidecar Canonicalization Plan

Date: 2026-03-30

Purpose: collapse the sidecar family to one canonical causal path so future
architecture work does not keep fragmenting across partially overlapping
trainers.

## Decision

The canonical causal sidecar lane is:

- [train_gpt_mlx_sidecar_canonical.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_sidecar_canonical.py)

which is a stable entrypoint for:

- [train_gpt_mlx_jepa_sidecar_chunkcausal.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_jepa_sidecar_chunkcausal.py)

This is the sidecar family that should count as architecture evidence going
forward.

## Why This One

The chunk-causal runner is the cleanest causal formulation already present in
the repo:

- chunk `c` is conditioned only on sidecar state from chunk `c-1`
- the scored path inside each chunk remains causal
- the sidecar is a compact recurrent belief-state object, not a broadcast
  summary of the current chunk
- the file already has the most explicit causal docstring and operational
  assumptions

That makes it the right base for:

- sidecar vs control validation
- sidecar plus curriculum routing
- sidecar plus early-exit composition
- later evaluation-side adaptation experiments

## Status Of The Other Variants

### Segment-long

- [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py)

Keep as the `slow-clock long-memory research` lane.

Interpretation:

- useful follow-on once the canonical chunk-causal sidecar is revalidated
- not the default sidecar trainer
- not the first place to add unrelated architectural features

### Superlong

- [train_gpt_mlx_superlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_superlong.py)

Treat as legacy V1 token-rate carry research.

Interpretation:

- keep for reference and compatibility
- do not use as the default for new sidecar work
- do not route future “mainline” ideas through this file first

### Older JEPA sidecar families

Keep only for artifact compatibility and historical comparisons.

They should not be the default for:

- new training runs
- new evaluator wrappers
- promotion-path notes

## Concrete Cleanup Completed

The repo now has:

- canonical entrypoint: [train_gpt_mlx_sidecar_canonical.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_sidecar_canonical.py)
- evaluator default moved to the canonical trainer module in [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py)
- curriculum-routed sidecar auxiliary scaling in [train_gpt_mlx_jepa_sidecar_chunkcausal.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_jepa_sidecar_chunkcausal.py) via [sidecar_aux.py](/home/zaytor/transformer_research/parameter-golf/sidecar_aux.py)
- canonical run template scaffold in [mlx_sidecar_canonical_ab3_1h.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_sidecar_canonical_ab3_1h.example.json)

This keeps future sidecar eval/adaptation work aimed at the causal lane by
default while preserving explicit override support for older artifacts.

The canonical lane now also has a first transition-aware reset surface:

- heuristic or learned transition-reset gate in [train_gpt_mlx_jepa_sidecar_chunkcausal.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_jepa_sidecar_chunkcausal.py)
- saved-artifact evaluation flags in [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py)
- reset-gate training auxiliaries:
  - sparsity penalty
  - weak prior-alignment penalty

This is the direct response to the residual-ACF finding that the sidecar helps
within regimes but carries stale belief across regime transitions.

## Next Engineering Steps

1. Revalidate the canonical sidecar against the current strong control.
2. Use the canonical run scaffold in [mlx_sidecar_canonical_ab3_1h.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_sidecar_canonical_ab3_1h.example.json) for clean sidecar-vs-control and sidecar-plus-curriculum A/Bs.
3. Test heuristic transition reset on existing saved sidecar artifacts before
   paying for a new training run.
4. Train the learned transition-reset variant only if the heuristic reset helps
   the cross-regime residual metrics.
   The first clean template for that is now:
   [mlx_sidecar_transition_reset_ab2_1h.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_sidecar_transition_reset_ab2_1h.example.json)
5. Test canonical sidecar plus early-exit once the early-exit promotion gate is
   resolved.
6. Keep segment-long as the next long-memory extension, not as the default
   entrypoint.

## Operating Rule

When a future note says `sidecar`, it should mean:

- the canonical chunk-causal sidecar path

unless the note explicitly says:

- `segment-long`
- `superlong`
- `legacy sidecar`
