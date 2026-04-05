# Freeze Family Mechanistic Read

This note closes the loop on the recovered `state_book` freeze family and separates exact BPB from controller diagnostics.

Primary sources:

- exact recovered results: [observed_results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/observed_results.json)
- control baseline: [observed_results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_034412_mlx-hardmax-trace-transfer-smoke/observed_results.json)
- controller diagnostics: [controller_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/hardmax_transfer_diagnostics/controller_summary.md)
- causal smoke eval: [causal_ablation_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/hardmax_transfer_diagnostics/causal_ablation_summary.md)
- logit-factor smoke eval: [factor_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/hardmax_transfer_diagnostics/factor_summary.md)
- freeze300 exact result recorded in lane note: [execution_trace_hardmax_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/execution_trace_hardmax_lane_20260403.md)

## Comparison Table

| Run | Exact val_bpb | Used states | Max state fraction | Self-transition fraction | `zero_hardmax` delta | Top factor EVR | Top factor \|coord\|-NLL corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1.88888776 | n/a | n/a | n/a | n/a | n/a | n/a |
| d32 random structonly | 1.85942869 | 1 | 1.000000 | 1.000000 | +0.000119 | 0.029882 | +0.038182 |
| trace `state_book` | 1.83989451 | 2 | 0.999750 | 0.999500 | -0.000016 | 0.030509 | +0.024013 |
| trace `state_book` + freeze100 | 1.83837788 | 1 | 1.000000 | 1.000000 | -0.000075 | 0.030503 | +0.022568 |
| trace `state_book` + freeze200 | 1.83088557 | 2 | 0.999348 | 0.998790 | -0.000273 | 0.030815 | +0.036637 |
| trace `state_book` + freeze300 | 1.82215938 | n/a | n/a | n/a | n/a | n/a | n/a |

## Read

- `state_book` transfer is real on exact BPB. It beats the matched `d32` random structural baseline by about `0.0195` BPB.
- `freeze100` and `freeze200` improve exact BPB further, and `freeze300` is the current best exact result.
- The discrete controller is still almost completely collapsed in every recovered diagnostic run:
  - `used_states` is only `1-2`
  - `max_state_fraction` is `0.999+`
  - `self_transition_fraction` is `0.998+`
- The gain from freeze is therefore not explained by a revived rich hard-state machine.

## Best Current Mechanistic Interpretation

- `state_book` transfer installs a useful structural prior.
- Early freeze preserves that transferred prior long enough for the LM trunk to adapt to it.
- The visible improvement tracks the soft control surface more than healthy discrete-state usage.

The strongest local evidence for that is:

- `trace state_book` by itself has near-flat confidence (`confidence std ~= 0.001845`) even though it improves BPB over random.
- `freeze100` and `freeze200` restore useful confidence variation (`0.131336` and `0.081667`) while the discrete state remains almost fully collapsed.
- The small causal smoke deltas are consistent with a weak-but-real mechanism, but they are not yet large enough to claim the hard discrete state is being causally used in a rich way.

## Decision

The bottleneck is no longer “does trace transfer work?” It does.

The bottleneck is:

- can we turn the transferred `state_book` from a helpful prior into a live, causally used discrete compiled state inside the LM?

That is why the next branch is anti-collapse `state_book` training on top of the `freeze300` anchor rather than more transfer archaeology.
