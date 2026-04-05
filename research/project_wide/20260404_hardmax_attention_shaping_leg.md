# Hardmax Attention-Shaping Leg (2026-04-04)

## Why this branch exists

The current hardmax lane now has a clean architectural read:

- structural conditioning helps
- residual-budget routing hurts
- trace supervision revives the controller pretrain
- transferred `state_book` priors improve LM BPB
- early warm-start freezing improves that transfer further

That combination points away from "better routing" and toward:

- always-on hardmax conditioning
- full trunk compute retained
- attention geometry shaped by structural state

## Minimal implementation now in trunk

The LM trunk in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py) now supports:

- `HARDMAX_STRUCT_CONDITION_MODE=residual`
  - existing behavior
  - hardmax returns a residual-conditioned hidden state

- `HARDMAX_STRUCT_CONDITION_MODE=q_bias`
  - controller state is projected into an additive `q` bias for later layers
  - trunk stays fully on

- `HARDMAX_STRUCT_CONDITION_MODE=q_bias_temp`
  - same `q` bias
  - plus a learned per-head temperature term derived from the controller state

Additional knobs:

- `HARDMAX_STRUCT_ATTN_Q_SCALE`
- `HARDMAX_STRUCT_ATTN_TAU_MIN`

The new path is intentionally narrow:

- no same-layer interleaving yet
- no repeated controller injection schedule yet
- no routing
- no extra controller recurrence beyond the existing microstep knob

## Current seam

The controller still updates at the configured structural layer.

What changes is how its state is used after that point:

- residual mode:
  - controller writes directly into the residual stream

- attention-shaping modes:
  - controller state is converted into attention-conditioning signals for subsequent layers
  - the controller output does not replace trunk compute

That makes this the direct architectural successor to the failed budget router.

## First planned matrix

The first staged sweep keeps capacity and transfer fixed and only changes the conditioning mode:

- residual baseline
- random-init `q_bias`
- trace `state_book + freeze200` `q_bias`
- trace `state_book + freeze200` `q_bias_temp`

Template:

- [mlx_hardmax_trace_transfer_attn_conditioning_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_transfer_attn_conditioning_smoke.example.json)

Generated runs:

- exploratory `freeze200` variant
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_191323_mlx-hardmax-trace-transfer-attn-conditioning-smoke/manifest.json)
  - [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_191323_mlx-hardmax-trace-transfer-attn-conditioning-smoke/README.md)

- canonical `freeze300` variant
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211052_mlx-hardmax-trace-transfer-attn-conditioning-smoke/manifest.json)
  - [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211052_mlx-hardmax-trace-transfer-attn-conditioning-smoke/README.md)

## Decision rule

Positive result:

- attention-shaping beats the current residual `state_book + freeze300` anchor
- or preserves BPB while making the controller more causally useful at eval

If positive, expand next in this order:

1. repeated conditioning across multiple layers
2. stronger pretrained state books from the memops export family
3. only then microsteps on top of the attention-shaped branch

If flat or negative:

- keep the code path
- do not expand this branch yet
- stay on the transfer-exploit line
