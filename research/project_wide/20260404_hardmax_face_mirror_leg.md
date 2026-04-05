# Hardmax Face / Mirror Leg

## Hypothesis

The hardmax path should not only condition the trunk. It should also emit a low-bandwidth
operator-state trace that can be:

- read later by the same model (`mirror`)
- published to other agents as a reliability / review signal (`face`)

This trace is meant to track operators that restructure downstream processing rather than
surface text alone. The first useful channels are:

- operator code / polarity seed
- blended polarity score
- hardmax confidence
- hardmax budget
- hard state index
- soft state usage mixture
- structural state norm

The immediate engineering goal is not a full introspection architecture. It is a prompt-level
export artifact that lets us inspect and compare traces from saved hardmax checkpoints.

## First Artifact

Tool:

- [export_hardmax_face_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/export_hardmax_face_trace.py)

What it exports for a free-text prompt:

- token ids and SentencePiece pieces
- next-token top-k predictions
- operator code and polarity fields
- per-token hardmax confidence / budget
- per-token hard state index
- top soft states per token
- aggregate usage summaries

The tool reads a saved config JSON plus a saved MLX artifact and runs the model through the
existing `forward_hidden_with_aux` path. This keeps H11 on the same inference seam as the
current hardmax transfer lane rather than introducing a separate introspection stack.

## Current Status

- implemented and syntax-checked locally
- local CPU-MLX import is now available, but actual MLX CPU execution on this Ubuntu host is still brittle because of the host JIT/compiler path
- Mini-first remote execution is now the preferred analysis path for H11 via:
  - [run_remote_saved_hardmax_analysis.py](/home/zaytor/transformer_research/parameter-golf/tools/run_remote_saved_hardmax_analysis.py)
- first practical smoke target is the recovered local artifact family from:
  - [20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze)
- first remote smoke now landed at:
  - [face_trace_freeze200_negation.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223900_hardmax-face-trace-smoke/face_trace_freeze200_negation.json)
- runtime parity blocker was resolved by shipping a small repo-root bundle with the export job
- the same remote path now also works for saved controller diagnostics:
  - [controller_freeze200_val2_remote.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/hardmax_transfer_diagnostics/controller_freeze200_val2_remote.json)

Observed result from the first real structural trace:

- `has_hardmax_structural=true`
- `confidence_mean=0.9241`
- `used_states=1`
- `max_state_fraction=1.0`

Interpretation:

- the face-trace path works end to end
- the exported trace is informative about soft control surface
- but the hard state itself is still collapsed to a single state in this `freeze200` winner

Observed result from the first Mini-first controller smoke on the same family:

- `used_states=2`
- `max_state_fraction=0.9985`
- `self_transition_fraction=0.9971`
- `confidence_std=0.0808`

Interpretation:

- the controller-analysis path now also works end to end through the same remote wrapper
- the broader validation view is consistent with the prompt-level face trace:
  the controller is not dead, but the discrete state remains almost fully collapsed

So H11 is no longer blocked on tooling. It is now blocked on controller richness.

## Planned Tests

### H11-A: Face export only

Export trace for short prompts containing:

- negation
- uncertainty
- boundary / topic shift
- quoted speech

Readout:

- whether operator-state channels vary in interpretable ways
- whether state changes align with negation and boundary transitions

### H11-B: Reviewer with face trace

Compare:

- text only
- text + face trace

for downstream review / weak-point localization.

### H11-C: Mirror self-read

Second-pass revision with:

- text only reconstruction of prior state
- direct access to stored face trace

If direct trace access helps revision or reliability estimation, the mirror hypothesis is real.

## Decision Rule

Advance H11 only if the exported trace is visibly nontrivial and varies in ways that are not
recoverable from surface text alone. If the trace is mostly constant or collapsed, H11 remains
blocked on the controller itself becoming richer.
