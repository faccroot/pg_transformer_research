# Compiled-State Supervision Leg

## Core distinction

The context window is the source program.

- raw text / code / dialogue
- human-readable
- not optimized for transformer execution

The KV cache is the compiled runtime state.

- the model's internal interpretation of the source program
- layerwise state used by future attention operations
- closer to the actual computation than the raw document

The weights are the compiler.

That means next-token training supervises only the product of the compiled program, not the
compilation process itself.

## Hypothesis

Standard next-token supervision is too thin relative to the latent state it tries to shape.

Language models should benefit from richer train-time supervision on properties of the
compiled state, such as:

- multi-horizon next-token targets
- rollout residuals between predicted future tokens and realized futures
- next internal-state summaries
- KV-relevance / memory-utility targets
- confidence / uncertainty trajectories

The general claim is:

- more supervision bits should come from the process that produced the token
- not only from the token itself

## Why this matters here

The hardmax lane is already moving in this direction.

- trace supervision worked better than shallow structural labels
- `state_book` transfer beat random init
- the face / mirror branch treats operator-state trace as an exportable artifact

This leg asks the more general question:

- what other train-time targets expose the model's own compiled state?

## Shared analysis seam

Saved-artifact factor analysis now exists at:

- [analyze_saved_logit_factors.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_saved_logit_factors.py)

That tool gives a first offline basis over output-distribution behavior using either:

- raw logit variance
- probability residuals against the realized next token

and reports factor summaries against NLL / entropy, with optional hardmax causal ablations.

## First concrete tests

### C1. Multi-horizon token heads

Train-time only auxiliary heads for:

- `t+1`
- `t+2`
- `t+4`
- `t+8`

Question:

- does richer horizon supervision improve BPB more than the same compute budget on pure next-token loss?

Existing engineering seam:

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py) already supports multi-horizon early-exit auxiliary heads
- first H12 smoke template:
  - [mlx_h12_multihorizon_statebook_freeze300_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_h12_multihorizon_statebook_freeze300_smoke.example.json)
- generated run:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/manifest.json)
- staged follow-on aux-weight ladder:
  - [mlx_h12_multihorizon_auxweight_ladder.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_h12_multihorizon_auxweight_ladder.example.json)

Important reframing:

- the current H12 smoke is best treated as a probe, not the intended long-term form
- naive horizon heads on small models are no longer the preferred target object
- the next reboot should move supervision onto register-like state slots and use forward curriculum rather than only parallel fixed-horizon heads

Decision rule:

- if the `0.05` auxiliary sweep is directionally positive but small, do not conclude the branch is weak yet
- next step should be the aux-weight ladder at `0.1` and `0.2`, with a random-init separator run
- if naive H12 is flat or weak, do not keep extending it blindly; reboot as:
  - register/state-slot futures
  - forward curriculum over horizons
  - later, one long-horizon future-summary head

### C1b. Register / curriculum supervision reboot

New principal variation for H12:

- use register-like future slots or hardmax-adjacent state slots
- predict horizons through those slots instead of bolting all heads directly onto the trunk
- train with forward curriculum (`1 -> 2 -> 4 -> 8`) rather than immediate full-horizon supervision

Intended read:

- test whether the supervision-density idea is right but the naive small-model head design is wrong

Related branch:

- [20260404_distribution_shape_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_distribution_shape_supervision_leg.md)

That branch asks the adjacent target question:

- if H12 adds more future-token targets, can H13 add better distribution-shape targets on the same forward pass?

### C2. Rollout residual supervision

At step `t`, predict a short future rollout.

At step `t+1`, compare:

- prior rollout prediction
- realized future prefix

Use the discrepancy as an auxiliary target.

Question:

- can the model learn from its own short-horizon forecasting errors in a way that improves later token prediction?

### C3. KV relevance targets

Use future attention or other retrieval proxies to label whether current-token state remains
useful later.

Question:

- can the model learn an explicit notion of memory utility that later supports cache compaction or context compression?

## Decision rule

Advance this branch only if the added supervision improves transfer or BPB without becoming
just another expensive auxiliary loss that does not survive export.

Current queue interpretation:

- keep the live H12 smoke as a probe
- but the next real H12 build should be the register/curriculum reboot, not more naive multi-horizon tuning
