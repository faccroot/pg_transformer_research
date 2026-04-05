# TurboQuant Transfer Handoff

This note is the current promotion-oriented handoff for the TurboQuant agent as of `2026-04-01`.

It separates:

- direct BPB-facing wins that already transferred on real Parameter Golf-style evaluations
- near-term structural branches that look promising but are not yet validated on final BPB
- things that should **not** be implemented in the main challenger

## Bottom Line

The current highest-confidence transfer path into Parameter Golf BPB is:

1. start from the trusted current-size Turbo/QAT base
2. add the validated phase-gated curriculum
3. add the early-exit auxiliary branch as the next direct BPB promotion candidate
4. test `segment-prev-read` as the first structural challenger
5. keep the program-cache / prefix-compiler lane as a separate high-upside research challenger, not the default stack

If the TurboQuant agent wants the shortest correct message:

- `phase-gated curriculum` is the strongest clean promotion
- `early-exit aux` is the strongest clean low-byte add-on
- `segment-prev-read` is the strongest currently implementable structural branch
- `prefix compiler` is the highest-upside speculative branch, but it has not yet been converted into end-to-end BPB

## Trusted Base

Start from the trusted current-size Turbo/QAT stack from [20260329_combined_challenger_handoff.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260329_combined_challenger_handoff.md):

- `9x512x2`
- untied embeddings
- no EMA
- `LeakyReLU^2` via `MLP_LEAKY_SLOPE=0.5`
- `turbo_block_v1`
- block size `256`
- `K` prod tensors, `Q/V/O + MLP + lm_head` MSE tensors
- Muon/Adam split
- plain Turbo export

Trusted strong-checkpoint reference:

- training-log exact roundtrip BPB: `1.43412685`
- artifact: `6,925,213` bytes
- source: [mini12_taskaware_export_20260326/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/mini12_taskaware_export_20260326/README.md)

## Most Promising Results

### 1. Phase-Gated Curriculum

This is the strongest current clean training-time promotion.

Exact one-hour result:

- baseline sequential: `1.52461262`
- curriculum order only: `1.53198915`
- curriculum order plus phase gating: `1.49965322`
- delta vs baseline: `-0.02495940`
- source: [20260326_060710_mlx-curriculum-ab3-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_060710_mlx-curriculum-ab3-1h/results_summary.md)

Transfer read:

- confidence: `high`
- expected PG BPB behavior: should transfer directly, because this was already measured on the same MLX/Turbo-style training path
- risk: low, as long as the agent keeps the gated variant and does **not** switch to order-only

Implementation:

- trainer: [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
- minimum env:
  - `CURRICULUM_ENABLED=1`
  - `CURRICULUM_FEATURES_PATH=/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/curriculum_train_shard0_features.npz`
  - `CURRICULUM_PHASE_PLAN_PATH=/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/curriculum_phase_plan.example.json`
  - `CURRICULUM_APPLY_QAT_PHASE_GATING=1`
  - `CURRICULUM_APPLY_EMA_PHASE_GATING=1`
  - `CURRICULUM_APPLY_FOCAL=0`

Reference promoted config:

- [02_current-size-curriculum.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260329_221157_mlx-h100-challenger-structural-ab4-1h/configs/02_current-size-curriculum.json)

### 2. Early-Exit Auxiliary

This is the strongest current clean low-byte improvement after curriculum.

Exact one-hour result:

- control exact int8-zlib: `1.59840419`
- early-exit aux exact int8-zlib: `1.58338580`
- delta: `-0.01501839`
- source: [20260330_031324_mlx-early-exit-aux-ab2-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_031324_mlx-early-exit-aux-ab2-1h/results_summary.md)

Mechanistic support:

- residual autocorrelation improved in [20260330_residual_autocorr_recent_1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260330_residual_autocorr_recent_1h/results_summary.md)
- boundary/whitespace prediction improved strongly in [20260401_prosody_diag_larger/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_prosody_diag_larger/results_summary.md)

Transfer read:

- confidence: `medium-high`
- expected PG BPB behavior: likely positive on the real stack, because it already improved exact exported BPB and does not cost export bytes if heads are stripped
- risk: moderate only because the strongest read is still one-hour, not a fully re-promoted longer run

Implementation:

- trainer: [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
- helper module: [early_exit_aux.py](/home/zaytor/transformer_research/parameter-golf/early_exit_aux.py)
- env:
  - `EARLY_EXIT_LAYER_INDEX=3`
  - `EARLY_EXIT_HORIZONS=1,2,3`
  - `EARLY_EXIT_AUX_WEIGHT=0.1`
  - keep training-only heads stripped from export

Code entry points:

- env parsing in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py#L300)
- export stripping logic already exists in the trainer

### 3. Segment-Previous-Read JEPA Branch

This is the strongest currently implementable structural branch that already has a positive exact probe.

Exact probe result:

- baseline: `2.10011343`
- `persist_train_carry=0`, `pred_weight=0.01`: `2.08463778`
- `persist_train_carry=0`, `pred_weight=0.0`: `2.08661388`
- deltas: `-0.01547565` and `-0.01349955`
- source: [20260329_turboquant_handoff_segmentlong.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260329_turboquant_handoff_segmentlong.md)

Transfer read:

- confidence: `medium`
- expected PG BPB behavior: likely positive if the architectural value survives the exact export path
- main risk: bytes; the best exact variant was about `94 KB` over the `16 MB` cap on the plain `int8+zlib` path

Implementation:

- trainer: [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py)
- first challenger env:
  - `LONGCTX_PERSIST_TRAIN_CARRY=0`
  - `LONGCTX_ENABLE_SIDECAR_READ=1`
  - `LONGCTX_SEGMENT_LEN=64`
  - `SIDECAR_STATE_DIM=64`
  - `SIDECAR_TAP_LAYER=3`
  - `SIDECAR_PRED_WEIGHT=0.01`
  - `SIDECAR_SIGREG_WEIGHT=0.0`
  - `SIDECAR_SPHERICAL_WEIGHT=0.0`

Byte-recovery fallback:

- same branch, but `SIDECAR_PRED_WEIGHT=0.0`

Implementation advice:

- compress `sidecar_read_proj`, `sidecar_read_scale`, and `sidecar_cell` aggressively
- if bytes are still tight, remove or simplify `sidecar_pred` first

### 4. Program Cache / Prefix Compiler

This is the highest-upside speculative branch, but it is **not** yet a direct BPB promotion.

Current evidence:

- cold-start loss headroom is real in [position_loss_1024_w4.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/position_loss_1024_w4.json)
- raw previous transcript helps a little
- hidden-state carry hurts
- learned compiled prefixes help a lot

Persisted small-budget results:

- free-form `s32` in [prefix_compiler_full_s32.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32.json)
  - `0-31`: `-0.7411`
  - `32-63`: `-0.4943`
  - `64-127`: `-0.4976`
- typed hierarchical `b8,k2,h1,hidden` in [prefix_compiler_typed_b8_k2_h1_hidden.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_hidden.json)
  - `0-31`: `-0.7538`
  - `32-63`: `-0.4859`
  - `64-127`: `-0.4259`

Transition-binned evaluation:

- the compiler helps **more** at boundaries than in stable regions
- naive hard reset is catastrophic
- sources:
  - [prefix_compiler_full_s32_transition_bins.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_transition_bins.json)
  - [prefix_compiler_full_s32_transition_bins_reset_hard.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_transition_bins_reset_hard.json)

Transfer read:

- confidence: `medium upside / low certainty`
- expected PG BPB behavior: potentially large if converted into a legal fixed-budget cross-window memory mechanism
- current blocker: it still lives as an eval-side/research compiler lane rather than a measured, artifact-fair mainline PG implementation

Implementation status:

- research tool only, not main trainer:
  - [train_prefix_compiler_ablation.py](/home/zaytor/transformer_research/parameter-golf/tools/train_prefix_compiler_ablation.py)
  - [analyze_compiler_transition_bins.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_compiler_transition_bins.py)
  - [analyze_prefix_slot_structure.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_prefix_slot_structure.py)

If the TurboQuant agent explores it anyway:

- promote candidates:
  - free-form `s32`
  - typed hierarchical `b8,k2,h1` with `utility_source=hidden`
- do **not** implement:
  - raw hidden carry
  - surprise-only typed selection
  - naive hard reset
- the typed variant likely needs slot-diversity pressure before it is worth exporting

## Expected Transfer To Parameter Golf BPB

The practical transfer ranking is:

1. `phase-gated curriculum`
   - already a clean direct BPB win
   - strongest promotion confidence

2. `early-exit aux`
   - already a clean direct exported BPB win
   - low export-byte risk

3. `segment-prev-read`
   - direct positive probe, likely real
   - byte-risk is the main obstacle, not modeling quality

4. `prefix compiler`
   - strongest conceptual upside
   - not yet proven on actual PG BPB
   - should stay a research challenger until it has a legal fixed-budget eval and export story

5. `corrected chunk-causal sidecar`
   - causally valid now
   - still not the best current promotion surface for TurboQuant

## What The TurboQuant Agent Should Actually Implement

### Mainline Promotion Stack

Implement this first:

1. trusted current-size Turbo/QAT base
2. `+` phase-gated curriculum
3. `+` early-exit auxiliary
4. plain Turbo export

This is the highest-confidence BPB-facing stack from the current results.

### First Structural Challenger

Implement next:

1. the mainline stack above
2. `+` `segment-prev-read`
3. run two variants:
   - `SIDECAR_PRED_WEIGHT=0.01`
   - `SIDECAR_PRED_WEIGHT=0.0`

If bytes are tight, keep the `0.0` fallback alive rather than abandoning the branch.

### Optional Research Challenger

If the agent has spare bandwidth after the above:

1. port the program-cache compiler idea as a fixed-budget cross-window memory branch
2. first compare:
   - free-form `s32`
   - typed hierarchical `b8,k2,h1,hidden`
3. do not add reset logic first
4. add slot-diversity pressure before trying to export the typed variant

## Do Not Implement In The Main Challenger

- curriculum order only
- token-category weighting
- context-delta weighting
- EMA teacher distillation
- inline low-compressibility filter
- old leaking sidecar families
- raw hidden prefix carry
- surprise-only typed compiler selection
- naive hard reset for the compiler lane
- task-aware export as the default on the strong current-size checkpoint family

## Suggested Message To The TurboQuant Agent

Use this verbatim or nearly verbatim:

> The highest-confidence transfer stack right now is: current-size Turbo/QAT base, plus phase-gated curriculum, plus early-exit auxiliary, with plain Turbo export. The clean direct gains are `-0.02496` BPB for curriculum and `-0.01502` BPB for early-exit aux. Implement those first in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py). For curriculum, use `CURRICULUM_ENABLED=1`, `CURRICULUM_FEATURES_PATH=/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/curriculum_train_shard0_features.npz`, `CURRICULUM_PHASE_PLAN_PATH=/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/curriculum_phase_plan.example.json`, `CURRICULUM_APPLY_QAT_PHASE_GATING=1`, `CURRICULUM_APPLY_EMA_PHASE_GATING=1`, and keep `CURRICULUM_APPLY_FOCAL=0`. For early-exit, use `EARLY_EXIT_LAYER_INDEX=3`, `EARLY_EXIT_HORIZONS=1,2,3`, `EARLY_EXIT_AUX_WEIGHT=0.1`, and keep the auxiliary heads stripped from export.
>
> The first structural challenger should be `segment-prev-read` in [train_gpt_mlx_segmentlong.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_segmentlong.py): `LONGCTX_PERSIST_TRAIN_CARRY=0`, `LONGCTX_ENABLE_SIDECAR_READ=1`, `LONGCTX_SEGMENT_LEN=64`, `SIDECAR_STATE_DIM=64`, `SIDECAR_TAP_LAYER=3`, first with `SIDECAR_PRED_WEIGHT=0.01`, then `0.0` as the byte-recovery fallback. Treat the architectural read path as the core value; the prediction head is optional garnish.
>
> Do not spend time on order-only curriculum, token weighting, old leaking sidecars, raw hidden carry, or hard reset in the compiler lane. If you explore the program-cache branch, the only promoted candidates are free-form `s32` and typed hierarchical `b8,k2,h1` with hidden-based utility; do not add boundary reset, because the transition-bin ablation falsified that hypothesis.

