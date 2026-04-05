# Combined Challenger Handoff

This note is the promotion-oriented handoff for the TurboQuant agent that will build a best-of challenger from the strongest findings across the recent sessions.

Related note for the new branching / early-exit lane:

- [branching_and_early_exit_ablation_lane.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/branching_and_early_exit_ablation_lane.md)

## Bottom Line

The safest current combined-challenger base is still the current-size Turbo/QAT stack:

- `9x512x2`
- untied embeddings
- no EMA
- `LeakyReLU^2` via `MLP_LEAKY_SLOPE=0.5`
- Turbo QAT on
- `turbo_block_v1`
- block size `256`
- `K` prod tensors, `Q/V/O + MLP + lm_head` MSE tensors
- Muon/Adam split
- plain Turbo export as the default export path

Best current 4-hour record for that family:

- training-log exact roundtrip BPB: `1.43412685`
- artifact: `6,925,213` bytes
- source: [mini12_taskaware_export_20260326/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/mini12_taskaware_export_20260326/README.md)

The main promotion recommendations on top of that base are:

1. Promote the validated plain phase-gated curriculum.
2. Keep the clean `segment-prev-read` JEPA feature as the primary structural challenger branch.
3. Keep the corrected chunk-causal sidecar as a research-to-challenger branch, but do not promote the older sidecar wins directly.
4. Keep plain Turbo export as the default export path on the strong combined-control lane.

## Promote Now

### 1. Phase-Gated Curriculum

This is the cleanest positive training-time composition result outside the export lane.

Clean 1-hour plain-stack result:

- baseline sequential: `1.52461262`
- curriculum order only: `1.53198915`
- curriculum order plus gating: `1.49965322`
- delta vs baseline: `-0.02495940`
- source: [20260326_060710_mlx-curriculum-ab3-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_060710_mlx-curriculum-ab3-1h/results_summary.md)

Read:

- phase gating is the thing that matters
- plain reordering alone is not robust
- token-local weighting variants should stay off the stack

Recommendation:

- promote `curriculum order + phase gating`
- do not promote order-only

### 2. Segment-Previous-Read JEPA Branch

This is the cleanest JEPA-derived feature currently in hand.

Exact probe results:

- baseline: `2.10011343`
- `persist_train_carry=0`, `pred_weight=0.01`: `2.08463778`
- `persist_train_carry=0`, `pred_weight=0.0`: `2.08661388`
- delta vs baseline: `-0.01547565` and `-0.01349955`
- source: [20260329_turboquant_handoff_segmentlong.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260329_turboquant_handoff_segmentlong.md)

Important byte note:

- the best exact variant is about `94 KB` over the `16 MB` total cap on the plain `int8+zlib` path
- that makes it a good TurboQuant branch, not an automatic default

Recommendation:

- treat `segment-prev-read` with `LONGCTX_PERSIST_TRAIN_CARRY=0` as the main JEPA challenger branch
- first try `SIDECAR_PRED_WEIGHT=0.01`
- keep `pred_weight=0.0` as the byte-recovery fallback

## Promote With Caution

### 3. Corrected Chunk-Causal Sidecar

The earlier sidecar wins were strong, but the older sidecar families are not safe to promote directly.

What happened:

- the older `sidecar_ref` family looked excellent in 1-hour and 4-hour composition runs
- later stronger causality probes found same-chunk leakage in the actual trainer path

Strong causality evidence:

- reproduced sidecar-ref A/B summary with verdict: [20260326_sidecar_ref_clean_ab_1h/results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_sidecar_ref_clean_ab_1h/results.json)
- same-chunk leak report: [20260326_sidecar_polarity_ab_1h/sidecar_ref_within_chunk_causality_true_ref.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_sidecar_polarity_ab_1h/sidecar_ref_within_chunk_causality_true_ref.json)

So:

- the flashy `1.5097`, `1.5085`, and `1.3476` sidecar-composition wins are best treated as upper bounds, not safe promotion numbers
- they are still useful as research signals
- they are not the basis for the combined challenger

What is safe now:

- corrected MLX trainer: [train_gpt_mlx_jepa_sidecar_chunkcausal.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_jepa_sidecar_chunkcausal.py)
- first Torch/CUDA port: [train_gpt_sidecar.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_sidecar.py)
- exact causality pass on the corrected chunk-causal MLX path: [20260327_chunkcausal_sidecar_smoke/causality_report.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260327_chunkcausal_sidecar_smoke/causality_report.json)

Current status:

- the corrected sidecar path is causally valid
- it does not yet have a finished 1-hour clean A/B good enough to promote into the combined challenger

Recommendation:

- keep the corrected chunk-causal sidecar as a branch to test
- do not promote any older sidecar checkpoint family directly

### 4. Rotated / Ternary Compression Branch

This remains the strongest alternate compression branch, but it is checkpoint-sensitive.

Key export-analysis result on the untied recipe:

- naive core-only ternary: `2.45958286`
- rotated core-only ternary: `2.43694264`
- improvement vs naive: `-0.02264022`
- improvement vs raw float checkpoint: `-0.06231510`
- source: [rope_gauge_deepdive.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/rope_gauge_deepdive.md)

But on the stronger 4-hour current-size combined-control checkpoint:

- rotated core-only ternary did not transfer
- source: [mini12_taskaware_export_20260326/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/mini12_taskaware_export_20260326/README.md)

Recommendation:

- keep rotated core-only ternary as a serious alternate challenger branch
- do not replace plain Turbo export with it by default on the current strong checkpoint family

## Do Not Promote

These paths are negative or too unstable to belong in the current challenger stack.

### Training-Time Negatives

- EMA teacher distillation
  - `1.56782029 -> 1.58544478`
  - delta `+0.01762449`
  - source: [20260326_160030_mlx-ema-teacher-distill-ab2-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_160030_mlx-ema-teacher-distill-ab2-1h/README.md)

- inline low-compressibility filter
  - `1.43519626 -> 1.45832297`
  - delta `+0.02312671`
  - source: [20260326_113727_mlx-combined-control-filter-ab2-4h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_113727_mlx-combined-control-filter-ab2-4h/README.md)

- token-category weighting
  - `1.52665040 -> 1.53971556`
  - delta `+0.01306516`
  - source: [20260326_104741_mlx-token-category-weighting-ab2-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_104741_mlx-token-category-weighting-ab2-1h/results_summary.md)

- exact short-vs-long context-delta weighting
  - `1.52621112 -> 1.56916658`
  - delta `+0.04295546`
  - source: [20260326_061206_mlx-context-delta-weighting-ab2-1h/results_summary.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_061206_mlx-context-delta-weighting-ab2-1h/results_summary.md)

Read:

- do not try to rescue token-local weighting in the challenger
- keep the validated curriculum, but stop short of per-token weighting

### Export-Time Negatives On The Strong 4h Control Lane

- task-aware export stack regressed on the strong 4-hour current-size combined-control checkpoint
- baseline Turbo full-val: `1.49196427`
- best task-aware full-val: `1.49466154`
- delta: `+0.00269726` BPB worse for `-150` bytes
- source: [taskaware_export_combined_control_4h_20260327_v1/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/analysis/taskaware_export_combined_control_4h_20260327_v1/README.md)

Recommendation:

- keep plain Turbo export as the default on the strong combined-control lane
- treat gauge / task-aware export as branch experiments, not as the promoted default

### Scale-Only Frontier

The current Mini-scale frontier says bigger trunks are not automatically better under the same wallclock.

1-hour scale frontier:

- `9x512x2`: `1.56400726`
- `12x640x2`: `1.59916346`
- `12x768x2`: `1.63517570`
- `15x640x2`: `1.62811976`
- source: [20260329_141454_mlx-plain-turbo-scale-frontier-1h/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260329_141454_mlx-plain-turbo-scale-frontier-1h/README.md)

Recommendation:

- keep `9x512x2` as the default current-size challenger base
- treat larger shapes as separate H100-only budget tests, not default promotions

## Potential Promotion Path

The cleanest combined-challenger construction order is:

1. Start from the trusted current-size Turbo/QAT control stack:
   - `9x512x2`
   - untied embeddings
   - no EMA
   - `LeakyReLU^2`
   - `turbo_block_v1`
   - block size `256`
   - `K` prod / `QVO+MLP+lm_head` MSE
   - Muon/Adam split
   - plain Turbo export

2. Promote the validated plain phase-gated curriculum.

3. Run the first structural challenger as:
   - base stack + `segment-prev-read`
   - `LONGCTX_PERSIST_TRAIN_CARRY=0`
   - `LONGCTX_ENABLE_SIDECAR_READ=1`
   - `LONGCTX_SEGMENT_LEN=64`
   - `SIDECAR_STATE_DIM=64`
   - `SIDECAR_TAP_LAYER=3`
   - `SIDECAR_PRED_WEIGHT=0.01`

4. Keep a byte-recovery fallback:
   - same branch, but `SIDECAR_PRED_WEIGHT=0.0`

5. Keep the corrected chunk-causal sidecar as the next structural branch:
   - only after it has a finished clean 1-hour A/B
   - do not use older `sidecar_ref` or detector-sidecar numbers as the decision surface

6. Keep rotated core-only ternary as the main alternate compression branch:
   - evaluate it as a separate challenger branch, not the default export replacement

7. Keep gauge / task-aware export only as a low-priority branch on top of a new checkpoint family if it wins there; do not pre-bake it into the promoted stack.

## Suggested Challenger Matrix

If the TurboQuant agent wants a compact ablation plan, use this order:

### A. Default Promotion Branch

- current-size Turbo/QAT base
- + phase-gated curriculum
- plain Turbo export

### B. JEPA Structural Branch

- A
- + `segment-prev-read`
- test `pred_weight=0.01` and `0.0`

### C. Compression Branch

- A
- + rotated core-only ternary export

### D. Sidecar Research-to-Challenger Branch

- A
- + corrected chunk-causal sidecar
- only after the clean 1-hour A/B lands

### E. Optional Small Shadow Branch

- A
- + 2-level clustered chain-rule token head
- source: [20260329_turboquant_handoff_segmentlong.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260329_turboquant_handoff_segmentlong.md)

This is intentionally not “everything at once.” The safe route is:

- lock the best clean base
- add curriculum
- test the clean JEPA segment branch
- then test compression alternatives
- only then revisit the corrected sidecar branch

## Research Notes That Matter, But Are Not Promotion Criteria Yet

- The contaminated sidecar families consistently showed operator-sensitive state subspaces and long-range post-operator benefit.
- That is useful for theory and future design.
- It is not yet a reason to promote those checkpoint families into the combined challenger.

Practical interpretation:

- explicit training-time detector / logic writes have repeatedly behaved like competing-gradient interventions
- clean structural priors with causal read paths are more promising than explicit training-time operator steering

## Final Recommendation

For the TurboQuant agent’s first best-of challenger:

1. Start from the trusted current-size Turbo/QAT base.
2. Add phase-gated curriculum.
3. Use plain Turbo export.
4. Make `segment-prev-read` the first JEPA challenger branch.
5. Keep rotated core-only ternary as the first alternate compression branch.
6. Keep corrected chunk-causal sidecar as a follow-on branch only after it has a clean 1-hour result.

That is the highest-confidence promotion path from the current session state.
