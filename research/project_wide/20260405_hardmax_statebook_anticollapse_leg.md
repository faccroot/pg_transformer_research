# Hardmax Statebook Anti-Collapse Leg

This branch shifts the hardmax lane from transfer archaeology to the actual bottleneck:

- the transferred `state_book` helps exact FineWeb BPB
- but the live discrete controller is still almost fully collapsed inside the LM

Mechanistic backfill:

- [20260404_freeze_family_mechanistic_read.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_freeze_family_mechanistic_read.md)

Current best exact anchor:

- `trace state_book + freeze300`
  - `val_bpb 1.82215938`

## Updated branch split

This leg is now split into two sub-branches:

### H15a. Naive anti-collapse on the `freeze300` anchor

Purpose:

- test whether occupancy / commitment / boundary-conditioned transition pressure alone can revive the live controller

This is the currently running baseline anti-collapse family.

### H15b. `SimVQ + NextLat` on the `freeze300` anchor

Purpose:

- treat the `state_book` explicitly as a codebook optimization problem rather than only a regularization problem
- replace "keep codes alive somehow" with:
  - shared latent-basis codebook reparameterization
  - explicit next-state prediction on the hardmax path

This is now the principal variation after H15a reads out.

The intended read is:

- H15a tells us whether naive anti-collapse pressure is enough
- H15b tests whether the real bottleneck is codebook optimization plus latent dynamics

## New trainer knobs

Added in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py):

- `HARDMAX_STRUCT_STATE_USAGE_ENTROPY_WEIGHT`
  - encourages broader batch-level occupancy over the transferred codebook
- `HARDMAX_STRUCT_STATE_COMMIT_WEIGHT`
  - anchors the live `state_book` to the transferred trace-pretrained vocabulary
- `HARDMAX_STRUCT_TRANSITION_BOUNDARY_WEIGHT`
  - encourages soft state transitions to align with prosodic boundary / quote-change spikes
- `HARDMAX_STRUCT_SIMVQ_ENABLED`
  - replaces the raw `state_book` readout with a shared learned projection over the full codebook
- `HARDMAX_STRUCT_NEXTLAT_WEIGHT`
  - adds explicit next-state prediction on the hardmax path against the next soft code distribution

These sit on top of the existing:

- `HARDMAX_STRUCT_USAGE_BALANCE_WEIGHT`
- `HARDMAX_STRUCT_DIVERSITY_WEIGHT`
- `HARDMAX_STRUCT_PREDICT_WEIGHT`
- `HARDMAX_STRUCT_CONFIDENCE_WEIGHT`

## First sweep

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_011602_mlx-hardmax-statebook-anticollapse-freeze300-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_011602_mlx-hardmax-statebook-anticollapse-freeze300-smoke/README.md)

Arms:

1. `freeze300-baseline`
2. `freeze300-usageH010`
3. `freeze300-commit010`
4. `freeze300-transbnd020`
5. `freeze300-usageH010-commit010`

The mixed `attention-shaping + anti-collapse` arm is intentionally deferred until H2 lands a positive result. The first read should isolate whether collapse control alone can turn the transferred `state_book` into a more causally used latent object.

## Next principal variation

If H15a is flat or only weakly positive, do not keep iterating occupancy-only fixes.

The next build should be H15b:

1. `SimVQ`-style `state_book` reparameterization
2. `NextLat`-style next-state prediction on the hardmax path
3. then commitment / occupancy on top of that base, not as the only mechanism

Planned H15b arms:

1. `freeze300 + SimVQ-state_book`
2. `freeze300 + SimVQ-state_book + commitment + occupancy`
3. `freeze300 + SimVQ-state_book + NextLat`
4. `freeze300 + SimVQ-state_book + commitment + occupancy + NextLat`

## H15b smoke

Template:

- [mlx_hardmax_statebook_simvq_nextlat_freeze300_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_statebook_simvq_nextlat_freeze300_smoke.example.json)

Generated sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_042706_mlx-hardmax-statebook-simvq-nextlat-freeze300-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_042706_mlx-hardmax-statebook-simvq-nextlat-freeze300-smoke/README.md)

Arms:

1. `freeze300-baseline`
2. `freeze300-simvq`
3. `freeze300-simvq-usageH010-commit010`
4. `freeze300-simvq-nextlat005`
5. `freeze300-simvq-usageH010-commit010-nextlat005`

Current status:

- the H15b smoke is now launched from the current sweep tooling
- first read should answer whether SimVQ alone stabilizes the transferred codebook
- second read is whether adding `NextLat` produces a richer controller without giving back the `freeze300` BPB gain

First landed read:

- `freeze300-simvq`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.87679221`
  - this is worse than the `freeze300` exact anchor `1.82215938`
  - so shared codebook reparameterization alone is not enough

Queue note:

- the first baseline / `SimVQ` attempts did finish and write exports
- they were retried because the queue wrapper treated a non-training dispatch failure as a run failure after artifacts had already been written
- this was a tooling issue, not evidence that the model crashed during training

Follow-up continuation sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_054811_mlx-hardmax-statebook-simvq-nextlat-freeze300-followup/manifest.json)

Current follow-up status:

- `freeze300-simvq-usageH010-commit010` running on `mini07`
- `freeze300-simvq-nextlat005` running on `mini04`
- `freeze300-simvq-usageH010-commit010-nextlat005` queued behind them

## H15b scoreboard

Authoritative source:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_042706_mlx-hardmax-statebook-simvq-nextlat-freeze300-smoke/manifest.json)

Completed exact reads:

1. `freeze300-baseline`
   - only `final_raw_export_ready_exact` was recovered cleanly
   - `val_bpb 1.84054539`
2. `freeze300-simvq`
   - `final_int8_zlib_roundtrip_exact val_bpb 1.87679221`
3. `freeze300-simvq-usageH010-commit010`
   - `final_int8_zlib_roundtrip_exact val_bpb 1.88071773`
4. `freeze300-simvq-nextlat005`
   - `final_int8_zlib_roundtrip_exact val_bpb 1.88572483`
5. `freeze300-simvq-usageH010-commit010-nextlat005`
   - `final_int8_zlib_roundtrip_exact val_bpb 1.87700218`

Read:

- every H15b arm is materially worse than the `freeze300` anchor `1.82215938`
- `SimVQ`-only is already a miss
- `NextLat` on top of the current `SimVQ` implementation is also a miss
- combined occupancy / commitment / `NextLat` recovers slightly relative to `NextLat` alone, but is still far from competitive

Conclusion:

- the current H15b implementation should be treated as a negative result
- do not scale this exact `SimVQ + NextLat` formulation further without a stronger mechanistic reason

Decision rule:

- keep H15a if it materially improves both exact BPB and controller richness
- otherwise promote H15b immediately as the main anti-collapse branch
