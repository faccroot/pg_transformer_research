# Hardmax Teacher-Distill Lane (2026-04-03)

## Purpose

Test a second hardmax supervision family in parallel with execution-trace pretraining:

- freeze a softmax LM teacher
- train a controller-only hardmax draft model on token sequences
- imitate the teacher's next-token certainty surface

This is the minimal "hardmax draft model" version of the user's idea:

- hardmax handles the teacher's high-confidence floor
- softmax teacher defines what "confident" means
- no routing or speculative decode yet

## Current Implementation

Trainer:

- [train_hardmax_teacher_distill.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_teacher_distill.py)

Key design:

- token embedding + position embedding
- `HardmaxStructuralController`
- full-vocab next-token head
- teacher targets:
  - teacher top-1 token
  - teacher top-1 probability
- losses:
  - token CE to teacher top-1 with heavier weight on teacher-confident positions
  - confidence regression to teacher top-1 probability
  - controller usage/diversity regularization

Outputs:

- controller-only init NPZ
- full draft-model NPZ
- summary JSON

## Local Teacher Asset

Resolved teacher config copied from local baseline:

- [plain_control_mini07_teacher_resolved.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_teacher_distill_assets/plain_control_mini07_teacher_resolved.json)

Teacher checkpoint:

- [plain_control_mini07_mlx_model.npz](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_sidecar_ref_clean_ab_1h/artifacts/plain_control_mini07_mlx_model.npz)

## Launched Smoke

Template:

- [mlx_hardmax_teacher_distill_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_teacher_distill_smoke.example.json)

Generated sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_044232_mlx-hardmax-teacher-distill-smoke/manifest.json)

Launched arms:

- `teacher-distill-1state`
- `teacher-distill-8state`
- `teacher-distill-8state-anticollapse`

Queue session:

- `34802`

## Relaunch after config fix

The first queue attempt exposed a wrapper mismatch:

- the queue wrapper invokes staged scripts as `script --config <json>`
- the original teacher-distill trainer only accepted direct CLI flags
- result: all first-launch attempts failed before startup with
  - `ValueError("Teacher config and checkpoint must be provided via --teacher-config/--teacher-checkpoint or env")`

Fix applied:

- [train_hardmax_teacher_distill.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_teacher_distill.py)
  - now understands the staged `--config` contract
  - applies both `env` and `args` payloads from the config JSON before parsing

Relaunched queue session:

- `27501`

Additional bring-up fixes after relaunch:

- moved teacher asset paths into config `env`, not only staged `args`
- added dataset/tokenizer envs to the sweep
- enabled teacher partial-load mode for this lane
- lowered the teacher partial-load floor to `0.40`
- fixed an MLX runtime issue in evaluation:
  - `mx.ones_like(..., dtype=...)` -> shape-based `mx.ones(...)`

Current state:

- the lane now reaches real teacher loading on cluster
- observed teacher-load diagnostics:
  - `matched_param_frac: 0.425`
  - `missing_tensors: 3`
  - `mismatched_tensors: 18`
- current active queue session: `27501`

## Clean rerun

After the bring-up fixes landed, a clean three-arm rerun was staged from the corrected template:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_051633_mlx-hardmax-teacher-distill-smoke/manifest.json)

Clean rerun queue session:

- `64151`

Purpose:

- recover the full `1-state` / `8-state` / `8-state + anti-collapse` comparison without the earlier failed retry history

Current status:

- the corrected rerun is launchable
- but the cluster is saturated, so the clean rerun has mostly burned retries on "No Minis available"
- this branch is now blocked on capacity, not on trainer wiring

## Retargeted teacher family

The original teacher-distill lane used a plain-control local teacher checkpoint.

A retargeted template now exists that points at the recovered hardmax transfer family instead:

- teacher config:
  - [04_structonly-step800-n8-d32-trace-statebook-freeze200.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/configs/04_structonly-step800-n8-d32-trace-statebook-freeze200.json)
- teacher checkpoint:
  - [mlx-hardmax-trace-transfer-statebook-freeze_structonly-step800-n8-d32-trace-statebook-freeze200_mlx_model.npz](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/artifacts/mlx-hardmax-trace-transfer-statebook-freeze_structonly-step800-n8-d32-trace-statebook-freeze200_mlx_model.npz)

Retargeted sweep template:

- [mlx_hardmax_teacher_distill_statebook_freeze200_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_teacher_distill_statebook_freeze200_smoke.example.json)

Generated retargeted sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211802_mlx-hardmax-teacher-distill-statebook-freeze200-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_211802_mlx-hardmax-teacher-distill-statebook-freeze200-smoke/README.md)

Purpose:

- move the certainty-floor supervision family closer to the current best hardmax LM branch
- reduce the gap between the teacher-distill branch and the winning transfer family

## What This Lane Should Answer

1. Does teacher-logit supervision keep the controller alive the way trace supervision did?
2. Is `8-state` better than `1-state` on teacher-match / coverage?
3. Does stronger anti-collapse help or hurt in this supervision regime?

## Immediate Next Gate

If the smoke is healthy:

- compare hardmax teacher-distill against a matched-budget small softmax draft baseline
- consider top-k / entropy targets after the first top-1 smoke

If it collapses:

- lower the teacher-confidence threshold
- increase uncertain-token weight
- add soft-to-hard temperature annealing before trying richer objectives
