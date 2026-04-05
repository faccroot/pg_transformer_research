# Trace Execution Verification Lane

Date: 2026-04-04

Purpose: separate three claims that were previously conflated:

1. the controller learns from execution traces
2. the controller generalizes next-step execution dynamics on held-out programs
3. the controller can actually roll forward a novel program in an execution-like
   way

## Current Truth

The trace pretrainer already reports held-out validation metrics such as:

- `val_opcode_acc`
- `val_write_acc`

on a train/val split of synthetic programs.

That means the model is doing more than memorizing a single training batch.

But the current pretraining task is not yet raw-program execution:

- inputs include rich current-trace features such as current read/write vars,
  branch label, env delta size, stack depth, and env size
- targets are the next trace event fields
- stack depth and env size are used as inputs but are not predicted by the
  current heads

So the strongest current supported statement is:

- the controller learns a held-out execution-dynamics model over rich trace
  state

Not yet:

- the controller executes raw programs from bytecode alone

## Planning Read

The main research question this lane needed to answer is now answered well
enough:

- the controller is not just memorizing one fixed trace batch
- held-out teacher-forced execution-dynamics prediction is real
- the transferred state vocabulary is therefore grounded in generalizable
  execution-state structure, not only in trace memorization

That means this lane is now primarily:

- mechanistic validation

not:

- the critical path for Parameter Golf BPB improvement

Open-loop autonomous rollout remains interesting future work, but it is no
longer a blocking question for the transfer story.

## New Tooling

Verifier CLI:

- [verify_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/verify_hardmax_execution_trace.py)

Reusable helper logic:

- [execution_trace_verifier.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_verifier.py)

Focused tests:

- [test_execution_trace_verifier.py](/home/zaytor/transformer_research/parameter-golf/tests/test_execution_trace_verifier.py)

Trainer export upgrade:

- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)
  now saves a full pretrainer checkpoint by default when `out_dir` is set

## What The Verifier Measures

### 1. Teacher-forced held-out dynamics

It evaluates held-out next-step prediction under a ladder of input ablations:

- `none`
- `drop_memops`
- `drop_branch`
- `drop_delta_output`
- `opcode_step_plus_sizes`
- `opcode_step_only`

This answers:

- how dependent is the controller on answer-like rich trace inputs?
- how much survives if only opcode/step type and coarse state summaries remain?

### 2. Semi-open-loop rollout

Because the current model does not predict next stack depth or next env size,
full raw open-loop execution is not yet possible.

The verifier therefore measures a bounded rollout:

- predicted next trace fields are fed back in
- stack depth and env size stay oracle-carried

Modes:

- `predicted_all_oracle_sizes`
- `predicted_opcode_step_only_oracle_sizes`

This answers:

- does the controller maintain useful execution-like state across multiple
  steps?
- how quickly does it drift once rich teacher forcing is removed?

## Immediate Use

Run this on the next trace-pretrain export that includes the new full-model
checkpoint.

Current live sweep for that purpose:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_023107_mlx-hardmax-trace-pretrain-export-memops-verify/manifest.json)

Purpose of the live sweep:

- produce a fresh mixed+memops export with:
  - controller init artifact for LM transfer
  - full pretrainer checkpoint for verification
- then run:
  - [run_trace_execution_verification.py](/home/zaytor/transformer_research/parameter-golf/tools/run_trace_execution_verification.py)

Decision ladder:

1. if `opcode_step_only` collapses completely:
   - the current model is mostly a trace-completion learner
2. if `opcode_step_plus_sizes` stays materially alive:
   - the controller has learned a useful coarse execution dynamics model
3. if semi-open-loop rollout remains accurate for several steps:
   - the controller is much closer to a learned executor than a one-step probe

## Blocking Limitation

The current 8-state export in
[trace_pretrain_export_8state_hardmax_controller_init.npz](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_transfer_assets/trace_pretrain_export_8state_hardmax_controller_init.npz)
is controller-only.

That is enough for LM transfer, but not enough for this verifier, which also
needs:

- trace input embeddings
- prediction heads

So verification becomes clean on the next export after the full-model checkpoint
change lands.

## First Verification Result

Verified export:

- [trace-pretrain-export-mixed-memops-verify.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_023107_mlx-hardmax-trace-pretrain-export-memops-verify/trace_execution_verification/trace-pretrain-export-mixed-memops-verify.json)

Summary:

- teacher-forced held-out dynamics are real
- semi-open-loop execution is not yet real
- strongest honest claim is now:
  - the controller learns a held-out transition model over execution-trace state

Teacher-forced held-out field accuracies on `256` val sequences:

- baseline `none`:
  - `opcode=0.4125`
  - `step=0.4362`
  - `write_var=0.8180`
  - `write_count=0.8701`
- `drop_memops`:
  - `opcode=0.4354`
  - `step=0.4338`
- `opcode_step_plus_sizes`:
  - `opcode=0.4190`
  - `step=0.4017`
- `opcode_step_only`:
  - `opcode=0.3927`
  - `step=0.3747`

Read:

- performance does not collapse when memops, branch, and delta/output features
  are removed
- coarse execution dynamics survive even under `opcode_step_plus_sizes`
- even `opcode_step_only` remains materially above zero

But rollout still fails immediately.

Semi-open-loop rollout on `64` val sequences:

- `predicted_all_oracle_sizes`:
  - `opcode=0.2863`
  - `step=0.2052`
  - `full_trace_exact_fraction=0.0`
  - `first_failure_step_mean=1.0`
- `predicted_opcode_step_only_oracle_sizes`:
  - `opcode=0.2700`
  - `step=0.2119`
  - `full_trace_exact_fraction=0.0`
  - `first_failure_step_mean=1.0`

So the current controller is:

- stronger than a pure trace memorizer
- weaker than a stable learned executor
- best described as a held-out execution-dynamics learner with poor open-loop
  stability

This means the transfer story should currently be phrased as:

- execution-dynamics supervision transfers into language modeling

not yet:

- a learned executor transfers into language modeling

## Second Verification Result: Predicted Size State

Patched export with explicit `stack_depth` / `env_size` heads:

- [trace-pretrain-export-mixed-memops-verify-v2.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_031502_mlx-hardmax-trace-pretrain-export-memops-verify-v2/trace_execution_verification/trace-pretrain-export-mixed-memops-verify-v2.json)

What changed:

- next `stack_depth` is now predicted
- next `env_size` is now predicted
- rollout can now test true predicted-size carry instead of only oracle size carry

Teacher-forced held-out dynamics improved substantially:

- `none`:
  - `opcode=0.6238`
  - `step=0.6664`
  - `stack_depth=0.6740`
  - `env_size=0.6252`
- `opcode_step_plus_sizes`:
  - `opcode=0.5007`
  - `step=0.4722`
  - `stack_depth=0.5023`
  - `env_size=0.6265`
- `opcode_step_only`:
  - `opcode=0.4451`
  - `step=0.4331`
  - `stack_depth=0.5130`
  - `env_size=0.4521`

So the transition model is materially better than the first verifier-compatible
export, and the missing machine-state sizes were genuinely learnable.

But the main bottleneck did not move:

- `predicted_all`: `first_failure_step_mean=1.0`
- `predicted_opcode_step_plus_sizes`: `first_failure_step_mean=1.0`
- `predicted_opcode_step_only`: `first_failure_step_mean=1.0`

Even the oracle-size carry variants still have:

- `predicted_all_oracle_sizes`: `first_failure_step_mean=1.0`
- `predicted_opcode_step_only_oracle_sizes`: `first_failure_step_mean=1.0`

So the size heads improved per-field rollout accuracy, but they did **not**
fix immediate open-loop instability.

Updated conclusion:

- the current controller is a stronger held-out transition learner than before
- predicting missing machine-state sizes helps
- but the real next bottleneck is no longer missing state heads
- it is rollout robustness under self-fed predicted inputs

That points directly to the next branch:

- short-horizon rollout-consistency training
- or another explicit anti-drift objective on self-fed predicted traces

## Active Follow-On: Short-Horizon Rollout Consistency

The pretrainer now has an explicit self-fed rollout objective in
[train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py):

- `--rollout-consistency-weight`
- `--rollout-consistency-horizon`
- `--rollout-consistency-mode`

Implementation details:

- keep the teacher-forced objective unchanged
- iteratively build self-fed input batches using the same rollout semantics as the verifier
- add a weighted short-horizon prediction loss on those self-fed traces
- default verifier-aligned mode for the first branch is `predicted_all`

Shared rollout construction now lives in:

- [execution_trace_verifier.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_verifier.py)

so the trainer and verifier are using the same transition semantics.

The first controlled branch is now staged and queued:

- spec:
  - [20260404_223013_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223013_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3.json)
- generated run bundle:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/manifest.json)

Arms:

1. `trace-pretrain-export-mixed-memops-baseline-v3`
2. `trace-pretrain-export-mixed-memops-rollout-h2w025-v3`

Decision target:

- rerun the same execution verifier on both arms
- keep the branch if `first_failure_step_mean` moves materially above `1.0`
- promote strongly if the rollout arm improves both:
  - teacher-forced ablated accuracy
  - and open-loop rollout depth

## Result: v3 Rollout-Consistency Branch

Recovered + verified iteration:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/manifest.json)
- baseline verify:
  - [trace-pretrain-export-mixed-memops-baseline-v3.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/trace_execution_verification/trace-pretrain-export-mixed-memops-baseline-v3.json)
- rollout verify:
  - [trace-pretrain-export-mixed-memops-rollout-h2w025-v3.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/trace_execution_verification/trace-pretrain-export-mixed-memops-rollout-h2w025-v3.json)

Headline:

- the rollout-consistency arm did **not** move `first_failure_step_mean`
- both baseline and rollout arm remain at:
  - `predicted_all first_failure_step_mean = 1.0`
  - `predicted_all_oracle_sizes first_failure_step_mean = 1.0`

Teacher-forced held-out accuracy actually regressed under the rollout loss:

- baseline `teacher_forced none`:
  - `opcode=0.5858`
  - `step=0.7240`
  - `stack_depth=0.5726`
  - `env_size=0.8511`
- rollout `teacher_forced none`:
  - `opcode=0.4310`
  - `step=0.4558`
  - `stack_depth=0.6170`
  - `env_size=0.5976`

Semi-open-loop `predicted_all` was mixed but not enough:

- baseline:
  - `opcode=0.2028`
  - `step=0.2028`
  - `stack_depth=0.4083`
  - `env_size=0.3153`
- rollout:
  - `opcode=0.2394`
  - `step=0.1829`
  - `stack_depth=0.3940`
  - `env_size=0.6338`

Interpretation:

- a naive 2-step rollout-consistency loss at weight `0.25` is too blunt
- it shifts some per-field rollout behavior, especially `env_size`
- but it does not buy even a single additional exact self-fed step
- and it degrades one-step teacher-forced dynamics too much

So this branch should currently be read as:

- useful negative result
- not yet the right anti-drift fix

Next likely fixes are:

1. lower rollout weight sharply (`0.05` / `0.10`)
2. restrict rollout loss to a subset of fields or later steps
3. try scheduled sampling / mixed-input corruption instead of full predicted-all replacement

## Active Follow-On: Scheduled Sampling / Mixed Inputs

The next anti-drift branch is now switched from explicit rollout loss to
scheduled-sampling-style input mixing.

Implementation:

- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)
  now supports:
  - `--scheduled-sampling-prob`
  - `--scheduled-sampling-mode`
  - `--scheduled-sampling-prefix-keep`

Mechanism:

- run one detached teacher-forced forward on the sampled batch
- build a verifier-aligned predicted-input batch
- replace only a small fraction of future inputs with model predictions
- keep the main training objective as the standard next-step teacher-forced loss

The first light branch is staged and running:

- spec:
  - [20260404_225508_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_225508_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4.json)
- generated bundle:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_035543_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4/manifest.json)

Arms:

1. `trace-pretrain-export-mixed-memops-baseline-v4`
2. `trace-pretrain-export-mixed-memops-schedmix-p010-opstepsize-v4`

Design choice:

- low `0.10` replacement probability
- `predicted_opcode_step_plus_sizes` only
- no explicit rollout-consistency loss

This is intentionally lighter than the failed v3 branch.

Current planning status:

- let v4 finish because it is already queued
- but do not expand more anti-drift sub-branches unless v4 is surprisingly
  positive
- main priority returns to:
  1. `H15b` SimVQ + NextLat anti-collapse
  2. `H16` residual mirror / error-memory
  3. `H12` register/curriculum supervision reboot
