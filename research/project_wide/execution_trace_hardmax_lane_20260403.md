# Execution Trace Hardmax Lane

This note records the shift from shallow structural supervision to execution-state supervision for the hardmax/controller line.

## Why the current hardmax lane is stuck

The current hardmax/controller experiments established three things:

1. A structural side-path is real.
   - `structonly` beat control in the step-matched AB3 run.
   - The best anti-collapse branch in AB4 reached `1.86285218` BPB.

2. The current router is not real.
   - `fullroute` lost in AB2 and AB3.
   - The routed branch effectively learned a nearly constant budget.

3. The hard discrete controller is collapsing.
   - hard state usage concentrated almost entirely in one state
   - the surviving signal lived more in the soft distribution and the adapter path than in the hard argmax path

The likely reason is simple: the current targets are too easy.

The controller is trained to predict:
- operator class
- token class
- boundary strength
- quote flag

One state is sufficient for most positions under that objective. The anti-collapse loss helps, but it is fighting the target rather than being supported by it.

## External work that points to the right supervision

The most relevant external pattern is not “more code tokens.” It is explicit intermediate execution state.

- [Emergent Representations of Program Semantics in Language Models Trained on Programs](https://arxiv.org/abs/2305.11169)
  Shows that plain code-trained LMs can carry latent program state in hidden representations.

- [Teaching Large Language Models to Reason about Code Execution (NExT)](https://arxiv.org/abs/2404.14662)
  Uses structured code-execution supervision rather than only source/output pairs.

- [Chain of Execution Supervision Promotes General Reasoning in Large Language Models](https://arxiv.org/abs/2510.23629)
  Argues that exposing intermediate execution traces is a better training signal than raw input/output alone.

- [Towards a Neural Debugger for Python](https://arxiv.org/abs/2603.09951)
  Explicitly treats Python execution traces as training data for neural reasoning/debugging.

- [wasm-interpreter-transformer](https://huggingface.co/eastlondoner/wasm-interpreter-transformer)
  Concrete evidence that a transformer can be trained around explicit executable state transitions rather than only surface text.

These all point in the same direction:

- code/source-only training forces the model to infer execution from weak supervision
- trace/state supervision provides the intermediate machine state directly
- that is exactly the kind of deterministic supervision a hardmax path should receive

## Current thesis

The hardmax/controller path should not be trained as a generic structural tagger. It should be trained as an execution-state model.

That means:

- not just operator labels
- not just boundaries
- not just “is this token easy”

Instead, the hardmax path should learn typed state transitions such as:

- load
- store
- arithmetic step
- compare
- branch taken / not taken
- stack state
- memory delta
- output delta
- halt / exception state

This gives the controller multiple real states to occupy. A single state is no longer sufficient.

## First concrete artifact

Implemented:

- [execution_trace_dataset.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_dataset.py)
- [execution_trace_pretrain_dataset.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_pretrain_dataset.py)
- [generate_execution_trace_corpus.py](/home/zaytor/transformer_research/parameter-golf/tools/generate_execution_trace_corpus.py)
- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)
- [mlx_hardmax_trace_pretrain_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_pretrain_smoke.example.json)
- [test_execution_trace_dataset.py](/home/zaytor/transformer_research/parameter-golf/tests/test_execution_trace_dataset.py)
- [test_execution_trace_pretrain_dataset.py](/home/zaytor/transformer_research/parameter-golf/tests/test_execution_trace_pretrain_dataset.py)

The first version is intentionally narrow:

- tiny imperative language
- stack VM IR
- deterministic execution
- explicit aligned views

The goal is not “full Python.” The goal is to create a clean synthetic execution substrate with exact traces.

## Trace schema

Each example currently contains:

- `source.text`
  Tiny imperative surface language

- `source.python_like_text`
  Python-style rendering of the same program

- `source.ast`
  Structured AST for the tiny language

- `source.python_ast_dump`
  Python AST dump for the Python-like rendering

- `ir.instructions`
  Typed stack-VM instructions

- `ir.opcode_text`
  Text rendering of the instruction sequence

- `trace`
  Per-step execution events with:
  - `pc`
  - `opcode`
  - `stack_before`
  - `stack_after`
  - `env_before`
  - `env_after`
  - `env_delta`
  - `memory_reads`
  - `memory_writes`
  - `output_before`
  - `output_after`
  - `output_delta`
  - `branch_taken`
  - `step_type`
  - `text`

- `views.trace_text`
  Human-readable step log

- `views.memory_trace_text`
  Human-readable memory/read-write log

- `python_runtime.events`
  A second aligned trace view from executing the Python-like rendering under `sys.settrace`

- `views.python_runtime_trace_text`
  Human-readable Python runtime trace

- `final`
  Final environment, output, and halt reason

## Why this is better than the old hardmax targets

The old targets are mostly low-entropy tags.
The new targets are stateful.

Execution traces force the model to differentiate:

- current control state
- current dataflow state
- memory reads vs writes
- temporary stack state
- branch outcome

This should create real pressure for multi-state specialization.

## Intended training use

This generator is not the final training setup. It is the first synthetic supervision source.

Expected phases:

1. Pretrain the hardmax/controller path on trace prediction.
   - next step type
   - next memory delta size
   - next read/write variable
   - next control-flow transition
   - next output event
   - maybe next environment state or compressed state view

2. Reattach that controller to the language model as `structonly`.

3. Re-run the old hardmax diagnostics:
   - state usage
   - confidence variability
   - residual ACF
   - boundary-binned eval

4. Only revisit routing once the controller is materially non-degenerate.

The first concrete training hook now exists:

- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)

It is intentionally narrow:

- controller-only
- execution-state targets
- no routing
- no full LM trunk

This script uses the encoded trace dataset to train the existing `HardmaxStructuralController` on:

- next opcode
- next step type
- next read variable
- next write variable
- next branch outcome
- next env-delta bucket
- next output flag

plus controller regularization and confidence calibration.

## Immediate next work

1. Expand the tiny language coverage.
   - nested branches
   - bounded loops
   - explicit temporary reuse
   - optional error traces

2. Add richer machine-state targets.
   - stack depth buckets
   - read/write address classes
   - branch target classes

3. Add a second trace family.
   - restricted Python subset or direct Python-bytecode tracing

4. Build a hardmax pretrain smoke.
   - controller trained only on trace supervision
   - no routing
   - then transfer back into the LM branch

5. Add more trace families.
   - nested control flow
   - bounded exceptions / error traces
   - restricted Python bytecode / line traces
   - eventually WASM or VM-level memory operations

## Transfer seam now implemented

The LM trainer now supports direct controller initialization from a trace-pretrained NPZ artifact.

Relevant files:

- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)
  - exports controller-only init artifacts with LM-compatible parameter names

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)
  - loads `HARDMAX_STRUCT_INIT_PATH`
  - supports `HARDMAX_STRUCT_INIT_MODE=core|state_book|full`
  - first intended transfer mode is `core`

The first transfer plan is intentionally narrow:

- keep the LM branch in `structonly`
- keep routing off
- align the LM hardmax state dim to the trace-pretrain state dim
- compare random-init vs trace-pretrained init before touching architecture or routing again

Templates staged for this:

- [mlx_hardmax_trace_pretrain_export.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_pretrain_export.example.json)
- [mlx_hardmax_trace_transfer_smoke.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_transfer_smoke.example.json)

## First exported controller artifact

The first reusable trace-pretrained controller artifact is now staged locally:

- [trace_pretrain_export_8state_hardmax_controller_init.npz](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_transfer_assets/trace_pretrain_export_8state_hardmax_controller_init.npz)
- [trace_pretrain_export_8state.summary.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_transfer_assets/trace_pretrain_export_8state.summary.json)

Key export metrics:

- `val_loss`: `3.6189`
- `opcode_acc`: `0.5666`
- `confidence_mean`: `0.3272`
- `hard_usage_peak_frac`: `0.4479`
- `soft_usage_perplexity`: `6.7070`

This is not yet a language-model result. It is the first exported controller init that stayed materially alive under execution-trace supervision and could be copied back into the repo for LM transfer.

## New verification branch: did it learn execution dynamics or only trace completion?

That question is now explicit:

- [20260404_trace_execution_verification_lane.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_trace_execution_verification_lane.md)

Current read:

- held-out `val_*` metrics already show next-step generalization on unseen
  synthetic programs
- but the current pretrainer consumes rich current-trace inputs, not raw
  program bytecode alone
- so the strongest supported claim today is "held-out execution-dynamics model,"
  not yet "verified learned executor"

New tooling now exists to resolve that distinction on the next export:

- [verify_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/verify_hardmax_execution_trace.py)
- [execution_trace_verifier.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_verifier.py)

And the trace pretrainer now saves a full checkpoint by default, so future
exports can be verified directly instead of only transferred.

First verifier-compatible export result:

- [trace-pretrain-export-mixed-memops-verify.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_023107_mlx-hardmax-trace-pretrain-export-memops-verify/trace_execution_verification/trace-pretrain-export-mixed-memops-verify.json)

Read:

- teacher-forced held-out transition prediction survives heavy input ablation
- `opcode_step_plus_sizes` remains alive:
  - `opcode=0.4190`
  - `step=0.4017`
- `opcode_step_only` degrades but does not collapse:
  - `opcode=0.3927`
  - `step=0.3747`
- semi-open-loop rollout is not stable:
  - `full_trace_exact_fraction=0.0`
  - `first_failure_step_mean=1.0`

Conclusion:

- the current controller is not yet a stable learned executor
- it is better described as a held-out execution-dynamics model whose one-step
  generalization is real, but whose open-loop rollout stability is still poor

Second verifier-compatible export result:

- [trace-pretrain-export-mixed-memops-verify-v2.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_031502_mlx-hardmax-trace-pretrain-export-memops-verify-v2/trace_execution_verification/trace-pretrain-export-mixed-memops-verify-v2.json)

What changed:

- explicit `stack_depth` and `env_size` prediction heads

What improved:

- teacher-forced held-out transition prediction improved sharply
  - baseline `opcode` rose from `0.4125` to `0.6238`
  - baseline `step` rose from `0.4362` to `0.6664`
  - new size accuracies landed at:
    - `stack_depth=0.6740`
    - `env_size=0.6252`

What did not improve enough:

- open-loop rollout still has `first_failure_step_mean=1.0`
  even when sizes are predicted and carried

So the bottleneck has narrowed:

- it is no longer “the model never learned machine-state size”
- it is now “the model is still not robust under self-fed predicted traces”

That makes the next execution-verification follow-on straightforward:

- short-horizon rollout-consistency / anti-drift training

## Active LM transfer sweep

The current live transfer sweep is:

- [20260404_034412_mlx-hardmax-trace-transfer-smoke](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_034412_mlx-hardmax-trace-transfer-smoke)

Arms:

- control
- `structonly` `n8,d32` random-init
- `structonly` `n8,d32` trace-pretrained `core`
- `structonly` `n8,d32` trace-pretrained `state_book`
- best old `structonly` `n8,d64` anti-collapse benchmark

This is the first direct test of the MCTS critical path:

- trace-pretrained controller -> LM transfer -> BPB/state-usage comparison

## Example

Generate a small corpus:

```bash
python3 /home/zaytor/transformer_research/parameter-golf/tools/generate_execution_trace_corpus.py \
  --num-examples 128 \
  --seed 7 \
  --task-family mixed \
  --output /tmp/execution_trace_corpus.jsonl
```

Inspect a sample:

```bash
python3 /home/zaytor/transformer_research/parameter-golf/tools/generate_execution_trace_corpus.py \
  --num-examples 1 \
  --seed 7 \
  --pretty-first
```

Run the first controller-only pretrain smoke on an MLX-capable machine:

```bash
python3 /home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py \
  --generate-examples 2048 \
  --seed 7 \
  --steps 200 \
  --batch-size 32 \
  --num-states 8 \
  --summary-out /tmp/hardmax_trace_pretrain_smoke.json
```

Or stage a small cluster sweep from:

```bash
python3 /home/zaytor/transformer_research/parameter-golf/tools/prepare_mlx_sweep.py \
  /home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_pretrain_smoke.example.json
```

## Current decision

Pause new hardmax routing experiments until the controller has seen real execution-state supervision.

The right next test is not a new router. It is a trace-pretrained controller.

## First smoke result

The first controller-only trace-pretrain smoke is staged in:

- [20260403_hardmax_trace_pretrain_smoke](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_pretrain_smoke)

The live three-arm result is already informative:

- `1-state` baseline:
  - `val_loss 4.2226`
  - `val_opcode_acc 0.4180`
  - `val_write_acc 0.7969`
  - `val_conf 1.0000`
  - `val_hard_peak 1.0000`
  - `val_soft_ppl 1.0000`

- `8-state` baseline:
  - `val_loss 4.2307`
  - `val_opcode_acc 0.4401`
  - `val_write_acc 0.7969`
  - `val_conf 0.2805`
  - `val_hard_peak 0.4564`
  - `val_soft_ppl 7.0810`

- `8-state + stronger anti-collapse`:
  - `val_loss 4.4405`
  - `val_opcode_acc 0.4232`
  - `val_write_acc 0.7969`
  - `val_conf 0.2476`
  - `val_hard_peak 0.5066`
  - `val_soft_ppl 7.1388`

This is the first positive sign that execution-state supervision changes the controller dynamics materially:

- the `1-state` run stays totally collapsed
- the `8-state` execution-trace run does **not** collapse immediately
- hard-state peak usage is far below `1.0`
- soft usage perplexity stays near the full state count

The anti-collapse variant did not win on this first smoke, but it also stayed non-degenerate. So the current read is:

- execution-trace supervision is already a better target family than the old structural labels
- the controller can stay alive under this target
- the next step is not more regularization first, but transfer back into the `structonly` LM lane

## Current branch state

The first LM transfer sweep is now active from:

- [20260404_034412_mlx-hardmax-trace-transfer-smoke](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_034412_mlx-hardmax-trace-transfer-smoke)

It compares:

- control
- `structonly` `n8,d32` random init
- `structonly` `n8,d32` trace-pretrained `core`
- `structonly` `n8,d32` trace-pretrained `state_book`
- prior `n8,d64` anti-collapse reference

In parallel, the next controller-only sweep is staged and launched from:

- [mlx_hardmax_trace_pretrain_ab2_nested_memops.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_pretrain_ab2_nested_memops.example.json)
- [20260403_230123_mlx-hardmax-trace-pretrain-ab2-nested-memops](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_230123_mlx-hardmax-trace-pretrain-ab2-nested-memops)

This second sweep does not change the architecture. It asks whether richer trace targets improve controller quality before transfer:

- nested control flow is now stable under the default trace-step budget
- `mixed` generation now includes the nested family
- new count targets supervise next-step memory-read and memory-write bucket counts
- the new ablation varies stronger memory-operation supervision and nested-only trace curricula

One backward-pass revision is already staged from the early live read:

- nested-only may be too narrow as a curriculum even when the target family is richer
- the next cleaner test is a **weighted mixed curriculum**, not a hard switch to nested-only

That support is now implemented via `task-family-mixture`, and the follow-up sweep spec is staged at:

- [mlx_hardmax_trace_pretrain_ab3_weighted_mix.example.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/templates/mlx_hardmax_trace_pretrain_ab3_weighted_mix.example.json)

## Live backward-pass read

The richer memop sweep at:

- [20260403_230123_mlx-hardmax-trace-pretrain-ab2-nested-memops](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_230123_mlx-hardmax-trace-pretrain-ab2-nested-memops)

already gives a useful split:

- `mixed baseline`:
  - `val_loss 4.8416`
  - `val_opcode_acc 0.5145`
  - `val_hard_peak 0.4936`
  - `val_soft_ppl 5.8491`

- `mixed + stronger memops`:
  - `val_loss 4.5144`
  - `val_opcode_acc 0.5486`
  - `val_hard_peak 0.5472`
  - `val_soft_ppl 5.9652`

- `nested-only + memops`:
  - `val_loss 4.2866`
  - `val_opcode_acc 0.7701`
  - `val_hard_peak 1.0000`
  - `val_soft_ppl 1.1353`

- `nested-only + memops + anti-collapse`:
  - `val_loss 4.7582`
  - `val_opcode_acc 0.6218`
  - `val_hard_peak 0.9886`
  - `val_soft_ppl 1.1795`

Interpretation:

- richer memory-operation supervision is clearly helping on the mixed curriculum
- nested-only traces teach the trace task well, but they almost fully collapse the controller again
- stronger anti-collapse does not repair that collapse enough
- the next good branch is a **weighted mixed curriculum** that biases toward nested traces without eliminating the rest of the trace family

## Weighted-mix follow-up

The weighted mixed-curriculum sweep was then staged at:

- [20260403_232141_mlx-hardmax-trace-pretrain-ab3-weighted-mix](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_232141_mlx-hardmax-trace-pretrain-ab3-weighted-mix)

Observed outcomes from the live dispatch:

- `mixed + memops`:
  - `val_loss 4.6754`
  - `val_opcode_acc 0.5790`
  - `val_hard_peak 0.6044`
  - `val_soft_ppl 4.1140`

- `mixed + memops + nestedx2`:
  - `val_loss 5.7517`
  - `val_opcode_acc 0.3633`
  - `val_hard_peak 0.4940`
  - `val_soft_ppl 6.0348`

- `mixed + memops + nestedx4`:
  - `val_loss 4.6981`
  - `val_opcode_acc 0.6006`
  - `val_hard_peak 0.6939`
  - `val_soft_ppl 3.3375`

- `mixed + memops + nestedx2 + anti-collapse`:
  - `val_loss 5.2298`
  - `val_opcode_acc 0.4293`
  - `val_hard_peak 0.6101`
  - `val_soft_ppl 4.1644`

Current read:

- weighted nested curricula did **not** beat the plain mixed+memops baseline
- `nestedx4` is the strongest of the weighted variants, but it is still slightly worse on loss and more collapse-prone than plain mixed+memops
- the curriculum branch should be pruned for now
- the next forward move is to export the improved mixed+memops controller and test transfer, rather than doing more curriculum search

## Parallel runtime leg

In parallel with the transfer gate, the first sequential microstep refinement branch is now staged:

- note: [20260403_hardmax_async_refinement_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260403_hardmax_async_refinement_leg.md)
- generated sweep: [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_050137_mlx-hardmax-trace-transfer-microsteps-smoke/manifest.json)

This is the `G1` ablation from the MCTS tree:

- same trace-core transfer init
- same `d32` controller capacity
- vary only `HARDMAX_STRUCT_FAST_REFINEMENT_STEPS` across `1 / 2 / 4`

That keeps the first runtime-leg test focused on the modeling question before any helper/runtime overlap work.

## Transfer gate update

The matched-capacity LM transfer sweep at:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_034412_mlx-hardmax-trace-transfer-smoke/manifest.json)

has now produced a finished `d32` random-init anchor.

Observed exact result from the live run:

- `structonly-step800-n8-d32-random`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.85942869`

This matters for Gate 1 because it is much better than the plain control anchor:

- `control-step800`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.88888776`

Current read:

- matched-capacity `d32` structural conditioning is clearly real
- the transfer question narrowed from "does transfer help at all?" to "which transfer slice helps?"
- `trace-statebook` is now the first positive transfer result

Observed exact result from the same transfer sweep:

- `structonly-step800-n8-d32-trace-statebook`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.83989451`

That beats:

- `structonly-step800-n8-d32-random`
  - `1.85942869`
- `structonly-step800-n8-d64-anticollapse`
  - `1.86368503`
- `control-step800`
  - `1.88888776`

Interpretation:

- trace pretraining now has a real positive transfer result into the LM
- the useful transferred object is, at minimum, the discrete state book
- the controller core is not yet validated as the best transfer slice

So the next A-branch exploit should bias toward:

- `A2`: state-book transfer
- `A4`: freeze/unfreeze around state-book transfer
- better exported controllers from the stronger memops pretrain family

## State-book freeze follow-up

To exploit the first positive transfer result, a dedicated follow-up sweep is now staged at:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/manifest.json)

This sweep keeps the transfer branch narrow and matched at `d32`:

- `structonly-step800-n8-d32-random`
- `structonly-step800-n8-d32-trace-statebook`
- `structonly-step800-n8-d32-trace-statebook-freeze100`
- `structonly-step800-n8-d32-trace-statebook-freeze200`

The implementation change is trainer-only:

- [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py)

New knob:

- `HARDMAX_STRUCT_STATEBOOK_FREEZE_STEPS`

Behavior:

- zeroes gradients for `hardmax_structural_controller.state_book`
- only for the first `N` optimizer steps
- leaves the rest of the controller and trunk trainable

Why this is the right next branch:

- `state_book` transfer is the first proven positive transfer slice
- `trace-core` has not yet shown a win
- so the next clean test is whether preserving the transferred discrete embedding prior during early LM adaptation improves or hurts the winning transfer

Current blocker remains unchanged:

- the winning `trace-statebook` artifact has not yet synced back locally
- so the existing diagnostics wrapper cannot yet be run on the positive transfer result

Observed exact finals from the live freeze sweep:

- `structonly-step800-n8-d32-trace-statebook-freeze100`
  - `final_int8_zlib_roundtrip_exact val_loss 3.10402405`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.83837788`

- `structonly-step800-n8-d32-trace-statebook-freeze200`
  - `final_int8_zlib_roundtrip_exact val_loss 3.09137359`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.83088557`

Observed exact final from the follow-up freeze-window refinement:

- `structonly-step800-n8-d32-trace-statebook-freeze300`
  - `final_int8_zlib_roundtrip_exact val_loss 3.07663979`
  - `final_int8_zlib_roundtrip_exact val_bpb 1.82215938`

Interpretation:

- short early freezing does not hurt the positive transfer
- a `200`-step freeze is materially better than the original unfrozen `state_book` transfer
- a `300`-step freeze improves further
- current best hardmax LM result is now `state_book + freeze300`

New ranking among matched-capacity LM runs:

- `state_book + freeze300`
  - `1.82215938`
- `state_book + freeze200`
  - `1.83088557`
- `state_book + freeze100`
  - `1.83837788`
- `state_book`
  - `1.83989451`
- `d32` random-init structonly
  - `1.85942869`
- old `d64` anti-collapse`
  - `1.86368503`
- control
  - `1.88888776`

So the next A-branch exploit has narrowed again:

- keep `state_book`
- keep the early warm-start freeze
- export stronger trace-pretrained state books from the best memops family when available
- do not prioritize `trace-core` or router work

## Controller diagnostics on recovered artifacts

The missing-copyback blocker turned out to be infrastructure, not missing files:

- `dispatch.sh` stages jobs onto Minis but does not copy artifacts back

Recovered artifacts from the remote job dirs were copied back locally into:

- [artifacts](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/artifacts)

Controller diagnostics were then run remotely on the Minis and copied back into:

- [hardmax_transfer_diagnostics/controller](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_073655_mlx-hardmax-trace-transfer-statebook-freeze/hardmax_transfer_diagnostics/controller)

Key read from the three-way comparison:

- `d32` random:
  - mean NLL `3.11617`
  - max-state fraction `1.00000`
  - usage perplexity `1.00000`
  - self-transition fraction `1.00000`
  - confidence std `0.10026`
  - `corr(confidence, NLL) = -0.17620`

- `state_book`:
  - mean NLL `3.07982`
  - max-state fraction `0.99975`
  - usage perplexity `1.00233`
  - self-transition fraction `0.99950`
  - confidence std `0.00184`
  - `corr(confidence, NLL) = 0.00161`

- `state_book + freeze200`:
  - mean NLL `3.06422`
  - max-state fraction `0.99935`
  - usage perplexity `1.00545`
  - self-transition fraction `0.99879`
  - confidence std `0.08167`
  - `corr(confidence, NLL) = -0.13741`

Interpretation:

- the original `state_book` transfer improved BPB while saturating the soft controller signal almost completely
- `freeze200` improved BPB further while restoring useful soft variation in confidence/budget
- the hard path is **still almost fully collapsed** in both transfer variants
- so the current gain is best read as:
  - transferred discrete embedding prior helps
  - freezing it early preserves a better soft control surface
  - but we still do **not** have a healthy dynamic hard state partition inside the LM

This means the next search move should stay narrow:

- exploit `state_book + freeze`
- use stronger trace-pretrained state books when available
- do not mistake the BPB gain for full controller revival

## Freeze-window refinement sweep

To refine the best current transfer schedule, a follow-up sweep is now staged at:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_181241_mlx-hardmax-trace-transfer-statebook-freeze-refine/manifest.json)

Run note:

- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_181241_mlx-hardmax-trace-transfer-statebook-freeze-refine/README.md)

Queue launcher:

- session `58763`

Arms:

- `structonly-step800-n8-d32-trace-statebook`
- `structonly-step800-n8-d32-trace-statebook-freeze200`
- `structonly-step800-n8-d32-trace-statebook-freeze300`
- `structonly-step800-n8-d32-trace-statebook-freeze400`

This is intentionally not a new architecture branch. It is a narrow exploit around the current best result:

- does the optimum freeze window stay near `200` steps
- or does a longer warm-start preserve the useful soft control surface better?

## Artifact recovery helper

The copyback failure is now addressed by a dedicated helper:

- [recover_iteration_cluster_artifacts.py](/home/zaytor/transformer_research/parameter-golf/tools/recover_iteration_cluster_artifacts.py)

Current read:

- it successfully recovered the earlier freeze-sweep artifacts into the generated iteration dir
- it has not found finished artifacts yet for the freeze-refine sweep
- it also did not recover the earlier transfer-smoke sweep on the latest re-run, so some historical artifacts may already be gone from Mini job dirs

Implication:

- exact BPB decisions can still come from logs
- but controller diagnostics remain gated on either recovered artifacts or manual remote evaluation

## Attention-shaping successor branch

The next architecture leg is now implemented and staged:

- [20260404_hardmax_attention_shaping_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_attention_shaping_leg.md)

New trunk modes in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py):

- `HARDMAX_STRUCT_CONDITION_MODE=residual`
- `HARDMAX_STRUCT_CONDITION_MODE=q_bias`
- `HARDMAX_STRUCT_CONDITION_MODE=q_bias_temp`

First staged sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_191323_mlx-hardmax-trace-transfer-attn-conditioning-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_191323_mlx-hardmax-trace-transfer-attn-conditioning-smoke/README.md)

This branch is staged but intentionally not launched yet.

Reason:

- the best exact transfer anchor is now `state_book + freeze300`
- the next branch is no longer more freeze tuning first
- it is:
  - H15a naive anti-collapse as the baseline read for that family
  - H15b `SimVQ + NextLat` is now a closed negative result in its current form
  - H16 residual feedback / error-memory is the next active principal variation
  - with H2 attention-shaping still the first conditioning branch to scale if it wins

## New supervision branches

The lane has now widened from transfer exploitation into supervision-density tests.

### H11. Face / mirror operator-state trace

Branch note:

- [20260404_hardmax_face_mirror_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_hardmax_face_mirror_leg.md)

First real trace artifact:

- [face_trace_freeze200_negation.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223900_hardmax-face-trace-smoke/face_trace_freeze200_negation.json)

Current read:

- `has_hardmax_structural=true`
- soft confidence/budget surface is alive
- hard state is still collapsed:
  - `used_states=1`
  - `max_state_fraction=1.0`

### H12. Compiled-state supervision density

Branch note:

- [20260404_compiled_state_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_compiled_state_supervision_leg.md)

Live smoke:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_222804_mlx-h12-multihorizon-statebook-freeze300-smoke/manifest.json)

Question:

- does train-time multi-horizon supervision provide enough extra supervision bits to improve on top of `state_book + freeze300`?

Updated read:

- the live H12 smoke is now best treated as a probe
- the next real H12 build should be register/curriculum supervision, not more naive small-model horizon heads

### H13. Distribution-shape supervision

Branch note:

- [20260404_distribution_shape_supervision_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260404_distribution_shape_supervision_leg.md)

Live smoke:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/manifest.json)

Question:

- does EMA-teacher KL on the full token distribution beat plain one-hot next-token supervision on the same anchor?

Updated read:

- the live EMA-KL smoke is now H13a
- the next real H13 follow-on should be calibrated uncertainty distillation, not plain sharp KL alone

### H15. Statebook anti-collapse on the `freeze300` anchor

Branch note:

- [20260405_hardmax_statebook_anticollapse_leg.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/20260405_hardmax_statebook_anticollapse_leg.md)

Generated sweep:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_011602_mlx-hardmax-statebook-anticollapse-freeze300-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_011602_mlx-hardmax-statebook-anticollapse-freeze300-smoke/README.md)

Reason:

- the freeze-family backfill now says the transfer object is real, but the live discrete controller is still almost fully collapsed
- so the next branch is no longer “more transfer archaeology”
- it is explicit anti-collapse training on top of the best exact anchor

Updated branch split:

- H15a naive anti-collapse
- H15b `SimVQ + NextLat`

The purpose of H15a is to test whether occupancy / commitment / transition pressure alone is enough.

The purpose of H15b is to attack the deeper bottleneck directly:

- codebook optimization
- next-state latent dynamics

New trainer knobs in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py):

- `HARDMAX_STRUCT_STATE_USAGE_ENTROPY_WEIGHT`
- `HARDMAX_STRUCT_STATE_COMMIT_WEIGHT`
- `HARDMAX_STRUCT_TRANSITION_BOUNDARY_WEIGHT`
- `HARDMAX_STRUCT_SIMVQ_ENABLED`
- `HARDMAX_STRUCT_NEXTLAT_WEIGHT`

First arms:

1. `freeze300-baseline`
2. `freeze300-usageH010`
3. `freeze300-commit010`
4. `freeze300-transbnd020`
5. `freeze300-usageH010-commit010`

H15b smoke is now live:

- [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_042706_mlx-hardmax-statebook-simvq-nextlat-freeze300-smoke/manifest.json)
- [README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_042706_mlx-hardmax-statebook-simvq-nextlat-freeze300-smoke/README.md)

Arms:

1. `freeze300-baseline`
2. `freeze300-simvq`
3. `freeze300-simvq-usageH010-commit010`
4. `freeze300-simvq-nextlat005`
5. `freeze300-simvq-usageH010-commit010-nextlat005`

Interpretation target:

- does shared codebook reparameterization alone revive the transferred state vocabulary?
- does explicit next-state prediction on the hardmax path create a richer controller without giving back the `freeze300` exact gain?

Current read:

- no
- the current H15b implementation is a negative result and should not be
  scaled as-is
- the next active BPB branch is now H16 residual feedback / error-memory on
  the same `freeze300` anchor

## Queue infrastructure upgrades

To reduce manual bookkeeping while these branches run, the queue/recovery path now has a structured observed-results layer:

- status helper:
  - [check_mlx_sweep_status.py](/home/zaytor/transformer_research/parameter-golf/tools/check_mlx_sweep_status.py)
  - now falls back to `launch.nohup.log` when `dispatch.out` is absent
  - prints observed `val_bpb` when an `observed_results.json` is present

- observed-results updater:
  - [update_iteration_observed_results.py](/home/zaytor/transformer_research/parameter-golf/tools/update_iteration_observed_results.py)
  - builds/refreshes per-iteration [observed_results.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_224347_mlx-h13-distribution-shape-statebook-freeze300-smoke/observed_results.json) from local status, recovered logs, and preserved prior observations

- recovery helper:
  - [recover_iteration_cluster_artifacts.py](/home/zaytor/transformer_research/parameter-golf/tools/recover_iteration_cluster_artifacts.py)
  - now refreshes `observed_results.json` after recovery by default
  - and can now optionally chain straight into:
    - [run_iteration_hardmax_transfer_diagnostics.py](/home/zaytor/transformer_research/parameter-golf/tools/run_iteration_hardmax_transfer_diagnostics.py)
    - with Mini-first remote analyzer mode for controller / causal / factor diagnostics

- future generated queue launchers:
  - [prepare_mlx_sweep.py](/home/zaytor/transformer_research/parameter-golf/tools/prepare_mlx_sweep.py)
  - now emit a stable `dispatch.out`
  - and run a best-effort post-run recover + observed-results refresh
  - and can now opt into post-run hardmax diagnostics from the sweep spec via:
    - `post_run_hardmax_diagnostics`
    - including Mini-first remote analyzer mode for controller / causal / factor outputs

- Mini-first saved-artifact analyzer wrapper:
  - [run_remote_saved_hardmax_analysis.py](/home/zaytor/transformer_research/parameter-golf/tools/run_remote_saved_hardmax_analysis.py)
  - now supports:
    - `controller`
    - `causal`
    - `factors`
    - `face`
  - this is now the preferred path for saved-artifact MLX analysis from the local Ubuntu shell

Implication:

- new branches should land with machine-readable status/finals much earlier
- and should depend less on manual queue-session note-taking

## Execution Verification Follow-On

The execution-verification read is now concrete:

- held-out transition prediction is real
- open-loop rollout still fails at step `1.0`
- predicting `stack_depth` / `env_size` improved one-step dynamics
- but did not fix self-fed drift

So the next execution branch is no longer more heads; it is rollout robustness.

Implemented follow-on:

- [train_hardmax_execution_trace.py](/home/zaytor/transformer_research/parameter-golf/tools/train_hardmax_execution_trace.py)
  - now supports short-horizon rollout-consistency loss on self-fed predicted traces
- [execution_trace_verifier.py](/home/zaytor/transformer_research/parameter-golf/execution_trace_verifier.py)
  - now provides the shared rollout input construction used by both training and verification

Queued controlled comparison:

- spec:
  - [20260404_223013_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_223013_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3.json)
- generated bundle:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/manifest.json)

Arms:

1. `baseline-v3`
2. `rollout-h2w025-v3`

The success criterion is simple:

- if rollout verification moves `first_failure_step_mean` materially above `1.0`,
  the controller has started to learn self-fed stability rather than only
  teacher-forced transition prediction

Result from the first controlled anti-drift branch:

- iteration:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/manifest.json)
- verification:
  - [baseline](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/trace_execution_verification/trace-pretrain-export-mixed-memops-baseline-v3.json)
  - [rollout](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_033054_mlx-hardmax-trace-pretrain-export-memops-rollout-verify-v3/trace_execution_verification/trace-pretrain-export-mixed-memops-rollout-h2w025-v3.json)

Read:

- `first_failure_step_mean` stayed at `1.0`
- naive rollout-consistency did **not** produce stable self-fed execution
- teacher-forced one-step dynamics regressed materially under the rollout arm

So the right update is:

- the anti-drift branch remains active
- but `predicted_all` 2-step consistency at weight `0.25` is the wrong shape
- next attempt should be lighter / mixed / scheduled-sampling-style, not a larger version of the same loss

Current queued follow-on:

- spec:
  - [20260404_225508_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_225508_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4.json)
- generated bundle:
  - [manifest.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260405_035543_mlx-hardmax-trace-pretrain-export-memops-schedmix-verify-v4/manifest.json)

Branch shape:

1. baseline-v4
2. schedmix-p010-opstepsize-v4

The new mechanism is scheduled-sampling-style input mixing on a small fraction
of future opcode/step/size inputs, while keeping the main objective teacher
forced.

Planning update:

- the execution lane has already answered its main question:
  teacher-forced held-out execution dynamics are real
- autonomous rollout robustness is now secondary future work, not the main
  project bottleneck
- v4 stays live as a light opportunistic follow-on, but the main priority
  shifts back to:
  - SimVQ + NextLat anti-collapse
  - residual mirror / error-memory
  - register/curriculum H12 reboot
