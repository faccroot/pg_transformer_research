# Hardmax Lane Results Review

This note consolidates the hardmax lane as of 2026-04-03 from completed artifacts, controller diagnostics, and trace-pretrain summaries.

It separates:

- results confirmed by saved files
- results currently confirmed only by live logs / lane notes
- diagnostics already run
- diagnostics still missing

## Evidence Sources

Primary saved artifacts:

- [hardmax_structural_controller_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/hardmax_structural_controller_lane_20260403.md)
- [execution_trace_hardmax_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/execution_trace_hardmax_lane_20260403.md)
- [20260403_hardmax_structural_diagnostics/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/README.md)
- [structonly_hardmax_controller_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/structonly_hardmax_controller_smoke.json)
- [fullroute_hardmax_controller_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/fullroute_hardmax_controller_smoke.json)
- [control_residual_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/control_residual_smoke.json)
- [structonly_residual_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_structural_diagnostics/structonly_residual_smoke.json)
- [20260403_hardmax_trace_pretrain_smoke/README.md](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_pretrain_smoke/README.md)
- [trace_pretrain_export_8state.summary.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260403_hardmax_trace_transfer_assets/trace_pretrain_export_8state.summary.json)
- [20260404_034412_mlx-hardmax-trace-transfer-smoke/configs](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260404_034412_mlx-hardmax-trace-transfer-smoke)

Secondary evidence:

- live run logs summarized into the two lane notes above
- currently active sessions for the transfer sweep and richer trace-pretrain sweep

## Run Ledger

### Structural LM lane

| Phase | Run | Metric | Result | Status |
|---|---|---:|---:|---|
| AB2 wallclock | control | final exact BPB | `1.72285249` | confirmed in lane note |
| AB2 wallclock | fullroute | final exact BPB | `1.98164836` | confirmed in lane note |
| AB3 step-match | control | final exact BPB | `1.88248754` | confirmed in diagnostics README |
| AB3 step-match | structonly | final exact BPB | `1.86855732` | confirmed in diagnostics README |
| AB3 step-match | fullroute | final exact BPB | `1.89072854` | confirmed in diagnostics README |
| AB4 collapse probe | control | final exact BPB | `1.88495975` | confirmed in diagnostics README |
| AB4 collapse probe | structonly `n1` | final exact BPB | `1.88201609` | confirmed in diagnostics README |
| AB4 collapse probe | structonly `n8` | final exact BPB | `1.87697373` | confirmed in diagnostics README |
| AB4 collapse probe | structonly `n8` anti-collapse | final exact BPB | `1.86285218` | confirmed in diagnostics README |
| AB5 adapter/anneal | static adapter | final exact BPB | `1.88671883` | live-log only, not archived locally |

### Trace/controller-only lane

| Phase | Run | Metric | Result | Status |
|---|---|---:|---:|---|
| Trace smoke | `1-state` | `val_loss` | `4.2226` | confirmed in README |
| Trace smoke | `1-state` | `val_hard_peak` | `1.0000` | confirmed in README |
| Trace smoke | `1-state` | `val_soft_ppl` | `1.0000` | confirmed in README |
| Trace smoke | `8-state` | `val_loss` | `4.2307` | confirmed in README |
| Trace smoke | `8-state` | `val_hard_peak` | `0.4564` | confirmed in README |
| Trace smoke | `8-state` | `val_soft_ppl` | `7.0810` | confirmed in README |
| Trace smoke | `8-state` anti-collapse | `val_loss` | `4.4405` | confirmed in README |
| Trace export | `8-state` | `val_loss` | `3.6189` | confirmed in summary JSON |
| Trace export | `8-state` | `opcode_acc` | `0.5666` | confirmed in summary JSON |
| Trace export | `8-state` | `hard_usage_peak_frac` | `0.4479` | confirmed in summary JSON |
| Trace export | `8-state` | `soft_usage_perplexity` | `6.7070` | confirmed in summary JSON |
| Trace memops AB2 | mixed baseline | `val_loss` | `4.8416` | lane note from live logs |
| Trace memops AB2 | mixed + memops | `val_loss` | `4.5144` | lane note from live logs |
| Trace memops AB2 | nested-only + memops | `val_loss` | `4.2866` | lane note from live logs |
| Trace memops AB2 | nested-only + memops | `val_hard_peak` | `1.0000` | lane note from live logs |
| Trace memops AB2 | nested-only + memops + anti-collapse | `val_loss` | `4.7582` | lane note from live logs |

## Confirmed Findings

### 1. The structural side-path is real, but the router is not.

This is the cleanest confirmed conclusion from the structural lane:

- `structonly` beat control in AB3 by `-0.01393` BPB
- `fullroute` lost to both control and `structonly`
- `fullroute` also lost badly in the 1-hour wallclock AB2

So the architecture split is:

- conditioning path: positive
- current confidence-to-budget routing rule: negative

### 2. The first hardmax LM win was mostly adapter gain, not a live discrete controller.

The controller-native structural diagnostics show:

- `structonly` confidence mean `1.0`
- `structonly` confidence std `0.0`
- `structonly` budget mean `0.40000007`
- `structonly` budget std effectively `0`

The same was true for `fullroute`.

Interpretation:

- the controller had fully collapsed
- routing became a constant compute downscale
- the surviving `structonly` gain came from a static learned side-path

### 3. More than one state helped in the structural lane, but only weakly.

AB4 shows:

- `1` state still slightly beat control
- `8` states beat `1`
- anti-collapse `8` states beat plain `8`

That means the branch is not purely “just another adapter,” but the magnitude of the state benefit was modest compared with the collapse problem.

### 4. Structural-lane residual diagnostics were mixed, not boundary-dominant.

Using the saved smoke JSONs:

- mean NLL improved by `-0.01334`
- generic NLL ACF got slightly worse
  - all mean: `0.02355 -> 0.02402`
  - within-regime mean: `0.02487 -> 0.02543`
- argmax residual ACF improved
  - all mean: `0.00881 -> 0.00732`
  - within-regime mean: `0.00900 -> 0.00756`
  - cross-regime mean: `0.00682 -> 0.00518`

Boundary-conditioned NLL did not show the hoped-for “largest gains at transitions” pattern:

- `none`: improved `-0.01574`
- `clause`: improved `-0.01062`
- `sentence`: worsened `+0.06846`
- `after_clause`: worsened `+0.02091`
- `after_sentence`: worsened `+0.06846`

So the early structural branch did not validate the strong “boundary scaffold” thesis.

### 5. Execution-trace supervision is the first intervention that clearly revived controller diversity.

Trace smoke:

- `1-state` exactly reproduced collapse
- `8-state` stayed materially alive
  - `val_hard_peak 0.4564`
  - `val_soft_ppl 7.0810`
- anti-collapse did not help the first trace smoke

This is the strongest evidence in the lane that the real blocker was supervision, not just architecture.

### 6. Trace export improved the controller materially over the first smoke.

The first exported trace-pretrained controller reached:

- `val_loss 3.6189`
- `opcode_acc 0.5666`
- `hard_usage_peak_frac 0.4479`
- `soft_usage_perplexity 6.7070`

That is a materially healthier controller than the structural lane ever produced locally.

### 7. Richer memop targets help, but nested-only curriculum re-collapses the controller.

From the richer trace-pretrain sweep:

- mixed baseline -> mixed + memops:
  - `val_loss 4.8416 -> 4.5144`
  - `val_opcode_acc 0.5145 -> 0.5486`
  - `val_hard_peak 0.4936 -> 0.5472`
  - `val_soft_ppl 5.8491 -> 5.9652`

That is a clear win for richer memory-operation supervision.

But nested-only traces are too narrow:

- `nested-only + memops`: `val_loss 4.2866`, `val_opcode_acc 0.7701`
- but `val_hard_peak 1.0000`, `val_soft_ppl 1.1353`

So nested-only improves the trace task while almost fully collapsing the controller again.

### 8. The trace-transfer sweep is the right next gate, but it currently has a built-in confound.

The active transfer sweep compares:

- `d32` random-init
- `d32` trace-core
- `d32` trace-statebook
- old best `d64` anti-collapse reference

This is good enough for first transfer testing, but not a clean apples-to-apples match against the old best LM branch because:

- `HARDMAX_STRUCT_DIM` differs (`32` vs `64`)
- anti-collapse weights differ
- temperature differs (`1.0` vs `2.0`)

So any first transfer result should be interpreted as:

- transfer signal vs `d32` random baseline
- not “trace transfer vs best old structural branch” in a perfectly controlled sense

## Diagnostics Already Run

Completed and useful:

1. Structural residual smoke
   - saved NLL ACF
   - saved boundary-conditioned NLL
   - saved transition-window NLL
   - saved residual-mode ACF

2. Structural controller smoke
   - confidence/budget collapse diagnosis
   - enough to reject routing

3. Trace controller export summary
   - enough to confirm trace supervision revives controller diversity

4. Broader nested-family generator validation
   - 200-seed smoke with no step-limit failures after revision

## Diagnostics Missing Or Under-Archived

These are the main gaps in the current lane.

### 1. Exact per-run results are not archived consistently.

Several important BPB numbers only live in:

- lane notes
- README summaries
- live terminal logs

rather than a per-iteration `results.json` or `results.csv`.

This is the biggest reproducibility gap in the lane.

### 2. AB5 is under-evidenced locally.

The static-adapter result exists only as a live-log note, not a saved result file.

### 3. The richer trace-pretrain sweep needs exported summaries per arm.

Right now the memops sweep conclusions are correct, but only partially archived in prose.

Each trace-pretrain run should emit and copy back:

- summary JSON
- controller init NPZ when relevant
- maybe a small liveness JSON

### 4. The transfer sweep needs post-run controller diagnostics immediately after completion.

This is already supported by:

- [run_iteration_hardmax_transfer_diagnostics.py](/home/zaytor/transformer_research/parameter-golf/tools/run_iteration_hardmax_transfer_diagnostics.py)

but it cannot run until local artifacts land.

## Most Important Missing Analytics To Run

Once the transfer sweep finishes, run these in order:

1. `d32` random vs `d32` trace-core vs `d32` trace-statebook
   - exact BPB/NLL
   - controller liveness retention

2. Residual smoke on each transfer arm
   - same outputs as the structural-lane smoke

3. Controller-native transfer diagnostics
   - hard state peak fraction
   - soft usage perplexity
   - confidence mean/std
   - confidence vs NLL correlation
   - confidence vs boundary correlation

4. Compare transfered controller vs exported controller
   - how much diversity is lost during LM fine-tuning?

5. Archive exact `results.json` for the sweep

## Current Best Interpretation

The lane has gone through three distinct states:

1. `Structural tags + routing`
   - small LM gain
   - routing loss
   - collapsed controller

2. `Structural tags + anti-collapse`
   - slightly better LM gain
   - soft signal partially revived
   - hard argmax path still nearly dead

3. `Execution-state supervision`
   - first clearly live multi-state controller
   - richer memop targets help
   - curriculum must remain mixed to avoid re-collapse

That means the main research thesis survived, but in a narrower form:

- the controller did not fail because hardmax is useless
- it failed because shallow structural supervision made one state sufficient
- execution-state supervision is the first target family that creates real pressure for specialization

## Decision Gates

The next hard gates should be:

1. Does trace-pretrained transfer beat `d32` random-init in LM BPB?
2. Does transfer preserve materially non-degenerate controller usage?
3. Does the controller stay alive better under weighted-mix trace curriculum than under the current export branch?

If the answer to `1` and `2` is yes, the hardmax lane becomes a real trace-conditioned side-path.

If `1` fails but controller-only trace metrics keep improving, the likely issue is transfer or conditioning, not the trace supervision thesis itself.
