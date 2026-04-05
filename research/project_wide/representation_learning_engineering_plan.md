# Representation Learning Merger Engineering Plan

Date: 2026-03-30

Companion design memo:

- `research/project_wide/representation_learning_complete_system_design_20260403.md`
- `research/project_wide/representation_learning_mcts_search_plan_20260404.md`

That document is the higher-bar architecture reference for the next tranche:

- DGCCA-style shared latent geometry
- stitching-based functional verification
- factor-aware continuous clearing
- downstream checkpoint synthesis as a separate late-stage branch

New foundation now implemented:

- a first linear multi-view GCCA shared-space artifact builder:
  - `tools/representation_learning/build_gcca_shared_geometry.py`
- artifact schema:
  - `SharedLatentGeometry` / `SharedLatentLayer`
- first cohort artifact:
  - `research/representation_learning/priors/qwen_phi_olmo_gcca_shared_geometry_v1.npz`
- first summary:
  - `research/representation_learning/reports/qwen_phi_olmo_gcca_shared_geometry_summary_v1.json`
- first stitching-based functional verification tools:
  - `tools/representation_learning/verify_model_stitching.py`
  - `tools/representation_learning/verify_model_stitching_cohort.py`

This means the repo now has the first complete shared-space plus functional-verification baseline:

- build a multi-view shared latent artifact
- select a shared calibration cohort from that artifact
- fit pairwise stitch maps on that cohort
- evaluate transferability through the target model's own output head

## Objective

The representation-learning lane is not an interpretability project with an optional merger side effect.

It is a model-merger project.

The goal is to:

- extract transferable factor structure from heterogeneous source models
- align those factors across model families and hidden sizes
- select or blend the strongest source per factor
- export one combined representation artifact
- use that artifact to improve a real target model

Interpretability remains useful, but only as:

- validation that the factors are real
- analysis of why one source wins a factor
- debugging when a merge fails

It is not the primary deliverable.

## Current Validated State

The current stack has already established the following:

- multi-model extraction works for at least:
  - `Qwen/Qwen3-4B`
  - `microsoft/Phi-4-mini-instruct`
  - `allenai/OLMo-2-0425-1B`
- the comparison pipeline produces stable cross-model outputs
- chimeric factor ownership is real
- unnamed factors matter more than the current named probe suite
- a simple activation-based spectral merge already produces a small positive result
- a first typed disagreement probe set and routing-kernel bootstrap now exist for the `Qwen/Phi/OLMo` cohort

The most important validated findings are:

### 1. Different models own different parts of the surface

From the current three-model comparison:

- `Qwen` wins `6/9` named concepts
- `Phi-4` wins `3/9` named concepts
- `OLMo` wins `0/9` named concepts
- `OLMo` still has the best mean shared chunk loss

### 2. Shared difficulty, different strategies

Pairwise chunk-loss correlations are high while functional overlap is much lower.

This means:

- models mostly agree on which chunks are hard
- they disagree on how those chunks are represented internally

That is the basic precondition for a useful merger.

### 3. The broad unnamed factors are real

The strongest two-model finding remains:

- `Qwen` wins all named concepts against `OLMo`
- `OLMo` still wins overall chunk loss

Advantage clustering shows:

- `Qwen` advantage is narrow and weird-tail
- `OLMo` advantage is broad and ordinary/noisy-web structure

This is exactly the kind of complementary factor ownership a merger should exploit.

### 4. A first merge already helps a little

The current activation merge on the tiny assembled target is small but positive:

- `Qwen+OLMo` beats the best single source by about `-0.0010` BPB
- it wins `39/64` paired batches against `Qwen`

That is not the end benchmark, but it is enough to justify continued architecture work.

### 5. The current projected-space gauge fix is not good enough

The first activation-space gauge-fixed merge is materially worse than the naive merge.

Implication:

- we should stop iterating that exact alignment branch
- better alignment should come from fuller activation statistics or Jacobian/subspace methods

### 6. Sequence-aware fusion is real, but the pure cascade is not yet the best merger

The new zoo-level sequence lane now has three concrete merger variants:

- flat activation merge
- sparse cascade residual merge
- hybrid cascade-guided spectral merge
- hardmax cascade-guided spectral merge
- moonshot stratified hardmax merge

Current local tiny-probe ordering is:

- best overall baseline: flat activation merge
- best sequence-aware branch: hybrid cascade-guided spectral merge
- hardmax branch: competitive in the two-model case but currently worse than hybrid in the three-model case
- moonshot branch: best local two-model strategy so far, still slightly worse than flat activation in the three-model case
- weakest of the family: pure cascade residual merge

Important nuance:

- the hybrid already beats the flat activation merge on the two-model `Qwen+OLMo` case
- on the three-model case it still trails the flat activation merge slightly, but it is much better than the pure cascade branch

Implication:

- sequence/autocorrelation structure is worth keeping
- but it should currently guide the merger, not replace the spectral merge skeleton outright
- hardmax should stay as an explicit ablation branch, not the default branch
- moonshot should be promoted into the real-target ablation set, because it is the first branch to clearly outperform both hybrid and hardmax on the two-model local probe

### 7. The disagreement-to-kernel bootstrap path now exists

The repo now has first-class artifacts for:

- `DisagreementProbeSet`
- `VerificationTable`
- `RoutingKernel`

And first concrete tools for:

- mining typed disagreement probes from extracted representations
- verifying winners with deterministic held-out chunk-loss rules
- building a first routing kernel from verified winner chunk geometry

The first real cohort bootstrap is:

- `255` disagreement probes across `Qwen/Phi/OLMo`
- mostly `directional_divergence`, with a non-trivial `joint_uncertainty` frontier
- a first routing kernel artifact with centroid-cluster rules plus lookup-style fallback rules for models whose current artifacts do not yet carry chunk projection geometry

Implication:

- the project no longer depends only on static spectral merge artifacts
- the kernel/routing path is now an active engineering lane, not just theory

## What Is Still Missing

We still do not have the number that actually matters:

- step-0 and short-run performance of a real target model initialized from the merged geometry

The real benchmark is:

- baseline random init on the actual target architecture
- single-source representation prior init
- merged-prior init
- same training recipe
- compare BPB and convergence

Until that exists, every other result is still an intermediate.

We also still do not have the first real functional-transfer matrix for the `Qwen/Phi/OLMo`
cohort. The stitching tools now exist, but the actual ordered-pair transfer results still need
to be run and compared against GCCA residuals and aligned-latent cosines.

## The Main Bottlenecks

The current architecture limitations are concrete:

### Runtime bottleneck 1. Priors only touch `Q/K`

Current runtime injection is limited to attention `Q` and `K`.

That is too narrow for a serious merger.

We need a staged path to:

- `Q/K`
- then `Q/K/V/O`
- then `MLP in/out`
- optionally later norms or residual-shaped low-rank structure

### Runtime bottleneck 2. Cross-architecture mapping is still crude

Current target mapping uses random Gaussian projection plus orthonormalization.

That is acceptable only as a smoke-test baseline.

It is not a believable long-term adapter.

We need a calibrated target adapter based on measured layer statistics rather than random projection.

Current concrete runtime adapter ablations now are:

- `random`
- `svd_carrier`
- `svd_carrier_matched`

### Runtime bottleneck 3. No real merged-checkpoint synthesis yet

The current path is:

- merged geometry
- injected as a prior into a target architecture

That is useful, but it is not yet:

- a direct cross-architecture merged checkpoint
- or a fully reconstructed target model from factor ownership

### Merger bottleneck 4. Alignment is still too weak

The current positive merge is from a relatively naive activation-space merge.

The better branches still need to be promoted:

- Jacobian-weighted geometry
- shared-subspace alignment
- GCCA / DGCCA-style multi-view latent construction
- better handling of shared vs unique factors

### Merger bottleneck 6. Sequence-aware fusion still needs a stronger inductive bias

The first pure cascade merger over compressed representation artifacts is useful analytically, but too weak as a direct merger engine.

The better current direction is:

- compute zoo-level order and novelty from the compressed artifacts
- use that signal to guide cluster champion selection inside the stronger flat spectral merge

That is the current bridge from “model-zoo autocorrelation” to a usable merged artifact.

### Evaluation bottleneck 5. Real-target eval not run yet

We have a real evaluator:

- `tools/representation_learning/eval_mlx_representation_init.py`

But there is still no real target-model report.

That is the immediate benchmark gap.

### Verification bottleneck 6. Functional interchangeability is still mostly unmeasured

The lane now has a first stitching verifier, but we still do not know:

- which ordered model pairs transfer well in practice
- whether transferability is symmetric
- whether GCCA-best-aligned pairs are also the best stitching pairs
- whether geometric alignment metrics predict logit-level functional transfer

The current verifier intentionally stays narrow:

- final hidden state only
- affine stitch map
- target-model output head for functional evaluation

That is enough for a real baseline, but it is not the end state.

### Kernel bottleneck 7. Typed probing and verification are still bootstrapped from chunk loss only

The current disagreement and verification path is intentionally cheap:

- disagreement typed from held-out chunk-loss structure
- verification from deterministic chunk-loss winner rules

That is good enough to bootstrap the kernel lane, but not enough for the final system.

The next upgrades are:

- richer disagreement mining from model output distributions, not just chunk-loss margins
- explicit code/fact verification on probes where that is available
- recursive probe generation from kernel uncertainty

### Kernel bottleneck 8. Some source artifacts still lack chunk projection geometry

The current routing-kernel builder can cluster winners only when the source representation includes `chunk_layer_projections`.

Right now this means:

- `Qwen` and `OLMo` support centroid-cluster routing rules
- `Phi` currently falls back to lookup-style rules because its current artifact lacks chunk projection geometry

Implication:

- the kernel path already preserves unique winners instead of dropping them
- but the next extraction refresh should ensure every source model used in routing carries chunk projection geometry

## Execution Principle

The project should now run in six parallel but clearly separated tracks:

1. `Benchmark`
   - prove the merged artifact helps a real target model
2. `Runtime architecture`
   - improve how priors map into the target model
3. `Merger geometry`
   - improve factor extraction, alignment, and merge quality
4. `Shared-space and functional verification`
   - measure which model pairs are actually interchangeable
5. `Analysis`
   - explain wins, failures, and contribution patterns
6. `Kernel`
   - mine disagreement regions, verify winners, and build a routing kernel

The benchmark track is the critical path.

## Workstream A: Real Target Benchmark

This is the highest-priority workstream.

### A1. Step-0 actual-target init eval

Run the current real-target evaluator on the actual `9x512` MLX GPT target.

Required arms:

- baseline random init
- `Qwen` prior
- `OLMo` prior
- `Phi` prior
- `Qwen+OLMo` activation merge
- `Qwen+Phi+OLMo` activation merge
- `Qwen+OLMo` hybrid cascade-guided merge
- optionally `Qwen+Phi+OLMo` hybrid cascade-guided merge once the two-model hybrid is benchmarked

Primary outputs:

- mean validation BPB at initialization
- paired batch deltas vs baseline
- paired batch deltas vs best single source

Decision gate:

- if merged prior beats baseline and best single-source prior at step 0, the merger path is directly validated
- if not, the runtime adapter and alignment become the main suspects

## Workstream F: Typed Probing And Routing Kernel

This is the moonshot lane that should eventually subsume static merge-only thinking.

### F1. Typed disagreement mining

Promote the new toolchain:

- `tools/representation_learning/mine_model_disagreements.py`

Current bootstrap types:

- `confidence_divergence`
- `directional_divergence`
- `joint_uncertainty`

Immediate next step:

- run it on every active cohort, not just the current `Qwen/Phi/OLMo` snapshot

### F2. Deterministic verification

Promote:

- `tools/representation_learning/verify_probe_outcomes.py`

Current method:

- held-out chunk-loss winner

Next methods to add:

- code-exec verification where probe text is code-like
- structured factual verification where a cheap lookup exists

### F3. Routing kernel build and audit

Promote:

- `tools/representation_learning/build_routing_kernel.py`

Required outputs per run:

- rule counts by winning model
- rule counts by probe type
- centroid-cluster vs lookup-fallback rule counts
- support mass by winning model

Decision gate:

- if the routing kernel remains concentrated in one model, disagreement mining is too weak
- if multiple models retain durable routing territory, the kernel path is working

### F4. Kernel-aware student training

After the current one-hour prior-init benchmarks settle, the next student path should be:

- baseline student
- single-source prior student
- static merged-prior student
- routing-kernel teacher student

That is the first real test of “verified-best-per-region distillation” versus static geometry prior injection.

Current status:

- first reusable teacher artifact now exists:
  - `tools/representation_learning/build_kernel_teacher_dataset.py`
  - `KernelTeacherDataset` in `tools/representation_learning/schemas.py`
- first real export artifact:
  - `research/representation_learning/priors/qwen_phi_olmo_kernel_teacher_v1.npz`
- text-conditioned student artifact and first consumer now exist:
  - `tools/representation_learning/build_kernel_teacher_text_dataset.py`
  - `tools/representation_learning/train_kernel_teacher_student.py`
  - `KernelTeacherTextDataset` in `tools/representation_learning/schemas.py`
  - `research/representation_learning/priors/qwen_phi_olmo_kernel_teacher_text_v1.npz`
- current teacher characteristics from the best `v5 full` ecology checkpoint:
  - `3060` examples
  - `64`-dim cleared embeddings
  - teacher routing accuracy `0.9261`
  - mean winner probability `0.9086`
- first real Mini smoke and follow-up ablation now exist:
  - frozen smoke:
    - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_smoke_v1/summary.json`
  - focused three-arm follow-up:
    - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_ablation_v1/comparison_v1.json`
- focused tuning sweep on the best follow-up recipe:
  - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_tuning_v1/comparison_v1.json`
- readout sweep on the tuned recipe:
  - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_readout_v1/comparison_v1.json`
- modest capacity sweep around the best readout:
  - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_capacity_v1/comparison_v1.json`
- first direct-vs-factorized student sweep:
  - `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_factorized_v1/comparison_v1.json`
- current best kernel-student recipe from the first ablation batch:
  - `unfrozen_distill_only`, `8 epochs @ 1e-3`, `readout_mode=mean_last`
  - `model_dim = 128`, `num_layers = 2`
  - `val_loss_total = 0.7344`
  - `val_mean_teacher_cosine = 0.2656`
- the first ablation batch also established:
  - unfreezing the backbone is necessary
  - pure teacher distillation currently beats the mixed `CE + distill` objective
  - simply scaling the small student from `128x2` to `256x4` did not help at the same learning rate / budget
  - longer training helps slightly, and `1e-3` currently beats both `3e-4` and `1e-4`
  - `mean_last` readout is the best tested pooling strategy so far, with `last` a close second
  - learned attention pooling is worse than the simpler readouts
  - modest width/depth increases (`160x2`, `128x3`, `160x3`) all regress relative to the `128x2` baseline
  - the first disagreement-factor student branch failed because the naive factor-coefficient loss is badly mis-scaled
  - direct embedding supervision currently remains much better than the raw factorized objective

Interpretation:

- the ecology checkpoint is now materially reusable as a best-of-N teacher, not just as a classifier report
- the first student step now consumes the text-conditioned teacher artifact directly rather than trying to read the ecology checkpoint directly
- the student lane is no longer blocked on plumbing; it is in real objective/architecture tuning
- the next benchmark move is no longer another frozen smoke
- the next student ablation should hold `unfrozen_distill_only` fixed as the baseline and test:
  - evaluation against single-source and static-merge teacher baselines, not just teacher-fit loss
  - stronger student heads over the current `mean_last` substrate rather than naive width/depth growth
  - teacher-target variants beyond the current best-of-N ecology teacher
  - normalized factorized objectives:
    - factor cosine / correlation losses instead of raw MSE
    - variance-normalized factor targets
    - reconstructive factor heads with a separate low-weight factor regularizer rather than dominant factor loss

### F5. Typed activation ecology model

The routing table is now a bootstrap object, not the intended endpoint.

We now have first-class typed-event substrates:

- `ActivationEventDataset`
- `EcologyTrainingSet`
- `tools/representation_learning/build_ecology_training_data.py`
- `tools/representation_learning/train_ecology_model.py`

The first concrete cohort build produced:

- `9180` typed activation events
- `3060` verified winner-labeled training examples
- a first learned ecology transformer trained on held-out chunk splits

Current local results on the `Qwen/Phi/OLMo` cohort are:

- first ecology-model ceiling:
  - `full` feature ecology model: `0.9089` held-out winner accuracy
  - `argmin_loss` baseline: `0.7031`
  - train-split static routing kernel baseline: `0.2435`
- repaired-`Phi` structural baseline:
  - `full`: `0.9075`
  - `structure_only`: `0.4635`
  - `cb05 structure_only`: `0.5217`

Interpretation:

- the learned ecology model is already substantially better than the current static routing kernel
- the remaining gap is not “can the ecology model learn the winner function?” but “can it predict the winner without oracle-like loss features?”
- the repaired-`Phi` run is still the best non-oracle structural baseline

Recent engineering updates:

- `Phi` chunk projection geometry was backfilled successfully, so it is no longer invisible to the structural router
- this materially improved the `full` ecology path and gave the plain `structure_only` path more real `Qwen` and `Phi` routing territory
- forward-signature sidecars were implemented and extracted, but the first scalar-forward pass regressed the ecology router

Current read on the scalar-forward branch:

- `qwen_phi_olmo_ecology_forward_signature_compare_v1.json`
- deltas vs repaired-`Phi` baseline:
  - `full`: `0.9075 -> 0.8099`
  - `structure_only`: `0.4635 -> 0.3867`
  - `cb05 structure_only`: `0.5217 -> 0.4909`
- conclusion:
  - scalar forward summaries carry some signal, but they wash out the geometric routing signal when mixed naively

New substrate added:

- `ForwardSignatureDataset`
- `tools/representation_learning/extract_forward_signatures.py`
- disagreement-factor decomposition inside `tools/representation_learning/build_ecology_training_data.py`

The first disagreement-factor experiment is now complete:

- dataset:
  - `qwen_phi_olmo_ecology_training_v5_disagreement_factors`
- targeted reports:
  - `qwen_phi_olmo_ecology_model_v5_disagreement_factors_targeted`
  - `qwen_phi_olmo_ecology_model_v5_disagreement_factors_targeted_cb05`
- compact compare:
  - `qwen_phi_olmo_ecology_disagreement_factor_compare_v1.json`

Results:

- plain `structure_only` improved over the repaired-`Phi` baseline:
  - `0.4635 -> 0.4909`
  - `Qwen`: `0.10 -> 0.15`
  - `Phi`: `0.4755 -> 0.5144`
  - `OLMo`: `0.5049 -> 0.5250`
- `cb05 structure_only` regressed:
  - `0.5217 -> 0.4961`
- `factor_only` is not enough:
  - plain: `0.4297`
  - `cb05`: `0.3932`
- `full` also regressed modestly vs repaired-`Phi`:
  - plain: `0.9075 -> 0.8815`
  - `cb05`: `0.8750`

Geometry-only isolation is also informative:

- `qwen_phi_olmo_ecology_disagreement_factor_geometry_compare_v1.json`
- vs the older geometry-only lane:
  - plain `geometry_only`: `0.3906 -> 0.4740`
  - `cb05 geometry_only`: `0.4688 -> 0.4427`

Interpretation:

- disagreement factors are useful signal, but only partially
- they help the plain non-oracle structural router
- they do not solve the `0.52 -> 0.91` gap
- they do not yet improve the class-balanced structural or full-oracle paths
- factor-only routing is too weak, so the factors are not yet a sufficient representation of the routing problem

Immediate next ecology tasks:

- stop investing in scalar forward summaries as the main path
- move from winner classification toward factor-aware mixture routing
- add cross-model per-factor topology features instead of only per-model factor signatures
- test “geometry + factors + topology” before any further scalar feature expansion
- extend the ecology target from “winner model” to “winner source-layer / factor subset” once the factor-topology path is stable
- keep the new `--torch-num-threads` and incremental `summary.partial.json` support in `train_ecology_model.py` for future long sweeps

### A2. One-hour training benchmark

Train the same target architecture for one hour with the same recipe for:

- baseline
- best single-source prior
- best merged prior
- best sequence-aware prior

Primary outputs:

- final BPB
- early validation curve
- improvement in time-to-quality, not just end quality

Decision gate:

- if merged init helps early training even when the final gap is small, the geometry is still valuable

### A3. Strength and target sweep

Once A1 exists, sweep:

- `REP_LEARN_INIT_STRENGTH`
- init targets:
  - `q`
  - `k`
  - `qk`

Later extend to:

- `qkv`
- `qkvo`

Purpose:

- separate “good artifact, bad strength” from “bad artifact”

## Workstream B: Runtime Architecture Improvements

### B1. Replace random projection with a calibrated target adapter

Current mapping in `runtime_mlx.py` is a random projection baseline.

Replace it with a measured adapter built from:

- target-model layer statistics
- calibration activations
- optionally source-to-target CCA or ridge projection

Milestone:

- deterministic adapter artifact for each target architecture

### B2. Expand runtime application beyond `Q/K`

Stage this deliberately:

1. support `qkv`
2. support `qkvo`
3. support MLP matrices

Each expansion should be benchmarked separately.

Do not expand the surface faster than we can measure.

### B3. Separate source geometry from target adapter

The runtime stack should clearly separate:

- source-side merged geometry
- target-side adapter

That makes it possible to:

- reuse one geometry artifact across several target architectures
- compare architecture sensitivity cleanly

### B4. Add runtime diagnostics

For each prior application, record:

- per-layer prior energy injected
- relative norm shift in each target matrix
- alignment between initialized target weights and source factor bases

These are required to debug “merge exists but init effect is weak”.

## Workstream C: Merger Geometry Improvements

### C1. Stabilize the activation-merge baseline

The naive activation merge is currently the best positive result.

Before replacing it, make it easier to analyze:

- export ownership counts per layer
- export retained factor energies
- export overlap between merged factors and named anchors

This becomes the baseline every improved merger must beat.

### C2. Promote the Jacobian branch

We already have:

- Jacobian extraction
- Jacobian view comparison
- Jacobian merge entrypoint

The missing piece is execution and comparison.

Required outputs:

- activation vs Jacobian agreement per source
- Jacobian merged artifact
- real-target evaluation of Jacobian merge vs activation merge

Decision gate:

- if Jacobian merge beats activation merge, promote Jacobian to the main path
- if not, keep it as a rigor branch

### C3. Move from vector alignment to shared-subspace alignment

The next real alignment improvement should be subspace-based, not another projected-space vector rotation trick.

Priority methods:

- CCA or related shared-subspace projection
- covariance-weighted layer geometry
- principal-angle matching for nearby factor groups

The goal is:

- stronger cross-model correspondence
- better handling of different hidden sizes
- less sensitivity to unstable individual vectors

### C4. Shared vs unique factor handling

The current merge is too close to winner-take-all at the single-direction level.

We need a more principled factor policy:

- shared factors:
  - select or softly blend by source strength and alignment confidence
- unique factors:
- preserve rather than average away

This is likely where multi-model merges will improve over the current two-model baseline.

### C5. Add zoo-level sequential residual fusion

Flat best-per-direction selection is a useful baseline, but it throws away the conditional structure between models.

We should also treat the model zoo as a sequence of compressed representation artifacts and measure:

- what the first, most central model explains
- what each later model adds conditionally as a residual
- which models are mostly redundant vs genuinely novel

This lane should operate only on the extracted representation artifacts, not on the original source checkpoints.

Immediate outputs:

- model centrality order
- leave-one-out novelty
- sequential residual merge artifact

This gives us a richer merger object without needing to run the large source models again.

Current local status:

- first sparse cascade implementation exists
- it is not yet the best direct merger

Next step:

- keep the pure cascade branch for analysis, novelty, and ordering
- promote the hybrid branch for actual merger benchmarking

### C6. Add hybrid cascade-guided spectral merge

Build a merger that keeps spectral clustering/champion selection as the main skeleton but conditions champion choice on:

- leave-one-out novelty
- greedy zoo order
- shared-vs-unique cluster structure

Purpose:

- preserve the strong empirical behavior of flat spectral merge
- inject sequence/autocorrelation information without forcing the whole merge to be sequential

Current local status:

- two-model hybrid beats flat activation merge on the tiny probe
- three-model hybrid is slightly worse than flat activation merge but much better than pure cascade

### C7. Keep hardmax as a moonshot ablation, not the default path

The hardmax branch now exists and is benchmarkable.

Current local status:

- on `Qwen+OLMo`, hardmax is effectively tied with the best hybrid setting
- on `Qwen+Phi+OLMo`, hardmax is clearly worse than the tuned hybrid

Implication:

- keep hardmax in the ablation matrix
- do not promote it over hybrid unless the real-target benchmark disagrees with the tiny local probe

### C8. Add stratified moonshot merge to the benchmark set

The moonshot branch now exists and is benchmarkable.

It differs from hybrid and hardmax by:

- reserving explicit capacity for robust shared clusters
- reserving explicit capacity for unique clusters
- allocating unique capacity with novelty-weighted per-model quotas

Current local status:

- on `Qwen+OLMo`, moonshot is the best local strategy so far
- on `Qwen+Phi+OLMo`, moonshot is better than hardmax but still slightly worse than tuned hybrid

Implication:

- moonshot belongs in the real-target init-eval and one-hour training benchmark set
- but flat activation remains the baseline to beat

### C9. Chain-of-merges remains later

Layer-sequential covariance updates and chain-of-merges are still interesting, but they should not block:

- real target eval
- better target adapters
- Jacobian-vs-activation comparison

Treat this as a later rigor upgrade, not the next blocker.

## Workstream D: Shared-Space and Functional Verification

### D1. Run the first stitching cohort on the GCCA artifact

Use:

- `tools/representation_learning/verify_model_stitching_cohort.py`

Against:

- `research/representation_learning/priors/qwen_phi_olmo_gcca_shared_geometry_v1.npz`

Required outputs:

- ordered pairwise transfer matrix
- best source model for each target model under stitch transfer
- relationship between GCCA residuals and stitch quality
- relationship between aligned-latent cosine and stitch quality

This is the first real test of whether the shared-space artifact is functionally meaningful
rather than only geometrically tidy.

### D2. Compare geometric and functional alignment explicitly

The next summary report should join:

- GCCA residuals
- pairwise aligned-latent cosine means
- pairwise stitching KL / JS / top-1 agreement

Decision gate:

- if good GCCA layers also stitch well, the shared-space path is validated
- if not, the shared space still needs a better construction or a better stitch operator

### D3. Upgrade from last-hidden stitching to layer-aware stitching

The current stitching baseline is intentionally narrow:

- final hidden state only
- affine map only

Next upgrades should be:

- selected internal layers, not just final hidden
- shared-latent-assisted stitch maps
- eventually richer stitch operators than plain affine maps

But those upgrades should happen only after the first cohort matrix exists.

## Workstream E: Analysis and Failure Interpretation

### E1. Formalize factor ownership reporting

Every comparison or merge report should expose:

- factor ownership counts
- factor energy by source
- shared vs unique factor totals
- coverage of eval-winning chunks where possible

### E2. Explain why the three-model merge underperformed the two-model merge

This is currently the most important failure analysis question.

Candidates include:

- weak `Phi` alignment
- target adapter loss
- adding partially aligned factors that hurt more than help

This should be analyzed explicitly rather than left as intuition.

### E3. Keep named probes as diagnostics only

Named probe results should still be exported, but they should no longer dominate planning.

The main analysis questions are:

- which source owns useful factors
- which factors transfer to the target
- which factors survive training

## Workstream G: Automation and Infrastructure

### G1. Durable run recipes

Promote current ad hoc commands into durable templates for:

- real target step-0 eval
- one-hour representation init ablation
- one-hour direct merger comparison
- Jacobian extraction
- Jacobian merge eval

Current durable templates now include:

- `research/iterations/templates/mlx_representation_init_eval.example.json`
- `research/iterations/templates/mlx_representation_merger_1h.example.json`

And the init-eval output can now be reduced with:

- `tools/representation_learning/summarize_init_eval_report.py`

### G2. Mini-cluster execution plan

Use the free Mini pool for:

- one node for real-target eval
- parallel nodes for Jacobian extractions
- one node for postprocessing and report collation

### G3. Incremental model-zoo path

Do not add more source models until:

- the real-target benchmark exists
- and the runtime adapter is better than random projection

After that, new source models should be added incrementally, not by restarting the project.

## Milestones

### M0. Current baseline

Already true:

- multi-model extraction works
- factor ownership is chimeric
- naive activation merge is slightly positive
- current gauge-fixed branch is negative

### M1. Real-target step-0 proof

Required:

- merged prior beats baseline and best single-source prior on the real target at initialization time

This is the next critical milestone.

### M2. Runtime adapter upgrade

Required:

- calibrated target adapter beats random projection
- at least one expanded injection target set beats `qk`

### M3. Jacobian promotion decision

Required:

- Jacobian merge evaluated on the real target
- clear comparison against activation merge

### M4. Training-lift proof

Required:

- merged prior improves one-hour training vs baseline and single-source prior

### M5. Generalized merger object

Required:

- geometry artifact and target adapter separated cleanly
- one merged artifact reusable across more than one target architecture

## Immediate Execution Order

The next autonomous sequence should be:

1. Run `A1` now:
   - real target step-0 init eval on single-source, flat merge, and hybrid merge artifacts
2. Run `C2` in parallel:
   - Jacobian extractions and Jacobian merge
3. Run `B1` next:
   - replace random projection with a calibrated target adapter
4. Re-run `A1` with the calibrated adapter
5. Only then run `A2`:
   - one-hour training benchmark
6. After that:
   - expand injection targets from `qk` to `qkvo`
   - then to MLP support if justified

## What Not To Do Next

Do not spend the next cycle on:

- more named-probe expansion
- more projected-space gauge-fix variants
- more toy assembled-model benchmarking
- adding many new source models before the target benchmark exists

Those are all lower value than:

- real target eval
- better target adapter
- Jacobian-vs-activation merge comparison

## Success Criterion

This lane is successful when we can say:

- a merged representation object built from multiple source models
- projected into a real target architecture
- improves BPB or convergence over both random init and any single-source prior

Everything else is in service of that.

## Student MCTS Frontier

The current principal variation for the kernel-student lane is:

- keep the small unfrozen `128x2` student
- keep `projection_mode=direct`
- keep `readout_mode=mean_last`
- stop spending frontier budget on naive width/depth growth
- stop spending frontier budget on raw factor-MSE supervision

The next highest-value branch is teacher quality, not architecture churn:

1. materialize matched baseline teachers from the existing ecology teacher artifact
   - `winner_only`
   - `static_mix` with global teacher-mean source weights
   - fixed-source `Qwen`
   - fixed-source `OLMo`
   - fixed-source `Phi`
2. train the exact same best student recipe against each teacher
3. compare downstream fit and student quality on equal footing

Durable assets for this branch:

- `tools/representation_learning/build_kernel_teacher_baseline_dataset.py`
- `research/iterations/templates/representation_learning_kernel_teacher_student_teacher_compare.example.json`

Result from the first full teacher comparison:

- `OLMo` fixed-source teacher is best on student fit
- static teacher-mean mixture is second
- ecology teacher is worse than both and worse than `Phi/Qwen` single-source teachers
- winner-only teacher is worst for this student recipe

That means the next student branch should stop assuming the ecology teacher is the right teacher.
The next product-oriented check is:

1. log explicit validation `loss_ce_probe` / `bpb_probe` in the student summary
2. rerun only the top-3 teachers (`OLMo`, `static_mix`, `ecology`)
3. rank them by both distillation fit and CE/BPB, not teacher cosine alone

Top-3 CE/BPB rerun result:

- `OLMo` remains best by both distillation fit and CE/BPB probe
- `static_mix` is second by fit but worse by CE/BPB
- `ecology` remains worst of the top-3 on fit and does not beat `OLMo` on CE/BPB

That makes the next branch narrower:

1. hold the `OLMo` teacher fixed
2. sweep small non-zero `ce_weight`
3. test whether product-side CE/BPB prefers mixed supervision even though pure distillation won the earlier teacher-fit branch

Durable asset for this branch:

- `research/iterations/templates/representation_learning_kernel_teacher_student_olmo_ce_sweep.example.json`

External LM evaluation on `compare_v1_calibration.jsonl` changed the ranking again:

- `olmo_ce025`: `bpb = 1.7304` (best)
- `olmo_ce005`: `1.9509`
- `olmo_ce010`: `2.3013`
- `ecology_ce0`: `10.0024`

So the next branch is no longer “teacher search.” It is product tuning around:

- `OLMo` teacher
- nonzero CE
- external-eval winner region around `ce_weight ~= 0.25`

Durable assets for that branch:

- `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_external_eval_top4_v1/comparison_v1.json`
- `research/iterations/templates/representation_learning_kernel_teacher_student_olmo_product_tuning.example.json`

The focused product-tuning sweep then improved the baseline materially on the same
external eval surface:

- `ce030_e12_lr5e4`: `bpb = 1.5018` (current best)
- `ce025_e12_lr5e4`: `1.5077`
- `ce030_e8_lr1e3`: `1.8030`
- `ce025_e12_lr1e3`: `1.8708`
- `ce020_e8_lr1e3`: `1.8908`
- `ce025_e8_lr1e3`: `1.9349`

This changes the principal variation again:

1. hold the `OLMo` teacher fixed
2. hold the small `128x2` `mean_last` student fixed
3. exploit the `ce_weight ~= 0.30`, `epochs >= 12`, `learning_rate ~= 5e-4`
   region
4. evaluate all follow-up checkpoints with the external LM suite, not just
   `bpb_probe`

Durable assets for the new frontier:

- `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_olmo_product_tuning_v1/comparison_v1.json`
- `research/representation_learning/reports/qwen_phi_olmo_kernel_teacher_student_olmo_product_tuning_external_eval_v1/comparison_v1.json`
- `research/iterations/templates/representation_learning_kernel_teacher_student_olmo_product_tuning_v2.example.json`
