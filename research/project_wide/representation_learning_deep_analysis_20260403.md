# Representation Learning Deep Analysis

Date: 2026-04-03

## Purpose

This document is a project-level synthesis of the representation-learning lane as it exists today.

It is meant to answer four questions:

1. What has been strongly validated?
2. What has only been partially validated?
3. What recent results changed the interpretation of the project?
4. What are the actual bottlenecks now?

This is not a design memo. The companion design documents are:

- `research/project_wide/representation_learning_complete_system_design_20260403.md`
- `research/project_wide/representation_learning_engineering_plan.md`

This document is an evidence-backed state assessment.

## Executive Summary

The representation-learning project has already established the scientific premise of the lane:

- multiple independently trained models appear to solve the same external language problem
- they do so with materially different internal geometry
- those differences are not random noise
- some of those differences are complementary enough to support positive merger signal

The strongest validated findings are:

- shared chunk difficulty is extremely correlated across `Qwen`, `Phi`, and `OLMo`
- internal subspace overlap is much lower than chunk-difficulty agreement
- different models own different concept and regime slices
- the strongest advantages are mostly not the named concept probes
- the first activation merge is positive
- the learned ecology model is much stronger than the first static routing kernel

The strongest negative or cautionary findings are:

- scalar forward summaries mostly hurt the non-oracle routing path
- the first gauge-fixed alignment branch is clearly bad
- current GCCA shared geometry is not yet enough to make final hidden states functionally interchangeable at the logit level
- the decisive real-target `9x512` training result is still unresolved in the repo

The current state is therefore:

- cross-model transfer looks feasible
- current routing/merger signals are real
- the main unsolved transition is:
  - `shared geometry -> functionally interchangeable representations -> target-model gain`

## 1. The Foundational Scientific Result

The core three-model comparison is:

- [qwen3_phi4_olmo2_compare_v5.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen3_phi4_olmo2_compare_v5.json)

The most important numbers in that report are:

- best mean shared chunk-loss model: `allenai/OLMo-2-0425-1B`
- pairwise shared chunk-loss correlations:
  - `Qwen` vs `Phi`: `0.950968651639414`
  - `Qwen` vs `OLMo`: `0.9451386571336365`
  - `Phi` vs `OLMo`: `0.9636623792920096`
- pairwise mean subspace overlaps:
  - `Qwen` vs `Phi`: `0.2434896132776645`
  - `Qwen` vs `OLMo`: `0.2584091638611394`
  - `Phi` vs `OLMo`: `0.24826962337400196`

This is the deepest positive result in the project so far.

Interpretation:

- the models largely agree on which text regions are hard
- they do not represent those regions in anything close to the same basis

This is exactly the condition under which cross-model transfer and chimeric merger can make sense.

If shared chunk-loss correlation had been low, the lane would be close to dead. It was not low. It was extremely high. At the same time, overlap was far from trivial. So the project is not trying to merge unrelated systems. It is trying to reconcile different compressions of the same external structure.

## 2. Complementary Ownership Is Real

The named concept ownership result from the same report is:

- `Qwen`: `6/9` named concepts
- `Phi`: `3/9` named concepts
- `OLMo`: `0/9` named concepts

But `OLMo` is still the best mean chunk-loss model overall.

That means:

- “best global predictor” is not the same as “best representer of the named concepts we probed”
- the current human-designed concept suite is not spanning the real surface

This is reinforced by the two-model `Qwen` vs `OLMo` comparison:

- [qwen3_vs_olmo2_compare_v7.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen3_vs_olmo2_compare_v7.json)

Key numbers:

- shared chunks: `128`
- `Qwen` better chunks: `13`, mean advantage `0.14610877403846154`
- `OLMo` better chunks: `115`, mean advantage `0.2789826766304348`

This is one of the clearest project-level findings:

- `Qwen` wins the named concept suite
- `OLMo` wins broad ordinary chunk prediction

So the useful geometry is not mostly in the named probe bank. It is in broader latent structure that the current probe bank only partially touches.

## 3. The Unnamed Structure Matters More Than The Named Structure

The advantage-clustering reports on the `Qwen` vs `OLMo` lane were the first clean proof that the pipeline can discover useful structure humans did not specify in advance.

The key report is:

- [qwen3_vs_olmo2_advantage_clusters_v4_thr097.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen3_vs_olmo2_advantage_clusters_v4_thr097.json)

The project-level interpretation from that lane has held up:

- `Qwen` owns a narrow weird-tail regime
- `OLMo` owns a broad ordinary/noisy-web regime

This mattered for the direction of the whole project because it established that:

- the pipeline is not only rediscovering hand-designed logical operators
- it is able to surface high-value latent registers and chunk families automatically

This is one reason the lane should not collapse back to “expand the named probe set.” The named probes are useful diagnostics. They are not the real surface.

## 4. The First Merger Signal Is Small But Real

The original positive merger result is still the clearest direct proof of concept:

- [activation_merge_zero_shot_eval_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/activation_merge_zero_shot_eval_v1.json)
- [activation_merge_gauge_eval_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/activation_merge_gauge_eval_v1.json)

Key numbers:

- `Qwen`: `10.37484244940494` BPB
- `OLMo`: `10.375331417579172`
- `Phi`: `10.379980001142837`
- `Qwen+OLMo` activation merge: `10.373815874213022`
- `Qwen+Phi+OLMo` activation merge: `10.37455560384113`

Against `Qwen` specifically:

- `Qwen+OLMo` merge delta: `-0.001026575191915735` BPB
- `Qwen+OLMo` wins `39/64` paired batches
- `Qwen+Phi+OLMo` merge delta: `-0.00028684556380784154`
- `Qwen+Phi+OLMo` wins `35/64`

These are small numbers, but the sign is the important part:

- naive chimeric merger helped
- it helped more in the 2-model case than the 3-model case

At the same time, the gauge-fixed branch failed clearly:

- `Qwen+OLMo` gauge-fixed delta vs `Qwen`: `+0.008861513574862895`
- `Qwen+Phi+OLMo` gauge-fixed delta vs `Qwen`: `+0.008323068141758622`

Interpretation:

- complementary transfer signal exists
- current alignment quality is still a severe bottleneck
- “more principled” alignment is not automatically better if it loses the useful structure

## 5. Sequence-Aware Static Merger Work: Useful But Secondary

The sequence-aware static merger lane produced meaningful information, but it did not overtake the main scientific result.

The best compact comparison is:

- [hybrid_vs_activation_vs_cascade_zero_shot_small_v2.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/hybrid_vs_activation_vs_cascade_zero_shot_small_v2.json)
- [moonshot_param_sweep_small_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/moonshot_param_sweep_small_v1.json)

Key findings:

- pure cascade residual merge is not the best merger
- sequence-aware hybrid guidance can beat flat activation in the 2-model case
- the moonshot branch is the best local 2-model strategy:
  - `Qwen+OLMo` flat activation: `10.00348370805423`
  - `Qwen+OLMo` moonshot: `10.003393331171202`
- the 3-model case remains harder:
  - `Qwen+Phi+OLMo` flat activation: `10.002647872371039`
  - `Qwen+Phi+OLMo` hybrid: `10.002809149101884`
  - `Qwen+Phi+OLMo` moonshot: `10.002825659436368`

Interpretation:

- sequence/autocorrelation structure is probably real
- but static heuristic-guided fusion is not the main bottleneck anymore
- this branch is now a baseline/control lane, not the center of gravity

## 6. The Routing Table Was A Bootstrap, Not The Solution

The typed disagreement -> verification -> routing-kernel pipeline was an important architectural step:

- [qwen_phi_olmo_kernel_pipeline_v1/summary.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/priors/qwen_phi_olmo_kernel_pipeline_v1/summary.json)
- [qwen_phi_olmo_routing_kernel_summary_v3_min1_fallback.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_routing_kernel_summary_v3_min1_fallback.json)
- [qwen_phi_olmo_routing_kernel_val_eval_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_routing_kernel_val_eval_v1.json)

What it established:

- `255` typed disagreement probes
- `255` verified outcomes
- `134` routing rules in the fallback-preserving kernel

Rule distribution:

- `Qwen`: `24` rules, support mass `144`
- `OLMo`: `107` rules, support mass `1632`
- `Phi`: `3` fallback rules, support mass `107`

But the validation accuracy was only:

- `0.24348958333333334`

That is not competitive with the learned ecology model. This turned out to be a very useful result because it clarified the role of the routing table:

- useful as bootstrap artifact
- not the final kernel

That conclusion held even after `Phi` geometry was repaired.

## 7. The Learned Ecology Model Is The Stronger Kernel Direction

The first learned ecology result was:

- [qwen_phi_olmo_ecology_leaderboard_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_leaderboard_v1.json)

Key numbers:

- `full`: `0.9088541666666666`
- `argmin_loss` baseline: `0.703125`
- `majority` baseline: `0.46875`
- static routing kernel baseline: `0.24348958333333334`

This is one of the strongest architectural findings in the project.

It says:

- the winner function is learnable
- a learned ecology model is dramatically stronger than a static routing table
- the routing-kernel path should be learned and continuous, not mostly lookup-based

But the ablations in that same report showed the gap immediately:

- `no_loss`: `0.4622395833333333`
- `structure_only`: `0.46875`
- `embedding_only`: `0.4895833333333333`

Interpretation:

- the ecology model was real
- but it was still depending heavily on oracle-like loss information

This became the central “gap” problem of the lane:

- oracle-ish ecology: about `0.91`
- best non-oracle structure-only routing: about `0.52`

## 8. Phi Geometry Repair Was The Last Clear Win In The Ecology Lane

Repairing `Phi` chunk projections was not cosmetic. It changed the ecology lane materially.

The key comparison is:

- [qwen_phi_olmo_ecology_phi_backfill_compare_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_phi_backfill_compare_v1.json)

Key improvements:

- `full`: `0.8203125 -> 0.9075342465753424`
- `structure_only`: `0.4609375 -> 0.4634703196347032`
- best non-oracle structural lane:
  - `v3_cb05 structure_only = 0.521689497716895`

Per-winner effects in the repaired plain structural router:

- `Qwen`: `0.0 -> 0.1`
- `Phi`: `0.4281609195402299 -> 0.47549019607843135`
- `OLMo`: `0.5694444444444444 -> 0.5049019607843137`

This was important because it proved two things:

- geometry quality matters a lot
- missing geometric coverage can silently make the structural router blind in a source model’s territory

This also means some earlier structural-routing conclusions were artificially pessimistic because `Phi` geometry was incomplete.

## 9. Scalar Forward Signatures Mostly Hurt

The first forward-signature sidecar tranche was a negative result, but an informative one:

- [qwen_phi_olmo_ecology_forward_signature_compare_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_forward_signature_compare_v1.json)
- [qwen_phi_olmo_ecology_forward_feature_ablation_summary_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_forward_feature_ablation_summary_v1.json)

Compared to the repaired-`Phi` baseline:

- `full`: `0.9075 -> 0.8099`
- `structure_only`: `0.4635 -> 0.3867`
- `cb05 structure_only`: `0.5217 -> 0.4909`

Feature-family reads:

- `forward_only = 0.4166666666666667`
- `geometry_only = 0.390625`
- `cb05 geometry_only = 0.46875`
- `cb05 forward_only = 0.3802083333333333`

This is the strongest evidence for the scalar-trap diagnosis.

Interpretation:

- scalar summaries like entropy and confidence did contain some signal
- but in their current form they blurred or washed out the stronger geometric signal
- richer typing is not enough if it is still scalarizing the useful structure away

This was the point at which the lane correctly pivoted from “more scalar features” toward:

- per-factor structure
- multi-view shared spaces
- functional verification

## 10. Disagreement Factors Help, But Only Partially

The disagreement-factor tranche showed that model-derived per-factor geometry is a real improvement over generic scalar summaries, but still not enough.

Key reports:

- [qwen_phi_olmo_ecology_disagreement_factor_compare_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_disagreement_factor_compare_v1.json)
- [qwen_phi_olmo_ecology_disagreement_factor_geometry_compare_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_ecology_disagreement_factor_geometry_compare_v1.json)

Main results:

- `structure_only`: `0.4635 -> 0.4909`
- `cb05 structure_only`: `0.5217 -> 0.4961`
- `factor_only`: `0.4297`
- `geometry_only`: `0.3906 -> 0.4740`
- `cb05 geometry_only`: `0.4688 -> 0.4427`

Interpretation:

- disagreement factors are real signal
- they help the plain non-oracle router
- they do not close the gap to the oracle ecology
- they can hurt class-balanced routing if they are not combined carefully

This is why the next move after disagreement factors should not be “more factor-only ablations.”

The better next moves are:

- per-factor cross-model topology
- continuous per-factor mixture routing
- better shared-space learning

## 11. GCCA Shared Geometry Is Real, But Not Yet Enough

The first real multi-view shared-space artifact is:

- [qwen_phi_olmo_gcca_shared_geometry_summary_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_gcca_shared_geometry_summary_v1.json)

Key facts:

- latent dim: `16`
- input dim: `64`
- layers built: `12`

Mean view residuals:

- `Qwen`: `0.5164368748664856`
- `OLMo`: `0.4883084297180176`
- `Phi`: `0.49984610080718994`

Best-aligned model by layer:

- `Qwen`: `2` layers
- `Phi`: `4` layers
- `OLMo`: `6` layers

Layer-12 pairwise aligned cosine means:

- `Qwen <-> OLMo`: `0.846746563911438`
- `Qwen <-> Phi`: `0.7462196350097656`
- `OLMo <-> Phi`: `0.7249842882156372`

This is a real improvement in architectural quality over the earlier ad hoc factor-PCA lane:

- it is truly multi-view
- it yields per-model projections into a shared latent
- it produces stable per-layer residuals and alignment diagnostics

So GCCA is a real foundation artifact.

What it is not, yet, is a proof of functional interchangeability.

## 12. The First Stitching Smoke Changed The Interpretation

The first real functional verification smoke is:

- [qwen_phi_olmo_stitching_cohort_smoke_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_stitching_cohort_smoke_v1.json)
- [qwen_phi_olmo_gcca_vs_stitch_smoke_v1.json](/home/zaytor/transformer_research/parameter-golf/research/representation_learning/reports/qwen_phi_olmo_gcca_vs_stitch_smoke_v1.json)

Setup:

- 3 models: `Qwen`, `Phi`, `OLMo`
- shared layer: `12`
- ordered pairs: `6`
- examples: `32`
- method: last-hidden affine stitching into the target model’s own output head

The smoke result:

- mean eval hidden cosine: `0.7391046682993571`
- mean eval target-logit KL: `6.033543507258098`
- mean eval target top-1 agreement: `0.0`

Best pair by target-logit KL:

- `OLMo -> Phi`: KL `5.105254650115967`, hidden cosine `0.8498367071151733`

Pairwise joined GCCA/stitch summary:

- aligned-latent cosine vs logit-KL correlation: `0.5069204252389741`
- mean pair residual vs logit-KL correlation: `-0.34402678824263927`

This is the most important recent interpretive shift in the entire project.

The stitching smoke says:

- there is meaningful shared hidden geometry
- but current shared geometry is not yet functionally interchangeable at the logit level
- better latent alignment does not yet cleanly imply better downstream functional transfer

In practical terms:

- the lane cannot treat shared-space similarity as equivalent to transferability
- shared geometry is necessary but currently not sufficient

This sharply narrows the main open problem:

- not “is there shared geometry?”
- but “what transform turns shared geometry into functionally consumable intermediate states?”

## 13. What Is Strongly Validated

The following claims are now strongly supported by the evidence:

### A. Shared external language structure exists across heterogeneous models

Support:

- chunk-loss correlations around `0.945` to `0.964`

### B. Internal representational strategies differ materially

Support:

- subspace overlaps around `0.243` to `0.258`

### C. Different source models own genuinely different parts of the surface

Support:

- named concept ownership split
- `Qwen` vs `OLMo` advantage asymmetry
- `OLMo` broad regime vs `Qwen` weird-tail regime

### D. Simple chimeric merge can produce positive signal

Support:

- `Qwen+OLMo` activation merge beats the best single source in the toy evaluation

### E. Learned routing is much better than static routing

Support:

- ecology `full` about `0.91`
- static routing kernel about `0.24`

### F. Geometry quality matters directly

Support:

- `Phi` backfill materially improved the ecology system

## 14. What Is Partially Validated

The following claims look plausible but are not yet established enough to treat as solved:

### A. Structure-only routing can recover a large fraction of the oracle ecology

Current status:

- best structure-only lane is about `0.52`
- full oracle lane is about `0.91`

Interpretation:

- some non-oracle routing is real
- but most of the winner signal is still missing

### B. Per-factor decomposition is the right upgrade over scalar summaries

Current status:

- disagreement factors improved some structural lanes
- but did not close the main gap

Interpretation:

- the direction is likely right
- the current implementation is still only partial

### C. Sequence-aware/static moonshot merge is better than flat merge

Current status:

- true in the 2-model toy lane
- not true in the 3-model toy lane

Interpretation:

- promising control branch
- not yet the mainline merger operator

### D. GCCA is the right shared-space substrate

Current status:

- it is a better foundation artifact than earlier ad hoc geometry
- but current layer-12 affine stitching does not yet show strong functional interchangeability

Interpretation:

- GCCA is good enough to continue
- not good enough yet to call the transfer problem solved

## 15. What Is Not Yet Validated

The decisive claims that remain unresolved are:

### A. Real target-model benefit

We still do not have a durable, collated answer to:

- does merged geometry improve the real `9x512` target model under the actual training recipe?

The generated manifests for that lane exist, but the final collated artifacts were not located in the repo during this synthesis.

### B. Functional interchangeability at a meaningful level

The stitching smoke says:

- moderate hidden cosine is not enough
- top-1 agreement is still `0.0`
- logit KL is still high

So the project does not yet have evidence that current cross-model representations can be swapped in a nearly plug-compatible way.

### C. Continuous factor mixture clearing

The project has not yet built and evaluated:

- per-factor mixture routing as the main ecology target

Current ecology remains primarily winner prediction, not factor-wise clearing.

### D. Student compiler benefit

We do not yet have:

- a single student trained against a learned routing/clearing kernel
- evidence that such a student beats the best source under fixed budget

## 16. The Main Bottlenecks Now

The current bottlenecks are not vague.

### Bottleneck 1. Shared geometry is not yet functionally consumable enough

Evidence:

- GCCA looks good geometrically
- stitching is only modestly good in hidden cosine
- stitching is still poor at the logit level

So the current bottleneck is not just “better shared space” in the abstract. It is:

- finding the right transfer operator from shared space into target-consumable function

### Bottleneck 2. Non-oracle routing is still too weak

Evidence:

- best non-oracle structure-only route is about `0.52`
- full winner oracle ecology is about `0.91`

So the current kernel still does not know enough about structural state without loss-derived features.

### Bottleneck 3. Scalar summaries are the wrong abstraction

Evidence:

- forward signature sidecars mostly regressed the ecology model

So the path forward is not more scalar feature engineering. It is richer high-dimensional or per-factor structure.

### Bottleneck 4. The real target benchmark is still unresolved

Until the real `9x512` training comparison is durable and collated, the project still lacks its final product-level metric.

## 17. What Should Be Pruned Or De-Prioritized

The following branches should now be clearly secondary:

- more projected-space gauge-fix iteration
- more scalar forward-signature tuning
- more pure cascade direct-merger tuning
- more named-probe expansion as a mainline activity
- more toy merger sweeps that do not answer a concrete gating question

These are not all useless, but they should not receive frontier budget ahead of:

- functional verification
- non-oracle kernel improvement
- real target-model benefit

## 18. The Most Important Decision Rules From Here

The next phase should be governed by a few explicit decision rules.

### Rule 1. Treat geometry and function as separate tests

Do not infer transferability from CCA/GCCA similarity alone.

If a shared-space branch improves geometry metrics but not stitching, it has not solved the important problem.

### Rule 2. Use stitching to judge the shared-space branch

The right next evaluation for shared-space improvements is:

- do they lower stitch KL?
- do they improve top-1 agreement?
- do they improve transfer asymmetry in the right pairs?

### Rule 3. Use the ecology lane to replace static routing, not just to imitate it

The routing table is now proven to be a bootstrap object.

The kernel must be learned.

### Rule 4. Use per-factor mixture routing as the next kernel target

Winner classification has taken the lane far enough.

The next kernel architecture should predict:

- per-factor mixture weights
- not only one winner label per chunk

### Rule 5. Recover the real target benchmark as soon as practical

Without the real `9x512` result, the merger/product question remains open.

## 19. The Highest-Value Next Experiments

In order:

1. Full stitching cohort, not just the smoke.
   - Sweep more layers, not only shared layer `12`
   - Determine whether transferability is layer-local or broadly weak

2. Improve shared-space construction only if stitching justifies it.
   - activation + Jacobian multi-view geometry
   - nonlinear DGCCA-style shared space if linear GCCA remains too weak

3. Promote ecology from winner classification to factor mixture routing.
   - use disagreement/topology factors
   - predict `factor x source` weights

4. Reconstruct or rerun the missing real `9x512` benchmark outputs.
   - baseline
   - best single source
   - flat merge
   - strongest sequence-aware / moonshot baseline

5. Only after the above, distill a student from the learned clearing kernel.

## Final Assessment

The representation-learning lane is not speculative anymore.

It has already demonstrated:

- shared-but-different geometry across real open models
- genuine complementary model ownership
- a small positive chimeric merge signal
- a strong learned-ecology advantage over static routing

At the same time, the newest results force a more precise statement of what remains hard:

- current shared geometry is not yet functionally interchangeable enough
- current non-oracle routing is not yet strong enough
- current scalar forward summaries are not the right abstraction

So the project is neither “solved” nor “stuck.”

It is in the phase where the important uncertainties are now narrow and concrete:

- how to turn shared geometry into functionally consumable transfer
- how to route without oracle loss
- how to show target-model benefit on the real benchmark

That is a good state for the project to be in. The scientific premise is validated. The remaining work is now primarily an architecture-quality and execution problem.
