# Project-Wide Notes

Use this folder for durable notes that should survive beyond any single run.

Current contents:

- `roadmap.md`: the operating model for this repository and the initial research priorities.
- `parameter_golf_architecture_and_hyperparameters.md`: a line-by-line baseline review of the starter scripts, including likely rationale and common alternatives.
- `curriculum_learning_lane.md`: the component-aware curriculum hypotheses, cautions, and implementation ladder.
- `test_time_adaptation_and_adaptive_depth_lane.md`: a split-lane review of adaptive depth, test-time adaptation, and how those interact with the existing curriculum and tokenizer work.
- `representation_learning_engineering_plan.md`: the implementation plan for extracting compact geometric priors from public teacher models and injecting them into the current training stack.
- `turboquant_agent_handoff_combined_challenger_20260329.md`: promotion-path handoff for the TurboQuant agent, including what should enter the combined challenger, what should remain side branches, and what should stay out.
- `branching_and_early_exit_ablation_lane.md`: the current causal-valid branching, mid-stack early-exit, and latent-state ablation ladder, including what is valid in training versus eval and how the live runs map onto that ladder.
- `architecture_backpass_20260330.md`: a results-driven architecture review of what is actually validated, what to stop doing, and which refinements should be built next while current promotion runs are still in flight.
- `autonomous_architecture_execution_plan_20260330.md`: the long-running execution plan for architecture work, result analysis, implementation cleanup, and autonomous stop/go rules across the mainline, branching, sidecar, and eval-adaptation lanes.
- `sidecar_canonicalization_plan_20260330.md`: the canonical sidecar decision, why chunk-causal is the main line, what remains research-only, and how future sidecar work should be routed.
- `residual_autocorrelation_framework_20260330.md`: residual-direction autocorrelation for saved language-model artifacts, regime-split ACF, transition examples, and how to interpret persistent residual structure.
- `fineweb_prosody_diagnostics_lane_20260331.md`: the first FineWeb text-native prosody analysis lane, including token-class loss decomposition, boundary-conditioned loss, and lightweight hidden-state probes for quote/boundary state.
- `fineweb_prosody_feature_state_lane_20260401.md`: the stronger FineWeb prosody architecture lane, including factorized token features, the exportable prosody state adapter, and the linked ablation ladder.

Environment status:

- A local Linux `.venv` has been created in the repo and is being populated from `requirements.txt`.
- Apple Silicon MLX setup cannot run natively on this Linux host, so cluster bootstrap is provided via `tools/setup_mlx_env.sh` and `requirements-mlx.txt`.
