# Hardmax Research Questions For Next Iteration

This note converts the latest external-literature memo into a concrete question set for the next research-agent pass.

It is anchored to the current lane state:

- [hardmax_structural_controller_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/hardmax_structural_controller_lane_20260403.md)
- [execution_trace_hardmax_lane_20260403.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/execution_trace_hardmax_lane_20260403.md)

The most important current empirical fact is:

- old shallow hardmax supervision collapsed
- execution-trace supervision kept the `8`-state controller materially alive in the first smoke

So the next research pass should stop asking “can hardmax help at all?” and instead answer:

- what exact training signal makes the controller stay alive?
- what architecture is most faithful to the literature while still trainable?
- what is genuinely novel vs already solved elsewhere?

## Priority 0: Questions That Directly Change Engineering

1. Which trace-supervision results are the strongest apples-to-apples precedents for our lane?
   We need the most concrete papers / repos where:
   - a model is trained on explicit execution traces
   - the trace contains machine state, not just natural-language rationales
   - improvements transfer to tasks without traces at test time

2. What trace representation is best supported by prior work?
   Compare:
   - line-level debugger traces
   - bytecode / opcode traces
   - abstract VM state
   - memory deltas
   - natural-language execution rationales distilled from traces

   The question is not “what exists,” but:
   - what representation improved learning most consistently?
   - what representation generalized best?
   - what representation is easiest to synthesize at scale?

3. Is there prior evidence that same-model mixed deterministic/probabilistic heads stay functionally distinct?
   We need concrete precedents for:
   - same-layer heterogeneous heads
   - mechanisms used to stop soft heads from silently re-implementing hard logic
   - whether per-layer supervision or ablation leverage was sufficient to enforce separation

4. What is the cleanest soft-to-hard training schedule in adjacent work?
   We need direct comparisons between:
   - straight-through hard assignments from step 0
   - Gumbel / Concrete relaxations
   - annealed temperature schedules
   - distillation from soft heads into hard heads

5. Where does Percepta’s WASM-executing transformer sit in the evidence hierarchy?
   We should answer:
   - what exactly is publicly documented?
   - what is proved constructively vs merely claimed?
   - what parts of the architecture are reusable for us?
   - what parts depend on analytically computed weights rather than SGD?

## Priority 1: Questions About Training Data

6. What existing datasets or pipelines already produce aligned code-execution traces at scale?
   Target data sources:
   - Python debugger traces
   - compiler IR traces
   - bytecode traces
   - WASM traces
   - small-step interpreter traces

   We should identify:
   - public corpora
   - trace generators
   - legal/operational constraints
   - whether source + trace + output alignments are already packaged

7. What multiview trace stacks have actually been tried?
   We are interested in joint training on:
   - source
   - AST
   - bytecode / IR / WASM
   - execution trace
   - memory state
   - final output

   We need to know:
   - whether anyone trained on all views jointly
   - whether benefits saturated after two or three views
   - whether some views hurt by introducing redundancy/noise

8. What machine-state components are most predictive and worth supervising first?
   Candidate targets:
   - program counter / next opcode
   - stack state
   - variable bindings
   - memory reads/writes
   - branch outcomes
   - exceptions / halt state

   We should look for:
   - ablations from prior work
   - evidence that some targets are enough to recover the rest
   - cases where richer state hurts due to sparsity or overfitting

9. How synthetic can the corpus be before transfer breaks?
   The current lane is using a tiny synthetic VM.
   We need evidence about transfer from:
   - toy DSLs
   - restricted Python subsets
   - full language/runtime traces

   The key question is:
   - what synthetic regime is the smallest one that still transfers to real code reasoning?

## Priority 2: Questions About Architecture

10. What is the best architectural locus for hard computation?
   Compare the literature support for:
   - sidecar controller at one layer
   - repeated conditioning at multiple layers
   - alternating hard/soft sublayers
   - same-layer mixed hard/soft heads
   - separate hard verifier tower

11. What exact role should hard heads play?
   We should separate:
   - state tracking
   - execution / verifier state transition
   - compute routing
   - type checking / error detection
   - memory addressing

   The next pass should ask which of these roles is already validated in the literature and which are still speculative.

12. How should we define “alive” for a hard controller?
   The literature review should collect operational criteria for:
   - state usage distribution
   - ablation leverage
   - causal necessity
   - inability of soft heads to bypass the mechanism
   - confidence calibration

13. What is the best way to prevent Hydra / bypass circuits?
   This is one of the most important questions for our design.
   We should specifically ask:
   - does per-layer supervision work in practice?
   - do auxiliary verifier losses help?
   - do bottlenecks or attention masks help?
   - do distillation and pruning reveal whether soft heads have copied the hard logic?

## Priority 3: Questions About Mechanistic Validation

14. In arithmetic and code tasks, what are the strongest causal tests that a model is executing a real internal algorithm?
   We should collect methods such as:
   - causal patching
   - ablation of carry/state heads
   - hidden-state linear probes for intermediate values
   - interventions on digit- or variable-specific heads

15. Is “one layer = one computational step” actually supported, or is it mostly metaphor?
   This matters for whether we want:
   - deeper interleaving
   - recurrent reuse of layers
   - controller recurrence separate from trunk depth

16. What evidence exists for program-counter-like or stack-like states in larger models?
   Not toy symbolic models only, but:
   - mid-sized or frontier code models
   - arithmetic models with mechanistic analysis
   - chain-of-thought models where explicit traces become latent

17. How should we test whether trace-pretrained hard states transfer into the LM lane?
   The next research pass should propose evaluation protocols for:
   - state usage before/after transfer
   - whether transfer preserves non-collapse
   - whether transferred states align with boundaries, operators, and code-execution structure
   - whether transferred states improve BPB / NLL beyond the current structural adapter

## Priority 4: Questions About Evaluation

18. What benchmark suite best tests the intended decomposition?
   We need a shortlist of tasks across:
   - arithmetic
   - formal languages / automata
   - code execution
   - theorem proving / verifier-style reasoning

   The main goal is not leaderboard breadth but coverage of:
   - exact state propagation
   - stack-like control
   - verifier/checker behavior
   - transfer from explicit traces to latent execution

19. How should we measure the structural vs inferential decomposition directly?
   The literature pass should look for prior attempts to decompose:
   - deterministic/mechanical next-token structure
   - uncertain/semantic residual

   We especially want metrics or experimental designs that could support:
   - per-factor perplexity
   - structural vs content entropy
   - machine-state predictability vs token predictability

20. What is the strongest negative test?
   We should identify tasks where:
   - pure softmax models look good behaviorally but fail mechanistically
   - shallow pattern matching can be separated from true state tracking
   - external verifier feedback and trace supervision should matter if the architecture is real

## Priority 5: Questions About Positioning / Novelty

21. What exactly is novel in our design relative to the literature?
   Candidate novelty claims:
   - same-model interleaving of hard execution heads and soft reasoning heads
   - training hard heads on machine-state traces, then transferring into a language-model trunk
   - using hard heads as an always-on execution substrate rather than only an external verifier
   - testing whether the verifier/runtime can be internalized rather than bolted on

   The next pass should determine which claim is defensible and narrow enough.

22. Which nearest neighbors should we explicitly compare against in any writeup?
   At minimum:
   - Percepta / WASM-executing transformer
   - trace-supervised code models such as CodeExecutor / NExT / CWM
   - proposer/verifier neuro-symbolic systems such as Logic-LM, AlphaGeometry, lambda-RLM
   - heterogeneous-head or adaptive-compute architectures

23. What would count as a decisive positive result?
   The next research pass should answer this before more engineering:
   - non-collapsed hard states under realistic supervision
   - transfer into the LM lane with better BPB/NLL than the best static adapter
   - causal evidence that the hard states matter
   - evidence that soft heads did not just reimplement the same logic

## Questions For A Literature-Collection Agent

The next literature agent should be asked to return concrete answers to:

1. Which papers provide the strongest direct evidence that execution traces outperform source/output supervision for code reasoning?
2. Which papers or repos use actual debugger, bytecode, IR, or WASM traces rather than natural-language chains of thought?
3. Which works study same-layer heterogeneous heads or mixed hard/soft attention in one model?
4. Which training tricks are most effective for keeping discrete components from collapsing?
5. Which works explicitly test whether soft pathways bypass the intended modular component?
6. Which datasets or generators are available for multiview code traces with machine state?
7. Which evaluation protocols best distinguish state tracking from shallow pattern matching?
8. Which mechanistic tools are most accepted for showing that a model implements variable binding, stack tracking, or arithmetic carries?
9. Which results support transfer from explicit external traces to implicit hidden-state computation?
10. Which parts of the Percepta system are public enough to treat as a reproducible engineering precedent rather than only inspiration?

## Questions For The Next Engineering-Oriented Research Pass

The next engineering-oriented agent should answer:

1. What is the smallest useful transfer experiment from the trace-pretrained controller into the `structonly` LM lane?
2. Should the first transfer initialize:
   - the full hard controller
   - only the state book
   - only the input/output projections
   - only auxiliary heads
3. What is the exact metric suite for the transfer?
   - BPB / NLL
   - state usage
   - confidence variability
   - residual ACF
   - boundary-binned evaluation
4. What ablation cleanly tests whether trace pretraining is better than just giving the adapter more capacity?
5. Which next trace family should be implemented first:
   - nested control flow
   - exceptions
   - Python bytecode
   - WASM / VM memory operations

## Current Recommendation

The next research-agent iteration should spend most of its time on five things:

1. Trace-supervision precedents that use real machine state.
2. Same-layer heterogeneous-head precedents and anti-bypass mechanisms.
3. Discrete-component training schedules that avoid collapse.
4. Datasets / generators for multiview execution traces.
5. Tight novelty positioning around “internalized proposer/verifier with explicit hard execution heads.”

That is the question set most likely to change the next engineering decision, not just expand the bibliography.
