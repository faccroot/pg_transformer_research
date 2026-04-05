# Test-Time Adaptation And Adaptive-Depth Lane

This note reviews three adjacent ideas that should not be merged blindly into the curriculum lane:

- hierarchical or morphology-aware tokenization
- adaptive depth or halting on shared recurrent blocks
- evaluation-time adaptation or model growth

The common theme is "use compute more intelligently than the baseline," but they touch different codepaths and have different competition-risk profiles.

## Current principal variation

The update to this lane is:

- token-level helper or adaptation updates are now effectively pruned
- the next useful adaptation forms are:
  - chunk-level updates
  - query-only updates
  - hard-span-triggered updates

So the lane should stop asking:

- "can we update a little every few tokens?"

and start asking:

- "what is the largest, cheapest, most selective update block that actually changes long-context behavior?"

## Immediate conclusions

### 1. H-net and morphology-aware tokenization are real BPB levers, but they belong in the tokenizer lane.

`val_bpb` is normalized by raw bytes, so a tokenizer change is allowed to win if the byte accounting remains correct. In that sense your framing is right: better tokens per byte can reduce the number of predictive events per byte.

The repo constraint is practical rather than theoretical:

- tokenizer changes are easy to mis-measure if the byte accounting or shard export path is wrong
- the current curriculum lane already depends on `sp1024`-specific operator detection and cached shards
- a tokenizer swap would invalidate some of the current offline curriculum artifacts

Near-term implication:

- `competitive path`: stay on the current SentencePiece family and only test conservative extensions that do not destabilize the stack
- `novelty path`: morphology-aware or hierarchical tokenization should remain a separate research lane until byte-accounting and shard rebuild tooling are explicit

This aligns with [tokenizer_lane.md](/home/zaytor/transformer_research/parameter-golf/research/project_wide/tokenizer_lane.md).

### 2. Adaptive depth is promising, but the first useful step is fixed shared recurrence, not learned halting.

The current model already has block sharing via `NUM_LAYER_TEMPLATES`; see [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py#L169) and [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py#L723). What it does not have is:

- per-token halting
- wave-based selective recurrence
- a trainable halting head
- any eval-only extra-depth control

So the practical ladder is:

1. fixed shared recurrence with explicit template reuse
2. eval-only extra recurrence depth on the same shared templates
3. chunk-level or phase-level depth scheduling
4. only then token-level halting or ACT-style mechanisms

The reason is simple: fixed extra depth tests the "test-time compute helps" thesis without introducing ragged execution, halting losses, or routing bugs.

Your sidecar-based halting idea is still useful, but it should come after we have evidence that extra recurrent steps help at all on this stack.

### 3. Evaluation-time growth is a separate non-record system, not an extension of curriculum training.

The current evaluation path is stateless: [eval_val()](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py#L1926) dispatches to non-overlapping or sliding-window validation and carries no persistent memory bank, no adaptation state, and no inner-loop optimizer.

That means evaluation-time training or memory accumulation should be implemented as a separate evaluator or wrapper, not as another branch inside the normal trainer flags.

The safest first slice is:

- no architecture mutation
- no full-model optimizer state
- adapt only a small sidecar or low-rank residual
- update only on already-scored validation tokens
- keep it opt-in and out of the default record path

Updated interpretation:

- if we pursue helper-worker adaptation here, the default form should be large chunk or query-only updates
- token-scale online updates are now the negative control, not the main bet

This is the cleanest way to test the legal and practical edge you described without tangling it into the standard `train_gpt_mlx.py` training loop.

### 3a. First empirical result: naive persistent sidecar carryover is not enough on the current checkpoints.

We now have a direct reset-vs-persistent evaluation on saved JEPA-sidecar artifacts using the opt-in evaluator at [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py) and the probe logs in [20260326_sidecar_eval_persistence_probe](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_sidecar_eval_persistence_probe).

Matched `512`-token evaluation on the best short detector-written sidecar checkpoint:

- reset: `1.99294778` BPB
- persistent: `1.99345176` BPB
- delta: `+0.00050398` BPB worse

Matched `512`-token evaluation on the plain sidecar reference checkpoint:

- reset: `2.00630008` BPB
- persistent: `2.00713968` BPB
- delta: `+0.00083960` BPB worse

Interpretation:

- simple cross-chunk state carryover by itself is not a free win
- if evaluation-time growth is going to help, it likely needs targeted adaptation or retrieval rather than just "do not reset the recurrent state"
- this pushes the next eval-time slice toward sidecar-only adaptation on already-scored chunks, not toward blindly enabling persistence everywhere

### 3b. Second empirical result: BOS-aware sidecar reset also failed on the saved short checkpoints.

The export path is not fully boundary-blind: each document is prefixed with `BOS=<s>` during tokenization, but the old JEPA-sidecar probes still carry recurrent state across chunk boundaries unless the evaluator resets it. To test the narrow "state contamination at boundaries" hypothesis directly, I added opt-in `SIDECAR_RESET_ON_BOS` handling to both the current JEPA sidecar loader and the older compatibility loader, then re-ran the saved short-checkpoint persistence probe on the artifact family that still matches the older loader.

Probe folder: [20260326_sidecar_eval_bos_reset_probe](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260326_sidecar_eval_bos_reset_probe)

Matched `512`-token evaluation on the detector-written checkpoint:

- old reset: `1.99294778`
- old persistent: `1.99345176`
- BOS-reset reset: `1.99481329`
- BOS-reset persistent: `1.99525541`

Matched `512`-token evaluation on the plain sidecar reference checkpoint:

- old reset: `2.00630008`
- old persistent: `2.00713968`
- BOS-reset reset: `2.00800817`
- BOS-reset persistent: `2.00882176`

Interpretation:

- the boundary issue is not "there is no document marker" because `BOS` already exists
- coarse BOS-aware recurrent resets did not recover BPB on the saved chunk-sidecar family
- the persistence gap shrank slightly, but both reset and persistent scores got worse overall
- this makes a strong case against spending more time on simple reset heuristics

So the boundary lane is now:

- `tested and negative`: naive persistence
- `tested and negative`: BOS-aware recurrent reset
- `still worth testing`: post-score adaptation, retrieval-style memory, or a truly token-causal sidecar family trained with the reset rule from the start

## What maps cleanly onto the current stack

### Good near-term fits

- Curriculum-conditioned depth schedule at the chunk or phase level
- Extra eval-only recurrence steps on shared templates
- Sidecar-only or LoRA-like test-time adaptation on scored validation chunks
- query-only or large-chunk helper updates
- Cross-chunk memory bank in a separate evaluator

### Medium-term fits

- MTP-confidence-driven depth control
- Learned halting based on sidecar state
- Population or thicket-style evaluation over a small family of quantized variants

### Poor immediate fits

- Full tokenizer replacement inside the current curriculum experiments
- Per-token ragged halting in the base MLX trainer
- Architecture growth inside the main training script
- token-level helper updates as the default adaptation form

## Recommended experiment order

### A. Keep the curriculum lane focused

- Finish the logic-enabled curriculum A/B sweep.
- Re-run the plain curriculum A/B at a longer wallclock once the smoke ranking is stable.

### B. Start a bounded recurrence lane

- Introduce a small recurrent-template sweep:
  - e.g. `NUM_LAYERS=12`, `NUM_LAYER_TEMPLATES=3 or 4`
- Add an eval-only "extra depth" control that reuses the same templates for a few additional passes during validation.
- Measure `val_bpb` against fixed compute before attempting halting.

### C. Build a separate eval-time adaptation surface

- Create a dedicated evaluation wrapper rather than modifying the baseline evaluator in-place.
- The first prototypes now exist:
  - [eval_saved_sidecar.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_sidecar.py) for persistent-vs-reset sidecar eval plus optional sidecar-only adaptation on already-scored chunks
  - [eval_saved_structural.py](/home/zaytor/transformer_research/parameter-golf/tools/eval_saved_structural.py) for structural analysis plus bounded score-first branch adaptation
- Current operating rule:
  - keep adaptation parameter scope small
  - keep update steps bounded
  - keep all of this out of the record-path trainer
  - compare evaluator lanes using explicit adaptation counts and post-rescore deltas, not just raw first-pass BPB
- Only after that works should we test:
  - low-rank adapters
  - memory-bank conditioning
  - small population search over quantized variants

### D. Leave H-net tokenization for the tokenizer novelty lane

- Keep the current `sp1024` path as the curriculum control.
- If we want a competitive tokenizer experiment soon, do a conservative standard-BPE or bigram-style extension first.
- Keep morphology-aware or hierarchical tokenization as a documented follow-up, not the next blocking change.

## Where the cited papers matter

The references you pasted support the direction, but they slot into different lanes:

- `TTT-E2E` supports treating evaluation-time adaptation as a first-class modeling choice rather than a hack.
- `SDPO` is most relevant if we create richer feedback or self-distillation loops for adaptation, not as a direct drop-in for current next-token training.
- `Neural Thickets` is most relevant to a later evaluation-time population search or perturb-and-select lane.

Those should guide design, but they do not change the immediate implementation order above.

## Bottom line

The stack should stay modular:

- curriculum remains a training-data orchestration lane
- tokenizer work remains a tokenizer lane
- adaptive depth becomes a bounded recurrence lane
- evaluation-time growth becomes a separate non-record evaluator

That separation is what keeps us from accidentally turning every `train_gpt_mlx.py` run into a moving target.
