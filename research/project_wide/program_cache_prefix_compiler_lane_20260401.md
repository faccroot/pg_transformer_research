# Program Cache Prefix Compiler Lane

This note records the current state of the "program cache" / synthetic-prefix memory branch as of `2026-04-01`.

## Thesis

The best object to carry across windows is not the raw transcript and not unadapted hidden states. It is a compact **compiled prefix state** optimized to reduce future prediction error.

Operationally:
- `raw_prev_all` is a useful upper-bound control for transcript carry.
- unadapted hidden-prefix carry is a negative control.
- a learned compiler from previous-window hidden states into a small prefix bank is the real branch.

## Current evidence

### 1. Front-of-window loss headroom is real

On the frozen untied Turbo baseline checkpoint:
- [position_loss_1024_w4.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/position_loss_1024_w4.json)

At `seq_len=1024`:
- `0-31`: `7.5051`
- `32-63`: `7.3975`
- `64-127`: `7.2823`

So the first tokens in a window are materially worse than the settled region.

### 2. Carry headroom exists, but naive latent prefixes fail

Files:
- [window_carry_1024_w4_oracle.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/window_carry_1024_w4_oracle.json)
- [window_carry_1024_w4_hidden_prev_all.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/window_carry_1024_w4_hidden_prev_all.json)

Key result:
- `raw_prev_all` improves `0-31` by `-0.0348` NLL
- `hidden_prev_all` worsens `0-31` by `+0.0043`

Interpretation:
- the model can use the previous transcript
- it cannot directly use arbitrary hidden states as prefix memory
- the problem is interface/compilation, not lack of memory headroom

### 3. Oracle synthetic prefixes are extremely strong

Files:
- [optimized_prefix_512_w3_s4_t10.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/optimized_prefix_512_w3_s4_t10.json)
- [optimized_prefix_512_w3_s16_t10.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/optimized_prefix_512_w3_s16_t10.json)
- [optimized_prefix_512_w3_s32_t10.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/optimized_prefix_512_w3_s32_t10.json)

Mean `0-31` delta vs no-carry:
- `4` slots: `-0.9163`
- `16` slots: `-1.8210`
- `32` slots: `-2.4984`

These are oracle upper bounds, not legal eval procedures. They show the model can exploit compact synthetic prefixes if they are compiled correctly.

### 4. First learned compiler smoke is positive

Validation-only smoke:
- [prefix_compiler_256_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_256_smoke.json)

Held-out mean deltas vs no-carry:
- `0-31`: compiler `-0.7927`, raw previous `-0.0397`
- `32-63`: compiler `-0.4318`, raw previous `-0.0101`

This established that a learned compiler can beat the transcript carry control on held-out windows.

### 5. Train-vs-val generalization also held

Files:
- [prefix_compiler_trainval_256_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_trainval_256_smoke.json)
- [prefix_compiler_trainval_512_smoke.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_trainval_512_smoke.json)

At `seq_len=256`:
- `0-31`: compiler `-0.8309`, raw previous `-0.0385`
- `32-63`: compiler `-0.5743`, raw previous `-0.0350`

At `seq_len=512`:
- `0-31`: compiler `-0.8249`, raw previous `-0.1278`
- `32-63`: compiler `-0.3701`, raw previous `-0.0321`
- `64-127`: compiler `-0.4244`, raw previous `+0.0124`

This is the first real signal that the branch generalizes beyond tiny validation overfit.

### 6. Typed control is competitive with the free-form compiler

Files:
- [prefix_compiler_trainval_512_typed_b8_k8.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_trainval_512_typed_b8_k8.json)
- [prefix_compiler_trainval_512_typed_b8_k2.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_trainval_512_typed_b8_k2.json)
- [prefix_compiler_trainval_512_typed_b8_k2_h1.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_smoke/prefix_compiler_trainval_512_typed_b8_k2_h1.json)

Compared to the free-form compiler at `seq_len=512`:

- free-form:
  - `0-31`: `-0.8249`
  - `32-63`: `-0.3701`
  - `64-127`: `-0.4244`
  - `128-255`: `-0.2320`

- typed, `8` blocks, no filter (`topk=8`):
  - `0-31`: `-0.8209`
  - `32-63`: `-0.3664`
  - `64-127`: `-0.4435`
  - `128-255`: `-0.2509`

- typed, `8` blocks, top-2 filter:
  - `0-31`: `-0.8012`
  - `32-63`: `-0.3407`
  - `64-127`: `-0.4063`
  - `128-255`: `-0.2054`

- typed, `8` blocks, top-2 filter, hierarchical summaries:
  - `0-31`: `-0.8277`
  - `32-63`: `-0.3649`
  - `64-127`: `-0.4316`
  - `128-255`: `-0.2371`

Interpretation:
- typed control is not hurting the branch
- the basic typed split/reduce runtime is already competitive with the free-form compiler
- aggressive filtering (`topk=2`) loses some quality on this small run
- adding a second summary scale recovers most of that loss and slightly improves the very front (`0-31`)

So the current best read is:
- do not promote hard top-k filtering by itself
- do promote the typed compiler family
- prioritize hierarchical typed summarization over stronger pruning

### 7. Persisted small-budget artifacts reproduce the branch

To make the lane evaluable beyond one-off smokes, the ablation runner now saves
compiler artifacts that can be reloaded and scored later.

Files:
- [prefix_compiler_full_s32.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32.json)
- [prefix_compiler_full_s32.pt](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32.pt)
- [prefix_compiler_typed_b8_k8.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k8.json)
- [prefix_compiler_typed_b8_k8.pt](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k8.pt)
- [prefix_compiler_typed_b8_k2_h1.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1.json)
- [prefix_compiler_typed_b8_k2_h1.pt](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1.pt)

These runs use a smaller CPU-safe budget:
- `train_windows=6`
- `eval_windows=6`
- `train_steps=20`

Even under that reduced budget, the branch remains strong:

- full `s32`:
  - `0-31`: `-0.7411`
  - `32-63`: `-0.4943`
  - `64-127`: `-0.4976`
- typed `b8,k8`:
  - `0-31`: `-0.7450`
  - `32-63`: `-0.4597`
  - `64-127`: `-0.4105`
- typed `b8,k2,h1`:
  - `0-31`: `-0.7556`
  - `32-63`: `-0.4783`
  - `64-127`: `-0.3977`

So the compiler signal is not an artifact of the larger earlier smoke budget.

### 8. Transition-binned evaluation does not show the expected stale-carry failure under the current heuristic

New files:
- [prefix_compiler_full_s32_transition_bins.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_transition_bins.json)
- [prefix_compiler_typed_b8_k8_transition_bins.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k8_transition_bins.json)
- [prefix_compiler_typed_b8_k2_h1_transition_bins.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_transition_bins.json)
- [prefix_compiler_typed_b8_k2_h1_hidden_transition_bins.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_hidden_transition_bins.json)

The analyzer bins windows into:
- `stable`
- `near_boundary`
- `post_boundary`

using tokenized prosody proxies around the window seam.

For the strongest free-form artifact (`full_s32`), front-band `0-31` deltas are:
- `stable`: `-0.5189`
- `near_boundary`: `-0.8367`
- `post_boundary`: `-0.7864`

For the strongest typed candidate so far (`typed_b8_k2_h1_hidden`), `0-31` deltas are:
- `stable`: `-0.5224`
- `near_boundary`: `-0.7970`
- `post_boundary`: `-0.7396`

Interpretation:
- the compiler gains do **not** flip sign at the current boundary bins
- under this heuristic, the carried state is helping at boundaries rather than becoming obviously anti-predictive
- if there is a real transition brittleness problem here, the current prosody-only seam heuristic is too weak or too indirect to isolate it

This is an important correction to the earlier expectation that compiler carry would necessarily collapse post-transition.

### 9. Naive boundary-driven flushing is bad

Files:
- [prefix_compiler_full_s32_transition_bins_reset_hard.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_transition_bins_reset_hard.json)
- [prefix_compiler_full_s32_transition_bins_reset_soft.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_transition_bins_reset_soft.json)

Using the same persisted `full_s32` artifact:

- no reset, `post_boundary`, `0-31`: `-0.7864`
- hard reset, `post_boundary`, `0-31`: `-0.0164`
- soft reset, `post_boundary`, `0-31`: `-0.7879`

The hard reset almost entirely destroys the gain on `post_boundary` windows.
The soft reset is nearly inert.

Interpretation:
- boundary-triggered flushing is **not** the missing ingredient for this lane under the current signal
- the compiler is carrying useful state across those seams
- any future reset mechanism should be based on stronger evidence than simple boundary priors

So the promotion read is now:
- do **not** promote naive hard reset
- do **not** assume a pause/boundary signal alone is a reliable stale-carry detector for this branch

### 10. Typed utility sweep favors hidden-first selection

Files:
- [prefix_compiler_typed_b8_k2_h1_hidden.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_hidden.json)
- [prefix_compiler_typed_b8_k2_h1_surprise.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_surprise.json)
- [prefix_compiler_typed_b8_k2_h1.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1.json)

For typed hierarchical `b8,k2,h1`:

- `hidden`:
  - `0-31`: `-0.7538`
  - `32-63`: `-0.4859`
  - `64-127`: `-0.4259`
  - `128-255`: `-0.2750`
- `hybrid`:
  - `0-31`: `-0.7556`
  - `32-63`: `-0.4783`
  - `64-127`: `-0.3977`
  - `128-255`: `-0.2444`
- `surprise`:
  - `0-31`: `-0.6702`
  - `32-63`: `-0.4552`
  - `64-127`: `-0.3530`
  - `128-255`: `-0.2246`

Interpretation:
- `surprise`-only selection is clearly worse
- `hidden` is the strongest typed selector overall
- `hybrid` only keeps a tiny edge on the very front band

So the best typed candidate is now:
- typed hierarchical `b8,k2,h1`, `utility_source=hidden`

### 11. Slot-structure probe: structure signal is real, but slot-role factorization is weak

New files:
- [prefix_compiler_full_s32_slot_structure.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_full_s32_slot_structure.json)
- [prefix_compiler_typed_b8_k2_h1_hidden_slot_structure.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/generated/20260401_program_cache_transition_bins/prefix_compiler_typed_b8_k2_h1_hidden_slot_structure.json)
- [tools/analyze_prefix_slot_structure.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_prefix_slot_structure.py)

This probe compares compiler-emitted prefix slots across windows and asks whether
they line up more by slot index or by local structural signature.

For the free-form `s32` compiler:
- same-slot, different-window cosine: `0.8412`
- different-slot, same-window cosine: `0.8727`
- same-slot, same-structure cosine: `0.9314`
- same-slot, different-structure cosine: `0.8341`
- nearest-neighbor same-slot rate: `0.1818`
- nearest-neighbor same-bin rate: `0.7614`
- nearest-neighbor same-structure rate: `0.6790`

For the typed hierarchical hidden-utility compiler:
- same-slot, different-window cosine: `0.9820`
- different-slot, same-window cosine: `0.9997`
- same-slot, same-structure cosine: `0.9956`
- same-slot, different-structure cosine: `0.9809`
- nearest-neighbor same-slot rate: `0.0000`
- nearest-neighbor same-bin rate: `0.9205`
- nearest-neighbor same-structure rate: `0.9148`

Interpretation:
- the compiler output is **not** primarily organized by slot index
- local structural signature matters more than slot identity
- the free-form compiler appears to carry a real structure-conditioned prefix state
- the typed hidden variant is much more compressed and close to slot collapse:
  slots within a window are nearly identical

So the current best mental model is:
- the free-form compiler is carrying shared structural scaffolding
- the typed compiler is carrying a cheaper but less factorized version of that scaffolding
- if we want a richer typed prefix bank, we likely need an explicit diversity/objective on slots, not just more selection logic

## Current code

Core utilities:
- [tools/torch_artifact_eval_utils.py](/home/zaytor/transformer_research/parameter-golf/tools/torch_artifact_eval_utils.py)

Diagnostics:
- [tools/analyze_window_position_loss.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_window_position_loss.py)
- [tools/analyze_window_carry.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_window_carry.py)
- [tools/optimize_synthetic_prefix.py](/home/zaytor/transformer_research/parameter-golf/tools/optimize_synthetic_prefix.py)

Compiler prototypes:
- [tools/train_prefix_compiler_smoke.py](/home/zaytor/transformer_research/parameter-golf/tools/train_prefix_compiler_smoke.py)
- [tools/train_prefix_compiler_ablation.py](/home/zaytor/transformer_research/parameter-golf/tools/train_prefix_compiler_ablation.py)
- [tools/analyze_compiler_transition_bins.py](/home/zaytor/transformer_research/parameter-golf/tools/analyze_compiler_transition_bins.py)

## Promotion read

Promote the lane to active challenger status.

What is now supported:
- warm-start loss is real
- transcript carry helps a bit
- raw hidden carry is the wrong interface
- compact synthetic prefixes have large oracle headroom
- a learned compiler already generalizes in small train-vs-val ablations
- persisted compiler artifacts can be reloaded and re-scored
- typed control remains competitive under persisted small-budget runs
- boundary-only hard resets are actively harmful

Current promoted candidates:
- free-form `s32`
- typed hierarchical `b8,k2,h1`, `utility_source=hidden`

## Next ablations

1. Larger-budget persisted comparison of the two promoted candidates
- free-form `s32`
- typed hierarchical `b8,k2,h1`, hidden utility

2. Better stale-state detectors
- residual-based seam disagreement
- carry-utility change rather than boundary-only priors
- document-boundary exact resets as a separate safe control

3. Slot diversity / anti-collapse ablation for typed compilers
- slot decorrelation penalty
- per-slot query specialization
- compare `16` diverse slots vs collapsed `16`

4. Distillation target sweep
- direct CE on next-window front tokens
- teacher target from `raw_prev_all`
- teacher target from oracle optimized prefix

5. Prefix compiler architecture sweep
- free-form compiler
- typed split/reduce compiler
- typed hierarchical compiler
- GRU or state-space summarizer

6. Focus-region sweep
- `0-31`
- `0-63`
- `0-127`
- weighted front-heavy objective

7. Eval semantics
- contiguous carry only
- reset on document boundary
- compare with existing sliding-eval settings

## What not to do

- Do not promote raw hidden-state prefix carry.
- Do not assume heuristic hidden compression is enough.
- Do not judge the lane by transcript-carry controls alone; the compiler is already materially better than raw transcript carry in the current smokes.
- Do not promote naive hard reset from boundary priors.
