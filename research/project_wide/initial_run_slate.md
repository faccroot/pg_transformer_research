# Revised Run Slate

This note supersedes the earlier first-wave ordering. It synthesizes the local baseline review, the public PR archive, and the multi-agent memo in [research_agent_feedback.md](/home/zaytor/transformer_research/parameter-golf/theory/research_agent_feedback.md).

## What changed after synthesis

Seven updates materially changed the execution order:

1. The mainline record-track stack is now clearly non-recurrent.
   Public PRs have converged on a family built around schedule tuning, a wider MLP, low-bit export, fp16 embedding handling, and longer-context sliding evaluation. That stack is now the baseline we need to beat before a more novel architecture matters.

2. Longer-context evaluation is already part of the real competition frontier.
   Agent-collected reports point to leaderboard-level gains from longer-context plus sliding-window evaluation. That lowers the bar for any architecture change: it now has to compete with a baseline family that may already live around the low `1.18x` range after eval improvements.

3. Shared recurrence is still interesting, but the delivered-quant path is now the blocker.
   Competitor reports suggest recurrence can look good before export and then collapse after low-bit roundtrip. That demotes recurrence from the main record lane to a bounded risk lane until we have a safe delivery recipe.

4. QAT and low-bit export remain first-class lanes because the scored metric is delivered BPB.
   Compression work is not cosmetic here; it directly attacks the submission artifact that is actually evaluated.

5. Multi-token prediction moved up; label smoothing moved down.
   The strongest training-objective recommendation is `K=2` multi-token prediction with a modest auxiliary weight. Label smoothing is more likely to hurt BPB than help it.

6. The tokenizer lane now splits into a main competitive path and a novelty path.
   Mainline: standard BPE around `2048-4096` on top of the strongest non-recurrent stack.
   Novelty lane: morphology-aware or meaning-aware tokenization after the main BPE path is benchmarked.

7. Document-aware packing still looks under-explored relative to its expected return.
   It costs almost no artifact bytes and is still not obviously saturated by the public PR set.

## Current execution order

### Stage 0: Already in place

- `D0` offline quantizer sweep tooling exists.
- `D1` periodic post-quant diagnostics exist in both trainers.
- Shared-template recurrence support exists as scaffolding, but not yet as the final recurrent design we want to test.
- The Mini fleet is bootstrapped and ready for single-node MLX ranking runs.

### Stage 1: Mainline record-track stack

These are the first things to run before deeper architecture changes.

#### S1. Converged optimizer and warmdown bundle

- Goal: match the now-standard leaderboard schedule stack before branching out.
- First target bundle:
  - `MATRIX_LR=0.02`
  - `SCALAR_LR=0.02`
  - `TIED_EMBED_LR=0.03`
  - `MUON_MOMENTUM=0.99`
  - `MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `WARMDOWN_ITERS` sweep centered around `2200-3000`
- Why first:
  - this is now table stakes for the record lane
  - it raises the bar for every tokenizer and data-path test that follows

#### S2. Wider MLP plus delivery-path stack

- Goal: land the strongest known non-recurrent body before tokenizer work.
- First target:
  - `MLP_MULT=3`
  - fp16 tied-embedding passthrough
  - low-bit export path, with `int6` as the first target and `zstd` as the secondary compression path to test
- Note:
  - some of this is still an implementation item in our branch; this stage includes the code work needed to expose it cleanly

#### S3. Longer-context and sliding-window evaluation sweep

- Goal: capture the best low-byte eval improvement on the current family.
- First sweep:
  - eval context lengths `{1024, 1536, 2048, 3072, 4096}`
  - stride sweep once the best context length is clear, starting at `64`
- Why now:
  - it costs almost no artifact bytes
  - it is already part of the public frontier

#### S4. Tokenizer co-design on the strong non-recurrent stack

- Goal: test whether `SP-2048` or `SP-4096` beats the field-standard `SP-1024` once the model stack is competitive.
- First path:
  - train `SP-2048` and `SP-4096`
  - measure bytes-per-token on validation
  - benchmark on top of the strong non-recurrent stack, not the baseline control
- Fallback if budget gets tight:
  - factor the embedding or add a low-rank output residual only after the plain larger-vocab path is measured

#### S5. Document-aware shard packing and sequence schedule

- Goal: stop conditioning on irrelevant previous-document tokens and reduce packing waste.
- First implementation:
  - sentence-aware splits for long documents when possible
  - better best-fit packing for short documents
  - short-to-long sequence schedule once packing is stable
  - optional offline teacher-scored curriculum ordering after the packing control exists
- Why now:
  - expected gain is large relative to the engineering cost
  - it is a data-path improvement, not a speculative model family rewrite

### Stage 2: Training objective and init layer

#### T1. Multi-token prediction

- Implementation:
  - start with `K=2`
  - first weight target: `lambda_2 = 0.25`
  - train-time-only auxiliary path
- Follow-up:
  - one mid-layer next-token head around layer `5`, `mu=0.1`, decayed off late
- Why:
  - best-ranked loss modification
  - improves sample efficiency without adding shipped artifact cost

#### T2. Cheap initialization wins

- First targets:
  - overtone / spectrum-shaped tied embedding init
  - residual-mix init schedules that reflect the stronger current priors from public PRs
- Why:
  - zero or near-zero artifact cost
  - easy to stack onto the mainline

### Stage 3: Bounded recurrence lane

This remains a parallel risk lane, not the default record path.

#### R1. One-block shared recurrence

- Implementation:
  - one shared block
  - `d=512`, `heads=8`, `kv=4`, `VOCAB_SIZE=1024`
  - `R_train=12`
  - `R_eval=16`, with optional eval sweep to `20`
- Keep initially:
  - `x0` anchor path
  - residual controls
  - quant diagnostics
- Add initially:
  - deterministic depth signal
  - tiny recurrent skip bank with fixed lags
- Why:
  - still the cleanest architecture-level byte reallocation mechanism
  - only worth promoting if the low-bit roundtrip is safe

#### R2. Safer sharing variants

- Implementation:
  - fp16-stored shared block
  - sandwich sharing
  - relaxed sharing with tiny per-depth deltas
- Why:
  - these variants directly target the export failure mode reported by competitors

### Stage 4: Delivery and compression lane

This lane remains critical because delivered BPB is the metric.

#### C1. Exact-export QAT

- Implementation:
  - fake-quantize only the large matrix weights that will actually ship quantized
  - keep control tensors in float as today
  - match export scaling rules
- Why:
  - directly attacks the delivered score
  - expected to recover part of the current low-bit loss

#### C2. Sensitivity-guided mixed precision

- Implementation:
  - rank tensors by post-quant harm per saved byte
  - first push middle-layer MLPs toward `int6`
  - only test `int4` after sensitivity ranking and QAT support exist
- Why:
  - safest path to real byte savings without blindly crushing sensitive tensors

#### C3. Compression-aware serialization

- Implementation:
  - bit-pack low-bit tensors densely when possible
  - separate streams by tensor type/bitwidth before `zlib` or `zstd`
- Why:
  - byte layout is likely leaving easy compression on the table

### Stage 5: Exploratory second-wave lanes

These are real ideas, but not in the first execution wave.

#### E1. Prefix-conditioned diagonal gate

- First form:
  - MLP-only
  - decoder-half only
  - tiny prefix summary
- Why:
  - best low-byte version of the operator-memory idea

#### E2. Attention ablations

- Order:
  - `8Q:2KV` first
  - then shared-basis or queryless-Q variants
- Why:
  - useful compression-aware attention lane
  - lower ceiling than tokenizer plus data-path plus delivery improvements

#### E3. Evolving `x0`

- Scope:
  - reparameterize the current `resid_mix` logic so `x0` becomes a slow-moving global state rather than a frozen anchor
- Why:
  - theory-aligned and relatively cheap

#### E4. Late-layer low-rank `4x` FFN

- Scope:
  - upper layers only
  - two-sided factorization
- Why:
  - strongest FFN redesign from the agent synthesis
  - still a second-wave structural bet

#### E5. Tokenizer novelty lane

- First novelty test:
  - morphology-aware comparison against the best standard BPE baseline
- Why:
  - potentially novel relative to the field
  - should not displace the competitive BPE path until the latter is benchmarked

#### E6. Binary embedding dims

- Constraint:
  - only test seriously once `VOCAB_SIZE=4096` or a similarly larger vocab path exists
- Why:
  - much more plausible there than at `1024`

## Run protocol

Every substantive implementation should still go through three stages:

1. `Smoke`
   - MLX path
   - `train_shards=1`
   - correctness, throughput, export, and logging validation

2. `Rank`
   - MLX path
   - `train_shards=10` when available on-node
   - enough signal to compare delivered curves and discard weak ideas

3. `Reference`
   - CUDA/H100 path
   - full challenge-style timed run or real eval sweep
   - only for candidates that clear the local ranking bar

The main selection metric remains post-export `val_bpb`, not pre-export `val_bpb`.

## Explicit de-prioritization

- Full ACT-style train-time halting
- MC-dropout / temperature-mixture eval tricks as a main strategy
- Label smoothing as a primary BPB improvement path
- Full SSM replacement, full MoE, or reversible-stack rewrites
- Dense hypernetworks or full generated operators
- Plain reshuffling without better packing or ordering
- Untied embeddings before recurrence or compression frees real budget
