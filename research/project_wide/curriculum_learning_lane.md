# Curriculum Learning Lane

This note turns the current curriculum-learning thesis into a testable research lane for the Parameter Golf stack.

## Core thesis

Curriculum likely matters more here than in frontier pretraining because the training budget is short, the model is small, and the run covers only about one pass over the cached corpus rather than washing out ordering effects over many epochs.

The useful unit is not "easy to hard" in the abstract. The useful unit is "which data best trains which component at which phase."

## Main claims to test

### H1. Purposeful ordering should beat plain sequential or reshuffled order in the 10-minute regime.

- Rationale: near-single-pass training makes early batches disproportionately important.
- Stronger form: order should be built around chunk utility, not just randomization.
- Comparison target: current contiguous shard streaming in [train_gpt_mlx.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx.py#L357).

### H2. Different training components want different data.

- `CE`: highest signal from hard or high-entropy token positions.
- `SIGReg`: highest signal from structurally diverse chunks that spread hidden states out.
- `Logic sidecar`: highest signal from operator-dense chunks.
- `JEPA sidecar`: highest signal from coherent, structurally rich sequences.
- `TurboQuant QAT`: easiest data is best during the ramp because geometry adaptation should not compete with new content learning.
- `EMA`: best used late on familiar or validation-like data, not during sharp representation changes.

### H3. The strongest curriculum should be phased, not monolithic.

Default phase plan:

1. `0.00-0.20`: structural foundation
   Focus on structurally diverse chunks. `CE + SIGReg`. No QAT. No EMA.
2. `0.20-0.40`: logic and polarity
   Bias toward operator-dense chunks. Activate logic sidecar and JEPA auxiliary path.
3. `0.40-0.60`: hard content
   Bias toward hard chunks and high-loss tokens. Add mild focal or entropy weighting.
4. `0.60-0.80`: QAT geometry adaptation
   Bias toward easy chunks. Full QAT. Keep gradients calm.
5. `0.80-1.00`: consolidation
   Bias toward moderate-difficulty validation-like chunks. Full QAT plus EMA.

### H4. Chunk-level structural features are the right offline proxy.

First-pass features worth building:

- hashed token histogram per chunk
- structural cluster id from histogram k-means
- operator density
- duplicate or near-duplicate proxy
- optional model-based difficulty or confidence score

The practical point is not that token histograms are perfect. The point is that they are cheap enough to compute offline and good enough to drive an initial curriculum.

### H5. One-pass classification can reclaim budget.

Chunks should be split into:

- `never`: duplicates, boilerplate, or already-mastered trivial chunks
- `once`: ordinary useful chunks
- `repeat`: hard structural or operator-dense chunks that deserve a second pass

This is a more defensible use of repetition than simply oversampling the whole "best" subset from the start.

## Cautions

Several parts of the broad thesis are still speculative and should be treated as benchmarks, not assumptions:

- The "full dataset scan in under 10 seconds on 8xH100" claim is plausible in principle but is not yet measured in this repo.
- A strict full-run easy to hard curriculum has weaker prior support than a curriculum warmup followed by mixed sampling.
- An extra no-grad or extra forward pass for focal loss may not pay for itself under a hard 10-minute cap unless the weighting is very light.
- Operator-density proxies are tokenizer-dependent and may need direct inspection on `sp1024`.
- Advanced adaptive rescoring should come after the static offline path is stable.

## Implementation ladder

### Slice 1: pure-Python curriculum scaffolding

- phase scheduler
- chunk feature structures
- hashed histogram extraction
- operator-density scoring
- structural clustering
- bucket classifier for `never` / `once` / `repeat`
- unit tests for these pieces

This slice should not depend on MLX and should run on the Linux box.

### Slice 2: offline feature build for shard subsets

- CLI that scans FineWeb token shards
- outputs `npz` metadata for chunk-level features
- enough for one-shard and smoke-tier Mini experiments first
- dedicated launch surface should use [train_gpt_mlx_curriculum.py](/home/zaytor/transformer_research/parameter-golf/train_gpt_mlx_curriculum.py), not the default trainer path
- queue-style cluster launches should use `dispatch.sh` via a generated `launch.sh` when support files are needed

### Slice 3: trainer integration

- curriculum-aware loader or ordered chunk sampler
- phase-driven component toggles
- optional per-token loss weighting
- logging of phase transitions and chunk-source mix

### Slice 4: adaptive curriculum

- periodic rescoring of held-out candidate chunks
- mastered-chunk skipping
- dynamic priority refresh every `N` steps

This is intentionally deferred until the static lane is clean.

## First experiment order

1. Static five-phase scheduler without data reordering.
   Goal: test whether turning components on and off by phase already helps.
2. Offline chunk ordering with structural diversity and operator density.
   Goal: test whether the data path adds signal beyond phase toggles alone.
3. Mild focal weighting in the hard-content phase only.
   Goal: test whether extra concentration on hard tokens helps enough to justify the compute.
4. Easy-biased QAT phase with late EMA.
   Goal: test the specific "geometry adaptation wants calm data" claim.

## Minimal success criteria

- Curriculum code remains independent from MLX for the feature-building and scheduling layer.
- We can generate chunk metadata deterministically for a shard subset.
- We can write tests that prove the phase schedule, scoring logic, and replay-bucket rules behave as intended.
- The first Mini sweep can compare at least:
  - baseline order
  - phase-only curriculum
  - phase + offline ordering

## Current position

This lane is upstream of the actual trainer changes. The immediate goal is not to prove the whole thesis at once. The immediate goal is to turn the vague idea of "curriculum" into a small number of deterministic, inspectable objects that the training loop can consume.
