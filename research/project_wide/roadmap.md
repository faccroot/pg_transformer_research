# Parameter Golf Research Roadmap

## Operating assumptions

- Primary edit loop: develop in this repo, validate quickly with `train_gpt_mlx.py` on the Mac mini cluster, and keep `train_gpt.py` as the CUDA reference path for later leaderboard-quality runs.
- Long-run research priority: core architecture and representation changes, not raw-compute brute force.
- Near-run execution priority: take the obvious low-byte wins first when they materially change the ranking bar for later architecture work.
- Every non-trivial change should be attached to an iteration ID before or at run launch.

## Working rules

- One main architectural hypothesis per iteration.
- Log the exact script, commit, command, dataset variant, train shard count, host, and resulting metrics for every run worth keeping.
- Keep JSONL as the source of truth for scale. Only create a per-iteration folder when the iteration needs extra notes, plots, patches, or artifacts.
- Treat local MLX tests as a filter for ideas, not as a substitute for the CUDA path when the system-level behavior matters.

## Current priority stack

1. `Primary record-track lane`
   - converge on the strongest non-recurrent stack first:
     - optimizer and warmdown bundle
     - `MLP_MULT=3`
     - fp16 tied-embedding passthrough plus low-bit export/compression
     - longer-context and sliding-window evaluation
   - benchmark standard BPE `2048-4096` on top of that stack, not in isolation
   - add document-aware packing, sequence scheduling, and safe offline curriculum ordering

2. `Bounded recurrence lane`
   - keep recurrence as a parallel, high-risk lane
   - only pursue fp16-stored sharing, sandwich sharing, or relaxed sharing with tiny per-depth deltas
   - do not make recurrence the record-track default until the quantized roundtrip is competitive

3. `Primary delivery lane`
   - export-matched QAT
   - per-tensor sensitivity ranking
   - mixed-precision packing and better byte layout before `zlib`/`zstd`

4. `Training-objective lane`
   - multi-token prediction first
   - one mid-layer auxiliary head second
   - only very mild BPB-aware difficulty weighting after those

5. `Second-wave representation lane`
   - evolving `x0`
   - prefix-conditioned diagonal gates or tiny low-rank operators
   - later MLP redesigns such as low-rank `4x` FFNs in upper layers

6. `Tokenizer lane`
   - competitive path: standard BPE around `2048-4096`, coupled to the strong non-recurrent stack
   - novelty path: morphology-aware or meaning-aware tokenization only after the main BPE path is benchmarked

## Near-term research buckets

- Non-recurrent delivery-stack improvements that now define the real record-track bar: schedule, wider MLP, low-bit export, fp16 embedding passthrough, and sliding eval.
- Quantization-aware architecture and export choices that improve delivered low-bit artifacts, not only raw bf16 checkpoints.
- Data-path improvements that cost almost no artifact bytes: doc-aware packing, sentence-aware splits, short-to-long sequence schedules, and safe offline curriculum ordering.
- Shared recurrence and test-time extra depth as a bounded architecture thesis, not the default first-wave record path.
- Residual-stream and control-path variants, since the baseline already exposes learnable residual controls.
- Attention and FFN compression ideas that preserve the tiny artifact budget: more aggressive GQA, query sharing, and late-layer low-rank FFNs.

## Tokenizer lane

- Keep tokenizer experiments in a separate lane from model architecture work so they can run in parallel.
- Treat `2048-4096` standard BPE as the main competitive tokenizer range on top of the strongest non-recurrent stack.
- Treat morphology-aware tokenization as a novelty lane, not the default first tokenizer bet.
- Start with one run per mini per tokenizer modification while the minis are uncoupled.
- When the 12-mini, 3-cluster setup arrives, keep per-host logging but group coordination by cluster.
- Use the dedicated note at [tokenizer_lane.md](tokenizer_lane.md) for experiment ordering and run logging.
