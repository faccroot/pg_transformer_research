# Cluster Queue / Runtime Metrics Bridge

Date: 2026-04-05

Purpose: make the Mac-mini queue, per-step run outputs, and branch-memory tree
work as one system without collapsing them into one store.

## The split

The clean architecture is:

- `~/cluster`
  - operational truth
  - Redis locks
  - host claims
  - `job_id`, `host`, `remote_dir`, and live execution state

- `research/iterations/generated/<iteration>/`
  - experiment-local truth
  - `manifest.json`
  - `observed_results.json`
  - recovered logs/artifacts
  - now also:
    - `cluster_queue_snapshot.json`
    - `cluster_queue_summary.md`
    - `runtime_state.json`
    - `runtime_metrics_manifest.json`
    - `runtime_summary.md`

- `research/iterations/derived/runtime_metrics.sqlite`
  - searchable numerical truth
  - per-run queue/runtime state
  - per-step scalar metrics extracted from logs

- `research/branch_memory/`
  - research-tree truth
  - hypotheses
  - cross-lane bridge nodes
  - forward-path tasks
  - links to sweeps and structured outputs

That means:

- queue is not the research graph
- branch-memory is not the scheduler
- the runtime bridge connects them

## What is implemented

Queue/runtime updater:

- [update_iteration_runtime_index.py](/home/zaytor/transformer_research/parameter-golf/tools/update_iteration_runtime_index.py)

It currently does three things:

1. `update`
   - snapshots queue/run state for one generated iteration
   - snapshots direct cluster host/lock/job state from `~/cluster`
   - parses available trainer logs
   - writes:
     - `cluster_queue_snapshot.json`
     - `cluster_queue_summary.md`
     - `runtime_state.json`
     - `runtime_metrics_manifest.json`
     - `runtime_summary.md`
   - updates the global SQLite metric index

2. `search-metric` / `list-runs`
   - makes indexed outputs searchable from the shell

3. `list-cluster-hosts` / `list-cluster-jobs`
   - exposes the latest indexed queue host/job state directly from SQLite

4. `search-analysis-metric`
   - exposes indexed residual/prosody/comparison summaries from saved analysis
     artifacts
   - works for both:
     - generated queue sweeps with `manifest.json`
     - analysis-only result bundles without a manifest

Generated queue sweeps now call the updater automatically in `post_run` via:

- [prepare_mlx_sweep.py](/home/zaytor/transformer_research/parameter-golf/tools/prepare_mlx_sweep.py)

That same `post_run` hook now also runs a direct branch-memory ingest, so
completed queue sweeps refresh:

- recovered artifacts
- observed results
- runtime/cluster snapshot artifacts
- branch-memory attachments/search

## Why this is useful

This fixes three gaps:

1. cluster state no longer lives only in `dispatch.out`
2. queue truth no longer lives only in human-readable `~/cluster/status.sh`
3. per-step outputs no longer live only in raw logs

Now we can ask things like:

- which host ran this arm?
- which `job_id` owns the remote directory?
- what was the latest `train_loss` or `val_bpb` before export?
- which runs showed `residnov_mean`, `prosody_reset`, or other lane-specific
  metrics moving in a useful way?
- which comparison bundle lowered `residual_modes.argmax_embedding.acf_summary.all.mean`?
- which hardmax variant reduced `factor_0` persistence?
- what was the exact `mean_nll` delta in the sidecar residual bundle?

without re-parsing raw text every time.

## Current limitations

- queue state is now snapshotted directly from `~/cluster`, but still as a
  poll/snapshot model rather than a persistent event stream
- only scalar metrics from existing trainer logs are indexed
- saved residual/prosody/comparison JSONs are indexed selectively by summary
  metrics, not exhaustively by every leaf
- token-level residual dumps are intentionally not stored by default
- branch-memory auto-attaches queue/runtime artifacts after ingest, but not the
  SQLite file itself

## Recommended next extensions

1. add a persistent cluster event stream or append-only snapshot history
2. add optional periodic residual-factor snapshots into the runtime index
3. add a small compare/query helper for:
   - same-host runs
   - latest metric deltas by run family
   - runs with missing exports but healthy step traces
4. optionally expose runtime-index summaries inside branch-memory frontier views

## Decision

Keep the three-layer split:

- scheduler in `~/cluster`
- metrics in `research/iterations/derived`
- research graph in `research/branch_memory`

Do not merge them into one database or one daemon.
