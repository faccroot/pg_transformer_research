# Iteration Tracking

This folder is set up for scale.

- `iteration_index.jsonl` is the primary index for architectural changes and research branches.
- `run_index.jsonl` is the primary index for actual executions.
- `derived/runtime_metrics.sqlite` is the searchable per-step metric index built from generated sweep logs.
- `archive/` is optional. Use it only when an iteration needs longer notes, plots, patches, or other supporting files.
- `templates/` contains lightweight note skeletons for manual write-ups.

Why JSONL first:

- Hundreds or thousands of iterations are easier to grep, diff, and post-process as append-only JSONL than as one directory per experiment.
- Per-iteration directories are still available when a particular branch needs richer artifacts.

Recommended workflow:

```bash
python tools/research_tracker.py new-iteration \
  --slug skip-gate-variant \
  --component residual-routing \
  --script train_gpt_mlx.py \
  --hypothesis "Per-feature residual routing will help small-model depth efficiency."
```

```bash
python tools/research_tracker.py log-run \
  --iteration-id iter_YYYYMMDD_HHMMSS_skip-gate-variant \
  --run-id mlx_skip_gate_001 \
  --host mini-03 \
  --script train_gpt_mlx.py \
  --dataset fineweb10B_sp1024 \
  --command 'RUN_ID=mlx_skip_gate_001 python3 train_gpt_mlx.py' \
  --metric val_bpb=1.2345 \
  --status completed
```

Suggested field discipline:

- Iteration entries should capture the hypothesis, owning component, base commit, and the note path.
- Run entries should capture the exact command, host, dataset/shard choice, metrics, log path, and artifact path when relevant.
- Keep cluster machine logs in `logs/` or your remote filesystem; keep only metadata links in the JSONL tracker.

Generated queue sweeps also support a structured runtime bridge:

```bash
python3 tools/update_iteration_runtime_index.py update research/iterations/generated/<iteration_dir>
python3 tools/update_iteration_runtime_index.py list-runs --run-like prosody
python3 tools/update_iteration_runtime_index.py search-metric val_bpb --phase val --run-like hardmax
python3 tools/update_iteration_runtime_index.py list-cluster-hosts --status BUSY
python3 tools/update_iteration_runtime_index.py list-cluster-jobs --status running
python3 tools/update_iteration_runtime_index.py search-analysis-metric mean_nll --iteration-like residual_autocorr_sidecar
```

That updater writes per-iteration artifacts:

- `cluster_queue_snapshot.json`
- `cluster_queue_summary.md`
- `runtime_state.json`
- `runtime_metrics_manifest.json`
- `runtime_summary.md`

and updates:

- `research/iterations/derived/runtime_metrics.sqlite`

Queue state comes from the external Mac-mini control plane under `~/cluster`,
but it is snapshotted into each generated iteration so queue truth, experiment
truth, and research-tree truth stay linked without sharing a single database.

Generated queue sweeps prepared by [prepare_mlx_sweep.py](/home/zaytor/transformer_research/parameter-golf/tools/prepare_mlx_sweep.py)
now run this bridge automatically in `post_run`, followed by a direct
branch-memory ingest.

The updater also supports analysis-only iteration directories without a
`manifest.json`, which is useful for standalone residual/prosody/hardmax result
bundles under `research/iterations/generated/`.
