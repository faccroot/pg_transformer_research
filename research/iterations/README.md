# Iteration Tracking

This folder is set up for scale.

- `iteration_index.jsonl` is the primary index for architectural changes and research branches.
- `run_index.jsonl` is the primary index for actual executions.
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

