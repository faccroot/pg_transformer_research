# Branch Memory

This directory is the canonical store for branch-native MCTS-style research memory.

The primary object is a node bundle under `nodes/<node_id>/`.

Each bundle contains:

- `node.json`: machine-readable metadata
- `notes.md`: local branch notes that agents or humans can update
- `source_snapshot.md`: ingest-time snapshot from source notes / sweeps
- `attachments/`: reserved for future materialized attachments

The derived search / graph index lives under:

- `derived/index.sqlite`

Use:

```bash
python3 tools/branch_memory.py ingest
python3 tools/branch_memory.py list-frontier
python3 tools/branch_memory.py search "hardmax freeze300"
python3 tools/branch_memory.py search "hardmax freeze300" --mode vector
python3 tools/branch_memory.py link-node --node-id some_child --parent-id some_parent
python3 tools/branch_memory.py create-merge-node \
  --title "Cross-lane bridge" \
  --from-node-id lane_a \
  --from-node-id lane_b
```

Daemon/client mode:

```bash
python3 tools/branch_memoryctl.py --ping
python3 tools/branch_memoryctl.py list-frontier --lane hardmax --status active,proposed,observed --limit 10
python3 tools/branch_memoryctl.py search "freeze300 hardmax" --lane hardmax
python3 tools/branch_memoryctl.py --shutdown
```

Notes:

- `branch_memoryctl.py` now autostarts the daemon by default if the socket is missing or stale.
- `search` defaults to `--mode hybrid`, which combines SQLite FTS/BM25 with a derived TF-IDF vector index at `research/branch_memory/derived/vector_index.json`.
- `link-node` is the non-destructive way to add cross-lane parentage to an existing node.
- `create-merge-node` creates a manual bridge/compaction node with multiple parents while preserving the original nodes.
- Ingested sweep nodes now auto-attach runtime and queue artifacts such as `runtime_summary.md`, `runtime_state.json`, `cluster_queue_snapshot.json`, and `cluster_queue_summary.md` when those files exist under a generated iteration.

User systemd template:

- [branch-memory.service](/home/zaytor/transformer_research/parameter-golf/research/branch_memory/systemd/branch-memory.service)

Example install:

```bash
mkdir -p ~/.config/systemd/user
cp research/branch_memory/systemd/branch-memory.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now branch-memory.service
```
