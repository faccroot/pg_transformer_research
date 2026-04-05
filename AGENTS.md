# AGENTS.md

## Branch Memory / MCTS Coordination

Use the branch-memory control plane for research-tree context before grepping raw notes or iteration dirs by hand.

Preferred interface:

```bash
python3 tools/branch_memoryctl.py list-frontier --lane hardmax --status active,proposed,observed --limit 10
python3 tools/branch_memoryctl.py search "freeze300 hardmax" --lane hardmax
python3 tools/branch_memoryctl.py search "state collapse commitment" --lane hardmax --mode vector
python3 tools/branch_memoryctl.py show sweep_20260405-011602-mlx-hardmax-statebook-anticollapse-freeze300-smoke
```

Important behavior:

- `tools/branch_memoryctl.py` auto-starts the local daemon if the socket is missing or stale.
- `search` defaults to `--mode hybrid`, which combines SQLite FTS/BM25 with the derived TF-IDF vector index.
- Canonical store is filesystem-native under:
  - `research/branch_memory/nodes/`
- Derived indices live under:
  - `research/branch_memory/derived/`

Use direct mode only when debugging the control plane itself:

```bash
python3 tools/branch_memoryctl.py --direct ingest
python3 tools/branch_memoryctl.py --direct list-frontier --lane hardmax
```

Manual branch/task creation:

```bash
python3 tools/branch_memoryctl.py create-node \
  --title "New branch idea" \
  --type task \
  --lane hardmax \
  --status proposed \
  --priority 60

python3 tools/branch_memoryctl.py attach-path \
  --node-id manual_new_branch_idea \
  --kind code \
  --path train_gpt_mlx.py

python3 tools/branch_memoryctl.py link-node \
  --node-id manual_new_branch_idea \
  --parent-id note_project-wide-hardmax-structural-controller-lane-20260403

python3 tools/branch_memoryctl.py create-merge-node \
  --title "Cross-lane bridge" \
  --lane general \
  --from-node-id note_project-wide-hardmax-structural-controller-lane-20260403 \
  --from-node-id note_project-wide-fineweb-prosody-feature-state-lane-20260401
```

If you change node bundles or source notes/manifests and search looks stale, rebuild the derived indices:

```bash
python3 tools/branch_memoryctl.py ingest
```

Relevant implementation/docs:

- `tools/branch_memory.py`
- `tools/branch_memoryd.py`
- `tools/branch_memoryctl.py`
- `research/branch_memory/README.md`
- `research/project_wide/20260404_branch_memory_mcts_infra.md`

## Surprise Alerts

- `train_gpt.py` writes the full source code into `logs/<RUN_ID>.txt` before the runtime metrics. Any parser that scans those logs must anchor on concrete metric lines and ignore template strings in the source dump, or it will mis-parse placeholders like `{q_val_loss:.8f}` as real values.
