# Branch Memory for MCTS-Style Research Planning (2026-04-04)

This note turns the current "persisted MCTS discipline" into a reusable infrastructure shape.

The key correction is:

- do not start with a heavyweight graph database
- start with a filesystem-native branch memory that agents can write directly
- derive search and graph semantics from that memory

## Why this object

The current research tree already exists, but it is fragmented across:

- `research/project_wide/*.md`
- generated sweep dirs with `manifest.json`
- `observed_results.json`
- recovered artifacts
- diagnostics JSONs

That is already close to a branch-native system.

The missing piece is a canonical place where a branch node can attach:

- results
- code refs
- artifacts
- follow-up tasks
- dependencies
- local notes

without forcing everything into a graph database row model first.

## Canonical store

The first scaffold lives at:

- `research/branch_memory/`

Each node is a directory bundle:

- `node.json`
- `notes.md`
- `source_snapshot.md`
- `attachments/`

The canonical object is the node bundle, not the database.

The database is derived.

## Derived indices

The first scaffold also builds a small SQLite index:

- `research/branch_memory/derived/index.sqlite`

That index stores:

- nodes
- attachment refs
- lightweight cross-node edges
- optional FTS search over titles, notes, snapshots, and attachment refs

This gives three retrieval modes:

1. exact path / attachment lookup
2. text search over node memory
3. structural traversal through derived edges

## Initial ingest sources

The first pass ingests:

1. `research/project_wide/*.md`
   - branch notes
   - hypothesis queues
   - lane notes

2. `research/iterations/generated/*/manifest.json`
   - sweep nodes
   - run metadata

3. `observed_results.json`
   - active / complete status
   - best observed metrics when present

4. linked markdown refs and nearby diagnostics JSONs
   - code attachments
   - result attachments
   - note-to-sweep edges

## CLI

The first tool is:

- [branch_memory.py](/home/zaytor/transformer_research/parameter-golf/tools/branch_memory.py)
- [branch_memoryd.py](/home/zaytor/transformer_research/parameter-golf/tools/branch_memoryd.py)
- [branch_memoryctl.py](/home/zaytor/transformer_research/parameter-golf/tools/branch_memoryctl.py)

Supported commands:

1. ingest current branch memory

```bash
python3 tools/branch_memory.py ingest
```

2. list frontier nodes

```bash
python3 tools/branch_memory.py list-frontier --limit 20
```

Lane-scoped frontier:

```bash
python3 tools/branch_memory.py list-frontier \
  --lane hardmax \
  --status active,proposed,observed \
  --limit 12
```

3. search notes, snapshots, and attachments

```bash
python3 tools/branch_memory.py search "freeze300 attention shaping"
```

Lane-scoped search:

```bash
python3 tools/branch_memory.py search "freeze300 hardmax" --lane hardmax
```

Vector-only search:

```bash
python3 tools/branch_memory.py search "freeze300 hardmax" --lane hardmax --mode vector
```

4. inspect one node bundle

```bash
python3 tools/branch_memory.py show sweep_20260405_011602_mlx-hardmax-statebook-anticollapse-freeze300-smoke
```

5. create a manual node

```bash
python3 tools/branch_memory.py create-node \
  --title "H2 expansion after q_bias win" \
  --type task \
  --lane hardmax \
  --status proposed \
  --priority 80
```

6. attach code or result paths

```bash
python3 tools/branch_memory.py attach-path \
  --node-id note_project-wide_20260404_hardmax_hypothesis_test_queue \
  --kind code \
  --path train_gpt_mlx.py
```

## Daemon / control-plane mode

The direct CLI is fine for humans and one-off agent calls.

For agent orchestration, a better shape is:

- one resident daemon holds the control plane
- agents send short socket requests through a thin client wrapper
- the daemon serializes writes and runs the existing branch-memory operations

Local start:

```bash
python3 tools/branch_memoryd.py
```

Agent-facing client:

```bash
python3 tools/branch_memoryctl.py --ping
python3 tools/branch_memoryctl.py list-frontier --lane hardmax --status active,proposed,observed --limit 10
python3 tools/branch_memoryctl.py search "freeze300 hardmax" --lane hardmax
python3 tools/branch_memoryctl.py --shutdown
```

Practical note:

- `branch_memoryctl.py` will now autostart the daemon if the socket is missing or stale, so agents can usually just call the client directly.

Systemd template:

- [branch-memory.service](/home/zaytor/transformer_research/parameter-golf/research/branch_memory/systemd/branch-memory.service)

This keeps the canonical object filesystem-native while making the agent interaction model feel more like a local control plane than a file-editing ritual.

## What this is for

This should become the coordination substrate for:

- branch-local agent context
- parent/child branch expansion
- result attachment
- experiment backprop into planning
- compute / task allocation later

The design target is:

- canonical object = node bundle
- graph semantics = derived
- search = first-class
- agents append files, not rows

## What is still missing

This first pass is intentionally narrow.

It does not yet include:

- learned/vector retrieval
- automatic budget / expected-value scoring
- compute scheduler integration
- node-level patch storage
- test obligations
- agent assignment workflow

The daemon layer does not replace those. It only makes the current store easier for many agents to query and mutate through one local endpoint.

The current retrieval stack is now:

- canonical node bundle store
- SQLite FTS/BM25 index
- derived TF-IDF vector index
- hybrid search via reciprocal-rank fusion

That is still a lightweight lexical vector layer rather than a learned embedding model, but it already gives agents a better "search over branches, notes, and artifacts" surface than raw grep or graph traversal alone.

Those should come after the node bundle shape proves useful in the repo.
