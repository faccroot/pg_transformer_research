# Research Workspace

This directory keeps the code and the research record in the same git history.

- `project_wide/` holds stable notes, architecture reviews, cluster workflow notes, and broader research framing.
- `iterations/` holds the scalable experiment registry. JSONL is the primary index so the system still works when the run count reaches the hundreds or thousands.
- `../theory/` holds higher-level research theses and speculative design notes, including the new representation-learning lane.

The repository is currently being used from the local branch `research/bootstrap`, which keeps local research setup work separate from upstream `main`.
