#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"
PROJECT_WIDE_DIR = RESEARCH_DIR / "project_wide"
ITERATIONS_GENERATED_DIR = RESEARCH_DIR / "iterations" / "generated"
STORE_DIR = RESEARCH_DIR / "branch_memory"
NODES_DIR = STORE_DIR / "nodes"
DERIVED_DIR = STORE_DIR / "derived"
DB_PATH = DERIVED_DIR / "index.sqlite"
VECTOR_INDEX_PATH = DERIVED_DIR / "vector_index.json"
SCHEMA_VERSION = 1

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]+")
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class Attachment:
    kind: str
    path: str
    label: str
    exists: bool


@dataclass
class NodeBundle:
    node_id: str
    title: str
    node_type: str
    lane: str
    status: str
    priority: float | None
    budget: float | None
    expected_value: float | None
    information_gain: float | None
    compute_cost: float | None
    fragility: float | None
    parent_ids: list[str]
    tags: list[str]
    summary: str
    metadata: dict[str, Any]
    attachments: list[Attachment]
    source_paths: list[str]
    notes_text: str
    source_snapshot_text: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_store_dirs() -> None:
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "node"


def node_dir(node_id: str) -> Path:
    return NODES_DIR / node_id


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_title(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback.replace("_", " ")


def extract_summary(text: str, max_chars: int = 420) -> str:
    paragraphs: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        stripped = "\n".join(line.rstrip() for line in chunk.splitlines()).strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        paragraphs.append(stripped)
    summary = " ".join(paragraphs[:2]).strip()
    if len(summary) > max_chars:
        return summary[: max_chars - 1].rstrip() + "…"
    return summary


def infer_lane(path: Path) -> str:
    text = path.as_posix().lower()
    if "hardmax" in text:
        return "hardmax"
    if "representation-learning" in text or "representation_learning" in text:
        return "representation_learning"
    if "prosody" in text:
        return "prosody"
    if "grouped-slim" in text or "slim" in text:
        return "slim"
    return "general"


def infer_note_type(path: Path, text: str) -> str:
    stem = path.stem.lower()
    if any(token in stem for token in ("mcts", "queue", "leg", "lane", "plan")):
        return "branch"
    if any(token in stem for token in ("review", "read", "results")):
        return "decision"
    if "architecture" in stem or "design" in stem:
        return "hypothesis"
    if "questions" in stem:
        return "task"
    return "note"


def infer_note_status(text: str) -> str:
    upper = text.upper()
    if "PRUNED" in upper:
        return "pruned"
    if "DEFERRED" in upper:
        return "deferred"
    if "BLOCKED" in upper:
        return "blocked"
    if "WINNER" in upper or "CURRENT BEST" in upper:
        return "won"
    return "active"


def infer_sweep_status(observed_payload: dict[str, Any] | None) -> str:
    if not observed_payload:
        return "proposed"
    statuses: list[str] = []
    for run in observed_payload.get("runs", {}).values():
        status = str(run.get("status", "")).strip().lower()
        if status:
            statuses.append(status)
    if not statuses:
        return "proposed"
    if any(any(token in status for token in ("progress", "running", "queued", "claimed", "launch")) for status in statuses):
        return "active"
    if any(any(token in status for token in ("exact_final", "observed_final", "complete", "completed")) for status in statuses):
        return "complete"
    if all(any(token in status for token in ("fail", "error", "missing")) for status in statuses):
        return "failed"
    return "observed"


def extract_local_links(text: str, source_path: Path) -> list[Attachment]:
    attachments: list[Attachment] = []
    seen: set[tuple[str, str]] = set()
    for label, target in LINK_RE.findall(text):
        target = target.strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = resolve_repo_path(target, source_path.parent)
        if resolved is None:
            continue
        key = (label, relpath(resolved))
        if key in seen:
            continue
        seen.add(key)
        attachments.append(
            Attachment(
                kind=infer_attachment_kind(resolved),
                path=relpath(resolved),
                label=label.strip() or resolved.name,
                exists=resolved.exists(),
            )
        )
    return attachments


def resolve_repo_path(target: str, relative_to: Path) -> Path | None:
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = (relative_to / target_path).resolve()
    else:
        target_path = target_path.resolve()
    try:
        target_path.relative_to(ROOT)
    except ValueError:
        return None
    return target_path


def infer_attachment_kind(path: Path) -> str:
    lower = path.name.lower()
    if lower == "manifest.json":
        return "manifest"
    if lower == "observed_results.json":
        return "observed_results"
    if lower == "runtime_state.json":
        return "runtime_state"
    if lower == "runtime_metrics_manifest.json":
        return "runtime_metrics"
    if lower == "cluster_queue_snapshot.json":
        return "cluster_queue_state"
    if lower.endswith(".md"):
        if lower == "runtime_summary.md":
            return "runtime_summary"
        if lower == "cluster_queue_summary.md":
            return "cluster_queue_summary"
        return "note"
    if lower.endswith(".json"):
        return "result_json"
    if lower.endswith((".py", ".sh")):
        return "code"
    if lower.endswith((".txt", ".log")):
        return "log"
    return "path"


def tokenize_tags(*texts: str) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for token in WORD_RE.findall(text.lower()):
            if len(token) <= 2:
                continue
            tokens.append(token)
    return sorted(set(tokens))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def tokenize_vector_text(text: str) -> list[str]:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    return [token for token in tokens if len(token) > 2]


def build_vector_index_payload(documents: list[dict[str, Any]]) -> dict[str, Any]:
    doc_term_counts: dict[str, Counter[str]] = {}
    doc_frequency: Counter[str] = Counter()
    for document in documents:
        node_id = document["id"]
        counts = Counter(tokenize_vector_text(document["search_text"]))
        doc_term_counts[node_id] = counts
        for term in counts:
            doc_frequency[term] += 1

    vocab = sorted(doc_frequency)
    term_to_index = {term: idx for idx, term in enumerate(vocab)}
    doc_count = max(len(documents), 1)
    idf = [math.log((1.0 + doc_count) / (1.0 + doc_frequency[term])) + 1.0 for term in vocab]
    nodes: dict[str, Any] = {}

    for document in documents:
        node_id = document["id"]
        counts = doc_term_counts[node_id]
        weights: list[tuple[int, float]] = []
        norm_sq = 0.0
        for term, count in counts.items():
            idx = term_to_index[term]
            weight = (1.0 + math.log(float(count))) * idf[idx]
            norm_sq += weight * weight
            weights.append((idx, weight))
        norm = math.sqrt(norm_sq) or 1.0
        nodes[node_id] = {
            "lane": document["lane"],
            "type": document["type"],
            "status": document["status"],
            "vector": [[idx, weight / norm] for idx, weight in sorted(weights)],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_count": doc_count,
        "term_order": vocab,
        "idf": idf,
        "nodes": nodes,
        "updated_at_utc": now_utc(),
    }


def dedupe_attachments(attachments: list[Attachment]) -> list[Attachment]:
    deduped: dict[tuple[str, str, str], Attachment] = {}
    for attachment in attachments:
        key = (attachment.kind, attachment.path, attachment.label)
        deduped[key] = attachment
    return sorted(deduped.values(), key=lambda item: (item.kind, item.path, item.label))


def safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError:
        return None


def collect_metric_values(obj: Any, target_key: str) -> list[float]:
    values: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key and isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.extend(collect_metric_values(value, target_key))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(collect_metric_values(value, target_key))
    return values


def summarize_observed_results(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"run_count": 0, "status_counts": {}, "best_val_bpb": None, "best_val_loss": None}
    runs = payload.get("runs", {})
    status_counts = Counter(str(run.get("status", "unknown")) for run in runs.values())
    all_bpb = collect_metric_values(runs, "val_bpb")
    all_loss = collect_metric_values(runs, "val_loss")
    return {
        "run_count": len(runs),
        "status_counts": dict(status_counts),
        "best_val_bpb": min(all_bpb) if all_bpb else None,
        "best_val_loss": min(all_loss) if all_loss else None,
    }


def find_diagnostic_attachments(iter_dir: Path) -> list[Attachment]:
    attachments: list[Attachment] = []
    patterns = [
        "*.json",
        "results_summary.md",
        "observed_results_summary.md",
        "runtime_summary.md",
        "cluster_queue_summary.md",
        "hardmax_transfer_diagnostics/*.json",
        "hardmax_transfer_diagnostics/**/*.json",
    ]
    seen: set[str] = set()
    ignored_names = {"manifest.json", "observed_results.json"}
    ignored_dirs = {"configs"}
    for pattern in patterns:
        for path in iter_dir.glob(pattern):
            if not path.is_file():
                continue
            if path.name in ignored_names:
                continue
            if any(part in ignored_dirs for part in path.parts):
                continue
            rel = relpath(path)
            if rel in seen:
                continue
            seen.add(rel)
            attachments.append(
                Attachment(
                    kind=infer_attachment_kind(path),
                    path=rel,
                    label=path.name,
                    exists=True,
                )
            )
    return sorted(attachments, key=lambda item: (item.kind, item.path))


def load_existing_fields(existing_path: Path) -> dict[str, Any]:
    if not existing_path.exists():
        return {}
    try:
        return json.loads(read_text(existing_path))
    except json.JSONDecodeError:
        return {}


def preserve_manual_value(existing: dict[str, Any], key: str, generated: Any) -> Any:
    if key in existing and existing[key] not in (None, "", []):
        return existing[key]
    return generated


def effective_status(existing: dict[str, Any], generated: str) -> str:
    metadata = existing.get("metadata", {})
    if isinstance(metadata, dict):
        override = metadata.get("manual_status_override")
        if isinstance(override, str) and override.strip():
            return override.strip()
    return generated


def merged_metadata(existing: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    current = existing.get("metadata", {})
    if not isinstance(current, dict):
        current = {}
    return {**current, **generated}


def write_node_bundle(bundle: NodeBundle) -> None:
    ensure_store_dirs()
    bundle_dir = node_dir(bundle.node_id)
    attachments_dir = bundle_dir / "attachments"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    attachments_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_fields(bundle_dir / "node.json")
    notes_path = bundle_dir / "notes.md"
    source_snapshot_path = bundle_dir / "source_snapshot.md"

    if not notes_path.exists():
        notes_path.write_text(bundle.notes_text.rstrip() + "\n", encoding="utf-8")
    source_snapshot_path.write_text(bundle.source_snapshot_text.rstrip() + "\n", encoding="utf-8")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "id": bundle.node_id,
        "title": bundle.title,
        "type": bundle.node_type,
        "lane": bundle.lane,
        "status": effective_status(existing, bundle.status),
        "priority": preserve_manual_value(existing, "priority", bundle.priority),
        "budget": preserve_manual_value(existing, "budget", bundle.budget),
        "expected_value": preserve_manual_value(existing, "expected_value", bundle.expected_value),
        "information_gain": preserve_manual_value(existing, "information_gain", bundle.information_gain),
        "compute_cost": preserve_manual_value(existing, "compute_cost", bundle.compute_cost),
        "fragility": preserve_manual_value(existing, "fragility", bundle.fragility),
        "parent_ids": preserve_manual_value(existing, "parent_ids", bundle.parent_ids),
        "tags": sorted(set(existing.get("tags", [])) | set(bundle.tags)),
        "summary": bundle.summary,
        "source_paths": bundle.source_paths,
        "attachments": [attachment.__dict__ for attachment in bundle.attachments],
        "notes_path": "notes.md",
        "source_snapshot_path": "source_snapshot.md",
        "attachments_dir": "attachments",
        "metadata": merged_metadata(existing, bundle.metadata),
        "updated_at_utc": now_utc(),
    }
    write_json(bundle_dir / "node.json", payload)


def load_node_payload(node_id: str) -> dict[str, Any]:
    path = node_dir(node_id) / "node.json"
    if not path.exists():
        raise FileNotFoundError(f"missing node bundle for {node_id}")
    return json.loads(read_text(path))


def build_note_bundle(note_path: Path) -> NodeBundle:
    text = read_text(note_path)
    title = parse_title(text, note_path.stem)
    attachments = dedupe_attachments([
        Attachment(kind="source_note", path=relpath(note_path), label=note_path.name, exists=True),
        *extract_local_links(text, note_path),
    ])
    stem_tokens = note_path.stem.replace(".", "_")
    node_id = f"note_{slugify(note_path.parent.name + '_' + stem_tokens)}"
    lane = infer_lane(note_path)
    node_type = infer_note_type(note_path, text)
    status = infer_note_status(text)
    source_snapshot = (
        f"# Source Snapshot: {title}\n\n"
        f"- Source: `{relpath(note_path)}`\n"
        f"- Lane: `{lane}`\n"
        f"- Type: `{node_type}`\n"
        f"- Status: `{status}`\n\n"
        f"{text.strip()}\n"
    )
    metadata = {
        "generated_from": relpath(note_path),
        "ingest_kind": "project_wide_note",
        "linked_attachment_count": len(attachments) - 1,
    }
    return NodeBundle(
        node_id=node_id,
        title=title,
        node_type=node_type,
        lane=lane,
        status=status,
        priority=None,
        budget=None,
        expected_value=None,
        information_gain=None,
        compute_cost=None,
        fragility=None,
        parent_ids=[],
        tags=tokenize_tags(note_path.stem, lane, node_type, title),
        summary=extract_summary(text),
        metadata=metadata,
        attachments=attachments,
        source_paths=[relpath(note_path)],
        notes_text="# Notes\n\n- Add local branch-specific commentary, decisions, or future tasks here.\n",
        source_snapshot_text=source_snapshot,
    )


def build_sweep_bundle(iter_dir: Path) -> NodeBundle:
    manifest_path = iter_dir / "manifest.json"
    observed_path = iter_dir / "observed_results.json"
    readme_path = iter_dir / "README.md"
    manifest = safe_json_load(manifest_path) or {}
    observed = safe_json_load(observed_path)
    readme_text = read_text(readme_path) if readme_path.exists() else ""
    title = manifest.get("sweep_slug") or iter_dir.name
    summary = extract_summary(readme_text) or f"Sweep directory `{iter_dir.name}`."
    observed_summary = summarize_observed_results(observed)
    attachments: list[Attachment] = [
        Attachment(kind="manifest", path=relpath(manifest_path), label="manifest.json", exists=manifest_path.exists()),
    ]
    source_paths = [relpath(manifest_path)]
    if readme_path.exists():
        attachments.append(Attachment(kind="note", path=relpath(readme_path), label="README.md", exists=True))
        attachments.extend(extract_local_links(readme_text, readme_path))
        source_paths.append(relpath(readme_path))
    if observed_path.exists():
        attachments.append(
            Attachment(kind="observed_results", path=relpath(observed_path), label="observed_results.json", exists=True)
        )
        source_paths.append(relpath(observed_path))
    attachments.extend(find_diagnostic_attachments(iter_dir))
    attachments = dedupe_attachments(attachments)
    node_id = f"sweep_{slugify(iter_dir.name)}"
    lane = infer_lane(iter_dir)
    status = infer_sweep_status(observed)
    run_slugs = [str(run.get("run_slug", "")) for run in manifest.get("runs", []) if run.get("run_slug")]
    metadata = {
        "generated_from": relpath(iter_dir),
        "ingest_kind": "iteration_sweep",
        "dispatch_mode": manifest.get("dispatch_mode"),
        "queue_parallelism": manifest.get("queue_parallelism"),
        "run_count": observed_summary["run_count"] or manifest.get("config_count", 0),
        "status_counts": observed_summary["status_counts"],
        "best_val_bpb": observed_summary["best_val_bpb"],
        "best_val_loss": observed_summary["best_val_loss"],
        "source_spec": manifest.get("source_spec"),
        "run_slugs": run_slugs,
    }
    observed_lines = [
        f"- Best observed val_bpb: `{observed_summary['best_val_bpb']}`" if observed_summary["best_val_bpb"] is not None else "- Best observed val_bpb: `pending`",
        f"- Best observed val_loss: `{observed_summary['best_val_loss']}`" if observed_summary["best_val_loss"] is not None else "- Best observed val_loss: `pending`",
        f"- Status counts: `{json.dumps(observed_summary['status_counts'], sort_keys=True)}`",
    ]
    source_snapshot = (
        f"# Source Snapshot: {title}\n\n"
        f"- Sweep dir: `{relpath(iter_dir)}`\n"
        f"- Lane: `{lane}`\n"
        f"- Status: `{status}`\n"
        f"- Run count: `{metadata['run_count']}`\n\n"
        f"## Observed summary\n"
        + "\n".join(observed_lines)
        + "\n\n## README snapshot\n\n"
        + (readme_text.strip() if readme_text else "_No README present._")
        + "\n"
    )
    return NodeBundle(
        node_id=node_id,
        title=title,
        node_type="sweep",
        lane=lane,
        status=status,
        priority=None,
        budget=None,
        expected_value=None,
        information_gain=None,
        compute_cost=None,
        fragility=None,
        parent_ids=[],
        tags=tokenize_tags(iter_dir.name, title, lane, *run_slugs),
        summary=summary,
        metadata=metadata,
        attachments=attachments,
        source_paths=sorted(set(source_paths)),
        notes_text="# Notes\n\n- Add local branch-specific commentary, decisions, or future tasks here.\n",
        source_snapshot_text=source_snapshot,
    )


def iter_all_node_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not NODES_DIR.exists():
        return payloads
    for path in sorted(NODES_DIR.glob("*/node.json")):
        try:
            payload = json.loads(read_text(path))
        except json.JSONDecodeError:
            continue
        payload["_node_dir"] = path.parent
        payloads.append(payload)
    return payloads


def gather_search_text(payload: dict[str, Any], notes_text: str, source_snapshot_text: str) -> str:
    attachment_bits = []
    for attachment in payload.get("attachments", []):
        attachment_bits.append(f"{attachment.get('kind', '')} {attachment.get('label', '')} {attachment.get('path', '')}")
    tag_text = " ".join(payload.get("tags", []))
    parent_text = " ".join(payload.get("parent_ids", []))
    return "\n".join(
        part
        for part in (
            payload.get("title", ""),
            payload.get("summary", ""),
            tag_text,
            parent_text,
            "\n".join(attachment_bits),
            notes_text,
            source_snapshot_text,
        )
        if part
    )


def rebuild_index() -> None:
    ensure_store_dirs()
    payloads = iter_all_node_payloads()
    vector_documents: list[dict[str, Any]] = []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("DROP TABLE IF EXISTS node_fts")
    cur.execute("DROP TABLE IF EXISTS edges")
    cur.execute("DROP TABLE IF EXISTS attachments")
    cur.execute("DROP TABLE IF EXISTS nodes")
    cur.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            lane TEXT NOT NULL,
            status TEXT NOT NULL,
            priority REAL,
            budget REAL,
            expected_value REAL,
            information_gain REAL,
            compute_cost REAL,
            fragility REAL,
            updated_at_utc TEXT,
            node_dir TEXT NOT NULL,
            notes_path TEXT,
            source_snapshot_path TEXT,
            summary TEXT,
            metadata_json TEXT,
            search_text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE attachments (
            node_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            label TEXT NOT NULL,
            exists_flag INTEGER NOT NULL,
            UNIQUE(node_id, kind, path)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE edges (
            src_node_id TEXT NOT NULL,
            dst_node_id TEXT,
            dst_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            label TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX edges_unique
        ON edges (src_node_id, dst_path, kind, ifnull(label, ''))
        """
    )
    fts_enabled = True
    try:
        cur.execute(
            """
            CREATE VIRTUAL TABLE node_fts USING fts5(
                node_id UNINDEXED,
                title,
                summary,
                tags,
                body,
                attachments_text
            )
            """
        )
    except sqlite3.OperationalError:
        fts_enabled = False

    path_to_node_id: dict[str, str] = {}
    known_node_ids = {payload["id"] for payload in payloads}
    for payload in payloads:
        for source_path in payload.get("source_paths", []):
            path_to_node_id[source_path] = payload["id"]

    for payload in payloads:
        bundle_dir = Path(payload["_node_dir"])
        notes_text = read_text(bundle_dir / payload.get("notes_path", "notes.md")) if (bundle_dir / payload.get("notes_path", "notes.md")).exists() else ""
        source_snapshot_text = (
            read_text(bundle_dir / payload.get("source_snapshot_path", "source_snapshot.md"))
            if (bundle_dir / payload.get("source_snapshot_path", "source_snapshot.md")).exists()
            else ""
        )
        search_text = gather_search_text(payload, notes_text, source_snapshot_text)
        vector_documents.append(
            {
                "id": payload["id"],
                "lane": payload.get("lane", "general"),
                "type": payload.get("type", "note"),
                "status": payload.get("status", "active"),
                "search_text": search_text,
            }
        )
        cur.execute(
            """
            INSERT INTO nodes (
                id, title, type, lane, status, priority, budget, expected_value, information_gain,
                compute_cost, fragility, updated_at_utc, node_dir, notes_path, source_snapshot_path,
                summary, metadata_json, search_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload.get("title", payload["id"]),
                payload.get("type", "note"),
                payload.get("lane", "general"),
                payload.get("status", "active"),
                payload.get("priority"),
                payload.get("budget"),
                payload.get("expected_value"),
                payload.get("information_gain"),
                payload.get("compute_cost"),
                payload.get("fragility"),
                payload.get("updated_at_utc"),
                relpath(bundle_dir),
                payload.get("notes_path"),
                payload.get("source_snapshot_path"),
                payload.get("summary", ""),
                json.dumps(payload.get("metadata", {}), sort_keys=True),
                search_text,
            ),
        )
        attachment_text_bits: list[str] = []
        for attachment in payload.get("attachments", []):
            cur.execute(
                """
                INSERT OR REPLACE INTO attachments (node_id, kind, path, label, exists_flag)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    attachment.get("kind", "path"),
                    attachment.get("path", ""),
                    attachment.get("label", ""),
                    1 if attachment.get("exists") else 0,
                ),
            )
            attachment_text_bits.append(f"{attachment.get('kind', '')} {attachment.get('label', '')} {attachment.get('path', '')}")
            dst_node_id = path_to_node_id.get(attachment.get("path", ""))
            if dst_node_id and dst_node_id != payload["id"]:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO edges (src_node_id, dst_node_id, dst_path, kind, label)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        payload["id"],
                        dst_node_id,
                        attachment.get("path", ""),
                        "attachment_ref",
                        attachment.get("label"),
                    ),
                )
        notes_links = extract_local_links(source_snapshot_text, bundle_dir)
        for attachment in notes_links:
            dst_node_id = path_to_node_id.get(attachment.path)
            if dst_node_id and dst_node_id != payload["id"]:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO edges (src_node_id, dst_node_id, dst_path, kind, label)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        payload["id"],
                        dst_node_id,
                        attachment.path,
                        "markdown_link",
                        attachment.label,
                    ),
                )
        for parent_id in payload.get("parent_ids", []):
            parent_known = parent_id in known_node_ids
            parent_path = relpath(node_dir(parent_id)) if parent_known else f"node://{parent_id}"
            cur.execute(
                """
                INSERT OR REPLACE INTO edges (src_node_id, dst_node_id, dst_path, kind, label)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    parent_id if parent_known else None,
                    parent_path,
                    "parent_ref",
                    parent_id,
                ),
            )
        if fts_enabled:
            cur.execute(
                """
                INSERT INTO node_fts (node_id, title, summary, tags, body, attachments_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload.get("title", ""),
                    payload.get("summary", ""),
                    " ".join(payload.get("tags", [])),
                    "\n".join(part for part in (notes_text, source_snapshot_text) if part),
                    "\n".join(attachment_text_bits),
                ),
            )
    conn.commit()
    conn.close()
    write_json(VECTOR_INDEX_PATH, build_vector_index_payload(vector_documents))


def ingest_notes() -> int:
    count = 0
    for note_path in sorted(PROJECT_WIDE_DIR.glob("*.md")):
        write_node_bundle(build_note_bundle(note_path))
        count += 1
    return count


def ingest_sweeps() -> int:
    count = 0
    if not ITERATIONS_GENERATED_DIR.exists():
        return count
    for iter_dir in sorted(ITERATIONS_GENERATED_DIR.iterdir()):
        if not iter_dir.is_dir():
            continue
        manifest_path = iter_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        write_node_bundle(build_sweep_bundle(iter_dir))
        count += 1
    return count


def cmd_ingest(_: argparse.Namespace) -> None:
    ensure_store_dirs()
    note_count = ingest_notes()
    sweep_count = ingest_sweeps()
    rebuild_index()
    payload = {
        "store_dir": relpath(STORE_DIR),
        "node_count": len(iter_all_node_payloads()),
        "ingested_project_wide_notes": note_count,
        "ingested_sweeps": sweep_count,
        "db_path": relpath(DB_PATH),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError("branch memory index missing; run `branch_memory.py ingest` first")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_vector_index() -> dict[str, Any]:
    if not VECTOR_INDEX_PATH.exists():
        raise FileNotFoundError("branch memory vector index missing; run `branch_memory.py ingest` first")
    return json.loads(read_text(VECTOR_INDEX_PATH))


def frontier_sql(limit: int, lane: str | None, status_filter: list[str] | None) -> tuple[str, list[Any]]:
    where_clauses = ["status NOT IN ('pruned', 'complete', 'failed', 'won')"]
    params: list[Any] = []
    if lane:
        where_clauses.append("lane = ?")
        params.append(lane)
    if status_filter:
        where_clauses.append(f"status IN ({','.join('?' for _ in status_filter)})")
        params.extend(status_filter)
    sql = f"""
        SELECT
            id,
            title,
            type,
            lane,
            status,
            priority,
            summary,
            metadata_json,
            (
                COALESCE(priority, 0.0) +
                CASE status
                    WHEN 'active' THEN 40
                    WHEN 'proposed' THEN 30
                    WHEN 'blocked' THEN 15
                    WHEN 'observed' THEN 10
                    WHEN 'deferred' THEN 5
                    ELSE 0
                END +
                CASE type
                    WHEN 'branch' THEN 8
                    WHEN 'sweep' THEN 6
                    ELSE 0
                END
            ) AS frontier_score
        FROM nodes
        WHERE {' AND '.join(where_clauses)}
        ORDER BY frontier_score DESC, updated_at_utc DESC, id ASC
        LIMIT ?
    """
    params.append(limit)
    return sql, params


def cmd_list_frontier(args: argparse.Namespace) -> None:
    conn = connect_db()
    status_filter = [value.strip() for value in args.status.split(",") if value.strip()] if args.status else None
    sql, params = frontier_sql(args.limit, args.lane, status_filter)
    rows = conn.execute(sql, params).fetchall()
    result = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        result.append(
            {
                "id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "lane": row["lane"],
                "status": row["status"],
                "priority": row["priority"],
                "frontier_score": round(row["frontier_score"], 3),
                "best_val_bpb": metadata.get("best_val_bpb"),
                "run_count": metadata.get("run_count"),
                "summary": row["summary"],
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    conn.close()


def search_with_fts(conn: sqlite3.Connection, query: str, limit: int, lane: str | None) -> list[dict[str, Any]]:
    try:
        where = "WHERE node_fts MATCH ?"
        params: list[Any] = [query]
        if lane:
            where += " AND nodes.lane = ?"
            params.append(lane)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT
                nodes.id,
                nodes.title,
                nodes.type,
                nodes.lane,
                nodes.status,
                nodes.summary,
                bm25(node_fts) AS rank
            FROM node_fts
            JOIN nodes ON nodes.id = node_fts.node_id
            {where}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "type": row["type"],
            "lane": row["lane"],
            "status": row["status"],
            "summary": row["summary"],
            "rank": row["rank"],
        }
        for row in rows
    ]


def search_with_like(conn: sqlite3.Connection, query: str, limit: int, lane: str | None) -> list[dict[str, Any]]:
    like = f"%{query.lower()}%"
    where = "WHERE lower(search_text) LIKE ?"
    params: list[Any] = [like]
    if lane:
        where += " AND lane = ?"
        params.append(lane)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, title, type, lane, status, summary
        FROM nodes
        {where}
        ORDER BY updated_at_utc DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "type": row["type"],
            "lane": row["lane"],
            "status": row["status"],
            "summary": row["summary"],
            "rank": None,
        }
        for row in rows
    ]


def query_vector_scores(query: str, vector_index: dict[str, Any], lane: str | None) -> dict[str, float]:
    vocab = vector_index.get("term_order", [])
    idf = vector_index.get("idf", [])
    term_to_index = {term: idx for idx, term in enumerate(vocab)}
    counts = Counter(tokenize_vector_text(query))
    weights: dict[int, float] = {}
    norm_sq = 0.0
    for term, count in counts.items():
        idx = term_to_index.get(term)
        if idx is None:
            continue
        weight = (1.0 + math.log(float(count))) * float(idf[idx])
        weights[idx] = weight
        norm_sq += weight * weight
    norm = math.sqrt(norm_sq)
    if norm == 0.0:
        return {}
    normalized = {idx: weight / norm for idx, weight in weights.items()}
    scores: dict[str, float] = {}
    for node_id, node_payload in vector_index.get("nodes", {}).items():
        if lane and node_payload.get("lane") != lane:
            continue
        score = 0.0
        for idx, weight in node_payload.get("vector", []):
            query_weight = normalized.get(int(idx))
            if query_weight is not None:
                score += query_weight * float(weight)
        if score > 0.0:
            scores[node_id] = score
    return scores


def search_with_vector(conn: sqlite3.Connection, query: str, limit: int, lane: str | None) -> list[dict[str, Any]]:
    vector_index = load_vector_index()
    scores = query_vector_scores(query, vector_index, lane)
    top_ids = [node_id for node_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]
    if not top_ids:
        return []
    placeholders = ",".join("?" for _ in top_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, type, lane, status, summary
        FROM nodes
        WHERE id IN ({placeholders})
        """,
        top_ids,
    ).fetchall()
    row_map = {row["id"]: row for row in rows}
    return [
        {
            "id": node_id,
            "title": row_map[node_id]["title"],
            "type": row_map[node_id]["type"],
            "lane": row_map[node_id]["lane"],
            "status": row_map[node_id]["status"],
            "summary": row_map[node_id]["summary"],
            "rank": None,
            "vector_score": scores[node_id],
        }
        for node_id in top_ids
        if node_id in row_map
    ]


def reciprocal_rank_fusion(
    result_lists: dict[str, list[dict[str, Any]]],
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_name, results in result_lists.items():
        for rank_idx, result in enumerate(results, start=1):
            node_id = result["id"]
            entry = merged.setdefault(
                node_id,
                {
                    "id": node_id,
                    "title": result["title"],
                    "type": result["type"],
                    "lane": result["lane"],
                    "status": result["status"],
                    "summary": result["summary"],
                    "hybrid_score": 0.0,
                    "sources": {},
                },
            )
            entry["hybrid_score"] += 1.0 / (rrf_k + rank_idx)
            entry["sources"][source_name] = {
                key: value
                for key, value in result.items()
                if key not in {"id", "title", "type", "lane", "status", "summary"}
            }
    ranked = sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)
    return ranked[:limit]


def cmd_search(args: argparse.Namespace) -> None:
    conn = connect_db()
    mode = args.mode
    try:
        if mode == "fts":
            results = search_with_fts(conn, args.query, args.limit, args.lane)
            if not results:
                results = search_with_like(conn, args.query, args.limit, args.lane)
        elif mode == "vector":
            results = search_with_vector(conn, args.query, args.limit, args.lane)
        elif mode == "like":
            results = search_with_like(conn, args.query, args.limit, args.lane)
        else:
            fts_results = search_with_fts(conn, args.query, max(args.limit * 4, 20), args.lane)
            vector_results = search_with_vector(conn, args.query, max(args.limit * 4, 20), args.lane)
            if not fts_results and not vector_results:
                results = search_with_like(conn, args.query, args.limit, args.lane)
            else:
                results = reciprocal_rank_fusion(
                    {"bm25": fts_results, "vector": vector_results},
                    args.limit,
                )
    except FileNotFoundError:
        results = search_with_fts(conn, args.query, args.limit, args.lane)
        if not results:
            results = search_with_like(conn, args.query, args.limit, args.lane)
    print(json.dumps(results, indent=2, sort_keys=True))
    conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    payload = load_node_payload(args.node_id)
    conn = connect_db()
    edges = conn.execute(
        """
        SELECT src_node_id, dst_node_id, dst_path, kind, label
        FROM edges
        WHERE src_node_id = ? OR dst_node_id = ?
        ORDER BY kind, dst_path
        """,
        (args.node_id, args.node_id),
    ).fetchall()
    notes_text = read_text(node_dir(args.node_id) / payload.get("notes_path", "notes.md"))
    snapshot_text = read_text(node_dir(args.node_id) / payload.get("source_snapshot_path", "source_snapshot.md"))
    output = {
        "node": payload,
        "edges": [dict(row) for row in edges],
        "notes_text": notes_text,
        "source_snapshot_text": snapshot_text,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    conn.close()


def cmd_create_node(args: argparse.Namespace) -> None:
    node_id = args.node_id or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{slugify(args.title)}"
    bundle = NodeBundle(
        node_id=node_id,
        title=args.title,
        node_type=args.node_type,
        lane=args.lane,
        status=args.status,
        priority=args.priority,
        budget=args.budget,
        expected_value=None,
        information_gain=None,
        compute_cost=None,
        fragility=None,
        parent_ids=args.parent_id or [],
        tags=sorted(set(args.tag)),
        summary=args.summary or args.title,
        metadata={"created_manually": True},
        attachments=[],
        source_paths=[],
        notes_text=args.notes or "# Notes\n\n- Fill in branch context here.\n",
        source_snapshot_text=f"# Source Snapshot: {args.title}\n\n_Manual node._\n",
    )
    write_node_bundle(bundle)
    notes_path = node_dir(node_id) / "notes.md"
    notes_path.write_text((args.notes or "# Notes\n\n- Fill in branch context here.\n").rstrip() + "\n", encoding="utf-8")
    rebuild_index()
    print(json.dumps({"created": node_id, "path": relpath(node_dir(node_id))}, indent=2, sort_keys=True))


def cmd_link_node(args: argparse.Namespace) -> None:
    payload = load_node_payload(args.node_id)
    payload["parent_ids"] = unique_preserve_order(payload.get("parent_ids", []) + (args.parent_id or []))
    payload["tags"] = sorted(set(payload.get("tags", [])) | set(args.tag or []))
    payload["updated_at_utc"] = now_utc()
    write_json(node_dir(args.node_id) / "node.json", payload)
    rebuild_index()
    print(
        json.dumps(
            {
                "node_id": args.node_id,
                "parent_ids": payload["parent_ids"],
                "tags": payload["tags"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_create_merge_node(args: argparse.Namespace) -> None:
    merge_of = unique_preserve_order(args.from_node_id or [])
    if not merge_of:
        raise SystemExit("create-merge-node requires at least one --from-node-id")
    node_id = args.node_id or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{slugify(args.title)}"

    attachments: list[Attachment] = []
    seen_attachments: set[tuple[str, str, str]] = set()
    source_paths: list[str] = []
    parent_titles: list[str] = []
    for parent_id in merge_of:
        parent_payload = load_node_payload(parent_id)
        parent_titles.append(parent_payload.get("title", parent_id))
        for source_path in parent_payload.get("source_paths", []):
            resolved = resolve_repo_path(source_path, ROOT)
            attachment = Attachment(
                kind="merge_source",
                path=source_path,
                label=f"{parent_id}:{Path(source_path).name}",
                exists=resolved.exists() if resolved else False,
            )
            key = (attachment.kind, attachment.path, attachment.label)
            if key in seen_attachments:
                continue
            seen_attachments.add(key)
            attachments.append(attachment)
            source_paths.append(source_path)

    notes_body = args.notes or (
        "# Notes\n\n"
        "This is a non-destructive compaction/bridge node.\n\n"
        "Merged parents:\n"
        + "".join(f"- `{parent_id}`\n" for parent_id in merge_of)
    )
    source_snapshot_text = (
        f"# Source Snapshot: {args.title}\n\n"
        f"- Merge of: `{', '.join(merge_of)}`\n"
        f"- Parent count: `{len(merge_of)}`\n"
        f"- Lane: `{args.lane}`\n\n"
        "## Parent titles\n"
        + "".join(f"- `{title}`\n" for title in parent_titles)
        + "\n"
    )
    bundle = NodeBundle(
        node_id=node_id,
        title=args.title,
        node_type=args.node_type,
        lane=args.lane,
        status=args.status,
        priority=args.priority,
        budget=args.budget,
        expected_value=None,
        information_gain=None,
        compute_cost=None,
        fragility=None,
        parent_ids=merge_of,
        tags=sorted(set((args.tag or []) + ["merge", "compaction"])),
        summary=args.summary or f"Compaction/bridge node spanning {len(merge_of)} parent branches.",
        metadata={
            "created_manually": True,
            "compaction": True,
            "merge_of": merge_of,
        },
        attachments=dedupe_attachments(attachments),
        source_paths=sorted(set(source_paths)),
        notes_text=notes_body,
        source_snapshot_text=source_snapshot_text,
    )
    write_node_bundle(bundle)
    (node_dir(node_id) / "notes.md").write_text(notes_body.rstrip() + "\n", encoding="utf-8")
    rebuild_index()
    print(
        json.dumps(
            {
                "created": node_id,
                "path": relpath(node_dir(node_id)),
                "merge_of": merge_of,
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_attach_path(args: argparse.Namespace) -> None:
    payload = load_node_payload(args.node_id)
    resolved = Path(args.path).resolve()
    attachment = {
        "kind": args.kind,
        "path": relpath(resolved),
        "label": args.label or resolved.name,
        "exists": resolved.exists(),
    }
    attachments = payload.get("attachments", [])
    if attachment not in attachments:
        attachments.append(attachment)
    payload["attachments"] = sorted(attachments, key=lambda item: (item.get("kind", ""), item.get("path", "")))
    payload["updated_at_utc"] = now_utc()
    write_json(node_dir(args.node_id) / "node.json", payload)
    rebuild_index()
    print(json.dumps({"attached_to": args.node_id, "attachment": attachment}, indent=2, sort_keys=True))


def cmd_set_status(args: argparse.Namespace) -> None:
    payload = load_node_payload(args.node_id)
    payload["status"] = args.status
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["manual_status_override"] = args.status
    payload["metadata"] = metadata
    payload["updated_at_utc"] = now_utc()
    write_json(node_dir(args.node_id) / "node.json", payload)
    rebuild_index()
    print(json.dumps({"node_id": args.node_id, "status": args.status}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filesystem-native branch memory for MCTS-style research planning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest project_wide notes and generated sweeps into branch memory.")
    ingest.set_defaults(func=cmd_ingest)

    frontier = subparsers.add_parser("list-frontier", help="List active frontier nodes sorted by a simple heuristic.")
    frontier.add_argument("--limit", type=int, default=20, help="Maximum number of nodes to print.")
    frontier.add_argument("--lane", help="Optional lane filter, for example hardmax.")
    frontier.add_argument(
        "--status",
        help="Optional comma-separated status filter, for example active,proposed,observed.",
    )
    frontier.set_defaults(func=cmd_list_frontier)

    search = subparsers.add_parser("search", help="Search node notes, snapshots, and attachment refs.")
    search.add_argument("query", help="FTS or substring query.")
    search.add_argument("--limit", type=int, default=10, help="Maximum number of hits.")
    search.add_argument("--lane", help="Optional lane filter, for example hardmax.")
    search.add_argument(
        "--mode",
        choices=("hybrid", "fts", "vector", "like"),
        default="hybrid",
        help="Retrieval mode. `hybrid` combines SQLite BM25 and a derived TF-IDF vector index.",
    )
    search.set_defaults(func=cmd_search)

    show = subparsers.add_parser("show", help="Show a node bundle with edges and note text.")
    show.add_argument("node_id", help="Node identifier.")
    show.set_defaults(func=cmd_show)

    create = subparsers.add_parser("create-node", help="Create a manual branch node.")
    create.add_argument("--node-id", help="Optional explicit node identifier.")
    create.add_argument("--title", required=True, help="Node title.")
    create.add_argument("--type", dest="node_type", default="task", help="Node type.")
    create.add_argument("--lane", default="general", help="Research lane.")
    create.add_argument("--status", default="proposed", help="Initial node status.")
    create.add_argument("--priority", type=float, help="Optional priority score.")
    create.add_argument("--budget", type=float, help="Optional budget share.")
    create.add_argument("--summary", help="Short summary.")
    create.add_argument("--notes", help="Initial notes body.")
    create.add_argument("--parent-id", action="append", default=[], help="Optional parent node id. Repeatable.")
    create.add_argument("--tag", action="append", default=[], help="Optional tag. Repeatable.")
    create.set_defaults(func=cmd_create_node)

    link = subparsers.add_parser("link-node", help="Add parent links or tags to an existing node without rewriting it.")
    link.add_argument("--node-id", required=True, help="Existing node id.")
    link.add_argument("--parent-id", action="append", default=[], help="Parent node id to add. Repeatable.")
    link.add_argument("--tag", action="append", default=[], help="Optional tag to add. Repeatable.")
    link.set_defaults(func=cmd_link_node)

    merge = subparsers.add_parser(
        "create-merge-node",
        help="Create a non-destructive bridge/compaction node spanning multiple parent nodes.",
    )
    merge.add_argument("--node-id", help="Optional explicit node identifier.")
    merge.add_argument("--title", required=True, help="Node title.")
    merge.add_argument("--type", dest="node_type", default="branch", help="Node type.")
    merge.add_argument("--lane", default="general", help="Research lane.")
    merge.add_argument("--status", default="proposed", help="Initial node status.")
    merge.add_argument("--priority", type=float, help="Optional priority score.")
    merge.add_argument("--budget", type=float, help="Optional budget share.")
    merge.add_argument("--summary", help="Short summary.")
    merge.add_argument("--notes", help="Initial notes body.")
    merge.add_argument("--from-node-id", action="append", default=[], help="Parent/source node id. Repeatable.")
    merge.add_argument("--tag", action="append", default=[], help="Optional tag. Repeatable.")
    merge.set_defaults(func=cmd_create_merge_node)

    attach = subparsers.add_parser("attach-path", help="Attach a repo path or artifact path to an existing node.")
    attach.add_argument("--node-id", required=True, help="Existing node id.")
    attach.add_argument("--kind", required=True, help="Attachment kind, for example code or result_json.")
    attach.add_argument("--path", required=True, help="Path to attach.")
    attach.add_argument("--label", help="Optional attachment label.")
    attach.set_defaults(func=cmd_attach_path)

    set_status = subparsers.add_parser("set-status", help="Update node status and rebuild the derived index.")
    set_status.add_argument("--node-id", required=True, help="Existing node id.")
    set_status.add_argument("--status", required=True, help="New status.")
    set_status.set_defaults(func=cmd_set_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
