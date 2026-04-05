#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from cluster_queue_snapshot import build_cluster_snapshot, write_snapshot as write_cluster_snapshot
    from check_mlx_sweep_status import summarize_sweep
    from plot_training_curves import parse_log
    from run_iteration_saved_diagnostics import build_run_entries, load_json, RunEntry
    from update_iteration_observed_results import (
        collect_search_roots,
        discover_named_file,
        extract_embedded_dispatch_logs,
        find_existing_file,
    )
except ModuleNotFoundError:
    from tools.cluster_queue_snapshot import build_cluster_snapshot, write_snapshot as write_cluster_snapshot
    from tools.check_mlx_sweep_status import summarize_sweep
    from tools.plot_training_curves import parse_log
    from tools.run_iteration_saved_diagnostics import build_run_entries, load_json, RunEntry
    from tools.update_iteration_observed_results import (
        collect_search_roots,
        discover_named_file,
        extract_embedded_dispatch_logs,
        find_existing_file,
    )


ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = ROOT / "research" / "iterations" / "derived"
DEFAULT_DB_PATH = DERIVED_DIR / "runtime_metrics.sqlite"
ARTIFACT_SUFFIXES = (
    "_mlx_model.npz",
    "_mlx_model.int8.ptz",
    "_int8zlib.pklz",
    "_trace_pretrain_model.npz",
    "_hardmax_controller_init.npz",
    ".summary.json",
)
IGNORED_ANALYSIS_JSON_NAMES = {
    "manifest.json",
    "observed_results.json",
    "runtime_state.json",
    "runtime_metrics_manifest.json",
    "cluster_queue_snapshot.json",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge generated MLX queue sweeps into structured runtime state and searchable metric indices."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser(
        "update",
        help="Write runtime_state.json plus a searchable SQLite metric index for one generated iteration.",
    )
    update.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json.")
    update.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    update.add_argument(
        "--runtime-output",
        default="",
        help="Output path for runtime_state.json. Defaults to <iteration_dir>/runtime_state.json",
    )
    update.add_argument(
        "--manifest-output",
        default="",
        help="Output path for runtime_metrics_manifest.json. Defaults to <iteration_dir>/runtime_metrics_manifest.json",
    )
    update.add_argument(
        "--summary-output",
        default="",
        help="Output path for runtime_summary.md. Defaults to <iteration_dir>/runtime_summary.md",
    )
    update.add_argument(
        "--cluster-root",
        default=str(Path.home() / "cluster"),
        help="Path to the external cluster control directory.",
    )
    update.add_argument("--redis-host", default="127.0.0.1", help="Redis host used by the cluster queue.")
    update.add_argument(
        "--cluster-snapshot-output",
        default="",
        help="Output path for cluster_queue_snapshot.json. Defaults to <iteration_dir>/cluster_queue_snapshot.json",
    )
    update.add_argument(
        "--cluster-summary-output",
        default="",
        help="Output path for cluster_queue_summary.md. Defaults to <iteration_dir>/cluster_queue_summary.md",
    )
    update.add_argument("--skip-cluster-snapshot", action="store_true", help="Skip direct cluster queue snapshot ingestion.")
    update.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional directory to search recursively for logs/artifacts.",
    )
    update.add_argument("--check-remote", action="store_true", help="SSH claimed hosts to confirm whether jobs are still running.")
    update.add_argument("--stdout-summary", action="store_true", help="Print a short JSON summary after writing outputs.")
    update.set_defaults(func=cmd_update)

    search = subparsers.add_parser(
        "search-metric",
        help="Query the runtime SQLite index for the latest value of a metric per run.",
    )
    search.add_argument("metric", help="Metric name, for example val_bpb or residnov_mean.")
    search.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    search.add_argument("--phase", default="", help="Optional phase filter: train, val, or final.")
    search.add_argument("--run-like", default="", help="Optional substring filter on run_slug.")
    search.add_argument("--iteration-like", default="", help="Optional substring filter on iteration_dir.")
    search.add_argument("--host", default="", help="Optional exact host filter.")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--order", choices=("asc", "desc"), default="", help="Sort direction for the metric value.")
    search.set_defaults(func=cmd_search_metric)

    runs = subparsers.add_parser(
        "list-runs",
        help="List indexed runs with latest queue/runtime state.",
    )
    runs.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    runs.add_argument("--status", default="", help="Optional queue status filter.")
    runs.add_argument("--run-like", default="", help="Optional substring filter on run_slug.")
    runs.add_argument("--iteration-like", default="", help="Optional substring filter on iteration_dir.")
    runs.add_argument("--host", default="", help="Optional exact host filter.")
    runs.add_argument("--limit", type=int, default=50)
    runs.set_defaults(func=cmd_list_runs)

    hosts = subparsers.add_parser(
        "list-cluster-hosts",
        help="List the latest indexed cluster host state.",
    )
    hosts.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    hosts.add_argument("--status", default="", help="Optional exact status filter.")
    hosts.add_argument("--host-like", default="", help="Optional substring filter on host.")
    hosts.add_argument("--limit", type=int, default=50)
    hosts.set_defaults(func=cmd_list_cluster_hosts)

    jobs = subparsers.add_parser(
        "list-cluster-jobs",
        help="List the latest indexed cluster job metadata.",
    )
    jobs.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    jobs.add_argument("--status", default="", help="Optional exact status filter.")
    jobs.add_argument("--host", default="", help="Optional exact host filter.")
    jobs.add_argument("--script-like", default="", help="Optional substring filter on script.")
    jobs.add_argument("--limit", type=int, default=100)
    jobs.set_defaults(func=cmd_list_cluster_jobs)

    analysis = subparsers.add_parser(
        "search-analysis-metric",
        help="Query indexed residual/prosody/comparison metrics from saved analysis artifacts.",
    )
    analysis.add_argument("metric", help="Metric path, for example mean_nll or nll_acf.summary.all.mean.")
    analysis.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite index path.")
    analysis.add_argument("--iteration-like", default="", help="Optional substring filter on iteration_dir.")
    analysis.add_argument("--subject-like", default="", help="Optional substring filter on subject_slug.")
    analysis.add_argument("--source-like", default="", help="Optional substring filter on source_path.")
    analysis.add_argument("--metric-like", default="", help="Optional substring filter on metric path.")
    analysis.add_argument("--limit", type=int, default=50)
    analysis.add_argument("--order", choices=("asc", "desc"), default="", help="Sort direction for value/delta.")
    analysis.add_argument("--delta-only", action="store_true", help="Order/search on delta_value instead of value.")
    analysis.set_defaults(func=cmd_search_analysis_metric)
    return parser.parse_args()


def compact_metrics_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, str)):
            out[key] = value
    return out or None


def load_observed_results(iteration_dir: Path) -> dict[str, Any]:
    path = iteration_dir / "observed_results.json"
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def find_log_for_entry(
    entry: RunEntry,
    *,
    observed_run: dict[str, Any],
    embedded_logs: dict[str, Path],
    search_roots: list[Path],
) -> Path | None:
    log_path = observed_run.get("log_path")
    if isinstance(log_path, str):
        found = find_existing_file(log_path, base_dirs=search_roots)
        if found is not None:
            return found
    embedded = embedded_logs.get(entry.run_slug)
    if embedded is not None and embedded.exists():
        return embedded.resolve()
    discovered = discover_named_file(entry, search_roots=search_roots, suffix=".txt")
    return None if discovered is None else discovered.resolve()


def find_artifact_for_entry(
    entry: RunEntry,
    *,
    observed_run: dict[str, Any],
    search_roots: list[Path],
) -> Path | None:
    artifact_path = observed_run.get("artifact_path")
    if isinstance(artifact_path, str):
        found = find_existing_file(artifact_path, base_dirs=search_roots)
        if found is not None:
            return found
    for suffix in ARTIFACT_SUFFIXES:
        discovered = discover_named_file(entry, search_roots=search_roots, suffix=suffix)
        if discovered is not None:
            return discovered.resolve()
    return None


def first_final_metrics(finals: dict[str, Any]) -> dict[str, Any] | None:
    preferred = (
        "final_int8_zlib_roundtrip_exact",
        "final_raw_export_ready_exact",
        "final_trace_pretrain_best_val",
    )
    for key in preferred:
        value = finals.get(key)
        if isinstance(value, dict):
            return value
    for value in finals.values():
        if isinstance(value, dict):
            return value
    return None


def numeric_metrics(row: dict[str, Any], *, exclude: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if key in exclude:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def discover_analysis_jsons(iteration_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    patterns = (
        "results/*.json",
        "results/**/*.json",
        "hardmax_transfer_diagnostics/*.json",
        "hardmax_transfer_diagnostics/**/*.json",
    )
    for pattern in patterns:
        candidates.extend(iteration_dir.glob(pattern))
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(candidates):
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if path.name in IGNORED_ANALYSIS_JSON_NAMES:
            continue
        ordered.append(resolved)
    return ordered


def safe_mean_from_trace(rows: list[dict[str, Any]] | Any) -> float | None:
    if not isinstance(rows, list):
        return None
    corrs = [float(item["corr"]) for item in rows if isinstance(item, dict) and isinstance(item.get("corr"), (int, float))]
    if not corrs:
        return None
    return sum(corrs) / float(len(corrs))


def safe_positive_area_from_trace(rows: list[dict[str, Any]] | Any) -> float | None:
    if not isinstance(rows, list):
        return None
    corrs = [max(0.0, float(item["corr"])) for item in rows if isinstance(item, dict) and isinstance(item.get("corr"), (int, float))]
    if not corrs:
        return None
    return sum(corrs)


def flatten_scalar_leaves(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_scalar_leaves(value, child))
    elif isinstance(obj, list):
        return out
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)):
        out.append((prefix, float(obj)))
    return out


def extract_comparison_metrics(
    payload: dict[str, Any],
    *,
    source_path: str,
    updated_at_utc: str,
) -> list[dict[str, Any]]:
    subject = f"{payload.get('left_label','left')}__vs__{payload.get('right_label','right')}"
    rows: list[dict[str, Any]] = []
    metrics = payload.get("metrics", [])
    if not isinstance(metrics, list):
        return rows
    for item in metrics:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("path", "") or "")
        if not metric:
            continue
        left = item.get("left")
        right = item.get("right")
        delta = item.get("delta")
        if not any(isinstance(value, (int, float)) for value in (left, right, delta)):
            continue
        rows.append(
            {
                "source_path": source_path,
                "source_kind": "comparison",
                "subject_slug": subject,
                "metric": metric,
                "value": float(delta) if isinstance(delta, (int, float)) else None,
                "left_value": float(left) if isinstance(left, (int, float)) else None,
                "right_value": float(right) if isinstance(right, (int, float)) else None,
                "delta_value": float(delta) if isinstance(delta, (int, float)) else None,
                "updated_at_utc": updated_at_utc,
            }
        )
    return rows


def extract_result_metrics(
    payload: dict[str, Any],
    *,
    source_path: str,
    updated_at_utc: str,
) -> list[dict[str, Any]]:
    subject = str(payload.get("label", "") or Path(source_path).stem)
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        rows.append(
            {
                "source_path": source_path,
                "source_kind": "result",
                "subject_slug": subject,
                "metric": metric,
                "value": float(value),
                "left_value": None,
                "right_value": None,
                "delta_value": None,
                "updated_at_utc": updated_at_utc,
            }
        )

    add("mean_nll", payload.get("mean_nll"))
    add("positions_analyzed", payload.get("positions_analyzed"))
    add("state_dim", payload.get("state_dim"))

    for metric, value in flatten_scalar_leaves(payload.get("analysis_params", {}), "analysis_params"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("regime", {}), "regime"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("token_class_loss", {}), "token_class_loss"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("prosody_correlations", {}), "prosody_correlations"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("prosody_probes", {}), "prosody_probes"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("boundary_conditioned", {}), "boundary_conditioned"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("quote_conditioned", {}), "quote_conditioned"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("transition_window", {}), "transition_window"):
        add(metric, value)
    for metric, value in flatten_scalar_leaves(payload.get("layerwise_regime", {}), "layerwise_regime"):
        add(metric, value)

    nll_acf = payload.get("nll_acf", {})
    for metric, value in flatten_scalar_leaves(nll_acf.get("summary", {}) if isinstance(nll_acf, dict) else {}, "nll_acf.summary"):
        add(metric, value)
    if isinstance(nll_acf, dict):
        add("nll_acf.all.trace_mean", safe_mean_from_trace(nll_acf.get("all")))
        add("nll_acf.within_regime.trace_mean", safe_mean_from_trace(nll_acf.get("within_regime")))
        add("nll_acf.cross_regime.trace_mean", safe_mean_from_trace(nll_acf.get("cross_regime")))
        add("nll_acf.all.positive_area_trace", safe_positive_area_from_trace(nll_acf.get("all")))

    residual_modes = payload.get("residual_modes", {})
    if isinstance(residual_modes, dict):
        for mode_name, mode_payload in residual_modes.items():
            if not isinstance(mode_payload, dict):
                continue
            base = f"residual_modes.{mode_name}"
            add(f"{base}.mean_residual_norm", mode_payload.get("mean_residual_norm"))
            for metric, value in flatten_scalar_leaves(mode_payload.get("acf_summary", {}), f"{base}.acf_summary"):
                add(metric, value)
            add(f"{base}.acf_all.trace_mean", safe_mean_from_trace(mode_payload.get("acf_all")))
            add(f"{base}.acf_within_regime.trace_mean", safe_mean_from_trace(mode_payload.get("acf_within_regime")))
            add(f"{base}.acf_cross_regime.trace_mean", safe_mean_from_trace(mode_payload.get("acf_cross_regime")))
            for metric, value in flatten_scalar_leaves(mode_payload.get("transition_window", {}), f"{base}.transition_window"):
                add(metric, value)
            factorized = mode_payload.get("factorized_acf", {})
            if isinstance(factorized, dict):
                add(f"{base}.factorized_acf.num_factors", factorized.get("num_factors"))
                add(
                    f"{base}.factorized_acf.explained_variance_ratio_sum",
                    factorized.get("explained_variance_ratio_sum"),
                )
                factors = factorized.get("factors", [])
                if isinstance(factors, list):
                    for factor in factors:
                        if not isinstance(factor, dict):
                            continue
                        idx = factor.get("factor_index")
                        if not isinstance(idx, int):
                            continue
                        fbase = f"{base}.factorized_acf.factor_{idx}"
                        add(f"{fbase}.explained_variance_ratio", factor.get("explained_variance_ratio"))
                        add(f"{fbase}.score_mean", factor.get("score_mean"))
                        add(f"{fbase}.score_std", factor.get("score_std"))
                        for metric, value in flatten_scalar_leaves(factor.get("acf_summary", {}), f"{fbase}.acf_summary"):
                            add(metric, value)
                        for metric, value in flatten_scalar_leaves(factor.get("transition_window_abs_score", {}), f"{fbase}.transition_window_abs_score"):
                            add(metric, value)
                        add(f"{fbase}.acf_all.trace_mean", safe_mean_from_trace(factor.get("acf_all")))
                        add(f"{fbase}.acf_within_regime.trace_mean", safe_mean_from_trace(factor.get("acf_within_regime")))
                        add(f"{fbase}.acf_cross_regime.trace_mean", safe_mean_from_trace(factor.get("acf_cross_regime")))

    return rows


def extract_analysis_rows(iteration_dir: Path, *, updated_at_utc: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_analysis_jsons(iteration_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        source_path = rel_to_iteration(iteration_dir, path)
        if isinstance(payload.get("metrics"), list) and {"left", "right", "left_label", "right_label"} <= set(payload.keys()):
            rows.extend(extract_comparison_metrics(payload, source_path=source_path, updated_at_utc=updated_at_utc))
        else:
            rows.extend(extract_result_metrics(payload, source_path=source_path, updated_at_utc=updated_at_utc))
    return rows


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, spec in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def rel_to_iteration(iteration_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(iteration_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            iteration_dir TEXT NOT NULL,
            run_slug TEXT NOT NULL,
            run_id TEXT NOT NULL,
            config_name TEXT NOT NULL,
            trainer_module TEXT,
            queue_status TEXT,
            observed_status TEXT,
            host TEXT,
            job_id TEXT,
            remote_dir TEXT,
            local_log_path TEXT,
            artifact_path TEXT,
            latest_train_step INTEGER,
            latest_train_loss REAL,
            latest_val_step INTEGER,
            latest_val_loss REAL,
            latest_val_bpb REAL,
            cluster_job_status TEXT,
            cluster_host_status TEXT,
            cluster_lock_owner TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (iteration_dir, run_slug)
        )
        """
    )
    ensure_columns(
        conn,
        "runs",
        {
            "cluster_job_status": "TEXT",
            "cluster_host_status": "TEXT",
            "cluster_lock_owner": "TEXT",
        },
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalar_points (
            iteration_dir TEXT NOT NULL,
            run_slug TEXT NOT NULL,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            point_key TEXT NOT NULL,
            step INTEGER NOT NULL,
            elapsed_min REAL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (iteration_dir, run_slug, phase, point_key, step, metric)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalar_points_metric ON scalar_points(metric, phase, value)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalar_points_run ON scalar_points(iteration_dir, run_slug, phase, step)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_host_status ON runs(host, queue_status)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_hosts (
            host TEXT PRIMARY KEY,
            status TEXT,
            load_avg TEXT,
            memory TEXT,
            job_display TEXT,
            job_status TEXT,
            disk_used TEXT,
            lock_owner TEXT,
            lock_ttl INTEGER,
            reachable INTEGER,
            python_procs INTEGER,
            active_job_id TEXT,
            active_job_status TEXT,
            active_script TEXT,
            active_remote_dir TEXT,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_jobs (
            job_id TEXT PRIMARY KEY,
            host TEXT,
            script TEXT,
            started TEXT,
            ended TEXT,
            status TEXT,
            pid TEXT,
            remote_dir TEXT,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_hosts_status ON cluster_hosts(status, host)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_jobs_status ON cluster_jobs(status, host, started)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_metrics (
            iteration_dir TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            subject_slug TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            left_value REAL,
            right_value REAL,
            delta_value REAL,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (iteration_dir, source_path, source_kind, subject_slug, metric)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_metrics_metric ON analysis_metrics(metric, value, delta_value)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_metrics_subject ON analysis_metrics(subject_slug, source_kind)")


def upsert_run(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            iteration_dir, run_slug, run_id, config_name, trainer_module, queue_status, observed_status,
            host, job_id, remote_dir, local_log_path, artifact_path,
            latest_train_step, latest_train_loss, latest_val_step, latest_val_loss, latest_val_bpb,
            cluster_job_status, cluster_host_status, cluster_lock_owner, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(iteration_dir, run_slug) DO UPDATE SET
            run_id=excluded.run_id,
            config_name=excluded.config_name,
            trainer_module=excluded.trainer_module,
            queue_status=excluded.queue_status,
            observed_status=excluded.observed_status,
            host=excluded.host,
            job_id=excluded.job_id,
            remote_dir=excluded.remote_dir,
            local_log_path=excluded.local_log_path,
            artifact_path=excluded.artifact_path,
            latest_train_step=excluded.latest_train_step,
            latest_train_loss=excluded.latest_train_loss,
            latest_val_step=excluded.latest_val_step,
            latest_val_loss=excluded.latest_val_loss,
            latest_val_bpb=excluded.latest_val_bpb,
            cluster_job_status=excluded.cluster_job_status,
            cluster_host_status=excluded.cluster_host_status,
            cluster_lock_owner=excluded.cluster_lock_owner,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            payload["iteration_dir"],
            payload["run_slug"],
            payload["run_id"],
            payload["config_name"],
            payload.get("trainer_module", ""),
            payload.get("queue_status", ""),
            payload.get("observed_status", ""),
            payload.get("host", ""),
            payload.get("job_id", ""),
            payload.get("remote_dir", ""),
            payload.get("local_log_path", ""),
            payload.get("artifact_path", ""),
            payload.get("latest_train_step"),
            payload.get("latest_train_loss"),
            payload.get("latest_val_step"),
            payload.get("latest_val_loss"),
            payload.get("latest_val_bpb"),
            payload.get("cluster_job_status", ""),
            payload.get("cluster_host_status", ""),
            payload.get("cluster_lock_owner", ""),
            payload["updated_at_utc"],
        ),
    )


def upsert_cluster_host(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO cluster_hosts (
            host, status, load_avg, memory, job_display, job_status, disk_used,
            lock_owner, lock_ttl, reachable, python_procs,
            active_job_id, active_job_status, active_script, active_remote_dir, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(host) DO UPDATE SET
            status=excluded.status,
            load_avg=excluded.load_avg,
            memory=excluded.memory,
            job_display=excluded.job_display,
            job_status=excluded.job_status,
            disk_used=excluded.disk_used,
            lock_owner=excluded.lock_owner,
            lock_ttl=excluded.lock_ttl,
            reachable=excluded.reachable,
            python_procs=excluded.python_procs,
            active_job_id=excluded.active_job_id,
            active_job_status=excluded.active_job_status,
            active_script=excluded.active_script,
            active_remote_dir=excluded.active_remote_dir,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            payload.get("host", ""),
            payload.get("status", ""),
            payload.get("load_avg", ""),
            payload.get("memory", ""),
            payload.get("job_display", ""),
            payload.get("job_status", ""),
            payload.get("disk_used", ""),
            payload.get("lock_owner", ""),
            payload.get("lock_ttl"),
            1 if payload.get("reachable") else 0,
            payload.get("python_procs"),
            payload.get("active_job_id", ""),
            payload.get("active_job_status", ""),
            payload.get("active_script", ""),
            payload.get("active_remote_dir", ""),
            payload.get("updated_at_utc", now_utc()),
        ),
    )


def upsert_cluster_job(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    job_id = str(payload.get("job_id", "") or "")
    if not job_id:
        return
    conn.execute(
        """
        INSERT INTO cluster_jobs (
            job_id, host, script, started, ended, status, pid, remote_dir, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            host=excluded.host,
            script=excluded.script,
            started=excluded.started,
            ended=excluded.ended,
            status=excluded.status,
            pid=excluded.pid,
            remote_dir=excluded.remote_dir,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            job_id,
            payload.get("host", ""),
            payload.get("script", ""),
            payload.get("started", ""),
            payload.get("ended", ""),
            payload.get("status", ""),
            payload.get("pid", ""),
            payload.get("remote_dir", ""),
            payload.get("updated_at_utc", now_utc()),
        ),
    )


def replace_analysis_metrics(
    conn: sqlite3.Connection,
    *,
    iteration_dir: str,
    rows: list[dict[str, Any]],
) -> int:
    conn.execute("DELETE FROM analysis_metrics WHERE iteration_dir = ?", (iteration_dir,))
    inserted = 0
    for row in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_metrics (
                iteration_dir, source_path, source_kind, subject_slug, metric,
                value, left_value, right_value, delta_value, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iteration_dir,
                row.get("source_path", ""),
                row.get("source_kind", ""),
                row.get("subject_slug", ""),
                row.get("metric", ""),
                row.get("value"),
                row.get("left_value"),
                row.get("right_value"),
                row.get("delta_value"),
                row.get("updated_at_utc", now_utc()),
            ),
        )
        inserted += 1
    return inserted


def replace_scalar_points(
    conn: sqlite3.Connection,
    *,
    iteration_dir: str,
    run_slug: str,
    run_id: str,
    parsed_log: dict[str, Any] | None,
) -> int:
    conn.execute("DELETE FROM scalar_points WHERE iteration_dir = ? AND run_slug = ?", (iteration_dir, run_slug))
    if parsed_log is None:
        return 0
    inserted = 0
    for phase_name in ("train", "val"):
        for row in parsed_log.get(phase_name, []):
            if not isinstance(row, dict):
                continue
            step = int(row.get("step", -1))
            elapsed_min = float(row.get("elapsed_min", 0.0)) if isinstance(row.get("elapsed_min"), (int, float)) else None
            for metric, value in numeric_metrics(row, exclude={"step", "iterations", "elapsed_min"}).items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scalar_points (
                        iteration_dir, run_slug, run_id, phase, point_key, step, elapsed_min, metric, value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (iteration_dir, run_slug, run_id, phase_name, "", step, elapsed_min, metric, value),
                )
                inserted += 1
    finals = parsed_log.get("finals", {})
    if isinstance(finals, dict):
        for final_key, final_row in finals.items():
            if not isinstance(final_row, dict):
                continue
            for metric, value in numeric_metrics(final_row, exclude=set()).items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scalar_points (
                        iteration_dir, run_slug, run_id, phase, point_key, step, elapsed_min, metric, value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (iteration_dir, run_slug, run_id, "final", str(final_key), -1, None, metric, value),
                )
                inserted += 1
    return inserted


def build_runtime_payload(iteration_dir: Path, *, check_remote: bool, search_roots: list[Path]) -> dict[str, Any]:
    manifest_path = iteration_dir / "manifest.json"
    entries: list[RunEntry] = []
    embedded_logs: dict[str, Path] = {}
    sweep_rows: list[dict[str, Any]] = []
    observed_runs: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        entries = build_run_entries(iteration_dir, manifest)
        embedded_logs = extract_embedded_dispatch_logs(iteration_dir, entries)
        sweep_rows = summarize_sweep(iteration_dir, check_remote=check_remote)
        observed = load_observed_results(iteration_dir)
        observed_runs = observed.get("runs", {}) if isinstance(observed, dict) else {}
        if not isinstance(observed_runs, dict):
            observed_runs = {}
    queue_by_slug = {str(row.get("run_slug", "")): row for row in sweep_rows if row.get("run_slug")}

    runs: dict[str, Any] = {}
    for entry in entries:
        queue_row = queue_by_slug.get(entry.run_slug, {})
        observed_run = observed_runs.get(entry.run_slug, {})
        if not isinstance(observed_run, dict):
            observed_run = {}
        log_path = find_log_for_entry(entry, observed_run=observed_run, embedded_logs=embedded_logs, search_roots=search_roots)
        parsed_log = parse_log(log_path) if log_path is not None and log_path.exists() else None
        latest_train = compact_metrics_row(parsed_log["train"][-1]) if parsed_log and parsed_log.get("train") else None
        latest_val = compact_metrics_row(parsed_log["val"][-1]) if parsed_log and parsed_log.get("val") else None
        finals = parsed_log.get("finals", {}) if parsed_log else {}
        final_metrics = compact_metrics_row(first_final_metrics(finals) if isinstance(finals, dict) else None)
        artifact_path = find_artifact_for_entry(entry, observed_run=observed_run, search_roots=search_roots)
        job_id = str(queue_row.get("job_id", "") or observed_run.get("job_id", "") or "")
        remote_dir = f"~/jobs/{job_id}" if job_id else ""
        runs[entry.run_slug] = {
            "run_slug": entry.run_slug,
            "run_id": entry.run_id,
            "config_name": entry.config_name,
            "trainer_module": entry.trainer_module,
            "queue_status": str(queue_row.get("status", observed_run.get("status", "prepared")) or "prepared"),
            "observed_status": str(observed_run.get("status", "") or ""),
            "host": str(queue_row.get("host", observed_run.get("host", "")) or ""),
            "job_id": job_id,
            "remote_dir": remote_dir,
            "local_log_path": log_path.as_posix() if log_path is not None else "",
            "artifact_path": artifact_path.as_posix() if artifact_path is not None else "",
            "latest_train": latest_train,
            "latest_val": latest_val,
            "final_metrics": final_metrics,
            "embedded_log": embedded_logs.get(entry.run_slug).as_posix() if entry.run_slug in embedded_logs else "",
            "parsed_train_rows": len(parsed_log.get("train", [])) if parsed_log else 0,
            "parsed_val_rows": len(parsed_log.get("val", [])) if parsed_log else 0,
            "final_keys": sorted(finals.keys()) if isinstance(finals, dict) else [],
        }
    return {
        "updated_at_utc": now_utc(),
        "iteration_dir": iteration_dir.resolve().as_posix(),
        "manifest_path": manifest_path.resolve().as_posix() if manifest_path.exists() else "",
        "manifest_missing": not manifest_path.exists(),
        "run_count": len(entries),
        "search_roots": [root.resolve().as_posix() for root in search_roots],
        "runs": runs,
    }


def write_runtime_summary(path: Path, payload: dict[str, Any], db_path: Path, total_points: int) -> None:
    lines: list[str] = []
    lines.append("# Runtime Summary")
    lines.append("")
    lines.append(f"- iteration_dir: `{payload['iteration_dir']}`")
    lines.append(f"- run_count: `{payload['run_count']}`")
    lines.append(f"- db_path: `{db_path.resolve().as_posix()}`")
    lines.append(f"- scalar_point_count: `{total_points}`")
    lines.append(f"- analysis_metric_count: `{payload.get('analysis_metric_count', 0)}`")
    lines.append(f"- analysis_source_count: `{payload.get('analysis_source_count', 0)}`")
    lines.append("")
    lines.append("Runs:")
    runs = payload.get("runs", {})
    for run_slug, row in sorted(runs.items()):
        if not isinstance(row, dict):
            continue
        metric = ""
        latest_val = row.get("latest_val")
        final_metrics = row.get("final_metrics")
        if isinstance(final_metrics, dict) and "val_bpb" in final_metrics:
            metric = f" final_bpb={float(final_metrics['val_bpb']):.6f}"
        elif isinstance(latest_val, dict) and "val_bpb" in latest_val:
            metric = f" latest_val_bpb={float(latest_val['val_bpb']):.6f}"
        lines.append(
            f"- `{run_slug}`: queue=`{row.get('queue_status','')}` host=`{row.get('host','')}` "
            f"job=`{row.get('job_id','')}` train_rows=`{row.get('parsed_train_rows',0)}` "
            f"val_rows=`{row.get('parsed_val_rows',0)}` cluster_job=`{row.get('cluster_job_status','')}` "
            f"cluster_host=`{row.get('cluster_host_status','')}`{metric}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_manifest(path: Path, payload: dict[str, Any], db_path: Path, total_points: int) -> dict[str, Any]:
    manifest = {
        "updated_at_utc": now_utc(),
        "iteration_dir": payload["iteration_dir"],
        "db_path": db_path.resolve().as_posix(),
        "run_count": payload["run_count"],
        "scalar_point_count": total_points,
        "analysis_metric_count": payload.get("analysis_metric_count", 0),
        "analysis_source_count": payload.get("analysis_source_count", 0),
        "cluster_snapshot_path": payload.get("cluster_snapshot_path", ""),
        "runs": {
            run_slug: {
                "queue_status": row.get("queue_status"),
                "host": row.get("host"),
                "job_id": row.get("job_id"),
                "cluster_job_status": row.get("cluster_job_status"),
                "cluster_host_status": row.get("cluster_host_status"),
                "cluster_lock_owner": row.get("cluster_lock_owner"),
                "latest_val_bpb": (
                    row.get("final_metrics", {}).get("val_bpb")
                    if isinstance(row.get("final_metrics"), dict) and "val_bpb" in row.get("final_metrics", {})
                    else (row.get("latest_val", {}).get("val_bpb") if isinstance(row.get("latest_val"), dict) else None)
                ),
                "local_log_path": row.get("local_log_path"),
                "artifact_path": row.get("artifact_path"),
            }
            for run_slug, row in payload.get("runs", {}).items()
            if isinstance(row, dict)
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def update_index(
    *,
    iteration_dir: Path,
    db_path: Path,
    check_remote: bool,
    search_roots: list[Path],
    runtime_output: Path,
    manifest_output: Path,
    summary_output: Path,
    cluster_root: Path,
    redis_host: str,
    cluster_snapshot_output: Path,
    cluster_summary_output: Path,
    include_cluster_snapshot: bool,
) -> dict[str, Any]:
    payload = build_runtime_payload(iteration_dir, check_remote=check_remote, search_roots=search_roots)
    cluster_snapshot: dict[str, Any] | None = None
    if include_cluster_snapshot:
        cluster_snapshot = build_cluster_snapshot(cluster_root=cluster_root, redis_host=redis_host)
        write_cluster_snapshot(cluster_snapshot_output, cluster_summary_output, cluster_snapshot)
        payload["cluster_snapshot_path"] = cluster_snapshot_output.resolve().as_posix()
        payload["cluster_summary_path"] = cluster_summary_output.resolve().as_posix()
        jobs_by_id = {
            str(row.get("job_id", "") or ""): row
            for row in cluster_snapshot.get("jobs", [])
            if isinstance(row, dict) and row.get("job_id")
        }
        hosts_by_name = {
            str(row.get("host", "") or ""): row
            for row in cluster_snapshot.get("hosts", [])
            if isinstance(row, dict) and row.get("host")
        }
        for row in payload.get("runs", {}).values():
            if not isinstance(row, dict):
                continue
            job_row = jobs_by_id.get(str(row.get("job_id", "") or ""), {})
            host_row = hosts_by_name.get(str(row.get("host", "") or ""), {})
            row["cluster_job_status"] = str(job_row.get("status", "") or "")
            row["cluster_host_status"] = str(host_row.get("status", "") or "")
            row["cluster_lock_owner"] = str(host_row.get("lock_owner", "") or "")
            if job_row.get("remote_dir"):
                row["remote_dir"] = str(job_row.get("remote_dir"))
    runtime_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    total_points = 0
    try:
        create_tables(conn)
        for run_slug, row in payload.get("runs", {}).items():
            if not isinstance(row, dict):
                continue
            upsert_run(
                conn,
                {
                    "iteration_dir": payload["iteration_dir"],
                    "run_slug": run_slug,
                    "run_id": row["run_id"],
                    "config_name": row["config_name"],
                    "trainer_module": row.get("trainer_module", ""),
                    "queue_status": row.get("queue_status", ""),
                    "observed_status": row.get("observed_status", ""),
                    "host": row.get("host", ""),
                    "job_id": row.get("job_id", ""),
                    "remote_dir": row.get("remote_dir", ""),
                    "local_log_path": row.get("local_log_path", ""),
                    "artifact_path": row.get("artifact_path", ""),
                    "latest_train_step": row.get("latest_train", {}).get("step") if isinstance(row.get("latest_train"), dict) else None,
                    "latest_train_loss": row.get("latest_train", {}).get("train_loss") if isinstance(row.get("latest_train"), dict) else None,
                    "latest_val_step": row.get("latest_val", {}).get("step") if isinstance(row.get("latest_val"), dict) else None,
                    "latest_val_loss": row.get("latest_val", {}).get("val_loss") if isinstance(row.get("latest_val"), dict) else None,
                    "latest_val_bpb": (
                        row.get("final_metrics", {}).get("val_bpb")
                        if isinstance(row.get("final_metrics"), dict) and "val_bpb" in row.get("final_metrics", {})
                        else (row.get("latest_val", {}).get("val_bpb") if isinstance(row.get("latest_val"), dict) else None)
                    ),
                    "cluster_job_status": row.get("cluster_job_status", ""),
                    "cluster_host_status": row.get("cluster_host_status", ""),
                    "cluster_lock_owner": row.get("cluster_lock_owner", ""),
                    "updated_at_utc": payload["updated_at_utc"],
                },
            )
            parsed_log = parse_log(Path(row["local_log_path"])) if row.get("local_log_path") else None
            total_points += replace_scalar_points(
                conn,
                iteration_dir=payload["iteration_dir"],
                run_slug=run_slug,
                run_id=str(row["run_id"]),
                parsed_log=parsed_log,
            )
        analysis_rows = extract_analysis_rows(iteration_dir, updated_at_utc=payload["updated_at_utc"])
        payload["analysis_metric_count"] = replace_analysis_metrics(
            conn,
            iteration_dir=payload["iteration_dir"],
            rows=analysis_rows,
        )
        payload["analysis_source_count"] = len({row["source_path"] for row in analysis_rows})
        if cluster_snapshot is not None:
            updated_at = cluster_snapshot.get("generated_at_utc", payload["updated_at_utc"])
            for host_row in cluster_snapshot.get("hosts", []):
                if not isinstance(host_row, dict):
                    continue
                upsert_cluster_host(conn, {**host_row, "updated_at_utc": updated_at})
            for job_row in cluster_snapshot.get("jobs", []):
                if not isinstance(job_row, dict):
                    continue
                upsert_cluster_job(conn, {**job_row, "updated_at_utc": updated_at})
        conn.commit()
    finally:
        conn.close()

    manifest = write_runtime_manifest(manifest_output, payload, db_path, total_points)
    write_runtime_summary(summary_output, payload, db_path, total_points)
    return manifest


def default_order_for_metric(metric: str) -> str:
    lower = metric.lower()
    if any(token in lower for token in ("loss", "bpb", "error", "mse")):
        return "asc"
    return "desc"


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def search_metric_rows(
    *,
    db_path: Path,
    metric: str,
    phase: str,
    run_like: str,
    iteration_like: str,
    host: str,
    limit: int,
    order: str,
) -> list[dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        where = ["p.metric = ?"]
        params: list[Any] = [metric]
        if phase:
            where.append("p.phase = ?")
            params.append(phase)
        if run_like:
            where.append("lower(p.run_slug) LIKE ?")
            params.append(f"%{run_like.lower()}%")
        if iteration_like:
            where.append("lower(p.iteration_dir) LIKE ?")
            params.append(f"%{iteration_like.lower()}%")
        if host:
            where.append("r.host = ?")
            params.append(host)
        direction = order or default_order_for_metric(metric)
        sql = f"""
            WITH latest AS (
                SELECT iteration_dir, run_slug, phase, metric, MAX(step) AS max_step
                FROM scalar_points
                WHERE metric = ?
                {'AND phase = ?' if phase else ''}
                GROUP BY iteration_dir, run_slug, phase, metric
            )
            SELECT
                p.iteration_dir,
                p.run_slug,
                p.run_id,
                p.phase,
                p.point_key,
                p.step,
                p.elapsed_min,
                p.metric,
                p.value,
                r.host,
                r.queue_status,
                r.latest_val_bpb,
                r.local_log_path
            FROM scalar_points p
            JOIN latest l
              ON p.iteration_dir = l.iteration_dir
             AND p.run_slug = l.run_slug
             AND p.phase = l.phase
             AND p.metric = l.metric
             AND p.step = l.max_step
            JOIN runs r
              ON r.iteration_dir = p.iteration_dir
             AND r.run_slug = p.run_slug
            WHERE {' AND '.join(where)}
            ORDER BY p.value {direction.upper()}, p.iteration_dir ASC, p.run_slug ASC
            LIMIT ?
        """
        query_params: list[Any] = [metric]
        if phase:
            query_params.append(phase)
        query_params.extend(params)
        query_params.append(limit)
        rows = conn.execute(sql, query_params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_runs_rows(
    *,
    db_path: Path,
    status: str,
    run_like: str,
    iteration_like: str,
    host: str,
    limit: int,
) -> list[dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        where = ["1=1"]
        params: list[Any] = []
        if status:
            where.append("queue_status = ?")
            params.append(status)
        if run_like:
            where.append("lower(run_slug) LIKE ?")
            params.append(f"%{run_like.lower()}%")
        if iteration_like:
            where.append("lower(iteration_dir) LIKE ?")
            params.append(f"%{iteration_like.lower()}%")
        if host:
            where.append("host = ?")
            params.append(host)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT
                iteration_dir, run_slug, run_id, queue_status, observed_status, host, job_id,
                latest_train_step, latest_train_loss, latest_val_step, latest_val_loss, latest_val_bpb,
                cluster_job_status, cluster_host_status, cluster_lock_owner,
                local_log_path, artifact_path, updated_at_utc
            FROM runs
            WHERE {' AND '.join(where)}
            ORDER BY updated_at_utc DESC, iteration_dir ASC, run_slug ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_cluster_hosts_rows(
    *,
    db_path: Path,
    status: str,
    host_like: str,
    limit: int,
) -> list[dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        where = ["1=1"]
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if host_like:
            where.append("lower(host) LIKE ?")
            params.append(f"%{host_like.lower()}%")
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT *
            FROM cluster_hosts
            WHERE {' AND '.join(where)}
            ORDER BY status ASC, host ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_cluster_jobs_rows(
    *,
    db_path: Path,
    status: str,
    host: str,
    script_like: str,
    limit: int,
) -> list[dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        where = ["1=1"]
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if host:
            where.append("host = ?")
            params.append(host)
        if script_like:
            where.append("lower(script) LIKE ?")
            params.append(f"%{script_like.lower()}%")
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT *
            FROM cluster_jobs
            WHERE {' AND '.join(where)}
            ORDER BY started DESC, job_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def search_analysis_metric_rows(
    *,
    db_path: Path,
    metric: str,
    iteration_like: str,
    subject_like: str,
    source_like: str,
    metric_like: str,
    limit: int,
    order: str,
    delta_only: bool,
) -> list[dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        where = ["metric = ?"]
        params: list[Any] = [metric]
        if iteration_like:
            where.append("lower(iteration_dir) LIKE ?")
            params.append(f"%{iteration_like.lower()}%")
        if subject_like:
            where.append("lower(subject_slug) LIKE ?")
            params.append(f"%{subject_like.lower()}%")
        if source_like:
            where.append("lower(source_path) LIKE ?")
            params.append(f"%{source_like.lower()}%")
        if metric_like:
            where.append("lower(metric) LIKE ?")
            params.append(f"%{metric_like.lower()}%")
        order_field = "delta_value" if delta_only else "value"
        direction = order or default_order_for_metric(metric)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT *
            FROM analysis_metrics
            WHERE {' AND '.join(where)}
            ORDER BY {order_field} {direction.upper()}, iteration_dir ASC, source_path ASC, subject_slug ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def cmd_update(args: argparse.Namespace) -> None:
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    runtime_output = Path(args.runtime_output).expanduser().resolve() if args.runtime_output else (iteration_dir / "runtime_state.json")
    manifest_output = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else (iteration_dir / "runtime_metrics_manifest.json")
    summary_output = Path(args.summary_output).expanduser().resolve() if args.summary_output else (iteration_dir / "runtime_summary.md")
    cluster_snapshot_output = (
        Path(args.cluster_snapshot_output).expanduser().resolve()
        if args.cluster_snapshot_output
        else (iteration_dir / "cluster_queue_snapshot.json")
    )
    cluster_summary_output = (
        Path(args.cluster_summary_output).expanduser().resolve()
        if args.cluster_summary_output
        else (iteration_dir / "cluster_queue_summary.md")
    )
    search_roots = collect_search_roots(iteration_dir, args.search_root)
    manifest = update_index(
        iteration_dir=iteration_dir,
        db_path=db_path,
        check_remote=bool(args.check_remote),
        search_roots=search_roots,
        runtime_output=runtime_output,
        manifest_output=manifest_output,
        summary_output=summary_output,
        cluster_root=Path(args.cluster_root).expanduser().resolve(),
        redis_host=str(args.redis_host),
        cluster_snapshot_output=cluster_snapshot_output,
        cluster_summary_output=cluster_summary_output,
        include_cluster_snapshot=not bool(args.skip_cluster_snapshot),
    )
    if args.stdout_summary:
        print(json.dumps(manifest, indent=2, sort_keys=True))


def cmd_search_metric(args: argparse.Namespace) -> None:
    rows = search_metric_rows(
        db_path=Path(args.db_path).expanduser().resolve(),
        metric=args.metric,
        phase=args.phase,
        run_like=args.run_like,
        iteration_like=args.iteration_like,
        host=args.host,
        limit=int(args.limit),
        order=args.order,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def cmd_list_runs(args: argparse.Namespace) -> None:
    rows = list_runs_rows(
        db_path=Path(args.db_path).expanduser().resolve(),
        status=args.status,
        run_like=args.run_like,
        iteration_like=args.iteration_like,
        host=args.host,
        limit=int(args.limit),
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def cmd_list_cluster_hosts(args: argparse.Namespace) -> None:
    rows = list_cluster_hosts_rows(
        db_path=Path(args.db_path).expanduser().resolve(),
        status=args.status,
        host_like=args.host_like,
        limit=int(args.limit),
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def cmd_list_cluster_jobs(args: argparse.Namespace) -> None:
    rows = list_cluster_jobs_rows(
        db_path=Path(args.db_path).expanduser().resolve(),
        status=args.status,
        host=args.host,
        script_like=args.script_like,
        limit=int(args.limit),
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def cmd_search_analysis_metric(args: argparse.Namespace) -> None:
    rows = search_analysis_metric_rows(
        db_path=Path(args.db_path).expanduser().resolve(),
        metric=args.metric,
        iteration_like=args.iteration_like,
        subject_like=args.subject_like,
        source_like=args.source_like,
        metric_like=args.metric_like,
        limit=int(args.limit),
        order=args.order,
        delta_only=bool(args.delta_only),
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
