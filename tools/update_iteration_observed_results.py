#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re

try:
    from check_mlx_sweep_status import summarize_sweep
    from plot_training_curves import parse_log
    from run_iteration_saved_diagnostics import RunEntry, build_run_entries, find_control_entry, load_json, name_tokens
except ModuleNotFoundError:
    from tools.check_mlx_sweep_status import summarize_sweep
    from tools.plot_training_curves import parse_log
    from tools.run_iteration_saved_diagnostics import (
        RunEntry,
        build_run_entries,
        find_control_entry,
        load_json,
        name_tokens,
    )


LOG_SUFFIX = ".txt"
MODEL_SUFFIXES = ("_mlx_model.npz", "_mlx_model.int8.ptz", "_int8zlib.pklz")
EMBEDDED_LOG_RE = re.compile(r"^logs/(?P<name>[^/\s]+\.txt)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build machine-readable observed_results.json for a generated MLX sweep from local logs/artifacts."
    )
    p.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json and configs/")
    p.add_argument(
        "--output",
        default="",
        help="Output path for observed_results.json. Defaults to <iteration_dir>/observed_results.json",
    )
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional directories to search recursively for recovered logs/artifacts.",
    )
    p.add_argument(
        "--check-remote",
        action="store_true",
        help="Ask the status helper to SSH claimed hosts and verify whether runs are still alive.",
    )
    p.add_argument(
        "--stdout-summary",
        action="store_true",
        help="Print a short summary JSON after writing observed_results.json.",
    )
    return p.parse_args()


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def find_existing_file(raw_path: str, *, base_dirs: list[Path]) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved if resolved.exists() else None
    for base_dir in base_dirs:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def discover_named_file(
    entry: RunEntry,
    *,
    search_roots: list[Path],
    suffix: str,
) -> Path | None:
    exact_patterns = [
        f"{entry.run_id}{suffix}",
        f"{entry.run_slug}{suffix}",
        f"{Path(entry.config_name).stem}{suffix}",
    ]
    exact_candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in exact_patterns:
            exact_candidates.extend(sorted(root.rglob(pattern)))
    if exact_candidates:
        return sorted({path.resolve() for path in exact_candidates}, key=lambda p: (0 if "artifacts" in p.parts else 1, len(p.parts), p.as_posix()))[0]

    slug_tokens = name_tokens(entry.run_slug) | name_tokens(Path(entry.config_name).stem) | name_tokens(entry.run_id)
    ranked: list[tuple[tuple[int, int, int, str], Path]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            stem_tokens = name_tokens(path.stem)
            overlap = len(slug_tokens & stem_tokens)
            if overlap <= 0:
                continue
            rank = (
                -overlap,
                0 if "artifacts" in path.parts else 1,
                len(path.parts),
                path.as_posix(),
            )
            ranked.append((rank, path.resolve()))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def collect_search_roots(iteration_dir: Path, extra_roots: list[str]) -> list[Path]:
    roots = [
        iteration_dir / "artifacts",
        iteration_dir / "logs",
        iteration_dir,
    ]
    for raw in extra_roots:
        roots.append(Path(raw).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def dispatch_path_for_iteration(iteration_dir: Path) -> Path | None:
    dispatch_path = iteration_dir / "dispatch.out"
    if dispatch_path.exists():
        return dispatch_path
    launch_path = iteration_dir / "launch.nohup.log"
    if launch_path.exists():
        return launch_path
    return None


def match_entry_for_log_name(log_name: str, entries: list[RunEntry]) -> RunEntry | None:
    for entry in entries:
        exact_names = {
            f"{entry.run_id}{LOG_SUFFIX}",
            f"{entry.run_slug}{LOG_SUFFIX}",
            f"{Path(entry.config_name).stem}{LOG_SUFFIX}",
        }
        if log_name in exact_names:
            return entry
    log_tokens = name_tokens(Path(log_name).stem)
    ranked: list[tuple[tuple[int, int, str], RunEntry]] = []
    for entry in entries:
        overlap = len(log_tokens & (name_tokens(entry.run_id) | name_tokens(entry.run_slug) | name_tokens(Path(entry.config_name).stem)))
        if overlap <= 0:
            continue
        rank = (-overlap, len(entry.run_slug), entry.run_slug)
        ranked.append((rank, entry))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def extract_embedded_dispatch_logs(iteration_dir: Path, entries: list[RunEntry]) -> dict[str, Path]:
    dispatch_path = dispatch_path_for_iteration(iteration_dir)
    if dispatch_path is None:
        return {}

    extracted_dir = iteration_dir / "observed_logs"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    entry_by_run_id = {entry.run_id: entry for entry in entries}
    buffers: dict[str, tuple[str, list[str]]] = {}
    current_entry: RunEntry | None = None
    current_name: str | None = None
    current_lines: list[str] = []
    pending_name: str | None = None

    def flush_current() -> None:
        nonlocal current_entry, current_name, current_lines
        if current_entry is None or current_name is None or not current_lines:
            current_entry = None
            current_name = None
            current_lines = []
            return
        buffers[current_entry.run_slug] = (current_name, list(current_lines))
        current_entry = None
        current_name = None
        current_lines = []

    raw_text = dispatch_path.read_text(encoding="utf-8", errors="replace")
    # Some remote stream joins drop a newline between the end of one log line and the
    # next embedded section marker. Normalize the obvious boundaries before parsing.
    raw_text = re.sub(r"(?<!saved_model:)(logs/[^/\s]+\.txt)", r"\n\1", raw_text)
    raw_text = re.sub(r"(?<!\n)(run_id:[^\s]+)", r"\n\1", raw_text)

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        match = EMBEDDED_LOG_RE.match(stripped)
        if match:
            flush_current()
            pending_name = match.group("name")
            continue
        if stripped.startswith("run_id:"):
            run_id = stripped.split(":", 1)[1]
            entry = entry_by_run_id.get(run_id)
            if entry is not None:
                flush_current()
                current_entry = entry
                current_name = pending_name or f"{run_id}{LOG_SUFFIX}"
                current_lines = [raw_line]
                pending_name = None
                continue
        if current_entry is None:
            continue
        current_lines.append(raw_line)
    flush_current()

    extracted: dict[str, Path] = {}
    for run_slug, (log_name, lines) in buffers.items():
        entry = next((item for item in entries if item.run_slug == run_slug), None)
        if entry is None or not lines:
            continue
        # Keep only sections that look like trainer logs, not arbitrary command noise.
        if not any(line.startswith("run_id:") or line.startswith("step:") or line.startswith("final_") for line in lines):
            continue
        out_path = extracted_dir / log_name
        out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        extracted[run_slug] = out_path
    return extracted


def finals_from_parsed_log(parsed: dict[str, object]) -> dict[str, dict[str, object]]:
    finals = parsed.get("finals", {})
    return finals if isinstance(finals, dict) else {}


def latest_metric_row(rows_obj: object) -> dict[str, object] | None:
    if not isinstance(rows_obj, list) or not rows_obj:
        return None
    row = rows_obj[-1]
    return row if isinstance(row, dict) else None


def compact_train_row(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    out: dict[str, object] = {}
    for key in ("step", "train_loss", "step_avg", "step_avg_ms", "ce", "hardmax_book_cos", "elapsed_min"):
        if key in row:
            out[key] = row[key]
    return out or None


def compact_val_row(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    out: dict[str, object] = {}
    for key in ("step", "val_loss", "val_bpb", "elapsed_min"):
        if key in row:
            out[key] = row[key]
    return out or None


def normalize_status(raw_status: str, *, has_final: bool, has_partial_final: bool, has_progress: bool) -> str:
    if has_final:
        return "observed_final"
    if has_partial_final:
        return "observed_partial_final"
    if has_progress:
        return "observed_progress"
    mapping = {
        "running": "running_or_not_observed",
        "claimed": "running_or_not_observed",
        "dispatching": "pending_or_not_observed",
        "prepared": "pending_or_not_observed",
        "released": "stopped_or_not_observed",
        "stopped": "stopped_or_not_observed",
        "dispatch_failed": "dispatch_failed",
        "summarized": "summarized_elsewhere",
    }
    return mapping.get(str(raw_status), str(raw_status or "unknown"))


def observation_score(payload: dict[str, object]) -> int:
    if isinstance(payload.get("final_int8_zlib_roundtrip_exact"), dict):
        return 3
    if isinstance(payload.get("final_raw_export_ready_exact"), dict):
        return 2
    if isinstance(payload.get("latest_observed_train"), dict) or isinstance(payload.get("latest_observed_val"), dict):
        return 1
    return 0


def merge_run_payload(existing: dict[str, object] | None, new: dict[str, object]) -> dict[str, object]:
    if not isinstance(existing, dict):
        return new
    merged = dict(existing)
    merged.update(new)
    if "artifacts" in existing and isinstance(existing["artifacts"], dict):
        artifacts = dict(existing["artifacts"])
        if isinstance(new.get("artifacts"), dict):
            artifacts.update(new["artifacts"])
        merged["artifacts"] = artifacts
    existing_log = existing.get("log_path")
    new_log = new.get("log_path")
    observation_keys = (
        "final_int8_zlib_roundtrip_exact",
        "final_raw_export_ready_exact",
        "latest_observed_train",
        "latest_observed_val",
    )
    if new_log or new.get("artifacts"):
        for key in observation_keys:
            if key not in new:
                merged.pop(key, None)
    if observation_score(existing) > observation_score(new) and not new_log and not new.get("artifacts"):
        for key in observation_keys:
            if key in existing and key not in new:
                merged[key] = existing[key]
    has_final = isinstance(merged.get("final_int8_zlib_roundtrip_exact"), dict)
    has_partial_final = isinstance(merged.get("final_raw_export_ready_exact"), dict)
    has_progress = isinstance(merged.get("latest_observed_train"), dict) or isinstance(merged.get("latest_observed_val"), dict)
    merged["status"] = normalize_status(
        str(merged.get("status", "")),
        has_final=has_final,
        has_partial_final=has_partial_final,
        has_progress=has_progress,
    )
    return merged


def build_observed_results(iteration_dir: Path, *, check_remote: bool, search_roots: list[Path]) -> dict[str, object]:
    manifest = load_json(iteration_dir / "manifest.json")
    entries = build_run_entries(iteration_dir, manifest)
    embedded_logs = extract_embedded_dispatch_logs(iteration_dir, entries)
    if embedded_logs:
        embedded_root = iteration_dir / "observed_logs"
        if embedded_root not in search_roots:
            search_roots = [embedded_root, *search_roots]
    status_rows = summarize_sweep(iteration_dir, check_remote=check_remote)
    status_by_config = {str(row.get("config")): row for row in status_rows}
    existing_path = iteration_dir / "observed_results.json"
    existing_runs: dict[str, object] = {}
    existing_notes: list[str] = []
    if existing_path.exists():
        try:
            existing_payload = load_json(existing_path)
            raw_runs = existing_payload.get("runs", {})
            if isinstance(raw_runs, dict):
                existing_runs = raw_runs
            raw_notes = existing_payload.get("notes", [])
            if isinstance(raw_notes, list):
                existing_notes = [str(note) for note in raw_notes]
        except (OSError, json.JSONDecodeError):
            existing_runs = {}
            existing_notes = []
    runs_payload: dict[str, dict[str, object]] = {}
    best_final: tuple[float, str] | None = None

    for entry in entries:
        row = status_by_config.get(entry.config_name, {})
        log_path = discover_named_file(entry, search_roots=search_roots, suffix=LOG_SUFFIX)
        parsed = parse_log(log_path) if log_path is not None else None
        finals = finals_from_parsed_log(parsed) if parsed is not None else {}
        final_exact = finals.get("final_int8_zlib_roundtrip_exact")
        final_raw = finals.get("final_raw_export_ready_exact")
        latest_train = compact_train_row(latest_metric_row(parsed.get("train")) if parsed is not None else None)
        latest_val = compact_val_row(latest_metric_row(parsed.get("val")) if parsed is not None else None)
        artifact_paths: dict[str, str] = {}
        for suffix in MODEL_SUFFIXES:
            found = discover_named_file(entry, search_roots=search_roots, suffix=suffix)
            if found is not None:
                artifact_paths[suffix] = str(found)

        status = normalize_status(
            str(row.get("status", "")),
            has_final=isinstance(final_exact, dict),
            has_partial_final=isinstance(final_raw, dict),
            has_progress=latest_train is not None or latest_val is not None,
        )
        payload: dict[str, object] = {
            "status": status,
            "config": entry.config_name,
            "run_id": entry.run_id,
        }
        if log_path is not None:
            payload["log_path"] = str(log_path)
        if isinstance(final_exact, dict):
            payload["final_int8_zlib_roundtrip_exact"] = final_exact
            val_bpb = final_exact.get("val_bpb")
            if isinstance(val_bpb, (int, float)):
                candidate = (float(val_bpb), entry.run_slug)
                if best_final is None or candidate[0] < best_final[0]:
                    best_final = candidate
        if isinstance(final_raw, dict):
            payload["final_raw_export_ready_exact"] = final_raw
        if latest_train is not None:
            payload["latest_observed_train"] = latest_train
        if latest_val is not None:
            payload["latest_observed_val"] = latest_val
        if artifact_paths:
            payload["artifacts"] = artifact_paths
        if row.get("host"):
            payload["host"] = row.get("host")
        if row.get("job_id"):
            payload["job_id"] = row.get("job_id")
        runs_payload[entry.run_slug] = merge_run_payload(existing_runs.get(entry.run_slug), payload)

    result: dict[str, object] = {
        "source": "local_logs_and_artifacts",
        "updated_at_local": now_local_iso(),
        "iteration_dir": str(iteration_dir),
        "runs": runs_payload,
    }
    for run_slug, payload in runs_payload.items():
        if not isinstance(payload, dict):
            continue
        final_exact = payload.get("final_int8_zlib_roundtrip_exact")
        if not isinstance(final_exact, dict):
            continue
        val_bpb = final_exact.get("val_bpb")
        if not isinstance(val_bpb, (int, float)):
            continue
        candidate = (float(val_bpb), run_slug)
        if best_final is None or candidate[0] < best_final[0]:
            best_final = candidate
    if best_final is not None:
        result["best_observed_final"] = {
            "run_slug": best_final[1],
            "val_bpb": best_final[0],
        }
    notes = [
        "Generated from local dispatch status, recovered per-run logs, and local artifacts.",
        "Runs without per-run logs may show status only until recovery lands.",
    ]
    for note in existing_notes:
        if note not in notes:
            notes.append(note)
    result["notes"] = notes
    return result


def write_observed_summary(iteration_dir: Path, payload: dict[str, object]) -> Path:
    out_path = iteration_dir / "observed_results_summary.md"
    runs = payload.get("runs", {})
    lines: list[str] = ["# Observed Sweep Results", ""]
    if not isinstance(runs, dict):
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    run_rows = [(slug, value) for slug, value in runs.items() if isinstance(value, dict)]
    control_slug = ""
    for slug, _value in run_rows:
        if "control" in slug:
            control_slug = slug
            break
    if not control_slug and run_rows:
        control_slug = str(run_rows[0][0])
    if control_slug:
        lines.append(f"Control: `{control_slug}`")
        lines.append("")

    lines.append("Runs:")
    for slug, value in sorted(run_rows, key=lambda item: item[0]):
        status = str(value.get("status", "unknown"))
        final_exact = value.get("final_int8_zlib_roundtrip_exact")
        raw_exact = value.get("final_raw_export_ready_exact")
        if isinstance(final_exact, dict) and "val_bpb" in final_exact:
            metric = f" final_int8_bpb={float(final_exact['val_bpb']):.6f}"
        elif isinstance(raw_exact, dict) and "val_bpb" in raw_exact:
            metric = f" final_raw_bpb={float(raw_exact['val_bpb']):.6f}"
        else:
            latest_val = value.get("latest_observed_val")
            metric = ""
            if isinstance(latest_val, dict) and "val_bpb" in latest_val:
                metric = f" latest_val_bpb={float(latest_val['val_bpb']):.6f}"
        lines.append(f"- `{slug}`: status=`{status}`{metric}")

    control_payload = runs.get(control_slug)
    control_bpb = None
    if isinstance(control_payload, dict):
        control_final = control_payload.get("final_int8_zlib_roundtrip_exact")
        if isinstance(control_final, dict) and "val_bpb" in control_final:
            control_bpb = float(control_final["val_bpb"])

    if control_bpb is not None:
        lines.append("")
        lines.append("Compared to control:")
        for slug, value in sorted(run_rows, key=lambda item: item[0]):
            if slug == control_slug:
                continue
            final_exact = value.get("final_int8_zlib_roundtrip_exact")
            if not isinstance(final_exact, dict) or "val_bpb" not in final_exact:
                continue
            bpb = float(final_exact["val_bpb"])
            lines.append(f"- `{slug}`: `{control_bpb:.6f} -> {bpb:.6f}` (delta `{bpb - control_bpb:+.6f}`)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else (iteration_dir / "observed_results.json")
    payload = build_observed_results(
        iteration_dir,
        check_remote=bool(args.check_remote),
        search_roots=collect_search_roots(iteration_dir, list(args.search_root)),
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_observed_summary(iteration_dir, payload)
    if args.stdout_summary:
        runs = payload.get("runs", {})
        observed_final = sum(
            1
            for value in (runs.values() if isinstance(runs, dict) else [])
            if isinstance(value, dict) and value.get("status") == "observed_final"
        )
        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "observed_final_runs": observed_final,
                    "best_observed_final": payload.get("best_observed_final"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
