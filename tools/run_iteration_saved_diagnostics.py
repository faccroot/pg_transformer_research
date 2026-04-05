#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZE_TOOL = ROOT / "tools" / "analyze_residual_autocorrelation.py"
COMPARE_TOOL = ROOT / "tools" / "compare_residual_autocorrelation.py"


@dataclass(frozen=True)
class RunEntry:
    config_name: str
    config_path: Path
    run_id: str
    run_slug: str
    notes: str
    trainer_module: str
    tokenizer_override: str
    data_override: str


ARTIFACT_SUFFIXES = ("_mlx_model.npz", "_int8zlib.pklz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run saved-artifact prosody/residual diagnostics for a generated MLX sweep iteration."
    )
    p.add_argument("iteration_dir", help="Generated sweep directory containing manifest.json and configs/")
    p.add_argument(
        "--output-dir",
        default="",
        help="Directory for analyzer JSONs and results_summary.md. Defaults to <iteration_dir>/saved_diagnostics",
    )
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional directory to scan recursively for artifacts. Can be passed multiple times.",
    )
    p.add_argument(
        "--control-slug",
        default="",
        help="Optional run_slug to treat as the control. Defaults to the first run whose slug contains `control`.",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable to use for the analyzer subprocesses.")
    p.add_argument("--skip-existing", action="store_true", help="Skip analyzer outputs that already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print planned analyzer commands without executing them.")
    p.add_argument("--val-max-seqs", type=int, default=1024)
    p.add_argument("--eval-seq-len", type=int, default=1024)
    p.add_argument("--eval-stride", type=int, default=0)
    p.add_argument("--eval-batch-seqs", type=int, default=None)
    p.add_argument("--cache-variant", default="sp1024")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--analysis-max-batches", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=64)
    p.add_argument("--probe-layers", default="-1")
    p.add_argument("--probe-max-samples", type=int, default=32768)
    p.add_argument("--probe-ridge", type=float, default=1.0)
    p.add_argument("--probe-train-frac", type=float, default=0.7)
    p.add_argument("--transition-window", type=int, default=32)
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--top-k-transitions", type=int, default=10)
    p.add_argument("--preview-radius", type=int, default=24)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict[str, object], path: tuple[str, ...]) -> object | None:
    cur: object = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


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


def resolve_tokenizer_override(config_path: Path, manifest: dict, env: dict) -> str:
    basename = Path(str(env.get("TOKENIZER_PATH", ""))).name
    raw = str(env.get("TOKENIZER_PATH", "")).strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved.as_posix() if resolved.exists() else ""
    explicit = find_existing_file(raw, base_dirs=[config_path.parent.parent])
    if explicit is not None:
        return explicit.as_posix()
    for item in manifest.get("support_files", []):
        support_path = Path(str(item)).expanduser().resolve()
        if support_path.exists() and support_path.name == basename:
            return support_path.as_posix()
    explicit = find_existing_file(raw, base_dirs=[ROOT, ROOT / "data" / "tokenizers"])
    if explicit is not None:
        return explicit.as_posix()
    return ""


def resolve_data_override(config_path: Path, env: dict) -> str:
    base_dirs = [config_path.parent.parent, ROOT]
    explicit = find_existing_file(str(env.get("DATA_PATH", "")), base_dirs=base_dirs)
    return "" if explicit is None else explicit.as_posix()


def infer_trainer_module(metadata: dict) -> str:
    raw = str(metadata.get("trainer_script_source") or metadata.get("trainer_script") or "").strip()
    if not raw:
        return ""
    stem = Path(raw).stem
    return "" if stem == "train_gpt_mlx" else stem


def build_run_entries(iteration_dir: Path, manifest: dict) -> list[RunEntry]:
    entries: list[RunEntry] = []
    for run in manifest.get("runs", []):
        config_path = (iteration_dir / str(run["config_path"])).resolve()
        config = load_json(config_path)
        env = config.get("env", {})
        metadata = config.get("metadata", {})
        if not isinstance(env, dict) or not isinstance(metadata, dict):
            continue
        entries.append(
            RunEntry(
                config_name=config_path.name,
                config_path=config_path,
                run_id=str(run.get("run_id") or env.get("RUN_ID") or config_path.stem),
                run_slug=str(run.get("run_slug") or metadata.get("run_slug") or config_path.stem),
                notes=str(run.get("notes") or metadata.get("notes") or ""),
                trainer_module=infer_trainer_module(metadata),
                tokenizer_override=resolve_tokenizer_override(config_path, manifest, env),
                data_override=resolve_data_override(config_path, env),
            )
        )
    return entries


def find_control_entry(entries: list[RunEntry], control_slug: str) -> RunEntry:
    if control_slug:
        for entry in entries:
            if entry.run_slug == control_slug:
                return entry
        raise ValueError(f"control slug {control_slug!r} not found in iteration manifest")
    for entry in entries:
        if "control" in entry.run_slug:
            return entry
    if not entries:
        raise ValueError("iteration manifest has no runs")
    return entries[0]


def name_tokens(text: str) -> set[str]:
    return {piece for piece in re.split(r"[^a-z0-9]+", text.lower()) if piece}


def scan_analyzable_artifacts(search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for suffix in ARTIFACT_SUFFIXES:
            candidates.extend(sorted(root.rglob(f"*{suffix}")))
    return sorted({path.resolve() for path in candidates})


def discover_artifact_for_run(run_id: str, run_slug: str, config_name: str, search_roots: list[Path]) -> Path | None:
    patterns = [f"{run_id}{suffix}" for suffix in ARTIFACT_SUFFIXES]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(root.rglob(pattern)))
    if not candidates:
        slug_tokens = name_tokens(run_slug) | name_tokens(Path(config_name).stem)
        ranked: list[tuple[tuple[int, int, int, str], Path]] = []
        for path in scan_analyzable_artifacts(search_roots):
            stem_tokens = name_tokens(path.stem)
            overlap = len(slug_tokens & stem_tokens)
            if overlap <= 0:
                continue
            rank = (
                -overlap,
                0 if "artifacts" in set(path.parts) else 1,
                len(path.parts),
                path.as_posix(),
            )
            ranked.append((rank, path))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]
    def rank(path: Path) -> tuple[int, int, str]:
        parts = set(path.parts)
        return (
            0 if "artifacts" in parts else 1,
            len(path.parts),
            path.as_posix(),
        )
    return sorted({path.resolve() for path in candidates}, key=rank)[0]


def analyzer_command(
    args: argparse.Namespace,
    *,
    entry: RunEntry,
    artifact_path: Path,
    result_json: Path,
) -> list[str]:
    cmd = [
        args.python,
        str(ANALYZE_TOOL),
        "--artifact",
        str(artifact_path),
        "--config-json",
        str(entry.config_path),
        "--label",
        entry.run_slug,
        "--result-json",
        str(result_json),
        "--val-max-seqs",
        str(int(args.val_max_seqs)),
        "--eval-seq-len",
        str(int(args.eval_seq_len)),
        "--cache-variant",
        str(args.cache_variant),
        "--train-shards",
        str(int(args.train_shards)),
        "--analysis-max-batches",
        str(int(args.analysis_max_batches)),
        "--max-lag",
        str(int(args.max_lag)),
        "--probe-layers",
        str(args.probe_layers),
        "--probe-max-samples",
        str(int(args.probe_max_samples)),
        "--probe-ridge",
        str(float(args.probe_ridge)),
        "--probe-train-frac",
        str(float(args.probe_train_frac)),
        "--transition-window",
        str(int(args.transition_window)),
        "--top-k-tokens",
        str(int(args.top_k_tokens)),
        "--top-k-transitions",
        str(int(args.top_k_transitions)),
        "--preview-radius",
        str(int(args.preview_radius)),
    ]
    if args.eval_stride:
        cmd.extend(["--eval-stride", str(int(args.eval_stride))])
    if args.eval_batch_seqs is not None:
        cmd.extend(["--eval-batch-seqs", str(int(args.eval_batch_seqs))])
    if entry.trainer_module:
        cmd.extend(["--trainer-module", entry.trainer_module])
    if entry.tokenizer_override:
        cmd.extend(["--tokenizer-path", entry.tokenizer_override])
    if entry.data_override:
        cmd.extend(["--data-path", entry.data_override])
    return cmd


def compare_command(
    args: argparse.Namespace,
    *,
    left_json: Path,
    right_json: Path,
    left_label: str,
    right_label: str,
    result_json: Path,
) -> list[str]:
    return [
        args.python,
        str(COMPARE_TOOL),
        "--left",
        str(left_json),
        "--right",
        str(right_json),
        "--left-label",
        left_label,
        "--right-label",
        right_label,
        "--result-json",
        str(result_json),
    ]


def run_subprocess(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def format_float(value: object | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def safe_float(payload: dict[str, object], path: tuple[str, ...]) -> float | None:
    value = nested_get(payload, path)
    return None if value is None else float(value)


def summarize_pair(control: dict[str, object], other: dict[str, object]) -> list[tuple[str, float | None, float | None]]:
    metrics = [
        ("mean NLL", ("mean_nll",)),
        ("content bits", ("token_class_loss", "content", "mean_bits")),
        ("whitespace bits", ("token_class_loss", "whitespace", "mean_bits")),
        ("punctuation bits", ("token_class_loss", "punctuation", "mean_bits")),
        ("inside-quote bits", ("quote_conditioned", "inside_quote", "mean_bits")),
        ("after-sentence bits", ("boundary_conditioned", "after_boundary_ge", "after_sentence", "mean_bits")),
        ("expected residual ACF mean", ("residual_modes", "expected_embedding", "acf_summary", "all", "mean")),
        ("expected residual within-regime", ("residual_modes", "expected_embedding", "acf_summary", "within_regime", "mean")),
        ("expected residual cross-regime", ("residual_modes", "expected_embedding", "acf_summary", "cross_regime", "mean")),
        ("argmax residual ACF mean", ("residual_modes", "argmax_embedding", "acf_summary", "all", "mean")),
        ("argmax residual within-regime", ("residual_modes", "argmax_embedding", "acf_summary", "within_regime", "mean")),
        ("argmax residual cross-regime", ("residual_modes", "argmax_embedding", "acf_summary", "cross_regime", "mean")),
        ("quote probe acc", ("prosody_probes", "-1", "inside_quote", "accuracy")),
        ("sentence-distance probe acc", ("prosody_probes", "-1", "sentence_distance_bucket", "accuracy")),
        ("noncontent probe acc", ("prosody_probes", "-1", "is_noncontent", "accuracy")),
    ]
    rows: list[tuple[str, float | None, float | None]] = []
    for label, path in metrics:
        rows.append((label, safe_float(control, path), safe_float(other, path)))
    return rows


def write_summary(
    out_path: Path,
    *,
    control_entry: RunEntry,
    result_paths: dict[str, Path],
    comparison_paths: dict[str, Path],
    artifact_paths: dict[str, Path],
    skipped: list[RunEntry],
) -> None:
    control_payload = load_json(result_paths[control_entry.run_slug])
    lines: list[str] = []
    lines.append("# Saved-Artifact Prosody/Residual Diagnostics")
    lines.append("")
    lines.append(f"Control: `{control_entry.run_slug}`")
    lines.append("")
    lines.append("Analyzed runs:")
    for run_slug, result_path in sorted(result_paths.items()):
        lines.append(f"- `{run_slug}`: [{result_path.name}]({result_path.as_posix()})")
    if skipped:
        lines.append("")
        lines.append("Missing/incomplete runs:")
        for entry in skipped:
            lines.append(f"- `{entry.run_slug}`: no local saved artifact found for run_id `{entry.run_id}`")
    for run_slug, result_path in sorted(result_paths.items()):
        if run_slug == control_entry.run_slug:
            continue
        payload = load_json(result_path)
        comparison_path = comparison_paths.get(run_slug)
        lines.append("")
        lines.append(f"## {control_entry.run_slug} vs {run_slug}")
        lines.append("")
        lines.append("Artifacts:")
        lines.append(f"- control: [{artifact_paths[control_entry.run_slug].name}]({artifact_paths[control_entry.run_slug].as_posix()})")
        lines.append(f"- challenger: [{artifact_paths[run_slug].name}]({artifact_paths[run_slug].as_posix()})")
        lines.append(f"- control config: [{control_entry.config_name}]({control_entry.config_path.as_posix()})")
        if comparison_path is not None:
            lines.append(f"- residual comparison: [{comparison_path.name}]({comparison_path.as_posix()})")
        rows = summarize_pair(control_payload, payload)
        lines.append("")
        for label, left, right in rows:
            lines.append(f"- {label}: `{format_float(left)} -> {format_float(right)}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    manifest = load_json(iteration_dir / "manifest.json")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (iteration_dir / "saved_diagnostics")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    entries = build_run_entries(iteration_dir, manifest)
    control_entry = find_control_entry(entries, args.control_slug)
    search_roots = [iteration_dir]
    search_roots.extend(Path(item).expanduser().resolve() for item in args.search_root)

    result_paths: dict[str, Path] = {}
    artifact_paths: dict[str, Path] = {}
    skipped: list[RunEntry] = []
    for entry in entries:
        artifact_path = discover_artifact_for_run(entry.run_id, entry.run_slug, entry.config_name, search_roots)
        if artifact_path is None:
            skipped.append(entry)
            continue
        artifact_paths[entry.run_slug] = artifact_path
        result_json = results_dir / f"{entry.run_slug}.json"
        result_paths[entry.run_slug] = result_json
        if args.skip_existing and result_json.exists():
            continue
        run_subprocess(
            analyzer_command(args, entry=entry, artifact_path=artifact_path, result_json=result_json),
            dry_run=args.dry_run,
        )

    if control_entry.run_slug not in result_paths:
        raise SystemExit(f"control run `{control_entry.run_slug}` has no discoverable local saved artifact")

    comparison_paths: dict[str, Path] = {}
    for entry in entries:
        if entry.run_slug == control_entry.run_slug or entry.run_slug not in result_paths:
            continue
        comparison_json = results_dir / f"comparison_{entry.run_slug}.json"
        comparison_paths[entry.run_slug] = comparison_json
        if not (args.skip_existing and comparison_json.exists()):
            run_subprocess(
                compare_command(
                    args,
                    left_json=result_paths[control_entry.run_slug],
                    right_json=result_paths[entry.run_slug],
                    left_label=control_entry.run_slug,
                    right_label=entry.run_slug,
                    result_json=comparison_json,
                ),
                dry_run=args.dry_run,
            )

    if not args.dry_run:
        write_summary(
            output_dir / "results_summary.md",
            control_entry=control_entry,
            result_paths=result_paths,
            comparison_paths=comparison_paths,
            artifact_paths=artifact_paths,
            skipped=skipped,
        )


if __name__ == "__main__":
    main()
