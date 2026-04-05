#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = ROOT / "research" / "competition_prs" / "open_current"
DEFAULT_OUT_DIR = ROOT / "research" / "competition_prs" / "analysis"


BASELINE_CONFIG = {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "9",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "2",
    "TRAIN_SEQ_LEN": "1024",
    "WARMDOWN_ITERS": "1200",
    "TIE_EMBEDDINGS": "1",
}


TRACKED_ENV_KEYS = [
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "MODEL_DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_MULT",
    "TRAIN_SEQ_LEN",
    "TRAIN_BATCH_TOKENS",
    "WARMDOWN_ITERS",
    "EVAL_STRIDE",
    "EVAL_BATCH_SEQS",
    "SW_EVAL_BATCH",
    "MATRIX_LR",
    "SCALAR_LR",
    "TIED_EMBED_LR",
    "EMBED_LR",
    "HEAD_LR",
    "MUON_MOMENTUM",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "QAT_START_FRAC",
    "MTP_HEADS",
    "MTP_WEIGHT",
    "FP16_KEEP_NAME_PATTERNS",
    "INT8_KEEP_FLOAT_NAME_PATTERNS",
    "INT8_COARSEN_OVERRIDES",
    "INT4_LAYERS",
    "INT4_STEP",
    "INT6_LAYERS",
    "NUM_RECURRENCE",
    "NUM_UNIQUE_LAYERS",
    "LORA_RANK",
    "TOK_EMB_QAT_MODE",
]


LEVER_PATTERNS: dict[str, list[str]] = {
    "sliding_window_eval": [r"\bsliding window\b", r"\bEVAL_STRIDE=", r"\bSW_EVAL_BATCH\b"],
    "long_context_train": [r"\bTRAIN_SEQ_LEN=(1536|2048|3072|4096|8192)\b", r"\bseq_len=4096\b", r"\bTRAIN_SEQ_LEN=4096\b"],
    "tokenizer_change": [r"\bsp2048\b", r"\bsp4096\b", r"\bVOCAB_SIZE=2048\b", r"\bVOCAB_SIZE=4096\b", r"\bfineweb_4096_bpe\.model\b"],
    "sp4096": [r"\bsp4096\b", r"\bVOCAB_SIZE=4096\b", r"\bfineweb_4096_bpe\.model\b"],
    "sp2048": [r"\bsp2048\b", r"\bVOCAB_SIZE=2048\b", r"\bfineweb_2048_bpe\.model\b"],
    "mixed_quant": [r"\bmixed quant\b", r"\bint6\b", r"\bint4\b", r"\bint8\b"],
    "int6_quant": [r"\bint6\b"],
    "int4_quant": [r"\bint4\b"],
    "fp16_embedding": [r"\bfp16.*embed", r"\bfp16 tied embedding\b", r"\bFP16_KEEP_NAME_PATTERNS=tok_emb\b", r"\bINT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb\.weight\b"],
    "qat": [r"\bQAT\b", r"\bfake[- ]quant", r"\bTOK_EMB_QAT_MODE\b", r"\bQAT_START_FRAC\b", r"\bSTE\b"],
    "recurrence": [r"\brecurr", r"\bshared block\b", r"\blooped\b", r"\bNUM_RECURRENCE\b", r"\bdepth-recurrent\b"],
    "lora": [r"\bLoRA\b", r"\bLORA_RANK\b"],
    "mtp": [r"\bMTP\b", r"\bmulti-token prediction\b", r"\bMTP_HEADS\b", r"\bMTP_WEIGHT\b"],
    "ttt": [r"\bTTT\b", r"\btest-time training\b"],
    "val_only_training": [r"\bval-only\b", r"\bmemorizes the evaluation data\b", r"\bvalonly\b"],
    "wider_mlp": [r"\bMLP_MULT=3", r"\bwider MLP\b", r"\bMLP 3x\b"],
    "swiglu": [r"\bSwiGLU\b"],
    "more_layers": [r"\bNUM_LAYERS=10\b", r"\bNUM_LAYERS=11\b", r"\bNUM_LAYERS=12\b", r"\b10 transformer layers\b", r"\bDepth12\b"],
    "optimizer_tuned": [r"\bMUON_MOMENTUM=0\.99\b", r"\bWARMDOWN_ITERS=3000\b", r"\bTIED_EMBED_LR=0\.03\b", r"\bMATRIX_LR=0\.02\b"],
    "queryless_attention": [r"\bqueryless\b", r"\bdrop(ping)? Wq\b", r"\bremove the Query projection\b"],
    "overtone_init": [r"\bovertone\b", r"\bspectral shaping\b"],
    "residual_init_trick": [r"\bphase-transition residual mixing\b", r"\bresid_mix init\b"],
    "binary_embedding": [r"\bbinary dims\b", r"\bbinary tail\b", r"\bbinary embedding\b"],
    "morphological_tokenizer": [r"\bmorfessor\b", r"\bmorpholog"],
    "document_packing": [r"\bdocument-aware\b", r"\bbest-fit pack", r"\bsentence-aware\b", r"\bpacking\b"],
}


VAL_BPB_PATTERNS = [
    re.compile(r"\bval_bpb[:=]\s*([0-9]+\.[0-9]+)"),
    re.compile(r"\bval_bpb\s+([0-9]+\.[0-9]+)"),
    re.compile(r"\b([0-9]+\.[0-9]+)\s*BPB\b", re.IGNORECASE),
]


@dataclass
class PRArtifacts:
    pr_dir: Path
    metadata: dict
    changed_files: list[str]
    relevant_text: str
    source_text_files: list[str]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_changed_files(patch_path: Path) -> list[str]:
    changed: list[str] = []
    diff_re = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    for line in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = diff_re.match(line)
        if match:
            changed.append(match.group(2))
    return changed


def tail_text(path: Path, max_chars: int = 200_000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def gather_relevant_text(pr_dir: Path, changed_files: list[str]) -> tuple[str, list[str]]:
    blobs: list[str] = []
    source_text_files: list[str] = []

    top_readme = pr_dir / "README.md"
    if top_readme.exists():
        blobs.append(top_readme.read_text(encoding="utf-8", errors="replace"))
        source_text_files.append("README.md")

    patch_path = pr_dir / "changes.patch"
    if patch_path.exists():
        added_lines = []
        for line in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("+++") or not line.startswith("+"):
                continue
            added_lines.append(line[1:])
        blobs.append("\n".join(added_lines))
        source_text_files.append("changes.patch(+)")

    source_root = pr_dir / "source"
    for rel_path in changed_files:
        path = source_root / rel_path
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".json", ".log", ".py", ".txt"}:
            continue
        try:
            text = tail_text(path) if path.suffix.lower() == ".log" else path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        blobs.append(text)
        source_text_files.append(rel_path)
    return "\n\n".join(blobs), source_text_files


def load_pr_artifacts(pr_dir: Path) -> PRArtifacts:
    metadata = load_json(pr_dir / "metadata.json")
    changed_files = parse_changed_files(pr_dir / "changes.patch")
    relevant_text, source_text_files = gather_relevant_text(pr_dir, changed_files)
    return PRArtifacts(
        pr_dir=pr_dir,
        metadata=metadata,
        changed_files=changed_files,
        relevant_text=relevant_text,
        source_text_files=source_text_files,
    )


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n")


def extract_env_values(text: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for key in TRACKED_ENV_KEYS:
        pattern = re.compile(rf"\b{re.escape(key)}\s*=\s*([^\s\\`]+)")
        for match in pattern.finditer(text):
            value = match.group(1).strip().strip(",")
            if value not in values[key]:
                values[key].append(value)
    return dict(values)


def choose_primary_value(values: list[str], baseline: str | None) -> str | None:
    if not values:
        return None
    non_baseline = [value for value in values if baseline is None or value != baseline]
    if len(non_baseline) == 1:
        return non_baseline[0]
    if non_baseline:
        return non_baseline[-1]
    return values[-1]


def derive_primary_config(env_values: dict[str, list[str]]) -> dict[str, str]:
    config: dict[str, str] = {}
    for key, values in env_values.items():
        chosen = choose_primary_value(values, BASELINE_CONFIG.get(key))
        if chosen is not None:
            config[key] = chosen
    return config


def derive_baseline_deltas(config: dict[str, str]) -> dict[str, dict[str, str]]:
    deltas: dict[str, dict[str, str]] = {}
    for key, baseline in BASELINE_CONFIG.items():
        value = config.get(key)
        if value is not None and value != baseline:
            deltas[key] = {"baseline": baseline, "value": value}
    return deltas


def extract_val_bpb_values(text: str) -> list[float]:
    values: set[float] = set()
    for pattern in VAL_BPB_PATTERNS:
        for match in pattern.finditer(text):
            try:
                values.add(float(match.group(1)))
            except ValueError:
                continue
    return sorted(values)


def extract_quant_gap_values(text: str) -> list[float]:
    values: set[float] = set()
    for match in re.finditer(r"\bquant gap\b[^0-9]*([0-9]+\.[0-9]+)", text, flags=re.IGNORECASE):
        try:
            values.add(float(match.group(1)))
        except ValueError:
            continue
    return sorted(values)


def extract_artifact_sizes(text: str) -> list[int]:
    values: set[int] = set()
    for match in re.finditer(r"\b([0-9]{6,}) bytes\b", text):
        try:
            values.add(int(match.group(1)))
        except ValueError:
            continue
    return sorted(values)


def extract_track_type(title: str, text: str) -> str:
    lowered = f"{title}\n{text}".lower()
    if "non-record" in lowered:
        return "non-record"
    if "wip" in lowered or "pending" in lowered or "draft" in lowered:
        return "wip"
    return "record-or-open"


def detect_levers(title: str, text: str) -> dict[str, bool]:
    blob = f"{title}\n{text}"
    levers: dict[str, bool] = {}
    for lever, patterns in LEVER_PATTERNS.items():
        levers[lever] = any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in patterns)
    return levers


def changed_areas(changed_files: Iterable[str]) -> list[str]:
    areas: set[str] = set()
    for path in changed_files:
        if path.startswith("data/"):
            areas.add("data")
        if path.startswith("records/"):
            areas.add("records")
        if path.startswith("tools/"):
            areas.add("tooling")
        if path.startswith("tests/"):
            areas.add("tests")
        if path.endswith("train_gpt.py") or path.endswith("train_gpt_mlx.py"):
            areas.add("trainer")
        if "mlx" in path.lower():
            areas.add("mlx")
        if "README" in path or path.endswith(".md"):
            areas.add("docs")
        if "submission" in path.lower():
            areas.add("submission")
    return sorted(areas)


def summarize_pr(pr: PRArtifacts) -> dict:
    title = pr.metadata["title"]
    text = normalize_text(pr.relevant_text)
    env_values = extract_env_values(text)
    config = derive_primary_config(env_values)
    val_bpb_values = extract_val_bpb_values(text)
    quant_gap_values = extract_quant_gap_values(text)
    artifact_sizes = extract_artifact_sizes(text)
    levers = detect_levers(title, text)
    active_levers = sorted([name for name, active in levers.items() if active])

    best_claimed_val_bpb = min(val_bpb_values) if val_bpb_values else None
    summary = {
        "pr_number": pr.metadata["number"],
        "title": title,
        "url": pr.metadata["html_url"],
        "author": pr.metadata["user"],
        "updated_at": pr.metadata["updated_at"],
        "draft": pr.metadata["draft"],
        "track_type": extract_track_type(title, text),
        "changed_files_count": len(pr.changed_files),
        "changed_files": pr.changed_files,
        "changed_areas": changed_areas(pr.changed_files),
        "source_text_files": pr.source_text_files,
        "env_values": env_values,
        "primary_config": config,
        "baseline_deltas": derive_baseline_deltas(config),
        "levers": levers,
        "active_levers": active_levers,
        "best_claimed_val_bpb": best_claimed_val_bpb,
        "all_claimed_val_bpb": val_bpb_values,
        "quant_gap_values": quant_gap_values,
        "artifact_sizes_bytes": artifact_sizes,
        "interesting_for_research": any(
            levers[name]
            for name in [
                "recurrence",
                "lora",
                "mtp",
                "overtone_init",
                "residual_init_trick",
                "binary_embedding",
                "morphological_tokenizer",
            ]
        ),
        "likely_score_chasing": any(
            levers[name]
            for name in [
                "sliding_window_eval",
                "mixed_quant",
                "wider_mlp",
                "optimizer_tuned",
                "val_only_training",
            ]
        ),
        "manual_review_status": "unreviewed",
        "manual_notes": "",
    }
    return summary


def build_lever_summary(entries: list[dict]) -> dict:
    lever_to_rows: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        for lever in entry["active_levers"]:
            lever_to_rows[lever].append(entry)

    summary: dict[str, dict] = {}
    for lever, rows in sorted(lever_to_rows.items()):
        scores = [row["best_claimed_val_bpb"] for row in rows if row["best_claimed_val_bpb"] is not None]
        summary[lever] = {
            "count": len(rows),
            "best_claimed_val_bpb": min(scores) if scores else None,
            "prs": sorted(
                [
                    {
                        "pr_number": row["pr_number"],
                        "title": row["title"],
                        "best_claimed_val_bpb": row["best_claimed_val_bpb"],
                    }
                    for row in rows
                ],
                key=lambda row: (float("inf") if row["best_claimed_val_bpb"] is None else row["best_claimed_val_bpb"], row["pr_number"]),
            ),
        }
    return summary


def build_combo_summary(entries: list[dict], combo_size: int = 2) -> dict[str, dict]:
    combos: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for entry in entries:
        levers = entry["active_levers"]
        if len(levers) < combo_size:
            continue
        for i in range(len(levers)):
            for j in range(i + 1, len(levers)):
                combo = tuple(sorted((levers[i], levers[j])))
                combos[combo].append(entry)

    summary: dict[str, dict] = {}
    for combo, rows in sorted(combos.items()):
        scores = [row["best_claimed_val_bpb"] for row in rows if row["best_claimed_val_bpb"] is not None]
        summary[" + ".join(combo)] = {
            "count": len(rows),
            "best_claimed_val_bpb": min(scores) if scores else None,
            "pr_numbers": sorted(row["pr_number"] for row in rows),
        }
    return summary


def write_outputs(out_dir: Path, entries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_jsonl = out_dir / "open_pr_lever_tracker.jsonl"
    with tracker_jsonl.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=lambda row: row["pr_number"]):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    manifest = {
        "generated_at_utc": now_utc(),
        "pr_count": len(entries),
        "entries": sorted(entries, key=lambda row: row["pr_number"]),
        "lever_summary": build_lever_summary(entries),
        "combo_summary_2": build_combo_summary(entries, combo_size=2),
    }
    write_text(out_dir / "open_pr_lever_tracker.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Open PR Lever Tracker",
        "",
        f"- Generated at: `{manifest['generated_at_utc']}`",
        f"- PR count: `{manifest['pr_count']}`",
        "",
        "| PR | Score | Track | Key Levers | Title |",
        "|---:|---:|---|---|---|",
    ]
    for entry in sorted(
        entries,
        key=lambda row: (float("inf") if row["best_claimed_val_bpb"] is None else row["best_claimed_val_bpb"], row["pr_number"]),
    ):
        score = "" if entry["best_claimed_val_bpb"] is None else f"{entry['best_claimed_val_bpb']:.4f}"
        levers = ", ".join(entry["active_levers"][:6])
        lines.append(f"| {entry['pr_number']} | {score} | {entry['track_type']} | {levers} | {entry['title'].replace('|', '\\|')} |")
    write_text(out_dir / "open_pr_lever_tracker.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a structured lever tracker from the archived competition PRs.")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR), help="PR archive directory created by sync_open_prs.py")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for JSON/JSONL/Markdown outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    entries: list[dict] = []
    for pr_dir in sorted(child for child in archive_dir.iterdir() if child.is_dir() and child.name.startswith("pr_")):
        try:
            pr = load_pr_artifacts(pr_dir)
            entries.append(summarize_pr(pr))
        except Exception as exc:
            entries.append(
                {
                    "pr_number": int(pr_dir.name.split("_", 2)[1]),
                    "title": pr_dir.name,
                    "manual_review_status": "extract_failed",
                    "manual_notes": str(exc),
                    "active_levers": [],
                    "best_claimed_val_bpb": None,
                    "track_type": "unknown",
                    "changed_files_count": 0,
                    "changed_files": [],
                    "changed_areas": [],
                    "source_text_files": [],
                    "env_values": {},
                    "primary_config": {},
                    "baseline_deltas": {},
                    "levers": {},
                    "all_claimed_val_bpb": [],
                    "quant_gap_values": [],
                    "artifact_sizes_bytes": [],
                    "interesting_for_research": False,
                    "likely_score_chasing": False,
                    "url": None,
                    "author": None,
                    "updated_at": None,
                    "draft": None,
                }
            )
    write_outputs(out_dir, entries)
    print(f"Wrote lever tracker for {len(entries)} PRs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
