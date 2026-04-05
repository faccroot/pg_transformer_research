#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUN_ID_RE = re.compile(r"^run_id:(\S+)$", re.MULTILINE)
STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms "
    r"step_avg:(?P<step_avg_ms>[0-9.]+)ms "
    r"tok_s:(?P<tok_s>[0-9.]+)"
    r"(?P<rest>.*)$"
)
STOP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) train_time:(?P<train_time_ms>[0-9.]+)ms"
)
FINAL_EXACT_RE = re.compile(r"final_(?P<kind>[^ ]+)_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)")
KV_RE = re.compile(r"([a-zA-Z0-9_]+):([^\s]+)")
KV_EQ_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")
MANAGER_WRITE_RE = re.compile(r"manager_write (?P<rest>.*)$")
TEACHER_BATCH_RE = re.compile(r"teacher_hidden_worker:batch (?P<rest>.*)$")


def parse_value(raw: str):
    if raw.endswith("ms"):
        raw = raw[:-2]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_kvs(raw: str) -> dict[str, object]:
    pairs = KV_RE.findall(raw) + KV_EQ_RE.findall(raw)
    return {key: parse_value(value) for key, value in pairs}


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    run_id_match = RUN_ID_RE.search(text)
    run_id = run_id_match.group(1) if run_id_match else path.stem

    last_step: dict[str, object] | None = None
    last_stop: dict[str, object] | None = None
    finals: dict[str, float] = {}
    external_applied = 0
    manager_last: dict[str, object] | None = None
    manager_write_count = 0
    teacher_hidden_batch_count = 0
    teacher_hidden_written_total = 0
    teacher_hidden_last: dict[str, object] | None = None

    for line in text.splitlines():
        if "external_controller:applied" in line:
            external_applied += 1
        manager_match = MANAGER_WRITE_RE.search(line)
        if manager_match:
            manager_write_count += 1
            manager_last = parse_kvs(manager_match.group("rest"))
            continue
        teacher_batch_match = TEACHER_BATCH_RE.search(line)
        if teacher_batch_match:
            teacher_hidden_batch_count += 1
            teacher_hidden_last = parse_kvs(teacher_batch_match.group("rest"))
            teacher_hidden_written_total += int(teacher_hidden_last.get("written", 0))
            continue
        step_match = STEP_RE.search(line)
        if step_match:
            row: dict[str, object] = {
                "step": int(step_match.group("step")),
                "iterations": int(step_match.group("iterations")),
                "train_loss": float(step_match.group("train_loss")),
                "train_time_ms": float(step_match.group("train_time_ms")),
                "step_avg_ms": float(step_match.group("step_avg_ms")),
                "tok_s": float(step_match.group("tok_s")),
            }
            row.update(parse_kvs(step_match.group("rest")))
            last_step = row
            continue
        stop_match = STOP_VAL_RE.search(line)
        if stop_match:
            last_stop = {
                "stop_step": int(stop_match.group("step")),
                "stop_iterations": int(stop_match.group("iterations")),
                "stop_val_loss": float(stop_match.group("val_loss")),
                "stop_val_bpb": float(stop_match.group("val_bpb")),
                "stop_train_time_ms": float(stop_match.group("train_time_ms")),
            }
            continue
        final_match = FINAL_EXACT_RE.search(line)
        if final_match:
            finals[f"final_{final_match.group('kind')}_bpb"] = float(final_match.group("val_bpb"))

    result: dict[str, object] = {
        "dispatch_log": str(path),
        "run_id": run_id,
        "external_controller_applied": external_applied,
        "manager_write_count": manager_write_count,
        "teacher_hidden_batch_count": teacher_hidden_batch_count,
        "teacher_hidden_written_total": teacher_hidden_written_total,
    }
    if last_step is not None:
        result.update(last_step)
    if last_stop is not None:
        result.update(last_stop)
    if manager_last is not None:
        result.update({f"manager_{key}": value for key, value in manager_last.items()})
    if teacher_hidden_last is not None:
        result.update({f"teacher_hidden_{key}": value for key, value in teacher_hidden_last.items()})
    result.update(finals)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manager/replay MLX dispatch logs.")
    parser.add_argument("suite_dir", help="Generated suite dir containing dispatch_logs/")
    parser.add_argument("--write-json", action="store_true", help="Write summary.json into the suite dir")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    dispatch_dir = suite_dir / "dispatch_logs"
    logs = sorted(dispatch_dir.glob("*.log"))
    if not logs:
        raise SystemExit(f"No dispatch logs found in {dispatch_dir}")

    rows = [parse_log(path) for path in logs]
    rows.sort(key=lambda row: str(row.get("run_id", "")))

    if args.write_json:
        (suite_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
