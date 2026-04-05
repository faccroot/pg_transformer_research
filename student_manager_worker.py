from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

from snapshot_signal_runtime import StudentSnapshotRuntime


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_latest_snapshot_meta(runtime: StudentSnapshotRuntime) -> dict[str, object] | None:
    if not runtime.index_path.is_file():
        return None
    try:
        with runtime.index_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _bounded_float(value: object, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(lo, min(hi, parsed))


def _bounded_int(value: object, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def build_decision(
    heartbeat: dict[str, object],
    snapshot_meta: dict[str, object] | None,
    *,
    source: str,
    policy: str,
    helper_statuses: dict[str, dict[str, object]] | None = None,
    helper_proposals: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    step = int(heartbeat.get("step", 0))
    if policy == "neutral":
        latest_snapshot_step = int(snapshot_meta.get("step", 0)) if snapshot_meta is not None else 0
        return {
            "decision_id": f"{source}-step{step:07d}",
            "source": source,
            "min_step": int(step),
            "heartbeat_step": int(step),
            "snapshot_step": int(latest_snapshot_step),
            "note": f"neutral step={step} latest_snapshot_step={latest_snapshot_step}",
        }
    train_loss = _bounded_float(heartbeat.get("train_loss"), default=0.0, lo=0.0, hi=1.0e9)
    replay_cached_examples = _bounded_int(heartbeat.get("replay_cached_examples"), default=0, lo=0, hi=10**9)
    quant_gap_bpb = heartbeat.get("quant_gap_bpb")
    quant_gap = _bounded_float(quant_gap_bpb, default=0.0, lo=-10.0, hi=10.0) if quant_gap_bpb is not None else None
    controller_metrics = heartbeat.get("controller_metrics")
    controller_dict = controller_metrics if isinstance(controller_metrics, dict) else {}
    teacher_disagree = _bounded_float(controller_dict.get("teacher_disagree_frac"), 0.0, 0.0, 1.0)
    teacher_hidden_cache_hit = _bounded_float(controller_dict.get("teacher_hidden_cache_hit_frac"), 0.0, 0.0, 1.0)
    helper_statuses = helper_statuses or {}
    helper_proposals = helper_proposals or {}
    hidden_helper = helper_statuses.get("teacher_hidden_worker", {})
    hidden_helper_state = str(hidden_helper.get("state", "unknown"))
    hidden_helper_cache_entries = _bounded_int(hidden_helper.get("cache_entries"), default=0, lo=0, hi=10**9)
    hidden_helper_last_written = _bounded_int(hidden_helper.get("last_written"), default=0, lo=0, hi=10**9)
    hidden_helper_elapsed_ms = _bounded_float(hidden_helper.get("elapsed_ms"), default=0.0, lo=0.0, hi=1.0e9)
    hidden_helper_proposal = helper_proposals.get("teacher_hidden_worker", {})
    hidden_helper_suggested_distill_mult = _bounded_float(
        hidden_helper_proposal.get("suggested_distill_weight_mult"),
        default=1.0,
        lo=0.0,
        hi=4.0,
    )

    sanitize_every = 4
    if step >= 64:
        sanitize_every = 8
    if step >= 128:
        sanitize_every = 16
    if step >= 256:
        sanitize_every = 32
    if train_loss > 4.0:
        sanitize_every = min(sanitize_every, 8)

    replay_mix_fraction = 0.0
    if replay_cached_examples >= 32:
        replay_mix_fraction = 0.10
    if replay_cached_examples >= 128:
        replay_mix_fraction = 0.20
    if replay_cached_examples >= 256:
        replay_mix_fraction = 0.25

    distill_weight_mult = 1.0
    branch_weight_mult = 1.0
    branch_extra_max_branches = 0
    if teacher_disagree >= 0.06:
        distill_weight_mult = 0.90
        branch_weight_mult = 1.20
    if teacher_disagree >= 0.12:
        distill_weight_mult = 0.80
        branch_weight_mult = 1.35
        branch_extra_max_branches = 1

    if hidden_helper_cache_entries >= 32 or teacher_hidden_cache_hit >= 0.25:
        distill_weight_mult = max(distill_weight_mult, 1.05)
    if hidden_helper_cache_entries >= 128 or teacher_hidden_cache_hit >= 0.50:
        distill_weight_mult = max(distill_weight_mult, 1.10)
    if hidden_helper_state == "active" and hidden_helper_last_written > 0 and hidden_helper_elapsed_ms <= 3000.0:
        distill_weight_mult = max(distill_weight_mult, 1.10)
    distill_weight_mult = max(distill_weight_mult, hidden_helper_suggested_distill_mult)

    if quant_gap is not None and quant_gap > 0.03:
        replay_mix_fraction *= 0.5
        branch_weight_mult *= 0.8

    latest_snapshot_step = int(snapshot_meta.get("step", 0)) if snapshot_meta is not None else 0
    note = (
        f"rule_based step={step} train_loss={train_loss:.4f} replay_cached={replay_cached_examples} "
        f"teacher_disagree={teacher_disagree:.3f} hidden_cache_hit={teacher_hidden_cache_hit:.3f} "
        f"hidden_helper_state={hidden_helper_state} hidden_helper_cache={hidden_helper_cache_entries} "
        f"latest_snapshot_step={latest_snapshot_step}"
    )
    return {
        "decision_id": f"{source}-step{step:07d}",
        "source": source,
        "min_step": int(step),
        "sanitize_every": int(sanitize_every),
        "replay_mix_fraction": float(replay_mix_fraction),
        "distill_weight_mult": float(distill_weight_mult),
        "branch_weight_mult": float(branch_weight_mult),
        "branch_extra_max_branches": int(branch_extra_max_branches),
        "heartbeat_step": int(step),
        "snapshot_step": int(latest_snapshot_step),
        "note": note,
    }


def main() -> None:
    run_id = os.environ.get("RUN_ID", "").strip()
    snapshot_root = os.environ.get("SNAPSHOT_BUS_DIR", "").strip()
    if not run_id or not snapshot_root:
        raise SystemExit("Set RUN_ID and SNAPSHOT_BUS_DIR")
    poll_seconds = max(float(os.environ.get("MANAGER_POLL_SECONDS", "5.0")), 0.1)
    once = bool(int(os.environ.get("MANAGER_ONCE", "0")))
    source = os.environ.get("MANAGER_SOURCE", "rule_manager")
    policy = os.environ.get("MANAGER_POLICY", "rule_based").strip().lower()
    helper_names = [
        name.strip()
        for name in os.environ.get("MANAGER_HELPER_NAMES", "teacher_hidden_worker").split(",")
        if name.strip()
    ]
    runtime = StudentSnapshotRuntime(Path(snapshot_root), run_id=run_id)

    last_heartbeat_mtime = -1
    idle_polls = 0
    max_idle_polls = max(int(os.environ.get("MANAGER_MAX_IDLE_POLLS", "0")), 0)

    while True:
        heartbeat_path = runtime.heartbeat_path
        heartbeat = None
        try:
            heartbeat_stat = heartbeat_path.stat()
            if heartbeat_stat.st_mtime_ns != last_heartbeat_mtime:
                heartbeat = _load_json(heartbeat_path)
                if heartbeat is not None:
                    last_heartbeat_mtime = int(heartbeat_stat.st_mtime_ns)
                    snapshot_meta = _load_latest_snapshot_meta(runtime)
                    helper_statuses = {
                        helper_name: payload
                        for helper_name in helper_names
                        if (payload := runtime.read_helper_status(helper_name)) is not None
                    }
                    helper_proposals = {
                        helper_name: payload
                        for helper_name in helper_names
                        if (payload := runtime.read_latest_helper_proposal(helper_name)) is not None
                    }
                    decision = build_decision(
                        heartbeat,
                        snapshot_meta,
                        source=source,
                        policy=policy,
                        helper_statuses=helper_statuses,
                        helper_proposals=helper_proposals,
                    )
                    runtime.write_controller_decision(decision)
                    print(
                        f"manager_write decision_id={decision['decision_id']} "
                        f"policy={policy} "
                        f"sanitize_every={decision.get('sanitize_every', '-') } "
                        f"distill_weight_mult={float(decision.get('distill_weight_mult', 1.0)):.2f} "
                        f"replay_mix_fraction={float(decision.get('replay_mix_fraction', 0.0)):.3f} "
                        f"branch_weight_mult={float(decision.get('branch_weight_mult', 0.0)):.2f} "
                        f"helper_statuses={','.join(sorted(helper_statuses)) or '-'}",
                        flush=True,
                    )
                    idle_polls = 0
                    if once:
                        return
        except FileNotFoundError:
            pass
        except OSError:
            pass

        idle_polls += 1
        if max_idle_polls > 0 and idle_polls >= max_idle_polls:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
