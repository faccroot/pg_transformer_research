from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _safe_name_component(value: str) -> str:
    text = value.strip()
    if not text:
        return "unnamed"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class StudentSnapshotRuntime:
    root_dir: Path
    run_id: str
    keep_last: int = 2

    def __post_init__(self) -> None:
        self.root_dir = self.root_dir.expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def snapshots_dir(self) -> Path:
        return self.root_dir / "snapshots"

    @property
    def heartbeat_path(self) -> Path:
        return self.root_dir / f"{self.run_id}.heartbeat.json"

    @property
    def index_path(self) -> Path:
        return self.root_dir / f"{self.run_id}.snapshot_index.jsonl"

    @property
    def controller_decision_path(self) -> Path:
        return self.root_dir / f"{self.run_id}.controller_decision.json"

    @property
    def helper_status_dir(self) -> Path:
        return self.root_dir / "helper_status"

    @property
    def helper_proposals_dir(self) -> Path:
        return self.root_dir / "helper_proposals"

    def snapshot_data_path(self, step: int, suffix: str) -> Path:
        clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.snapshots_dir / f"{self.run_id}_step{int(step):07d}{clean_suffix}"

    def snapshot_npz_path(self, step: int) -> Path:
        return self.snapshot_data_path(step, ".npz")

    def snapshot_pt_path(self, step: int) -> Path:
        return self.snapshot_data_path(step, ".pt")

    def snapshot_meta_path(self, step: int) -> Path:
        return self.snapshots_dir / f"{self.run_id}_step{int(step):07d}.json"

    def helper_status_path(self, helper_name: str) -> Path:
        return self.helper_status_dir / f"{self.run_id}.{_safe_name_component(helper_name)}.json"

    def helper_proposals_path(self, helper_name: str) -> Path:
        return self.helper_proposals_dir / f"{self.run_id}.{_safe_name_component(helper_name)}.jsonl"

    def write_heartbeat(self, payload: dict[str, object]) -> None:
        _atomic_write_text(self.heartbeat_path, json.dumps(payload, sort_keys=True, indent=2) + "\n")

    def write_controller_decision(self, payload: dict[str, object]) -> None:
        _atomic_write_text(self.controller_decision_path, json.dumps(payload, sort_keys=True, indent=2) + "\n")

    def read_controller_decision(self) -> dict[str, object] | None:
        return _read_json(self.controller_decision_path)

    def write_helper_status(self, helper_name: str, payload: dict[str, object]) -> None:
        stamped = {
            **payload,
            "run_id": payload.get("run_id", self.run_id),
            "helper_name": payload.get("helper_name", helper_name),
        }
        _atomic_write_text(self.helper_status_path(helper_name), json.dumps(stamped, sort_keys=True, indent=2) + "\n")

    def read_helper_status(self, helper_name: str) -> dict[str, object] | None:
        return _read_json(self.helper_status_path(helper_name))

    def read_all_helper_statuses(self) -> dict[str, dict[str, object]]:
        statuses: dict[str, dict[str, object]] = {}
        for path in sorted(self.helper_status_dir.glob(f"{self.run_id}.*.json")):
            payload = _read_json(path)
            if payload is None:
                continue
            helper_name = str(payload.get("helper_name", path.stem.split(".", 1)[-1]))
            statuses[helper_name] = payload
        return statuses

    def append_helper_proposal(self, helper_name: str, payload: dict[str, object]) -> None:
        stamped = {
            **payload,
            "run_id": payload.get("run_id", self.run_id),
            "helper_name": payload.get("helper_name", helper_name),
        }
        path = self.helper_proposals_path(helper_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, sort_keys=True))
            handle.write("\n")

    def read_latest_helper_proposal(self, helper_name: str) -> dict[str, object] | None:
        path = self.helper_proposals_path(helper_name)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
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

    def write_snapshot_metadata(self, step: int, payload: dict[str, object]) -> None:
        meta_path = self.snapshot_meta_path(step)
        _atomic_write_text(meta_path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
        self.prune_old_snapshots()

    def prune_old_snapshots(self) -> None:
        keep = max(int(self.keep_last), 0)
        if keep <= 0:
            return
        meta_files = sorted(self.snapshots_dir.glob(f"{self.run_id}_step*.json"))
        stale_meta = meta_files[:-keep]
        for meta_path in stale_meta:
            stem = meta_path.stem
            for path in self.snapshots_dir.glob(f"{stem}.*"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
