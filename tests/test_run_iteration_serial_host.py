from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_iteration_saved_diagnostics import build_run_entries, load_json
from tools.run_iteration_serial_host import dispatch_command, observed_run_status, select_entries


class RunIterationSerialHostTests(unittest.TestCase):
    def make_iteration_dir(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="serial_host_"))
        (tmp / "configs").mkdir(parents=True, exist_ok=True)
        manifest = {
            "dispatch_script": "/tmp/dispatch.sh",
            "wrapper_script": "/tmp/wrapper.py",
            "script": "/tmp/train.py",
            "support_files": ["/tmp/a.py", "/tmp/b.py"],
            "runs": [
                {
                    "config_path": "configs/01_control.json",
                    "run_id": "iter_control",
                    "run_slug": "control",
                    "notes": "baseline",
                },
                {
                    "config_path": "configs/02_novelty.json",
                    "run_id": "iter_novelty",
                    "run_slug": "novelty",
                    "notes": "challenger",
                },
            ],
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name in ("01_control.json", "02_novelty.json"):
            payload = {
                "env": {
                    "RUN_ID": f"run_{name}",
                    "TOKENIZER_PATH": "",
                    "DATA_PATH": "",
                },
                "metadata": {
                    "trainer_script": "/tmp/train.py",
                },
            }
            (tmp / "configs" / name).write_text(json.dumps(payload), encoding="utf-8")
        return tmp

    def test_observed_run_status(self) -> None:
        observed = {"runs": {"control": {"status": "observed_final"}}}
        self.assertEqual(observed_run_status(observed, "control"), "observed_final")
        self.assertEqual(observed_run_status(observed, "missing"), "")

    def test_select_entries_skips_observed_final(self) -> None:
        iteration_dir = self.make_iteration_dir()
        manifest = load_json(iteration_dir / "manifest.json")
        entries = build_run_entries(iteration_dir, manifest)
        observed = {"runs": {"control": {"status": "observed_final"}}}
        selected = select_entries(
            entries,
            observed=observed,
            run_slugs=set(),
            start_at_slug="",
            skip_observed_finals=True,
        )
        self.assertEqual([entry.run_slug for entry in selected], ["novelty"])

    def test_select_entries_honors_start_and_filter(self) -> None:
        iteration_dir = self.make_iteration_dir()
        manifest = load_json(iteration_dir / "manifest.json")
        entries = build_run_entries(iteration_dir, manifest)
        selected = select_entries(
            entries,
            observed={},
            run_slugs={"novelty"},
            start_at_slug="novelty",
            skip_observed_finals=False,
        )
        self.assertEqual([entry.run_slug for entry in selected], ["novelty"])

    def test_dispatch_command_uses_manifest_paths(self) -> None:
        iteration_dir = self.make_iteration_dir()
        manifest = load_json(iteration_dir / "manifest.json")
        entries = build_run_entries(iteration_dir, manifest)
        cmd = dispatch_command(manifest, entries[0], "mini09")
        self.assertEqual(
            cmd[:6],
            ["bash", "/tmp/dispatch.sh", "--host", "mini09", "/tmp/wrapper.py", "/tmp/train.py"],
        )
        self.assertIn(str(entries[0].config_path), cmd)
        self.assertTrue(cmd[-2:] == ["/tmp/a.py", "/tmp/b.py"])


if __name__ == "__main__":
    unittest.main()
