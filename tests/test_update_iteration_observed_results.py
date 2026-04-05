import json
import tempfile
import unittest
from pathlib import Path

from tools.update_iteration_observed_results import (
    build_observed_results,
    collect_search_roots,
    extract_embedded_dispatch_logs,
    merge_run_payload,
    write_observed_summary,
)


def _write_config(path: Path) -> None:
    payload = {
        "env": {},
        "metadata": {
            "trainer_script_source": "/tmp/train_gpt_mlx.py",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class UpdateIterationObservedResultsTests(unittest.TestCase):
    def test_merge_drops_stale_finals_when_log_source_changes(self) -> None:
        existing = {
            "status": "observed_final",
            "log_path": "/tmp/old.txt",
            "final_int8_zlib_roundtrip_exact": {"val_bpb": 1.5},
        }
        new = {
            "status": "observed_progress",
            "log_path": "/tmp/new.txt",
            "latest_observed_train": {"step": 10},
        }
        merged = merge_run_payload(existing, new)
        self.assertNotIn("final_int8_zlib_roundtrip_exact", merged)
        self.assertEqual(merged["status"], "observed_progress")

    def test_extracts_embedded_logs_and_observed_finals_from_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iteration_dir = root / "iter"
            configs_dir = iteration_dir / "configs"
            configs_dir.mkdir(parents=True)
            _write_config(configs_dir / "01_control.json")
            _write_config(configs_dir / "02_aux.json")
            manifest = {
                "runs": [
                    {
                        "config_path": "configs/01_control.json",
                        "run_id": "run_control",
                        "run_slug": "combined-control-1h",
                    },
                    {
                        "config_path": "configs/02_aux.json",
                        "run_id": "run_aux",
                        "run_slug": "combined-prosody-aux-1h",
                    },
                ],
            }
            (iteration_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            dispatch_text = "\n".join(
                [
                    "[2026-04-01T00:00:00-05:00] dispatch config=01_control.json host=auto attempt=1/8",
                    "[job_1] Claimed mini01 (load: 1.0)",
                    "[job_1] Running on mini01...",
                    "logs/run_control.txt",
                    "run_id:run_control",
                    "iterations:60000 max_wallclock_seconds:3600.0 warmdown_iters:1200 warmdown_fraction:0.18 matrix_lr:0.04 scalar_lr:0.04 embed_lr:0.05",
                    "step:100/60000 train_loss:3.0000 train_time:100000ms",
                    "step:120/60000 val_loss:2.8000 val_bpb:1.6500 train_time:3600000ms",
                    "final_int8_zlib_roundtrip_exact val_loss:2.80000000 val_bpb:1.65000000",
                    "[job_1] Released mini01",
                    "[2026-04-01T00:00:01-05:00] dispatch config=02_aux.json host=auto attempt=1/8",
                    "[job_2] Claimed mini02 (load: 1.0)",
                    "[job_2] Running on mini02...",
                    "logs/run_aux.txt",
                    "run_id:run_aux",
                    "iterations:60000 max_wallclock_seconds:3600.0 warmdown_iters:1200 warmdown_fraction:0.18 matrix_lr:0.04 scalar_lr:0.04 embed_lr:0.05",
                    "step:100/60000 train_loss:2.9000 train_time:100000ms",
                    "step:120/60000 val_loss:2.7000 val_bpb:1.6200 train_time:3600000ms",
                    "final_int8_zlib_roundtrip_exact val_loss:2.70000000 val_bpb:1.62000000",
                    "[job_2] Released mini02",
                ]
            )
            (iteration_dir / "dispatch.out").write_text(dispatch_text + "\n", encoding="utf-8")

            search_roots = collect_search_roots(iteration_dir, [])
            payload = build_observed_results(iteration_dir, check_remote=False, search_roots=search_roots)

            runs = payload["runs"]
            self.assertEqual(runs["combined-control-1h"]["status"], "observed_final")
            self.assertEqual(runs["combined-prosody-aux-1h"]["status"], "observed_final")
            self.assertAlmostEqual(
                runs["combined-control-1h"]["final_int8_zlib_roundtrip_exact"]["val_bpb"],
                1.65,
            )
            self.assertAlmostEqual(
                runs["combined-prosody-aux-1h"]["final_int8_zlib_roundtrip_exact"]["val_bpb"],
                1.62,
            )
            self.assertEqual(payload["best_observed_final"]["run_slug"], "combined-prosody-aux-1h")

            extracted = extract_embedded_dispatch_logs(iteration_dir, [])
            self.assertEqual(extracted, {})

    def test_extract_embedded_logs_matches_entries_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iteration_dir = root / "iter"
            configs_dir = iteration_dir / "configs"
            configs_dir.mkdir(parents=True)
            _write_config(configs_dir / "01_control.json")
            manifest = {
                "runs": [
                    {
                        "config_path": "configs/01_control.json",
                        "run_id": "run_control",
                        "run_slug": "combined-control-1h",
                    }
                ],
            }
            (iteration_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (iteration_dir / "dispatch.out").write_text(
                "\n".join(
                    [
                        "logs/run_control.txt",
                        "run_id:run_control",
                        "step:10/60000 train_loss:3.0 train_time:1000ms",
                        "final_raw_export_ready_exact val_loss:3.1 val_bpb:1.8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            from tools.run_iteration_saved_diagnostics import build_run_entries, load_json

            entries = build_run_entries(iteration_dir, load_json(iteration_dir / "manifest.json"))
            extracted = extract_embedded_dispatch_logs(iteration_dir, entries)
            self.assertIn("combined-control-1h", extracted)
            self.assertTrue(extracted["combined-control-1h"].exists())

            payload = {
                "runs": {
                    "combined-control-1h": {
                        "status": "observed_partial_final",
                        "final_raw_export_ready_exact": {"val_bpb": 1.8},
                    }
                }
            }
            summary_path = write_observed_summary(iteration_dir, payload)
            text = summary_path.read_text(encoding="utf-8")
            self.assertIn("combined-control-1h", text)
            self.assertIn("final_raw_bpb=1.800000", text)


if __name__ == "__main__":
    unittest.main()
