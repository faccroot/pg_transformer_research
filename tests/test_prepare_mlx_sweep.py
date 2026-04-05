from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_mlx_sweep import prepare_sweep


ROOT = Path("/home/zaytor/transformer_research/parameter-golf")


class PrepareMlxSweepTests(unittest.TestCase):
    def test_queue_launch_script_skips_completed_runs_after_nonzero_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            spec_path = tmp_path / "queue_retry_guard.json"
            spec_path.write_text("{}", encoding="utf-8")
            out_dir = tmp_path / "out"
            spec = {
                "sweep_slug": "queue-retry-guard",
                "dispatch_mode": "queue",
                "script": str(ROOT / "train_gpt_mlx.py"),
                "wrapper_script": str(ROOT / "run_train_gpt_mlx_config.py"),
                "runs": [
                    {
                        "slug": "baseline",
                        "env": {
                            "RUN_ID": "queue_retry_guard_baseline",
                        },
                    }
                ],
            }

            prepare_sweep(spec, spec_path, out_dir)

            launch_text = (out_dir / "launch.sh").read_text(encoding="utf-8")
            self.assertIn("run_has_completed_outputs()", launch_text)
            self.assertIn("dispatch_skip_completed", launch_text)
            self.assertIn("dispatch_nonzero_but_completed", launch_text)

    def test_queue_mode_stages_env_file_values_and_rewrites_to_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            priors_path = tmp_path / "priors_v1.npz"
            priors_path.write_bytes(b"fake-priors")
            spec_path = tmp_path / "rep_ablation.json"
            spec_path.write_text("{}", encoding="utf-8")
            out_dir = tmp_path / "out"
            spec = {
                "sweep_slug": "rep-ablation",
                "dispatch_mode": "queue",
                "script": str(ROOT / "train_gpt_mlx.py"),
                "wrapper_script": str(ROOT / "run_train_gpt_mlx_config.py"),
                "support_files": [
                    str(ROOT / "train_gpt_mlx_representation.py"),
                    str(ROOT / "tools" / "representation_learning" / "runtime_mlx.py"),
                    str(ROOT / "tools" / "representation_learning" / "schemas.py"),
                ],
                "base_env": {
                    "TOKENIZER_PATH": "~/transformer_research/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
                },
                "runs": [
                    {
                        "slug": "rep-init",
                        "script": str(ROOT / "train_gpt_mlx_representation.py"),
                        "env": {
                            "RUN_ID": "rep_init",
                            "REP_LEARN_PRIORS_PATH": str(priors_path),
                            "REP_LEARN_QK_INIT": "1",
                        },
                    }
                ],
            }

            _configs_dir, manifest_path, manifest = prepare_sweep(spec, spec_path, out_dir)

            config_path = out_dir / manifest["runs"][0]["config_path"]
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["env"]["REP_LEARN_PRIORS_PATH"], "priors_v1.npz")
            self.assertIn(str(priors_path.resolve()), manifest["support_files"])
            self.assertIn(
                str((ROOT / "tools" / "representation_learning" / "runtime_mlx.py").resolve()),
                manifest["support_files"],
            )
            self.assertTrue(manifest_path.is_file())

    def test_queue_mode_stages_file_valued_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            priors_path = tmp_path / "merge_v1.npz"
            priors_path.write_bytes(b"fake-priors")
            spec_path = tmp_path / "rep_eval.json"
            spec_path.write_text("{}", encoding="utf-8")
            out_dir = tmp_path / "out"
            spec = {
                "sweep_slug": "rep-eval",
                "dispatch_mode": "queue",
                "script": str(ROOT / "tools" / "representation_learning" / "eval_mlx_representation_init.py"),
                "wrapper_script": str(ROOT / "run_cli_json_config.py"),
                "support_files": [
                    str(ROOT / "tools" / "representation_learning" / "runtime_mlx.py"),
                    str(ROOT / "tools" / "representation_learning" / "schemas.py"),
                ],
                "base_args": {
                    "output": str(tmp_path / "report.json"),
                    "artifact": [f"merge={priors_path}"],
                },
                "runs": [
                    {
                        "slug": "rep-eval",
                        "args": {
                            "data_path": "./data/datasets/fineweb10B_sp1024",
                        },
                    }
                ],
            }

            _configs_dir, manifest_path, manifest = prepare_sweep(spec, spec_path, out_dir)

            config_path = out_dir / manifest["runs"][0]["config_path"]
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["args"]["artifact"], ["merge=merge_v1.npz"])
            self.assertIn(str(priors_path.resolve()), manifest["support_files"])
            self.assertIn(str((ROOT / "run_cli_json_config.py").resolve()), manifest["wrapper_script"])
            self.assertTrue(manifest_path.is_file())

    def test_queue_mode_ignores_binary_support_files_during_import_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            binary_path = tmp_path / "tokenizer.model"
            binary_path.write_bytes(b"\x80binary-tokenizer")
            spec_path = tmp_path / "binary_support.json"
            spec_path.write_text("{}", encoding="utf-8")
            out_dir = tmp_path / "out"
            spec = {
                "sweep_slug": "binary-support",
                "dispatch_mode": "queue",
                "script": str(ROOT / "train_gpt_mlx.py"),
                "wrapper_script": str(ROOT / "run_train_with_manager.py"),
                "support_files": [
                    str(ROOT / "run_train_with_manager.py"),
                    str(binary_path),
                ],
                "base_env": {
                    "TOKENIZER_PATH": str(binary_path),
                    "STUDENT_SNAPSHOT_DIR": "logs/student_snapshot_bus",
                    "START_MANAGER": 0,
                },
                "runs": [
                    {
                        "slug": "binary-support-smoke",
                        "env": {
                            "RUN_ID": "binary_support_smoke",
                        },
                    }
                ],
            }

            _configs_dir, manifest_path, manifest = prepare_sweep(spec, spec_path, out_dir)

            self.assertIn(str(binary_path.resolve()), manifest["support_files"])
            self.assertTrue(manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
