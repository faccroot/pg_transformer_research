import json
import tempfile
import unittest
from pathlib import Path

from tools.run_iteration_saved_diagnostics import (
    build_run_entries,
    discover_artifact_for_run,
    find_control_entry,
    infer_trainer_module,
    resolve_data_override,
    resolve_tokenizer_override,
)


class RunIterationSavedDiagnosticsTests(unittest.TestCase):
    def test_infer_trainer_module_for_base_and_custom_trainers(self) -> None:
        self.assertEqual(infer_trainer_module({"trainer_script_source": "/tmp/train_gpt_mlx.py"}), "")
        self.assertEqual(
            infer_trainer_module({"trainer_script_source": "/tmp/train_gpt_mlx_sidecar_canonical.py"}),
            "train_gpt_mlx_sidecar_canonical",
        )

    def test_resolve_tokenizer_override_prefers_manifest_support_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iteration_dir = root / "iter"
            configs_dir = iteration_dir / "configs"
            configs_dir.mkdir(parents=True)
            tokenizer = root / "fineweb_1024_bpe.model"
            tokenizer.write_text("stub", encoding="utf-8")
            config_path = configs_dir / "01_control.json"
            config_path.write_text("{}", encoding="utf-8")
            got = resolve_tokenizer_override(
                config_path,
                {"support_files": [str(tokenizer)]},
                {"TOKENIZER_PATH": "fineweb_1024_bpe.model"},
            )
            self.assertEqual(got, tokenizer.resolve().as_posix())

    def test_resolve_data_override_uses_existing_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "dataset"
            data_dir.mkdir()
            config_path = root / "configs" / "01_control.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}", encoding="utf-8")
            got = resolve_data_override(config_path, {"DATA_PATH": str(data_dir)})
            self.assertEqual(got, data_dir.resolve().as_posix())

    def test_discover_artifact_prefers_artifacts_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            other = root / "other"
            other.mkdir()
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (other / "run123_mlx_model.npz").write_text("x", encoding="utf-8")
            preferred = artifacts / "run123_mlx_model.npz"
            preferred.write_text("y", encoding="utf-8")
            got = discover_artifact_for_run("run123", "run123", "01_run123.json", [root])
            self.assertEqual(got, preferred.resolve())

    def test_discover_artifact_falls_back_to_slug_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            target = artifacts / "plain_control_mini07_mlx_model.npz"
            target.write_text("x", encoding="utf-8")
            got = discover_artifact_for_run("unmatched_run_id", "plain-control", "01_plain-control.json", [root])
            self.assertEqual(got, target.resolve())

    def test_build_entries_and_find_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iteration_dir = root / "iter"
            configs_dir = iteration_dir / "configs"
            configs_dir.mkdir(parents=True)
            tokenizer = root / "fineweb_1024_bpe.model"
            tokenizer.write_text("stub", encoding="utf-8")
            data_dir = root / "dataset"
            data_dir.mkdir()
            manifest = {
                "support_files": [str(tokenizer)],
                "runs": [
                    {
                        "config_path": "configs/01_control.json",
                        "run_id": "run_control",
                        "run_slug": "combined-control-1h",
                        "notes": "control",
                    },
                    {
                        "config_path": "configs/02_sidecar.json",
                        "run_id": "run_sidecar",
                        "run_slug": "combined-sidecar-1h",
                        "notes": "sidecar",
                    },
                ],
            }
            for name, trainer in (
                ("01_control.json", "train_gpt_mlx.py"),
                ("02_sidecar.json", "train_gpt_mlx_sidecar_canonical.py"),
            ):
                payload = {
                    "env": {
                        "TOKENIZER_PATH": "fineweb_1024_bpe.model",
                        "DATA_PATH": str(data_dir),
                    },
                    "metadata": {
                        "trainer_script_source": f"/tmp/{trainer}",
                        "notes": name,
                    },
                }
                (configs_dir / name).write_text(json.dumps(payload), encoding="utf-8")
            entries = build_run_entries(iteration_dir, manifest)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[1].trainer_module, "train_gpt_mlx_sidecar_canonical")
            control = find_control_entry(entries, "")
            self.assertEqual(control.run_slug, "combined-control-1h")


if __name__ == "__main__":
    unittest.main()
