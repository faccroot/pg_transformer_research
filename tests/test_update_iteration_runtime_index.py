import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.update_iteration_runtime_index import (
    list_cluster_hosts_rows,
    list_cluster_jobs_rows,
    search_analysis_metric_rows,
    search_metric_rows,
    update_index,
)


def _write_config(path: Path) -> None:
    payload = {
        "env": {},
        "metadata": {
            "trainer_script_source": "/tmp/train_gpt_mlx.py",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class UpdateIterationRuntimeIndexTests(unittest.TestCase):
    def test_update_builds_runtime_snapshot_and_searchable_metrics(self) -> None:
        def fake_snapshot(*, cluster_root: Path, redis_host: str) -> dict:
            return {
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "cluster_root": cluster_root.as_posix(),
                "redis_host": redis_host,
                "status_source": "test",
                "hosts": [
                    {
                        "host": "mini01",
                        "status": "BUSY",
                        "lock_owner": "job_1",
                        "lock_ttl": 123,
                        "active_job_id": "job_1",
                        "active_job_status": "running",
                        "active_script": "run_control.py",
                        "active_remote_dir": "~/jobs/job_1",
                    }
                ],
                "locks": [
                    {"host": "mini01", "key": "cluster:lock:mini01", "owner": "job_1", "ttl": 123},
                ],
                "jobs": [
                    {
                        "job_id": "job_1",
                        "host": "mini01",
                        "script": "run_control.py",
                        "started": "2026-04-05T00:00:00-05:00",
                        "status": "running",
                        "pid": "100",
                        "remote_dir": "~/jobs/job_1",
                    }
                ],
            }

        def fake_write_snapshot(output_path: Path, summary_path: Path, payload: dict) -> None:
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            summary_path.write_text("# test\n", encoding="utf-8")

        with patch("tools.update_iteration_runtime_index.build_cluster_snapshot", side_effect=fake_snapshot), patch(
            "tools.update_iteration_runtime_index.write_cluster_snapshot", side_effect=fake_write_snapshot
        ):
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
                            "notes": "control",
                        }
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
                        "step:100/60000 train_loss:3.0000 train_time:100000ms ce:2.9000 residnov_mean:0.4000",
                        "step:120/60000 val_loss:2.8000 val_bpb:1.6500 train_time:3600000ms",
                        "final_int8_zlib_roundtrip_exact val_loss:2.80000000 val_bpb:1.65000000",
                        "[job_1] Released mini01",
                    ]
                )
                (iteration_dir / "dispatch.out").write_text(dispatch_text + "\n", encoding="utf-8")
                results_dir = iteration_dir / "results"
                results_dir.mkdir()
                (results_dir / "control_analysis.json").write_text(
                    json.dumps(
                        {
                            "label": "control",
                            "mean_nll": 2.5,
                            "positions_analyzed": 128,
                            "nll_acf": {
                                "summary": {
                                    "all": {"mean": 0.1, "positive_area": 0.2},
                                    "within_regime": {"mean": 0.11},
                                }
                            },
                            "residual_modes": {
                                "argmax_embedding": {
                                    "acf_summary": {"all": {"mean": 0.02}},
                                    "factorized_acf": {
                                        "num_factors": 2,
                                        "explained_variance_ratio_sum": 0.3,
                                        "factors": [
                                            {
                                                "factor_index": 0,
                                                "explained_variance_ratio": 0.2,
                                                "score_std": 1.5,
                                                "acf_summary": {"all": {"mean": 0.5}},
                                            }
                                        ],
                                    },
                                }
                            },
                            "token_class_loss": {"content": {"mean_bits": 4.2}},
                            "prosody_probes": {"-1": {"inside_quote": {"accuracy": 0.8, "lift": 0.1}}},
                        }
                    ),
                    encoding="utf-8",
                )
                (results_dir / "comparison.json").write_text(
                    json.dumps(
                        {
                            "left_label": "control",
                            "right_label": "challenger",
                            "left": "a.json",
                            "right": "b.json",
                            "metrics": [
                                {"path": "mean_nll", "left": 2.5, "right": 2.3, "delta": -0.2},
                                {"path": "nll_acf.summary.all.mean", "left": 0.1, "right": 0.08, "delta": -0.02},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                db_path = root / "runtime_metrics.sqlite"
                runtime_output = iteration_dir / "runtime_state.json"
                manifest_output = iteration_dir / "runtime_metrics_manifest.json"
                summary_output = iteration_dir / "runtime_summary.md"
                cluster_output = iteration_dir / "cluster_queue_snapshot.json"
                cluster_summary = iteration_dir / "cluster_queue_summary.md"
                manifest_payload = update_index(
                    iteration_dir=iteration_dir,
                    db_path=db_path,
                    check_remote=False,
                    search_roots=[iteration_dir / "artifacts", iteration_dir / "logs", iteration_dir],
                    runtime_output=runtime_output,
                    manifest_output=manifest_output,
                    summary_output=summary_output,
                    cluster_root=root / "cluster",
                    redis_host="127.0.0.1",
                    cluster_snapshot_output=cluster_output,
                    cluster_summary_output=cluster_summary,
                    include_cluster_snapshot=True,
                )

                self.assertTrue(runtime_output.exists())
                self.assertTrue(manifest_output.exists())
                self.assertTrue(summary_output.exists())
                self.assertTrue(cluster_output.exists())
                self.assertTrue(cluster_summary.exists())
                self.assertTrue(db_path.exists())

                runtime_payload = json.loads(runtime_output.read_text(encoding="utf-8"))
                run = runtime_payload["runs"]["combined-control-1h"]
                self.assertEqual(run["host"], "mini01")
                self.assertEqual(run["job_id"], "job_1")
                self.assertEqual(run["queue_status"], "stopped")
                self.assertEqual(run["cluster_job_status"], "running")
                self.assertEqual(run["cluster_host_status"], "BUSY")
                self.assertEqual(run["cluster_lock_owner"], "job_1")
                self.assertAlmostEqual(run["latest_train"]["train_loss"], 3.0)
                self.assertAlmostEqual(run["latest_val"]["val_bpb"], 1.65)
                self.assertAlmostEqual(run["final_metrics"]["val_bpb"], 1.65)

                self.assertEqual(manifest_payload["run_count"], 1)
                self.assertTrue(manifest_payload["scalar_point_count"] >= 5)
                self.assertEqual(manifest_payload["runs"]["combined-control-1h"]["cluster_job_status"], "running")
                self.assertTrue(manifest_payload["analysis_metric_count"] >= 8)
                self.assertEqual(manifest_payload["analysis_source_count"], 2)

                rows = search_metric_rows(
                    db_path=db_path,
                    metric="val_bpb",
                    phase="val",
                    run_like="control",
                    iteration_like="iter",
                    host="mini01",
                    limit=5,
                    order="asc",
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["run_slug"], "combined-control-1h")
                self.assertAlmostEqual(rows[0]["value"], 1.65)

                host_rows = list_cluster_hosts_rows(db_path=db_path, status="BUSY", host_like="mini", limit=5)
                self.assertEqual(len(host_rows), 1)
                self.assertEqual(host_rows[0]["host"], "mini01")

                job_rows = list_cluster_jobs_rows(db_path=db_path, status="running", host="mini01", script_like="control", limit=5)
                self.assertEqual(len(job_rows), 1)
                self.assertEqual(job_rows[0]["job_id"], "job_1")

                analysis_rows = search_analysis_metric_rows(
                    db_path=db_path,
                    metric="mean_nll",
                    iteration_like="iter",
                    subject_like="control",
                    source_like="comparison",
                    metric_like="",
                    limit=10,
                    order="asc",
                    delta_only=True,
                )
                self.assertEqual(len(analysis_rows), 1)
                self.assertAlmostEqual(analysis_rows[0]["delta_value"], -0.2)

                factor_rows = search_analysis_metric_rows(
                    db_path=db_path,
                    metric="residual_modes.argmax_embedding.factorized_acf.factor_0.acf_summary.all.mean",
                    iteration_like="iter",
                    subject_like="control",
                    source_like="control_analysis",
                    metric_like="",
                    limit=10,
                    order="desc",
                    delta_only=False,
                )
                self.assertEqual(len(factor_rows), 1)
                self.assertAlmostEqual(factor_rows[0]["value"], 0.5)

    @patch("tools.update_iteration_runtime_index.write_cluster_snapshot")
    @patch("tools.update_iteration_runtime_index.build_cluster_snapshot")
    def test_update_supports_analysis_only_iteration_without_manifest(self, build_snapshot_mock, write_snapshot_mock) -> None:
        build_snapshot_mock.return_value = {
            "generated_at_utc": "2026-04-05T00:00:00Z",
            "cluster_root": "/tmp/cluster",
            "redis_host": "127.0.0.1",
            "status_source": "test",
            "hosts": [],
            "locks": [],
            "jobs": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iteration_dir = root / "analysis_only"
            results_dir = iteration_dir / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "comparison.json").write_text(
                json.dumps(
                    {
                        "left_label": "a",
                        "right_label": "b",
                        "left": "a.json",
                        "right": "b.json",
                        "metrics": [{"path": "mean_nll", "left": 3.0, "right": 2.5, "delta": -0.5}],
                    }
                ),
                encoding="utf-8",
            )
            db_path = root / "runtime_metrics.sqlite"
            manifest_payload = update_index(
                iteration_dir=iteration_dir,
                db_path=db_path,
                check_remote=False,
                search_roots=[iteration_dir],
                runtime_output=iteration_dir / "runtime_state.json",
                manifest_output=iteration_dir / "runtime_metrics_manifest.json",
                summary_output=iteration_dir / "runtime_summary.md",
                cluster_root=root / "cluster",
                redis_host="127.0.0.1",
                cluster_snapshot_output=iteration_dir / "cluster_queue_snapshot.json",
                cluster_summary_output=iteration_dir / "cluster_queue_summary.md",
                include_cluster_snapshot=True,
            )
            self.assertEqual(manifest_payload["run_count"], 0)
            self.assertEqual(manifest_payload["analysis_source_count"], 1)
            rows = search_analysis_metric_rows(
                db_path=db_path,
                metric="mean_nll",
                iteration_like="analysis_only",
                subject_like="a__vs__b",
                source_like="comparison",
                metric_like="",
                limit=5,
                order="asc",
                delta_only=True,
            )
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["delta_value"], -0.5)


if __name__ == "__main__":
    unittest.main()
