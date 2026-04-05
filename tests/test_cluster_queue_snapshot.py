import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.cluster_queue_snapshot import build_cluster_snapshot, write_snapshot


def _completed(stdout: str) -> object:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, text: str) -> None:
            self.stdout = text

    return Result(stdout)


class ClusterQueueSnapshotTests(unittest.TestCase):
    def test_build_cluster_snapshot_merges_status_and_jobs(self) -> None:
        status_payload = {
            "generated_at_utc": "2026-04-05T00:00:00Z",
            "hosts": [
                {
                    "host": "mini01",
                    "status": "BUSY",
                    "load_avg": "1.23",
                    "memory": "24GB (90% free)",
                    "job_display": "LOCKED: job_1",
                    "job_status": "LOCKED: job_1",
                    "disk_used": "6% of 460Gi",
                    "lock_owner": "job_1",
                    "lock_ttl": 123,
                    "reachable": True,
                    "python_procs": 1,
                }
            ],
            "locks": [
                {"host": "mini01", "key": "cluster:lock:mini01", "owner": "job_1", "ttl": 123},
            ],
        }

        def fake_run_capture(cmd: list[str]):
            if len(cmd) == 3 and cmd[0] == "bash" and cmd[2] == "--json":
                return _completed(json.dumps(status_payload))
            if cmd[:5] == ["redis-cli", "-h", "127.0.0.1", "--raw", "KEYS"] and cmd[5] == "cluster:job:*":
                return _completed("cluster:job:job_1\n")
            if cmd[:5] == ["redis-cli", "-h", "127.0.0.1", "--raw", "HGETALL"] and cmd[5] == "cluster:job:job_1":
                return _completed(
                    "\n".join(
                        [
                            "host",
                            "mini01",
                            "script",
                            "run_control.py",
                            "started",
                            "2026-04-05T00:00:00-05:00",
                            "status",
                            "running",
                            "pid",
                            "100",
                            "remote_dir",
                            "~/jobs/job_1",
                        ]
                    )
                    + "\n"
                )
            self.fail(f"unexpected command: {cmd}")

        with tempfile.TemporaryDirectory() as tmpdir:
            cluster_root = Path(tmpdir) / "cluster"
            cluster_root.mkdir(parents=True)
            (cluster_root / "status.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            with patch("tools.cluster_queue_snapshot.run_capture", side_effect=fake_run_capture):
                snapshot = build_cluster_snapshot(cluster_root=cluster_root, redis_host="127.0.0.1")

        self.assertEqual(snapshot["status_source"], "status_sh_json")
        self.assertEqual(len(snapshot["hosts"]), 1)
        self.assertEqual(snapshot["hosts"][0]["active_job_id"], "job_1")
        self.assertEqual(snapshot["hosts"][0]["active_job_status"], "running")
        self.assertEqual(snapshot["hosts"][0]["active_script"], "run_control.py")
        self.assertEqual(len(snapshot["jobs"]), 1)
        self.assertEqual(snapshot["jobs"][0]["remote_dir"], "~/jobs/job_1")

    def test_write_snapshot_writes_json_and_summary(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-05T00:00:00Z",
            "cluster_root": "/tmp/cluster",
            "redis_host": "127.0.0.1",
            "status_source": "test",
            "hosts": [
                {
                    "host": "mini01",
                    "status": "BUSY",
                    "lock_owner": "job_1",
                    "lock_ttl": 123,
                    "active_job_id": "job_1",
                    "active_job_status": "running",
                }
            ],
            "locks": [{"host": "mini01", "key": "cluster:lock:mini01", "owner": "job_1", "ttl": 123}],
            "jobs": [{"job_id": "job_1", "status": "running"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "cluster_queue_snapshot.json"
            summary = root / "cluster_queue_summary.md"
            write_snapshot(output, summary, payload)
            self.assertTrue(output.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["redis_host"], "127.0.0.1")
            summary_text = summary.read_text(encoding="utf-8")
            self.assertIn("Host statuses:", summary_text)
            self.assertIn("Locked hosts:", summary_text)


if __name__ == "__main__":
    unittest.main()
