from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from tools.recover_iteration_cluster_artifacts import RECOVER_SUFFIXES, ssh_find_remote_path


class RecoverIterationClusterArtifactsTests(unittest.TestCase):
    def test_recover_suffixes_include_trace_pretrain_outputs(self) -> None:
        self.assertIn("_trace_pretrain_model.npz", RECOVER_SUFFIXES)
        self.assertIn("_hardmax_controller_init.npz", RECOVER_SUFFIXES)
        self.assertIn(".summary.json", RECOVER_SUFFIXES)

    @mock.patch("tools.recover_iteration_cluster_artifacts.subprocess.run")
    def test_ssh_find_remote_path_uses_integer_connect_timeout(self, run_mock: mock.Mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="/Users/test/jobs/job_123/logs/run_trace_pretrain_model.npz\n",
            stderr="",
        )
        out = ssh_find_remote_path("mini10", "run_trace_pretrain_model.npz", timeout=2.0)
        self.assertEqual(out, "/Users/test/jobs/job_123/logs/run_trace_pretrain_model.npz")
        cmd = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args.args[0]
        self.assertIn("ConnectTimeout=2", cmd)
        self.assertNotIn("ConnectTimeout=2.000", cmd)

    @mock.patch("tools.recover_iteration_cluster_artifacts.subprocess.run")
    def test_ssh_find_remote_path_rounds_up_subsecond_timeout(self, run_mock: mock.Mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        out = ssh_find_remote_path("mini10", "run_trace_pretrain_model.npz", timeout=0.1)
        self.assertEqual(out, "")
        cmd = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args.args[0]
        self.assertIn("ConnectTimeout=1", cmd)


if __name__ == "__main__":
    unittest.main()
