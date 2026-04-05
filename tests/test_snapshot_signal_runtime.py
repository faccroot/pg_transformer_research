from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from snapshot_signal_runtime import StudentSnapshotRuntime
from student_manager_worker import build_decision


class SnapshotSignalRuntimeTests(unittest.TestCase):
    def test_helper_status_and_latest_proposal_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = StudentSnapshotRuntime(Path(tmpdir), run_id="run123")
            runtime.write_helper_status(
                "teacher_hidden_worker",
                {
                    "state": "active",
                    "cache_entries": 42,
                    "last_written": 8,
                },
            )
            runtime.append_helper_proposal(
                "teacher_hidden_worker",
                {
                    "proposal_id": "p1",
                    "kind": "teacher_hidden_cache_batch",
                    "suggested_distill_weight_mult": 1.05,
                },
            )
            runtime.append_helper_proposal(
                "teacher_hidden_worker",
                {
                    "proposal_id": "p2",
                    "kind": "teacher_hidden_cache_batch",
                    "suggested_distill_weight_mult": 1.20,
                },
            )

            status = runtime.read_helper_status("teacher_hidden_worker")
            proposal = runtime.read_latest_helper_proposal("teacher_hidden_worker")
            all_statuses = runtime.read_all_helper_statuses()

            self.assertIsNotNone(status)
            self.assertEqual(status["state"], "active")
            self.assertEqual(int(status["cache_entries"]), 42)
            self.assertEqual(status["helper_name"], "teacher_hidden_worker")
            self.assertIsNotNone(proposal)
            self.assertEqual(proposal["proposal_id"], "p2")
            self.assertAlmostEqual(float(proposal["suggested_distill_weight_mult"]), 1.20)
            self.assertIn("teacher_hidden_worker", all_statuses)

    def test_manager_rule_uses_helper_status_and_proposal(self) -> None:
        heartbeat = {
            "step": 128,
            "train_loss": 3.0,
            "replay_cached_examples": 16,
            "controller_metrics": {
                "teacher_disagree_frac": 0.0,
                "teacher_hidden_cache_hit_frac": 0.60,
            },
        }
        snapshot_meta = {"step": 120}
        helper_statuses = {
            "teacher_hidden_worker": {
                "helper_name": "teacher_hidden_worker",
                "state": "active",
                "cache_entries": 128,
                "last_written": 12,
                "elapsed_ms": 800.0,
            }
        }
        helper_proposals = {
            "teacher_hidden_worker": {
                "proposal_id": "teacher_hidden_worker-step0000128",
                "suggested_distill_weight_mult": 1.20,
            }
        }

        decision = build_decision(
            heartbeat,
            snapshot_meta,
            source="rule_manager",
            policy="rule_based",
            helper_statuses=helper_statuses,
            helper_proposals=helper_proposals,
        )

        self.assertGreaterEqual(float(decision["distill_weight_mult"]), 1.20)
        self.assertIn("hidden_helper_state=active", str(decision["note"]))
        self.assertIn("hidden_cache_hit=0.600", str(decision["note"]))


if __name__ == "__main__":
    unittest.main()
