from __future__ import annotations

import unittest

from tools.representation_learning.schemas import DisagreementProbeSet
from tools.representation_learning.verify_probe_outcomes import verify_probe_set


class VerifyProbeOutcomesTests(unittest.TestCase):
    def test_verifies_with_chunk_loss_margin(self) -> None:
        probe_set = DisagreementProbeSet(
            source_models=["m1", "m2"],
            probes=[
                {
                    "probe_id": "p1",
                    "chunk_id": "c1",
                    "probe_type": "directional_divergence",
                    "loss_by_model": {"m1": 0.2, "m2": 0.5},
                    "source_models": ["m1", "m2"],
                },
                {
                    "probe_id": "p2",
                    "chunk_id": "c2",
                    "probe_type": "directional_divergence",
                    "loss_by_model": {"m1": 0.2, "m2": 0.21},
                    "source_models": ["m1", "m2"],
                },
            ],
        )
        table = verify_probe_set(probe_set, min_verification_confidence=0.1)
        self.assertEqual(len(table.entries), 2)
        self.assertEqual(table.entries[0]["verified_winner_model"], "m1")
        self.assertTrue(table.entries[0]["verified"])
        self.assertFalse(table.entries[1]["verified"])
        self.assertEqual(table.metadata["verified_count"], 1)


if __name__ == "__main__":
    unittest.main()
