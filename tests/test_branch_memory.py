import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import tools.branch_memory as bm


class BranchMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        self._old_paths = {
            "ROOT": bm.ROOT,
            "RESEARCH_DIR": bm.RESEARCH_DIR,
            "PROJECT_WIDE_DIR": bm.PROJECT_WIDE_DIR,
            "ITERATIONS_GENERATED_DIR": bm.ITERATIONS_GENERATED_DIR,
            "STORE_DIR": bm.STORE_DIR,
            "NODES_DIR": bm.NODES_DIR,
            "DERIVED_DIR": bm.DERIVED_DIR,
            "DB_PATH": bm.DB_PATH,
            "VECTOR_INDEX_PATH": bm.VECTOR_INDEX_PATH,
        }
        bm.ROOT = root
        bm.RESEARCH_DIR = root / "research"
        bm.PROJECT_WIDE_DIR = bm.RESEARCH_DIR / "project_wide"
        bm.ITERATIONS_GENERATED_DIR = bm.RESEARCH_DIR / "iterations" / "generated"
        bm.STORE_DIR = bm.RESEARCH_DIR / "branch_memory"
        bm.NODES_DIR = bm.STORE_DIR / "nodes"
        bm.DERIVED_DIR = bm.STORE_DIR / "derived"
        bm.DB_PATH = bm.DERIVED_DIR / "index.sqlite"
        bm.VECTOR_INDEX_PATH = bm.DERIVED_DIR / "vector_index.json"
        bm.PROJECT_WIDE_DIR.mkdir(parents=True, exist_ok=True)
        bm.ITERATIONS_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for key, value in self._old_paths.items():
            setattr(bm, key, value)
        self._tmpdir.cleanup()

    def _run_cmd(self, func, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            func(argparse.Namespace(**kwargs))

    def test_link_node_adds_parent_ids_and_parent_edges(self) -> None:
        self._run_cmd(
            bm.cmd_create_node,
            node_id="manual_parent",
            title="Parent Node",
            node_type="branch",
            lane="general",
            status="active",
            priority=None,
            budget=None,
            summary="Parent summary",
            notes="# Notes\n",
            parent_id=[],
            tag=["parent"],
        )
        self._run_cmd(
            bm.cmd_create_node,
            node_id="manual_child",
            title="Child Node",
            node_type="task",
            lane="general",
            status="proposed",
            priority=None,
            budget=None,
            summary="Child summary",
            notes="# Notes\n",
            parent_id=[],
            tag=[],
        )
        self._run_cmd(
            bm.cmd_link_node,
            node_id="manual_child",
            parent_id=["manual_parent"],
            tag=["bridge"],
        )

        payload = bm.load_node_payload("manual_child")
        self.assertEqual(payload["parent_ids"], ["manual_parent"])
        self.assertIn("bridge", payload["tags"])

        conn = bm.connect_db()
        try:
            rows = conn.execute(
                """
                SELECT src_node_id, dst_node_id, kind, label
                FROM edges
                WHERE src_node_id = ?
                """,
                ("manual_child",),
            ).fetchall()
        finally:
            conn.close()
        self.assertTrue(
            any(
                row["kind"] == "parent_ref"
                and row["dst_node_id"] == "manual_parent"
                and row["label"] == "manual_parent"
                for row in rows
            )
        )

    def test_create_merge_node_preserves_parents_and_source_paths(self) -> None:
        note_a = bm.PROJECT_WIDE_DIR / "lane_a.md"
        note_b = bm.PROJECT_WIDE_DIR / "lane_b.md"
        note_a.write_text("# Lane A\n\nAlpha branch.\n", encoding="utf-8")
        note_b.write_text("# Lane B\n\nBeta branch.\n", encoding="utf-8")
        self._run_cmd(bm.cmd_ingest)

        node_a = bm.build_note_bundle(note_a).node_id
        node_b = bm.build_note_bundle(note_b).node_id
        self._run_cmd(
            bm.cmd_create_merge_node,
            node_id="manual_merge",
            title="Merged Bridge",
            node_type="branch",
            lane="general",
            status="active",
            priority=70.0,
            budget=None,
            summary="Compaction node",
            notes="# Notes\n\nMerged.\n",
            from_node_id=[node_a, node_b],
            tag=["bridge"],
        )

        payload = bm.load_node_payload("manual_merge")
        self.assertEqual(payload["parent_ids"], [node_a, node_b])
        self.assertTrue(payload["metadata"]["compaction"])
        self.assertEqual(payload["metadata"]["merge_of"], [node_a, node_b])
        self.assertIn("bridge", payload["tags"])
        self.assertIn("merge", payload["tags"])
        self.assertIn("compaction", payload["tags"])
        self.assertEqual(
            sorted(payload["source_paths"]),
            sorted([bm.relpath(note_a), bm.relpath(note_b)]),
        )
        self.assertEqual(
            sorted(attachment["kind"] for attachment in payload["attachments"]),
            ["merge_source", "merge_source"],
        )

        conn = bm.connect_db()
        try:
            rows = conn.execute(
                """
                SELECT dst_node_id, kind
                FROM edges
                WHERE src_node_id = ?
                ORDER BY dst_node_id
                """,
                ("manual_merge",),
            ).fetchall()
        finally:
            conn.close()
        parent_refs = [(row["dst_node_id"], row["kind"]) for row in rows if row["kind"] == "parent_ref"]
        self.assertEqual(parent_refs, [(node_a, "parent_ref"), (node_b, "parent_ref")])

    def test_manual_status_override_survives_reingest_for_note_nodes(self) -> None:
        note_path = bm.PROJECT_WIDE_DIR / "residual_lane.md"
        note_path.write_text("# Residual Lane\n\nActive by default.\n", encoding="utf-8")
        self._run_cmd(bm.cmd_ingest)
        node_id = bm.build_note_bundle(note_path).node_id

        self._run_cmd(
            bm.cmd_set_status,
            node_id=node_id,
            status="deferred",
        )
        self._run_cmd(bm.cmd_ingest)

        payload = bm.load_node_payload(node_id)
        self.assertEqual(payload["status"], "deferred")
        self.assertEqual(payload["metadata"]["manual_status_override"], "deferred")


if __name__ == "__main__":
    unittest.main()
