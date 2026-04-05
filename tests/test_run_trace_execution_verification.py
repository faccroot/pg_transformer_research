from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.run_trace_execution_verification import executable_path


class RunTraceExecutionVerificationTests(unittest.TestCase):
    def test_executable_path_preserves_absolute_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target = tmp / "python-real"
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            link = tmp / "python"
            link.symlink_to(target)
            self.assertEqual(executable_path(str(link)), link)


if __name__ == "__main__":
    unittest.main()
