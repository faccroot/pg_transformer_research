from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/home/zaytor/transformer_research/parameter-golf")


class RunCliJsonConfigTests(unittest.TestCase):
    def test_wrapper_supports_positionals_and_flag_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "echo_args.py"
            output_path = tmp_path / "out.json"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "import sys",
                        "from pathlib import Path",
                        f"Path({str(output_path)!r}).write_text(json.dumps({{'argv': sys.argv[1:], 'env': os.environ.get('TEST_ENV')}}), encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "positionals": ["first", 7, "third"],
                        "args": {
                            "alpha": "beta",
                            "toggle": True,
                            "skip_me": False,
                            "repeat": ["x", "y"],
                        },
                        "env": {"TEST_ENV": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_cli_json_config.py"),
                    str(script_path),
                    str(config_path),
                ],
                check=True,
                cwd=ROOT,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["argv"],
                ["first", "7", "third", "--alpha", "beta", "--toggle", "--repeat", "x", "--repeat", "y"],
            )
            self.assertEqual(payload["env"], "ok")


if __name__ == "__main__":
    unittest.main()
