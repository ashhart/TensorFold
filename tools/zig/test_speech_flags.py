"""Speech-only flags are refused before HTTP startup."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "zig-out/native/bin/tensorfold-native"
MESSAGE = "--silence-floor-dbfs, --silence-seconds, --silence-onset-s, --silence-onset-voiced-s and the silence-mid flags are for Drama speech: serve without them"


@unittest.skipUnless(BINARY.is_file(), "build native before running the CLI admission test")
class SpeechFlags(unittest.TestCase):
    def test_refuses_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text('{"model_type":"nemotron_h"}', encoding="utf-8")
            for flags in (("--silence-floor-dbfs=-80",), ("--silence-seconds=0",),
                          ("--silence-floor-dbfs=-80", "--silence-seconds=1"),
                          ("--silence-floor-dbfs=-80", "--silence-seconds=2", "--silence-onset-s=1",
                           "--silence-onset-voiced-s=0.5"),
                          ("--silence-floor-dbfs=-80", "--silence-seconds=1", "--silence-mid-s=0.5",
                           "--silence-mid-min-voiced-s=1")):
                with self.subTest(flags=flags):
                    result = subprocess.run(
                        [str(BINARY), "serve", folder, *flags],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                        capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(MESSAGE, result.stderr)


if __name__ == "__main__":
    unittest.main()
