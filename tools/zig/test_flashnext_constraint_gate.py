"""Host checks for the gate's native-server launch."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import flashnext_constraint_gate as gate


class ServeTests(unittest.TestCase):
    def test_mac_launch_and_health_use_their_respective_backend_names(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(gate.subprocess, "Popen") as popen:
                popen.return_value.poll.return_value = None
                with patch.object(gate.urllib.request, "urlopen") as urlopen:
                    urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps({"backend": "metal"}).encode())
                    with gate.serve("tensorfold-native", "checkpoint", 2, 4096, 0, Path(directory), "candidate") as base:
                        self.assertTrue(base.startswith("http://127.0.0.1:"))
                        command = popen.call_args.args[0]
                        self.assertEqual(command[command.index("--backend") + 1], "mlx")
                        self.assertEqual(command[command.index("--parallel") + 1], "2")
                        self.assertNotIn("--ple-on-ssd", command)
                popen.return_value.terminate.assert_called_once()
                popen.return_value.wait.assert_called_once_with(timeout=60)

    def test_non_metal_health_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(gate.subprocess, "Popen") as popen:
                popen.return_value.poll.return_value = None
                with patch.object(gate.urllib.request, "urlopen") as urlopen:
                    urlopen.return_value.__enter__.return_value = io.BytesIO(b'{"backend":"cuda"}')
                    with self.assertRaisesRegex(AssertionError, "gate requires Metal"):
                        with gate.serve("tensorfold-native", "checkpoint", 1, 4096, 0, Path(directory), "candidate"):
                            self.fail("non-Metal server was accepted")
                popen.return_value.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
