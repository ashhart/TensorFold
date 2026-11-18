"""Host checks for the gate's native-server launch, and its draft and seed checks against fake replies."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
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


def fake_post(fault):
    """``post`` for a healthy server, or one with ``fault``: undrafted, plain_drafts or unsampled (seed ignored)."""
    def post(_base, body):
        seed = body["seed"] if body["temperature"] and fault != "unsampled" else None
        shape = [body["messages"], body.get("response_format"), body.get("tool_choice"), seed]
        ids = list(hashlib.sha256(json.dumps(shape).encode()).digest()[:8])
        message = {"content": '{"n": 7}'} if "response_format" in body else {"content": "def add(a, b): ..."}
        if "tools" in body:
            message = {"content": "", "tool_calls": [{"function": {"name": "choose", "arguments": '{"n": 7}'}}]}
        drafting = (body["draft"] or fault == "plain_drafts") and fault != "undrafted"
        return {"choices": [{"finish_reason": "stop", "message": message}], "tensorfold": {"token_ids": ids},
                "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                "speculative": {"drafted": 3 * drafting, "accepted": 2 * drafting}}, 0.01
    return post


def run_gate(fault=None):
    """The gate's main() on fake servers: None when it passes, else its AssertionError's text."""
    @contextlib.contextmanager
    def serve(*_):
        yield "http://fake"
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / "tensorfold-native"
        binary.write_bytes(b"binary")
        argv = ["gate", "--baseline", str(binary), "--candidate", str(binary), "--model", "m", "--revision", "r",
                "--quantization", "q", "--gpu", "g", "--output", str(Path(directory) / "out"), "--reps", "1"]
        with patch.object(sys, "argv", argv), patch.object(gate, "serve", serve), \
                patch.object(gate, "post", fake_post(fault)), contextlib.redirect_stdout(io.StringIO()):
            try:
                gate.main()
            except AssertionError as error:
                return str(error)
    return None


class ExercisedTests(unittest.TestCase):
    def test_a_healthy_server_passes(self):
        self.assertIsNone(run_gate())

    def test_unexercised_drafts_and_seeds_fail_by_name(self):
        for fault, words in (("undrafted", "no draft accepted at t0"), ("plain_drafts", "plain cases drafted"),
                             ("unsampled", "seeds 7 and 1904 gave the same tokens")):
            with self.subTest(fault=fault):
                self.assertIn(words, run_gate(fault) or "passed")


if __name__ == "__main__":
    unittest.main()
