"""Model-free checks for the strict chat-history snapshot gate."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "exl3_snapshot_gate.py"
SPEC = importlib.util.spec_from_file_location("exl3_snapshot_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class SnapshotGateTests(unittest.TestCase):
    def invoke(self, args, cached, runtime_override=None):
        requests = []

        def open_request(request, timeout):
            body = json.loads(request.data)
            requests.append((request.full_url, body))
            digest = hashlib.sha256(json.dumps(body["messages"]).encode()).hexdigest()
            runtime = {"token_sha256": digest}
            if runtime_override is not None:
                runtime = runtime_override
            return Response({
                "usage": {"prompt_tokens": 80, "completion_tokens": 2,
                          "prompt_tokens_details": {"cached_tokens": cached}},
                "tensorfold": runtime,
            })

        output, errors = io.StringIO(), io.StringIO()
        with patch.object(gate.urllib.request, "urlopen", open_request), redirect_stdout(output), redirect_stderr(errors):
            try:
                gate.main(["http://example.invalid", "--tokens", "40", "--decode", "2", *args])
                code = 0
            except AssertionError as error:
                errors.write(str(error))
                code = 1
        return code, output.getvalue(), errors.getvalue(), requests

    def test_records_cold_reference_then_requires_exact_chat_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            receipt = Path(directory) / "receipt.json"
            code, _, _, cold_requests = self.invoke(["--record-reference", "--output", str(reference)], 0)
            self.assertEqual(code, 0)
            self.assertEqual(len(cold_requests), 8)
            self.assertTrue(all(url.endswith("/v1/chat/completions") for url, _ in cold_requests))
            self.assertTrue(all(len(body["messages"]) == 3 for _, body in cold_requests))
            code, output, _, warm_requests = self.invoke(["--reference", str(reference), "--output", str(receipt)], 40)
            self.assertEqual(code, 0)
            self.assertEqual(len(warm_requests), 8)
            self.assertIn("PASS:", output)
            saved = json.loads(receipt.read_text())
            for phase in ("warm", "branch", "rewind"):
                self.assertTrue(all(0 < lane["cached_tokens"] < lane["prompt_tokens"]
                                    for lane in saved["phases"][phase]))

    def test_missing_warm_cache_hit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            self.invoke(["--record-reference", "--output", str(reference)], 0)
            code, output, errors, _ = self.invoke(["--reference", str(reference)], 0)
            self.assertEqual(code, 1)
            self.assertIn("did not restore a cached prefix", errors)

    def test_missing_or_inexact_full_digest_cannot_record_a_reference(self):
        for runtime in ({"token_sha": "a" * 12}, {"token_sha256": "a" * 12},
                        {"token_sha256": "z" * 64}, {"token_sha256": None}):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as directory:
                reference = Path(directory) / "reference.json"
                code, _, errors, _ = self.invoke(
                    ["--record-reference", "--output", str(reference)], 0, runtime)
                self.assertEqual(code, 1)
                self.assertIn("missing full output token SHA256", errors)
                self.assertFalse(reference.exists())


if __name__ == "__main__":
    unittest.main()
