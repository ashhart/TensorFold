"""Interrupted image sessions retain failure evidence and cannot pass as successful CLI runs."""
import contextlib
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_image_session
from test_bench_image_session import png

CLIENT = Path(bench_image_session.__file__).resolve()
KEY = "tf-image-fixture-credential"
ENV = "TENSORFOLD_TEST_IMAGE_KEY"
DONE = b"data: [DONE]\n\n"


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def reply(text="complete reply", done=True):
    wire = event({"choices": [{"delta": {"content": text}}]})
    wire += event({"choices": [], "usage": {"completion_tokens": 4,
                   "prompt_tokens_details": {"cached_tokens": 2}}})
    return wire + (DONE if done else b"")


@contextlib.contextmanager
def server(responses):
    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, wire = responses[min(len(bodies) - 1, len(responses) - 1)]
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", bodies
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=5)


class ImageSessionFailures(unittest.TestCase):
    def cli(self, responses, turns=3):
        with server(responses) as (base, bodies), tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture.png"
            image.write_bytes(png())
            output = Path(directory) / "receipt.jsonl"
            argv = [sys.executable, str(CLIENT), base, "--image", str(image), "--model", "fixture",
                    "--turns", str(turns), "--out", str(output), "--api-key-env", ENV]
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15,
                                    env={**os.environ, ENV: KEY})
            receipt = output.read_text()
        self.assertEqual(result.stdout, receipt)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn(KEY, result.stdout + result.stderr + receipt)
        return result.returncode, [json.loads(line) for line in receipt.splitlines()], bodies

    def assert_failure(self, status, wire, previous=0, error="RuntimeError"):
        responses = [(200, reply())] * previous + [(status, wire)]
        code, rows, bodies = self.cli(responses)
        self.assertEqual(code, 1)
        self.assertEqual(len(bodies), previous + 1, "A failed reply must never enter the next turn's history")
        self.assertEqual(len(rows), previous + 2)
        failed, summary = rows[-2:]
        self.assertEqual(failed, {"turn": previous + 1, "sha256": None, "completion_tokens": None,
                                 "cached_tokens": None, "ttft_s": None, "error": error})
        self.assertTrue(summary["summary"])
        self.assertEqual(summary["turns"], previous + 1)
        self.assertEqual(summary["errors"], 1)
        self.assertIsNone(summary["sha256"][-1])
        self.assertEqual(summary["completion_tokens"], 4 if previous else None)
        self.assertEqual(summary["cached_tokens"], 2 if previous else None)

    def test_clean_eof_never_qualifies_a_turn_even_with_usage_or_finish_reason(self):
        for wire in (b"", event({"choices": [{"delta": {"content": KEY}}]}), reply(KEY, done=False),
                     reply(KEY, done=False) + event({"choices": [{"delta": {}, "finish_reason": "stop"}]})):
            for previous in (0, 1):
                with self.subTest(wire=wire, previous=previous):
                    self.assert_failure(200, wire, previous)

    def test_request_and_stream_errors_exit_nonzero_and_keep_sanitized_evidence(self):
        cases = [(503, KEY.encode(), "HTTPError"),
                 (200, event({"error": {"message": KEY}}) + DONE, "RuntimeError"),
                 (200, b"data: {\n\n", "JSONDecodeError")]
        for status, wire, error in cases:
            with self.subTest(status=status, error=error):
                self.assert_failure(status, wire, previous=1, error=error)

    def test_successful_cli_keeps_exit_zero_receipt_shape_and_complete_history(self):
        code, rows, bodies = self.cli([(200, reply("first café")), (200, reply("second reply"))], turns=2)
        self.assertEqual(code, 0)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["messages"][1], {"role": "assistant", "content": "first café"})
        for index, text in enumerate(("first café", "second reply")):
            self.assertEqual(set(rows[index]), {"turn", "sha256", "completion_tokens", "cached_tokens", "ttft_s"})
            self.assertEqual(rows[index]["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(rows[-1]["errors"], 0)
        self.assertEqual(rows[-1]["completion_tokens"], 8)

    def test_terminal_marker_accepts_empty_reply_and_length_finish_without_usage(self):
        for chunks in ([], [{"choices": [{"delta": {"content": "budget reached"}, "finish_reason": "length"}]}]):
            wire = b"".join(event(chunk) for chunk in chunks) + DONE
            with self.subTest(chunks=chunks), patch.object(bench_image_session.urllib.request, "urlopen",
                                                         return_value=io.BytesIO(wire)):
                got = bench_image_session.turn("http://unused", "fixture", [], 4, 0, None, None)
            self.assertIsNone(got["completion_tokens"])
            self.assertIsNone(got["cached_tokens"])
            self.assertEqual(got["reply"], "budget reached" if chunks else "")


if __name__ == "__main__":
    unittest.main()
