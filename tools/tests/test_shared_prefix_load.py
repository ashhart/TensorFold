"""Model-free checks for shared-prefix stream completion and load summaries."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import mock_open, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared_prefix_load as load


def response(done=True, tokens=4, chunks=None):
    if chunks is None:
        chunks = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"completion_tokens": tokens, "prompt_tokens": 8}},
        ]
    wire = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    return io.BytesIO(wire + (b"data: [DONE]\n\n" if done else b""))


class StreamTests(unittest.TestCase):
    item = {"id": 0, "messages": [{"role": "user", "content": "A public fixture."}]}

    def request(self):
        return load.one("http://127.0.0.1:8080", "fixture", self.item, 4, 0, 1234, True)

    def read(self, reply):
        with patch.object(load.urllib.request, "urlopen", return_value=reply):
            return self.request()

    def test_complete_stream_retains_counts(self):
        row = self.read(response())
        self.assertIsNone(row["error"])
        self.assertEqual(row["tokens"], 4)
        self.assertEqual(row["prompt_tokens"], 8)
        self.assertEqual(row["finish"], "length")
        self.assertEqual(row["pieces"], 2)

    def test_eof_without_done_is_incomplete(self):
        for chunks in [None, []]:
            with self.subTest(empty=chunks == []):
                row = self.read(response(done=False, chunks=chunks))
                self.assertEqual(row["error"], "incomplete_stream")

    def test_done_stops_reading(self):
        wire = response().getvalue() + b"data: not-json\n\n"
        self.assertIsNone(self.read(io.BytesIO(wire))["error"])

    def test_server_error_is_preserved(self):
        error = {"message": "fixture server error"}
        for done in [False, True]:
            with self.subTest(done=done):
                row = self.read(response(done=done, chunks=[{"error": error}]))
                self.assertEqual(row["error"], str(error))

    def test_transport_error_is_preserved(self):
        with patch.object(load.urllib.request, "urlopen", side_effect=OSError("fixture connection error")):
            self.assertEqual(self.request()["error"], "OSError: fixture connection error")

    def test_summary_excludes_incomplete_stream(self):
        prompts = json.dumps({"items": [self.item, {**self.item, "id": 1}]})
        argv = ["shared_prefix_load.py", "http://127.0.0.1:8080", "fixture", "prompts.json",
                "--concurrency", "1"]
        replies = [response(tokens=4), response(done=False, tokens=100)]
        with patch("builtins.open", mock_open(read_data=prompts)):
            with patch.object(sys, "argv", argv):
                with patch.object(load.urllib.request, "urlopen", side_effect=replies):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        load.main()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["output_tokens"], 4)
        self.assertEqual(summary["prompt_tokens_total"], 8)


if __name__ == "__main__":
    unittest.main()
