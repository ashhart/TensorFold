"""Completed SSE metadata cannot substitute for observed text or reasoning in cache receipts."""
import contextlib
import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_cache
import bench_concurrent
import bench_openai
from bench_evidence import Evidence, token_equal

COUNTS = {"prompt_tokens": 8, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}
HASHES = {"token_sha": "a" * 12, "token_sha256": "a" * 64}
TOOL_CHOICES = [
    {"delta": {"role": "assistant", "content": None, "tool_calls": [
        {"index": 0, "id": "call_fixture", "type": "function", "function": {"name": "lookup", "arguments": ""}}]}},
    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"key":'}}]}},
    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"value"}'}}]}},
    {"delta": {}, "finish_reason": "tool_calls"},
]


def response(choices=(), done=True):
    events = [{"choices": [choice]} for choice in choices]
    events.append({"choices": [], "usage": COUNTS, "tensorfold": HASHES})
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (b"data: [DONE]\n\n" if done else b"")


class EmptyOutputTests(unittest.TestCase):
    def test_metadata_role_and_empty_chunks_never_supply_equality_fingerprints(self):
        for choices in [[], [{"text": ""}], [{"delta": {"role": "assistant"}}],
                        [{"delta": {"content": "", "reasoning_content": ""}}], TOOL_CHOICES]:
            with self.subTest(choices=choices):
                evidence = Evidence()
                for choice in choices:
                    evidence.add(choice)
                row = evidence.finish(COUNTS, HASHES, True)
                self.assertTrue(row["complete"])
                self.assertEqual(row["tokens"], 4)
                self.assertEqual(row["cache_state"], "cold")
                for field in ["token_sha", "token_sha256", "output_sha256"]:
                    self.assertIsNone(row[field])
                self.assertIsNone(token_equal(row, row))
                self.assertFalse(bench_cache.qualify([row]))

    def test_existing_nonempty_hashes_and_reasoning_only_replies_are_preserved(self):
        for choices, content, reasoning in [
            ([{"text": "buffered reply"}], "buffered reply", ""),
            ([{"delta": {"reasoning_content": "reason"}}], "", "reason"),
            ([{"delta": {"content": "caf"}}, {"delta": {"content": "é", "reasoning_content": "why"}}], "café", "why"),
        ]:
            evidence = Evidence()
            for choice in choices:
                evidence.add(choice)
            row = evidence.finish(COUNTS, HASHES, True)
            digest = hashlib.sha256(hashlib.sha256(content.encode()).digest() + hashlib.sha256(reasoning.encode()).digest()).hexdigest()
            self.assertEqual(row["output_sha256"], digest)
            self.assertEqual(row["token_sha256"], HASHES["token_sha256"])
            self.assertTrue(bench_cache.qualify([row]))
            self.assertIsNone(evidence.finish(COUNTS, HASHES, False)["output_sha256"])

    def test_both_sse_clients_keep_usage_without_publishing_empty_output_fingerprints(self):
        item = {"name": "fixture", "kind": "chat", "prompt": "A public fixture."}
        for client in [bench_openai, bench_concurrent]:
            for choices in [[], TOOL_CHOICES]:
                with self.subTest(client=client.__name__, choices=choices):
                    with patch.object(client.urllib.request, "urlopen", return_value=io.BytesIO(response(choices))):
                        row = client.stream("http://127.0.0.1:1", "fixture", item, 4, 0, 1234)
                    self.assertTrue(row["complete"])
                    self.assertEqual(row["tokens"], 4)
                    for field in ["output_sha256", "token_sha", "token_sha256", "decode_tps", "ttft_s"]:
                        self.assertIsNone(row[field])

    def test_mixed_reply_fingerprints_cover_text_and_reasoning_but_not_tool_calls(self):
        item = {"name": "fixture", "kind": "chat", "prompt": "A public fixture."}
        for client in [bench_openai, bench_concurrent]:
            for field in ["content", "reasoning_content"]:
                for tools in [TOOL_CHOICES, [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_other", "type": "function",
                     "function": {"name": "other", "arguments": '{"key":"different"}'}}]}}]]:
                    with self.subTest(client=client.__name__, field=field, tools=tools):
                        rows = []
                        for choices in [[{"delta": {field: "observed"}}], [{"delta": {field: "observed"}}] + tools]:
                            with patch.object(client.urllib.request, "urlopen", return_value=io.BytesIO(response(choices))):
                                rows.append(client.stream("http://127.0.0.1:1", "fixture", item, 4, 0, 1234))
                        self.assertIsNotNone(rows[0]["output_sha256"])
                        self.assertEqual(rows[0]["output_sha256"], rows[1]["output_sha256"])
                        self.assertTrue(bench_cache.qualify([bench_cache.receipt(rows[1], "first")]))
                        self.assertIsNone(rows[1]["decode_tps"])

    def test_cache_cli_rejects_empty_output_in_each_phase_and_saves_the_failed_receipt(self):
        for empty_choices, empty_at in [(choices, i) for choices in [[], TOOL_CHOICES] for i in range(4)]:
            wires = [response([{"text": "a buffered reply"}]) for _ in range(4)]
            wires[empty_at] = response(empty_choices)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "receipt.json"
                with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=[io.BytesIO(w) for w in wires]):
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        code = bench_cache.main(["http://127.0.0.1:1", "fixture", "--repeats", "1", "--streams", "2", "--output", str(output)])
                report = json.loads(output.read_text())
                self.assertEqual(report, json.loads(stdout.getvalue()))
            self.assertEqual(code, 1)
            self.assertFalse(report["qualified"])
            self.assertEqual(len(report["rows"]), 4)
            missing = [r for r in report["rows"] if r["output_sha256"] is None]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["phase"], ["first", "repeated", "concurrent", "concurrent"][empty_at])

    def test_real_http_usage_and_tool_only_fail_but_text_and_reasoning_pass(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            choices = []
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                wire = response(type(self).choices)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            for choices, expected in [([], 1), (TOOL_CHOICES, 1), ([{"text": "buffered"}], 0),
                                      ([{"delta": {"reasoning_content": "reason"}}], 0),
                                      ([{"delta": {"content": "buffered"}}] + TOOL_CHOICES, 0)]:
                Handler.choices = choices
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = bench_cache.main([f"http://127.0.0.1:{server.server_port}", "fixture", "--repeats", "1", "--streams", "2", "--require-cold"])
                self.assertEqual(code, expected)
                self.assertEqual(json.loads(stdout.getvalue())["qualified"], expected == 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
