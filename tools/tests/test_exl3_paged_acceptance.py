"""Model-free acceptance harness checks using a local HTTP server."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "exl3_paged_acceptance.py"
SPEC = importlib.util.spec_from_file_location("exl3_paged_acceptance", SCRIPT)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def reply(body):
    digest = hashlib.sha256(json.dumps(body["prompt"]).encode()).hexdigest()
    return {
        "id": "fake-response", "choices": [{"text": "fake output", "finish_reason": "length"}],
        "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"],
                  "prompt_tokens_details": {"cached_tokens": 0}},
        "tensorfold": {"token_sha256": digest, "token_sha": digest[:12]},
    }


@contextmanager
def server(transform=lambda body, payload: payload, status=200, raw=None, observe=None):
    requests = []
    lock = threading.Lock()
    simultaneous = threading.Barrier(2)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append({"body": body, "request_id": self.headers.get("X-Request-ID"),
                                 "path": self.path})
            if "branch-" in self.headers.get("X-Request-ID", ""):
                try:
                    simultaneous.wait(timeout=3)
                except threading.BrokenBarrierError:
                    pass
            payload = transform(body, reply(body))
            if observe is not None:
                observe(self.headers.get("X-Request-ID"), body)
            encoded = json.dumps(payload).encode() if raw is None else raw
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Request-ID", "fake-http-request")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture_a = Path(self.directory.name) / "a.bin"
        self.fixture_b = Path(self.directory.name) / "b.bin"
        self.fixture_a.write_bytes(struct.pack("<5I", 11, 12, 13, 14, 15))
        self.fixture_b.write_bytes(struct.pack("<4I", 11, 12, 90, 91))

    def invoke(self, base, *extra):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = harness.main([
                str(self.fixture_a), str(self.fixture_b), "--base-url", base, "--model", "fake",
                "--context-length", "4", "--completion-tokens", "3", *extra,
            ])
        return code, json.loads(output.getvalue()), errors.getvalue()

    def test_exact_concurrent_requests_and_receipt(self):
        output = Path(self.directory.name) / "receipt.json"
        with server() as (base, requests):
            code, report, errors = self.invoke(base, "--output", str(output))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual(len(requests), 2)
        self.assertEqual({tuple(item["body"]["prompt"]) for item in requests}, {(11, 12, 13, 14), (11, 12, 90, 91)})
        self.assertTrue(all(item["path"] == "/v1/completions" for item in requests))
        self.assertTrue(all(item["body"]["stream"] is False for item in requests))
        self.assertEqual(report["fixtures"][0]["source_tokens"], 5)
        self.assertTrue(report["fixtures"][0]["selected_prefix"])
        self.assertEqual(report["fixtures"][0]["prompt_u32le_sha256"],
                         hashlib.sha256(struct.pack("<4I", 11, 12, 13, 14)).hexdigest())
        self.assertTrue(report["http_overlap"]["observed"])
        self.assertGreater(report["http_overlap"]["seconds"], 0)
        self.assertFalse(report["server_lane_residency_proven"])
        self.assertFalse(report["snapshot_restore_proven"])
        self.assertIn("HTTP overlap alone", errors)
        self.assertGreater(len(report["correlation_evidence_needed"]), 0)
        for row in report["requests"]:
            self.assertEqual(row["prompt_tokens"], 4)
            self.assertEqual(row["completion_tokens"], 3)
            self.assertTrue(row["complete"])
            self.assertEqual(row["server_response_id"], "fake-response")
            self.assertEqual(row["response_request_id"], "fake-http-request")
            self.assertEqual(len(row["token_sha256"]), 64)
            self.assertLess(row["start"]["monotonic_ns"], row["end"]["monotonic_ns"])
            self.assertTrue(row["start"]["utc"].endswith("+00:00"))
            self.assertEqual(row["tensorfold"]["token_sha256"], row["token_sha256"])

    def test_warm_shared_prefix_precedes_concurrent_branches(self):
        warmed = threading.Event()

        def reuse(body, payload):
            if len(body["prompt"]) == 2:
                warmed.set()
            elif warmed.is_set():
                payload["usage"]["prompt_tokens_details"]["cached_tokens"] = 2
            return payload

        with server(reuse) as (base, requests):
            code, report, _ = self.invoke(base, "--warm-prefix-tokens", "2")
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]["body"]["prompt"], [11, 12])
        self.assertEqual(requests[0]["body"]["max_tokens"], 1)
        self.assertTrue(all(row["cached_tokens"] == 2 for row in report["requests"]))
        self.assertLessEqual(report["warmup"]["end"]["monotonic_ns"],
                             min(row["start"]["monotonic_ns"] for row in report["requests"]))

    def test_warmup_failure_stops_dependent_requests(self):
        def incomplete(body, payload):
            payload["choices"][0]["finish_reason"] = None
            return payload

        with server(incomplete) as (base, requests):
            code, report, _ = self.invoke(base, "--warm-prefix-tokens", "2")
        self.assertEqual(code, 1)
        self.assertEqual(len(requests), 1)
        self.assertFalse(report["warmup"]["complete"])
        self.assertEqual(report["requests"], [])

    def test_required_reuse_fails_when_missing_zero_or_invalid(self):
        for cached in (None, 0, True, "2", 2.0, -1, 5):
            with self.subTest(cached=cached):
                def change(body, payload):
                    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
                    return payload

                with server(change) as (base, _):
                    code, report, _ = self.invoke(base, "--require-cache-reuse")
                self.assertEqual(code, 1)
                self.assertTrue(all(not row["cache_reuse_observed"] for row in report["requests"]))

    def test_warm_phase_implicitly_requires_reuse(self):
        with server() as (base, _):
            code, report, _ = self.invoke(base, "--warm-prefix-tokens", "2")
        self.assertEqual(code, 1)
        self.assertTrue(report["configuration"]["require_cache_reuse"])

    def test_usage_counts_must_be_exact_non_boolean_integers(self):
        for field, value in (("prompt_tokens", 3), ("prompt_tokens", None), ("prompt_tokens", True),
                             ("prompt_tokens", 4.0), ("completion_tokens", 2), ("completion_tokens", None)):
            with self.subTest(field=field, value=value):
                def change(body, payload):
                    payload["usage"][field] = value
                    return payload

                with server(change) as (base, _):
                    code, report, _ = self.invoke(base)
                self.assertEqual(code, 1)
                self.assertFalse(report["qualified"])

    def test_one_bad_branch_fails_pair(self):
        def change(body, payload):
            if body["prompt"][-1] == 91:
                payload["usage"]["prompt_tokens"] = 3
            return payload

        with server(change) as (base, _):
            code, report, _ = self.invoke(base)
        self.assertEqual(code, 1)
        self.assertTrue(report["requests"][0]["qualified"])
        self.assertFalse(report["requests"][1]["qualified"])

    def test_server_errors_incomplete_and_bad_hashes_fail(self):
        changes = [
            {"error": {"message": "pager overflow"}},
            {"choices": [{"finish_reason": None}]},
            {"choices": [{"finish_reason": "error"}]},
            {"choices": []}, {"choices": [None]}, {"complete": False}, {"status": "incomplete"},
            {"tensorfold": {"complete": False, "token_sha256": "a" * 64}},
            {"tensorfold": {}}, {"tensorfold": {"token_sha256": "not-a-hash"}},
            {"tensorfold": {"token_sha256": "a" * 64, "token_sha": "b" * 12}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                def change(body, payload):
                    payload.update(copy.deepcopy(changed))
                    return payload

                with server(change) as (base, _):
                    code, report, _ = self.invoke(base)
                self.assertEqual(code, 1)
                self.assertTrue(all(row["errors"] for row in report["requests"]))

    def test_http_error_and_invalid_or_truncated_json_fail(self):
        for status, raw in ((500, None), (200, b'{"choices":'), (200, b'[]'),
                            (200, b'{"tensorfold":{"seconds":NaN}}')):
            with self.subTest(status=status, raw=raw):
                with server(status=status, raw=raw) as (base, _):
                    code, report, _ = self.invoke(base)
                self.assertEqual(code, 1)
                self.assertTrue(all(not row["complete"] for row in report["requests"]))
                self.assertTrue(all(row["end"] is not None for row in report["requests"]))

    def test_timeout_is_a_failed_receipt_with_both_request_errors(self):
        release = threading.Event()

        def stalled(body, payload):
            release.wait(timeout=0.1)
            return payload

        with server(stalled) as (base, _):
            code, report, _ = self.invoke(base, "--timeout", "0.03")
            release.set()
        self.assertEqual(code, 1)
        self.assertTrue(all(not row["complete"] and row["errors"] for row in report["requests"]))

    def test_nonoverlapping_intervals_fail_cli_even_with_valid_responses(self):
        with server() as (base, _):
            with patch.object(harness, "overlap", return_value={
                "seconds": 0, "start_utc": None, "end_utc": None, "observed": False,
            }):
                code, report, _ = self.invoke(base)
        self.assertEqual(code, 1)
        self.assertTrue(all(row["qualified"] for row in report["requests"]))
        self.assertIn("two HTTP request intervals did not overlap", report["errors"])

    def test_short_misaligned_and_missing_fixtures_fail_before_http(self):
        for raw in (struct.pack("<3I", 1, 2, 3), b'bad', None):
            with self.subTest(raw=raw):
                if raw is None:
                    self.fixture_b.unlink()
                else:
                    self.fixture_b.write_bytes(raw)
                with server() as (base, requests):
                    code, report, _ = self.invoke(base)
                self.assertEqual(code, 2)
                self.assertEqual(requests, [])
                self.assertFalse(report["qualified"])

    def test_different_shared_prefix_fails_before_http(self):
        with server() as (base, requests):
            code, report, _ = self.invoke(base, "--warm-prefix-tokens", "3")
        self.assertEqual(code, 2)
        self.assertEqual(requests, [])
        self.assertIn("do not share", report["errors"][0])

    def test_overlap_uses_monotonic_intervals(self):
        rows = [
            {"start": {"monotonic_ns": 1, "utc": "first-start"},
             "end": {"monotonic_ns": 4, "utc": "first-end"}},
            {"start": {"monotonic_ns": 3, "utc": "second-start"},
             "end": {"monotonic_ns": 8, "utc": "second-end"}},
        ]
        self.assertEqual(harness.overlap(rows), {"seconds": 1e-9, "observed": True,
                                               "start_utc": "second-start", "end_utc": "first-end"})
        rows[1]["start"]["monotonic_ns"] = 4
        self.assertFalse(harness.overlap(rows)["observed"])
        rows[1]["start"] = None
        self.assertFalse(harness.overlap(rows)["observed"])


if __name__ == "__main__":
    unittest.main()
