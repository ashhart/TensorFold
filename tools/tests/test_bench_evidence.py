"""Model-free regression tests for observed cache state and benchmark exactness evidence."""
import contextlib
import io
import http.server
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_cache
import bench_concurrent
import bench_openai
from bench_evidence import Evidence, token_equal


def usage(cached=0):
    return {"prompt_tokens": 8, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": cached}}


def response(counts=None, done=True, chunks=None, runtime=None):
    chunks = chunks if chunks is not None else [{"choices": [{"text": t}]} for t in ["H", "ell", "o", " world"]]
    chunks.append({"choices": [], "usage": counts, "tensorfold": runtime or {"token_sha": "a" * 12}})
    return io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
                      + (b"data: [DONE]\n\n" if done else b""))


class UsageTests(unittest.TestCase):
    def test_cold_and_reuse_require_affirmative_evidence(self):
        self.assertEqual(Evidence().finish(usage(0), {}, True)["cache_state"], "cold")
        self.assertEqual(Evidence().finish(usage(7), {}, True)["cache_state"], "reused")
        for value in [None, {}, {"prompt_tokens": 8}, [], "invalid"]:
            with self.subTest(value=value):
                self.assertEqual(Evidence().finish(value, {}, True)["cache_state"], "unverified")

    def test_invalid_counts_cannot_be_coerced_to_cold(self):
        for field, values in [("prompt_tokens", [None, -1, True, "8", 8.0]),
                              ("cached_tokens", [None, -1, False, "0", 0.0, 9])]:
            for value in values:
                counts = usage()
                target = counts if field == "prompt_tokens" else counts["prompt_tokens_details"]
                target[field] = value
                self.assertEqual(Evidence().finish(counts, {}, True)["cache_state"], "unverified")
        self.assertEqual(Evidence().finish(usage(), {}, False)["cache_state"], "unverified")

    def test_missing_and_malformed_hashes_never_pass_exactness(self):
        row = Evidence().finish(usage(), {}, True)
        self.assertIsNone(token_equal(row, row))
        for value in [None, True, "not-a-hash", "a" * 11]:
            self.assertIsNone(Evidence().finish(usage(), {"token_sha": value}, True)["token_sha"])
        a = Evidence().finish(usage(), {"token_sha": "a" * 12}, True)
        b = Evidence().finish(usage(), {"token_sha": "b" * 12}, True)
        self.assertTrue(token_equal(a, a))
        self.assertFalse(token_equal(a, b))
        self.assertIsNone(token_equal(a, {**a, "complete": False}))
        self.assertIsNone(token_equal(a, {**a, "error": "failure"}))

    def test_unicode_hashes_ignore_chunks_and_distinguish_reasoning(self):
        a, b, c = Evidence(), Evidence(), Evidence()
        a.add({"delta": {"content": "café 🚀", "reasoning_content": "reason"}})
        for piece in ["ca", "fé ", "🚀"]:
            b.add({"delta": {"content": piece}})
        b.add({"delta": {"reasoning_content": "reason"}})
        c.add({"delta": {"content": "reason", "reasoning_content": "café 🚀"}})
        self.assertEqual(a.finish(usage(), {}, True)["output_sha256"], b.finish(usage(), {}, True)["output_sha256"])
        self.assertNotEqual(a.finish(usage(), {}, True)["output_sha256"], c.finish(usage(), {}, True)["output_sha256"])
        self.assertIsNone(a.finish(usage(), {}, False)["output_sha256"])


class ClientTests(unittest.TestCase):
    item = {"name": "public-fixture", "kind": "completion", "prompt": "A public fixture."}

    def stream(self, module, **kwargs):
        with patch.object(module.urllib.request, "urlopen", return_value=response(**kwargs)):
            return module.stream("http://127.0.0.1:8080", "fixture", self.item, 4, 0, 1234)

    def test_both_clients_retain_actual_prompt_cache_and_output(self):
        for module in [bench_openai, bench_concurrent]:
            with self.subTest(module=module.__name__):
                row = self.stream(module, counts=usage(7))
                self.assertEqual(row["prompt_tokens"], 8)
                self.assertEqual(row["cached_tokens"], 7)
                self.assertEqual(row["cache_state"], "reused")
                self.assertEqual(row["token_sha"], "a" * 12)
                self.assertEqual(len(row["output_sha256"]), 64)

    def test_missing_usage_and_empty_replies_are_unmeasured(self):
        for module in [bench_openai, bench_concurrent]:
            row = self.stream(module, counts=None, chunks=[])
            self.assertIsNone(row["prompt_tokens"])
            self.assertIsNone(row["cached_tokens"])
            self.assertIsNone(row["tokens"])
            self.assertIsNone(row["ttft_s"])
            self.assertIsNone(row["decode_tps"])
            self.assertEqual(row["cache_state"], "unverified")

    def test_truncated_stream_cannot_supply_rate_or_exactness(self):
        for module in [bench_openai, bench_concurrent]:
            row = self.stream(module, counts=usage(), done=False)
            self.assertFalse(row["complete"])
            self.assertIsNone(row["decode_tps"])
            self.assertIsNone(row["token_sha"])
            self.assertEqual(row["cache_state"], "unverified")

    def test_stream_errors_are_not_successful_empty_generations(self):
        row = self.stream(bench_concurrent, chunks=[{"error": {"message": "private.example"}}])
        self.assertEqual(row["error"], "RuntimeError")
        self.assertNotIn("private.example", json.dumps(row))
        self.assertFalse(row["complete"])

    def test_connection_error_details_cannot_leak_into_receipt(self):
        with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=OSError("private.example")):
            row = bench_concurrent.stream("http://127.0.0.1:8080", "fixture", self.item, 4, 0, 1)
        self.assertEqual(row["error"], "OSError")
        self.assertNotIn("private.example", json.dumps(bench_cache.receipt(row, "first")))

    def test_concurrent_requests_retain_evidence_in_request_order(self):
        with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=lambda *a, **k: response(usage(0))):
            rows = bench_concurrent.together("http://127.0.0.1:8080", "fixture", [(self.item, 1)] * 3, 4, 0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["cache_state"] == "cold" for row in rows))
        self.assertTrue(all(token_equal(rows[0], row) for row in rows))

    def test_concurrent_cli_preserves_unverified_exactness_and_per_request_evidence(self):
        argv = ["bench_concurrent.py", "http://127.0.0.1:8080", "fixture", "--levels", "1", "--reps", "1",
                "--temperatures", "0", "--alone", "--serial"]
        with patch.object(sys, "argv", argv):
            with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=lambda *a, **k: response(usage(), runtime={"other": 1})):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    bench_concurrent.main()
        cells = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertTrue(all(cell["alone"]["equal"] == 0 and cell["alone"]["failed"] == 1 for cell in cells))
        self.assertTrue(all(cell["serial"]["equal"] == 0 and cell["serial"]["failed"] == 1 for cell in cells))

    def test_cli_missing_usage_and_empty_output_do_not_fabricate_timing(self):
        for module in [bench_concurrent, bench_openai]:
            argv = [module.__name__, "http://127.0.0.1:8080", "fixture", "--reps", "1", "--temperatures", "0"]
            if module is bench_concurrent:
                argv.extend(["--levels", "1"])
            with patch.object(sys, "argv", argv):
                with patch.object(module.urllib.request, "urlopen", side_effect=lambda *a, **k: response(None, chunks=[])):
                    with contextlib.redirect_stdout(io.StringIO()) as out:
                        module.main()
            cells = [json.loads(line) for line in out.getvalue().splitlines()]
            key = "ttft_s_max" if module is bench_concurrent else "ttft_s_median"
            self.assertTrue(all(cell[key] is None for cell in cells))


class QualificationTests(unittest.TestCase):
    def row(self, phase="first", cached=0):
        return bench_cache.receipt(Evidence().finish(usage(cached), {}, True), phase)

    def test_claims_fail_closed_on_missing_evidence_and_wrong_token_count(self):
        row = self.row()
        self.assertTrue(bench_cache.qualify([row], 8, True))
        self.assertFalse(bench_cache.qualify([row], 9, True))
        self.assertFalse(bench_cache.qualify([{**row, "cache_state": "unverified"}], 8, True))
        self.assertFalse(bench_cache.qualify([self.row(cached=1)], 8, True))
        self.assertFalse(bench_cache.qualify([row, self.row("repeated")], require_reuse=True))
        self.assertTrue(bench_cache.qualify([row, self.row("repeated", 7)], require_reuse=True))
        self.assertFalse(bench_cache.qualify([{**row, "equal_first_output": False}]))
        self.assertFalse(bench_cache.qualify([{**row, "tokens": 0}]))

    def test_receipt_excludes_all_unlisted_input_and_output_fields(self):
        row = {**Evidence().finish(usage(), {}, True), "prompt": "private", "model": "private", "base": "private",
               "text": "private", "runtime": {"private": "private"}}
        self.assertNotIn("private", json.dumps(bench_cache.receipt(row, "first")))

    def test_cli_runs_first_repeated_and_concurrent_with_no_warmup(self):
        with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=lambda *a, **k: response(usage(0))) as http:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = bench_cache.main(["http://127.0.0.1:8080", "private-alias", "--require-cold", "--repeats", "1"])
        report = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(http.call_count, 4)
        self.assertEqual([r["phase"] for r in report["rows"]], ["first", "repeated", "concurrent", "concurrent"])
        self.assertNotIn("private-alias", out.getvalue())

    def test_cli_rejects_missing_cache_evidence_for_cold_claim(self):
        with patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=lambda *a, **k: response(None)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = bench_cache.main(["http://127.0.0.1:8080", "fixture", "--require-cold", "--repeats", "1"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out.getvalue())["qualified"])


class HTTPTests(unittest.TestCase):
    def test_real_http_first_repeated_and_concurrent_cache_evidence(self):
        seen = []
        lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    cached = 0 if not seen else 8
                    seen.append(body)
                wire = response(usage(cached)).getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = bench_cache.main(["http://127.0.0.1:" + str(server.server_port), "fixture",
                                         "--repeats", "1", "--streams", "3", "--expected-prompt-tokens", "8",
                                         "--require-cold", "--require-reuse"])
            report = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(len(seen), 5)
            self.assertTrue(all(body == seen[0] for body in seen))
            self.assertEqual(report["rows"][0]["cache_state"], "cold")
            self.assertTrue(all(row["cache_state"] == "reused" for row in report["rows"][1:]))
            self.assertTrue(all(row["equal_first_tokens"] for row in report["rows"][1:]))
            self.assertNotIn("127.0.0.1", out.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
