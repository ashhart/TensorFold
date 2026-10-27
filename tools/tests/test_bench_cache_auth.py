"""Cache probes share the benchmark clients' explicit bearer authentication contract."""
import contextlib
import io
import http.server
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_cache
from test_bench_auth import ENV, KEY, response, serve


class CacheAuthTests(unittest.TestCase):
    def invoke(self, base, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = bench_cache.main([base, "fixture", "--repeats", "1", "--streams", "2",
                                             "--output", str(output), *extra])
                except SystemExit as exc:
                    code = exc.code
            return code, stdout.getvalue(), stderr.getvalue(), output.read_text() if output.exists() else ""

    def test_real_http_authenticates_first_repeated_and_concurrent_probes(self):
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append((self.headers.get("Authorization"), body))
                if self.headers.get("Authorization") != "Bearer " + KEY:
                    self.send_error(401)
                    return
                wire = response().getvalue()
                self.send_response(200)
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        with serve(Handler) as base, patch.dict(os.environ, {ENV: KEY}):
            code, stdout, stderr, report = self.invoke(base, ("--api-key-env", ENV))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(auth == "Bearer " + KEY and KEY.encode() not in body for auth, body in seen))
        self.assertEqual([r["phase"] for r in json.loads(report)["rows"]],
                         ["first", "repeated", "concurrent", "concurrent"])
        self.assertNotIn(KEY, stdout + stderr + report)

    def test_echoed_credential_fingerprint_is_redacted(self):
        key = "a" * 64
        with patch.dict(os.environ, {ENV: key}):
            with patch.object(bench_cache.bench.urllib.request, "urlopen", side_effect=lambda *a, **k: response()):
                code, stdout, stderr, report = self.invoke("http://127.0.0.1:8080", ("--api-key-env", ENV))
        self.assertEqual(code, 0, stderr)
        self.assertNotIn(key, stdout + stderr + report)
        self.assertIn("[REDACTED]", report)

    def test_default_ignores_environment_credentials(self):
        with patch.dict(os.environ, {ENV: KEY, "OPENAI_API_KEY": KEY}):
            with patch.object(bench_cache.bench.urllib.request, "urlopen", side_effect=lambda *a, **k: response()) as http:
                code, _, _, _ = self.invoke("http://127.0.0.1:8080")
        self.assertEqual(code, 0)
        self.assertEqual(http.call_count, 4)
        self.assertTrue(all(call.args[0].get_header("Authorization") is None for call in http.call_args_list))

    def test_invalid_selected_credentials_fail_before_any_probe(self):
        for key in (None, "", " ", KEY + "\r\nInjected: yes", KEY + "\u00e9"):
            with self.subTest(key_present=key is not None):
                with patch.dict(os.environ, {} if key is None else {ENV: key}, clear=True):
                    with patch.object(bench_cache.bench.urllib.request, "urlopen") as http:
                        code, stdout, stderr, report = self.invoke("http://127.0.0.1:8080", ("--api-key-env", ENV))
                self.assertEqual(code, 2)
                http.assert_not_called()
                self.assertIn("API key", stderr)
                self.assertNotIn(KEY, stdout + stderr + report)


if __name__ == "__main__":
    unittest.main()
