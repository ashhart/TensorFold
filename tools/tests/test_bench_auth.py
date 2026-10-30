"""Model-free authentication, redirect and credential-disclosure checks for benchmark clients."""
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_openai

KEY = "tf-fixture-credential-7eb2f1"
ENV = "TENSORFOLD_TEST_API_KEY"
MODULES = (bench_openai, bench_concurrent)


def response(pieces=None):
    pieces = ["a", "b", "c", "d"] if pieces is None else pieces
    chunks = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                   "tensorfold": {"token_sha": "a" * 12, "token_sha256": "a" * 64, "ignored_secret": KEY}})
    wire = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    return io.BytesIO(wire + b"data: [DONE]\n\n")


class AuthTests(unittest.TestCase):
    def cli(self, module, reply=None, key=KEY, extra=()):
        seen = []

        def open_request(request, **kwargs):
            seen.append(request)
            if isinstance(reply, Exception):
                raise reply
            return response(reply)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = [module.__name__, "http://127.0.0.1:8080", "fixture", "--reps", "1",
                    "--temperatures", "1", "--api-key-env", ENV, "--output", str(output), *extra]
            if module is bench_concurrent:
                argv.extend(["--levels", "2", "--alone", "--serial"])
            stdout, stderr = io.StringIO(), io.StringIO()
            code = 0
            with patch.dict(os.environ, {ENV: key} if key is not None else {}, clear=True):
                with patch.object(sys, "argv", argv):
                    with patch.object(module.urllib.request, "urlopen", side_effect=open_request):
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            try:
                                module.main()
                            except SystemExit as exc:
                                code = exc.code
            report = output.read_text() if output.exists() else ""
        return code, seen, stdout.getvalue(), stderr.getvalue(), report

    def test_default_requests_omit_auth_even_with_environment_key(self):
        for module in MODULES:
            with self.subTest(client=module.__name__):
                with patch.dict(os.environ, {"OPENAI_API_KEY": KEY, ENV: KEY}):
                    with patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                        module.stream("http://127.0.0.1:8080", "fixture", module.PROMPTS[0], 4, 0, 1)
                self.assertIsNone(http.call_args.args[0].get_header("Authorization"))

    def test_bearer_headers_on_completion_and_chat_requests(self):
        for module in MODULES:
            for item in module.PROMPTS:
                with self.subTest(client=module.__name__, kind=item["kind"]):
                    with patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                        module.stream("http://127.0.0.1:8080", "fixture", item, 4, 0, 1, api_key=KEY)
                    req = http.call_args.args[0]
                    self.assertEqual(req.get_header("Authorization"), "Bearer " + KEY)
                    self.assertNotIn(KEY, req.full_url)
                    self.assertNotIn(KEY.encode(), req.data)

    def test_cli_auth_reaches_warmup_alone_serial_and_concurrent_requests(self):
        for module, extra in [(bench_openai, ()), (bench_concurrent, ()),
                              (bench_concurrent, ("--stagger-ms", "1"))]:
            with self.subTest(client=module.__name__, extra=extra):
                code, seen, stdout, stderr, report = self.cli(module, extra=extra)
                self.assertEqual(code, 0, stderr)
                self.assertEqual(len(seen), 4 if module is bench_openai else 14)
                self.assertTrue(all(req.get_header("Authorization") == "Bearer " + KEY for req in seen))
                self.assertEqual({req.full_url.rsplit("/v1/", 1)[1] for req in seen},
                                 {"completions", "chat/completions"})
                if module is bench_concurrent:
                    self.assertTrue(any(json.loads(req.data).get("draft") is False for req in seen))
                for text in (stdout, stderr, report):
                    self.assertNotIn(KEY, text)
                    self.assertNotIn("Authorization", text)
                self.assertTrue(json.loads(report))

    def test_echoed_credentials_are_redacted_before_sample_truncation(self):
        for module in MODULES:
            with self.subTest(client=module.__name__):
                pieces = ["x" * 150, KEY[:10], KEY[10:], " end"]
                code, _, stdout, stderr, report = self.cli(module, reply=pieces, extra=("--label", KEY))
                self.assertEqual(code, 0, stderr)
                for text in (stdout, stderr, report):
                    self.assertNotIn(KEY, text)
                    self.assertNotIn(KEY[:10], text)
                self.assertIn("[REDACTED]", report)
                if module is bench_openai:
                    self.assertTrue(json.loads(report)[0]["sample"].startswith("x" * 150 + "[REDACTED]"))

    def test_missing_empty_and_invalid_keys_fail_before_requests(self):
        for module in MODULES:
            for key in (None, "", " ", KEY + "\r\nX-Injected: yes", KEY + "\u00e9"):
                with self.subTest(client=module.__name__, missing=key is None):
                    code, seen, stdout, stderr, report = self.cli(module, key=key)
                    self.assertEqual(code, 2)
                    self.assertEqual(seen, [])
                    self.assertIn("API key", stderr)
                    self.assertNotIn(KEY, stdout + stderr + report)

    def test_http_and_transport_errors_do_not_expose_credentials(self):
        for module in MODULES:
            for error in (urllib.error.HTTPError("http://127.0.0.1", 401, KEY, {}, None),
                          urllib.error.URLError(KEY), OSError(KEY)):
                with self.subTest(client=module.__name__, error=type(error).__name__):
                    code, seen, stdout, stderr, report = self.cli(module, reply=error)
                    self.assertTrue(seen)
                    self.assertNotIn(KEY, stdout + stderr + report)
                    if module is bench_openai:
                        self.assertEqual(code, 1)
                        self.assertIn(type(error).__name__, stderr)
                    else:
                        self.assertEqual(code, 0)
                        self.assertTrue(all(cell["failed"] == 2 for cell in json.loads(report)["cells"]))
                if isinstance(error, urllib.error.HTTPError):
                    error.close()  # an unclosed one warns with its reason text, the key here, when it is collected

    def test_stream_error_payload_does_not_expose_credentials(self):
        wire = b"data: " + json.dumps({"error": {"message": KEY}}).encode() + b"\n\n"
        for module in MODULES:
            argv = [module.__name__, "http://127.0.0.1:8080", "fixture", "--reps", "1",
                    "--temperatures", "0", "--api-key-env", ENV]
            with self.subTest(client=module.__name__):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.dict(os.environ, {ENV: KEY}):
                    with patch.object(sys, "argv", argv):
                        with patch.object(module.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(wire)):
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                try:
                                    module.main()
                                except SystemExit as exc:
                                    self.assertEqual(exc.code, 1)
                self.assertNotIn(KEY, stdout.getvalue() + stderr.getvalue())
                self.assertIn("RuntimeError", stdout.getvalue() + stderr.getvalue())


@contextlib.contextmanager
def serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class HTTPTests(unittest.TestCase):
    def test_real_http_requires_bearer_auth_for_both_clients(self):
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append((self.path, self.headers.get("Authorization"), body))
                if self.headers.get("Authorization") != "Bearer " + KEY:
                    self.send_error(401)
                    return
                wire = response().getvalue()
                self.send_response(200)
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        with serve(Handler) as base:
            for module in MODULES:
                if module is bench_openai:
                    with self.assertRaises(urllib.error.HTTPError):
                        module.stream(base, "fixture", module.PROMPTS[0], 4, 0, 1)
                else:
                    self.assertEqual(module.stream(base, "fixture", module.PROMPTS[0], 4, 0, 1)["error"], "HTTPError")
                for item in module.PROMPTS:
                    row = module.stream(base, "fixture", item, 4, 0, 1, api_key=KEY)
                    self.assertTrue(row["complete"])
                    self.assertEqual(row["tokens"], 4)
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(KEY.encode() not in body for _, _, body in seen))

    def test_redirect_does_not_forward_bearer_header(self):
        original, redirected = [], []

        class Target(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                redirected.append(self.headers.get("Authorization"))
                wire = response().getvalue()
                self.send_response(200)
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        with serve(Target) as target:
            class Redirect(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    self.rfile.read(int(self.headers["Content-Length"]))
                    original.append(self.headers.get("Authorization"))
                    self.send_response(302)
                    self.send_header("Location", target + "/redirected")
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            with serve(Redirect) as base:
                for module in MODULES:
                    row = module.stream(base, "fixture", module.PROMPTS[0], 4, 0, 1, api_key=KEY)
                    self.assertTrue(row["complete"])
        self.assertEqual(original, ["Bearer " + KEY] * 2)
        self.assertEqual(redirected, [None, None])


if __name__ == "__main__":
    unittest.main()
