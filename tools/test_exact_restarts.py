"""tools/exact_restarts.py low-level checks against a stub server scripting its own token_ids."""
import json
import sys
import threading
import urllib.error
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import exact_restarts as er  # noqa: E402

BASE_IDS = list(range(100, 116))
PROMPT = {"name": "p", "kind": "completion", "prompt": "x"}


class Stub(BaseHTTPRequestHandler):
    ids = []

    def do_POST(self):  # noqa: N802 (the library's name)
        self.rfile.read(int(self.headers["Content-Length"]))
        data = json.dumps({"tensorfold": {"token_ids": Stub.ids}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


class ExactRestartTests(unittest.TestCase):
    def setUp(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), Stub)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def collect(self):
        return er.collect(self.base, [PROMPT], 16)

    def test_identical_runs_report_no_flip(self):
        Stub.ids = BASE_IDS
        runs = {"a": self.collect(), "b": self.collect()}
        part = er.compare(["p"], runs)
        self.assertEqual(part["flips"], 0)
        self.assertIsNone(part["rows"][0]["first_diff"])

    def test_one_token_flip_is_found_at_its_position(self):
        for pos in (0, 7, 15):
            with self.subTest(pos=pos):
                Stub.ids = BASE_IDS
                first = self.collect()
                flipped = BASE_IDS.copy()
                flipped[pos] += 1000
                Stub.ids = flipped
                part = er.compare(["p"], {"a": first, "b": self.collect()})
                self.assertEqual(part["flips"], 1)
                self.assertEqual(part["rows"][0]["first_diff"], pos)

    def test_shorter_run_flips_at_the_first_missing_position(self):
        Stub.ids = BASE_IDS
        first = self.collect()
        Stub.ids = BASE_IDS[:9]
        part = er.compare(["p"], {"a": first, "b": self.collect()})
        self.assertEqual(part["rows"][0]["first_diff"], 9)

    def test_server_that_fails_to_start_is_named(self):
        bad = er.Server([sys.executable, "-c", "import sys; sys.exit(3)"], port=1)
        with self.assertRaisesRegex(er.ServerFailed, "exited 3"):
            bad.__enter__()
        bad.stop()

    def test_request_ids_reads_the_tensorfold_block(self):
        Stub.ids = BASE_IDS[:4]
        self.assertEqual(er.request_ids(self.base, PROMPT, 4), BASE_IDS[:4])

    def test_a_busy_peer_error_fails_the_run(self):
        calls = []

        def fake(base, item, tokens, api_key=None, timeout=600):
            calls.append(1)
            if len(calls) > 1:
                raise urllib.error.HTTPError(base, 500, "peer failed", {}, None)
            return BASE_IDS

        with patch.object(er, "request_ids", fake):
            with self.assertRaisesRegex(urllib.error.HTTPError, "peer failed"):
                er.collect("http://127.0.0.1:1", [PROMPT], 16, busy=3)

    def test_a_busy_peer_with_different_tokens_fails_the_run(self):
        calls = []

        def fake(base, item, tokens, api_key=None, timeout=600):
            calls.append(1)
            return BASE_IDS if len(calls) == 1 else BASE_IDS[:3] + [999] + BASE_IDS[4:]

        with patch.object(er, "request_ids", fake):
            with self.assertRaisesRegex(RuntimeError, "busy peer"):
                er.collect("http://127.0.0.1:1", [PROMPT], 16, busy=3)

    def test_fresh_server_disables_prompt_cache(self):
        argv = er.serve_argv("binary", "model", 8081, 8, "draft", cache_disabled=True)
        self.assertEqual(argv[-2:], ["--prompt-cache-gib", "0"])
        self.assertIn("--parallel", argv)
        self.assertIn("--drafter", argv)


if __name__ == "__main__":
    unittest.main()
