"""The 27B pair's oracle tools import together and refuse a pair serve or another GPU family from its own /health."""
import contextlib
import http.server
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen27"))
import pair_accept  # noqa: E402
import record_oracle  # noqa: E402

PAIR_POLICY = "speed-up mode: a two-Mac pair, rank 0 answers greedy requests only"
ORACLE = {"gpu_family": "apple-m4", "cases": []}


class FakeServe(http.server.BaseHTTPRequestHandler):
    """A serve on loopback: /health answers the class's document, and every POST is counted, never answered."""

    health = {}
    posts = 0

    def log_message(self, *_):
        pass

    def do_GET(self):
        body = json.dumps(type(self).health).encode("utf-8")
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).posts += 1
        self.send_response(500)
        self.end_headers()


@contextlib.contextmanager
def serving(health):
    """A fresh FakeServe class answering `health`, its base URL, and the class to read its POST count from."""
    handler = type("Serve", (FakeServe,), {"health": health, "posts": 0})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:{}".format(server.server_address[1]), handler
    finally:
        server.shutdown()
        server.server_close()


def one_mac(device="apple-m4"):
    return {"status": "ok", "version": "1.2.7", "device": device}


class OracleTools(unittest.TestCase):
    def test_the_recorder_imports_only_names_pair_accept_defines(self):
        names = ("DEFAULT_MODEL", "load_oracle", "post", "reply_text", "request_body", "server_device", "token_sha")
        for name in names:
            self.assertTrue(hasattr(pair_accept, name), name)
        self.assertIs(record_oracle.server_device, pair_accept.server_device)

    def test_a_single_mac_serve_names_its_runtime(self):
        with serving(one_mac()) as (base, _):
            self.assertEqual("tensorfold-native 1.2.7", record_oracle.served_runtime(base, 5.0))
            self.assertIsNone(record_oracle.family_refusal(ORACLE, base, 5.0))

    def test_the_recorder_refuses_a_pair_serve_before_any_completion(self):
        health = dict(one_mac(), policy=PAIR_POLICY)
        with serving(health) as (base, handler):
            oracle = str(Path(pair_accept.__file__).with_name("qwen38_27b_oracle.json"))
            argv = ["record_oracle.py", "--base-url", base, "--oracle", oracle, "--timeout", "5"]
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as stop:
                    record_oracle.main()
            self.assertEqual(2, stop.exception.code)
            self.assertIn("pair", stderr.getvalue())
            self.assertEqual(0, handler.posts)

    def test_the_recorder_refuses_another_gpu_family_from_the_serves_own_health(self):
        with serving(one_mac("apple-m5")) as (base, _):
            why = record_oracle.family_refusal(ORACLE, base, 5.0)
            self.assertIn("apple-m4", why)
            self.assertIn("apple-m5", why)
        with serving({"status": "ok", "version": "1.2.7"}) as (base, _):
            self.assertIn("did not report its GPU family", record_oracle.family_refusal(ORACLE, base, 5.0))
        self.assertIn("no gpu_family", record_oracle.family_refusal({"cases": []}, "http://127.0.0.1:9", 5.0))

    def test_acceptance_reads_each_ranks_family_from_its_own_serve(self):
        with serving(one_mac()) as (zero, _), serving(one_mac("apple-m5")) as (one, _):
            self.assertIsNone(pair_accept.pair_refusal(ORACLE, zero, zero, 5.0))
            why = pair_accept.pair_refusal(ORACLE, zero, one, 5.0)
            self.assertIn("rank 1", why)
            self.assertIn("apple-m5", why)


if __name__ == "__main__":
    unittest.main()
