"""tools/exact_restarts.py against a stub server scripting its own token_ids."""

import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import exact_restarts as er

BASE_IDS = list(range(100, 116))
PROMPT = {"name": "p", "kind": "completion", "prompt": "x"}


class Stub(BaseHTTPRequestHandler):
    ids = []

    def do_POST(self):  # noqa: N802 (the library's name)
        self.rfile.read(int(self.headers["Content-Length"]))
        body = {"tensorfold": {"token_ids": Stub.ids}}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_identical_runs_report_no_flip(server):
    Stub.ids = BASE_IDS
    runs = {"a": collect(server), "b": collect(server)}
    part = er.compare(["p"], runs)
    assert part["flips"] == 0 and part["rows"][0]["first_diff"] is None


@pytest.mark.parametrize("pos", [0, 7, 15])
def test_one_token_flip_is_found_at_its_position(server, pos):
    Stub.ids = BASE_IDS
    first = collect(server)
    flipped = BASE_IDS.copy()
    flipped[pos] += 1000
    Stub.ids = flipped
    part = er.compare(["p"], {"a": first, "b": collect(server)})
    assert part["flips"] == 1
    assert part["rows"][0]["first_diff"] == pos


def test_shorter_run_flips_at_the_first_missing_position(server):
    Stub.ids = BASE_IDS
    first = collect(server)
    Stub.ids = BASE_IDS[:9]
    part = er.compare(["p"], {"a": first, "b": collect(server)})
    assert part["rows"][0]["first_diff"] == 9


def collect(base):
    return er.collect(base, [PROMPT], 16)["p"]


def test_server_that_fails_to_start_is_named(tmp_path):
    bad = er.Server([sys.executable, "-c", "import sys; sys.exit(3)"], port=1)
    with pytest.raises(er.ServerFailed, match="exited 3"):
        bad.__enter__()
    bad.stop()


def test_request_ids_reads_the_tensorfold_block(server):
    Stub.ids = BASE_IDS[:4]
    assert er.request_ids(server, PROMPT, 4) == BASE_IDS[:4]
