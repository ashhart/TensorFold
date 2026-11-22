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


def test_a_busy_peer_error_fails_the_run(monkeypatch):
    calls = []

    def fake(base, item, tokens, api_key=None, timeout=600):
        calls.append(1)
        if len(calls) > 1:
            raise urllib.error.HTTPError(base, 500, "peer failed", {}, None)
        return BASE_IDS

    monkeypatch.setattr(er, "request_ids", fake)
    with pytest.raises(urllib.error.HTTPError, match="peer failed"):
        er.collect("http://127.0.0.1:1", [PROMPT], 16, busy=3)


def test_a_busy_peer_with_different_tokens_fails_the_run(monkeypatch):
    calls = []

    def fake(base, item, tokens, api_key=None, timeout=600):
        calls.append(1)
        return BASE_IDS if len(calls) == 1 else BASE_IDS[:3] + [999] + BASE_IDS[4:]

    monkeypatch.setattr(er, "request_ids", fake)
    with pytest.raises(RuntimeError, match="busy peer"):
        er.collect("http://127.0.0.1:1", [PROMPT], 16, busy=3)


def test_attached_mode_reports_unrunnable_scenarios_as_not_run(server, monkeypatch, tmp_path, capsys):
    Stub.ids = BASE_IDS
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([PROMPT]))
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["exact_restarts.py", "--base", server, "--drafter", "d",
                                    "--prompts", str(prompts), "--output", str(out)])
    assert er.main() == 0
    report = json.loads(out.read_text())
    assert report["scenarios"] == ["busy"]
    assert set(report["not_run"]) == {"restart", "parallel", "draft"}
    assert "not run" in capsys.readouterr().out
