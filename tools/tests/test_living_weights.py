"""Model-free tests for the Living Weights harness and console proxy, against a fake server."""
import contextlib
import http.server
import io
import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "living_weights"))
import console
import lw_client
import lw_facts
import lw_rules

LEARN_STREAM = (b"event: fact\ndata: {\"id\": 1, \"text\": \"T\"}\n\nevent: learning\ndata: {\"id\": 1}\n\n"
                b"event: learned\ndata: {\"id\": 1, \"recalled\": true}\n\nevent: learned\ndata: [DONE]\n\n")


class FakeTensorFold(http.server.BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, fmt, *args):
        pass

    def _reply(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._reply(200, json.dumps({"data": [{"id": "fake"}]}).encode())
        self._reply(200, json.dumps({"path": self.path}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTensorFold.bodies.append((self.path, body))
        if self.path == "/v1/slide/learn":
            return self._reply(200, LEARN_STREAM, "text/event-stream")
        question = body["messages"][-1]["content"]
        reply = {"choices": [{"message": {"content": f"echo: {question}"}}]}
        self._reply(200, json.dumps(reply).encode())


@contextlib.contextmanager
def serving(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class ClientTests(unittest.TestCase):
    def test_parse_sse_names_events_and_decodes_json(self):
        events = list(lw_client.parse_sse(io.BytesIO(LEARN_STREAM)))
        self.assertEqual([n for n, _ in events], ["fact", "learning", "learned", "learned"])
        self.assertEqual(events[2][1], {"id": 1, "recalled": True})
        self.assertEqual(events[3][1], "[DONE]")

    def test_hit_ignores_case(self):
        self.assertTrue(lw_client.hit("It is 1,312 Metres long", ["1,312 metres"]))
        self.assertFalse(lw_client.hit("About 2.7 km", ["1,312", "quillon"]))

    def test_learn_sends_text_and_source(self):
        FakeTensorFold.bodies.clear()
        with serving(FakeTensorFold) as url:
            events = lw_client.learn(url, "The Quillon bridge is 1,312 metres long.", "document", echo=False)
        self.assertEqual(FakeTensorFold.bodies, [("/v1/slide/learn", {"text": "The Quillon bridge is 1,312 metres long.",
                                                                      "source": "document"})])
        self.assertEqual(events[2]["data"]["recalled"], True)

    def test_ask_is_greedy(self):
        FakeTensorFold.bodies.clear()
        with serving(FakeTensorFold) as url:
            self.assertEqual(lw_client.ask(url, "Q?"), "echo: Q?")
        self.assertEqual(FakeTensorFold.bodies[0][1]["temperature"], 0)


class SummaryTests(unittest.TestCase):
    def test_facts_summary_counts_each_kind(self):
        rows = [{"kind": "recall", "ok": True}, {"kind": "recall", "ok": False},
                {"kind": "neighbour", "leaked": True}, {"kind": "control", "ok": True}]
        self.assertEqual(lw_facts.summarise(rows), {"recall_ok": "1/2", "neighbour_leaked": "1/1", "control_ok": "1/1"})

    def test_rules_in_context_question_carries_the_rule(self):
        spec = {"general": [], "rules": [{"id": "r", "text": "Rule.", "recall": {"q": "R?", "expect": ["x"]},
                                          "problems": [{"q": "P?", "expect": ["echo: Rule. P?"]}]}]}
        with serving(FakeTensorFold) as url:
            rows = lw_rules.probe(url, spec)
        self.assertEqual(lw_rules.summarise(rows), {"general": "0/0", "rule_recall": "0/1", "from_memory": "0/1",
                                                    "in_context": "1/1"})


class ConsoleTests(unittest.TestCase):
    def test_build_targets_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            console.build_targets("http://127.0.0.1:1", extra=["no-equals"])
        with self.assertRaises(ValueError):
            console.build_targets("file:///etc/passwd")

    def test_route_allows_learning_only_on_the_taught_server(self):
        targets = console.build_targets("http://127.0.0.1:1", "http://127.0.0.1:2", ["other=http://127.0.0.1:3"])
        self.assertEqual(console.route(targets, "/m/taught/v1/slide/learn"), ("http://127.0.0.1:1", "/v1/slide/learn"))
        self.assertEqual(console.route(targets, "/m/clean/v1/slide/learn")[0], 403)
        self.assertEqual(console.route(targets, "/m/other/v1/slide/graph")[0], 403)
        self.assertEqual(console.route(targets, "/m/other/v1/models"), ("http://127.0.0.1:3", "/v1/models"))
        self.assertEqual(console.route(targets, "/m/missing/v1/models")[0], 404)
        self.assertEqual(console.route(targets, "/slide"), ("http://127.0.0.1:1", "/slide"))

    def test_proxy_streams_learn_events_and_refuses_reference(self):
        with serving(FakeTensorFold) as taught, serving(FakeTensorFold) as clean:
            console.Handler.targets = console.build_targets(taught, clean)
            with serving(console.Handler) as page:
                req = urllib.request.Request(f"{page}/m/taught/v1/slide/learn", data=b'{"text": "T"}',
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    self.assertEqual(resp.read(), LEARN_STREAM)
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    urllib.request.urlopen(urllib.request.Request(f"{page}/m/clean/v1/slide/learn", data=b"{}"),
                                           timeout=10)
                self.assertEqual(refused.exception.code, 403)
                with urllib.request.urlopen(f"{page}/targets", timeout=10) as resp:
                    self.assertEqual(json.load(resp)["clean"], {"learn": False, "reference": True})


if __name__ == "__main__":
    unittest.main()
