"""Model-free tests for the conformance kit's comparisons, its gates and their planted controls."""
import contextlib
import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conformance
from conformance import drafted_gate, equality_gate, refusals_gate, reply_row, resumed_gate, same, tally

PROFILE = Path(__file__).resolve().parents[1] / "conformance_profiles" / "qwen35-cuda.json"


def doc(sha="a" * 64, cached=0, accepted=2, drafted=3, text="hi"):
    return {"choices": [{"message": {"content": text}}], "tensorfold": {"token_sha256": sha},
            "speculative": {"accepted": accepted, "drafted": drafted},
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": cached}}}


class RowTests(unittest.TestCase):
    def test_reply_row_reads_hash_counts_and_text(self):
        row = reply_row(200, doc(cached=5))
        self.assertEqual((row["sha"], row["cached"], row["accepted"], row["drafted"], row["text"]),
                         ("a" * 64, 5, 2, 3, "hi"))

    def test_malformed_hashes_and_counts_stay_unknown(self):
        for sha in [None, "a" * 12, "A" * 64, True, "z" * 64]:
            self.assertIsNone(reply_row(200, doc(sha=sha))["sha"])
        for cached in [None, -1, True, "0", 0.0]:
            self.assertIsNone(reply_row(200, doc(cached=cached))["cached"])
        self.assertIsNone(reply_row(200, "not an object")["sha"])

    def test_error_bodies_keep_status_and_message(self):
        row = reply_row(400, {"error": {"message": "n must be 1", "type": "invalid_request_error"}})
        self.assertEqual((row["status"], row["error"]), (400, "n must be 1"))
        self.assertEqual(reply_row(500, {})["error"], "HTTP 500")


class ComparisonTests(unittest.TestCase):
    def test_same_needs_two_full_hashes(self):
        a, b = reply_row(200, doc()), reply_row(200, doc(sha="b" * 64))
        self.assertTrue(same(a, a))
        self.assertFalse(same(a, b))
        self.assertIsNone(same(a, reply_row(200, doc(sha=None))))
        self.assertIsNone(same(a, {**a, "error": "x"}))
        self.assertEqual(tally([(a, a), (a, b), (a, {})]), {"equal": 1, "unequal": 1, "unverified": 1})

    def test_equality_gate_fails_on_unequal_unverified_or_empty(self):
        a, b = reply_row(200, doc()), reply_row(200, doc(sha="b" * 64))
        self.assertTrue(equality_gate("g", [(a, a)])["passed"])
        self.assertFalse(equality_gate("g", [(a, a), (a, b)])["passed"])
        self.assertFalse(equality_gate("g", [(a, {})])["passed"])
        self.assertFalse(equality_gate("g", [])["passed"])

    def test_drafted_gate_is_vacuous_without_an_accepted_draft(self):
        plain, drafted = reply_row(200, doc(accepted=0, drafted=0)), reply_row(200, doc(accepted=0, drafted=7))
        gate = drafted_gate([(plain, drafted)])
        self.assertFalse(gate["passed"])
        self.assertIn("vacuous", gate["line"])
        self.assertTrue(drafted_gate([(plain, reply_row(200, doc()))])["passed"])

    def test_resumed_gate_asserts_both_cache_counts(self):
        def turns(fresh_cached, resumed_cached):
            out = []
            for i, (f, r) in enumerate(zip(fresh_cached, resumed_cached)):
                out.append(({**reply_row(200, doc(cached=f)), "turn": i}, {**reply_row(200, doc(cached=r)), "turn": i}))
            return out
        self.assertTrue(resumed_gate(turns([0, 0, 0], [0, 4, 9]))["passed"])
        cold = resumed_gate(turns([0, 0, 0], [0, 0, 9]))
        self.assertFalse(cold["passed"])
        self.assertIn("each must be > 0", cold["line"])
        warm_fresh = resumed_gate(turns([0, 3, 0], [0, 4, 9]))
        self.assertFalse(warm_fresh["passed"])
        self.assertIn("--prompt-cache-gib 0", warm_fresh["line"])
        self.assertFalse(resumed_gate(turns([0], [0]))["passed"])

    def test_refusals_need_status_and_text(self):
        expect = {"status": 400, "error": "n must be 1"}
        refused = {"status": 400, "error": "n must be 1; multiple choices are not supported"}
        self.assertTrue(refusals_gate([("n", refused, expect)])["passed"])
        self.assertFalse(refusals_gate([("n", {**refused, "status": 503}, expect)])["passed"])
        self.assertFalse(refusals_gate([("n", {"status": 200, "sha": "a" * 64}, expect)])["passed"])
        self.assertFalse(refusals_gate([])["passed"])


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """A deterministic chat server: the reply hash depends on messages and, when sampling, the seed."""

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        error = refusal(body)
        if error:
            return self.reply(400, {"error": {"message": error, "type": "invalid_request_error"}})
        seed = body["seed"] if body.get("temperature") else None
        sha = hashlib.sha256(json.dumps([body["messages"], seed]).encode()).hexdigest()
        cached = 5 if self.server.cache and len(body["messages"]) > 1 else 0
        accepted = 2 if body.get("draft", True) else 0
        self.reply(200, doc(sha=sha, cached=cached, accepted=accepted, drafted=accepted, text="r" + sha[:6]))

    def reply(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def refusal(body):
    if body.get("n", 1) != 1:
        return "n must be 1; multiple choices are not supported"
    if "top_logprobs" in body and not body.get("logprobs"):
        return "top_logprobs requires logprobs: true"
    if abs(body.get("presence_penalty", 0)) > 2:
        return "presence_penalty must be between -2 and 2, or null"
    if body.get("max_tokens", 0) > 100000:
        return "This server's maximum context length is 16384 tokens"
    return None


@contextlib.contextmanager
def fake(cache):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    server.cache = cache
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class GateRunTests(unittest.TestCase):
    """Every gate passes on the fake engine and goes red when its fault is planted."""

    def run_kit(self, *extra):
        out = io.StringIO()
        with fake(True) as url, fake(False) as fresh, contextlib.redirect_stdout(out):
            code = conformance.main([str(PROFILE), "--url", url, "--fresh-url", fresh, *extra])
        return code, [line for line in out.getvalue().splitlines() if line.startswith(("PASS", "FAIL"))]

    def test_all_gates_pass(self):
        code, lines = self.run_kit()
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines), len(conformance.GATES))
        self.assertTrue(all(line.startswith("PASS") for line in lines), lines)

    def test_each_planted_fault_turns_its_gate_red(self):
        names = {"drafted": "drafted == plain", "repeat": "repeat == first", "concurrent": "concurrent == solo",
                 "resumed": "resumed == fresh", "refusals": "refusals"}
        for gate in conformance.GATES:
            with self.subTest(gate=gate):
                code, lines = self.run_kit("--gates", gate, "--plant", gate)
                self.assertEqual(code, 1, lines)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("FAIL " + names[gate]), lines)

    def test_resume_without_a_fresh_server_fails(self):
        out = io.StringIO()
        with fake(True) as url, contextlib.redirect_stdout(out):
            code = conformance.main([str(PROFILE), "--url", url, "--gates", "resumed"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL resumed == fresh: no cache-off server", out.getvalue())

    def test_profiles_parse_and_name_their_gates(self):
        for path in sorted(PROFILE.parent.glob("*.json")):
            with self.subTest(profile=path.name):
                profile = json.loads(path.read_text())
                self.assertGreaterEqual(len(profile["conversation"]), 2)
                self.assertTrue(profile["refusals"])
                self.assertTrue(all(r["name"] and r["error"] for r in profile["refusals"]))
                self.assertGreater(profile["sampled"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
