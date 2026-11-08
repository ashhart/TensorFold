"""Model-free tests for the conformance kit's comparisons, its gates and their planted controls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conformance
from conformance import drafted_gate, equality_gate, refusals_gate, reply_row, resumed_gate, same, tally
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_served_api import doc, fake

PROFILE = Path(__file__).resolve().parents[1] / "conformance_profiles" / "qwen35-cuda.json"


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


class GateRunTests(unittest.TestCase):
    """Every gate passes on the clean fake; each fault the fake plants turns exactly its gate red."""

    NAMES = {"drafted": "drafted == plain", "repeat": "repeat == first", "concurrent": "concurrent == solo",
             "resumed": "resumed == fresh", "refusals": "refusals"}

    def run_kit(self, *extra, fault=None):
        out = io.StringIO()
        with fake(True, fault) as url, fake(False) as fresh, contextlib.redirect_stdout(out):
            code = conformance.main([str(PROFILE), "--url", url, "--fresh-url", fresh, *extra])
        return code, [line for line in out.getvalue().splitlines() if line.startswith(("PASS", "FAIL"))]

    def assert_only_red(self, fault):
        code, lines = self.run_kit(fault=fault)
        self.assertEqual(code, 1, lines)
        self.assertEqual(len(lines), len(conformance.GATES), lines)
        red = [line for line in lines if line.startswith("FAIL")]
        self.assertEqual(len(red), 1, lines)
        self.assertTrue(red[0].startswith("FAIL " + self.NAMES[fault]), lines)

    def test_clean_fake_passes_every_gate(self):
        code, lines = self.run_kit()
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines), len(conformance.GATES))
        self.assertTrue(all(line.startswith("PASS") for line in lines), lines)

    def test_fault_drafted_differs_from_plain(self):
        self.assert_only_red("drafted")

    def test_fault_repeat_differs(self):
        self.assert_only_red("repeat")

    def test_fault_concurrent_differs_from_solo(self):
        self.assert_only_red("concurrent")

    def test_fault_resumed_reports_no_cached_tokens(self):
        self.assert_only_red("resumed")

    def test_fault_listed_refusal_accepted(self):
        self.assert_only_red("refusals")

    def test_each_plant_flag_turns_its_gate_red(self):
        for gate in conformance.GATES:
            with self.subTest(gate=gate):
                code, lines = self.run_kit("--gates", gate, "--plant", gate)
                self.assertEqual(code, 1, lines)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("FAIL " + self.NAMES[gate]), lines)

    def test_resume_without_a_fresh_server_fails(self):
        out = io.StringIO()
        with fake(True) as url, contextlib.redirect_stdout(out):
            code = conformance.main([str(PROFILE), "--url", url, "--gates", "resumed"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL resumed == fresh: no cache-off server", out.getvalue())

    def test_context_items_lengthen_only_the_first_turn(self):
        sent = []

        class Recorder:
            def post(self, body):
                sent.append(body["messages"])
                return {"status": 200, "sha": "a" * 64, "text": "ok"}
        profile = {**json.loads(PROFILE.read_text()), "conversation_context_items": 50}
        rows = conformance.conversation(Recorder(), conformance.Plan(profile, None), "greedy", True)
        self.assertEqual([r["turn"] for r in rows], list(range(len(profile["conversation"]))))
        self.assertTrue(sent[0][0]["content"].startswith(conformance.inventory(50)))
        self.assertTrue(all(m["content"] == t for m, t in zip(sent[-1][2::2], profile["conversation"][1:])))

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
