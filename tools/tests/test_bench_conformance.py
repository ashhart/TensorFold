"""Model-free tests for the conformance kit's comparisons, its gates and their planted controls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conformance
from conformance import drafted_gate, equality_gate, refusals_gate, reply_row, resumed_gate, same, tally
from conformance_lifecycle import cancel_gate, census, evict_gate
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_served_api import FAULTS, doc, fake, script, serve

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


class LifecycleVerdictTests(unittest.TestCase):
    """The cancel and evict verdicts: vacuous, uncounted, held and unevicted runs fail with their reason."""

    A, B = reply_row(200, doc()), reply_row(200, doc(sha="b" * 64))
    CUT, BASE = {"chunks": 4, "ended": False}, {"running": 0, "waiting": 0, "active": 0, "lanes_free": 8}

    def test_cancel_needs_clean_counted_settled_cuts(self):
        ok = cancel_gate([(self.A, self.A)], [self.CUT], [True], [True], 1, self.BASE)
        self.assertTrue(ok["passed"], ok["line"])
        cases = {"vacuous": ([(self.A, self.A)], [{"chunks": 4, "ended": True}], [True], [True], 1, self.BASE),
                 "counted 0": ([(self.A, self.A)], [self.CUT], [True], [True], 0, self.BASE),
                 "counted 2": ([(self.A, self.A)], [self.CUT], [True], [True], 2, self.BASE),
                 "did not return": ([(self.A, self.A)], [self.CUT], [False], [True], 1, self.BASE),
                 "not ok": ([(self.A, self.A)], [self.CUT], [True], [False], 1, self.BASE),
                 "no idle census": ([(self.A, self.A)], [self.CUT], [True], [True], 1, None),
                 "1 unequal": ([(self.A, self.B)], [self.CUT], [True], [True], 1, self.BASE)}
        for why, args in cases.items():
            with self.subTest(why=why):
                gate = cancel_gate(*args)
                self.assertFalse(gate["passed"])
                self.assertIn(why, gate["line"])

    def test_evict_needs_a_kept_prompt_that_leaves(self):
        def row(cached, sha="a"):
            return {**reply_row(200, doc(sha=sha * 64)), "cached": cached}
        ok = evict_gate(row(0), [row(0), row(25)], row(0), [row(0)], True, self.BASE)
        self.assertTrue(ok["passed"], ok["line"])
        cases = {"vacuous": (row(0), [row(0), row(0)], row(0), [row(0)], True, self.BASE),
                 "nothing evicted": (row(0), [row(0), row(25)], row(25), [], True, self.BASE),
                 "1 unequal": (row(0), [row(0), row(25)], row(0, "b"), [row(0)], True, self.BASE),
                 "did not return": (row(0), [row(0), row(25)], row(0), [row(0)], False, self.BASE),
                 "failed": (row(0), [row(0), row(25)], row(0), [{"status": 0, "error": "x"}], True, self.BASE)}
        for why, args in cases.items():
            with self.subTest(why=why):
                gate = evict_gate(*args)
                self.assertFalse(gate["passed"])
                self.assertIn(why, gate["line"])

    def test_the_census_keeps_every_pool(self):
        pools = [{"promised_tokens": 0, "available_tokens": 90, "reclaimable_tokens": 10},
                 {"promised_tokens": 64, "available_tokens": 7}]
        load = {"running": 0, "waiting": 0, "http_active": {"chat_completions": 0}, "capacity": {"pools": pools}}
        client = type("Stub", (), {"get": lambda _self, path: (200, json.dumps(load).encode())})()
        self.assertEqual(census(client)["pools"], [{"promised": 0, "unheld": 100}, {"promised": 64, "unheld": 7}])


class GateRunTests(unittest.TestCase):
    """Every gate passes on the clean fake; each fault the fake plants turns exactly its gate red."""

    NAMES = {"drafted": "drafted == plain", "repeat": "repeat == first", "concurrent": "concurrent == solo",
             "resumed": "resumed == fresh", "refusals": "refusals", "cancel": "cancel", "evict": "evict",
             "cancel_lane": "cancel", "cancel_free": "cancel", "cancel_reply": "cancel", "cancel_health": "cancel"}
    GATE_OF = {fault: name.split()[0] for fault, name in NAMES.items() if fault in FAULTS}  # the gate a fault turns red

    def run_kit(self, *extra, fault=None, shape=None, **changes):
        profile = json.loads(PROFILE.read_text())
        for block in ("cancel", "evict"):
            profile[block]["settle_s"] = 1  # a held lane fails in a second, not a minute
        profile.update(changes)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, fake(True, fault, **(shape or {})) as url, fake(False) as fresh, \
                contextlib.redirect_stdout(out):
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(profile))
            code = conformance.main([str(path), "--url", url, "--fresh-url", fresh, *extra])
        return code, [line for line in out.getvalue().splitlines() if line.startswith(("PASS", "FAIL", "SKIP"))]

    def assert_only_red(self, fault, also=()):
        code, lines = self.run_kit(fault=fault)
        self.assertEqual(code, 1, lines)
        self.assertEqual(len(lines), len(conformance.GATES), lines)
        red = [line for line in lines if line.startswith("FAIL")]
        self.assertEqual(len(red), 1 + len(also), lines)
        self.assertTrue(red[0].startswith("FAIL " + self.NAMES[fault]), lines)
        self.assertEqual([line.split(":")[0] for line in red[1:]], ["FAIL " + g for g in also], lines)

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

    def test_fault_cancel_keeps_a_lane(self):
        # A request left running keeps the server from ever going idle, so evict has no idle census either.
        self.assert_only_red("cancel_lane", also=("evict",))

    def test_fault_cancel_loses_only_a_free_lane(self):
        self.assert_only_red("cancel_free")

    def test_fault_cancel_changes_the_next_reply(self):
        self.assert_only_red("cancel_reply")

    def test_fault_cancel_leaves_the_server_unhealthy(self):
        self.assert_only_red("cancel_health")

    def test_fault_evict_answers_from_stale_state(self):
        self.assert_only_red("evict")

    def test_skip_names_the_gate_and_its_reason(self):
        code, lines = self.run_kit("--gates", "cancel,evict", skip={"evict": "no prompt cache"})
        self.assertEqual(code, 0, lines)
        self.assertEqual(lines[1], "SKIP evict: no prompt cache")
        code, lines = self.run_kit("--gates", "cancel", cancel={})
        self.assertEqual(code, 1, lines)
        self.assertIn("no cancel block", lines[0])

    def test_skip_takes_only_cancel_and_evict_and_plant_refuses_a_skipped_gate(self):
        cases = {"only cancel and evict": ("--gates", "drafted", {"drafted": "x"}),
                 "skips that gate": ("--gates", "evict", "--plant", "evict", {"evict": "no prompt cache"}),
                 "a non-empty string": ("--gates", "evict", {"evict": True}),
                 "each skipped gate": ("--gates", "evict", ["evict"])}
        for why, (*argv, skip) in cases.items():
            with self.subTest(why=why), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as exit_:
                    self.run_kit(*argv, skip=skip)
            self.assertEqual(exit_.exception.code, 2)
            self.assertIn(why, err.getvalue())

    def test_a_stream_that_ends_before_the_cut_is_vacuous(self):
        short = {"prompt": "count", "tokens": 2, "after_chunks": 8, "rounds": 1, "settle_s": 1}
        code, lines = self.run_kit("--gates", "cancel", cancel=short)
        self.assertEqual(code, 1, lines)
        self.assertIn("2 cuts came after the reply ended or failed (vacuous)", lines[0])  # one a sampling mode

    def test_a_server_busy_after_each_reply_and_without_lanes_free_settles_first(self):
        # Like Metal qwen27: no capacity.lanes_free, and each reply is followed by a background prefill of the next turn.
        profile = json.loads(PROFILE.read_text())
        blocks = {b: {k: v for k, v in profile[b].items() if k != "settle_s"} for b in ("cancel", "evict")}
        shape = {"warm": 0.4, "lanes_free": False}
        code, lines = self.run_kit("--gates", "cancel,evict", shape=shape, settle_s=5, **blocks)
        self.assertEqual(code, 0, lines)
        self.assertTrue(all(line.startswith("PASS") for line in lines), lines)
        self.assertIn("censuses back to {'running': 0, 'waiting': 0, 'active': 0}", lines[0])
        code, lines = self.run_kit("--gates", "cancel,evict", shape=shape, settle_s=0.1, **blocks)
        self.assertEqual(code, 1, lines)
        self.assertTrue(all(line.startswith("FAIL") for line in lines), lines)
        self.assertIn("no idle census from GET /v1/load (busy past settle_s)", lines[0])

    def test_each_plant_flag_turns_its_gate_red(self):
        for gate in conformance.GATES:
            with self.subTest(gate=gate):
                code, lines = self.run_kit("--gates", gate, "--plant", gate)
                self.assertEqual(code, 1, lines)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("FAIL " + self.NAMES[gate]), lines)

    def test_sequential_requests_under_the_concurrent_fault_stay_unsalted(self):
        # A handler still counted in flight after its reply would make the next sequential request read as batched.
        with fake(True, "concurrent", linger=0.3) as url:
            client = conformance.Client(url, "fake", timeout=10)
            rows = [client.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}) for _ in range(3)]
        self.assertTrue(all(row["status"] == 200 for row in rows), rows)
        self.assertEqual(len({row["sha"] for row in rows}), 1, rows)

    def test_streamed_replies_pass_clean_and_each_fault_turns_its_gate_red(self):
        code, lines = self.run_kit("--stream")
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines), len(conformance.GATES))
        self.assertTrue(all(line.startswith("PASS") for line in lines), lines)
        for fault, gate in self.GATE_OF.items():
            with self.subTest(fault=fault):
                code, lines = self.run_kit("--stream", "--gates", gate, fault=fault)
                self.assertEqual(code, 1, lines)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("FAIL " + self.NAMES[fault]), lines)

    def test_a_streamed_row_is_the_json_row(self):
        body = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                             {"role": "user", "content": "c"}], "max_tokens": 4, "draft": True}
        with fake(True) as url:
            plain = conformance.Client(url, "fake", timeout=10).post(body)
            streamed = conformance.Client(url, "fake", timeout=10, stream=True).post(body)
            refused = conformance.Client(url, "fake", timeout=10, stream=True).post({**body, "n": 2}, stream=False)
        self.assertEqual(streamed.pop("completion"), "finished")
        self.assertEqual(streamed, plain)
        self.assertEqual((plain["cached"], plain["accepted"]), (5, 2))
        self.assertEqual((refused["status"], refused["error"]), (400, "n must be 1; multiple choices are not supported"))

    def test_a_stream_that_errs_or_stops_early_is_unverified(self):
        words = [(0, "content", w) for w in ("a", "b", "c")]
        cases = {"finished": {}, "fixture failure": {"error_after": 1}, "incomplete_stream": {"done": False}}
        rows = {}
        for name, options in cases.items():
            with serve(lambda body, o=options: script(words, [1], **o)) as url:
                rows[name] = conformance.Client(url, "fake", timeout=10, stream=True).post({"messages": []})
        self.assertEqual((rows["finished"]["sha"], rows["finished"]["text"]), ("d" * 64, "abc"))
        for name in ("fixture failure", "incomplete_stream"):
            self.assertEqual((rows[name]["status"], rows[name]["error"]), (200, name))
            self.assertIsNone(same(rows["finished"], rows[name]))

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

    def test_fresh_env_reaches_only_servers_keeping_no_prompts(self):
        profile = {"name": "p", "fresh_env": {"TF_QWEN27_SLOT_RESUME": 0}}
        fresh, cached = (conformance.Server(None, profile, gib, Path(".")).env() for gib in (0, 2))
        self.assertEqual(fresh["TF_QWEN27_SLOT_RESUME"], "0")
        self.assertEqual(cached, dict(conformance.os.environ))
        self.assertEqual(conformance.Server(None, {"name": "p"}, 0, Path(".")).env(), dict(conformance.os.environ))

    def test_profiles_parse_and_name_their_gates(self):
        for path in sorted(PROFILE.parent.glob("*.json")):
            with self.subTest(profile=path.name):
                profile = json.loads(path.read_text())
                self.assertGreaterEqual(len(profile["conversation"]), 2)
                self.assertTrue(profile["refusals"])
                self.assertTrue(all(r["name"] and r["error"] for r in profile["refusals"]))
                self.assertGreater(profile["sampled"]["temperature"], 0)
                self.assertTrue(all(isinstance(v, str) for v in profile.get("fresh_env", {}).values()))
                self.assertEqual(conformance.lifecycle_gaps(profile), [])  # each lifecycle gate runs or says why not

    def test_an_empty_lifecycle_block_is_a_gap(self):
        base = {"cancel": {"prompt": "count"}, "evict": {"fill_prompts": 12}}
        self.assertEqual(conformance.lifecycle_gaps(base), [])
        for gate, block in (("cancel", {}), ("evict", {}), ("evict", {"fill_prompts": 0}), ("cancel", {"prompt": ""})):
            with self.subTest(gate=gate, block=block):
                self.assertEqual(conformance.lifecycle_gaps({**base, gate: block}), [gate])
                self.assertEqual(conformance.lifecycle_gaps({**base, gate: block, "skip": {gate: "why"}}), [])


if __name__ == "__main__":
    unittest.main()
