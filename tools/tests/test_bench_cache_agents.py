"""Served cache-eviction receipts, real HTTP framing, strict evidence and deterministic A/B traffic."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_cache_agents as bench
import bench_agent_compare as compare
from agent_http_fixture import Fixture, MemoryHTTP, loopback

KEY = "tf-public-fixture-key"
ENV = "TF_AGENT_TEST_KEY"


def invoke(fixture, extra=(), base="http://127.0.0.1:1", memory=True):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "receipt.json"
        argv = ["run", base, fixture.model, "--trace", "agent_loops", "--conversations", "2",
                "--rounds", "4", "--prefix-tokens", "32", "--tokens", "2", "--output", str(output), *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        transport = patch.object(bench.urllib.request, "urlopen", side_effect=MemoryHTTP(fixture)) if memory else contextlib.nullcontext()
        with transport, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = bench.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, json.loads(output.read_text()) if output.exists() else None, stdout.getvalue() + stderr.getvalue()


class AgentTests(unittest.TestCase):
    def test_agent_order_grows_and_each_reply_is_closed_before_the_next_request(self):
        fixture = Fixture(cached=8)
        code, report, _ = invoke(fixture)
        self.assertEqual(code, 0)
        self.assertEqual(len(fixture.chat), 24)
        self.assertEqual([r["kind"] for r in report["rows"]], ["agent", "one_off", "one_off"] * 8)
        self.assertEqual([r["conversation"] for r in report["rows"] if r["kind"] == "agent"], [0, 1] * 4)
        agents = fixture.chat[0::3]
        self.assertEqual(agents[0]["messages"][0], agents[1]["messages"][0])
        self.assertEqual([len(agents[i]["messages"]) for i in (0, 2, 4, 6)], [2, 4, 6, 8])
        self.assertEqual(agents[2]["messages"][2], {"role": "assistant", "content": "MODEL-PRIVATE-REPLY"})
        self.assertEqual(len(agents[0]["messages"][0]["content"].encode()), 32)
        self.assertTrue(all(r["prompt_tokens"] >= 10001 for r in report["rows"]))
        self.assertEqual(report["rows"][0]["cache_state"], "cold")
        self.assertEqual(report["rows"][1]["cached_tokens"], 8)
        self.assertTrue(all(r["ttft_s"] >= 0 for r in report["rows"]))
        self.assertTrue(all(b["stream_options"]["include_usage"] and b["temperature"] == 0 for b in fixture.chat))
        self.assertEqual([b["seed"] for b in fixture.chat], list(range(7, 31)))

    def test_replacement_cohort_has_new_system_and_sessions_and_restarts_histories(self):
        fixture = Fixture()
        code, report, _ = invoke(fixture, ("--trace", "agent_cohorts"))
        self.assertEqual(code, 0)
        agents = fixture.chat[0::3]
        self.assertEqual([r["conversation"] for r in report["rows"] if r["kind"] == "agent"], [0, 1, 0, 1, 2, 3, 2, 3])
        self.assertNotEqual(agents[0]["messages"][0], agents[4]["messages"][0])
        self.assertEqual(len(agents[4]["messages"]), 2)
        self.assertEqual(len(report["system_prefixes"]), 2)

    def test_seed_reproduces_requests_and_different_seed_changes_the_fixture(self):
        a, b, c = Fixture(), Fixture(cached=8), Fixture()
        _, left, _ = invoke(a)
        _, right, _ = invoke(b)
        _, changed, _ = invoke(c, ("--seed", "11"))
        self.assertEqual(a.chat, b.chat)
        self.assertEqual(left["fixture_sha256"], right["fixture_sha256"])
        self.assertNotEqual(left["fixture_sha256"], changed["fixture_sha256"])

    def test_missing_usage_stays_unverified_and_no_rebuilt_sum_is_invented(self):
        code, report, _ = invoke(Fixture("missing_usage"))
        self.assertEqual(code, 1)
        self.assertTrue(report["completed_schedule"])
        self.assertTrue(all(r["cached_tokens"] is None and r["prompt_tokens"] is None for r in report["rows"]))
        self.assertTrue(all(r["cache_state"] == "unverified" for r in report["rows"]))
        self.assertIsNone(report["summary"]["rebuilt_rows"])
        self.assertFalse(compare.compare(report, report)["qualified"])

    def test_invalid_cache_counters_are_unverified(self):
        for mode in ("bad_cache", "too_large_cache", "negative_cache", "float_cache", "string_cache", "bad_prompt", "missing_cache"):
            with self.subTest(mode=mode):
                code, report, _ = invoke(Fixture(mode))
                self.assertEqual(code, 1)
                self.assertEqual(report["rows"][0]["cache_state"], "unverified")
                self.assertIsNone(report["summary"]["rebuilt_rows"])

    def test_incomplete_stream_is_retained_and_stops_without_retry(self):
        fixture = Fixture("incomplete")
        code, report, _ = invoke(fixture)
        self.assertEqual(code, 1)
        self.assertEqual(len(fixture.chat), 1)
        self.assertFalse(report["completed_schedule"])
        row = report["rows"][0]
        self.assertFalse(row["complete"])
        self.assertIsNone(row["ttft_s"])
        self.assertIsNone(row["token_sha256"])
        self.assertEqual(row["cache_state"], "unverified")

    def test_missing_full_hash_and_empty_or_tool_only_output_cannot_qualify(self):
        for mode in ("no_hash", "no_output", "tool_only"):
            with self.subTest(mode=mode):
                code, report, _ = invoke(Fixture(mode))
                self.assertEqual(code, 1)
                self.assertIsNone(compare.compare(report, report)["token_exact"])

    def test_reasoning_only_output_has_ttft_and_verified_fingerprints(self):
        fixture = Fixture("reasoning")
        code, report, _ = invoke(fixture)
        self.assertEqual(code, 0)
        self.assertTrue(all(r["ttft_s"] is not None and r["token_sha256"] for r in report["rows"]))
        self.assertEqual(fixture.chat[6]["messages"][2]["reasoning_content"], "fixture thought")

    def test_ttft_starts_at_content_not_headers_role_or_final_usage(self):
        original = bench.bench_stream.Collector

        class TimedCollector(original):
            def __init__(self, sent):
                super().__init__(sent)
                arrivals = iter([sent + 1, sent + 2, sent + 3])
                self.clock = lambda: next(arrivals)

        with patch.object(bench.bench_stream, "Collector", TimedCollector):
            code, report, _ = invoke(Fixture("role_first"))
        self.assertEqual(code, 0)
        self.assertTrue(all(row["ttft_s"] == 3 for row in report["rows"]))

    def test_unsupported_thinking_switch_is_omitted_and_its_mode_is_recorded(self):
        fixture = Fixture("thinking_default")
        code, report, _ = invoke(fixture)
        self.assertEqual(code, 0)
        self.assertEqual(report["thinking"]["thinking"], "default (thinking off unsupported)")
        self.assertTrue(all("chat_template_kwargs" not in body for body in fixture.chat))

    def test_http_failures_do_not_echo_server_text_and_are_not_retried(self):
        fixture = Fixture("http_error", key=KEY)
        with patch.dict(os.environ, {ENV: KEY}):
            code, report, shown = invoke(fixture, ("--api-key-env", ENV))
        self.assertEqual(code, 1)
        self.assertEqual(len(fixture.chat), 1)
        self.assertEqual(report["rows"][0]["error"], "HTTPError")
        self.assertNotIn(KEY, shown)

    def test_echoed_credential_is_not_sent_back_as_conversation_text(self):
        fixture = Fixture("echo_key_text", key=KEY)
        with patch.dict(os.environ, {ENV: KEY}):
            code, report, shown = invoke(fixture, ("--api-key-env", ENV))
        self.assertEqual(code, 1)
        self.assertEqual(len(fixture.chat), 1)
        self.assertEqual(report["rows"][0]["error"], "credential_in_reply")
        self.assertNotIn(KEY, json.dumps(fixture.chat) + shown)

    def test_selected_bearer_covers_discovery_tokenizer_and_chat_without_disclosure(self):
        fixture = Fixture(key=KEY)
        fixture.model = "PRIVATE-MODEL-ALIAS"
        with patch.dict(os.environ, {ENV: KEY}):
            code, report, shown = invoke(fixture, ("--api-key-env", ENV))
        self.assertEqual(code, 0)
        self.assertTrue(all(auth == "Bearer " + KEY for _, _, auth in fixture.seen))
        self.assertEqual({p for p, _, _ in fixture.seen}, {"/v1/models", "/tokenize", "/detokenize", "/v1/chat/completions"})
        for private in (KEY, fixture.model, "MODEL-PRIVATE-REPLY", "Conversation 0 begins", "http://127.0.0.1:1"):
            self.assertNotIn(private, shown + json.dumps(report))

    def test_default_ignores_environment_credentials(self):
        fixture = Fixture()
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY, ENV: KEY}):
            code, _, _ = invoke(fixture)
        self.assertEqual(code, 0)
        self.assertTrue(all(auth is None for _, _, auth in fixture.seen))

    def test_invalid_selected_key_fails_before_requests(self):
        fixture = Fixture()
        with patch.dict(os.environ, {ENV: KEY + "\n"}):
            code, report, shown = invoke(fixture, ("--api-key-env", ENV))
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertEqual(fixture.seen, [])
        self.assertNotIn(KEY, shown)

    def test_tokenizer_failure_preserves_an_unqualified_empty_receipt(self):
        fixture = Fixture("bad_tokenizer")
        code, report, _ = invoke(fixture)
        self.assertEqual(code, 1)
        self.assertEqual(report["rows"], [])
        self.assertEqual(fixture.chat, [])
        self.assertIsNotNone(report["setup_error"])

    def test_reply_hash_that_echoes_the_key_is_redacted_and_unverified(self):
        key = "a" * 64
        with patch.dict(os.environ, {ENV: key}):
            code, report, shown = invoke(Fixture("echo_key", key=key), ("--api-key-env", ENV))
        self.assertEqual(code, 1)
        self.assertNotIn(key, shown)
        self.assertIn("[REDACTED]", json.dumps(report))

    def test_credential_colliding_with_a_receipt_field_does_not_crash_or_qualify(self):
        with patch.dict(os.environ, {ENV: "rows"}):
            code, report, _ = invoke(Fixture(key="rows"), ("--api-key-env", ENV))
        self.assertEqual(code, 1)
        self.assertFalse(report["qualified"])

    def test_invalid_arguments_fail_before_http(self):
        for extra in (("--conversations", "0"), ("--prefix-tokens", "-1"), ("--timeout", "nan"),
                      ("--seed", "-1"), ("--trace", "agent_cohorts", "--rounds", "1")):
            with self.subTest(extra=extra):
                fixture = Fixture()
                code, _, _ = invoke(fixture, extra)
                self.assertEqual(code, 2)
                self.assertEqual(fixture.seen, [])


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = invoke(Fixture())[1]
        cls.b = invoke(Fixture(cached=8))[1]

    def test_equal_tokens_and_per_trace_counter_and_ttft_sums(self):
        a, b = copy.deepcopy(self.a), copy.deepcopy(self.b)
        for row in a["rows"]:
            row["ttft_s"] = 0.25
        for row in b["rows"]:
            row["ttft_s"] = 0.125
        result = compare.compare(a, b)
        self.assertTrue(result["qualified"])
        self.assertTrue(result["token_exact"])
        self.assertTrue(all(row["token_equal"] for row in result["per_request"]))
        trace = result["traces"]["agent_loops"]
        self.assertEqual(trace["rebuilt_rows_b_minus_a"], -8 * 23)
        self.assertEqual(trace["a"]["ttft_sum_s"], 6)
        self.assertEqual(trace["b"]["ttft_sum_s"], 3)

    def test_token_mismatch_fails_even_when_output_text_matches(self):
        bad = invoke(Fixture("mismatch"))[1]
        self.assertEqual(self.a["rows"][0]["output_sha256"], bad["rows"][0]["output_sha256"])
        result = compare.compare(self.a, bad)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["token_exact"])
        self.assertIsNone(result["traces"]["agent_loops"]["rebuilt_rows_b_minus_a"])

    def test_matching_token_hashes_do_not_hide_different_visible_output(self):
        bad = copy.deepcopy(self.b)
        bad["rows"][0]["output_sha256"] = "f" * 64
        result = compare.compare(self.a, bad)
        self.assertTrue(result["token_exact"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["per_request"][0]["output_equal"])

    def test_mismatched_inputs_and_incomplete_schedules_cannot_pass(self):
        bad = copy.deepcopy(self.b)
        bad["thinking"]["thinking"] = "different"
        self.assertFalse(compare.compare(self.a, bad)["qualified"])
        bad = copy.deepcopy(self.b)
        bad["rows"].pop()
        bad["fixture_sha256"] = bench.digest([r["request_sha256"] for r in bad["rows"]])
        bad["completed_schedule"] = False
        self.assertFalse(compare.compare(self.a, bad)["qualified"])

    def test_boolean_cached_tokens_and_invalid_time_are_not_valid_evidence(self):
        for field, value in (("cached_tokens", True), ("ttft_s", -1), ("ttft_s", float("nan"))):
            with self.subTest(field=field, value=str(value)):
                bad = copy.deepcopy(self.b)
                bad["rows"][0][field] = value
                self.assertFalse(compare.compare(self.a, bad)["qualified"])

    def test_request_fingerprint_changes_cannot_be_hidden_by_matching_reply_hashes(self):
        bad = copy.deepcopy(self.b)
        bad["rows"][0]["request_sha256"] = "f" * 64
        bad["fixture_sha256"] = bench.digest([r["request_sha256"] for r in bad["rows"]])
        result = compare.compare(self.a, bad)
        self.assertFalse(result["qualified"])
        self.assertIn("request_identities_differ", result["issues"])

    def test_matching_hashes_with_different_prompt_usage_are_not_comparable(self):
        bad = copy.deepcopy(self.b)
        bad["rows"][0]["prompt_tokens"] += 1
        result = compare.compare(self.a, bad)
        self.assertFalse(result["qualified"])
        self.assertIn("prompt_usage_unverified_or_different", result["issues"])

    def test_performance_changes_are_reported_without_an_implicit_pass_threshold(self):
        result = compare.compare(self.b, self.a)
        self.assertTrue(result["qualified"])
        self.assertFalse(result["traces"]["agent_loops"]["rows_no_worse"])

    def test_both_trace_shapes_can_be_collected_and_compared(self):
        a = invoke(Fixture(), ("--trace", "agent_cohorts"))[1]
        b = invoke(Fixture(cached=8), ("--trace", "agent_cohorts"))[1]
        result = compare.compare(a, b)
        self.assertTrue(result["qualified"])
        self.assertEqual(set(result["traces"]), {"agent_cohorts"})

    def test_compare_cli_returns_one_for_a_mismatch_and_two_for_invalid_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, out = [Path(directory) / name for name in ("a.json", "b.json", "out.json")]
            a.write_text(json.dumps(self.a))
            bad = copy.deepcopy(self.b)
            bad["rows"][0]["token_sha256"] = "f" * 64
            b.write_text(json.dumps(bad))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(["compare", str(a), str(b), "--output", str(out)]), 1)
                b.write_text('{"broken": true}')
                self.assertEqual(bench.main(["compare", str(a), str(b), "--output", str(out)]), 2)


class LoopbackTests(unittest.TestCase):
    def check(self, mode):
        fixture = Fixture(mode)
        try:
            with loopback(fixture) as base:
                return invoke(fixture, base=base, memory=False)
        except PermissionError:
            self.skipTest("loopback socket creation denied by host")

    def test_real_http_sequential_traffic(self):
        code, report, _ = self.check("normal")
        self.assertEqual(code, 0)
        self.assertEqual(len(report["rows"]), 24)

    def test_real_http_missing_usage(self):
        code, report, _ = self.check("missing_usage")
        self.assertEqual(code, 1)
        self.assertIsNone(report["summary"]["rebuilt_rows"])


if __name__ == "__main__":
    unittest.main()
