"""Recorded synthetic fixtures exercise strict IDs, BF16 diagnostics and served/replay request contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import draft_divergence as harness

FIXTURES = Path(__file__).parent / "fixtures" / "draft_divergence"


class DraftDivergenceTests(unittest.TestCase):
    def setUp(self):
        self.recording = harness.load(FIXTURES / "recording.json")
        self.profiles = harness.load(FIXTURES / "profiles.json")
        self.equal = harness.load(FIXTURES / "equal.json")
        self.calls = []

    def evaluate(self, observations=None, **kwargs):
        observations = self.equal if observations is None else observations

        def fetch(profile, body, case_id, depth):
            self.calls.append((copy.deepcopy(profile), copy.deepcopy(body), case_id, depth))
            return copy.deepcopy(observations[case_id + ":" + depth])

        return harness.evaluate(self.recording, self.profiles, fetch, lookback=2, tokens=8, **kwargs)

    def test_equal_runs_and_exact_request_contract(self):
        report = self.evaluate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "pass")
        case, = report["cases"]
        self.assertEqual(case["near_tie_offset"], 2)
        self.assertTrue(case["source_reproduced"])
        self.assertEqual([v["configured_depth"] for v in case["variants"]], [1, 4, 8])
        self.assertTrue(all(v["token_equal"] for v in case["variants"]))
        self.assertEqual(len(self.calls), 4)
        for index, (profile, body, name, depth) in enumerate(self.calls):
            self.assertEqual(body["prompt"], [1, 2])
            self.assertEqual(body["max_tokens"], 8)
            self.assertEqual(body["temperature"], 0)
            self.assertIs(body["draft"], index > 0)
            self.assertIs(body["return_token_ids"], True)
            self.assertIs(body["ignore_eos"], True)
            self.assertIs(body["stream"], False)
            self.assertEqual(body["model"], profile["model"])
            self.assertNotIn("draft_depth", body)
            self.assertEqual(name, "fixture:2")
            self.assertEqual(depth, ["plain", "1", "4", "8"][index])
        self.assertEqual(harness.summary(report),
                         "DRAFT_DIVERGENCE status=pass cases=1 comparisons=3 divergences=0")

    def test_divergence_at_token_n_reports_both_raw_captures(self):
        report = self.evaluate(harness.load(FIXTURES / "divergent.json"))
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "divergent")
        difference = report["cases"][0]["variants"][1]["first_divergence"]
        self.assertEqual((difference["output_index"], difference["absolute_position"]), (2, 4))
        self.assertEqual((difference["plain_token_id"], difference["drafted_token_id"]), (12, 99))
        self.assertEqual(difference["plain_top2"]["ids"], [12, 99])
        self.assertEqual(difference["drafted_top2"]["ids"], [99, 12])
        self.assertEqual(difference["plain_top2"]["logits"], [16.0, 15.9375])
        self.assertEqual(difference["drafted_top2"]["logits"], [16.0, 15.9375])
        self.assertEqual(difference["label"], "near-tie")
        self.assertIs(difference["strict_equal"], False)
        self.assertIn('first="fixture:2"@2 depth=4', harness.summary(report))

    def test_bf16_boundary_is_inclusive_and_uses_absolute_raw_logits(self):
        row = self.recording["traces"][0]["top2"][2]
        row["logits"] = [16, 15.875]
        case, = harness.select_prompts(self.recording, lookback=1, tokens=8)
        self.assertEqual(case["bf16_spacing"], 0.125)
        self.assertEqual(case["prompt"], [1, 2, 10])
        self.assertEqual(case["near_tie_offset"], 1)
        row["logits"][1] = 15.874
        with self.assertRaisesRegex(ValueError, "no recorded BF16 near-ties"):
            harness.select_prompts(self.recording)
        self.assertTrue(harness.near({"logits": [16, 15.9375]}))
        self.assertFalse(harness.near({"logits": [-0.68, -0.7425]}))

    def test_nearest_supported_caps_are_reported(self):
        for profile, depth in zip(self.profiles, [1, 3, 6]):
            profile["depth"] = depth
        report = self.evaluate()
        self.assertTrue(report["passed"])
        variants = report["cases"][0]["variants"]
        self.assertEqual([(v["requested_depth"], v["configured_depth"]) for v in variants],
                         [(1, 1), (4, 3), (8, 6)])
        self.assertIn("not per-request", variants[0]["depth_control"])

    def test_no_draft_work_never_passes(self):
        for value in (0, None):
            observations = copy.deepcopy(self.equal)
            observations["fixture:2:4"]["response"]["speculative"]["drafted"] = value
            with self.subTest(value=value):
                report = self.evaluate(observations)
                self.assertEqual(report["status"], "incomplete")
                self.assertFalse(report["passed"])

    def test_equal_early_stops_do_not_pass(self):
        observations = copy.deepcopy(self.equal)
        for observation in observations.values():
            response = observation["response"]
            response["tensorfold"]["token_ids"] = response["tensorfold"]["token_ids"][:2]
            response["usage"]["completion_tokens"] = 2
            response["choices"][0]["finish_reason"] = "stop"
            observation["top2"] = observation["top2"][:2]
        report = self.evaluate(observations)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["passed"])

    def test_length_difference_has_missing_id_and_fails(self):
        observations = copy.deepcopy(self.equal)
        observation = observations["fixture:2:4"]
        observation["response"]["tensorfold"]["token_ids"].pop()
        observation["response"]["usage"]["completion_tokens"] -= 1
        observation["response"]["choices"][0]["finish_reason"] = "stop"
        observation["top2"].pop()
        report = self.evaluate(observations)
        self.assertEqual(report["status"], "divergent")
        difference = report["cases"][0]["variants"][1]["first_divergence"]
        self.assertEqual(difference["output_index"], 7)
        self.assertEqual(difference["plain_token_id"], 17)
        self.assertIsNone(difference["drafted_token_id"])

    def test_live_shape_without_raw_captures_labels_historical_logits(self):
        observations = harness.load(FIXTURES / "divergent.json")
        for observation in observations.values():
            observation.pop("top2")
        report = self.evaluate(observations)
        difference = report["cases"][0]["variants"][1]["first_divergence"]
        self.assertFalse(report["passed"])
        self.assertEqual(difference["label"], "recorded-near-tie")
        self.assertIsNone(difference["plain_top2"])
        self.assertIsNone(difference["drafted_top2"])
        self.assertEqual(difference["recorded_logits_for_tokens"], {"plain": 16.0, "drafted": 15.9375})
        self.assertIn("recorded source", difference["logit_source"])

    def test_changed_plain_history_does_not_reuse_source_logits(self):
        observations = harness.load(FIXTURES / "divergent.json")
        for observation in observations.values():
            observation.pop("top2")
            observation["response"]["tensorfold"]["token_ids"][0] = 77
        report = self.evaluate(observations)
        difference = report["cases"][0]["variants"][1]["first_divergence"]
        self.assertFalse(difference["recorded_history_matches"])
        self.assertEqual(difference["recorded_logits_for_tokens"], {"plain": None, "drafted": None})
        self.assertEqual(difference["label"], "divergence")

    def test_nonreproduced_source_never_passes_even_when_modes_agree(self):
        observations = copy.deepcopy(self.equal)
        for observation in observations.values():
            observation.pop("top2")
            observation["response"]["tensorfold"]["token_ids"][0] = 77
        report = self.evaluate(observations)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["cases"][0]["source_reproduced"])

    def test_malformed_responses_refused(self):
        mutations = [
            lambda r: r["tensorfold"].pop("token_ids"),
            lambda r: r["tensorfold"]["token_ids"].__setitem__(0, True),
            lambda r: r["tensorfold"].__setitem__("drafts", True),
            lambda r: r["usage"].__setitem__("completion_tokens", 7),
            lambda r: r.__setitem__("model", "wrong"),
            lambda r: r["choices"][0].__setitem__("finish_reason", "error"),
            lambda r: r.__setitem__("error", "failure"),
            lambda r: r["speculative"].__setitem__("drafted", 2),
        ]
        for index, mutate in enumerate(mutations):
            observations = copy.deepcopy(self.equal)
            mutate(observations["fixture:2:plain"]["response"])
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self.evaluate(observations)

    def test_bad_recorded_logits_refused(self):
        mutations = [
            lambda r: r.__setitem__("logits", [15, 16]),
            lambda r: r.__setitem__("logits", [float("nan"), 15]),
            lambda r: r.__setitem__("logits", [float("inf"), 15]),
            lambda r: r.__setitem__("ids", [12, 12]),
            lambda r: r.__setitem__("ids", [13, 99]),
            lambda r: r.__setitem__("position", 999),
        ]
        for index, mutate in enumerate(mutations):
            recording = copy.deepcopy(self.recording)
            mutate(recording["traces"][0]["top2"][2])
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                harness.select_prompts(recording)

    def test_bad_profiles_refused(self):
        mutations = [
            lambda p: p[1]["identity"].__setitem__("tokenizer", "different"),
            lambda p: p[1].__setitem__("depth", 1),
            lambda p: p[1].__setitem__("url", p[0]["url"] + "/"),
            lambda p: p[1].__setitem__("url", p[0]["url"] + "/v1/completions"),
            lambda p: p[1].__setitem__("url", "http://secret@fixture.invalid"),
            lambda p: p[1].__setitem__("url", "http://fixture.invalid?key=secret"),
        ]
        for index, mutate in enumerate(mutations):
            profiles = copy.deepcopy(self.profiles)
            mutate(profiles)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                harness.profiles_for(profiles, self.recording["identity"], (1, 4, 8))

    def test_distinct_prefix_selection_is_bounded(self):
        trace = self.recording["traces"][0]
        trace["top2"][3]["logits"] = [16, 15.99]
        cases = harness.select_prompts(self.recording, limit=1, lookback=0, tokens=4)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["id"], "fixture:3")
        self.assertEqual(cases[0]["prompt"], [1, 2, 10, 11, 12])
        self.assertEqual(cases[0]["tokens"], [13, 14, 15, 16])

    def test_bad_per_run_capture_refused(self):
        observations = copy.deepcopy(self.equal)
        observations["fixture:2:4"]["top2"][2]["ids"] = [99, 12]
        with self.assertRaisesRegex(ValueError, "argmax disagree"):
            self.evaluate(observations)

    def test_redirect_is_refused(self):
        with self.assertRaisesRegex(ValueError, "redirect refused"):
            harness.NoRedirect().redirect_request(None, None, 302, "found", {}, "http://fixture.invalid")

    def test_cli_replay_is_offline_and_exit_codes_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for filename, code, status in [("equal.json", 0, "pass"), ("divergent.json", 1, "divergent")]:
                stdout = io.StringIO()
                with patch.object(harness, "live_fetch", side_effect=AssertionError("unexpected HTTP")):
                    with contextlib.redirect_stdout(stdout):
                        actual = harness.main(["--recording", str(FIXTURES / "recording.json"), "--profiles",
                                               str(FIXTURES / "profiles.json"), "--replay", str(FIXTURES / filename),
                                               "--lookback", "2", "--tokens", "8", "--output", str(output)])
                self.assertEqual(actual, code)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], status)
                self.assertEqual(report["mode"], "recorded-replay")
                self.assertEqual(len(stdout.getvalue().splitlines()), 1)
                self.assertIn("input_sha256", report)

    def test_cli_errors_are_json_and_one_summary_line(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = harness.main(["--recording", str(FIXTURES / "recording.json"), "--profiles",
                                     str(FIXTURES / "profiles.json"), "--depths", "1,1", "--replay",
                                     str(FIXTURES / "equal.json"), "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "error")
            self.assertEqual(len(stdout.getvalue().splitlines()), 1)

    def test_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "nonfinite.json"
            source.write_text('{"logit": NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                harness.load(source)

    def test_served_client_posts_exact_ids_without_network(self):
        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return json.dumps(self_response).encode()

        self_response = self.equal["fixture:2:plain"]["response"]
        case, = harness.select_prompts(self.recording, lookback=2, tokens=8)
        body = harness.request(case, self.profiles[0], False)
        with patch.object(harness.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = Reply()
            observation = harness.live_fetch(12)(self.profiles[0], body, case["id"], "plain")
            (request,), kwargs = build.return_value.open.call_args
            self.assertEqual(request.full_url, "http://fixture.invalid:8001/v1/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(json.loads(request.data), body)
            self.assertEqual(kwargs, {"timeout": 12})
            self.assertEqual(observation["response"], self_response)


if __name__ == "__main__":
    unittest.main()
