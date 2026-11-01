"""The DSpark matrix verdict: below-plain cells show, and unverified, mismatched or missing evidence fails."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_lfm2_dspark_verdict as q

TEMPS, WIDTHS, SEED = [0.0, 1.0], [1, 4, 8, 16], 1234
META = {"label": "QUALIFICATION", "smoke": False, "binary": {"sha256": "a" * 64},
        "source_revision": "test-source", "target": {"revision": "test-target"},
        "drafter": {"revision": "test-drafter"},
        "settings": {"reps": 3, "widths": WIDTHS, "temperatures": TEMPS, "prompts": ["code", "chat"],
                     "tokens": 512, "seed": SEED, "rotation_balanced": True}}
DRAFTED = {"rounds": 3, "drafted": 9, "accepted": 7}
NONE = {"rounds": 0, "drafted": 0, "accepted": 0}
WEIGHTS = {"plain": None, "bf16": 40.0, "bits8": 21.5}
READY = [{"event": "ready", "label": f"r{rep}-{arm}", "arm": arm,
          "dspark_mode": None if arm == "plain" else arm,
          "dspark_weights_mib": WEIGHTS[arm], "binary_sha256": META["binary"]["sha256"]}
         for rep in range(3) for arm in q.ARMS]
SHUTDOWN = [{"event": "shutdown", "label": e["label"], "forced_kill": False, "returncode": 0} for e in READY]
EVENTS = READY + SHUTDOWN


def digest(temp, seed):
    return f"{seed:032x}{int(temp):032x}"


def request(temp, seed, **fields):
    return {"seed": seed, "tokens": 512, "complete": True, "error": None, "unmeasured": None, "decode_tps": 50.0,
            "token_sha256": digest(temp, seed), "activity": NONE, **fields}


def evidence(speed, arm_activity=lambda arm: DRAFTED):
    """Every expected cell and oracle record of a clean run; speed(arm, streams) is each rep's aggregate tps."""
    out = []
    for rep in range(3):
        for arm in q.ARMS:
            activity = NONE if arm == "plain" else arm_activity(arm)
            for prompt in META["settings"]["prompts"]:
                for t in TEMPS:
                    seeds = list(dict.fromkeys(q.seeds_for(SEED, WIDTHS, t)))
                    out += [{"kind": "oracle", "rep": rep, "arm": arm, "prompt": prompt, "temperature": t, "seed": sd,
                             "solo_token_sha256": digest(t, sd), "draft_false_token_sha256": digest(t, sd),
                             "activity": activity, "failures": [], "ok": True} for sd in seeds]
                    for w in WIDTHS:
                        out.append({"kind": "cell", "rep": rep, "arm": arm, "prompt": prompt, "temperature": t,
                                    "streams": w, "aggregate_tps": speed(arm, w), "activity": activity, "failures": [],
                                    "ok": True, "requests": [request(t, sd, activity=activity)
                                                             for sd in q.seeds_for(SEED, WIDTHS, t)[:w]]})
    return out


def faster(arm, w):
    return 100.0 if arm == "plain" else 110.0


def whats(summary):
    return {f["what"] for f in summary["failures"]}


class Verdict(unittest.TestCase):
    def test_clean_run_with_every_cell_at_or_above_plain_passes(self):
        out = q.summarize(META, evidence(faster), EVENTS)
        self.assertEqual((out["verdict"], out["exit_code"], out["failures"]), ("pass", q.EXIT_PASS, []))
        self.assertEqual(out["performance"]["negative_cells"], [])

    def test_one_cell_below_plain_is_reported_whatever_the_others_gain(self):
        slower = lambda arm, w: 100.0 if arm == "plain" else 99.0 if (arm, w) == ("bits8", 4) else 150.0  # noqa: E731
        out = q.summarize(META, evidence(slower), EVENTS)
        self.assertEqual(out["exit_code"], q.EXIT_PERF)
        self.assertEqual({(n["arm"], n["streams"]) for n in out["performance"]["negative_cells"]},
                         {("bits8", 4)})

    def test_no_oracle_records_never_pass(self):
        out = q.summarize(META, [r for r in evidence(faster) if r["kind"] == "cell"], EVENTS)
        self.assertEqual(out["exit_code"], q.EXIT_FAIL)
        self.assertIn("missing_oracles", whats(out))

    def test_one_missing_oracle_or_cell_fails(self):
        records = evidence(faster)
        gone = next(r for r in records if r["kind"] == "oracle" and r["arm"] == "bf16")
        self.assertIn("missing_oracles", whats(q.summarize(META, [r for r in records if r is not gone], EVENTS)))
        self.assertIn("missing_cells", whats(q.summarize(META, records[:-1], EVENTS)))

    def test_cross_arm_hash_mismatch_fails_as_exactness(self):
        for kind, field in (("oracle", "solo_token_sha256"), ("oracle", "draft_false_token_sha256")):
            records = evidence(faster)
            bad = next(r for r in records if r["kind"] == kind and r["arm"] == "bits8")
            bad[field] = "f" * 64
            out = q.summarize(META, records, EVENTS)
            self.assertEqual(out["exit_code"], q.EXIT_FAIL)
            self.assertIn("cross_arm_mismatch", whats(out))
            self.assertGreater(out["exactness"]["failures"], 0)
        records = evidence(faster)
        cell = next(r for r in records if r["kind"] == "cell" and r["arm"] == "bf16")
        cell["requests"][0]["token_sha256"] = "e" * 64
        self.assertIn("cross_arm_mismatch", whats(q.summarize(META, records, EVENTS)))

    def test_unverifiable_cross_arm_hashes_fail(self):
        records = evidence(faster)
        for r in records:
            if r["arm"] == "plain" and r["kind"] == "oracle" and r["temperature"] == 1.0 and r["seed"] == SEED:
                r["solo_token_sha256"] = r["draft_false_token_sha256"] = None
        for r in records:
            if r["arm"] == "plain" and r["kind"] == "cell" and r["temperature"] == 1.0:
                r["requests"][0]["token_sha256"] = None
        self.assertIn("cross_arm_no_plain_hash", whats(q.summarize(META, records, EVENTS)))

    def test_a_cell_record_claiming_ok_is_checked_against_its_requests(self):
        for change in ({"tokens": 100}, {"complete": False}, {"token_sha256": None}, {"decode_tps": None},
                       {"error": "URLError"}, {"seed": 7}):
            records = evidence(faster)
            cell = next(r for r in records if r["kind"] == "cell" and r["arm"] == "plain" and r["streams"] == 4)
            cell["requests"][1] = {**cell["requests"][1], **change}
            out = q.summarize(META, records, EVENTS)
            self.assertEqual(out["exit_code"], q.EXIT_FAIL, change)

    def test_drafter_arms_need_some_speculative_activity_and_plain_may_have_it(self):
        out = q.summarize(META, evidence(faster, lambda arm: NONE if arm == "bits8" else DRAFTED), EVENTS)
        self.assertEqual((out["exit_code"], whats(out)), (q.EXIT_FAIL, {"no_speculative_activity"}))
        records = evidence(faster)
        for r in records:  # core copy proposals may run with no external drafter: recorded, never rejected
            if r["arm"] == "plain":
                r["activity"] = DRAFTED
        out = q.summarize(META, records, EVENTS)
        self.assertEqual(out["exit_code"], q.EXIT_PASS)

    def test_startup_logs_must_state_each_arms_dspark_mode(self):
        def ready(**by_arm):
            return [{**e, "dspark_mode": by_arm.get(e["arm"], e["dspark_mode"])} for e in READY] + SHUTDOWN
        cases = {"plain_arm_loaded_dspark": ready(plain="bf16"),
                 "dspark_mode_log_missing_or_wrong": ready(bits8="bf16")}
        for what, events in cases.items():
            self.assertIn(what, whats(q.summarize(META, evidence(faster), events)), what)
        for events in (READY[:-1], [], ready(bf16="unrecognized")):
            self.assertIn("dspark_mode_log_missing_or_wrong", whats(q.summarize(META, evidence(faster), events)))


    def test_unbalanced_run_cannot_claim_qualification(self):
        meta = {**META, "settings": {**META["settings"], "reps": 2, "rotation_balanced": False}}
        records = [r for r in evidence(faster) if r["rep"] < 2]
        events = READY[:6] + SHUTDOWN[:6]
        self.assertIn("unbalanced_rotation", whats(q.summarize(meta, records, events)))
        self.assertEqual(q.summarize({**meta, "smoke": True}, records, events)["exit_code"], q.EXIT_PASS)

    def test_each_launch_must_prove_the_same_binary(self):
        for sha in (None, "b" * 64):
            events = [{**e, "binary_sha256": sha} if i == 1 else e for i, e in enumerate(READY)]
            self.assertIn("binary_changed", whats(q.summarize(META, evidence(faster), events + SHUTDOWN)))

    def test_abort_and_forced_shutdown_fail(self):
        for event in ({"event": "abort", "detail": "x"}, {"event": "shutdown", "forced_kill": True}):
            self.assertEqual(q.summarize(META, evidence(faster), EVENTS + [event])["exit_code"], q.EXIT_FAIL)

    def test_shutdown_must_be_clean_and_account_for_every_launch(self):
        for code in (-11, 1, None):
            events = READY + [{**e, "returncode": code} if i == 0 else e for i, e in enumerate(SHUTDOWN)]
            self.assertIn("unclean_shutdown", whats(q.summarize(META, evidence(faster), events)))
        for events in (READY, EVENTS[:-1], READY + SHUTDOWN[:-1] + [SHUTDOWN[0]]):
            self.assertIn("missing_or_duplicate_shutdown", whats(q.summarize(META, evidence(faster), events)))

    def test_partial_persisted_matrix_cannot_be_relabelled_qualification(self):
        for change in ({"widths": [1, 4]}, {"tokens": 64}, {"prompts": ["code"]}, {"temperatures": [0.0]}):
            meta = {**META, "settings": {**META["settings"], **change}}
            records = evidence(faster)
            if "tokens" in change:
                for record in records:
                    for run in record.get("requests", []):
                        run["tokens"] = change["tokens"]
            self.assertIn("incomplete_qualification_matrix", whats(q.summarize(meta, records, EVENTS)))

    def test_qualification_without_source_or_model_revision_fails(self):
        for field in ("source_revision", "target", "drafter"):
            for missing in (None, "unknown"):
                meta = {**META, field: missing if field == "source_revision" else {"revision": missing}}
                self.assertIn("missing_revision", whats(q.summarize(meta, evidence(faster), EVENTS)))


class Requests(unittest.TestCase):
    def test_only_observed_complete_hashed_full_length_runs_are_clean(self):
        good = {"tokens": 512, "complete": True, "token_sha256": "a" * 64, "seed": 1, "decode_tps": 5.0}
        self.assertFalse(q.problems(good, 512, True))
        for change in ({"tokens": 511}, {"token_sha256": None}, {"complete": False}, {"decode_tps": None},
                       {"error": "URLError"}):
            self.assertTrue(q.problems({**good, **change}, 512, True), change)
        self.assertFalse(q.problems({**good, "decode_tps": None}, 512, False))

    def test_unequal_or_unverified_hashes_are_never_a_pass(self):
        ok = {"complete": True, "token_sha256": "a" * 64}
        self.assertIsNone(q.compare(ok, {**ok}, "x", 1))
        self.assertEqual(q.compare(ok, {**ok, "token_sha256": "b" * 64}, "x", 1)["class"], "exactness")
        self.assertEqual(q.compare(ok, {**ok, "token_sha256": None}, "x", 1)["class"], "scenario")
        self.assertEqual(q.compare(ok, {**ok, "complete": False}, "x", 1)["class"], "scenario")



if __name__ == "__main__":
    unittest.main()
