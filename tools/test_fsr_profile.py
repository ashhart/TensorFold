import json
import unittest
from fsr_profile import TraceError, analyze


def fixture():
    out = [{"format": "tensorfold-fsr-v1", "layers": 2, "experts": 4,
            "slot_bytes": 10, "workload": "synthetic", "checkpoint": "synthetic-v1",
            "modes": {"target": [0, 1]}}]
    for wid in range(3):
        out += [{"event": "begin", "window": wid, "mode": "target"}]
        for layer in range(2):
            out += [{"event": "route", "window": wid, "layer": layer, "ids": [0, 1, 0]}]
        out += [{"event": "end", "window": wid}]
    return out + [{"event": "closed", "windows": 3}]


def run(events, budgets=(40,)):
    return analyze([json.dumps(x) for x in events], budgets, 10)


class LocalityTests(unittest.TestCase):
    def test_hit_observations_preserve_union_and_whole_budget(self):
        result = run(fixture(), (41, 20, 10))
        self.assertEqual(result["observed_union_bytes"], 40)
        fits, pressure, refused = result["frontier"]
        self.assertEqual(fits["misses"], 4)
        self.assertEqual([x["misses"] for x in fits["per_window"]], [4, 0, 0])
        self.assertEqual(fits["unused_tail_bytes"], 1)
        self.assertTrue(fits["observed_union_fits"])
        self.assertEqual(pressure["misses"], 12)
        self.assertFalse(pressure["observed_union_fits"])
        self.assertFalse(refused["supported_window_fits"])
        self.assertNotIn("misses", refused)
        self.assertEqual(pressure["storage_seconds_lower_bound"], 12)

    def test_truncation_missing_layer_duplicate_and_false_close_refuse(self):
        cases = [fixture()[:-1], fixture()[0:1] + fixture()[2:],
                 fixture()[:3] + fixture()[2:3] + fixture()[3:],
                 fixture()[:3] + fixture()[4:]]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(TraceError):
                run(events)
        events = fixture()
        events[-1]["windows"] = 100
        with self.assertRaises(TraceError):
            run(events)

    def test_bad_metadata_and_forecasts_cannot_claim_actual_union(self):
        for key, value in (("slot_bytes", 0), ("experts", True), ("workload", "")):
            events = fixture()
            events[0][key] = value
            with self.subTest(key=key), self.assertRaises(TraceError):
                run(events)
        events = fixture()
        events[2]["event"] = "forecast"
        with self.assertRaises(TraceError):
            run(events)
        events = fixture()
        events[2]["ids"] = [4]
        with self.assertRaises(TraceError):
            run(events)


if __name__ == "__main__":
    unittest.main()
