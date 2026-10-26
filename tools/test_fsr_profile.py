import json
import unittest
from fsr_profile import TraceError, analyze
from fsr_synthetic import generate


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
    def test_synthetic_routes_are_complete_repeatable_and_preserve_raw_rows(self):
        for locality in ("uniform", "hot80", "sticky"):
            events = list(generate(2, locality, 1039, rows=3, streams=2, prompt=True))
            self.assertEqual(events, list(generate(2, locality, 1039, rows=3, streams=2, prompt=True)))
            self.assertEqual(events[0]["modes"], {"prompt": list(range(48)), "mtp": [48]})
            self.assertEqual(events[0]["slot_bytes"], 3072000)
            for route in (x for x in events if x.get("event") == "route"):
                self.assertEqual(len(route["ids"]), 30)
                for at in range(0, 30, 10):
                    self.assertEqual(len(set(route["ids"][at:at + 10])), 10)
            result = analyze([json.dumps(x) for x in events], [16 * 1024**3], tokens=6)
            self.assertEqual(result["windows"], 4)
            self.assertTrue(result["frontier"][0]["supported_window_fits"])

    def test_uniform_component_io_model_counts_coalescing_and_labels_predictions(self):
        events = fixture()
        events[0]["slot_bytes"] = 36
        events[0]["component_bytes"] = [4] * 9
        got = analyze([json.dumps(x) for x in events], [144], 1000, tokens=3, resident_tps=100,
                      read_latency_ns=100000)["frontier"][0]
        self.assertEqual(got["demands"], 12)
        self.assertAlmostEqual(got["hit_rate"], 2 / 3)
        self.assertEqual(got["component_reads_without_coalescing"], 36)
        self.assertEqual(got["coalesced_read_requests"], 18)
        self.assertEqual(got["mean_read_bytes"], 8)
        self.assertEqual(got["cold_bytes_per_token"], 48)
        self.assertGreater(got["serialized_batched_tps"], got["serialized_baseline_tps"])
        self.assertIn("uncalibrated", got["prediction"])

    def test_malformed_io_geometry_and_unmeasured_rates_refuse(self):
        for parts in ([4] * 8, [True] * 9, [4] * 9):
            events = fixture()
            events[0]["component_bytes"] = parts
            with self.subTest(parts=parts), self.assertRaises(TraceError):
                run(events)
        for kwargs in (dict(tokens=0), dict(resident_tps=float("nan")), dict(read_latency_ns=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(TraceError):
                analyze([json.dumps(x) for x in fixture()], [40], **kwargs)

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

    def test_native_batch_rows_and_streams_remain_visible(self):
        events = fixture()
        events[1].update(rows=8, streams=3)
        result = run(events)
        self.assertEqual(result["growth"][0]["rows"], 8)
        self.assertEqual(result["growth"][0]["streams"], 3)
        for key, bad in (("rows", True), ("streams", 0)):
            events = fixture()
            events[1][key] = bad
            with self.subTest(key=key), self.assertRaises(TraceError):
                run(events)

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
