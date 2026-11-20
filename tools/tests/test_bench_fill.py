"""bench_fill's waits and exactness verdict, without a model."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_fill


def receipt(sent, times, sha="a" * 64, cached=0, ttft=None):
    return {"sent": sent, "completed": times[-1] + 0.25, "pieces": [(t, 1) for t in times],
            "ttft_s": times[0] - sent if ttft is None else ttft, "token_sha": sha[:12], "token_sha256": sha,
            "cached_tokens": cached, "complete": True, "error": None, "tokens": len(times),
            "prefill_seconds": 4.0, "time_to_first_token": 4.5, "prefill_widths": [1024, 1024]}


def orders(reply_sha="a" * 64, long_cached=0):
    alone = {"reply": receipt(0.0, [1.0, 2.0]), "long": receipt(0.0, [5.0, 6.0], sha="b" * 64)}
    first = {"reply": receipt(10.0, [10.5, 11.0, 13.0], sha=reply_sha),
             "long": receipt(10.5, [20.5, 21.0], sha="b" * 64, cached=long_cached)}
    second = {"reply": receipt(35.0, [36.0, 37.0]), "long": receipt(30.0, [40.0, 41.0], sha="b" * 64)}
    return alone, first, second


def trace(long_rows=2048, between=True):
    events = []

    def add(kind, id_, to, from_=0):
        events.append({"t0_ns": len(events), "t1_ns": len(events), "kind": kind, "id": id_, "stream": id_,
                       "rows_from": from_, "rows_to": to, "ok": True})

    for id_, rows in [(1, 10), (2, long_rows), (3, 10), (4, 10), (5, long_rows), (6, long_rows), (7, 10)]:
        add("submit", id_, rows)
    for long_id, reply_id in [(5, 4), (6, 7)]:
        half = long_rows // 2
        add("fill", long_id, half)
        if between:
            add("decode", reply_id, 1)
        add("fill", long_id, long_rows, half)
        if not between:
            add("decode", reply_id, 1)
    return events


class Fill(unittest.TestCase):
    def test_waits(self):
        report = bench_fill.summary(*orders())
        self.assertEqual(report["alone"], {"reply_s": 2.0, "long_ttft_s": 5.0})
        self.assertEqual(report["reply_first"]["long_native_prefill_wall_s"], 4.0)
        self.assertEqual(report["reply_first"]["long_server_ttft_s"], 4.5)
        self.assertEqual(report["reply_first"]["long_wall_s"], 10.75)
        self.assertEqual(report["reply_first"]["long_prefill_widths"], [1024, 1024])
        self.assertEqual(report["mixed_load"], "unknown")
        self.assertEqual(report["reply_first"]["reply_s"], 3.0)
        self.assertEqual(report["reply_first"]["long_ttft_s"], 10.0)
        self.assertEqual(report["reply_first"]["both_done_s"], 11.0)
        self.assertEqual(report["long_first"]["both_done_s"], 11.0)
        self.assertEqual(report["reply_first"]["reply_gap_max_ms"], 2000.0)

    def test_owned_decode_between_fills_proves_both_mixed_orders(self):
        report = bench_fill.summary(*orders(), trace=trace(), long_rows=2048)
        self.assertEqual(report["mixed_load"], "verified_mixed")
        self.assertEqual(report["reply_first"]["native_mix"], "verified_mixed")
        self.assertEqual(report["long_first"]["native_mix"], "verified_mixed")

    def test_serialized_control_cannot_masquerade_as_mixed_load(self):
        report = bench_fill.summary(*orders(), trace=trace(between=False), long_rows=2048)
        self.assertEqual(report["mixed_load"], "verified_serialized")

    def test_short_control_is_not_a_long_mixed_load(self):
        report = bench_fill.summary(*orders(), trace=trace(long_rows=512), long_rows=512)
        self.assertEqual(report["mixed_load"], "not_long")

    def test_legacy_rounds_without_owned_decode_are_unknown(self):
        events = [e for e in trace() if e["kind"] != "decode"]
        events.append({**events[-1], "kind": "round", "id": 0, "stream": 1})
        report = bench_fill.summary(*orders(), trace=events, long_rows=2048)
        self.assertEqual(report["mixed_load"], "unknown")

    def test_exact_and_cold(self):
        report = bench_fill.summary(*orders())
        self.assertTrue(report["exact"])
        self.assertTrue(report["cold"])

    def test_a_different_reply_is_not_exact(self):
        self.assertFalse(bench_fill.summary(*orders(reply_sha="c" * 64))["exact"])

    def test_a_missing_hash_is_not_exact(self):
        alone, first, second = orders()
        first["reply"]["token_sha256"] = None
        self.assertFalse(bench_fill.summary(alone, first, second)["exact"])

    def test_a_cached_long_prompt_is_not_cold(self):
        self.assertFalse(bench_fill.summary(*orders(long_cached=8))["cold"])

    def test_gaps_with_one_piece(self):
        self.assertEqual(bench_fill.largest_gaps([(1.0, 1)]), (None, None))


if __name__ == "__main__":
    unittest.main()
