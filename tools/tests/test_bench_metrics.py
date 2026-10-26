"""ITL, round gaps and prefill rates follow the definitions in docs/cache-benchmarks.md."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_metrics as metrics


def run(times, tokens, **fields):
    base = dict(complete=True, tokens=tokens, pieces=[(t, 3) for t in times], sent=0.0, first=times[0],
                ttft_s=times[0], prompt_tokens=1000, cache_state="cold")
    return {**base, **fields}


class Definitions(unittest.TestCase):
    def test_one_token_a_chunk_gives_the_chunk_gaps(self):
        got = metrics.request(run([0.1, 0.2, 0.4, 0.5], 4))
        self.assertAlmostEqual(got["itl_mean_ms"], 400 / 3)
        self.assertAlmostEqual(got["itl_p50_ms"], 100)
        self.assertAlmostEqual(got["itl_p95_ms"], 200)
        self.assertEqual(got["tokens_per_chunk"], 1)
        self.assertAlmostEqual(got["prefill_tps"], 10000)

    def test_a_chunk_of_several_tokens_adds_zeros_and_keeps_the_mean(self):
        got = metrics.request(run([0.1, 0.4, 0.7], 7))  # 3 chunks, 7 tokens: 2 gaps of 300 ms and 4 zeros
        self.assertAlmostEqual(got["itl_mean_ms"], 600 / 6)
        self.assertEqual(got["itl_p50_ms"], 0)
        self.assertAlmostEqual(got["itl_p95_ms"], 300)
        self.assertAlmostEqual(got["round_gap_p50_ms"], 300)
        self.assertAlmostEqual(got["round_gap_mean_ms"], 300)
        self.assertAlmostEqual(got["tokens_per_chunk"], 7 / 3)

    def test_the_itl_mean_is_the_decode_span_over_tokens_after_the_first(self):
        times, tokens = [1.0, 1.05, 1.3, 1.31, 1.9], 11
        got = metrics.request(run(times, tokens))
        self.assertAlmostEqual(got["itl_mean_ms"], (times[-1] - times[0]) * 1000 / (tokens - 1))

    def test_percentiles_are_nearest_rank(self):
        values = [float(v) for v in range(1, 21)]
        self.assertEqual(metrics.rank(values, 0.5), 10)
        self.assertEqual(metrics.rank(values, 0.95), 19)
        self.assertEqual(metrics.rank([7.0], 0.95), 7)
        self.assertIsNone(metrics.rank([], 0.5))

    def test_unmatched_or_unverified_streams_give_no_figures(self):
        for fields in [dict(tokens=2, pieces=[(0.1, 1), (0.2, 1), (0.3, 1)]), dict(complete=False), dict(tokens=None),
                       dict(tokens=True), dict(tokens=1), dict(pieces=[(0.3, 1), (0.2, 1)]), dict(pieces=[])]:
            with self.subTest(fields=fields):
                self.assertIsNone(metrics.request({**run([0.1, 0.2], 5), **fields})["itl_mean_ms"])

    def test_prefill_rate_only_for_a_cold_request(self):
        for fields in [dict(cache_state="reused"), dict(cache_state="unverified"), dict(prompt_tokens=None),
                       dict(prompt_tokens=0), dict(ttft_s=None), dict(ttft_s=0.0), dict(ttft_s=True)]:
            with self.subTest(fields=fields):
                self.assertIsNone(metrics.request(run([0.1, 0.2], 5, **fields))["prefill_tps"])

    def test_pooled_samples_and_aggregate_prefill(self):
        a = run([1.0, 1.1], 2, sent=0.0, first=1.0)
        b = run([2.0, 2.3], 2, sent=0.5, first=2.0)
        got = metrics.pooled([a, b])
        self.assertEqual(got["streams_timed"], 2)
        self.assertAlmostEqual(got["itl_mean_ms"], 200)
        self.assertAlmostEqual(got["aggregate_prefill_tps"], 2000 / 2.0)
        self.assertIsNone(metrics.pooled([a, {**b, "cache_state": "reused"}])["aggregate_prefill_tps"])
        self.assertEqual(metrics.pooled([a, {**b, "complete": False}])["streams_timed"], 1)


if __name__ == "__main__":
    unittest.main()
