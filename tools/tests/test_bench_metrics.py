"""ITL, round gaps and prefill rates follow the definitions in docs/cache-benchmarks.md."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_metrics as metrics
import bench_openai

FIGURES = ("prefill_tps", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "round_gap_p50_ms", "round_gap_p95_ms",
           "tokens_per_chunk")


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


def response(texts=("H", "ell", "o", " world"), completion=4, cached=0):
    chunks = [{"choices": [{"text": t}]} for t in texts]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": completion,
                                            "prompt_tokens_details": {"cached_tokens": cached}},
                   "tensorfold": {"token_sha": "a" * 12, "token_sha256": "a" * 64}})
    return io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n")


def clock(*times):
    """perf_counter replaced by these times in order, then the last one forever."""
    it = iter(times)
    last = [times[-1]]

    def now():
        last[0] = next(it, last[0])
        return last[0]
    return now


class Clients(unittest.TestCase):
    def test_bench_openai_times_each_chunk(self):
        with patch.object(bench_openai.urllib.request, "urlopen", return_value=response(completion=7)), \
                patch.object(bench_openai.time, "perf_counter", clock(0.0, 0.5, 0.6, 0.8, 0.9)):
            got = bench_openai.stream("http://unused", "fixture", bench_openai.PROMPTS[0], 7, 0, 1)
        self.assertEqual(len(got["pieces"]), 4)
        self.assertAlmostEqual(got["ttft_s"], 0.5)
        self.assertAlmostEqual(got["prefill_tps"], 8 / 0.5)
        self.assertAlmostEqual(got["itl_mean_ms"], 400 / 6)  # 3 chunk gaps and 3 zeros
        self.assertEqual(got["itl_p50_ms"], 0)
        self.assertAlmostEqual(got["round_gap_p50_ms"], 100)
        self.assertAlmostEqual(got["tokens_per_chunk"], 7 / 4)

    def test_bench_concurrent_times_each_chunk(self):
        with patch.object(bench_concurrent.urllib.request, "urlopen", return_value=response(cached=8)), \
                patch.object(bench_concurrent.time, "perf_counter", clock(0.0, 0.5, 0.6, 0.8, 0.9, 1.0)):
            got = bench_concurrent.stream("http://unused", "fixture", bench_concurrent.PROMPTS[0], 4, 0, 1)
        self.assertIsNone(got["prefill_tps"])  # 8 cached tokens: not a cold request
        self.assertAlmostEqual(got["itl_mean_ms"], 400 / 3)
        self.assertAlmostEqual(got["itl_p95_ms"], 200)
        self.assertEqual(got["tokens_per_chunk"], 1)

    def test_bench_openai_rows_add_medians_and_keep_their_fields(self):
        argv = ["bench_openai.py", "http://unused", "fixture", "--reps", "2", "--temperatures", "0"]
        with patch.object(sys, "argv", argv), \
                patch.object(bench_openai.urllib.request, "urlopen", side_effect=lambda *a, **k: response()), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            bench_openai.main()
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(rows), len(bench_openai.PROMPTS))
        for row in rows:
            for key in ("label", "prompt", "temperature", "decode_tps_median", "decode_tps_all", "ttft_s_median"):
                self.assertIn(key, row)
            self.assertEqual({k + "_median" for k in FIGURES} - set(row), set())
            self.assertEqual(row["tokens_per_chunk_median"], 1)

    def test_bench_concurrent_cells_add_request_and_pooled_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            argv = ["bench_concurrent.py", "http://unused", "fixture", "--levels", "2", "--reps", "2",
                    "--temperatures", "0", "--output", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.object(bench_concurrent.urllib.request, "urlopen", side_effect=lambda *a, **k: response()), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                bench_concurrent.main()
            report = json.loads(output.read_text())
        printed = [json.loads(line) for line in out.getvalue().splitlines()]
        for cell, line in zip(report["cells"], printed):
            for key in ("aggregate_tps", "per_stream_tps", "ttft_s_max", "timing_valid", "failed", "unmeasured"):
                self.assertIn(key, line)
            self.assertEqual(set(cell["request"]), {"ttft_s", "decode_tps", *FIGURES})
            self.assertEqual(cell["request"]["tokens_per_chunk"], 1)
            self.assertEqual(cell["pooled"]["tokens_per_chunk"], 1)
            self.assertIsNotNone(cell["pooled"]["aggregate_prefill_tps"])
            for rep in cell["reps"]:
                self.assertEqual(len(rep["requests"]), 2)
                self.assertEqual(rep["pooled"]["streams_timed"], 2)


if __name__ == "__main__":
    unittest.main()
