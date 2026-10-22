"""Workload timing includes every requested worker and the full response interval."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent as concurrent
import bench_cache as cache


def row(start=2, end=12, tokens=80, **fields):
    return dict(sent=start, completed=end, tokens=tokens, complete=True, decode_tps=None,
                prompt="code", seed=1234, first=start + 2, last=end - 1, pieces=[], ttft_s=2, **fields)


class CompleteTiming(unittest.TestCase):
    def test_overlapping_full_intervals_and_bursts(self):
        for rows, span, total in [([row(), row(7, 22, 120)], 20, 200), ([row()], 10, 80)]:
            result = concurrent.aggregates(rows)
            self.assertEqual(result.get("makespan_s"), span)
            self.assertEqual(result.get("requested_requests"), len(rows))
            self.assertTrue(result.get("timing_valid"))
            self.assertAlmostEqual(result["end_to_end_tps"], total / span)

    def test_partial_or_malformed_workers_invalidate_the_entire_workload(self):
        invalid = [dict(complete=False), dict(complete=1), dict(error="lost"), dict(tokens=0),
                   dict(tokens=True), dict(tokens=None), dict(tokens=2.5), dict(tokens=-1),
                   dict(sent=True), dict(sent=-1), dict(sent=10**1000), dict(completed=None),
                   dict(completed=float("nan")), dict(completed=float("inf")), dict(completed=2)]
        for fields in invalid:
            with self.subTest(fields=fields):
                result = concurrent.aggregates([row(), {**row(), **fields}])
                self.assertEqual(result.get("requested_requests"), 2)
                self.assertEqual(result.get("timed_requests"), 1)
                self.assertIs(result.get("timing_valid"), False)
                self.assertIsNone(result.get("end_to_end_tps"))
        self.assertIs(concurrent.aggregates([]).get("timing_valid"), False)

    def test_worker_exception_outside_stream_retains_its_order_and_seed(self):
        class Inline:
            def __init__(self, target, args):
                self.target, self.args = target, args
            def start(self):
                self.target(*self.args)
            def join(self):
                pass
        specs = [(concurrent.PROMPTS[0], n) for n in (8, 19, 31)]
        def fake_stream(*args, **kwargs):
            if args[5] == 19:
                raise RuntimeError("synthetic failure")
            return {**row(), "seed": args[5]}
        with patch.object(concurrent.threading, "Thread", Inline), patch.object(concurrent, "stream", fake_stream):
            runs = concurrent.together("unused", "fixture", specs, 80, 0)
        self.assertEqual([r["seed"] for r in runs], [8, 19, 31])
        self.assertEqual(runs[1]["error"], "RuntimeError")
        self.assertEqual(concurrent.aggregates(runs)["failed"], 1)
        self.assertIs(concurrent.aggregates(runs)["timing_valid"], False)

    def test_invalid_delay_starts_no_worker(self):
        for delay in [-1, True, float("nan"), float("inf")]:
            with patch.object(concurrent.threading, "Thread") as worker:
                with self.assertRaises(ValueError):
                    concurrent.together("unused", "fixture", [(concurrent.PROMPTS[0], 0)], 1, 0, delay)
                worker.assert_not_called()
        self.assertEqual(concurrent.together("unused", "fixture", [], 1, 0), [])

    def test_real_stream_records_chunk_arrivals_and_final_usage_delivery(self):
        class Response:
            def __enter__(self):
                chunks = [dict(choices=[dict(text="first")]), dict(choices=[dict(text="second")]),
                          dict(usage=dict(completion_tokens=80, prompt_tokens=10))]
                return iter([("data: " + json.dumps(c)).encode() for c in chunks] + [b"data: [DONE]"])
            def __exit__(self, *args):
                pass
        with patch.object(concurrent.urllib.request, "urlopen", return_value=Response()), \
                patch.object(concurrent.time, "perf_counter", side_effect=[10, 11, 14, 18]):
            run = concurrent.stream("http://unused", "fixture", concurrent.PROMPTS[0], 80, 0, 1)
        self.assertTrue(run["complete"])
        self.assertEqual(run.get("elapsed_s"), 8)
        self.assertEqual(run.get("chunk_offsets_s"), [1, 4])
        self.assertEqual(run.get("chunk_gaps_ms"), dict(count=1, p50=3000, p95=3000, max=3000))
        self.assertEqual(concurrent.aggregates([run])["end_to_end_tps"], 10)
        self.assertIsNone(run["decode_tps"])

    def test_cache_report_carries_concurrent_timing_and_observed_chunk_fields(self):
        good = {**row(), "cache_state": "reused", "elapsed_s": 10, "chunk_offsets_s": [2], "chunk_gaps_ms": None}
        with patch.object(cache.bench, "stream", return_value=good), \
                patch.object(cache.bench, "together", return_value=[good, row(4, 22, 120)]), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cache.main(["unused", "fixture", "--repeats", "1", "--streams", "2"]), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report.get("concurrent_timing", {}).get("makespan_s"), 20)
        self.assertEqual(report["rows"][0].get("chunk_offsets_s"), [2])
        self.assertEqual(report["rows"][0].get("elapsed_s"), 10)

    def test_cli_does_not_cherry_pick_successful_repetitions(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "receipts.json"
            argv = ["bench_concurrent", "unused", "fixture", "--levels", "1", "--temperatures", "0",
                    "--reps", "2", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(concurrent, "PROMPTS", [concurrent.PROMPTS[0]]), \
                    patch.object(concurrent, "together", side_effect=[[row()], [row()], [{**row(), "complete": False}]]), \
                    contextlib.redirect_stdout(io.StringIO()):
                concurrent.main()
            report = json.loads(output.read_text())["cells"][0]
        self.assertIs(report.get("timing_valid"), False)
        self.assertIsNone(report.get("end_to_end_tps"))
        interval = report["reps"][0]["client_intervals"][0]
        self.assertEqual([interval.get(k) for k in ("start", "first", "last", "end")], [2, 4, 11, 12])


if __name__ == "__main__":
    unittest.main()
