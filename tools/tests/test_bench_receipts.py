"""Receipts written by origin/main 857b11bb8's clients still read, and new receipts keep every key they had."""
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
import bench_metrics
import bench_openai
import bench_thinking

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OLD = {"openai": FIXTURES / "receipt-openai-857b11bb8.json",
       "concurrent": FIXTURES / "receipt-concurrent-857b11bb8.json"}


def wire(*_args, **_kwargs):
    """The reply reports/eng01/capture_old_receipts.py served to main's clients for the fixtures."""
    chunks = [{"choices": [{"index": 0, "text": t}]} for t in ("Fib", "onacci numbers", " add", " the two before")]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 18, "completion_tokens": 9,
                                            "prompt_tokens_details": {"cached_tokens": 0}},
                   "tensorfold": {"token_sha": "c" * 12, "token_sha256": "c" * 64}})
    return io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n")


def write(module, argv):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        argv = [module.__name__, "http://127.0.0.1:9", "fixture", *argv, "--output", str(path)]
        with patch.object(sys, "argv", argv), \
                patch.object(module.urllib.request, "urlopen", side_effect=wire), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            module.main()
        return json.loads(path.read_text()), [json.loads(line) for line in out.getvalue().splitlines()]


def old(name):
    return json.loads(OLD[name].read_text())


class OldReceipts(unittest.TestCase):
    def test_comparison_reader_takes_old_receipts(self):
        for name, path in OLD.items():
            with self.subTest(receipt=name):
                rows = bench_thinking.load_comparison(str(path))
                self.assertTrue(rows)
                self.assertTrue(bench_thinking.comparison(rows[0], rows)["thinking_comparable"])

    def test_clients_compare_against_old_receipts(self):
        _, printed = write(bench_openai, ["--reps", "1", "--temperatures", "0",
                                          "--compare-receipt", str(OLD["openai"])])
        self.assertTrue(all(row["thinking_comparable"] for row in printed))
        _, printed = write(bench_concurrent, ["--levels", "1,2", "--reps", "1", "--temperatures", "0",
                                              "--compare-receipt", str(OLD["concurrent"])])
        self.assertTrue(all(cell["thinking_comparable"] for cell in printed))

    def test_new_receipts_keep_every_old_key(self):
        new, _ = write(bench_openai, ["--reps", "2", "--temperatures", "0"])
        for before, after in zip(old("openai"), new):
            self.assertEqual(set(before) - set(after), set())
            self.assertEqual(set(before["evidence"][0]) - set(after["evidence"][0]), set())
        new, _ = write(bench_concurrent, ["--levels", "1,2", "--reps", "1", "--temperatures", "0"])
        self.assertEqual(set(old("concurrent")) - set(new), set())
        for before, after in zip(old("concurrent")["cells"], new["cells"]):
            for part in ("request", "pooled"):
                self.assertEqual(set(before[part]) - set(after[part]), set())
            self.assertEqual(set(before) - set(after), set())
            for key in ("requests", "evidence", "client_intervals"):
                self.assertEqual(set(before["reps"][0][key][0]) - set(after["reps"][0][key][0]), set())
            self.assertEqual(set(before["reps"][0]["pooled"]) - set(after["reps"][0]["pooled"]), set())

    def test_deprecated_keys_are_marked_and_carry_the_new_value(self):
        new, _ = write(bench_concurrent, ["--levels", "2", "--reps", "1", "--temperatures", "0"])
        self.assertEqual(set(new["deprecated_keys"]), set(bench_metrics.DEPRECATED))
        self.assertGreater(new["collector_us_per_chunk"], 0)
        for cell in new["cells"]:
            self.assertEqual(cell["request"]["prefill_tps"], cell["request"]["prompt_ttft_tps"])
            self.assertEqual(cell["pooled"]["aggregate_prefill_tps"], cell["pooled"]["aggregate_prompt_ttft_tps"])
            self.assertEqual(cell["completions"], {"unfinished": 2})
        rows, _ = write(bench_openai, ["--reps", "1", "--temperatures", "0"])
        for row in rows:
            self.assertIn("prefill_tps", row["deprecated_keys"])
            self.assertEqual(row["prefill_tps_median"], row["prompt_ttft_tps_median"])

    def test_old_receipts_show_the_padded_itl_that_new_ones_leave_unknown(self):
        before = old("concurrent")["cells"][0]["request"]
        self.assertEqual((before["itl_p50_ms"], before["tokens_per_chunk"]), (0.0, 2.25))
        new, _ = write(bench_concurrent, ["--levels", "1", "--reps", "1", "--temperatures", "0"])
        after = new["cells"][0]["request"]
        self.assertIsNone(after["itl_p50_ms"])
        self.assertIsNotNone(after["chunk_tpot_estimate_ms"])
        self.assertEqual(after["tokens_per_chunk"], 2.25)


if __name__ == "__main__":
    unittest.main()
