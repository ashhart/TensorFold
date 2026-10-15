"""Host measurement tests; fake native lines never qualify CUDA or model memory."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import memory_receipts as receipt


def fixture(rank=1):
    rows = []
    for i, phase in enumerate(["load_begin", "load_end", "cleanup_begin", "cleanup_end"]):
        rows.append({"schema": 1, "pid": 123, "rank": rank, "phase": phase, "reset": i == 0,
                     "monotonic_ns": i, "scope": "process_owned_cuda_buffers",
                     "usage": {"device": 0, "host": 0, "peak": 20, "host_peak": 10}})
    return rows


def lines(rows):
    return "\n".join("TF_MEMORY " + json.dumps(row) for row in rows)


class Receipts(unittest.TestCase):
    def test_both_ranks_require_complete_lifecycle(self):
        for rank in [0, 1]:
            rows = fixture(rank)
            self.assertEqual(receipt.native_receipts(lines(rows), rank)["status"], "recorded")
            self.assertEqual(receipt.native_receipts(lines(rows[:-1]), rank)["status"], "unmeasured")
            rows[0]["reset"] = False
            self.assertEqual(receipt.native_receipts(lines(rows), rank)["status"], "unmeasured")

    def test_invalid_identity_clock_peak_and_missing_fields(self):
        self.assertEqual(receipt.native_receipts(lines(fixture()), 1, 999)["status"], "unmeasured")
        for key, value in [("rank", 0), ("monotonic_ns", None), ("scope", "physical")]:
            rows = fixture()
            rows[1][key] = value
            self.assertEqual(receipt.native_receipts(lines(rows), 1)["status"], "unmeasured")
        rows = fixture()
        rows[1]["usage"]["host_peak"] = -1
        self.assertEqual(receipt.native_receipts(lines(rows), 1)["status"], "unmeasured")
        self.assertEqual(receipt.native_receipts("TF_MEMORY {}", 1)["status"], "unmeasured")

    def test_resets_are_retained_as_separate_intervals(self):
        rows = fixture()
        rows.insert(2, dict(rows[1], phase="reset", reset=True))
        got = receipt.native_receipts(lines(rows), 1)
        self.assertEqual(got["status"], "recorded")
        self.assertEqual(got["rows"][2]["phase"], "reset")

    @unittest.skipUnless(sys.platform == "linux", "Linux capture validation")
    def test_invalid_interval_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            for interval in [0, -1, float("nan"), float("inf")]:
                output = Path(tmp) / "must-not-exist"
                with self.assertRaises(ValueError):
                    receipt.capture(["must-not-run"], 0, 2048, output, interval)
                self.assertFalse(output.exists())

    @unittest.skipUnless(sys.platform == "linux", "Linux procfs/wait4 capture")
    def test_actual_child_memory_exit_and_missing_native_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            rc = receipt.capture([sys.executable, "-c", "x=bytearray(8000000); print(len(x))"], 1, 2048, output, .005)
            got = json.loads((output / "receipt.json").read_text())
            self.assertEqual(rc, 0)
            self.assertGreater(got["resident_peak"]["bytes"], 0)
            self.assertEqual(got["native"]["status"], "unmeasured")
            self.assertEqual(got["physical_footprint"]["status"], "unsupported")
            self.assertIsNone(got["physical_footprint"]["bytes"])
            self.assertLess(got["process_start_ns"], got["process_exit_ns"])
            with self.assertRaises(FileExistsError):
                receipt.capture(["unused"], 1, 2048, output, .1)

    @unittest.skipUnless(sys.platform == "linux", "Linux wait4 capture")
    def test_failed_child_cannot_record_qualified_native_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            code = ("import os,json; rows=" + repr(fixture()) +
                    "; [row.update(pid=os.getpid()) for row in rows]; "
                    "[print('TF_MEMORY '+json.dumps(row)) for row in rows]; raise SystemExit(7)")
            self.assertEqual(receipt.capture([sys.executable, "-c", code], 1, 2048, output, .01), 7)
            report = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt.native_receipts((output / "process.log").read_text(), 1, report["pid"])["status"], "recorded")
            self.assertEqual(report["native"]["status"], "unmeasured")


if __name__ == "__main__":
    unittest.main()
