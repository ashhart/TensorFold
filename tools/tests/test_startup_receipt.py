"""Startup receipts reject incomplete loads and clean up only their owned child."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import startup_receipt as run
import startup_receipt_format as fmt


def record(event="ready"):
    rows = []
    for name in fmt.PHASES:
        at = 100 if name == "server_ready" else 10
        pending = name in ("first_completion", "metal_source_build", "metal_pipeline_build")
        rows.append({"phase": name, "start_ns": None if pending else 0,
                     "end_ns": None if pending else at, "total_ns": 0 if pending else at,
                     "calls": 0 if pending else 1})
    value = {"startup": {"schema": 1, "event": event, "deployment": "qwen27-metal", "clock": "awake_monotonic",
                         "origin": "native_main_entry", "units": "ns", "inclusive": True, "at_ns": 100,
                         "phases": rows}, "port": 12345, "token_sha256": None,
             "resources": {"physical_footprint_bytes": 10, "lifetime_peak_physical_footprint_bytes": 20,
                           "disk_io_bytes_read": 30}}
    if event == "first_completion":
        value["startup"]["at_ns"] = 200
        rows[-1].update(start_ns=120, end_ns=200, total_ns=80, calls=1)
        value["port"], value["token_sha256"] = None, "a" * 64
    return value


def line(value):
    return fmt.PREFIX + json.dumps(value).encode() + b"\n"


class Parsing(unittest.TestCase):
    def test_complete_pair_and_ordinary_logs(self):
        ready, done = record(), record("first_completion")
        self.assertIsNone(fmt.parse_line(b"ordinary server log\n"))
        self.assertEqual(fmt.parse_line(line(ready)), ready)
        self.assertEqual(fmt.paired(ready, done, "a" * 64)["reply_token_sha256"], "a" * 64)
        done["startup"]["phases"][-3].update(start_ns=150, end_ns=160, total_ns=10, calls=1)
        fmt.paired(ready, done, "a" * 64)

    def test_bad_timestamps_resources_and_incomplete_loads(self):
        for field in ("end_ns", "start_ns"):
            bad = record()
            bad["startup"]["phases"][3][field] = None
            with self.assertRaises(fmt.InvalidReceipt):
                fmt.parse_line(line(bad))
        cases = []
        bad = record(); bad["startup"]["at_ns"] = True; cases.append(bad)
        bad = record(); bad["startup"]["phases"][0]["end_ns"] = 101; cases.append(bad)
        bad = record(); bad["startup"]["phases"].pop(); cases.append(bad)
        bad = record(); bad["resources"]["disk_io_bytes_read"] = -1; cases.append(bad)
        bad = record(); bad["resources"]["lifetime_peak_physical_footprint_bytes"] = 9; cases.append(bad)
        bad = record(); bad["startup"]["schema"] = 1.0; cases.append(bad)
        bad = record(); bad["port"] = True; cases.append(bad)
        bad = record(); bad["startup"]["inclusive"] = 1; cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(fmt.InvalidReceipt):
                fmt.parse_line(line(bad))
        for raw in (b"{", b"null", b"{}", b"[]"):
            with self.assertRaises(fmt.InvalidReceipt):
                fmt.parse_line(fmt.PREFIX + raw)

    def test_mismatched_hash_and_mutated_startup(self):
        ready, done = record(), record("first_completion")
        with self.assertRaises(fmt.InvalidReceipt):
            fmt.paired(ready, done, "b" * 64)
        done["startup"]["phases"][3]["end_ns"] += 1
        with self.assertRaises(fmt.InvalidReceipt):
            fmt.paired(ready, done, "a" * 64)

    def test_real_response_hash_is_required(self):
        response = {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 64},
                    "tensorfold": {"token_sha256": "a" * 64}}
        self.assertEqual(fmt.completion_hash(json.dumps(response)), "a" * 64)
        for changed in (dict(response, error={}), dict(response, choices=[{"finish_reason": None}]),
                        dict(response, usage={"completion_tokens": 0}), dict(response, tensorfold={"token_sha256": "short"})):
            with self.assertRaises(fmt.InvalidReceipt):
                fmt.completion_hash(json.dumps(changed))


class ChildFailures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, source, timeout=1):
        return run.attempt([sys.executable, "-c", source], dict(os.environ), self.root / "attempt", timeout, 0.1)

    def test_load_failure_has_no_ready_and_child_is_reaped(self):
        result = self.attempt("import sys; print('load failed', file=sys.stderr); sys.exit(1)")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["ready"])
        self.assertIsNone(result["first_completion"])
        self.assertTrue(result["child_reaped"])

    def test_timeout_stops_child_and_preserves_unrelated_process(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            result = self.attempt("import time; time.sleep(60)", timeout=0.05)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["ready"])
            self.assertTrue(result["child_reaped"])
            self.assertIsNone(other.poll())
        finally:
            other.terminate(); other.wait(timeout=5)

    def test_malformed_ready_is_not_a_ready_event(self):
        result = self.attempt("print('startup-receipt {}', flush=True)")
        self.assertIsNone(result["ready"])
        self.assertTrue(result["child_reaped"])

    def test_http_failure_and_interrupt_clean_up_owned_child(self):
        source = "import time; print(" + repr(line(record()).decode().strip()) + ", flush=True); time.sleep(60)"
        for error in (run.RunFailed("HTTP failed"), KeyboardInterrupt()):
            folder = self.root / "attempt"
            if folder.exists():
                folder.rename(self.root / "previous")
            with mock.patch.object(run, "request", side_effect=error):
                result = self.attempt(source)
            self.assertNotEqual(result["status"], "ok")
            self.assertTrue(result["child_reaped"])
            self.assertIsNone(result["first_completion"])

    def test_success_uses_exact_response_hash_and_reaps_child(self):
        ready, done = record(), record("first_completion")
        source = "import time; print(" + repr(line(ready).decode().strip()) + ", flush=True); print(" + repr(
            line(done).decode().strip()) + ", flush=True); time.sleep(60)"
        body = json.dumps({"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 64},
                           "tensorfold": {"token_sha256": "a" * 64}}).encode()
        with mock.patch.object(run, "request", side_effect=(b"{}", body)) as request:
            result = self.attempt(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reply_token_sha256"], "a" * 64)
        self.assertTrue(result["child_reaped"])
        self.assertEqual(request.call_args_list[1].args[3], "/v1/completions")

    def test_repeated_ready_is_refused(self):
        source = "import time; print(" + repr(line(record()).decode().strip()) + ", flush=True); print(" + repr(
            line(record()).decode().strip()) + ", flush=True); time.sleep(60)"
        body = json.dumps({"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 64},
                           "tensorfold": {"token_sha256": "a" * 64}}).encode()
        with mock.patch.object(run, "request", side_effect=(b"{}", body)):
            result = self.attempt(source)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["child_reaped"])

    def test_cold_case_never_launches_or_purges(self):
        out = self.root / "cold"
        with mock.patch.object(run.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            self.assertEqual(run.main(["--binary", "absent", "--model", "absent", "--output", str(out),
                                       "--case", "cold"]), 2)
        self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "unsupported")

    def test_prewarmer_hashes_bytes_without_changing_weights(self):
        model = self.root / "model"; model.mkdir()
        config = {"hidden_size": 5120, "num_hidden_layers": 64, "quantization": {"bits": 4, "group_size": 64}}
        (model / "config.json").write_text(json.dumps(config))
        (model / "model.safetensors").write_bytes(b"synthetic host fixture")
        before = (model / "model.safetensors").read_bytes()
        first, second = run.warm_checkpoint(model), run.warm_checkpoint(model)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual((model / "model.safetensors").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
