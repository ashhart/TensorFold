"""Invalid benchmark workloads fail before sending requests or writing receipts."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_openai
from test_bench_auth import response


class WorkloadArguments(unittest.TestCase):
    def assert_invalid(self, module, argument, message):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = [module.__name__, "http://unused", "fixture", "--reps", "1", "--tokens", "1",
                    "--temperatures", "0", "--output", str(output), argument]
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(module.urllib.request, "urlopen") as http, \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exited:
                    module.main()
            http.assert_not_called()
            self.assertEqual(exited.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(output.exists())

    def test_nonpositive_tokens_and_repetitions(self):
        for module in (bench_openai, bench_concurrent):
            for option in ("tokens", "reps"):
                for value in (0, -1):
                    with self.subTest(client=module.__name__, option=option, value=value):
                        self.assert_invalid(module, f"--{option}={value}", f"--{option} must be greater than 0")

    def test_every_concurrency_level_must_be_a_positive_integer(self):
        for levels in ("0", "-1", "1,0", "1,-1", "-1,2", "", ",", "1,", ",1", "1,,2", "x", "1.5"):
            with self.subTest(levels=levels):
                self.assert_invalid(bench_concurrent, f"--levels={levels}",
                                    "--levels must be a comma-separated list of positive integers")

    def test_positive_workloads_still_send_requests_and_write_receipts(self):
        for module in (bench_openai, bench_concurrent):
            with self.subTest(client=module.__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                argv = [module.__name__, "http://unused", "fixture", "--reps", "1", "--tokens", "1",
                        "--temperatures", "0", "--output", str(output)]
                if module is bench_concurrent:
                    argv.extend(["--levels", " 1 , 2 "])
                with patch.object(sys, "argv", argv), \
                        patch.object(module.urllib.request, "urlopen", side_effect=lambda *a, **k: response()) as http, \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                self.assertEqual(http.call_count, 4 if module is bench_openai else 8)
                self.assertTrue(json.loads(output.read_text()))
                if module is bench_concurrent:
                    self.assertEqual([c["streams"] for c in json.loads(output.read_text())["cells"]], [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
