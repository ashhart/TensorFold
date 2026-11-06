"""Temperature lists fail locally when invalid and preserve the selected workloads when valid."""
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

CLIENTS = (bench_openai, bench_concurrent)
INVALID = ("", " ", ",", ",1", "1,", "1,,0", "x", "0,x", "x,0", "1 0", "None",
           "nan", "NaN,0", "0,+inf", "-Infinity", "1e309", "0,-1e309")


def argv(module, output, extra):
    args = [module.__name__, "http://unused", "fixture", "--tokens", "4", "--reps", "1", "--output", str(output)]
    return args + (["--levels", "1"] if module is bench_concurrent else []) + list(extra)


class TemperatureArguments(unittest.TestCase):
    def assert_invalid(self, module, value, existing=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            original = b"previous benchmark report\n"
            if existing:
                output.write_bytes(original)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", argv(module, output, ["--temperatures=" + value])), \
                    patch.object(module.urllib.request, "urlopen") as http, \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exited:
                    module.main()
            http.assert_not_called()
            self.assertEqual(exited.exception.code, 2)
            self.assertIn("--temperatures", stderr.getvalue())
            self.assertIn("comma-separated finite numbers", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            if existing:
                self.assertEqual(output.read_bytes(), original)
            else:
                self.assertFalse(output.exists())

    def test_invalid_lists_fail_before_any_request_or_report(self):
        for module in CLIENTS:
            for value in INVALID:
                with self.subTest(client=module.__name__, value=value):
                    self.assert_invalid(module, value)

    def test_invalid_list_preserves_an_existing_report(self):
        for module in CLIENTS:
            with self.subTest(client=module.__name__):
                self.assert_invalid(module, "0,broken", existing=True)

    def test_valid_lists_preserve_order_duplicates_and_request_temperatures(self):
        cases = [((), [1.0, 0.0]), (("--temperatures=0",), [0.0]),
                 (("--temperatures= 0 , 1e-1, +1.5, 1, 1, -0.5, 3 ",), [0, 0.1, 1.5, 1, 1, -0.5, 3])]
        for module in CLIENTS:
            for extra, temperatures in cases:
                with self.subTest(client=module.__name__, extra=extra):
                    self.assert_workloads(module, extra, temperatures)

    def test_mixed_keeps_its_sampled_temperature_override(self):
        for extra in ([], ["--temperatures=0,0.5"], ["--temperatures=ignored"]):
            with self.subTest(extra=extra):
                self.assert_workloads(bench_concurrent, ["--mixed", *extra], [1.0], prompts=1)

    def assert_workloads(self, module, extra, temperatures, prompts=2):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            requests = []

            def open_request(request, **kwargs):
                requests.append(json.loads(request.data))
                return response()

            with patch.object(sys, "argv", argv(module, output, extra)), \
                    patch.object(module.urllib.request, "urlopen", side_effect=open_request), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            report = json.loads(output.read_text())
        cells = report["cells"] if module is bench_concurrent else report
        self.assertEqual([cell["temperature"] for cell in cells], [t for t in temperatures for _ in range(prompts)])
        self.assertEqual([r["temperature"] for r in requests], [t for t in temperatures for _ in range(prompts * 2)])
        for request in requests:
            self.assertEqual(request.get("top_k"), 20 if request["temperature"] > 0 else None)
            self.assertEqual(request.get("top_p"), 0.95 if request["temperature"] > 0 else None)


if __name__ == "__main__":
    unittest.main()
