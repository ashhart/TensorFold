"""CPU fixtures ensure a red prompt cannot hide later prompts or become a passing near-tie."""
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen27_margins import bf16_spacing, verdict

spec = importlib.util.spec_from_file_location("gate", Path(__file__).with_name("qwen27_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTests(unittest.TestCase):
    def test_near_tie_is_still_a_strict_failure(self):
        result = verdict({"ids": [421, 314], "logits": [10.0, 9.9375]},
                         {"ids": [314, 421], "logits": [10.0, 9.9375]}, 314, 421)
        self.assertEqual(result["verdict"], "near-tie")
        self.assertFalse(result["token_equal"])
        large = verdict({"ids": [421, 314], "logits": [10.0, 9.0]},
                        {"ids": [314, 421], "logits": [10.0, 9.0]}, 314, 421)
        self.assertEqual(large["verdict"], "bug")
        self.assertEqual(bf16_spacing(10.0), 0.0625)

    def test_all_prompts_run_after_red_and_error_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            wanted = [0] * 256
            wanted[42] = 314
            names = ["copy", "story", "arithmetic", "code"]
            record = {"version": "0.6.6", "drafts": False, "prompts": names, "results": {}}
            for name in names:
                record["results"][name] = {"tokens": wanted, "top2": [{"ids": [314, 421], "logits": [10.0, 9.9375]}] * 256}
            (out / "oracle.json").write_text(json.dumps(record))
            called = []

            def run(command, **_):
                name = Path(command[command.index("--tokens") + 1]).name.split(".")[0]
                called.append(name)
                result = {"tokens": list(wanted), "prompt_state_byte_equal": True, "first_difference": None}
                code = 0
                if name == "story":
                    result["tokens"][42] = 421
                    result["first_difference"] = {"position": 42, "top2": {"ids": [421, 314], "logits": [10.0, 9.9375]}}
                    code = 1
                if name == "arithmetic":
                    return argparse.Namespace(stdout="", stderr="native error", returncode=1)
                return argparse.Namespace(stdout=json.dumps(result), stderr="", returncode=code)

            args = argparse.Namespace(out=out, binary=Path("fake"), model=Path("unused"), no_speed=True)
            with patch.object(gate.subprocess, "run", run), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "All four"):
                    gate.check(args)
            self.assertEqual(called, names)
            report = json.loads((out / "token-gate.json").read_text())
            self.assertEqual(report["story"]["diagnostic"]["first_mismatch"], 42)
            self.assertEqual(report["story"]["diagnostic"]["verdict"], "near-tie")
            self.assertEqual(report["code"]["diagnostic"]["verdict"], "equal")


if __name__ == "__main__":
    unittest.main()
