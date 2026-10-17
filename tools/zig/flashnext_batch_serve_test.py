"""CPU regressions for served-gate routing and honest qualification receipts."""
import contextlib
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import flashnext_batch_serve as gate


class ServedGateTests(unittest.TestCase):
    def exercise(self, modes="1,default,auto,4", shared=True, mismatch=False, startup_only=False,
                 shared_blocks=None, legacy=False, late_log=False):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        fixtures = root / "fixtures"
        fixtures.mkdir()
        for name in ("short", "thinking", "sparse", "sparse-thinking"):
            (fixtures / f"{name}.json").write_text(json.dumps({"text": name, "thinking": "thinking" in name}))
        output = root / "out"
        commands = []
        current = {}

        class Process:
            pid = 12345
            returncode = None

            def __init__(self, command, stdout, **kwargs):
                commands.append(command)
                current.update(command=command, log=stdout)
                if startup_only:
                    stdout.write("info: flashnext batch: streams 4, rows 8, shared rounds 99\n")
                    stdout.flush()

            def poll(self):
                return None

            def terminate(self):
                self.returncode = 0

            def wait(self, **kwargs):
                return 0

        def get(path):
            if path == "/health":
                return '{"status":"ok"}'
            return "tensorfold:requests_running 4\n"

        def request(case, tokens, barrier):
            barrier.wait(timeout=2)
            multi = "--parallel" not in current["command"] or current["command"][current["command"].index("--parallel") + 1] != "1"
            if shared and multi and (shared_blocks is None or len(commands) - 1 in shared_blocks):
                current["log"].write("info: flashnext batch: streams 4, rows 8, shared rounds 1\n")
                current["log"].flush()
            return {"reply": {"message": {"content": case["text"]}, "finish_reason": "length"},
                    "tokens": tokens, "prompt_tokens": 12, "cached_tokens": 0,
                    "token_sha": ("b" if mismatch and multi else "a") * 12, "seconds": 0.01}

        argv = ["gate", "binary", "model", str(fixtures), str(output), "--parallel", modes,
                "--context", "32768", "--rounds", "1", "--require-shared-rounds"]
        run = gate.run
        thread = gate.threading.Thread

        class MonitorThread(thread):
            def __init__(self, *args, **kwargs):
                self.monitor = getattr(kwargs.get("target"), "__name__", "") == "monitor"
                super().__init__(*args, **kwargs)

            def join(self, *args, **kwargs):
                if late_log and self.monitor:
                    current["log"].write("info: flashnext batch: streams 4, rows 8, shared rounds 99\n")
                    current["log"].flush()
                return super().join(*args, **kwargs)

        def invoke(args, slots, cases, block):
            if legacy:
                args = argparse.Namespace(**{k: v for k, v in vars(args).items()
                                             if k not in {"context", "require_shared_rounds"}})
                slots = int(slots)
            return run(args, slots, cases, block)

        try:
            with patch("sys.argv", argv), patch.object(gate.subprocess, "Popen", Process), \
                    patch.object(gate, "get", get), patch.object(gate, "request", request), \
                    patch.object(gate, "run", invoke), \
                    patch.object(gate.threading, "Thread", MonitorThread), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                gate.main()
        finally:
            self.receipt = json.loads((output / "receipt.json").read_text()) if output.exists() else None
            self.commands = commands
        return self.receipt

    def test_default_auto_and_fixed_commands_use_same_context_and_qualify(self):
        report = self.exercise()
        self.assertTrue(report["exact"])
        self.assertTrue(report["shared_rounds_verified"])
        self.assertEqual(report["context"], 32768)
        self.assertEqual(len(self.commands), 4)
        self.assertNotIn("--parallel", self.commands[1])
        self.assertEqual(self.commands[2][self.commands[2].index("--parallel") + 1], "auto")
        self.assertTrue(all(c[c.index("--context") + 1] == "32768" for c in self.commands))
        self.assertEqual(list(report["runs"]), ["1", "default", "auto", "4"])

    def test_running_requests_alone_cannot_prove_shared_rounds(self):
        with self.assertRaisesRegex(RuntimeError, "shared GPU rounds"):
            self.exercise(shared=False)
        self.assertFalse(self.receipt["exact"])
        self.assertFalse(self.receipt["shared_rounds_verified"])
        self.assertIn("default", self.receipt["runs"])

    def test_mismatch_preserves_failing_block_and_cannot_qualify(self):
        with self.assertRaisesRegex(RuntimeError, "differ"):
            self.exercise(mismatch=True)
        self.assertFalse(self.receipt["exact"])
        self.assertIn("default", self.receipt["runs"])

    def test_startup_rounds_cannot_qualify_requests(self):
        with self.assertRaisesRegex(RuntimeError, "shared GPU rounds"):
            self.exercise(shared=False, startup_only=True)
        self.assertFalse(self.receipt["shared_rounds_verified"])

    def test_rounds_after_requests_cannot_qualify_them(self):
        with self.assertRaisesRegex(RuntimeError, "shared GPU rounds"):
            self.exercise(shared=False, late_log=True)
        self.assertFalse(self.receipt["shared_rounds_verified"])

    def test_every_repeated_auto_block_needs_its_own_shared_rounds(self):
        with self.assertRaisesRegex(RuntimeError, "shared GPU rounds"):
            self.exercise(modes="1,auto,auto", shared_blocks={1})
        self.assertFalse(self.receipt["exact"])
        self.assertEqual([b["shared_rounds"] > 0 for b in self.receipt["runs"]["auto"]], [True, False])

    def test_legacy_integer_run_without_new_arguments_still_works(self):
        report = self.exercise(modes="1,4", legacy=True)
        self.assertTrue(report["exact"])
        self.assertTrue(all(c[c.index("--context") + 1] == "16384" for c in self.commands))

    def test_invalid_modes_fail_before_loading_a_model(self):
        for modes in ("auto,1", "1,17", "1,0", "1,typo", "1,"):
            with self.subTest(modes=modes), self.assertRaises(SystemExit):
                self.exercise(modes=modes)
            self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
