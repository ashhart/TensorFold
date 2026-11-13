"""The model-code sandbox: wall timeout, memory, output limits, an empty environment and the OS sandbox's denials."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_sandbox

UNIX = eval_sandbox.resource is not None


def sandbox(**limits) -> eval_sandbox.Sandbox:
    """The host's sandbox, or rlimits alone where it has none, so these tests run on any Unix runner."""
    return eval_sandbox.Sandbox.detect(unsafe=True, warn=lambda message: None, **limits)


@unittest.skipUnless(UNIX, "model code runs only where rlimits exist")
class Limits(unittest.TestCase):
    def test_an_infinite_loop_stops_at_the_wall_timeout(self):
        result = sandbox(timeout_s=1).run("while True:\n    pass\n")
        self.assertEqual(result["ended"], "timeout")
        self.assertTrue(result["error"])
        self.assertLess(result["seconds"], 10)
        self.assertIn("time limit", result["text"])

    def test_the_timeout_stops_the_programs_children_too(self):
        code = ("import subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(child.pid, flush=True)\n"
                "while True:\n    pass\n")
        result = sandbox(timeout_s=1).run(code)
        self.assertEqual(result["ended"], "timeout")
        pid = int(result["stdout"].split()[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
            if not state.stdout.strip() or state.stdout.strip().startswith("Z"):
                return
            time.sleep(0.05)
        self.fail("the program's child outlived the timeout")

    def test_memory_over_the_limit_is_an_error(self):
        result = sandbox(timeout_s=10, memory_mb=256).run("x = b'x' * (1 << 30)\nimport time\ntime.sleep(10)\n")
        self.assertTrue(result["error"])
        self.assertLess(result["seconds"], 10)
        self.assertTrue(result["ended"] == "memory" or "MemoryError" in result["stderr"], result)

    def test_exit_status_and_traceback_reach_the_model(self):
        result = sandbox().run("print('partial')\nimport sys\nsys.exit(3)\n")
        self.assertEqual((result["exit"], result["error"]), (3, True))
        self.assertEqual(result["text"], "partial\n[exit status 3]")
        result = sandbox().run("1 / 0\n")
        self.assertIn("ZeroDivisionError", result["text"])
        self.assertTrue(result["error"])

    def test_clean_run_and_silent_run(self):
        self.assertEqual(sandbox().run("print(6 * 7)\n")["text"], "42")
        silent = sandbox().run("x = 1\n")
        self.assertEqual((silent["text"], silent["error"]), (eval_sandbox.NO_OUTPUT, False))

    def test_output_swapped_for_a_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("do not read")
            code = (f"import os\nprint('mine', flush=True)\nos.remove('.stdout')\n"
                    f"os.symlink({str(secret)!r}, '.stdout')\n")
            result = sandbox().run(code)
        self.assertEqual(result["stdout"].strip(), "mine")

    def test_long_output_keeps_its_head_and_tail(self):
        result = sandbox(output_chars=1000).run("print('a' * 20000)\nraise SystemExit('the end')\n")
        self.assertLess(len(result["text"]), 1100)
        self.assertIn("characters cut", result["text"])
        self.assertTrue(result["text"].startswith("aaa"))
        self.assertIn("[exit status 1]", result["text"])

    def test_environment_holds_only_path_and_each_run_gets_a_new_directory(self):
        box = sandbox()
        code = ("import json, os\nopen('left.txt', 'w').write('x')\n"
                "print(json.dumps([sorted(os.environ), os.getcwd()]))\n")
        with patch.dict(os.environ, {"EVAL_SANDBOX_SECRET": "x", "HOME": "/nowhere"}):
            first = box.run(code)
        names, cwd = json.loads(first["stdout"])
        # macOS adds __CF_USER_TEXT_ENCODING to every process and Python sets LC_CTYPE when it coerces the C locale
        self.assertEqual(set(names) - {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}, {"PATH"})
        second = box.run("import json, os\nprint(json.dumps([os.path.exists('left.txt'), os.getcwd()]))\n")
        left, cwd2 = json.loads(second["stdout"])
        self.assertFalse(left)
        self.assertNotEqual(cwd, cwd2)
        self.assertFalse(os.path.exists(cwd))


@unittest.skipUnless(UNIX, "model code runs only where rlimits exist")
class Isolation(unittest.TestCase):
    def setUp(self):
        self.box = sandbox()
        if self.box.kind == "none":
            self.skipTest("this host has no sandbox")
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(self.listener.close)

    def test_the_network_is_denied(self):
        port = self.listener.getsockname()[1]
        code = f"import socket\nsocket.create_connection(('127.0.0.1', {port}), timeout=3)\nprint('connected')\n"
        result = self.box.run(code)
        self.assertNotIn("connected", result["stdout"])
        self.assertTrue(result["error"])

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec denies writes; unshare -n only the network")
    def test_writes_outside_the_directory_and_home_reads_are_denied(self):
        with tempfile.TemporaryDirectory() as outside:
            target = os.path.join(os.path.realpath(outside), "escaped.txt")
            result = self.box.run(f"open({target!r}, 'w').write('x')\n")
            self.assertIn("PermissionError", result["stderr"])
            self.assertFalse(os.path.exists(target))
        home = os.path.realpath(os.path.expanduser("~"))
        result = self.box.run(f"import os\nprint(os.listdir({home!r}))\n")
        self.assertIn("PermissionError", result["stderr"])


class Detection(unittest.TestCase):
    def test_no_sandbox_warns_then_refuses_unless_unsafe(self):
        if not UNIX:
            with self.assertRaises(eval_sandbox.SandboxUnavailable):
                eval_sandbox.Sandbox.detect(unsafe=True)
            return
        warned = []
        with patch.object(eval_sandbox.shutil, "which", return_value=None):
            with self.assertRaises(eval_sandbox.SandboxUnavailable):
                eval_sandbox.Sandbox.detect(warn=warned.append)
            box = eval_sandbox.Sandbox.detect(unsafe=True, warn=warned.append)
        self.assertEqual(box.kind, "none")
        self.assertEqual(len(warned), 2)
        self.assertIn("no sandbox", warned[0])

    @unittest.skipUnless(UNIX, "the stand-in unshare is a shell script")
    def test_linux_takes_unshare_n_then_falls_back_to_a_user_namespace(self):
        for refuse_plain, kind in ((False, "unshare -n"), (True, "unshare -r -n")):
            with tempfile.TemporaryDirectory() as bin_dir, self.subTest(kind=kind):
                fake = Path(bin_dir) / "unshare"
                refuse = '[ "$1" = "-n" ] && exit 1\n' if refuse_plain else ""
                fake.write_text(f"#!/bin/sh\n{refuse}exit 0\n")
                fake.chmod(0o755)
                path = bin_dir + os.pathsep + os.environ.get("PATH", "")
                with patch.object(eval_sandbox.sys, "platform", "linux"), patch.dict(os.environ, {"PATH": path}):
                    box = eval_sandbox.Sandbox.detect()
                self.assertEqual(box.kind, kind)
                self.assertEqual(box.command("/work"), [str(fake), *kind.split()[1:]])

    def test_mac_profile_denies_network_and_outside_writes(self):
        profile = eval_sandbox.mac_profile("/private/tmp/work dir")
        self.assertIn("(deny network*)", profile)
        self.assertIn("(deny file-write*)", profile)
        self.assertIn('(allow file-write* (subpath "/private/tmp/work dir")', profile)

    def test_cut_keeps_both_ends(self):
        self.assertEqual(eval_sandbox.cut("short", 10), "short")
        text = eval_sandbox.cut("h" * 50 + "t" * 50, 30)
        self.assertTrue(text.startswith("h" * 20) and text.endswith("t" * 10))
        self.assertIn("70 characters cut", text)


if __name__ == "__main__":
    unittest.main()
