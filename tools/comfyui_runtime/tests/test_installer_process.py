from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tf_installer import build, pins

try:
    import fcntl
except ImportError:
    fcntl = None


@unittest.skipUnless(fcntl is not None, "Process group tests require Linux/POSIX")
class ProcessGroupTests(unittest.TestCase):
    def test_descendants_stop_after_success_failure_and_timeout(self):
        for mode in ("success", "failure", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "worker.py").write_text(
                    "import fcntl,os,pathlib,signal,time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "lock = open('worker.lock', 'w')\n"
                    "fcntl.flock(lock, fcntl.LOCK_EX)\n"
                    "pathlib.Path('pid').write_text(str(os.getpid()))\n"
                    "pathlib.Path('ready').touch()\n"
                    "time.sleep(60)\n")
                parent = (
                    "import pathlib,subprocess,sys,time\n"
                    "subprocess.Popen([sys.executable, 'worker.py'])\n"
                    "while not pathlib.Path('ready').exists(): time.sleep(0.01)\n"
                    "if sys.argv[1] == 'timeout': time.sleep(60)\n"
                    "raise SystemExit(7 if sys.argv[1] == 'failure' else 0)\n")
                try:
                    command = [sys.executable, "-c", parent, mode]
                    if mode == "success":
                        build.run_command(command, root, root / "run.log", timeout=5)
                    else:
                        expected = "time or log-size" if mode == "timeout" else "exited 7"
                        with self.assertRaisesRegex(pins.InstallError, expected):
                            build.run_command(command, root, root / "run.log", timeout=1 if mode == "timeout" else 5)
                    self.assertTrue((root / "ready").exists())
                    deadline = time.monotonic() + 3
                    with (root / "worker.lock").open("w") as lock:
                        while True:
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                self.assertLess(time.monotonic(), deadline)
                                time.sleep(0.01)
                finally:
                    if (root / "pid").is_file():
                        try:
                            os.kill(int((root / "pid").read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_original_failure_survives_cleanup_failure(self):
        stop = build.stop_process_group

        def fail_after_stop(process):
            stop(process)
            raise pins.InstallError("cleanup diagnostic")

        cases = (("raise SystemExit(7)", 5, "exited 7"), ("import time; time.sleep(60)", 0.1, "time or log-size"))
        for code, timeout, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory, patch.object(build, "stop_process_group", fail_after_stop), redirect_stderr(io.StringIO()) as diagnostic:
                root = Path(directory)
                with self.assertRaisesRegex(pins.InstallError, expected):
                    build.run_command([sys.executable, "-c", code], root, root / "run.log", timeout=timeout)
            self.assertIn("cleanup diagnostic", diagnostic.getvalue())

    def test_final_reap_has_a_bound(self):
        with patch.object(build.os, "killpg") as kill, patch.object(build.subprocess, "Popen") as constructor:
            process = constructor.return_value
            process.wait.side_effect = subprocess.TimeoutExpired("worker", 5)
            with self.assertRaisesRegex(pins.InstallError, "SIGKILL"):
                build.stop_process_group(process)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGKILL])
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [5])


if __name__ == "__main__":
    unittest.main()
