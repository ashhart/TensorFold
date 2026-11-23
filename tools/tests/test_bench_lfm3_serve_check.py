"""tools/lfm3/serve_check.py on the fake served API: clean it passes; unexercised drafts or sampling fail by name."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_served_api import fake

SPEC = importlib.util.spec_from_file_location("lfm3_serve_check", Path(__file__).resolve().parents[1] / "lfm3" /
                                              "serve_check.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def fake_server(fault):
    """``serve_check.Server`` with a fake served API (with ``fault``) in place of the native binary."""

    class Fake(check.Server):
        def __enter__(self):
            self.command = ["fake", str(self.cache)]
            self.log = tempfile.NamedTemporaryFile(mode="w+b", dir=self.log_directory, delete=False)
            self.running = fake(self.cache > 0, fault)
            self.base = self.running.__enter__()
            return self

        def __exit__(self, *exc):
            self.running.__exit__(*exc)
            self.log.close()
    return Fake


def run(fault=None):
    """The exit code and the failed checks' names of one ``--sampled`` run against the fake."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps({"model_type": "lfm3"}))
        (root / "binary").write_bytes(b"fake")
        argv = ["serve_check.py", "--binary", str(root / "binary"), "--model-dir", str(root), "--output",
                str(root / "out.json"), "--sampled"]
        with mock.patch.object(check, "Server", fake_server(fault)), mock.patch.object(sys, "argv", argv), \
                mock.patch.object(check.tempfile, "mkdtemp", lambda prefix: tmp), \
                contextlib.redirect_stdout(io.StringIO()):
            code = check.main()
        receipt = json.loads((root / "out.json").read_text())
    return code, [t["name"] for t in receipt["tests"] if t["status"] == "failed"]


class Lfm3ServeCheckTests(unittest.TestCase):
    def test_clean_fake_passes(self):
        self.assertEqual(run(), (0, []))

    def test_unexercised_drafts_fail(self):
        for fault in ("undrafted", "plain_drafts"):
            with self.subTest(fault=fault):
                self.assertEqual(run(fault), (1, ["drafts/exercised"]))

    def test_a_sampler_that_never_ran_fails(self):
        self.assertEqual(run("unsampled"), (1, ["sampled_code/sampler_ran", "sampled_chat/sampler_ran"]))


if __name__ == "__main__":
    unittest.main()
