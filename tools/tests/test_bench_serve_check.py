"""zig/tests/qwen35/serve_check.py against the fake served API: clean it passes, and each planted fault fails by name."""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_served_api import fake

SCRIPT = Path(__file__).resolve().parents[2] / "zig" / "tests" / "qwen35" / "serve_check.py"


def run(fault=None):
    """The script's exit code and its FAIL lines, against a cached fake with ``fault`` and a cache-off fake."""
    with fake(True, fault) as url, fake(False) as fresh:
        done = subprocess.run([sys.executable, str(SCRIPT), url, "fake", "8", fresh], capture_output=True, text=True,
                              timeout=120)
    return done.returncode, [line for line in done.stdout.splitlines() if line.startswith("FAIL")]


class ServeCheckTests(unittest.TestCase):
    def test_clean_fake_passes(self):
        self.assertEqual(run(), (0, []))

    def test_a_resume_that_resumed_nothing_fails(self):
        # The vacuous pass of #2460: a later turn that resumed no tokens compared two fresh replies.
        code, failed = run("resumed")
        self.assertEqual(code, 1)
        self.assertEqual([line.split(":")[0] for line in failed], ["FAIL resumed == fresh"], failed)
        self.assertIn("resumed no tokens", failed[0])

    def test_unexercised_drafts_fail(self):
        for fault, labels in (("plain_drafts", ["greedy", "t0.8"]), ("sampled_drafts", ["t0.8"])):
            with self.subTest(fault=fault):
                code, failed = run(fault)
                self.assertEqual(code, 1)
                self.assertEqual([line.split(":")[0] for line in failed], [f"FAIL {x}" for x in labels], failed)
                self.assertTrue(all("drafted == serial" in line for line in failed), failed)

    def test_a_sampler_that_never_ran_fails(self):
        code, failed = run("unsampled")
        self.assertEqual(code, 1)
        self.assertEqual(failed, ["FAIL t0.8: the same twice, unlike greedy"])

    def test_empty_replies_are_never_equal(self):
        code, failed = run("empty")
        self.assertEqual(code, 1)
        self.assertGreaterEqual(len(failed), 10, failed)


if __name__ == "__main__":
    unittest.main()
