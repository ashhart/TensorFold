"""bench_concurrent.py and image_cache_turns.py against the fake served API: comparisons that compared nothing say so.

bench_concurrent is a reporter: its ``alone`` and ``serial`` counts carry ``vacuous`` when no two streams decoded at
once, or no draft landed, or the plain runs drafted. image_cache_turns' resumed == fresh fails when the fresh server
resumed or no later turn did.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_served_api import fake  # noqa: E402

TOOLS = HERE.parent


def concurrent(fault=None):
    """Each cell's ``alone`` and ``serial`` counts from a two-stream greedy run with ``--alone --serial``.

    512 tokens keep each stream ~0.1 s long at the fake's token gap, so the two streams overlap on a slow runner too
    (20 Nov: at 64 tokens a hosted Windows runner decoded them one after the other and main went red)."""
    with tempfile.TemporaryDirectory() as folder, fake(False, fault) as url:
        out = Path(folder) / "out.json"
        subprocess.run([sys.executable, str(TOOLS / "bench_concurrent.py"), url, "fake", "--levels", "2", "--reps",
                        "1", "--temperatures", "0", "--tokens", "512", "--alone", "--serial", "--output", str(out)],
                       capture_output=True, text=True, timeout=120, check=True)
        return [(c["alone"], c["serial"]) for c in json.loads(out.read_text())["cells"]]


def turns(cached_fault=None, fresh_cache=False):
    """image_cache_turns' exit code and FAIL lines, the fresh side a cache-off fake unless ``fresh_cache``."""
    with tempfile.TemporaryDirectory() as folder, fake(True, cached_fault) as url, fake(fresh_cache) as fresh:
        image = Path(folder) / "a.png"
        image.write_bytes(b"\x89PNG")
        done = subprocess.run([sys.executable, str(TOOLS / "image_cache_turns.py"), "--url", url, "--fresh-url",
                               fresh, "--image", str(image), "--turns", "3", "--tokens", "8"],
                              capture_output=True, text=True, timeout=120)
    return done.returncode, [line for line in done.stdout.splitlines() if line.startswith("FAIL ")]


class ConcurrentEvidenceTests(unittest.TestCase):
    def test_a_healthy_server_proves_both(self):
        for alone, serial in concurrent():
            self.assertNotIn("vacuous", alone)
            self.assertNotIn("vacuous", serial)
            self.assertGreater(serial["accepted"], 0)

    def test_a_server_serving_one_at_a_time_is_named(self):
        for alone, _ in concurrent("serial"):
            self.assertIn("no two streams decoding at once", alone.get("vacuous", ""))

    def test_undrafted_or_plain_drafting_is_named(self):
        for fault, words in (("undrafted", "accepted no draft"), ("plain_drafts", "runs drafted")):
            with self.subTest(fault=fault):
                for _, serial in concurrent(fault):
                    self.assertIn(words, serial.get("vacuous", ""))


class ImageTurnsTests(unittest.TestCase):
    def test_a_resuming_server_against_a_fresh_one_passes(self):
        self.assertEqual(turns(), (0, []))

    def test_a_server_that_never_resumed_fails_by_name(self):
        code, failed = turns("resumed")
        self.assertEqual(code, 1)
        self.assertEqual([line.split(":")[0] for line in failed], ["FAIL text", "FAIL image", "FAIL repeat"])
        self.assertTrue(all("compared fresh with fresh" in line for line in failed), failed)

    def test_a_fresh_server_that_resumed_fails_by_name(self):
        code, failed = turns(fresh_cache=True)
        self.assertEqual(code, 1)
        self.assertTrue(failed and all("(not fresh)" in line for line in failed), failed)


if __name__ == "__main__":
    unittest.main()
