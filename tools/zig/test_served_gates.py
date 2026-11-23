"""Served gate scripts against the fake served API: clean they pass, and each planted fault fails the gate by name."""
# drafted == plain: no draft, plain drafting or empty replies fail; cancel: too late, a held lane or bad health fail.
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
from fake_served_api import fake  # noqa: E402
import reuse_served  # noqa: E402

DRAFT_FAULTS = ("undrafted", "plain_drafts", "empty")


def script(name, *argv, timeout=120):
    """A tools/zig script's exit code and stdout."""
    done = subprocess.run([sys.executable, str(HERE / f"{name}.py"), *argv], capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout


def named(out, words):
    """The verdict of the check line that names ``words``: True for "ok", False for "FAIL", None when absent."""
    lines = [line for line in out.splitlines() if words in line]
    return None if not lines else all(line.startswith("ok") for line in lines)


class DraftedEqualsPlainTests(unittest.TestCase):
    def structured(self, fault=None):
        with fake(False, fault) as url:
            _, out = script("structured_served", "--url", url, "--tokens", "8", "--thinking-tokens", "8")
        rows = [line for line in out.splitlines() if "drafted==plain=" in line or "drafted == plain:" in line]
        return bool(rows) and not any("drafted==plain=False" in r or r.startswith("FAIL drafted") for r in rows)

    def test_structured_served(self):
        self.assertTrue(self.structured())
        for fault in (*DRAFT_FAULTS, "sampled_drafts"):
            with self.subTest(fault=fault):
                self.assertFalse(self.structured(fault))

    def vision(self, fault=None):
        with tempfile.TemporaryDirectory() as folder, fake(False, fault) as url:
            for name in ("a.png", "b.png"):
                (Path(folder) / name).write_bytes(b"\x89PNG" + name.encode())
            _, out = script("vision_served", "--url", url, "--images", folder, "--tokens", "8")
        return named(out, 'drafted equals "draft": false with an image')

    def test_vision_served(self):
        self.assertIs(self.vision(), True)
        for fault in DRAFT_FAULTS:
            with self.subTest(fault=fault):
                self.assertIs(self.vision(fault), False)

    def video(self, fault=None):
        ids = types.SimpleNamespace(tolist=lambda: [1, 2])

        class Processor:
            video_processor = None

            def apply_chat_template(self, *_, **__):
                return ""

            def __call__(self, **_):
                return {"input_ids": [ids]}
        stubs = {"glm_video_fixtures": types.SimpleNamespace(load=lambda *_: (None, None)),
                 "transformers": types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(
                     from_pretrained=lambda _: Processor()))}
        with patch.dict(sys.modules, stubs), tempfile.TemporaryDirectory() as clips, fake(False, fault) as url:
            for name in ("h264.mp4", "hevc.mp4", "h264-odd.mp4", "short.mp4"):
                (Path(clips) / name).write_bytes(name.encode())
            sys.modules.pop("video_served", None)
            video_served = importlib.import_module("video_served")
            out = io.StringIO()
            argv = ["video_served.py", "--url", url, "--model", "fake", "--clips", clips, "--tokens", "8"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                video_served.main()
            sys.modules.pop("video_served", None)
        return named(out.getvalue(), 'drafted equals "draft": false with a video')

    def test_video_served(self):
        self.assertIs(self.video(), True)
        for fault in DRAFT_FAULTS:
            with self.subTest(fault=fault):
                self.assertIs(self.video(fault), False)

    def xpu(self, fault=None):
        with fake(False, fault) as url:
            code, out = script("xpu_draft_receipt", url, "fake", "--tokens", "8", "--reps", "1")
        return code, out.splitlines()[-1]

    def test_xpu_draft_receipt_exits_1_on_fail(self):
        self.assertEqual(self.xpu(), (0, "DRAFT IDENTITY: PASS"))
        for fault in (*DRAFT_FAULTS, "sampled_drafts"):
            with self.subTest(fault=fault):
                code, last = self.xpu(fault)
                self.assertEqual(code, 1, last)
                self.assertTrue(last.startswith("DRAFT IDENTITY: FAIL"), last)


class ReuseCancelTests(unittest.TestCase):
    def cancel(self, fault=None, prefill=2.0):
        """Failures after scenario 5's cancel row, against a fake that holds each request ``prefill`` seconds."""
        with tempfile.TemporaryDirectory() as folder, fake(False, fault, prefill=prefill) as url:
            rec = reuse_served.Recorder(reuse_served.Client(url, "fake"), str(Path(folder) / "rows.jsonl"), 8)
            rec.c.SETTLE_S = 1
            conv = reuse_served.Conv("C", "You audit code.", "Audit it.", None, False, False)
            body = conv.body("fake", 8, False)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rec.row(conv, "cancel", body, rec.c.cancel(body, 0.2), None, 0)
            rec.out.close()
        return rec.failures, out.getvalue()

    def test_a_cut_in_prefill_holds(self):
        self.assertEqual(self.cancel()[0], 0)

    def test_a_cancel_that_cancelled_nothing_fails(self):
        failures, out = self.cancel(prefill=0)
        self.assertEqual(failures, 1, out)
        self.assertIn("the reply came before the cut", out)

    def test_a_cancel_that_holds_a_lane_or_breaks_health_fails(self):
        for fault, words in (("cancel_lane", "did not return to"), ("cancel_health", "/health not ok")):
            with self.subTest(fault=fault):
                failures, out = self.cancel(fault)
                self.assertEqual(failures, 1, out)
                self.assertIn(words, out)


if __name__ == "__main__":
    unittest.main()
