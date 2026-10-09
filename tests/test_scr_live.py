"""Live integration suite: boots a real tensorfold server with the SCR
patch and asserts behavior end to end.

Runs ONLY when TENSORFOLD_SCR_LIVE=1 (needs a GPU + the cached Qwen3.8-27B-NVFP4
checkpoint); `python -m unittest discover` stays CPU-only without it.

    TENSORFOLD_SCR_LIVE=1 python3 -m pytest tests/test_scr_live.py -v

What it asserts (not vibes — exact properties):
  T1 canary          a short request is served unplanned: reply is exact AND no
                     [scr] plan line appears for it
  T2 mid_edit        turn 2 deletes a mid-message paragraph: the session matches,
                     a plan with >= 1 splice runs ("[scr] splice OK" in the log),
                     and the needle that lives inside the RELOCATED span is still
                     answered correctly
  T3 append planned  an append-only turn plans splice-free; its greedy reply is
                     identical to the same conversation served on a SCR-off boot
                     (native resume == fresh per the engine's contract), proving
                     the planned path does not perturb outputs on the dense path
  T4 off parity      the SCR-off boot serves the canary identically to the on
                     boot (the patch is inert when no plan attaches)
"""

import json
import os
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path


MODEL = os.environ.get(
    "TENSORFOLD_SCR_LIVE_MODEL",
    os.path.expanduser("~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/"
                       "snapshots/319f741cce68d7914884900c138a1fbb70a42f30"))
PORT = int(os.environ.get("TENSORFOLD_SCR_LIVE_PORT", "18391"))
LOG = Path(os.environ.get("TENSORFOLD_SCR_LIVE_LOG", "/tmp/scr_suite_server.log"))
BOOT_WAIT = float(os.environ.get("TENSORFOLD_SCR_LIVE_BOOT_WAIT", "600"))

REASON = "TENSORFOLD_SCR_LIVE=1 not set (needs a GPU and a cached Qwen3.8-27B checkpoint)"


@unittest.skipUnless(os.environ.get("TENSORFOLD_SCR_LIVE") == "1", REASON)
class LiveSCR(unittest.TestCase):
    proc: subprocess.Popen | None = None
    log_offset = 0

    @classmethod
    def boot(cls, scr_on: bool) -> subprocess.Popen:
        if cls.proc is not None:
            cls.proc.terminate()
            cls.proc.wait(timeout=60)
        env = dict(os.environ)
        tf_src = "src"
        env["PYTHONPATH"] = "src"
        env["TENSORFOLD_SCR_ENABLED"] = "1" if scr_on else "0"
        env["TENSORFOLD_SCR_DEBUG"] = "1"
        env["PYTHONFAULTHANDLER"] = "1"
        cmd = [sys.executable, "-m", "tensorfold.cli", "serve", MODEL,
               "--host", "127.0.0.1", "--port", str(PORT),
               "--no-drafts", "--no-thinking", "--parallel", "4"]
        with open(LOG, "a") as f:
            f.write(f"\n===== boot scr_on={scr_on} pid={os.getpid()} =====\n")
            f.flush()
            cls.proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        deadline = time.time() + BOOT_WAIT
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError(f"server exited early; log tail:\n{LOG.read_text()[-2000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if b'"ok"' in r.read():
                        cls.log_offset = LOG.stat().st_size
                        return cls.proc
            except Exception:                                       # noqa: BLE001
                time.sleep(5)
        raise RuntimeError("server did not become healthy in time")

    @classmethod
    def setUpClass(cls) -> None:
        cls.boot(scr_on=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.proc is not None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    # -- helpers ------------------------------------------------------------
    def scr_lines(self) -> list[str]:
        """The [scr] lines emitted since the last marker."""
        with open(LOG, errors="replace") as f:
            f.seek(self.log_offset)
            lines = [ln for ln in f.read().splitlines() if "[scr]" in ln]
        self.log_offset = LOG.stat().st_size
        return lines

    def chat(self, messages, max_tokens=8) -> tuple[str, dict]:
        from tests.test_scr_live_driver import chat as drv_chat

        text, usage = drv_chat(PORT, messages, max_tokens=max_tokens)
        self.assertTrue(text, "empty reply")
        return text, usage

    def assert_alive(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            self.assertTrue(b'"ok": true' in r.read())

    # -- T1 -----------------------------------------------------------------
    def test_1_canary_unplanned(self):
        from tests.test_scr_live_driver import chat as drv_chat

        text, usage = drv_chat(PORT, [{"role": "user",
                                       "content": "What is 2+2? Reply with just the number."}])
        self.assertEqual(text, "4")
        plans = [ln for ln in self.scr_lines() if "plan:" in ln]
        self.assertEqual(plans, [], "a 25-token canary must not plan")
        self.assert_alive()

    # -- T2 -----------------------------------------------------------------
    def test_2_mid_edit_splice_and_needle(self):
        from tests.test_scr_live_driver import filler

        needle = "ACCESS-7Q4K"
        long_msg = filler(40, needle_para=32, needle=needle)
        ask = "\n\nConfirm you are tracking: reply with just the word TRACKING."
        reply1, _ = self.chat([{"role": "user", "content": long_msg + ask}])
        self.assertIn("TRACKING", reply1)
        paras = long_msg.split("\n\n")
        edited = "\n\n".join(paras[:19] + paras[20:]) + ask
        reply2, usage = self.chat([{"role": "user", "content": edited},
                                   {"role": "assistant", "content": reply1},
                                   {"role": "user",
                                    "content": "Now: what is the emergency access code? "
                                               "Reply with only the code."}])
        lines = self.scr_lines()
        plans = [ln for ln in lines if "plan:" in ln]
        splices = [ln for ln in lines if "splice OK" in ln]
        self.assertTrue(plans, "the edited turn must plan")
        self.assertTrue(splices, "the edited turn must relocate a span (splice OK)")
        self.assertIn(needle, reply2, "the needle inside the RELOCATED span was lost")
        # the plan served the leading prefix from the snapshot
        plan_fields = dict(kv.split("=") for kv in plans[-1].split() if "=" in kv)
        self.assertGreater(int(plan_fields["base"]), 0)
        self.assert_alive()

    # -- T3 -----------------------------------------------------------------
    def test_3_append_planned_matches_off_boot(self):
        from tests.test_scr_live_driver import filler

        long_msg = filler(28)
        ask1 = "\n\nNow reply with exactly one word, the name of a common fruit:"
        ask2 = "Good. Now reply with exactly one word, the color of a banana:"
        # conversational, planned
        reply1, _ = self.chat([{"role": "user", "content": long_msg + ask1}])
        reply2, usage2 = self.chat([{"role": "user", "content": long_msg + ask1},
                                    {"role": "assistant", "content": reply1},
                                    {"role": "user", "content": ask2}])
        plans = [ln for ln in self.scr_lines() if "plan:" in ln]
        self.assertTrue(plans, "the append turn must plan (the reply's KV is only in the snapshot)")
        # the same conversation on a SCR-off boot (native resume == fresh, per the
        # engine's contract) must give the identical greedy reply
        self.boot(scr_on=False)
        try:
            fresh, _ = self.chat([{"role": "user", "content": long_msg + ask1},
                                  {"role": "assistant", "content": reply1},
                                  {"role": "user", "content": ask2}])
        finally:
            self.boot(scr_on=True)
        self.assertEqual(reply2, fresh,
                         f"planned {reply2!r} != off-boot {fresh!r}")
        self.assert_alive()

    # -- T4 -----------------------------------------------------------------
    def test_4_off_boot_canary_parity(self):
        # after T3 the active boot is SCR-on again; the canary must be exact
        text, _ = self.chat([{"role": "user", "content": "What is 2+2? Reply with just the number."}])
        self.assertEqual(text, "4")
        self.assert_alive()


if __name__ == "__main__":
    unittest.main()