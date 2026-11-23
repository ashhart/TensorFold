"""drama3_replay_gate.sh: only replay.zig's PCM-only outcome, after every exact check, passes; any other stop fails."""
import subprocess
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "tools/zig/drama3_replay_gate.sh"
SIZES = {1: "{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }", 4: "{ 4, 4, 4, 4, 1 }", 16: "{ 16, 1 }"}
EVERY_DECODE = (1, 1, 4, 16, 1)


def pcm_only(chunk: int, max_abs: str = "3.052e-5") -> str:
    return (f"Drama replay PCM only: codec chunk {chunk} against 1, max abs {max_abs}, RMS 4.317e-6; "
            "every exact check passed")


def log(*chunks: int, last: str = "error: ReplayMismatch", info: tuple[str, ...] = ()) -> list[str]:
    """replay.zig's report: one codec line per decode reached, any outcome lines, then the run's last line."""
    return [f"codec requested {c}, actual sizes {SIZES[c]}" for c in chunks] + list(info) + [last]


def gate(lines: list[str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".log") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        return subprocess.run(["sh", str(GATE), f.name]).returncode


class ReplayGateTests(unittest.TestCase):
    def test_chunk_4_pcm_only_passes(self):
        self.assertEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(4),))), 0)

    def test_chunk_16_pcm_only_passes(self):
        self.assertEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(16),))), 0)

    def test_chunk_4_and_16_pcm_only_passes(self):
        self.assertEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(4), pcm_only(16)))), 0)

    def test_fork_frame_mismatch_after_chunk_16_fails(self):
        """Igor's fixture on #4463: PCM differed at chunk 4/16, then a forked frame differed (replay.zig frame loop)."""
        self.assertNotEqual(gate(log(1, 1, 4, 16)), 0)

    def test_frame_mismatch_fails(self):
        self.assertNotEqual(gate(log(1)), 0)

    def test_chunk_1_pcm_mismatch_fails(self):
        self.assertNotEqual(gate(log(1, 1)), 0)

    def test_fork_lane_pcm_mismatch_fails(self):
        self.assertNotEqual(gate(log(*EVERY_DECODE)), 0)

    def test_pcm_only_error_without_its_line_fails(self):
        self.assertNotEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly")), 0)

    def test_pcm_only_line_before_another_error_fails(self):
        self.assertNotEqual(gate(log(*EVERY_DECODE, last="error: BadCodecChunk", info=(pcm_only(4),))), 0)

    def test_chunk_1_pcm_only_line_fails(self):
        self.assertNotEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(1),))), 0)

    def test_malformed_pcm_only_line_fails(self):
        for line in (pcm_only(4).replace(", RMS 4.317e-6", ""), pcm_only(4).replace("max abs 3.052e-5", "max abs x"),
                     pcm_only(4) + " extra"):
            with self.subTest(line=line):
                self.assertNotEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(line,))), 0)

    def test_pcm_past_reported_bound_fails(self):
        for max_abs in ("3.053e-5", "2.500e-1", "1.000e0", "1.200e3"):
            with self.subTest(max_abs=max_abs):
                self.assertNotEqual(gate(log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(16, max_abs),))), 0)

    def test_bad_chunk_4_before_a_bounded_chunk_16_fails(self):
        """Ash's case on #4463: the last PCM line is in bounds, the one before it is not."""
        for max_abs in ("nan", "inf", "2.500e-1"):
            with self.subTest(max_abs=max_abs):
                lines = log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(4, max_abs), pcm_only(16)))
                self.assertNotEqual(gate(lines), 0)

    def test_nonfinite_rms_fails(self):
        lines = log(*EVERY_DECODE, last="error: ReplayPcmOnly", info=(pcm_only(4).replace("RMS 4.317e-6", "RMS nan"), pcm_only(16)))
        self.assertNotEqual(gate(lines), 0)


if __name__ == "__main__":
    unittest.main()
