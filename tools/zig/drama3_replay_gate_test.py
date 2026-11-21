"""drama3_replay_gate.sh: a stop at the chunk 4 or 16 PCM compare passes; any earlier or other failure fails."""
import subprocess
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "tools/zig/drama3_replay_gate.sh"
SIZES = {1: "{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }", 4: "{ 4, 4, 4, 4, 1 }", 16: "{ 16, 1 }"}


def log(*chunks: int, last: str = "error: ReplayMismatch") -> list[str]:
    """replay.zig's report: one codec line per decode reached, then the run's last line."""
    return [f"codec requested {c}, actual sizes {SIZES[c]}" for c in chunks] + [last]


def gate(lines: list[str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".log") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        return subprocess.run(["sh", str(GATE), f.name]).returncode


class ReplayGateTests(unittest.TestCase):
    def test_chunk_4_pcm_mismatch_passes(self):
        self.assertEqual(gate(log(1, 1, 4)), 0)

    def test_chunk_16_pcm_mismatch_passes(self):
        self.assertEqual(gate(log(1, 1, 4, 16)), 0)

    def test_frame_mismatch_fails(self):
        self.assertNotEqual(gate(log(1)), 0)

    def test_chunk_1_pcm_mismatch_fails(self):
        self.assertNotEqual(gate(log(1, 1)), 0)

    def test_fork_lane_pcm_mismatch_fails(self):
        self.assertNotEqual(gate(log(1, 1, 4, 16, 1)), 0)

    def test_other_error_at_chunk_4_fails(self):
        self.assertNotEqual(gate(log(1, 1, 4, last="error: BadCodecChunk")), 0)


if __name__ == "__main__":
    unittest.main()
