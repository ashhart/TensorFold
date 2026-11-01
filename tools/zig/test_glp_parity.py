"""The GLP parity script's comparison: the fixture both readers see, and the difference both must report."""
from __future__ import annotations

import struct
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glp_parity as parity

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def gguf(tensors: list[tuple[str, list[int], bytes]], metadata: dict) -> bytes:
    """A GGUF v3 container with F32 tensors and typed metadata, the shape the control vectors use."""
    out = bytearray(b"GGUF" + struct.pack("<I", 3))
    out += struct.pack("<QQ", len(tensors), len(metadata))
    for key, value in metadata.items():
        raw = key.encode()
        out += struct.pack("<Q", len(raw)) + raw
        if isinstance(value, str):
            raw = value.encode()
            out += struct.pack("<I", 8)  # the string value type
            out += struct.pack("<Q", len(raw)) + raw
        elif isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)  # the uint32 value type
        elif isinstance(value, float):
            out += struct.pack("<I", 6) + struct.pack("<f", value)  # the float32 value type
        else:
            raise TypeError(f"the fixture holds no {type(value).__name__} values")
    offset = 0
    for name, dims, payload in tensors:
        raw = name.encode()
        out += struct.pack("<Q", len(raw)) + raw
        out += struct.pack("<I", len(dims)) + struct.pack("<Q" * len(dims), *dims)
        out += struct.pack("<Q", offset)
        out += struct.pack("<I", 0)  # the F32 tensor type
        offset += len(payload)
    while len(out) % 32 != 0:
        out += b"\0"
    for _, _, payload in tensors:
        out += payload
    return bytes(out)


def fixture() -> bytes:
    """One direction, a unit vector, in a file that names the layer it belongs to."""
    direction = [0.0] * 8
    direction[3] = 1.0
    payload = struct.pack("<8f", *direction)
    return gguf(
        [("direction.7", [8], payload)],
        {
            "general.architecture": "controlvector",
            "glp.spec_version": 1,
            "glp.mode": "project",
            "glp.rank": 1,
            "glp.hook_point": "residual_stream_post_layer",
            "glp.alpha_default": 1.0,
            "glp.layer_ids_zero_based": "7",
        },
    )


class Compare(unittest.TestCase):
    """The comparison the script runs on the two readers' views of one file."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is required")
        self.meta = {
            "glp.mode": "project",
            "glp.hook_point": "residual_stream_post_layer",
            "glp.alpha_default": "1.0",
            "glp.layer_ids_zero_based": "7",
        }
        self.tensors = {"direction.7": numpy.zeros(8, dtype=numpy.float32)}
        self.tensors["direction.7"][3] = 1.0
        self.ours = {
            "mode": "project",
            "hook_point": "residual_stream_post_layer",
            "alpha": 1.0,
            "method": "",
            "contrast": "",
            "base_model": "",
            "base_revision": "",
            "sha256": "0" * 64,
            "layers": [7],
        }
        self.blob = self.tensors["direction.7"].astype("<f4").tobytes()

    def test_the_same_file_is_the_same_view(self):
        problems: list[str] = []
        parity.compare(self.meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual([], problems)

    def test_a_different_layer_is_reported(self):
        problems: list[str] = []
        ours = dict(self.ours, layers=[8])
        parity.compare(self.meta, self.tensors, ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("layer ids differ", problems[0])

    def test_a_different_number_is_reported(self):
        problems: list[str] = []
        blob = bytearray(self.blob)
        blob[0] = 0x3F  # a different first element
        parity.compare(self.meta, self.tensors, self.ours, bytes(blob), problems)
        self.assertEqual(1, len(problems))
        self.assertIn("direction.7 differs", problems[0])

    def test_a_different_mode_is_reported(self):
        problems: list[str] = []
        ours = dict(self.ours, mode="add")
        parity.compare(self.meta, self.tensors, ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("glp.mode", problems[0])

    def test_a_layer_list_that_does_not_match_the_tensors_is_reported(self):
        problems: list[str] = []
        meta = dict(self.meta, **{"glp.layer_ids_zero_based": "9"})
        parity.compare(meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("names layers", problems[0])


class Fixture(unittest.TestCase):
    """The container this test writes: our reader and the reference's reader must both take it."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is required")
        self.exe = Path(__file__).resolve().parents[2] / "zig-out" / "bin" / "tf-glp-check"
        if not self.exe.is_file():
            self.skipTest("tf-glp-check is not built: zig build tf-glp-check")
        self.temp = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip())
        self.addCleanup(self.temp.rmdir)
        self.addCleanup(lambda: [p.unlink() for p in self.temp.glob("*") if p.is_file()])
        self.path = self.temp / "vector.gguf"
        self.path.write_bytes(fixture())

    def test_our_reader_and_the_reference_read_the_same_fixture(self):
        receipt = self.temp / "ours.json"
        blob = self.temp / "ours.bin"
        subprocess.run([str(self.exe), "read", str(self.path), str(receipt), str(blob)], check=True)
        ours = parity.json.loads(receipt.read_text())
        self.assertEqual([7], ours["layers"])
        self.assertEqual("project", ours["mode"])
        self.assertEqual(8 * 4, blob.stat().st_size)
        self.assertEqual(self.tensors_bytes(), blob.read_bytes())

    def tensors_bytes(self) -> bytes:
        return struct.pack("<8f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
