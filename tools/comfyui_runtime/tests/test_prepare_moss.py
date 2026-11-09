import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_moss import prepare, split_header


def fixture():
    return {"layers.0.proj.qkv.weight": {"dtype": "BF16", "shape": [6, 2], "data_offsets": [0, 24]},
            "rope.bands": {"dtype": "F32", "shape": [1], "data_offsets": [24, 28]}}


class PreparationTests(unittest.TestCase):
    def test_payload_and_qkv_row_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.safetensors", Path(directory) / "prepared.safetensors"
            header = json.dumps(fixture()).encode()
            payload = bytes(range(28))
            source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            original = source.read_bytes()
            prepare(source, output)
            with output.open("rb") as stream:
                converted = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
                self.assertEqual(stream.read(), payload)
            for index, part in enumerate(("q", "k", "v")):
                tensor = converted[f"layers.0.proj.{part}.weight"]
                self.assertEqual(tensor["shape"], [2, 2])
                start, stop = tensor["data_offsets"]
                self.assertEqual(payload[start:stop], payload[index * 8:(index + 1) * 8])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare(source, output)

    def test_malformed_extents_and_mixed_weights_are_rejected(self):
        for change in ("dtype", "shape", "overlap", "mixed", "truncated"):
            with self.subTest(change=change):
                header = fixture()
                if change == "dtype":
                    header["layers.0.proj.qkv.weight"]["dtype"] = "F32"
                if change == "shape":
                    header["layers.0.proj.qkv.weight"]["shape"] = [2, 6]
                if change == "overlap":
                    header["rope.bands"]["data_offsets"] = [0, 4]
                if change == "mixed":
                    header["layers.0.proj.q.weight"] = header["rope.bands"]
                with self.assertRaises(ValueError):
                    split_header(header, 27 if change == "truncated" else 28)

    def test_copy_failure_removes_its_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            header = json.dumps(fixture()).encode()
            source.write_bytes(struct.pack("<Q", len(header)) + header + bytes(range(28)))
            original = source.read_bytes()
            with patch("prepare_moss.shutil.copyfileobj", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    prepare(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_invalid_header_never_creates_an_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "bad", Path(directory) / "output"
            for data in (b"", struct.pack("<Q", 99), struct.pack("<Q", 2) + b"[]"):
                source.write_bytes(data)
                with self.assertRaises(ValueError):
                    prepare(source, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
