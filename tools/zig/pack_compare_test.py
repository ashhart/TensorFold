"""Pack qualification must reject incomplete tensors even when their common prefix matches."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pack_compare


class PackCompareTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.built, self.reference = self.root / "built", self.root / "reference"
        for directory in (self.built, self.reference):
            directory.mkdir()
            for name in pack_compare.PACKS:
                self.write(directory / name)

    def write(self, path, payload=b"abcdefgh", shape=None, offsets=None, dtype="U32"):
        header = {"w": {"dtype": dtype, "shape": [2] if shape is None else shape,
                         "data_offsets": [0, 8] if offsets is None else offsets}}
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def compare(self):
        with patch("sys.argv", ["pack_compare", str(self.built), str(self.reference)]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = pack_compare.main()
        self.report = out.getvalue()
        return code

    def test_identical_complete_packs_pass(self):
        self.assertEqual(self.compare(), 0)
        self.assertIn("packs match", self.report)

    def test_truncated_matching_prefix_fails_on_either_side(self):
        for directory in (self.built, self.reference):
            with self.subTest(side=directory.name):
                path = directory / pack_compare.PACKS[0]
                self.write(path, payload=b"abcd")
                self.assertEqual(self.compare(), 1)
                self.assertIn("INVALID", self.report)
                self.assertNotIn("packs match", self.report)
                self.write(path)

    def test_both_identically_truncated_files_still_fail(self):
        for directory in (self.built, self.reference):
            self.write(directory / pack_compare.PACKS[0], payload=b"abcd")
        self.assertEqual(self.compare(), 1)

    def test_header_byte_range_must_match_shape_and_dtype(self):
        for directory in (self.built, self.reference):
            self.write(directory / pack_compare.PACKS[0], payload=b"abcd", offsets=[0, 4])
        self.assertEqual(self.compare(), 1)

    def test_complete_byte_mismatch_still_fails(self):
        self.write(self.built / pack_compare.PACKS[0], payload=b"abcdEfgh")
        self.assertEqual(self.compare(), 1)
        self.assertIn("DIFF", self.report)

    def test_complete_empty_and_scalar_tensors_are_valid(self):
        for directory in (self.built, self.reference):
            self.write(directory / pack_compare.PACKS[0], payload=b"", shape=[0], offsets=[0, 0])
            self.write(directory / pack_compare.PACKS[1], payload=b"abcd", shape=[], offsets=[0, 4])
        self.assertEqual(self.compare(), 0)


if __name__ == "__main__":
    unittest.main()
