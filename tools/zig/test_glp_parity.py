"""The GLP parity comparator rejects mismatched values and layer identities."""
from __future__ import annotations

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


class Compare(unittest.TestCase):
    """The comparison the script runs on the two readers' views of one file."""

    def setUp(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy is required")
        self.meta = {
            "glp.mode": "add",
            "glp.hook_point": "residual_stream_post_layer",
            "glp.alpha_default": "1.0",
            "glp.layer_ids_zero_based": "7",
        }
        self.tensors = {"direction.7": numpy.zeros(8, dtype=numpy.float32)}
        self.tensors["direction.7"][3] = 1.0
        self.ours = {
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

    def test_a_layer_list_that_does_not_match_the_tensors_is_reported(self):
        problems: list[str] = []
        meta = dict(self.meta, **{"glp.layer_ids_zero_based": "9"})
        parity.compare(meta, self.tensors, self.ours, self.blob, problems)
        self.assertEqual(1, len(problems))
        self.assertIn("names layers", problems[0])


if __name__ == "__main__":
    unittest.main()
