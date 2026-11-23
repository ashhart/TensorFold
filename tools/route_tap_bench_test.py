"""Trace row decoding and malformed/truncated-record regressions; no device or service."""
import struct
import tempfile
from pathlib import Path
import unittest
import route_tap_bench as tap


class Records(unittest.TestCase):
    def round_trip(self, rec):
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "trace.tfr1")
            tap.encode_file(p, [rec])
            return tap.decode_path(p)["records"][0]

    def test_rows_have_their_own_ids_and_weights(self):
        rec = self.round_trip(dict(step=9, layer=7, rows=2, slots=2,
                                  ids=[3, 4, 5, 6], weights=[.25, .75, .5, .5]))
        self.assertEqual(rec["row_records"], [{"ids": [3, 4], "weights": [.25, .75]},
                                              {"ids": [5, 6], "weights": [.5, .5]}])
        self.assertEqual(rec["rows"], 2)
        self.assertFalse(rec["truncated"])

    def test_truncation_keeps_whole_rows_and_original_dimensions(self):
        rec = self.round_trip(dict(step=0, layer=1, rows=3, slots=2, truncated=True,
                                  ids=[3, 4, 5, 6], weights=[.25, .75, .5, .5]))
        self.assertEqual((rec["rows"], rec["kept_rows"], len(rec["row_records"])), (3, 2, 2))
        self.assertTrue(rec["truncated"])
        for n, truncated in [(3, True), (4, False)]:
            with self.assertRaises(ValueError):
                self.round_trip(dict(step=0, layer=0, rows=3, slots=2, truncated=truncated,
                                     ids=[1] * n, weights=[1.] * n))

    def test_decoder_rejects_short_headers_and_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trace.tfr1"
            tap.encode_file(str(p), [dict(step=0, layer=0, rows=1, slots=2,
                                        ids=[1, 2], weights=[.5, .5])])
            raw = p.read_bytes()
            for suffix in [raw[:-1], raw[:17], raw[:16] + struct.pack("<IHHHBB", 0, 0, 1, 2, 2, 1)]:
                p.write_bytes(suffix)
                with self.assertRaises(ValueError):
                    tap.decode_path(str(p))


if __name__ == "__main__":
    unittest.main()
