#!/usr/bin/env python3
"""Tests for the synthetic image fixture, its token fixture and the receipts tied to the withdrawn photograph."""
import json
import os
import pathlib
import struct
import subprocess
import sys
import unittest
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import synthetic_image as fixture  # noqa: E402

RECEIPTS = fixture.ROOT / "docs/receipts/lfm3"
MANIFEST = json.loads(fixture.MANIFEST.read_text())
TIED = MANIFEST["receipts_tied_to_withdrawn_photo"]
SECTIONS = ("direct_image_identity", "derived_artifact_identity", "named_without_hash", "references_only", "prose")


def decode(data: bytes) -> tuple[int, int, bytes]:
    """An independent reader for the PNGs this fixture writes: chunks and checksums, zlib, filter 0 rows."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat, header = 8, b"", None
    while at < len(data):
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        (crc,) = struct.unpack(">I", data[at + 8 + length:at + 12 + length])
        assert crc == zlib.crc32(kind + body), kind
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        at += 12 + length
    width, height, depth, color, _, _, interlace = header
    assert (depth, color, interlace) == (8, 2, 0)
    raw, stride = zlib.decompress(idat), width * 3
    assert len(raw) == height * (stride + 1)
    assert all(raw[y * (stride + 1)] == 0 for y in range(height))
    return width, height, b"".join(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)] for y in range(height))


def tiny_tokenizer() -> fixture.Tokenizer:
    """A byte-level BPE whose merges, chained left to right, make exactly the words the prompt uses single tokens."""
    letters = fixture.byte_unicode()
    vocab = {letters[b]: b for b in range(256)}
    merges: list[tuple[str, str]] = []
    for word in ("user", "Describe", " this", " image", "assistant", "The", " contains", " a", " cat"):
        symbols = [letters[b] for b in word.encode()]
        for end in range(2, len(symbols) + 1):
            merge = ("".join(symbols[:end - 1]), symbols[end - 1])
            if merge not in merges:
                merges.append(merge)
                vocab["".join(symbols[:end])] = len(vocab)
    config = {"image_token_id": 9000, "vision_start_token_id": 9001, "vision_end_token_id": 9002}
    added = [{"content": "<|im_start|>", "id": 9003}, {"content": "<|im_end|>", "id": 9004}]
    return fixture.Tokenizer({"model": {"type": "BPE", "vocab": vocab, "merges": [list(m) for m in merges]}, "added_tokens": added}, config)


class Image(unittest.TestCase):
    def test_generator_is_deterministic_and_matches_the_pins(self):
        self.assertEqual(fixture.png(), fixture.png())
        self.assertEqual(fixture.check_png(MANIFEST), [])

    def test_png_decodes_to_the_generated_raster(self):
        width, height, raster = decode(fixture.PNG.read_bytes())
        self.assertEqual((width, height), (fixture.WIDTH, fixture.HEIGHT))
        self.assertEqual(raster, fixture.rgb())

    def test_stored_zlib_inflates_to_its_input_across_block_boundaries(self):
        for size in (0, 1, 65534, 65535, 65536, 131070, 131071):
            data = bytes(i * 7 & 255 for i in range(size))
            self.assertEqual(zlib.decompress(fixture.stored_zlib(data)), data, size)

    def test_image_has_its_shapes(self):
        self.assertEqual(fixture.pixel(105, 96), (235, 40, 36))
        self.assertEqual(fixture.pixel(230, 75), (244, 204, 36))
        self.assertEqual({fixture.pixel(0, 160), fixture.pixel(10, 160)}, {(250, 250, 250), (16, 16, 16)})
        self.assertNotEqual(fixture.pixel(0, 0), fixture.pixel(299, 159))


class Tokens(unittest.TestCase):
    def test_npy_header_is_aligned_and_round_trips(self):
        for count in (0, 1, 90, 1000):
            data = fixture.npy_u4(list(range(count)))
            (length,) = struct.unpack("<H", data[8:10])
            self.assertEqual((10 + length) % 64, 0)
            self.assertEqual(struct.unpack(f"<{count}I", data[10 + length:]), tuple(range(count)))

    def test_npy_matches_numpy_save(self):
        np = _numpy()
        import io
        buffer = io.BytesIO()
        np.save(buffer, np.array(list(range(90)), dtype="<u4"))
        self.assertEqual(buffer.getvalue(), fixture.npy_u4(list(range(90))))

    def test_prompt_layout(self):
        t = tiny_tokenizer()
        word = lambda text: t.vocab["".join(t.letters[b] for b in text.encode())]  # noqa: E731
        newline = t.piece("\n")
        expected = ([9003, word("user"), newline, 9001] + [9000] * 70
                    + [9002, word("Describe"), word(" this"), word(" image"), word("."), 9004, newline, 9003,
                       word("assistant"), newline, word("The"), word(" image"), word(" contains"), word(" a"), word(" cat"),
                       word(".")])
        self.assertEqual(t.prompt("The image contains a cat."), expected)
        self.assertEqual(expected[4:74], [9000] * 70)
        self.assertEqual(expected[74], 9002)

    def test_pieces_are_single_tokens_or_an_error(self):
        t = tiny_tokenizer()
        self.assertEqual(t.piece(" cat"), t.vocab["".join(t.letters[b] for b in b" cat")])
        with self.assertRaises(ValueError):
            t.piece(" dog")
        with self.assertRaises(ValueError):
            t.words("A cat, 2 dogs")

    def test_checkpoint_tokenizer_reproduces_the_photograph_fixture_and_pins(self):
        model = os.environ.get("LFM3_VL_MODEL_DIR")
        if not model:
            self.skipTest("set LFM3_VL_MODEL_DIR to an official LFM3-VL-2B MLX checkpoint")
        self.assertEqual(fixture.check_tokens(MANIFEST, pathlib.Path(model)), [])


class Receipts(unittest.TestCase):
    def path(self, name: str) -> pathlib.Path:
        return fixture.ROOT / name if "/" in name else RECEIPTS / name

    def test_photograph_is_gone_and_replacement_differs(self):
        self.assertFalse(self.path(MANIFEST["withdrawn_photo"]["path"]).exists())
        self.assertNotEqual(MANIFEST["image"]["png_sha256"], MANIFEST["withdrawn_photo"]["png_sha256"])

    def test_every_listed_tie_occurs_in_its_file(self):
        names = TIED["identifiers"]
        for section in SECTIONS:
            for entry in TIED[section]:
                text = self.path(entry["file"]).read_text()
                for tie in entry["ties"]:
                    self.assertIn(names.get(tie, tie), text, f"{entry['file']}: {tie}")

    def test_no_unlisted_receipt_carries_the_photograph_identity(self):
        listed = {self.path(e["file"]).name for s in SECTIONS for e in TIED[s]} | {fixture.MANIFEST.name}
        for path in sorted(RECEIPTS.glob("*.json")):
            text = path.read_text()
            hits = [k for k, v in TIED["identifiers"].items() if v in text]
            self.assertTrue(not hits or path.name in listed, f"{path.name} carries {hits} but is not listed")

    def test_tree_has_no_reference_to_the_removed_png(self):
        files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=fixture.ROOT, capture_output=True, text=True,
                               check=True).stdout.split("\n")
        needle = "real-image-fp64-input"
        found = [f for f in files if f and (fixture.ROOT / f).is_file() and f != "tools/lfm3/test_synthetic_image.py"
                 and needle in (fixture.ROOT / f).read_text(errors="ignore")]
        self.assertEqual(found, ["docs/receipts/lfm3/synthetic-image-fixture.json"])


def _numpy():
    try:
        import numpy
    except ImportError:
        raise unittest.SkipTest("NumPy is not installed")
    return numpy


if __name__ == "__main__":
    unittest.main()
