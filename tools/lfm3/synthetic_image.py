#!/usr/bin/env python3
"""Generate the synthetic LFM3-VL image fixture and its token fixture, and check both against their pinned identities.

Every pixel comes from integer formulas in this file and the PNG is written with stored (uncompressed) deflate blocks, so
the bytes do not depend on a zlib build. Nothing here is third-party content; see docs/receipts/lfm3/synthetic-image-fixture.json.
"""
import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/receipts/lfm3/synthetic-image-fixture.json"
PNG = ROOT / "docs/receipts/lfm3/synthetic-image-input.png"
PHOTO_RECEIPT = ROOT / "docs/receipts/lfm3/real-image-fp64.json"
WIDTH, HEIGHT = 300, 200
IMAGE_ROWS = 70  # a 14 x 20 patch grid after the processor's 65,536-pixel floor, merged 2 x 2
ANSWER = "The image contains a circle."
PHOTO_ANSWER = "The image contains a cat."


def pixel(x: int, y: int) -> tuple[int, int, int]:
    """A gradient sky, a shaded red disc, a yellow bordered rectangle and a black and white checker strip."""
    if y >= 160:
        return (250, 250, 250) if (x // 10 + (y - 160) // 10) % 2 == 0 else (16, 16, 16)
    dx, dy = x - 105, y - 96
    squared = dx * dx + dy * dy
    if squared <= 56 * 56:
        shade = squared * 70 // (56 * 56)
        return 235 - shade, 40 + shade // 2, 36 + shade // 2
    if 190 <= x < 272 and 38 <= y < 112:
        edge = x < 193 or x >= 269 or y < 41 or y >= 109
        return (120, 90, 10) if edge else (244, 204, 36)
    return 24 + x * 40 // (WIDTH - 1), 60 + y * 120 // (HEIGHT - 1), 190 - y * 90 // (HEIGHT - 1)


def rgb() -> bytes:
    """The raster, row by row, three bytes a pixel."""
    return bytes(c for y in range(HEIGHT) for x in range(WIDTH) for c in pixel(x, y))


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def stored_zlib(data: bytes) -> bytes:
    """A zlib stream of stored blocks: byte-identical on every platform."""
    out = bytearray(b"\x78\x01")
    for at in range(0, max(len(data), 1), 65535):
        block = data[at:at + 65535]
        out += struct.pack("<BHH", int(at + 65535 >= len(data)), len(block), len(block) ^ 0xFFFF) + block
    return bytes(out) + struct.pack(">I", zlib.adler32(data))


def png() -> bytes:
    """8-bit RGB, no interlace, filter 0 on every row."""
    raster = rgb()
    stride = WIDTH * 3
    rows = b"".join(b"\x00" + raster[y * stride:(y + 1) * stride] for y in range(HEIGHT))
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", stored_zlib(rows)) + chunk(b"IEND", b"")


def npy_u4(values: list[int]) -> bytes:
    """What numpy.save writes for a little-endian uint32 vector."""
    header = "{'descr': '<u4', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + struct.pack(f"<{len(values)}I", *values)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def byte_unicode() -> dict[int, str]:
    """GPT-2's reversible byte to printable character map, which the LFM3 tokenizer's vocabulary is written in."""
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = iter(range(256, 256 + 256 - len(kept)))
    return {b: chr(b if b in kept else next(extra)) for b in range(256)}


class Tokenizer:
    """The checkpoint's byte-level BPE, applied to one pre-token at a time, with its added tokens by name."""

    def __init__(self, tokenizer_json: dict, config: dict):
        model = tokenizer_json["model"]
        if model["type"] != "BPE":
            raise ValueError("expected a BPE tokenizer")
        self.vocab = model["vocab"]
        self.rank = {tuple(pair.split(" ")) if isinstance(pair, str) else tuple(pair): i for i, pair in enumerate(model["merges"])}
        self.added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
        self.config = config
        self.letters = byte_unicode()

    def piece(self, text: str) -> int:
        """The one token a pre-token is; BPE merging it to anything else is an error, not a guess."""
        symbols = [self.letters[b] for b in text.encode()]
        while len(symbols) > 1:
            rank, at = min((self.rank.get((symbols[i], symbols[i + 1]), 1 << 60), i) for i in range(len(symbols) - 1))
            if rank == 1 << 60:
                break
            symbols[at:at + 2] = [symbols[at] + symbols[at + 1]]
        if len(symbols) != 1 or symbols[0] not in self.vocab:
            raise ValueError(f"{text!r} is not a single token")
        return self.vocab[symbols[0]]

    def words(self, text: str) -> list[int]:
        """ASCII words, each with its leading space, and full stops: the pre-tokens the processor's regular expression gives."""
        pieces = re.findall(r" ?[A-Za-z]+|\.", text)
        if "".join(pieces) != text:
            raise ValueError("only ASCII words, single spaces and full stops are supported")
        return [self.piece(p) for p in pieces]

    def prompt(self, answer: str) -> list[int]:
        """The teacher-forced image prompt: one image of 70 rows at position 4, a question and the answer."""
        c = self.config
        return ([self.added["<|im_start|>"], self.piece("user"), self.piece("\n"), c["vision_start_token_id"]]
                + [c["image_token_id"]] * IMAGE_ROWS
                + [c["vision_end_token_id"], *self.words("Describe this image."), self.added["<|im_end|>"], self.piece("\n"),
                   self.added["<|im_start|>"], self.piece("assistant"), self.piece("\n"), *self.words(answer)])


def load_tokenizer(model_dir: pathlib.Path) -> Tokenizer:
    return Tokenizer(json.loads((model_dir / "tokenizer.json").read_text()), json.loads((model_dir / "config.json").read_text()))


def identities() -> dict:
    """What a fresh generation produces; model-independent."""
    data = png()
    return {"png_sha256": sha256(data), "png_bytes": len(data), "pixels_sha256": sha256(rgb())}


def check_png(manifest: dict) -> list[str]:
    """Differences between the generator, the committed file and the pins."""
    fresh, pinned, found = identities(), manifest["image"], []
    for key in ("png_sha256", "png_bytes", "pixels_sha256"):
        if fresh[key] != pinned[key]:
            found.append(f"generator {key} {fresh[key]} differs from the pinned {pinned[key]}")
    if not PNG.is_file():
        found.append(f"{PNG} is missing")
    elif PNG.read_bytes() != png():
        found.append(f"{PNG} is not what the generator writes")
    return found


def check_tokens(manifest: dict, model_dir: pathlib.Path) -> list[str]:
    """The fixture's pins against this checkpoint's tokenizer, and the tokenizer against the retained photograph fixture."""
    found, pinned = [], manifest["token_fixture"]
    tokenizer = load_tokenizer(model_dir)
    tokenizer_sha = sha256((model_dir / "tokenizer.json").read_bytes())
    if tokenizer_sha != pinned["tokenizer_json_sha256"]:
        found.append(f"tokenizer.json {tokenizer_sha} differs from the pinned {pinned['tokenizer_json_sha256']}")
    tokens = tokenizer.prompt(ANSWER)
    if sha256(npy_u4(tokens)) != pinned["npy_sha256"] or len(tokens) != pinned["count"]:
        found.append("the synthetic token fixture differs from its pin")
    historical = json.loads(PHOTO_RECEIPT.read_text())["token_fixture_sha256"]
    if sha256(npy_u4(tokenizer.prompt(PHOTO_ANSWER))) != historical:
        found.append("this tokenizer path does not reproduce the retained photograph token fixture")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    image = sub.add_parser("image", help="check the committed PNG and the generator against the pins, or write it")
    image.add_argument("--write", type=pathlib.Path, help="write the PNG here (the committed fixture is PNG's default path)")
    tokens = sub.add_parser("tokens", help="build the 90-token fixture from a checkpoint's tokenizer and check it")
    tokens.add_argument("--model-dir", type=pathlib.Path, required=True)
    tokens.add_argument("--output", type=pathlib.Path, help="write the fixture as a little-endian uint32 .npy")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "image":
        if args.write:
            args.write.write_bytes(png())
        found = check_png(manifest) if not args.write or args.write != PNG else []
        print("identities:", json.dumps(identities()))
    else:
        found = check_tokens(manifest, args.model_dir)
        if args.output:
            args.output.write_bytes(npy_u4(load_tokenizer(args.model_dir).prompt(ANSWER)))
    for line in found:
        print("MISMATCH", line)
    print("synthetic fixture matches its pins" if not found else f"{len(found)} mismatches")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
