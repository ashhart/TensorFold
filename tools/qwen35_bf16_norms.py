#!/usr/bin/env python3
"""Write a copy of a Qwen3.5 checkpoint whose GDN norm weights are bf16, for the native CUDA qwen3_5 family.

Stock `Qwen/Qwen3.5-9B` stores `linear_attn.norm.weight` as f32; the qwen3_5 CUDA family reads those weights as bf16 and
refuses an f32 one rather than round it silently at load. This writes a converted copy: each such tensor rounded to
bf16 (nearest, ties to even: torch's `.to(torch.bfloat16)`), every other tensor and file unchanged. The model then
reads its norms as a bf16 release would, e.g. Cloudflare's merged clef-flash, whose norms are the base's rounded so.

    python3 tools/qwen35_bf16_norms.py SRC_DIR OUT_DIR

Shards without an f32 GDN norm are hard-linked (copied where links fail); the index is rewritten with the new sizes.
Standard library only. Prints each converted tensor; exits non-zero if SRC_DIR has none.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

NORM = ".linear_attn.norm.weight"


def bf16_bits(w: int) -> int:
    """An f32's bits rounded to bf16, nearest with ties to even (NaN kept quiet)."""
    if (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF):
        return ((w >> 16) | 0x40) & 0xFFFF
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16) & 0xFFFF


def read(path: Path):
    """A safetensors file's header and the offset its data starts at."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def convert(src: Path, out: Path) -> list[str]:
    """`out`: `src`'s shard with every f32 GDN norm rounded to bf16; the names converted. Streams the other tensors."""
    header, base = read(src)
    names = [k for k, v in header.items() if k.endswith(NORM) and v["dtype"] == "F32"]
    if not names:
        return []
    entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
    entries.sort(key=lambda kv: kv[1]["data_offsets"][0])
    new = {k: v for k, v in header.items() if k == "__metadata__"}
    at = 0
    for k, v in entries:
        lo, hi = v["data_offsets"]
        size = (hi - lo) // 2 if k in names else hi - lo
        new[k] = {**v, "dtype": "BF16" if k in names else v["dtype"], "data_offsets": [at, at + size]}
        at += size
    head = json.dumps(new, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(src, "rb") as f, open(out, "wb") as o:
        o.write(struct.pack("<Q", len(head)) + head)
        for k, v in entries:
            lo, hi = v["data_offsets"]
            f.seek(base + lo)
            if k in names:
                words = struct.unpack(f"<{(hi - lo) // 4}I", f.read(hi - lo))
                o.write(struct.pack(f"<{len(words)}H", *(bf16_bits(w) for w in words)))
                continue
            left = hi - lo
            while left:
                chunk = f.read(min(left, 1 << 24))
                o.write(chunk)
                left -= len(chunk)
    return names


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0] + "\nusage: qwen35_bf16_norms.py SRC_DIR OUT_DIR", file=sys.stderr)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    shards = sorted(src.glob("*.safetensors"))
    if not any(k.endswith(NORM) and v.get("dtype") == "F32" for f in shards for k, v in read(f)[0].items()):
        print(f"{src} has no f32 {NORM[1:]}: nothing to convert", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=False)
    done: list[str] = []
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if f.suffix == ".safetensors":
            names = convert(f, out / f.name)
            if names:
                done += names
                continue
        link(f, out / f.name)
    index = out / "model.safetensors.index.json"
    if index.exists():
        doc = json.loads((src / index.name).read_text())
        doc.setdefault("metadata", {})["total_size"] = sum(
            p.stat().st_size for p in out.glob("*.safetensors"))
        index.unlink()
        index.write_text(json.dumps(doc, indent=2))
    for n in done:
        print("bf16:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
