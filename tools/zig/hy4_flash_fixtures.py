#!/usr/bin/env python3
"""Rebuild zig/src/families/hy4_flash/fixtures/header.txt from a Hy4-flash checkpoint's safetensors headers.

    python3 tools/zig/hy4_flash_fixtures.py CHECKPOINT_DIR > zig/src/families/hy4_flash/fixtures/header.txt

One line a tensor: name, dtype, dims. Kept: layers 0, 1 and 2 (dense, sparse with Engram, sliding), experts 0 and
the last of each layer, drafter layer 0, every tensor outside the layers, and the vision tower's first block.
"""

import glob
import json
import re
import struct
import sys

KEEP_LAYERS = {"0", "1", "2"}


def headers(root):
    for path in sorted(glob.glob(f"{root}/*.safetensors")):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            for name, meta in json.loads(f.read(n)).items():
                if name != "__metadata__":
                    yield name, meta["dtype"], meta["shape"]


def keep(name, last_expert):
    m = re.match(r"model\.(layers|dfly_layers)\.(\d+)\.", name)
    if m and (m.group(2) not in KEEP_LAYERS or (m.group(1) == "dfly_layers" and m.group(2) != "0")):
        return False
    m = re.match(r"vit\.layers\.(\d+)\.", name)
    if m and m.group(1) != "0":
        return False
    m = re.search(r"\.experts\.(\d+)\.", name)
    return not m or m.group(1) in ("0", last_expert)


def main():
    rows = list(headers(sys.argv[1]))
    last = str(max(int(m.group(1)) for n, _, _ in rows if (m := re.search(r"\.experts\.(\d+)\.", n))))
    for name, dtype, shape in sorted(rows):
        if keep(name, last):
            print(name, dtype, "x".join(map(str, shape)) or "scalar")


if __name__ == "__main__":
    main()
