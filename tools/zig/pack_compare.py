"""Compare two Flash Next pack directories tensor by tensor. Exit 0 when names, dtypes, shapes and bytes match."""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

PACKS = ("pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors")
WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
          "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def load_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, n + 8


def read_tensor(path: Path, base: int, entry: dict) -> np.ndarray:
    start, end = entry["data_offsets"]
    shape = entry["shape"]
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError("invalid tensor byte range")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid tensor shape")
    width = WIDTHS.get(entry["dtype"])
    if width is None:
        raise ValueError("unsupported tensor dtype")
    expected = math.prod(shape) * width
    if end - start != expected:
        raise ValueError(f"tensor byte range has {end - start} bytes, shape and dtype require {expected}")
    if base + end > path.stat().st_size:
        raise ValueError("tensor byte range extends beyond the file")
    with path.open("rb") as f:
        f.seek(base + start)
        raw = f.read(expected)
    if len(raw) != expected:
        raise ValueError(f"short tensor read: expected {expected} bytes, read {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8)


def as_float(raw: np.ndarray, dtype: str, count: int) -> np.ndarray:
    if dtype == "F32":
        return raw.view("<f4")
    if dtype == "BF16":
        return (raw.view("<u2").astype("<u4") << 16).view("<f4")
    if dtype == "F16":
        return raw.view("<f2").astype("<f4")
    return raw  # not a float dtype


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    built_dir, reference_dir = (Path(sys.argv[1]), Path(sys.argv[2]))
    clean = True
    for pack in PACKS:
        built_path, reference_path = built_dir / pack, reference_dir / pack
        if not built_path.exists() or not reference_path.exists():
            print(f"{pack}: MISSING ({'built' if not built_path.exists() else 'reference'} side)")
            clean = False
            continue
        built, built_base = load_header(built_path)
        reference, reference_base = load_header(reference_path)
        built.pop("__metadata__", None)
        reference.pop("__metadata__", None)
        built_names, reference_names = set(built), set(reference)
        for name in sorted(reference_names - built_names):
            print(f"{pack} {name}: missing from the build")
            clean = False
        for name in sorted(built_names - reference_names):
            print(f"{pack} {name}: missing from the reference")
            clean = False
        identical = differ = 0
        first = True
        for name in sorted(built_names & reference_names):
            mine, theirs = built[name], reference[name]
            if mine["dtype"] != theirs["dtype"] or mine["shape"] != theirs["shape"]:
                print(f"{pack} {name}: STRUCTURE built {mine['dtype']}{mine['shape']}"
                      f" vs reference {theirs['dtype']}{theirs['shape']}")
                clean = False
                differ += 1
                continue
            try:
                a = read_tensor(built_path, built_base, mine)
                b = read_tensor(reference_path, reference_base, theirs)
            except ValueError as err:
                print(f"{pack} {name}: INVALID {err}")
                clean = False
                differ += 1
                continue
            diff = np.flatnonzero(a != b)
            if diff.size == 0 and a.size == b.size:
                identical += 1
                continue
            differ += 1
            clean = False
            dtype, shape = mine["dtype"], mine["shape"]
            count = int(diff.size)
            extra = ""
            if dtype in ("F32", "BF16", "F16") and diff.size:
                width = 4 if dtype == "F32" else 2
                elems = np.unique(diff // width)
                fa, fb = as_float(a, dtype, a.size), as_float(b, dtype, b.size)
                gaps = np.abs(fa[elems].astype("<f8") - fb[elems].astype("<f8"))
                worst = int(elems[int(np.argmax(gaps))])
                scale = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                extra = (f", max |a-b| = {float(gaps.max()):.6g} at flat index {worst}"
                         f" (row {worst // scale if scale else 0})")
            tag = "FIRST DIFF" if first and differ == 1 else "DIFF"
            first = False
            print(f"{pack} {tag} {name} {dtype}{shape}: {count} of {a.size} bytes differ{extra}")
        print(f"{pack}: {identical} identical, {differ} differ")
    print("packs match" if clean else "packs differ")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
