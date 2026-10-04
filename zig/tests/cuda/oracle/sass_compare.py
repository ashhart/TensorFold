"""Compares cuobjdump -sass of two images kernel by kernel (keyed by name and template arguments, namespaces ignored)."""

from __future__ import annotations

import re
import sys

KEY = re.compile(r"(\d+(?:replay|tree)_kernelI.*?EE)")


def kernels(path: str) -> dict[str, list[str]]:
    """Function key -> SASS lines (instruction text and encodings) for every function in the dump."""

    out: dict[str, list[str]] = {}
    current = None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            k = KEY.search(m.group(1))
            current = k.group(1) if k else None
            if current is not None:
                out[current] = []
            continue
        if current is not None and re.match(r"\s*(/\*[0-9a-f]{4}\*/|/\* 0x)", line):
            out[current].append(re.sub(r"\s+", " ", line.strip()))
    return out


def main() -> int:
    python, zig = kernels(sys.argv[1]), kernels(sys.argv[2])
    both = sorted(set(python) & set(zig))
    bad = 0
    for key in both:
        same = python[key] == zig[key]
        bad += not same
        print(f"{'SASS-EQUAL' if same else 'SASS-DIFFER'} {key}: {len(zig[key])} lines")
    print(f"compared {len(both)} kernels; only in python: {sorted(set(python) - set(zig))}; "
          f"only in zig: {sorted(set(zig) - set(python))}")
    return 1 if bad or not both else 0


if __name__ == "__main__":
    raise SystemExit(main())
