"""Compares cuobjdump -sass of the Python extension's object and our fatbin, kernel by kernel, namespaces ignored."""

from __future__ import annotations

import re
import sys

# experts.cu's integer planner: the named namespace places place_items' static shared array at another offset
LAYOUT_ONLY = {"11plan_kernelEPKiiiiPiS2_S2_"}


def key(symbol: str) -> str | None:
    """The mangled name past its namespace: _ZN<n><namespace n chars> is dropped, the rest kept."""

    m = re.match(r"_ZN(\d+)", symbol)
    if m is None:
        return None
    return symbol[m.end() + int(m.group(1)):]


def kernels(path: str) -> dict[str, list[str]]:
    """Kernel key (its mangled name from the length-prefixed base name on) -> SASS lines with their encodings."""

    out: dict[str, list[str]] = {}
    current = None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            current = key(m.group(1))
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
        allowed = not same and key in LAYOUT_ONLY and len(python[key]) == len(zig[key])
        bad += not same and not allowed
        label = "SASS-EQUAL" if same else "SASS-LAYOUT" if allowed else "SASS-DIFFER"
        print(f"{label} {key}: {len(zig[key])} lines")
    missing = sorted(set(zig) - set(python))
    print(f"compared {len(both)} kernels; ours missing from python: {missing}")
    return 1 if bad or missing or not both else 0


if __name__ == "__main__":
    raise SystemExit(main())
