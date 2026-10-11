"""Compares cuobjdump -sass of the reference extension objects with our fatbins: every kernel of ours, same SASS."""

from __future__ import annotations

import re
import sys

def key(name: str) -> str:
    """The name with its first (length-prefixed) namespace component replaced by @."""
    m = re.match(r"_ZN(\d+)", name)
    if not m:
        return name
    n = int(m.group(1))
    return "_ZN@" + name[m.end() + n:]


def kernels(path: str, sm: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    arch_ok = True
    for line in open(path):
        a = re.match(r"\s*arch = sm_(\d+)", line)
        if a:
            arch_ok = a.group(1) == sm
            continue
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            current = key(m.group(1)) if arch_ok else None
            if current is not None:
                out[current] = []
            continue
        if current is not None and re.match(r"\s*(/\*[0-9a-f]{4}\*/|/\* 0x)", line):
            out[current].append(re.sub(r"\s+", " ", line.strip()))
    return out


def main() -> int:
    sm = "121"
    args = sys.argv[1:]
    if "--sm" in args:
        i = args.index("--sm")
        sm = args[i + 1]
        del args[i:i + 2]
    python, zig = kernels(args[0], sm), kernels(args[1], sm)
    bad = missing = 0
    for k in sorted(zig):
        if k not in python:
            missing += 1
            print(f"ONLY-ZIG {k}")
            continue
        if python[k] != zig[k]:
            bad += 1
            print(f"SASS-DIFFER {k}: {len(zig[k])} / {len(python[k])} lines")
    extra = sorted(set(python) - set(zig))
    print(f"sm_{sm}: {len(zig)} kernels of ours, {len(zig) - bad - missing} SASS-equal, {bad} differ, {missing} not in the "
          f"Python build; {len(extra)} only in the Python build" + (f" ({', '.join(extra[:6])}...)" if extra else ""))
    return 1 if bad or missing or not zig else 0


if __name__ == "__main__":
    raise SystemExit(main())
