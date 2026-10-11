#!/usr/bin/env python3
"""Merge one AOT set (aot.json + cubins/<hash>.cubin) into another: the kernels the destination lacks, by hash."""
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    dst, src = Path(sys.argv[1]), Path(sys.argv[2])
    a = json.loads((dst / "aot.json").read_text())
    b = json.loads((src / "aot.json").read_text())
    have = {k["hash"] for k in a["kernels"]}
    added = 0
    for k in b["kernels"]:
        if k["hash"] in have:
            continue
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", dst / "cubins" / f"{k['hash']}.cubin")
        a["kernels"].append(k)
        have.add(k["hash"])
        added += 1
    (dst / "aot.json").write_text(json.dumps(a, indent=1) + "\n")
    print(json.dumps({"added": added, "kernels": len(a["kernels"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
