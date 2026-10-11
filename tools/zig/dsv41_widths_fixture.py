#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
"""Per-rank expert listing as the Zig fixture: each block's routed and shared widths, then the rank's trellis bytes."""

import json
import sys


def hist(widths: dict) -> str:
    # bits (as "2.0") -> K2 = 2 x bits
    return ",".join(f"{int(round(2 * float(b)))}:{n}" for b, n in sorted(widths.items(), key=lambda kv: float(kv[0])))


def main() -> int:
    d = json.load(open(sys.argv[1]))
    print(f"# expert widths of a 2.8 bpw EXL3 pack, world {d['world']} rank {d['rank']}: layer w1 w2 w3 shared_k2(w1,w2,w3) rank_bytes")
    for b in d["blocks"]:
        sh = b["shared"]
        shared = ",".join(str(int(round(2 * sh[w]))) for w in ("w1", "w2", "w3")) if sh else "-"
        print(b["layer"], hist(b["w1"]["widths"]), hist(b["w2"]["widths"]), hist(b["w3"]["widths"]), shared, b["rank_bytes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
