#!/usr/bin/env python3
"""The two RoPE tables (SWA plain theta, compressed YaRN) built by the reference code on this host's CPU, for Zig."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=6144, help="positions: the slot limit + the prefill chunk (4096 + 2048)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tensorfold.families.deepseek_v41.cuda import rope
    from tensorfold.families.deepseek_v41.cuda.config import Config

    cfg = Config.from_file(Path(a.config))
    t = rope.tables(cfg, a.rows)
    plain, comp = t[0].cs.contiguous(), t[1].cs.contiguous()
    head = b"DSV41RP1" + struct.pack("<II", plain.shape[0], plain.shape[1])
    Path(a.out).write_bytes(head + plain.numpy().tobytes() + comp.numpy().tobytes())
    print(f"{a.out}: 2 x [{plain.shape[0]}, {plain.shape[1]}] fp32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
