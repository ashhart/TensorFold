#!/usr/bin/env python3
"""The capture's Triton manifest as the Zig engine's kernel set: aot.json (launch facts and keys) plus the cubins."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def const(v):
    """A constexpr as Zig compares it: ints and bools as integers, floats by their fp32 bits."""

    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, dict) and "fp32_bits" in v:
        return {"f32": int(v["fp32_bits"], 16)}
    raise SystemExit(f"constexpr {v!r} has no Zig form")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True, help="a capture's manifest (repeatable)")
    ap.add_argument("--cache", action="append", required=True, help="that capture's Triton cache, in the same order")
    ap.add_argument("--jit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jit = json.loads(Path(a.jit).read_text())
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    kernels, seen = [], set()
    for manifest, cache in zip(a.manifest, a.cache, strict=True):
        for k in json.loads(Path(manifest).read_text())["kernels"]:
            if k["hash"] not in seen:
                seen.add(k["hash"])
                kernels.append(entry(k, jit, Path(cache), out))
    kernels.sort(key=lambda x: (x["fn"], x["hash"]))
    meta = {"generator": "zig/tests/cuda/nemotron/aot_pack.py", "kernels": kernels}
    (out / "aot.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(kernels)} kernels -> {out}")
    return 0


def entry(k: dict, jit: dict, cache: Path, out: Path) -> dict:
    """One specialization's launch facts; its cubin copied beside them by hash."""

    fn = jit[k["function"]]
    nospec = set(fn["do_not_specialize"])
    attrs = {n for n, v in k["attrs"].items() if v}
    runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
    ptx = [x["name"] for x in k["abi"]]
    if ptx != runtime + ["global_scratch", "profile_scratch"]:
        raise SystemExit(f"{k['name']} {k['hash']}: PTX parameters {ptx} are not the runtime arguments {runtime}")
    cubin = cache / k["cubin"]
    data = cubin.read_bytes()
    if hashlib.sha256(data).hexdigest() != k["cubin_sha256"]:
        raise SystemExit(f"{cubin} changed since the manifest was written")
    (out / "cubins" / f"{k['hash']}.cubin").write_bytes(data)
    md = k["metadata"]
    return {
        "fn": k["name"], "hash": k["hash"], "name": md["name"], "num_warps": md["num_warps"],
        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
        "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
        "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
        "params": [{"name": n, "type": k["signature"][n], "div16": n in attrs, "nospec": n in nospec}
                   for n in runtime],
        "consts": {n: const(v) for n, v in k["constexprs"].items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
