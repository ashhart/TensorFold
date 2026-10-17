#!/usr/bin/env python3
"""Triton captures as the Zig engine's kernel set: aot.json (launch facts and keys) plus cubins/<hash>.cubin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DIV16 = ["tt.divisibility", 16]


def const(v):
    """A constexpr as Zig compares it: ints and bools as integers, floats by their fp32 bits."""

    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, dict) and "fp32_bits" in v:
        return {"f32": int(v["fp32_bits"], 16)}
    raise SystemExit(f"constexpr {v!r} has no Zig form")


def kernels_of(doc: dict) -> list[tuple[str, dict]]:
    """(hash, kernel) pairs from a manifest's list or a Recorder dump's mapping keyed by hash."""

    ks = doc["kernels"]
    return list(ks.items()) if isinstance(ks, dict) else [(k["hash"], k) for k in ks]


def located(path: str, cache: Path | None) -> Path:
    """A cache file where recorded, or under --cache by its last two parts (the cache seen from another mount)."""

    p = Path(path)
    for q in (p if p.is_absolute() else None, cache and cache / p, cache and cache / p.parent.name / p.name):
        if q is not None and q.exists():
            return q
    raise SystemExit(f"{path}: not found; give the capture's Triton cache as this host sees it with --cache")


def entry(h: str, k: dict, nospec: set[str], cache: Path | None, out: Path) -> dict:
    """One specialization's launch facts without Triton's scratch pointers; its cubin copied beside them by hash."""

    sig = k["signature"]
    runtime = [n for n in k.get("params") or list(sig) if sig.get(n, "constexpr") != "constexpr"]
    if "abi" in k and [x["name"] for x in k["abi"]] != runtime + ["global_scratch", "profile_scratch"]:
        raise SystemExit(f"{k['name']} {h}: PTX parameters are not the runtime arguments {runtime} then scratch")
    cubin = located(k["cubin"] if "cubin" in k else next(p for n, p in k["files"].items() if n.endswith(".cubin")),
                    cache)
    data = cubin.read_bytes()
    if "cubin_sha256" in k and hashlib.sha256(data).hexdigest() != k["cubin_sha256"]:
        raise SystemExit(f"{cubin} changed since the manifest was written")
    (out / "cubins" / f"{h}.cubin").write_bytes(data)
    md, div16 = k["metadata"], {n for n, v in k["attrs"].items() if DIV16 in v}
    return {
        "fn": k["name"], "hash": h, "name": k["name"], "num_warps": md["num_warps"],
        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
        "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
        "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
        "params": [{"name": n, "type": sig[n], "div16": n in div16, "nospec": n in nospec} for n in runtime],
        # a None argument Triton folded into the kernel (an unused pointer) leaves nothing to match
        "consts": {n: const(v) for n, v in k["constexprs"].items() if n not in runtime and v is not None},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True,
                    help="a manifest.json or a Recorder's launches.json (repeatable)")
    ap.add_argument("--cache", action="append", help="each capture's Triton cache as seen here, in --manifest order")
    ap.add_argument("--jit", action="append",
                    help="a capture's jit.json (repeatable, merged); without one no parameter is do_not_specialize")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = [json.loads(Path(m).read_text()) for m in a.manifest]
    caches = a.cache or [d.get("cache") for d in docs]
    if len(caches) != len(docs):
        raise SystemExit("--cache is given once per --manifest, or not at all")
    jit = {f: v for j in a.jit for f, v in json.loads(Path(j).read_text()).items()} if a.jit else None
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    kernels, seen = [], set()
    for doc, cache in zip(docs, caches):
        for h, k in kernels_of(doc):
            if h not in seen:
                seen.add(h)
                nospec = set(jit[k["function"]]["do_not_specialize"]) if jit else set()
                kernels.append(entry(h, k, nospec, Path(cache) if cache else None, out))
    kernels.sort(key=lambda x: (x["fn"], x["hash"]))
    meta = {"generator": "tools/zig/triton_aot_pack.py", "kernels": kernels}
    (out / "aot.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(kernels)} kernels from {len(docs)} captures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
