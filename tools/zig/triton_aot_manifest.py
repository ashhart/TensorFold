#!/usr/bin/env python3
"""Record every Triton and extension launch a run makes, then map each launched Triton specialization to its cached cubin."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any

SITE_FRAMES = 8             # innermost frames of the caller's own code kept per launch site


def _jsonable(v: Any) -> Any:
    """Constants and metadata as JSON: floats keep their fp32 and fp64 bit patterns, dtypes and targets their names."""

    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return {"float": v, "fp32_bits": "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0],
                "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "value") and type(v).__name__ == "constexpr":
        return _jsonable(v.value)
    if hasattr(v, "__dict__") and type(v).__name__ == "GPUTarget":
        return {k: _jsonable(x) for k, x in vars(v).items()}
    return str(v)


def describe(v: Any) -> Any:
    """One launch argument: a tensor's dtype, shape, strides and 16-byte alignment, or a scalar's value."""

    if hasattr(v, "data_ptr") and hasattr(v, "shape"):
        return {"dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape), "stride": list(v.stride()),
                "align16": v.data_ptr() % 16 == 0, "offset": int(v.storage_offset())}
    if isinstance(v, (list, tuple)):
        return [describe(x) for x in v]
    return _jsonable(v)


def call_site(skip: tuple[str, ...] = ()) -> list[str]:
    """``file:line function`` of the innermost frames outside Triton, torch and this recorder."""

    out, f = [], sys._getframe(2)
    while f is not None and len(out) < SITE_FRAMES:
        name = f.f_code.co_filename
        if not any(s in name for s in ("/triton/", "/torch/", __file__, *skip)):
            parts = Path(name).parts
            short = "/".join(parts[parts.index("tensorfold"):]) if "tensorfold" in parts else "/".join(parts[-2:])
            out.append(f"{short}:{f.f_lineno} {f.f_code.co_name}")
        f = f.f_back
    return out


class Recorder:
    """Counts launches per phase; with ``detail`` it also logs each launch's arguments and site in order."""

    def __init__(self) -> None:
        self.kernels: dict[str, dict] = {}
        self.counts: Counter = Counter()
        self.grids: dict[str, set] = defaultdict(set)
        self.sites: dict[str, set] = defaultdict(set)
        self.log: list[dict] = []
        self.phase = "startup"
        self.detail = False
        self._seq = 0

    def install(self) -> "Recorder":
        from triton.runtime.jit import JITFunction

        original = JITFunction.run
        rec = self

        def run(fn, *args, grid, warmup, **kwargs):
            kernel = original(fn, *args, grid=grid, warmup=warmup, **kwargs)
            if kernel is not None and not warmup:
                rec._triton(fn, kernel, args, kwargs, grid)
            return kernel

        JITFunction.run = run
        return self

    def wrap(self, module: Any, names: tuple[str, ...], label: str) -> None:
        """Log calls of an extension module's functions (our .cu launches) beside the Triton launches."""

        for name in names:
            fn = getattr(module, name, None)
            if fn is None or getattr(fn, "_recorded", False):
                continue

            def wrapper(*args, _fn=fn, _name=name, **kwargs):
                self._count(f"{label}.{_name}", "ext", None)
                if self.detail:
                    self._entry({"kind": "ext", "name": f"{label}.{_name}", "args": describe(list(args)),
                                 "kwargs": {k: describe(v) for k, v in kwargs.items()}, "site": call_site()})
                return _fn(*args, **kwargs)

            wrapper._recorded = True
            setattr(module, name, wrapper)

    @contextmanager
    def scope(self, phase: str, detail: bool = False):
        """A named phase (also a ``tf::`` profiler range when torch is loaded)."""

        before = (self.phase, self.detail)
        self.phase, self.detail = phase, detail
        ctx = None
        if "torch" in sys.modules:
            import torch

            ctx = torch.profiler.record_function(f"tf::{phase}")
            ctx.__enter__()
        try:
            yield self
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
            self.phase, self.detail = before

    def _count(self, key: str, kind: str, grid) -> None:
        self.counts[(self.phase, key)] += 1

    def _entry(self, row: dict) -> None:
        row.update(seq=self._seq, phase=self.phase)
        self._seq += 1
        self.log.append(row)

    def _triton(self, fn, kernel, args, kwargs, grid) -> None:
        h = kernel.hash
        if h not in self.kernels:
            self.kernels[h] = self._info(fn, kernel)
        grid = tuple(grid) if not callable(grid) else ("callable",)
        self.counts[(self.phase, h)] += 1
        self.grids[h].add(grid)
        site = call_site()
        self.sites[h].add(" <- ".join(site[:2]))
        if self.detail:
            names = fn.arg_names
            bound = dict(zip(names, args))
            bound.update({k: v for k, v in kwargs.items() if k in names})
            options = {k: v for k, v in kwargs.items() if k not in names}
            self._entry({"kind": "triton", "name": kernel.name, "hash": h, "grid": list(grid),
                         "args": {k: describe(v) for k, v in bound.items()}, "options": _jsonable(options),
                         "site": site})

    @staticmethod
    def _info(fn, kernel) -> dict:
        names = fn.arg_names
        src = kernel.src

        def named(paths: dict) -> dict:
            return {".".join([names[p[0]], *map(str, p[1:])]): v for p, v in paths.items()}

        meta = kernel.metadata._asdict()
        return {"name": kernel.name, "function": f"{fn.fn.__module__}.{fn.fn.__qualname__}",
                "source": {"file": fn.fn.__code__.co_filename, "line": fn.fn.__code__.co_firstlineno},
                "params": list(names), "signature": dict(src.signature),
                "constexprs": {k: _jsonable(v) for k, v in named(src.constants).items()},
                "attrs": {k: _jsonable(v) for k, v in named(src.attrs).items()},
                "metadata": _jsonable(meta), "files": dict(kernel.metadata_group),
                "n_regs": getattr(kernel, "n_regs", None), "n_spills": getattr(kernel, "n_spills", None)}

    def dump(self, path: str | Path) -> None:
        phases = defaultdict(dict)
        for (phase, key), n in sorted(self.counts.items()):
            phases[phase][key] = n
        out = {"generator": "tools/zig/triton_aot_manifest.py", "written_unix": time.time(), "kernels": self.kernels,
               "phases": phases, "grids": {h: sorted(map(list, g)) for h, g in self.grids.items()},
               "sites": {h: sorted(s) for h, s in self.sites.items()}, "log": self.log}
        Path(path).write_text(json.dumps(out, indent=1, default=str) + "\n")


def ptx_entry(ptx: str) -> dict:
    """The PTX entry's parameter list and launch directives (the cubin's ABI as the driver sees it)."""

    m = re.search(r"\.entry\s+(\S+)\s*\((.*?)\)\s*(.*?)\{", ptx, re.S)
    if m is None:
        return {}
    params = [" ".join(p.split()) for p in m.group(2).split(",") if p.strip()]
    directives = [" ".join(d.split()) for d in m.group(3).splitlines() if d.strip()]
    return {"symbol": m.group(1), "params": params, "directives": directives}


def abi(info: dict, entry: dict) -> list[dict]:
    """Runtime parameters in launch order: the signature's non-constexpr arguments, then Triton's scratch pointers."""

    runtime = [(n, info["signature"].get(n)) for n in info["params"] if info["signature"].get(n) != "constexpr"]
    params = entry.get("params", [])
    out = []
    for i, ptx in enumerate(params):
        if i < len(runtime):
            name, ty = runtime[i]
            out.append({"name": name, "type": ty, "divisibility_16": [["tt.divisibility", 16]] == info["attrs"].get(name),
                        "ptx": ptx})
        else:
            extra = ("global_scratch", "profile_scratch")
            out.append({"name": extra[i - len(runtime)] if i - len(runtime) < 2 else f"extra_{i}", "type": "*i8",
                        "ptx": ptx})
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(cache: Path, launches: Path | None, out: Path, mount: str | None) -> dict:
    """``manifest.json``: each launched kernel's cubin (relative to the cache), metadata, ABI, phases and call sites."""

    rec = json.loads(launches.read_text()) if launches else {"kernels": {}, "phases": {}, "grids": {}, "sites": {}}
    per_phase = defaultdict(dict)
    for phase, counts in rec.get("phases", {}).items():
        for key, n in counts.items():
            per_phase[key][phase] = n
    entries, seen = [], set()
    for h, info in sorted(rec["kernels"].items(), key=lambda kv: (kv[1]["name"], kv[0])):
        files = {}
        for leaf, where in info["files"].items():
            p = Path(where)
            if mount and str(p).startswith(mount):
                p = cache / Path(str(p)[len(mount):].lstrip("/"))
            files[leaf] = p
        cubin = next(p for leaf, p in files.items() if leaf.endswith(".cubin"))
        ptx = next((p for leaf, p in files.items() if leaf.endswith(".ptx")), None)
        meta = json.loads(next(p for leaf, p in files.items() if leaf.endswith(".json")).read_text())
        entry = ptx_entry(ptx.read_text()) if ptx is not None else {}
        seen.add(cubin.parent.name)
        keep = ("name", "num_warps", "num_ctas", "num_stages", "maxnreg", "shared", "tmem_size", "global_scratch_size",
                "global_scratch_align", "profile_scratch_size", "profile_scratch_align", "cluster_dims", "launch_pdl",
                "launch_cooperative_grid", "ptx_version", "arch", "enable_fp_fusion", "default_dot_input_precision",
                "allowed_dot_input_precisions", "supported_fp8_dtypes", "deprecated_fp8_dot_operand_dtypes",
                "debug", "sanitize_overflow", "triton_version", "target", "hash")
        entries.append({
            "name": info["name"], "function": info["function"], "source": info["source"], "hash": h,
            "cubin": str(cubin.relative_to(cache)), "cubin_sha256": sha256(cubin), "cubin_bytes": cubin.stat().st_size,
            "ptx": str(ptx.relative_to(cache)) if ptx is not None else None,
            "metadata": {k: meta[k] for k in keep if k in meta},
            "block": [32 * meta.get("num_warps", 4) * meta.get("num_ctas", 1), 1, 1],
            "dynamic_shared_bytes": meta.get("shared"),
            "n_regs": info.get("n_regs"), "n_spills": info.get("n_spills"),
            "signature": info["signature"], "constexprs": info["constexprs"], "attrs": info["attrs"],
            "entry": entry, "abi": abi(info, entry),
            "launches": per_phase.get(h, {}), "grids": rec["grids"].get(h, []), "call_sites": rec["sites"].get(h, []),
        })
    extensions = {key: phases for key, phases in per_phase.items() if key not in rec["kernels"]}
    unlaunched = []
    for meta_path in sorted(cache.glob("*/*.json")):
        if meta_path.name.startswith("__grp__") or meta_path.parent.name in seen:
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            continue
        if "num_warps" in meta:
            unlaunched.append({"name": meta.get("name"), "dir": meta_path.parent.name})
    manifest = {"generator": "tools/zig/triton_aot_manifest.py", "cache": str(cache), "kernels": entries,
                "extension_calls": extensions, "compiled_not_launched": unlaunched,
                "env": rec.get("env", {})}
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="the run's TRITON_CACHE_DIR as seen here")
    ap.add_argument("--launches", help="the Recorder's dump (launches.json)")
    ap.add_argument("--mount", help="TRITON_CACHE_DIR as the run saw it, when it ran in a container")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = build(Path(a.cache), Path(a.launches) if a.launches else None, Path(a.out), a.mount)
    print(json.dumps({"kernels": len(m["kernels"]), "compiled_not_launched": len(m["compiled_not_launched"]),
                      "out": a.out}))


if __name__ == "__main__":
    main()
