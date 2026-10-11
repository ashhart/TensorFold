#!/usr/bin/env python3
"""Triton variants compiled ahead of time from the served Triton source for every specialization a Zig config needs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from triton_fill_compile import (HERE, KERNEL_DIRS, OWN_DIR, OWN_PKG, OPTION_KEYS, jsonable, unjson, load_tree,
    full_name, sigs, resolve, _DT, torch_dtype, stand_in, Target, specialize, compile_one, entry_of, need_of_manifest,
    options_of, RULES, SOURCE, rule_options, derive_options)
from triton_fill_stamp import (elf_sections, uleb, line_files, stamps_of_cubins, stamped_root, in_bwrap,
    reexec_stamped, sass)

SERVED_PY = "/dsv41-tf/src"
SERVED_TRITON = "/usr/local/lib/python3.12/dist-packages/triton"


# ---- commands ---------------------------------------------------------------------------------------------------

def fresh_cache() -> None:
    """A fresh Triton cache for the run: its key does not cover the line-info map or the stamps."""

    if not os.environ.get("TRITON_FILL_STAMPED"):
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-fill-cache-")
    import atexit

    atexit.register(shutil.rmtree, os.environ["TRITON_CACHE_DIR"], True)


def cmd_verify(a) -> int:
    reexec_stamped(a)
    fresh_cache()
    kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
    man = json.loads(Path(a.manifest).read_text())
    cub = Path(a.cubins) if a.cubins else None
    tgts: dict = {}
    res = {"kernels": 0, "spec_equal": 0, "sass_equal": 0, "bytes_equal": 0, "sha_equal": 0, "no_reference": 0,
           "failed": []}
    for k in man["kernels"]:
        if a.only and k["name"] not in a.only:
            continue
        res["kernels"] += 1
        arch = k["metadata"]["target"]["arch"]
        tgt = tgts.setdefault(arch, Target(arch))
        try:
            fn = resolve(kernels, k["name"], k["function"], [])
            consts, runtime = need_of_manifest(k, fn)
            got, signature, constexprs, attrs = compile_one(fn, tgt, consts, runtime, options_of(k))
        except Exception as e:
            res["failed"].append({"hash": k["hash"], "name": k["name"], "error": f"{type(e).__name__}: {e}"})
            continue
        e = entry_of(fn, got, signature, constexprs, attrs)
        want_attrs = {n: v for n, v in k.get("attrs", {}).items() if v}
        have_attrs = {fn.arg_names[p[0]]: [list(x) for x in v] for p, v in attrs.items() if v}
        same_spec = e["signature"] == k["signature"] and have_attrs == want_attrs and \
            json.dumps(e["constexprs"], sort_keys=True) == json.dumps(k["constexprs"], sort_keys=True)
        res["spec_equal"] += same_spec
        mine = got.asm["cubin"]
        res["sha_equal"] += hashlib.sha256(mine).hexdigest() == k.get("cubin_sha256")
        ref = (cub / f"{k['hash']}.cubin") if cub else None
        if ref is None or not ref.exists():
            res["no_reference"] += 1
            if not same_spec:
                res["failed"].append({"hash": k["hash"], "name": k["name"], "error": "specialization differs"})
            continue
        theirs = ref.read_bytes()
        same_sass = sass(mine) == sass(theirs)
        res["sass_equal"] += same_sass
        res["bytes_equal"] += mine == theirs
        if not (same_spec and same_sass):
            res["failed"].append({"hash": k["hash"], "name": k["name"], "spec": same_spec, "sass": same_sass})
    print(json.dumps(res, indent=1))
    return 0 if not res["failed"] else 1


def have_specs(dirs: list[str]) -> set:
    """Variant keys of existing aot sets (aot.json): what the engine would already find."""

    out = set()
    for d in dirs:
        p = Path(d) / "aot.json"
        if p.exists():
            for k in json.loads(p.read_text())["kernels"]:
                out.add(spec_key(k))
    return out


def spec_key(k: dict) -> str:
    """aot.zig's identity of a variant: fn, constexprs (ints, fp32 bits), runtime params (type, div16)."""

    params = [(p["name"], p["type"], bool(p["div16"])) for p in k["params"]]
    return json.dumps([k["fn"], sorted((n, v.get("int"), v.get("f32")) for n, v in k["consts"].items()), params])


def cmd_compile(a) -> int:
    reexec_stamped(a)
    fresh_cache()
    sys.path.insert(0, str(HERE))
    import dsv41_triton_capture as cap

    kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
    rules = json.loads(Path(a.options).read_text())["functions"]
    needs = json.loads(Path(a.needs).read_text())["kernels"]
    have = have_specs(a.have or [])
    jit = sigs_jit(kernels)
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    aot_path = out / "aot.json"
    packed = json.loads(aot_path.read_text())["kernels"] if aot_path.exists() else []
    have |= {spec_key(k) for k in packed}
    tgt = Target(a.arch)
    res = {"needs": len(needs), "had": 0, "compiled": 0, "failed": []}
    for n in needs:
        try:
            fn = resolve(kernels, n["fn"], n.get("function"), list(n["consts"]) + list(n["runtime"]))
            consts = {k: unjson(v) for k, v in n["consts"].items()}
            opts = rule_options(rules, full_name(fn), consts)
            sp = specialize(fn, tgt, consts, n["runtime"], opts)
        except Exception as e:
            res["failed"].append({"fn": n["fn"], "consts": n["consts"], "error": f"{type(e).__name__}: {e}"})
            continue
        # the variant's identity before compiling it: an existing set's variant is not compiled again
        key = spec_key(packed_of(fn, cap, jit, *sp[1:], None))
        if key in have:
            res["had"] += 1
            continue
        try:
            import triton
            from triton.compiler import ASTSource

            k = triton.compile(ASTSource(fn, sp[1], sp[2], sp[3]), target=tgt.target, options=sp[0].__dict__)
        except Exception as e:
            res["failed"].append({"fn": n["fn"], "consts": n["consts"], "error": f"{type(e).__name__}: {e}"})
            continue
        packed_k = packed_of(fn, cap, jit, sp[1], sp[2], sp[3], k)
        if packed_k["global_scratch"] or packed_k["profile_scratch"]:
            res["failed"].append({"fn": n["fn"], "error": "global / profile scratch (aot.zig refuses it)"})
            continue
        (out / "cubins" / f"{k.hash}.cubin").write_bytes(k.asm["cubin"])
        packed.append(packed_k)
        have.add(key)
        res["compiled"] += 1
    aot_path.write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "kernels": packed}, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if not res["failed"] else 1


def packed_of(fn, cap, jit: dict, signature: dict, constexprs: dict, attrs: dict, k) -> dict:
    """The variant in aot.json's form (dsv41_triton_capture.pack_aot's fields); `k` None: its identity only."""

    names = fn.arg_names
    div = {names[p[0]]: [list(x) for x in v] == [["tt.divisibility", 16]] for p, v in attrs.items()}
    runtime = [n for n, t in signature.items() if t != "constexpr"]
    e = {"function": full_name(fn), "signature": dict(signature),
         "constexprs": {names[p[0]]: jsonable(v) for p, v in constexprs.items()},
         "abi": [{"name": n, "divisibility_16": div.get(n, False)} for n in runtime]}
    out = {"fn": fn.fn.__name__, "params": cap.aot_params(e, runtime, jit), "consts": cap.aot_consts(e)}
    if k is None:
        return out
    md = k.metadata._asdict()
    return {"fn": k.name, "hash": k.hash, "name": md["name"], "num_warps": md["num_warps"],
            "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
            "global_scratch": md.get("global_scratch_size", 0) or 0,
            "global_align": md.get("global_scratch_align", 1) or 1,
            "profile_scratch": md.get("profile_scratch_size", 0) or 0, "pdl": bool(md.get("launch_pdl", False)),
            "params": out["params"], "consts": out["consts"], "filled": True}


def sigs_jit(kernels: dict) -> dict:
    """dsv41_triton_capture.specialization()'s form: {function: {do_not_specialize: [...]}}."""

    return {full_name(fn): {"params": [p.name for p in fn.params],
                            "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize]}
            for fns in kernels.values() for fn in fns}


def cmd_sass(a) -> int:
    """Every filled variant whose specialization a served set also holds: SASS (nvdisasm -c) and bytes compared."""

    def load(d):
        return {spec_key(k): k for k in json.loads((Path(d) / "aot.json").read_text())["kernels"]}

    fill, served = load(a.fill), load(a.served)
    res = {"filled": len(fill), "served": len(served), "both": 0, "sass_equal": 0, "bytes_equal": 0, "only_filled": 0,
           "differ": []}
    for key, k in fill.items():
        s = served.get(key)
        if s is None:
            res["only_filled"] += 1
            continue
        res["both"] += 1
        mine = (Path(a.fill) / "cubins" / f"{k['hash']}.cubin").read_bytes()
        theirs = (Path(a.served) / "cubins" / f"{s['hash']}.cubin").read_bytes()
        same = sass(mine) == sass(theirs)
        res["sass_equal"] += same
        res["bytes_equal"] += mine == theirs
        if not same:
            res["differ"].append({"fn": k["fn"], "filled": k["hash"], "served": s["hash"]})
    print(json.dumps(res, indent=1))
    return 0 if not res["differ"] else 1


def cmd_merge(a) -> int:
    """The filled variants a served set lacks (by specialization, not hash: the fill's hashes are another host's)."""

    dst, src = Path(a.dst), Path(a.src)
    d = json.loads((dst / "aot.json").read_text())
    have = {spec_key(k) for k in d["kernels"]}
    added = 0
    for k in json.loads((src / "aot.json").read_text())["kernels"]:
        if spec_key(k) in have:
            continue
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", dst / "cubins" / f"{k['hash']}.cubin")
        d["kernels"].append(k)
        have.add(spec_key(k))
        added += 1
    (dst / "aot.json").write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({"added": added, "kernels": len(d["kernels"])}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tree(p):
        p.add_argument("--py", required=True, help="the served Python tree's src/ (the served engine's commit)")
        p.add_argument("--served-py", default=SERVED_PY)
        p.add_argument("--served-triton", default=SERVED_TRITON)
        p.add_argument("--stamps", help="stamps.json: compile where the served paths hold the image's files")

    p = sub.add_parser("sigs")
    tree(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("options")
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stamps")
    p.add_argument("--cubins", nargs="*", default=[])
    p.add_argument("--root", nargs="*", default=[], help="DIR[=SERVED]: stat the .py files under DIR, served at "
                   "SERVED (in the served image: /dsv41-tf/src and its Triton)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    tree(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--cubins", help="the captured cubins (<hash>.cubin): SASS and bytes")
    p.add_argument("--only", nargs="*")
    p = sub.add_parser("compile")
    tree(p)
    p.add_argument("--needs", required=True)
    p.add_argument("--options", required=True)
    p.add_argument("--have", nargs="*", help="aot sets whose variants are not compiled again")
    p.add_argument("--arch", type=int, default=121)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sass")
    p.add_argument("--fill", required=True, help="the filled aot dir")
    p.add_argument("--served", required=True, help="a served aot dir (its captured cubins)")
    p = sub.add_parser("merge")
    p.add_argument("dst")
    p.add_argument("src")
    a = ap.parse_args()
    if a.cmd == "sass":
        return cmd_sass(a)
    if a.cmd == "merge":
        return cmd_merge(a)
    if a.cmd == "sigs":
        kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
        Path(a.out).write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "kernels": sigs(kernels)},
                                          indent=1) + "\n")
        print(json.dumps({"kernels": len(kernels), "out": a.out}))
        return 0
    if a.cmd == "options":
        rules, bad = derive_options([Path(m) for m in a.manifests])
        for f, (kw, site) in SOURCE.items():
            rules.setdefault(f, dict(kw, source=site))
        Path(a.out).write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "functions": rules,
                                           "rules": sorted(RULES)}, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"functions": len(rules), "disagree": bad}, indent=1))
        return 0 if not bad else 1
    if a.cmd == "stamps":
        st = stamps_of_cubins([Path(d) for d in a.cubins])
        for r in a.root:
            d, _, served = r.partition("=")
            for f in Path(d).rglob("*.py"):
                s = f.stat()
                st[(served or d).rstrip("/") + "/" + str(f.relative_to(d))] = (int(s.st_mtime), s.st_size)
        Path(a.out).write_text(json.dumps(st, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"files": len(st), "out": a.out}))
        return 0
    if a.cmd == "verify":
        return cmd_verify(a)
    return cmd_compile(a)


if __name__ == "__main__":
    raise SystemExit(main())
