"""triton_fill's served-tree loading, one variant through the JIT's steps, and the options rules."""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# where the served family's kernels live (the EXL3 prefill GEMM is shared code): other families reuse kernel names
KERNEL_DIRS = ("tensorfold/families/deepseek_v41", "tensorfold/cuda")
# the Zig engine's own Triton kernels: a package beside this script, imported after the served tree (shared helpers)
OWN_DIR = HERE / "dsv41_triton"
OWN_PKG = "dsv41_zig_triton"
# call-site options a launch passes (the rest are the backend's defaults, the same for every launch)
OPTION_KEYS = ("num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion", "launch_pdl",
               "launch_cooperative_grid")


def jsonable(v):
    """A constexpr as triton_aot_manifest records it (floats with their fp32 / fp64 bits)."""

    if isinstance(v, float):
        return {"float": v, "fp32_bits": "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0],
                "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}
    if hasattr(v, "value") and type(v).__name__ == "constexpr":
        return jsonable(v.value)
    return v


def unjson(v):
    """A recorded constexpr back to its Python value (a float from its fp64 bits: exact)."""

    if isinstance(v, dict) and "fp64_bits" in v:
        return struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]
    if isinstance(v, dict) and "float" in v:
        return float(v["float"])
    return v


# ---- the served tree's kernels ------------------------------------------------------------------------------------

def load_tree(py: Path, served_py: str, served_triton: str) -> dict:
    """Import every family module defining a Triton kernel, line info mapped to the served paths: {name: [JIT, ...]}."""

    sys.path.insert(0, str(py))
    import triton
    import triton.compiler.code_generator as cg
    from triton.runtime.jit import JITFunction

    pairs = [(str(py), served_py), (os.path.dirname(triton.__file__), served_triton)]
    orig = cg.get_jit_fn_file_line

    def mapped(fn):
        name, line = orig(fn)
        for a, b in pairs:
            if name.startswith(a):
                return b + name[len(a):], line
        return name, line

    cg.get_jit_fn_file_line = mapped
    sys.path.append(str(OWN_DIR))
    out: dict = {}
    files = [(f, py) for d in KERNEL_DIRS for f in sorted((py / d).rglob("*.py"))]
    files += [(f, OWN_DIR) for f in sorted((OWN_DIR / OWN_PKG).rglob("*.py"))]
    for f, root in files:
        text = f.read_text(errors="replace")
        if "triton.jit" not in text:
            continue
        mod = ".".join(f.relative_to(root).with_suffix("").parts)
        try:
            m = importlib.import_module(mod)
        except Exception as e:                      # a module that needs a GPU extension at import
            print(f"[fill] skip {mod}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for v in vars(m).values():
            if isinstance(v, JITFunction) and v.fn.__module__ == mod:
                out.setdefault(v.fn.__name__, []).append(v)
    return out


def full_name(fn) -> str:
    return f"{fn.fn.__module__}.{fn.fn.__qualname__}"


def sigs(kernels: dict) -> dict:
    """{name: [{function, params: [{name, constexpr, nospec, noalign}]}]}: what the Zig side reduces calls with."""

    out = {}
    for name, fns in sorted(kernels.items()):
        out[name] = [{"function": full_name(fn), "params": [
            {"name": p.name, "constexpr": bool(p.is_constexpr), "nospec": bool(p.do_not_specialize),
             "noalign": bool(p.do_not_specialize_on_alignment)} for p in fn.params]} for fn in fns]
    return out


def resolve(kernels: dict, name: str, function: str | None, arg_names) -> object:
    """The JITFunction a launch names: by its module path when known, else the one whose parameters hold the call's."""

    fns = kernels.get(name) or []
    if function:
        for fn in fns:
            if full_name(fn) == function:
                return fn
        raise KeyError(f"{function}: not in the tree")
    fit = [fn for fn in fns if set(arg_names) <= set(fn.arg_names)]
    if len(fit) != 1:
        raise KeyError(f"{name}: {len(fit)} kernels take {sorted(arg_names)}")
    return fit[0]


# ---- one variant through the JIT's steps ------------------------------------------------------------------------

_DT = None


def torch_dtype(ptr_type: str):
    global _DT
    import torch

    if _DT is None:
        _DT = {"*bf16": torch.bfloat16, "*fp16": torch.float16, "*fp32": torch.float32, "*fp64": torch.float64,
               "*i8": torch.int8, "*u8": torch.uint8, "*i16": torch.int16, "*i32": torch.int32, "*i64": torch.int64,
               "*i1": torch.bool, "*fp8e4nv": torch.float8_e4m3fn, "*fp8e5": torch.float8_e5m2}
    return _DT[ptr_type]


def stand_in(spec: dict):
    """A runtime argument of the recorded specialization: a CPU tensor (aligned or not), an int, a float."""

    import torch

    ty = spec["type"]
    if ty.startswith("*"):
        t = torch.zeros(64, dtype=torch_dtype(ty))
        return t if spec.get("div16", False) else t[1:]
    if ty in ("fp32", "fp64", "fp16", "bf16"):
        return float(spec.get("value", 0.5))
    if "value" in spec:
        return int(spec["value"])
    big = ty in ("i64", "u64")
    return (1 << 36 if big else 0) + (16 if spec.get("div16") else 17)


class Target:
    def __init__(self, arch: int) -> None:
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend

        self.target = GPUTarget("cuda", arch, 32)
        self.backend = make_backend(self.target)


def specialize(fn, tgt: Target, consts: dict, runtime: dict, options: dict):
    """(options, signature, constexprs, attrs) as JITFunction.run's binder + _pack_args make them."""

    from triton.runtime.jit import create_function_from_signature

    binder = create_function_from_signature(fn.signature, fn.params, tgt.backend)
    kwargs = {}
    for p in fn.params:
        if p.name in consts:
            kwargs[p.name] = consts[p.name]
        elif p.name in runtime:
            kwargs[p.name] = stand_in(runtime[p.name])
    kwargs.update(options)
    bound, spec, opts = binder(**kwargs)
    return fn._pack_args(tgt.backend, kwargs, bound, spec, opts)


def compile_one(fn, tgt: Target, consts: dict, runtime: dict, options: dict):
    import triton
    from triton.compiler import ASTSource

    opts, signature, constexprs, attrs = specialize(fn, tgt, consts, runtime, options)
    k = triton.compile(ASTSource(fn, signature, constexprs, attrs), target=tgt.target, options=opts.__dict__)
    return k, signature, constexprs, attrs


def entry_of(fn, k, signature: dict, constexprs: dict, attrs: dict) -> dict:
    """The compiled variant as a manifest entry (dsv41_triton_capture.pack_aot's input)."""

    names = fn.arg_names
    md = k.metadata._asdict()
    named_c = {names[p[0]]: jsonable(v) for p, v in constexprs.items()}
    div = {names[p[0]]: [list(a) for a in v] == [["tt.divisibility", 16]] for p, v in attrs.items()}
    runtime = [n for n, t in signature.items() if t != "constexpr"]
    return {"name": k.name, "function": full_name(fn), "hash": k.hash, "signature": dict(signature),
            "constexprs": named_c, "abi": [{"name": n, "divisibility_16": div.get(n, False)} for n in runtime],
            "metadata": {x: md.get(x) for x in ("name", "num_warps", "num_ctas", "num_stages", "maxnreg", "shared",
                                                 "global_scratch_size", "global_scratch_align", "profile_scratch_size",
                                                 "launch_pdl", "arch", "triton_version")}}


def need_of_manifest(k: dict, fn) -> tuple[dict, dict]:
    """A captured kernel as (constexprs, runtime arguments): the inverse of what Triton recorded."""

    consts = {n: unjson(v) for n, v in k["constexprs"].items()}
    attrs = k.get("attrs", {})
    runtime = {}
    for n, ty in k["signature"].items():
        if ty == "constexpr":
            continue
        runtime[n] = {"type": ty, "div16": [["tt.divisibility", 16]] == attrs.get(n)}
    return consts, runtime


def options_of(k: dict) -> dict:
    md = k["metadata"]
    return {x: md[x] for x in OPTION_KEYS if x in md and md[x] is not None}


# ---- options rules ----------------------------------------------------------------------------------------------

# call sites whose options follow their constexprs (the served code, mirrored; `options` checks them on every capture)
RULES = {
    # mhc_dec.launch: maxnreg = TF_DSV41_MHC_DEC_MAXNREG (128) when RB <= 4 and num_warps == 4
    "tensorfold.families.deepseek_v41.cuda.mhc_dec._site_dec":
        lambda c, o: {**o, **({"maxnreg": 128} if c.get("RB", 99) <= 4 and o.get("num_warps") == 4 else {})},
}


# call sites of kernels no captured run launched: their launch's options as passed (the backend fills the rest)
SOURCE = {
    "tensorfold.families.deepseek_v41.cuda.csa2.dtopk._dtopk": ({"num_warps": 8}, "csa2/dtopk.py:167 num_warps=WARPS"),
    "tensorfold.families.deepseek_v41.cuda.prune._prune": ({"num_warps": 1}, "prune.py:296 num_warps=1"),
}


def rule_options(rules: dict, function: str, consts: dict) -> dict:
    r = rules.get(function)
    if r is None:
        raise KeyError(f"{function}: no options rule (no captured variant; add one to options.json)")
    base = {k: v for k, v in r.items() if k != "source"}
    fx = RULES.get(function)
    return fx(consts, base) if fx else base


def derive_options(manifests: list[Path]) -> tuple[dict, list]:
    """{function: options} (the captured variants' options without RULES' parts), and every capture they disagree on."""

    seen: dict = {}
    for m in manifests:
        for k in json.loads(m.read_text()).get("kernels", []):
            o = options_of(k)
            f = k["function"]
            if f in RULES:
                o.pop("maxnreg", None)
            seen.setdefault(f, {}).setdefault(json.dumps(o, sort_keys=True), []).append((str(m), k["hash"]))
    rules, bad = {}, []
    for f, opts in seen.items():
        if len(opts) > 1:
            bad.append({"function": f, "options": list(opts)})
            continue
        rules[f] = json.loads(next(iter(opts)))
    for m in manifests:
        for k in json.loads(m.read_text()).get("kernels", []):
            f = k["function"]
            if f in rules and rule_options(rules, f, {n: unjson(v) for n, v in k["constexprs"].items()}) != options_of(k):
                bad.append({"function": f, "hash": k["hash"], "manifest": str(m), "rule": "differs"})
    return rules, bad
