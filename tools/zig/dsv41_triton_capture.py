#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
"""Op capture: reference blocks on one GPU at rank 0's TP=2 shapes, each launch recorded with buffer bytes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = {"A": (1, 2, 3), "B": (20, 21, 22, 23, 24), "C": (40,), "D": (8, 13, 14, 15),
        "P": (20, 36, 37, 38, 39)}
# D: Engram mid-stack (8 is 13-15's KV source); P: DSpark's drafting pass (20 is 36-39's KV and candidate source)

# DSV41_CAPTURE_WINDOWS (default 1,3,16): the decode windows' rows in order, up to 64; meta.json "windows" carries them
WINDOWS = tuple(int(x) for x in (os.environ.get("DSV41_CAPTURE_WINDOWS", "") or "1,3,16").split(","))
if not WINDOWS or any(not 1 <= n <= 64 for n in WINDOWS):
    raise ValueError(f"DSV41_CAPTURE_WINDOWS={os.environ.get('DSV41_CAPTURE_WINDOWS')!r}: rows 1..64, comma separated")
PREFILL = int(os.environ.get("DSV41_CAPTURE_PREFILL_ROWS", "256"))   # rows of the recorded prefill segment after the windows
#   (the fast prefill kernels; 2048 = the served prefill chunk, the Zig prefill's launch list)
KV_MODE = "split" if os.environ.get("DSV41_CAPTURE_SPLIT", "0") == "1" else ("pool" if os.environ.get("DSV41_CAPTURE_PAGED", "0") == "1" else "")
#   the forward's slot in a KV pool (DSV41_CAPTURE_PAGED) or a split pool (DSV41_CAPTURE_SPLIT); "" = contiguous stores
SCAN_MAX = 1 << 20          # int64 elements scanned for device addresses a tensor at most


def fbits(v: float) -> dict:
    return {"t": "float", "v": v, "f32": "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0],
            "f64": "0x%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}


class Capture:
    """Buffers by storage, weights by name, blobs by content; the op log."""

    def __init__(self, out: Path) -> None:
        import torch

        self.torch = torch
        self.out = out
        (out / "blobs").mkdir(parents=True, exist_ok=True)
        self.ops = open(out / "ops.jsonl", "w")
        self.weights: list[tuple[int, int, str]] = []      # (base, end, name), sorted
        self.keep: list = []                                # every named tensor, alive until the end
        self.recording = False
        self.set = ""
        self.phase = ""
        self.seq = 0
        self.blob_bytes = 0
        self.counts: dict[str, int] = {}
        self.kv_sets: dict[str, dict] = {}                 # each set's pool (dsv41_kv_ref.KV.describe)

    # -- weights ---------------------------------------------------------------------------------------------------
    def name_weights(self, rank_w, layers) -> dict:
        """Every tensor of the rank's tree by the Zig loader's names; returns the digests for weights.json."""

        torch = self.torch
        found: dict[str, torch.Tensor] = {}

        def x3(prefix, v) -> None:
            for f in ("trellis", "suh", "svh", "data"):
                if hasattr(v, f) and isinstance(getattr(v, f), torch.Tensor):
                    found[f"{prefix}.{f}"] = getattr(v, f)

        for lw in layers:
            p = f"L{lw.index}"
            found[f"{p}.attn_norm"], found[f"{p}.ffn_norm"] = lw.attn_norm, lw.ffn_norm
            for which, hc in (("hc_attn", lw.hc_attn), ("hc_ffn", lw.hc_ffn)):
                for f in ("fn", "base", "scale"):
                    found[f"{p}.{which}.{f}"] = getattr(hc, f)
            a = lw.attn
            for f in ("wq_a", "wkv", "wq_b", "wo_b", "comp_wkv", "comp_wgate", "ix_wq_b", "ix_wk"):
                if getattr(a, f) is not None:
                    x3(f"{p}.attn.{f}", getattr(a, f))
            for j, g in enumerate(a.wo_a):
                x3(f"{p}.attn.wo_a.{j}", g)
            for f, n in (("q_norm", "q_norm"), ("kv_norm", "kv_norm"), ("sink", "sink"), ("comp_norm", "comp_norm"),
                         ("ix_wp", "ix_wp"), ("ix_knorm", "ix_knorm")):
                if getattr(a, f) is not None:
                    found[f"{p}.attn.{n}"] = getattr(a, f)
            m = lw.moe
            found[f"{p}.moe.gate"], found[f"{p}.moe.bias"] = m.gate, m.bias
            for f in ("w1", "w2", "w3"):
                x3(f"{p}.moe.{f}", getattr(m, f))
            for i, (w1, w2, w3) in enumerate(m.shared):
                for f, v in (("w1", w1), ("w2", w2), ("w3", w3)):
                    x3(f"{p}.moe.shared.{i}.{f}", v)
            if lw.engram is not None:
                x3(f"{p}.engram.wkv", lw.engram.wkv)
                found[f"{p}.engram.qk"] = lw.engram.qk
        found["embed"], found["norm"] = rank_w.embed, rank_w.norm
        x3("head", rank_w.head)
        dw = getattr(self, "draft_w", None)          # set P: DSpark's heads by the Zig loader's names (named.dspark)
        if dw is not None:
            x3("dspark.main_proj", dw.main_proj)
            found["dspark.main_norm"], found["dspark.norm"] = dw.main_norm, dw.norm
            found["dspark.markov.w1"], found["dspark.markov.w2"] = dw.w1, dw.w2
            if dw.conf is not None:
                found["dspark.conf"] = dw.conf
        digests, spans = {}, []
        for name, t in found.items():
            t = t.contiguous() if not t.is_contiguous() else t
            raw = t.view(torch.uint8).reshape(-1) if t.numel() else torch.empty(0, dtype=torch.uint8)
            digests[name] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
                             "nbytes": int(raw.numel()), "sha256": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()}
            orig = found[name]
            if orig.is_cuda and orig.numel():
                spans.append((orig.data_ptr(), orig.data_ptr() + raw.numel(), name))
        # keep every named tensor alive for the capture: a freed weight's addresses would rebind a later buffer
        self.keep.extend(found.values())
        self.weights = sorted(set(self.weights) | set(spans))
        return digests

    def weight_of(self, addr: int):
        for b, e, n in self.weights:
            if b <= addr < e:
                return b, n
        return None

    # -- blobs -----------------------------------------------------------------------------------------------------
    def blob(self, storage) -> str:
        torch = self.torch
        n = storage.nbytes()
        t = torch.empty(0, dtype=torch.uint8, device=storage.device)
        t.set_(storage, 0, (n,))
        data = t.cpu().numpy().tobytes()
        h = hashlib.sha256(data).hexdigest()
        p = self.out / "blobs" / h[:2] / f"{h}.bin"
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
            self.blob_bytes += n
        return h

    # -- arguments -------------------------------------------------------------------------------------------------
    def describe(self, v, bufs: dict, tensors: list):
        """An argument as the replay reads it; tensors register their storage (``bufs``: key -> entry)."""

        torch = self.torch
        if isinstance(v, torch.Tensor):
            st = v.untyped_storage()
            base, off = st.data_ptr(), v.data_ptr() - st.data_ptr()
            d = {"t": "tensor", "dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape),
                 "stride": list(v.stride())}
            w = self.weight_of(v.data_ptr()) if v.is_cuda else None
            if w is not None:
                d.update(weight=w[1], offset=v.data_ptr() - w[0])
                return d
            if not v.is_cuda:
                d.update(host=True)
                return d
            key = f"{base}:{st.nbytes()}"
            if key not in bufs:
                bufs[key] = {"id": len(bufs), "key": key, "base": base, "nbytes": st.nbytes(), "storage": st}
            d.update(buf=bufs[key]["id"], offset=off)
            tensors.append((v, bufs[key]))
            return d
        if isinstance(v, bool):
            return {"t": "bool", "v": v}
        if isinstance(v, int):
            d = {"t": "int", "v": v}
            w = self.weight_of(v) if v > 1 << 32 else None
            if w is not None:
                d.update(ptr={"weight": w[1], "delta": v - w[0]})
            return d
        if isinstance(v, float):
            return fbits(v)
        if v is None:
            return {"t": "none"}
        if isinstance(v, (list, tuple)):
            return {"t": "list", "items": [self.describe(x, bufs, tensors) for x in v]}
        if type(v).__name__ == "constexpr":
            return self.describe(v.value, bufs, tensors)
        return {"t": "other", "repr": repr(v)[:200]}

    def relocs(self, tensors: list, bufs: dict) -> list:
        """Device addresses stored in int64 tensors: each points into a weight or a buffer of this op."""

        torch = self.torch
        out, seen = [], set()
        ranges = sorted((b["base"], b["base"] + b["nbytes"], b["id"]) for b in bufs.values())
        for v, b in tensors:
            if v.dtype not in (torch.int64, torch.uint64) or v.numel() > SCAN_MAX or (b["id"], v.data_ptr()) in seen:
                continue
            seen.add((b["id"], v.data_ptr()))
            vals = v.detach().reshape(-1).cpu().tolist()
            starts = [v.data_ptr() - b["base"] + i * 8 * v.stride(-1) for i in range(len(vals))] if v.dim() == 1 \
                else None
            for i, x in enumerate(vals):
                if x < 1 << 32:
                    continue
                at = starts[i] if starts is not None else None
                if at is None:                      # non-1-D tables: element offset of a contiguous tensor only
                    if not v.is_contiguous():
                        continue
                    at = v.data_ptr() - b["base"] + 8 * i
                w = self.weight_of(x)
                if w is not None:
                    out.append({"buf": b["id"], "at": at, "weight": w[1], "delta": x - w[0]})
                    continue
                for lo, hi, bid in ranges:
                    if lo <= x < hi:
                        out.append({"buf": b["id"], "at": at, "target": bid, "delta": x - lo})
                        break
                else:
                    out.append({"buf": b["id"], "at": at, "unresolved": x})
        return out

    # -- one launch ------------------------------------------------------------------------------------------------
    def record(self, row: dict, args: list, kwargs: dict, call, named: bool = False):
        """Dump every buffer the arguments touch, run ``call``, dump them again; log the op (``named``: Triton)."""

        torch = self.torch
        bufs: dict = {}
        tensors: list = []
        row["args"] = [{"name": v[0], **self.describe(v[1], bufs, tensors)} if named else self.describe(v, bufs, tensors)
                       for v in args]
        row["kwargs"] = {k: self.describe(v, bufs, tensors) for k, v in kwargs.items()}
        row["relocs"] = self.relocs(tensors, bufs)
        torch.cuda.synchronize()
        before = {k: self.blob(b["storage"]) for k, b in bufs.items()}
        ret = call()
        torch.cuda.synchronize()
        rbufs, rtensors = {}, []
        if isinstance(ret, torch.Tensor):
            row["ret"] = self.describe(ret, rbufs, rtensors)
            for k, b in rbufs.items():          # a returned tensor's storage: new, unless an argument's
                if k in bufs:
                    row["ret"]["buf"] = bufs[k]["id"]
                else:
                    b["id"] = len(bufs)
                    row["ret"]["buf"] = b["id"]
                    bufs[k] = b
                    before[k] = None
        row["buffers"] = [{"id": b["id"], "key": k, "nbytes": b["nbytes"], "before": before[k],
                           "after": self.blob(b["storage"])} for k, b in bufs.items()]
        row.update(seq=self.seq, set=self.set, phase=self.phase)
        self.seq += 1
        self.counts[row["name"]] = self.counts.get(row["name"], 0) + 1
        self.ops.write(json.dumps(row) + "\n")
        return ret


def install(cap: Capture, rec) -> None:
    """Triton's JITFunction.run and every extension module tensorfold.cuda.build.load returns, wrapped."""

    from triton.runtime.jit import JITFunction
    import tensorfold.cuda.build as build

    original = JITFunction.run

    def run(fn, *args, grid, warmup, **kwargs):
        if warmup or not cap.recording:
            return original(fn, *args, grid=grid, warmup=warmup, **kwargs)
        names = fn.arg_names
        bound = inspect.signature(fn.fn).bind_partial(*args, **{k: v for k, v in kwargs.items() if k in names})
        bound.apply_defaults()
        g = grid(dict(bound.arguments)) if callable(grid) else grid
        g = [int(x) for x in (tuple(g) + (1, 1, 1))[:3]]
        row = {"kind": "triton", "name": fn.fn.__name__, "function": f"{fn.fn.__module__}.{fn.fn.__qualname__}",
               "grid": g, "options": {k: (v if isinstance(v, (int, bool, str)) or v is None else repr(v))
                                      for k, v in kwargs.items() if k not in names}}
        got = {}

        def call():
            k = got["k"] = original(fn, *args, grid=grid, warmup=warmup, **kwargs)
            row["hash"], row["name"] = k.hash, k.name
            rec._triton(fn, k, args, kwargs, grid)

        cap.record(row, [(n, bound.arguments[n]) for n in names if n in bound.arguments], {}, call, named=True)
        return got.get("k")

    JITFunction.run = run
    loads = build.load

    def load(name, *a, **kw):
        mod = loads(name, *a, **kw)
        for f in dir(mod):
            fnc = getattr(mod, f)
            if f.startswith("_") or not callable(fnc) or getattr(fnc, "_captured", False):
                continue

            def wrapper(*args, _f=fnc, _name=f, **kwargs):
                if not cap.recording:
                    return _f(*args, **kwargs)
                row = {"kind": "ext", "ext": name, "func": _name, "name": f"{name}.{_name}"}
                return cap.record(row, list(args), kwargs, lambda: _f(*args, **kwargs))

            wrapper._captured = True
            setattr(mod, f, wrapper)
        return mod

    build.load = load


class ZeroPeer:
    """World 2, rank 0: the peer's half of every all-gather is zeros."""

    rank, world = 0, 2

    def all_gather(self, send, recv) -> None:
        n = send.numel()
        recv[:n].copy_(send.reshape(-1))
        recv[n:].zero_()

    def barrier(self) -> None:
        pass


class IndexRows:
    """Engram rows as a function of the row index: fp32 [N, 256] in (-0.25, 0.25), multiples of 1/1024."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def rows(self, index):
        import torch

        i = index.to(torch.int64).reshape(-1, 1)
        j = torch.arange(self.dim, dtype=torch.int64, device=i.device).reshape(1, -1)
        return (((i * 2654435761 + j * 40503) % 511) - 255).to(torch.float32) / 1024.0


def run_draft(cap, rec, s, loader, cfg, tree, layers, tmap, dev, a, g, vh, Forward, Seg, Triton, NgramHasher):
    """Set P: backbone layers 36-39 with taps, a prompt, then the drafter's ingests and passes (i0 d0 i1 d1)."""

    import torch
    from tensorfold.families.deepseek_v41.cuda import drafter as DR
    from tensorfold.families.deepseek_v41.cuda.forward import greedy

    dw = DR.load(loader)
    cap.draft_w = dw
    digests = cap.name_weights(tree, layers + list(dw.layers))
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=ZeroPeer(), device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048, taps=True)
    dr = DR.Drafter(fw, dw)
    dr.reset(0)
    ids = torch.randint(3, vh, (a.prompt + 1,), generator=g).tolist()
    logits = fw.prompt(ids[:a.prompt])
    anchor = int(greedy(fw, logits)[0])
    cap.set = s

    def rec_phase(name, fn):
        cap.phase = name
        cap.recording = True
        try:
            with rec.scope(f"{s}-{name}"):
                return fn()
        finally:
            cap.recording = False

    taps = fw.slot.taps
    n0 = next(iter(taps.values())).shape[0]
    rec_phase("i0", lambda: dr.ingest(0, fw.slot.pos - n0, taps))
    rec_phase("d0", lambda: dr.propose([0], [anchor], [fw.slot.pos]))
    lg = fw.run([Seg(0, fw.slot.pos, (anchor,))])
    nxt = int(greedy(fw, lg)[0])
    fw.keep(0, 0)
    rec_phase("i1", lambda: dr.ingest(0, fw.slot.pos - 1, fw.slot.taps))
    rec_phase("d1", lambda: dr.propose([0], [nxt], [fw.slot.pos]))
    # DSV41_CAPTURE_DRAFT_SLOTS=k > 1: one pass over k slots in row mode, phase "q<k>"; slots 1.. have empty contexts
    k = int(os.environ.get("DSV41_CAPTURE_DRAFT_SLOTS", "") or 1)
    if k > 1:
        if not dr.rowmode:
            raise ValueError("DSV41_CAPTURE_DRAFT_SLOTS: the drafter is not in row mode (csa2 'triton')")
        for sl in range(1, k):
            dr.reset(sl)
        pos = fw.slot.pos
        rec_phase(f"q{k}", lambda: dr.propose(list(range(k)), [nxt] * k, [pos] * k))
    torch.cuda.synchronize()
    cap.draft_w = None
    del fw, dr, dw, tree, layers
    return digests


def run_set(cap, rec, s, loader, W, cfg, embed, norm, head, tmap, dev, a, g, vh, Forward, Seg, Triton, NgramHasher):
    """One set: blocks loaded, a prompt, recorded decode windows and prefill segment; returns its weight digests."""

    import torch

    layers = [loader.layer(L) for L in SETS[s]]
    tree = W.RankW(0, 2, 0, embed, layers, norm, head, None, {})
    if s == "P":
        if KV_MODE:
            raise ValueError("set P: contiguous stores only (DSV41_CAPTURE_PAGED / DSV41_CAPTURE_SPLIT unset)")
        return run_draft(cap, rec, s, loader, cfg, tree, layers, tmap, dev, a, g, vh, Forward, Seg, Triton, NgramHasher)
    digests = cap.name_weights(tree, layers)
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    kv = None
    if KV_MODE:                 # the slot in a (split) KV pool, as serving binds it (dsv41_kv_ref.KV)
        import dsv41_kv_ref as KR
        from tensorfold.families.deepseek_v41.cuda.slots import SlotsForward

        fw = SlotsForward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=ZeroPeer(),
                          device=dev, limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap),
                          slots=1, contiguous=False, pchunk=2048)
        kv = KR.KV(fw, Path(a.pack), KV_MODE, 0, 2, a.limit, dev)
    else:
        fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=ZeroPeer(), device=dev,
                     limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                     contiguous=True, pchunk=2048)
    # ids in rank 0's vocabulary half (its embedding rows; the peer's are zero)
    ids = torch.randint(3, vh, (a.prompt + sum(WINDOWS) + PREFILL,), generator=g).tolist()
    if kv is not None:
        kv.grow(a.prompt)
    fw.prompt(ids[:a.prompt])
    pos = a.prompt
    cap.set = s
    # DSV41_CAPTURE_PREFILL_ROWS=0: decode windows only (the prefill blobs are large)
    for n, prefill in [(n, False) for n in WINDOWS] + ([(PREFILL, True)] if PREFILL > 0 else []):
        cap.phase = f"p{n}" if prefill else f"w{n}"
        if kv is not None:
            kv.grow(fw.slot.pos + n)            # the round's ensure: pages mapped before the run
        cap.recording = True
        try:
            with rec.scope(f"{s}-{cap.phase}"):
                fw.run([Seg(0, fw.slot.pos, tuple(ids[pos:pos + n]))], prefill=prefill)
        finally:
            cap.recording = False
        fw.keep(0, n - 1)
        pos += n
    torch.cuda.synchronize()
    if kv is not None:
        cap.kv_sets[s] = kv.describe(fw)
    del fw, tree, layers, kv
    return digests


def aot_consts(k: dict) -> dict:
    """A kernel's constexprs as aot.zig matches them: ints and bools as int, floats as fp32 bits, others left out."""

    out = {}
    for name, v in k.get("constexprs", {}).items():
        if isinstance(v, bool):
            out[name] = {"int": int(v)}
        elif isinstance(v, int):
            out[name] = {"int": v}
        elif isinstance(v, float):
            out[name] = {"f32": struct.unpack("<I", struct.pack("<f", v))[0]}
        elif isinstance(v, dict) and "fp32_bits" in v:      # triton_aot_manifest's float form
            out[name] = {"f32": int(v["fp32_bits"], 16)}
    return out


def aot_params(k: dict, runtime: list, jit: dict) -> list:
    """Runtime parameters with Triton's specialization: div16 from the signature, nospec from do_not_specialize."""

    div = {x["name"]: bool(x.get("divisibility_16", False)) for x in k.get("abi", [])}
    nospec = set(jit.get(k.get("function", ""), {}).get("do_not_specialize", []))
    return [{"name": n, "type": k["signature"][n], "div16": div.get(n, False), "nospec": n in nospec} for n in runtime]


def pack_aot(manifest: Path, cache: Path | None, out: Path, jit: dict | None = None) -> dict:
    """The launched kernels in aot.json form (cubins by hash, constexprs, div16, nospec); misfits go to problems."""

    import shutil

    jit = jit or {}
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    kernels, problems = [], []
    for k in json.loads(manifest.read_text())["kernels"]:
        runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
        ptx = [x["name"] for x in k["abi"]]
        if ptx and ptx != runtime + ["global_scratch", "profile_scratch"]:
            problems.append({"hash": k["hash"], "name": k["name"], "ptx": ptx, "runtime": runtime})
            continue
        if cache is not None:
            shutil.copyfile(cache / k["cubin"], out / "cubins" / f"{k['hash']}.cubin")
        md = k["metadata"]
        kernels.append({"fn": k["name"], "hash": k["hash"], "name": md["name"], "num_warps": md["num_warps"],
                        "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
                        "global_scratch": md.get("global_scratch_size", 0) or 0,
                        "global_align": md.get("global_scratch_align", 1) or 1,
                        "profile_scratch": md.get("profile_scratch_size", 0) or 0,
                        "pdl": bool(md.get("launch_pdl", False)),
                        "params": aot_params(k, runtime, jit), "consts": aot_consts(k)})
    (out / "aot.json").write_text(json.dumps({"generator": "tools/zig/dsv41_triton_capture.py", "kernels": kernels},
                                             indent=1) + "\n")
    return {"kernels": len(kernels), "problems": problems}


def specialization() -> dict:
    from triton.runtime.jit import JITFunction
    import gc

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            out[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                "params": [p.name for p in fn.params], "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--prompt", type=int, default=1536, help="prompt tokens before the windows (past the top-512)")
    ap.add_argument("--limit", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import triton_aot_manifest as aot

    cap = Capture(out)
    rec = aot.Recorder()
    install(cap, rec)
    if KV_MODE == "split":
        os.environ["TF_DSV41_KV_SPLIT"] = "1"           # the knob the split stands for (meta.json's env)
        import dsv41_kv_ref as KR

        KR.install_glue(cap)
    import torch
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg

    dev = torch.device("cuda:0")
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, 0, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", 0, vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    meta = {"pack": str(pack), "sets": {}, "windows": list(WINDOWS), "prompt": a.prompt, "limit": a.limit,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("TF_DSV41", "GLM53_TF", "TRITON"))},
            "peer": "zeros", "engram_rows": "index function"}
    import triton

    meta["triton"] = triton.__version__
    if KV_MODE:
        meta["kv"] = {"mode": "paged" if KV_MODE == "pool" else KV_MODE, "prefill_rows": PREFILL,
                      "glue": ["kvsplit.gather", "kvsplit.all_gather", "kvsplit.var_gather"] if KV_MODE == "split"
                      else [], "peer_rows": "zeros" if KV_MODE == "split" else None}
    weights: dict = {}
    g = torch.Generator().manual_seed(4101)
    meta["failed_sets"] = {}
    for s in a.sets.split(","):
        t0 = time.time()
        try:
            got = run_set(cap, rec, s, loader, W, cfg, embed, norm, head, tmap, dev, a, g, vh, Forward, Seg, Triton,
                          NgramHasher)
        except Exception as e:          # a set that fails leaves the others' records whole
            import traceback

            traceback.print_exc()
            cap.recording = False
            meta["failed_sets"][s] = repr(e)[:500]
            torch.cuda.empty_cache()
            continue
        weights.update(got)
        ls = list(SETS[s]) + (list(range(cfg.num_hidden_layers, cfg.num_hidden_layers + cfg.num_nextn_predict_layers))
                              if s == "P" else [])          # P: the drafter's blocks load too
        meta["sets"][s] = {"layers": ls, "load_and_run_s": round(time.time() - t0, 1), "ops": cap.seq}
        if s in cap.kv_sets:
            meta["sets"][s]["kv"] = cap.kv_sets[s]
        torch.cuda.empty_cache()
    cap.ops.close()
    (out / "weights.json").write_text(json.dumps(weights, indent=0, sort_keys=True) + "\n")
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    aot.build(cache, out / "launches.json", out / "manifest.json", None)
    meta["aot"] = pack_aot(out / "manifest.json", cache, out / "aot", json.loads((out / "jit.json").read_text()))
    meta.update(ops=cap.seq, blob_bytes=cap.blob_bytes, counts=cap.counts)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"ops": cap.seq, "blob_MB": round(cap.blob_bytes / 1e6, 1), "kernels": len(rec.kernels)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
