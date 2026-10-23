#!/usr/bin/env python3
"""MoE tail candidate versus frozen v5 put_slot+combine, on an idle authorized SM121a GPU.

No model/service changes. Mandatory frozen fatbins provide an independently built baseline.
All five repeats compare bytes, guards, immutable inputs and a separate FP64 reference;
NaN payloads and signed zero compare bytewise too. Optional CUDA-graph timings exclude host launch gaps.
This focused local receipt is not upstream/all-hardware or whole-engine qualification.
"""
import argparse
import ctypes as C
import hashlib
import json
import statistics
from pathlib import Path

import torch


def report(**values):
    print(json.dumps(values), flush=True)


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation}: CUDA driver status {status}")


class Module:
    def __init__(self, path):
        self.d = C.CDLL("libcuda.so.1")
        self.module = C.c_void_p()
        self.d.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.d.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.d.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        self.d.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [
            C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        check(self.d.cuModuleLoad(C.byref(self.module), str(path).encode()), f"load {path}")

    def function(self, name):
        f = C.c_void_p()
        check(self.d.cuModuleGetFunction(C.byref(f), self.module, name.encode()), name)
        resources = {}
        for label, attribute in (("max_threads", 0), ("static", 1), ("local", 3), ("registers", 4)):
            value = C.c_int()
            check(self.d.cuFuncGetAttribute(C.byref(value), attribute, f), label)
            resources[label] = value.value
        report(kind="resources", symbol=name, **resources)
        return f

    def launch(self, fn, grid, block, values):
        args = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        check(self.d.cuLaunchKernel(fn, *grid, block, 1, 1, 0,
                                   C.c_void_p(torch.cuda.current_stream().cuda_stream), args, None), "launch")


class Guard:
    def __init__(self, value):
        self.storage = torch.full((value.numel() + 128,), 0x13579BDF, dtype=torch.int32, device="cuda")
        self.tensor = self.storage[64:-64].view(torch.float32).view(value.shape)
        self.tensor.copy_(value)

    def intact(self):
        assert bool((self.storage[:64] == 0x13579BDF).all()), "prefix guard overwritten"
        assert bool((self.storage[-64:] == 0x13579BDF).all()), "suffix guard overwritten"


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def classification(x):
    return torch.where(torch.isnan(x), 1, torch.where(torch.isposinf(x), 2,
                                                   torch.where(torch.isneginf(x), 3, 0)))


def fixture(rows, width, kind):
    # Two untouched rows before and after exercise nonzero lo, hi slices and output boundaries.
    generator = torch.Generator().manual_seed(713 + 13 * rows + width)
    y = torch.randn((rows + 4, 9, width), generator=generator)
    shared = torch.randn((rows + 4, width), generator=generator)
    wts = torch.rand((rows + 4, 9), generator=generator) * 2 - 1
    if kind == "stress":
        positive = torch.randint(262144, 262656, (rows + 4, 4, width), generator=generator).float()
        y[:, :4] = positive
        y[:, 4:8] = -positive + 17
        shared.fill_(3)
        wts.fill_(1)
    elif kind == "nonfinite":
        wts.fill_(1)
        y[:, 0, :7] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0, 0, 0, float("inf")])
        shared[:, :7] = torch.tensor([0, 0, 0, float("nan"), float("inf"), -float("inf"), -float("inf")])
        # Noncanonical NaN payloads: a raw copy must not canonicalize before the actual FMA.
        shared.view(torch.int32)[:, 3] = 0x7FC00123
        if rows > 1:
            wts[3, 8] = float("nan")
    elif kind == "zero_subnormal":
        tiny = torch.tensor([0, -0.0, 1.401298464324817e-45, -1.401298464324817e-45,
                             1.1754943508222875e-38, -1.1754943508222875e-38, 0, -0.0])
        y.copy_(tiny.repeat(width // 8).view(1, 1, width).expand_as(y))
        shared.copy_(-tiny.repeat(width // 8).view(1, width).expand_as(shared))
        wts.copy_(torch.tensor([1, -1, 0, -0.0, .5, -.5, 1, -1, .5]))
    y[:, 8].fill_(float("nan"))  # candidate must not read or write this stale slot
    return y, shared, wts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path, help="new glm_glue.fatbin")
    p.add_argument("baseline_glue", type=Path, help="frozen v5 glm_glue.fatbin")
    p.add_argument("baseline_ops", type=Path, help="frozen v5 glm_ops.fatbin")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--rounds", type=int, default=500)
    args = p.parse_args()
    assert args.rounds > 0
    assert args.candidate.resolve() != args.baseline_glue.resolve(), "frozen baseline must be separate"
    torch.set_num_threads(4)
    torch.cuda.init()
    anchor = torch.empty(1, device="cuda")
    assert torch.cuda.get_device_capability() == (12, 1), "This experiment is scoped to GB10 SM121"
    candidate, old, ops = [Module(v) for v in (args.candidate, args.baseline_glue, args.baseline_ops)]
    fused = candidate.function("tf_glm_combine_shared")
    combine = old.function("tf_glm_combine")
    put = ops.function("tf_glm_put_slot")
    report(kind="provenance", gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
           artifacts={str(v): hashlib.sha256(v.read_bytes()).hexdigest()
                      for v in (args.candidate, args.baseline_glue, args.baseline_ops)},
           scope="FP32 decode tail only; TP2 local width 4096, 8 routed plus 1 shared; no cross-rank change")
    cases = [(r, 4096, k) for r in range(1, 33) for k in ("random", "stress", "nonfinite", "zero_subnormal")]
    cases += [(r, d, "random") for r in (1, 32) for d in (8, 1016, 1024, 1032)]
    for rows, width, kind in cases:
        y, shared, wts = fixture(rows, width, kind)
        guards = [Guard(v) for v in (y, shared, wts, y,
                                    torch.full_like(shared, -123), torch.full_like(shared, -123))]
        gy, gs, gw, by, bo, co = [v.tensor for v in guards]
        immutable = [v.clone() for v in (gy, gs, gw)]
        lo = 2
        ey = C.c_void_p(gy[lo].data_ptr())
        sy = C.c_void_p(gs[lo].data_ptr())
        wt = C.c_void_p(gw[lo].data_ptr())
        legacy_y = C.c_void_p(by[lo].data_ptr())
        grid = (rows, (width + 1023) // 1024, 1)

        def chain(mode):
            if mode == "old":
                ops.launch(put, ((rows * width + 255) // 256, 1, 1), 256,
                           [sy, legacy_y, C.c_int(rows), C.c_int(width), C.c_int(9), C.c_int(8), C.c_int(0)])
                old.launch(combine, grid, 128, [legacy_y, C.c_int(1), wt, C.c_void_p(bo[lo].data_ptr()),
                                               C.c_int64(width), C.c_int(9)])
            else:
                candidate.launch(fused, grid, 128, [ey, sy, wt, C.c_void_p(co[lo].data_ptr()),
                                                    C.c_int64(width), C.c_int(9)])

        first = None
        for repeat in range(5):
            bo.fill_(-123)
            co.fill_(-123)
            for mode in (("old", "fused") if repeat % 2 == 0 else ("fused", "old")):
                chain(mode)
            torch.cuda.synchronize()
            for guard in guards:
                guard.intact()
            assert same(bo, co), ("frozen/candidate bytes differ", rows, width, kind)
            assert bool((co[:lo] == -123).all() and (co[lo + rows:] == -123).all()), "subrange escaped"
            assert all(same(a, b) for a, b in zip(immutable, (gy, gs, gw))), "candidate input mutated"
            assert same(by[:, :8], gy[:, :8]), "baseline routed slots mutated"
            assert same(by[:lo], gy[:lo]) and same(by[lo + rows:], gy[lo + rows:]), "baseline range escaped"
            assert same(by[lo:lo + rows, 8], gs[lo:lo + rows]), "baseline put_slot wrong"
            if first is None:
                first = co.clone()
            assert same(first, co), "repeat bytes changed"
        values = y[lo:lo + rows].double().clone()
        values[:, 8] = shared[lo:lo + rows].double()
        terms = values * wts[lo:lo + rows, :, None].double()
        ref = terms.sum(1)
        got = co[lo:lo + rows].cpu().double()
        assert torch.equal(classification(got), classification(ref)), "FP64 classification mismatch"
        if kind == "stress":
            assert torch.equal(got, ref), "FP32 integer cancellation failed"
        elif kind == "random":
            bound = terms.abs().sum(1) * (9 * 2**-24 / (1 - 9 * 2**-24))
            assert bool(((got - ref).abs() <= bound).all()), "outside FP32 FMA rounding bound"
        co.fill_(-123)
        for row in range(lo, lo + rows):
            candidate.launch(fused, (1, grid[1], 1), 128,
                             [C.c_void_p(gy[row].data_ptr()), C.c_void_p(gs[row].data_ptr()),
                              C.c_void_p(gw[row].data_ptr()), C.c_void_p(co[row].data_ptr()),
                              C.c_int64(width), C.c_int(9)])
        torch.cuda.synchronize()
        assert same(first, co), "window rows differ from the same rows launched alone"
        report(kind="case", rows=rows, width=width, fixture=kind, repeats=5, bytes_equal=True,
               guards=True, inputs_immutable=True, fp64_reference=True, rows_match_solo=True, status="PASS")
        if args.bench and kind == "random" and width == 4096 and rows in (1, 2, 4, 8, 16, 32):
            for mode in ("old", "fused"):
                for _ in range(10):
                    chain(mode)
            torch.cuda.synchronize()
            graphs = {}
            for mode in ("old", "fused"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):
                        chain(mode)
                graphs[mode] = graph
            samples = {mode: [] for mode in graphs}
            for rep in range(6):
                for mode in (("old", "fused") if rep % 2 == 0 else ("fused", "old")):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(args.rounds):
                        graphs[mode].replay()
                    end.record()
                    end.synchronize()
                    samples[mode].append(start.elapsed_time(end) / (args.rounds * 20))
            assert same(bo, co)
            report(kind="graph_microbenchmark", rows=rows, samples_ms=samples,
                   median_ms={k: statistics.median(v) for k, v in samples.items()},
                   speedup=statistics.median(samples["old"]) / statistics.median(samples["fused"]))
    report(kind="summary", cases=len(cases), status="PASS", qualification="focused local only; no served proof")
    del anchor


if __name__ == "__main__":
    main()
