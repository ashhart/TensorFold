#!/usr/bin/env python3
"""Compare direct KDA stores with frozen-v5 fatbins; run only on an authorized idle GPU.

Both directories contain qmmf.fatbin, glm_mm.fatbin and glm_ops.fatbin. The test
uses the native packed formats and the served TP2 geometry, not torch matmul.
CUDA event microtimings omit host TP scheduling and use ordinary dependent
launches without PDL; they are not a served performance claim.
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


def check(code, operation):
    if code:
        raise RuntimeError(f"{operation}: CUDA error {code}")


def ptr(t):
    return C.c_void_p(t.data_ptr())


def ints(*values):
    return [C.c_int(v) for v in values]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Native:
    def __init__(self, directory):
        self.driver = C.CDLL("libcuda.so.1")
        self.driver.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
        self.driver.cuLaunchKernel.argtypes = [
            C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
            C.c_uint, C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.modules, self.functions, self.hashes = {}, {}, {}
        for name in ("qmmf", "glm_mm", "glm_ops"):
            path = directory / (name + ".fatbin")
            module = C.c_void_p()
            check(self.driver.cuModuleLoad(C.byref(module), str(path).encode()), str(path))
            self.modules[name] = module
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def launch(self, module, symbol, values, grid, threads=128, shared=0):
        key = module, symbol
        if key not in self.functions:
            function = C.c_void_p()
            check(self.driver.cuModuleGetFunction(C.byref(function), self.modules[module], symbol.encode()), symbol)
            if shared:
                check(self.driver.cuFuncSetAttribute(function, 8, shared), "dynamic shared memory")
            self.functions[key] = function
        arguments = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        check(self.driver.cuLaunchKernel(self.functions[key], *grid, threads, 1, 1, shared,
              C.c_void_p(torch.cuda.current_stream().cuda_stream), arguments, None), symbol)

    def qkv(self, x, w, scales, out, rows, n, k, ldy=None):
        bm = 16 if rows <= 16 else 32
        shared = 4 * ((bm * 128 + 4096 + 128 + 127) // 128 * 128)
        prefix = [ptr(x), ptr(w), ptr(scales), C.c_float(1), ptr(out)]
        if ldy is not None:
            symbol = f"qmmf_mxfp8_stride{bm}"
            values = prefix + ints(rows, n, k, (n + 127) // 128 * 128, x.stride(0), 1, ldy)
        else:
            symbol = (f"_ZN7tf_qmmf11qmmf_kernelILi2ELi{bm}ELi64ELi1ELi4ELi4ELb0ELb0ELb0"
                      "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii")
            values = prefix + [C.c_void_p(0)] + ints(rows, n, k, 1, (n + 127) // 128 * 128, x.stride(0), 1)
        self.launch("qmmf", symbol, values, ((n + 63) // 64, 1, 1), shared=shared)

    def rest(self, x, w, out, partial, rows, n, k, ldy=None):
        bm = 16 if rows <= 16 else 32
        sk = 1
        while sk < 8 and (n + 63) // 64 * sk < 192 and (k // 64) % (sk * 2) == 0 and k // 64 // (sk * 2) >= 4:
            sk *= 2
        assert sk > 1
        values = [ptr(x), C.c_longlong(x.stride(0)), ptr(w), C.c_void_p(0), ptr(out), ptr(partial)]
        values += ints(0, rows, n, k, sk)
        shared = 3 * (bm * 128 + 8 * 32 * 8 * 4 + 128)
        self.launch("glm_mm", f"glm_mm16_{bm}", values, ((n + 63) // 64, 1, sk), shared=shared)
        values = [ptr(partial), ptr(out), C.c_longlong(rows * n), C.c_int(sk)]
        symbol = "glm_mm_reduce" if ldy is None else "glm_mm_reduce_stride"
        values += ints(0) if ldy is None else ints(n, ldy)
        self.launch("glm_mm", symbol, values, ((rows * n + 255) // 256, 1, 1), threads=256)

    def copy(self, source, out, rows, n, ldy):
        values = [ptr(source), C.c_longlong(n * 2), ptr(out), C.c_longlong(ldy * 2)] + ints(rows, n * 2)
        self.launch("glm_ops", "tf_glm_copy_rows", values, ((rows * n * 2 + 255) // 256, 1, 1), threads=256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rows", default="1,2,3,4,5,6,7,8,15,16,17,31,32")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(72519)
    torch.cuda.init()
    anchor = torch.empty(1, device="cuda")
    base, candidate = Native(args.baseline), Native(args.candidate)
    report(kind="provenance", baseline=base.hashes, candidate=candidate.hashes,
           gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
           pdl=False, qualification="frozen-fatbin byte replacement and dependent microbench")
    q, n, k = 12288, 288, 4096
    width = q + n
    codes = torch.randint(0, 256, (q * k,), dtype=torch.uint8, device="cuda")
    codes.masked_fill_((codes & 127) == 127, 126)
    scales = torch.randint(119, 132, (q * k // 32,), dtype=torch.uint8, device="cuda")
    source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") / 32
    tiled = torch.empty_like(source)
    candidate.launch("glm_mm", "glm_mm_tile16", [ptr(source), ptr(tiled)] + ints(n, k),
                     (((n // 8) * (k // 64) * 32 + 255) // 256, 1, 1), threads=256)
    torch.cuda.synchronize()
    inputs = [codes, scales, source, tiled]
    input_hashes = [digest(t) for t in inputs]
    for rows in map(int, args.rows.split(",")):
        assert 1 <= rows <= 32
        x = torch.randn((rows, k + 16), dtype=torch.bfloat16, device="cuda")
        saved = x.clone()
        partial = torch.empty((8, rows, n), dtype=torch.float32, device="cuda")
        temporary = torch.empty((rows * q,), dtype=torch.bfloat16, device="cuda")
        storage = torch.full((rows * (width + 16) + 64,), -123, dtype=torch.bfloat16, device="cuda")
        out = storage[32:-32].view(rows, width + 16)
        y = out.flatten()

        def chain(engine, direct):
            if direct:
                engine.qkv(x, codes, scales, y, rows, q, k, out.stride(0))
                engine.rest(x, tiled, y[q:], partial, rows, n, k, out.stride(0))
            else:
                engine.qkv(x, codes, scales, temporary, rows, q, k)
                engine.copy(temporary, y, rows, q, out.stride(0))
                engine.rest(x, tiled, temporary, partial, rows, n, k)
                engine.copy(temporary, y[q:], rows, n, out.stride(0))

        for fixture in ("random", "zero", "extreme", "nonfinite"):
            x.copy_(saved)
            if fixture == "zero":
                x.zero_()
            elif fixture == "extreme":
                x[:, 0::2] *= 2 ** 30
                x[:, 1::2] *= 2 ** -30
            elif fixture == "nonfinite":
                x[0, 7] = float("nan")
                if rows > 1:
                    x[1, 13] = float("inf")
                if rows > 2:
                    x[2, 23] = -float("inf")
            before = digest(x)
            expected = None
            for repetition in range(5):
                for engine, direct in ((base, False), (candidate, False), (candidate, True)):
                    storage.fill_(-123)
                    chain(engine, direct)
                    torch.cuda.synchronize()
                    got = storage.view(torch.uint8).clone()
                    if expected is None:
                        expected = got
                    assert torch.equal(got, expected), f"Byte mismatch rows={rows} {fixture} direct={direct}"
                    assert bool((storage[:32] == -123).all() and (storage[-32:] == -123).all())
                    assert bool((out[:, width:] == -123).all()), "Output row padding changed"
                    if fixture == "zero":
                        assert torch.count_nonzero(out[:, :width]) == 0
            assert digest(x) == before, "Activation bytes changed"
            report(kind="correctness", rows=rows, fixture=fixture, five_repeats=True, status="PASS")
        x.copy_(saved)
        samples = {"baseline_v5": [], "candidate_off": [], "candidate_on": []}
        variants = [("baseline_v5", base, False), ("candidate_off", candidate, False), ("candidate_on", candidate, True)]
        for _, engine, direct in variants:
            for _ in range(10):
                chain(engine, direct)
        torch.cuda.synchronize()
        for rep in range(args.reps):
            for name, engine, direct in variants if rep % 2 == 0 else reversed(variants):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.rounds):
                    chain(engine, direct)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.rounds)
        report(kind="timing", rows=rows, samples_ms=samples,
               median_ms={name: statistics.median(values) for name, values in samples.items()})
    assert input_hashes == [digest(t) for t in inputs], "Weight bytes changed"
    report(kind="summary", status="PASS", input_bytes_unchanged=True,
           limits="Microbench only; actual runtime skips direct stores at one row and during prefill")
    del anchor


if __name__ == "__main__":
    main()
