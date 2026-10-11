#!/usr/bin/env python3
"""Byte oracle against the selected Triton RMS + SWA store kernels, guards and padding included."""
import argparse
import ctypes as C
import hashlib
import importlib
import os
from pathlib import Path
import sys


class Args(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ("x", "w", "cs", "v", "s", "pos", "sl", "norm_out")] + [
                (name, C.c_int64) for name in
                ("x_stride", "cs_stride", "v_stride", "s_stride")] + [
                ("inv_k", C.c_float), ("eps", C.c_float),
                ("ring", C.c_int32), ("rows", C.c_int32)]


assert C.sizeof(Args) == 112
assert [getattr(Args, name).offset for name, _ in Args._fields_] == [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 100, 104, 108]


class Driver:
    def __init__(self, fatbin, device=0):
        try:
            self.lib = C.CDLL("libcuda.so.1")
        except OSError as error:
            raise RuntimeError("CUDA driver unavailable; KV byte oracle requires a CUDA GPU") from error
        self.functions = {}
        signatures = {
            "cuInit": [C.c_uint],
            "cuDeviceGetCount": [C.POINTER(C.c_int)],
            "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
            "cuDevicePrimaryCtxRetain": [C.POINTER(C.c_void_p), C.c_int],
            "cuDevicePrimaryCtxRelease_v2": [C.c_int],
            "cuCtxSetCurrent": [C.c_void_p],
            "cuCtxGetCurrent": [C.POINTER(C.c_void_p)],
            "cuModuleLoad": [C.POINTER(C.c_void_p), C.c_char_p],
            "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
            "cuLaunchKernel": [C.c_void_p] + [C.c_uint] * 7 +
                              [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
            "cuModuleUnload": [C.c_void_p],
            "cuStreamBeginCapture": [C.c_void_p, C.c_int],
            "cuStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "cuGraphGetNodes": [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
            "cuGraphInstantiateWithFlags": [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            "cuGraphLaunch": [C.c_void_p, C.c_void_p],
            "cuGraphExecDestroy": [C.c_void_p],
            "cuGraphDestroy": [C.c_void_p],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = signature, C.c_int
            self.functions[name] = function
        self.device, self.context = C.c_int(), C.c_void_p()
        self.module, self.kernel = C.c_void_p(), C.c_void_p()
        self.call("cuInit", 0)
        count = C.c_int()
        self.call("cuDeviceGetCount", C.byref(count))
        if count.value == 0:
            raise RuntimeError("No CUDA device present; KV byte oracle requires a CUDA GPU")
        if not 0 <= device < count.value:
            raise RuntimeError(f"CUDA device ordinal {device} unavailable ({count.value} devices)")
        self.call("cuDeviceGet", C.byref(self.device), device)
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.context), self.device)
        try:
            self.call("cuCtxSetCurrent", self.context)
            self.call("cuModuleLoad", C.byref(self.module), os.fsencode(fatbin))
            self.call("cuModuleGetFunction", C.byref(self.kernel), self.module, b"kv_norm_store")
        except BaseException:
            self.close()
            raise

    def call(self, name, *args):
        status = self.functions[name](*args)
        if status == 100:  # CUDA_ERROR_NO_DEVICE, including cuInit on a GPU-free host
            raise RuntimeError(f"{name}: no CUDA device present; KV byte oracle requires a CUDA GPU")
        if status:
            raise RuntimeError(f"{name}: CUDA driver error {status}")

    def assert_current(self):
        current = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(current))
        if current.value != self.context.value:
            raise RuntimeError("CUDA context changed: Torch reference and fused kernel must use "
                               "the selected device's same primary context")

    def close(self):
        try:
            if self.module.value:
                self.call("cuCtxSetCurrent", self.context)
                self.call("cuModuleUnload", self.module)
                self.module = C.c_void_p()
        finally:
            if self.context.value:
                self.call("cuDevicePrimaryCtxRelease_v2", self.device)
                self.context = C.c_void_p()

    def launch(self, arguments, rows, stream, block):
        self.assert_current()
        parameters = (C.c_void_p * 1)(C.addressof(arguments))
        self.call("cuLaunchKernel", self.kernel, rows, 1, 1, block, 1, 1,
                  0, C.c_void_p(stream.cuda_stream), parameters, None)

    def capture(self, function, stream, expected_nodes):
        self.assert_current()
        handle = C.c_void_p(stream.cuda_stream)
        self.call("cuStreamBeginCapture", handle, 0)
        graph = C.c_void_p()
        try:
            function()
        finally:
            self.call("cuStreamEndCapture", handle, C.byref(graph))
        executable = C.c_void_p()
        try:
            count = C.c_size_t()
            self.call("cuGraphGetNodes", graph, None, C.byref(count))
            if count.value != expected_nodes:
                raise AssertionError(f"expected {expected_nodes} graph nodes, got {count.value}")
            self.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, 0)
        finally:
            self.call("cuGraphDestroy", graph)
        return executable


def bytes_equal(torch, actual, expected, label):
    a, b = actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    different = (a != b).flatten().nonzero().flatten()
    if different.numel():
        offsets = different[:8].cpu().tolist()
        av, bv = a.flatten(), b.flatten()
        detail = [(i, int(av[i]), int(bv[i])) for i in offsets]
        raise AssertionError(f"{label}: {different.numel()} differing bytes; (offset, fused, baseline)={detail}")


def fixture(torch, rows, mode, pattern, strided):
    ring, slots = 256, 3 if mode != "scalar" else 1
    xs, cs_stride = (544, 80) if strided else (512, 64)
    vs, ss = (608, 24) if strided else (576, 8)
    guard, records = 64, ring * slots
    # Storage offsets exercise nonzero pointers; full backing storage is guarded.
    xbase = torch.full((64 + rows * xs,), 19., dtype=torch.bfloat16, device="cuda")
    x = xbase.as_strided((rows, 512), (xs, 1), 32)
    columns = torch.arange(512, dtype=torch.float64)
    r = torch.arange(rows, dtype=torch.float64)[:, None]
    values = torch.sin((r + 1) * (columns + 1) * .03125) * 2
    weight = torch.ones(512, dtype=torch.float32)
    if pattern == "zeros":
        values.zero_()
    elif pattern == "subnormal":
        values = ((columns.to(torch.int64) % 7 - 3).double() * 2.**-133).expand(rows, -1).clone()
    elif pattern == "large_finite":
        values = ((columns.to(torch.int64) % 5 - 2).double() * 2.**120).expand(rows, -1).clone()
    elif pattern == "mixed_exponents":
        values = torch.pow(2., columns % 120 - 60)[None, :].expand(rows, -1).clone()
        values[:, 1::2].neg_()
    elif pattern == "signed_zero":
        values.zero_()
        values[:, 1::2] = -0.0
        weight[::3] = -1
    elif pattern == "weight_signs":
        weight = (torch.cos(columns * .17) * 3).float()
        weight[::11] = 0
    elif pattern == "input_nan":
        values[:, ::37] = float("nan")
    elif pattern == "input_inf":
        values[:, ::37] = float("inf")
        values[:, 1::37] = -float("inf")
    elif pattern == "weight_nan":
        weight[::37] = float("nan")
    elif pattern == "weight_inf":
        weight[::37] = float("inf")
        weight[1::37] = -float("inf")
    elif pattern == "reduction_stress":
        # Exact bf16 powers span far beyond fp64's mantissa; placement spans per-thread lanes and warp boundaries.
        values.fill_(1)
        values[:, ::127] = 2.**30
        values[:, 5::31] = 2.**-40
        values[:, 1::17] = -2.**10
    elif pattern == "bf16_ties":
        values.fill_(1)
        ties = torch.tensor([1 + 1 / 256, 1 + 3 / 256, 2 + 1 / 128,
                             -1 - 1 / 256, -1 - 3 / 256, 2.**-134,
                             3 * 2.**-134], dtype=torch.float64)
        # x=1 yields rn=1 after adding 1e-20, exposing exact fp32 midpoints in the fp64->fp32->bf16 normalization.
        weight = ties.repeat((512 + len(ties) - 1) // len(ties))[:512].float().clone()
    elif pattern == "fp8_ties":
        values.fill_(1)
        ties = torch.tensor([0., -0., .0009765625, .0029296875, .0146484375,
                             1.0625, 1.1875, 2.125, 3.875, 15.5, 31., 62., 124., 248.])
        weight = ties.repeat((512 + len(ties) - 1) // len(ties))[:512].clone()
        weight[1::2].neg_()
        # Set tile maxima to448, keeping the NoPE quantization exponent at0.
        weight[::64] = 448
    elif pattern not in ("random", "rope_quadrants", "rope_identity", "eps_1e5"):
        raise ValueError(pattern)
    x.copy_(values.to(torch.bfloat16).cuda())
    weight = weight.cuda()
    if mode == "scalar":
        position = torch.tensor([ring - 3], dtype=torch.int32, device="cuda")
        sl = None
        max_position = ring - 3 + rows
    else:
        position = (torch.arange(rows, dtype=torch.int64) * 13 + 7).cuda()
        sl = (torch.arange(rows, dtype=torch.int64) % slots).cuda()
        sl[::5] = -1
        if mode == "all_padding":
            sl.fill_(-1)
        max_position = int(position.max()) + 1
    csbase = torch.full((64 + (max_position + 1) * cs_stride,), 29., dtype=torch.float32, device="cuda")
    cs = csbase.as_strided((max_position + 1, 64), (cs_stride, 1), 16)
    q = torch.arange(max_position + 1, dtype=torch.float64)[:, None]
    frequencies = torch.arange(32, dtype=torch.float64)[None, :]
    angle = q * (frequencies + 1) * .037
    if pattern == "rope_quadrants":
        angle = ((q + frequencies) % 8) * (torch.pi / 2)
    elif pattern == "rope_identity":
        angle.zero_()
    cs.copy_(torch.cat((angle.cos(), angle.sin()), dim=1).float().cuda())

    def cache(width, stride):
        base = torch.full((guard * 2 + records * stride,), 0xA5, dtype=torch.uint8, device="cuda")
        return base, base.as_strided((records, width), (stride, 1), guard)

    vb, v = cache(576, vs)
    sb, s = cache(8, ss)
    fvbase, fsbase = vb.clone(), sb.clone()
    fv = fvbase.as_strided(v.shape, v.stride(), guard)
    fs = fsbase.as_strided(s.shape, s.stride(), guard)
    nb = torch.full((64 + rows * 512,), 31., dtype=torch.bfloat16, device="cuda")
    fnbase = nb.clone()
    norm = nb.as_strided((rows, 512), (512, 1), 32)
    fnorm = fnbase.as_strided(norm.shape, norm.stride(), 32)
    return dict(x=x, w=weight, cs=cs, pos=position, sl=sl, ring=ring,
                eps=1e-5 if pattern == "eps_1e5" else 1e-20,
                v=v, s=s, fv=fv, fs=fs, vb=vb, sb=sb, fvb=fvbase, fsb=fsbase,
                norm=norm, fnorm=fnorm, nb=nb, fnb=fnbase,
                inputs=[xbase, weight, csbase, position] + ([] if sl is None else [sl]))


def baseline(f, rms, store, rows):
    rms._rms[(rows,)](f["x"], f["x"].stride(0), f["w"], f["norm"], 512,
                      512, 1 / 512, f["eps"], BK=512, HAS_W=True, NARROW=True,
                      PDL=False, num_warps=4, **rms.LAUNCH)
    store._kv_store[(rows,)](f["norm"], 512, f["cs"], f["cs"].stride(0),
                            f["v"], f["s"], f["v"].stride(0), f["s"].stride(0),
                            f["pos"], RATIO=0, RING=f["ring"], PT=None, PSH=0,
                            SL=f["sl"], PTS=0, ROWS=f["sl"] is not None,
                            PDL=False, num_warps=4)


def arguments(f, debug):
    ptr = lambda value: 0 if value is None else value.data_ptr()
    return Args(*(ptr(f[key]) for key in ("x", "w", "cs", "fv", "fs", "pos", "sl")),
                ptr(f["fnorm"]) if debug else 0, f["x"].stride(0), f["cs"].stride(0),
                f["fv"].stride(0), f["fs"].stride(0), 1 / 512, f["eps"],
                f["ring"], f["sl"] is not None)


def event_us(torch, function, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fatbin", required=True)
    parser.add_argument("--python-tree", default=os.environ.get("TF_DSV41_PYTHON_TREE"))
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--block", type=int, choices=(128,), default=128)
    args = parser.parse_args()
    if not args.python_tree:
        parser.error("--python-tree or TF_DSV41_PYTHON_TREE must select the pinned reference source")
    if args.reps < 1:
        parser.error("--reps must be positive")
    tree = Path(args.python_tree).resolve()
    sys.path.insert(0, str(tree / "src" if (tree / "src").is_dir() else tree))
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; run in the pinned reference Python environment")
    rms = importlib.import_module("tensorfold.families.deepseek_v41.cuda.rmsnorm")
    store = importlib.import_module("tensorfold.families.deepseek_v41.cuda.csa2.compress")
    for module in (rms, store):
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(tree):
            raise RuntimeError(f"baseline imported outside selected tree: {source}")
        print("BASELINE", source, "sha256", hashlib.sha256(source.read_bytes()).hexdigest())
    print("DEVICE", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
    torch.cuda.init()
    device = torch.cuda.current_device()
    driver = Driver(Path(args.fatbin).resolve(), device)
    patterns = ("random", "zeros", "subnormal", "large_finite", "mixed_exponents", "signed_zero",
                "weight_signs", "fp8_ties", "rope_quadrants", "rope_identity", "eps_1e5",
                "input_nan", "input_inf", "weight_nan", "weight_inf", "reduction_stress", "bf16_ties")
    tested = 0
    try:
        stream = torch.cuda.Stream(device=device)
        driver.assert_current()  # Torch stream creation must retain the same primary context.
        with torch.cuda.stream(stream):
            for rows in (1, 2, 3, 4, 5, 6, 9, 12, 15, 16, 20, 24, 32, 48, 64):
                for mode in ("scalar", "rows", "all_padding"):
                    for strided in (False, True):
                        for pattern in patterns:
                            f = fixture(torch, rows, mode, pattern, strided)
                            snapshots = [value.clone() for value in f["inputs"]]
                            call = arguments(f, True)
                            label = f"R={rows} mode={mode} strided={strided} pattern={pattern}"
                            for repetition in range(3):
                                baseline(f, rms, store, rows)
                                driver.launch(call, rows, stream, args.block)
                                stream.synchronize()
                                for actual, expected, field in ((f["fvb"], f["vb"], "values+guards"),
                                                               (f["fsb"], f["sb"], "scales+guards"),
                                                               (f["fnb"], f["nb"], "norm+guards")):
                                    bytes_equal(torch, actual, expected, f"{label} rep={repetition} {field}")
                            # Optional debug output must not affect the actual KV output.
                            driver.launch(arguments(f, False), rows, stream, args.block)
                            stream.synchronize()
                            bytes_equal(torch, f["fvb"], f["vb"], label + " no-debug values")
                            bytes_equal(torch, f["fsb"], f["sb"], label + " no-debug scales")
                            for actual, expected in zip(f["inputs"], snapshots):
                                bytes_equal(torch, actual, expected, label + " immutable input")
                            tested += 1
                print("BYTE_CASES_PASS", "R", rows, "cumulative", tested, flush=True)
            print("TIMING_US R mode baseline_eager fused_eager baseline_graph fused_graph graph_nodes")
            # Captured DSpark ingest uses scalar positions at R1-6/9; attention/backbone row tables use R12-48.
            for rows, mode in ([(r, "scalar") for r in (1, 2, 3, 4, 5, 6, 9)] +
                               [(r, "rows") for r in (12, 15, 16, 20, 24, 32, 48, 64)]):
                f = fixture(torch, rows, mode, "random", False)
                call = arguments(f, False)
                unfused = lambda: baseline(f, rms, store, rows)
                fused = lambda: driver.launch(call, rows, stream, args.block)
                for _ in range(10):
                    unfused()
                    fused()
                stream.synchronize()
                old = driver.capture(unfused, stream, 2)
                new = None
                try:
                    new = driver.capture(fused, stream, 1)
                    timings = [event_us(torch, function, args.reps) for function in
                               (unfused, fused,
                                lambda: driver.call("cuGraphLaunch", old, C.c_void_p(stream.cuda_stream)),
                                lambda: driver.call("cuGraphLaunch", new, C.c_void_p(stream.cuda_stream)))]
                    print("TIMING_US", rows, mode, *(f"{value:.6f}" for value in timings), "2/1")
                finally:
                    stream.synchronize()
                    driver.call("cuGraphExecDestroy", old)
                    if new is not None:
                        driver.call("cuGraphExecDestroy", new)
            print("PASS", tested, "byte cases; timings are GPU event durations, not throughput forecasts")
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            driver.close()


if __name__ == "__main__":
    main()
