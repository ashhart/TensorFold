#!/usr/bin/env python3
"""Same-bit KDA MXFP8 projection qualification against an immutable v5 fatbin.

Run only on an authorized idle GPU. This script does not manage services or
compile kernels. Positional paths are qmmf fatbins (or directories containing
qmmf.fatbin). Candidate symbols follow KDA_PROJECTION_V9.md. No torch matmul is
used as the performance baseline; the actual v5 kernel and SK4 cluster reduction
are retained. An independent FP64 reference checks sampled output columns.

Examples:
  python bench_kda_projection_v9.py --self-test
  python bench_kda_projection_v9.py v5/qmmf.fatbin v9/qmmf.fatbin --mode correctness
  python bench_kda_projection_v9.py v5/qmmf.fatbin v9/qmmf.fatbin --variants s2w4,s3w8
  python bench_kda_projection_v9.py v5/qmmf.fatbin v9/qmmf.fatbin --include-v5-split --projections o

JSONL results include raw ABBA/BAAB samples, A/A controls and both fatbin hashes.
Warm timings replay a graph containing --iters identical sequential projections;
cold timings replay one projection after every cache eviction, with the eviction
outside the measured event interval. Both timing events are nodes INSIDE the
captured graph, excluding Python submission gaps even for a single projection.
Neither is an end-to-end speed claim.
"""

import argparse
import ctypes as C
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace


class Dim3(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("z", C.c_uint)]


class LaunchValue(C.Union):
    _fields_ = [("pad", C.c_ubyte * 64), ("align8", C.c_void_p),
                ("integer", C.c_int), ("cluster_dim", Dim3)]


class LaunchAttribute(C.Structure):
    _fields_ = [("id", C.c_uint), ("pad", C.c_uint), ("value", LaunchValue)]


class LaunchConfig(C.Structure):
    _fields_ = [("grid_x", C.c_uint), ("grid_y", C.c_uint), ("grid_z", C.c_uint),
                ("block_x", C.c_uint), ("block_y", C.c_uint), ("block_z", C.c_uint),
                ("shared_bytes", C.c_uint), ("stream", C.c_void_p),
                ("attrs", C.POINTER(LaunchAttribute)), ("num_attrs", C.c_uint)]


VARIANTS = {f"s{s}w{w}": (s, w) for s in (2, 3, 4) for w in (4, 8)}
SPLIT_VARIANTS = {"baseline_v5_split", "candidate_build_split"}
PROJECTIONS = {"qkv": {"n": 12288, "sk": 1, "wide": False},
               "o": {"n": 4096, "sk": 4, "wide": True}}
K = 4096
GUARD = 64
SENTINEL = -123.0
torch = None  # CPU-only ABI/layout self-test and --help do not import torch.


def report(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def kin(lane, byte):
    """fp8.zig kin(): K32 fragment swizzle, not a row-major approximation."""
    p = 16 * (byte // 4) + 4 * (lane % 4) + byte % 4
    return 16 * (p // 16) + 2 * ((p % 16) // 4) + p % 2 + 8 * ((p % 4) // 2)


def code_tile_order():
    return [(a * 8 + lane // 4) * 64 + h * 32 + kin(lane, byte)
            for a in range(8) for lane in range(32) for h in range(2) for byte in range(8)]


def cpu_self_test():
    """Standard-library-only ABI and independent coordinate checks."""
    assert C.sizeof(LaunchValue) == 64 and C.alignment(LaunchValue) == 8
    assert C.sizeof(LaunchAttribute) == 72 and LaunchAttribute.value.offset == 8
    assert C.sizeof(LaunchConfig) == 56 and LaunchConfig.stream.offset == 32
    assert LaunchConfig.attrs.offset == 40 and LaunchConfig.num_attrs.offset == 48
    order = code_tile_order()
    assert sorted(order) == list(range(4096)), "Packed code tile is not a permutation"
    assert sorted(kin(lane, byte) for lane in range(4) for byte in range(8)) == list(range(32))
    idx = (((3 * 32 + 9) * 2 + 1) * 8 + 5)
    assert order[idx] == 26 * 64 + 32 + kin(9, 5)
    # Tiny independent round trip of the full multi-tile code and scale layouts.
    n, k = 130, 128
    npad, kg = 256, k // 64
    packed = []
    for tile in range(npad // 64):
        for group in range(kg):
            for at in order:
                row, col = tile * 64 + at // 64, group * 64 + at % 64
                packed.append(row * k + col if row < n else -1)
    assert len(packed) == npad * k
    assert sorted(v for v in packed if v >= 0) == list(range(n * k))
    scales = [(tile * 64 + row) * (k // 32) + group * 2 + half
              if tile * 64 + row < n else -1
              for tile in range(npad // 64) for group in range(kg)
              for row in range(64) for half in range(2)]
    assert sorted(v for v in scales if v >= 0) == list(range(n * k // 32))
    cpu_split_launch_test()
    report(kind="cpu_self_test", status="PASS", launch_attribute_bytes=C.sizeof(LaunchAttribute),
           launch_config_bytes=C.sizeof(LaunchConfig), stream_offset=LaunchConfig.stream.offset,
           packed_tile_permutation=True, padded_layout_round_trip=True,
           split_launch_abi_mock=True)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def pack_codes(codes):
    """Exactly fp8.packCodes: [tile, K64, a, lane, half, byte]."""
    n, k = codes.shape
    npad = (n + 127) // 128 * 128
    padded = torch.zeros((npad, k), dtype=torch.uint8)
    padded[:n].copy_(codes)
    tiles = padded.view(npad // 64, 64, k // 64, 64).permute(0, 2, 1, 3).contiguous()
    order = torch.tensor(code_tile_order(), dtype=torch.int64)
    return tiles.view(npad // 64, k // 64, 4096).index_select(2, order).contiguous().flatten()


def pack_scales(scales):
    """Exactly mxfp8.packScales: [tile, K64, row, K32 half]."""
    n, groups = scales.shape
    npad = (n + 127) // 128 * 128
    padded = torch.full((npad, groups), 127, dtype=torch.uint8)
    padded[:n].copy_(scales)
    return padded.view(npad // 64, 64, groups // 2, 2).permute(0, 2, 1, 3).contiguous().flatten()


def validate_tensor_packing(codes, scales, packed_codes, packed_scales):
    """Coordinate check uses scalar formulas independently of tensor reshape."""
    n, k = codes.shape
    for tile, group, a, lane, half, byte in (
            (0, 0, 0, 0, 0, 0), (0, 1, 3, 9, 1, 5),
            (n // 64 - 1, k // 64 - 1, 7, 31, 1, 7)):
        at = (((((tile * (k // 64) + group) * 8 + a) * 32 + lane) * 2 + half) * 8 + byte)
        row, col = tile * 64 + a * 8 + lane // 4, group * 64 + half * 32 + kin(lane, byte)
        assert int(packed_codes[at]) == int(codes[row, col])
        scale_at = ((tile * (k // 64) + group) * 64 + (row % 64)) * 2 + half
        assert int(packed_scales[scale_at]) == int(scales[row, group * 2 + half])


def symbol_baseline(projection, cluster=None):
    wide = int(PROJECTIONS[projection]["wide"])
    cluster = int(PROJECTIONS[projection]["sk"] > 1 if cluster is None else cluster)
    return (f"_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb{wide}ELb{cluster}ELb0"
            "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii")


def shared_bytes(stages, warps):
    stage = (16 * 64 * 2 + 64 * 64 + 64 * 2 + 127) // 128 * 128
    partials = 1 * (64 // warps // 8) * 4 * (warps * 32) * 4
    return max(stages * stage, partials)


class Native:
    def __init__(self, path, name):
        self.path = path / "qmmf.fatbin" if path.is_dir() else path
        self.name = name
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.driver = C.CDLL("libcuda.so.1")
        d = self.driver
        d.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        d.cuModuleUnload.argtypes = [C.c_void_p]
        d.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        d.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
        d.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        d.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [
            C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        d.cuLaunchKernelEx.argtypes = [C.POINTER(LaunchConfig), C.c_void_p,
                                      C.POINTER(C.c_void_p), C.c_void_p]
        d.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        self.module = C.c_void_p()
        self.check(d.cuModuleLoad(C.byref(self.module), str(self.path).encode()), str(self.path))
        self.functions = {}

    def check(self, status, operation):
        if status:
            name = C.c_char_p()
            self.driver.cuGetErrorName(status, C.byref(name))
            raise RuntimeError(f"{operation}: CUDA {status} {name.value!r}")

    def function(self, symbol, shared):
        if symbol not in self.functions:
            function = C.c_void_p()
            self.check(self.driver.cuModuleGetFunction(C.byref(function), self.module,
                                                     symbol.encode()), symbol)
            if shared:
                self.check(self.driver.cuFuncSetAttribute(function, 8, shared), "dynamic shared memory")
            attributes = {}
            for attr, label in enumerate(("max_threads", "static_shared_bytes", "const_bytes",
                                           "local_bytes", "registers", "ptx_version", "binary_version")):
                value = C.c_int()
                self.check(self.driver.cuFuncGetAttribute(C.byref(value), attr, function), label)
                attributes[label] = value.value
            report(kind="kernel_resources", module=self.name, symbol=symbol,
                   dynamic_shared_bytes=shared, **attributes)
            self.functions[symbol] = function
        return self.functions[symbol]

    def projection(self, variant, projection, x, codes, scales, out, rows, partial=None):
        config = PROJECTIONS[projection]
        n, sk = config["n"], config["sk"]
        split = variant == "baseline_v5_split"
        is_v5 = variant in ("baseline_v5", "baseline_v5_split")
        if split:
            assert projection == "o" and sk == 4 and partial is not None
            assert partial.dtype == torch.float32 and partial.numel() == sk * rows * n
        stages, warps = (4, 4) if is_v5 else VARIANTS[variant]
        symbol = (symbol_baseline(projection, cluster=False if split else None) if is_v5 else
                  f"tf_kda_mxfp8_s{stages}_w{warps}_{projection}")
        shared = shared_bytes(stages, warps)
        function = self.function(symbol, shared)
        # Match qmmf.zig exactly, including row-1 ldx and native band=1.
        values = [ptr(x), ptr(codes), ptr(scales), C.c_float(1), ptr(out),
                  ptr(partial) if split else C.c_void_p(0)]
        values += [C.c_int(v) for v in (rows, n, K, sk, (n + 127) // 128 * 128,
                                       K if rows == 1 else x.stride(0), 1)]
        arguments = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        if sk == 1 or split:
            status = self.driver.cuLaunchKernel(function, n // 64, 1, sk, warps * 32, 1, 1,
                                                shared, stream, arguments, None)
        else:
            attribute = LaunchAttribute()
            attribute.id = 4  # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION.
            attribute.value.cluster_dim = Dim3(1, 1, sk)
            launch = LaunchConfig(n // 64, 1, sk, warps * 32, 1, 1, shared,
                                  stream, C.pointer(attribute), 1)
            status = self.driver.cuLaunchKernelEx(C.byref(launch), function, arguments, None)
        self.check(status, symbol)
        if split:
            # Same immutable-v5 slices and FP32 0+1+2+3 ordering, but store
            # partials to global memory instead of exchanging them in a cluster.
            reduce_symbol = "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif"
            reduce_function = self.function(reduce_symbol, 0)
            reduce_values = [ptr(partial), ptr(out), C.c_longlong(rows * n), C.c_int(sk), C.c_float(1)]
            reduce_arguments = (C.c_void_p * len(reduce_values))(
                *(C.cast(C.byref(v), C.c_void_p) for v in reduce_values))
            self.check(self.driver.cuLaunchKernel(reduce_function, (rows * n + 255) // 256,
                                                 1, 1, 256, 1, 1, 0, stream,
                                                 reduce_arguments, None), reduce_symbol)

    def close(self):
        if self.module:
            self.check(self.driver.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = C.c_void_p()


def cpu_split_launch_test():
    """No CUDA/torch import: verify both builds use the ordinary SK4 + reducer ABI."""
    global torch
    previous = torch
    torch = SimpleNamespace(float32="f32", cuda=SimpleNamespace(
        current_stream=lambda: SimpleNamespace(cuda_stream=123)))

    class Tensor:
        dtype = "f32"

        def __init__(self, address, count=0):
            self.address, self.count = address, count

        def data_ptr(self):
            return self.address

        def stride(self, axis):
            return K + 16

        def numel(self):
            return self.count

    class Driver:
        def __init__(self):
            self.calls = []

        def cuLaunchKernel(self, *args):
            values = args[9]
            if not self.calls:
                decoded = [C.cast(values[i], C.POINTER(C.c_int)).contents.value for i in range(6, 13)]
            else:
                decoded = [C.cast(values[2], C.POINTER(C.c_longlong)).contents.value,
                           C.cast(values[3], C.POINTER(C.c_int)).contents.value,
                           C.cast(values[4], C.POINTER(C.c_float)).contents.value]
            self.calls.append((args[1:8], decoded))
            return 0

    try:
        for external_name in sorted(SPLIT_VARIANTS):
            for rows in (1, 8):
                engine = Native.__new__(Native)
                engine.name, engine.driver = external_name, Driver()
                symbols = []

                def function(symbol, shared):
                    symbols.append((symbol, shared))
                    return C.c_void_p(1)

                engine.function = function
                engine.projection("baseline_v5_split", "o", Tensor(1), Tensor(2), Tensor(3),
                                  Tensor(4), rows, Tensor(5, 4 * rows * K))
                assert engine.driver.calls == [
                    ((64, 1, 4, 128, 1, 1, 25088), [rows, K, K, 4, K, K if rows == 1 else K + 16, 1]),
                    ((rows * K // 256, 1, 1, 256, 1, 1, 0), [rows * K, 4, 1.0])]
                assert "ELb1ELb0ELb0" in symbols[0][0]
                assert symbols[1][0] == "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif"
    finally:
        torch = previous


def reference_check(codes, scales, x, out, wide, columns):
    """Independent E4M3/UE8M0 decode and FP64 dot; samples only N, never K."""
    n = codes.shape[0]
    chosen = sorted(set([0, 1, 7, 8, 31, 63, 64, n // 2, n - 1] +
                        [i * (n - 1) // max(1, columns - 1) for i in range(columns)]))
    c = codes[chosen].to(torch.int32)
    exponent, fraction = (c >> 3) & 15, c & 7
    exponent_f = exponent.to(torch.float64)
    weight = torch.where(exponent == 0, fraction.to(torch.float64) * (2.0 ** -9),
                         (1 + fraction.to(torch.float64) / 8) * torch.pow(2.0, exponent_f - 7))
    weight *= torch.where((c & 128) != 0, -1.0, 1.0)
    s = torch.pow(2.0, scales[chosen].to(torch.float64) - 127).repeat_interleave(32, dim=1)
    weight *= s
    activation = x.detach().cpu().to(torch.float64)
    want = activation @ weight.T
    magnitude = activation.abs() @ weight.abs().T
    got = out[:, chosen].detach().cpu().to(torch.float64)
    tolerance = magnitude * 2e-6 + want.abs() * (1e-6 + (0 if wide else 1 / 256)) + 1e-30
    difference = (got - want).abs()
    assert bool(torch.isfinite(got).all()), "Finite fixture produced nonfinite values"
    assert bool((difference <= tolerance).all()), "Independent FP64 reference check failed"
    return {"columns": len(chosen), "full_k": K,
            "max_abs_error": float(difference.max()),
            "max_tolerance_fraction": float((difference / tolerance).max())}


def apply_fixture(x, saved, fixture):
    x.copy_(saved)
    if fixture == "zero":
        x.zero_()
    elif fixture == "extreme":
        x[:, 0::2] *= 2 ** 30
        x[:, 1::2] *= 2 ** -30
    elif fixture == "nan":
        x[0, 7] = float("nan")
    elif fixture == "posinf":
        x[0, 13] = float("inf")
    elif fixture == "neginf":
        x[0, 23] = -float("inf")
    elif fixture == "nonfinite":
        x[0, 7] = float("nan")
        x[min(1, x.shape[0] - 1), 13] = float("inf")
        x[min(2, x.shape[0] - 1), 23] = -float("inf")


def correctness(launches, projection, rows, x, saved, x_storage, storage, out,
                codes_cpu, scales_cpu, reference_columns, partial_storage=None):
    for fixture in ("finite", "zero", "extreme", "nan", "posinf", "neginf", "nonfinite"):
        apply_fixture(x, saved, fixture)
        before = digest(x_storage)
        expected = None
        ref = None
        for repetition in range(5):
            for name, call in launches.items():
                storage.fill_(SENTINEL)
                if partial_storage is not None:
                    partial_storage.fill_(SENTINEL)
                    partial_storage[GUARD:-GUARD].fill_(float("nan"))
                call()
                torch.cuda.synchronize()
                got = storage.view(torch.uint8).clone()
                if expected is None:
                    expected = got
                    if fixture in ("finite", "extreme"):
                        ref = reference_check(codes_cpu, scales_cpu, x, out,
                                              PROJECTIONS[projection]["wide"], reference_columns)
                assert torch.equal(got, expected), (
                    f"Byte mismatch: {projection} rows={rows} fixture={fixture} "
                    f"variant={name} repetition={repetition}")
                assert bool((storage[:GUARD] == SENTINEL).all() and
                            (storage[-GUARD:] == SENTINEL).all()), "Output guard changed"
                if partial_storage is not None:
                    assert bool((partial_storage[:GUARD] == SENTINEL).all() and
                                (partial_storage[-GUARD:] == SENTINEL).all()), "Partial guard changed"
                    if name in SPLIT_VARIANTS and fixture in ("finite", "zero", "extreme"):
                        assert bool(torch.isfinite(partial_storage[GUARD:-GUARD]).all()), (
                            "Split projection left poisoned or nonfinite FP32 partials")
                if fixture == "zero":
                    assert int(torch.count_nonzero(out)) == 0, "Zero input produced nonzero output"
                if fixture in ("finite", "zero", "extreme"):
                    assert bool(torch.isfinite(out).all()), "Finite fixture produced NaN/Inf"
        assert digest(x_storage) == before, "Activation or activation guards changed"
        report(kind="correctness", projection=projection, rows=rows, fixture=fixture,
               variants=list(launches), repeats=5, status="PASS", byte_equal=True,
               output_guards=True, partial_guards=partial_storage is not None,
               immutable_input=True, independent_reference=ref)
    x.copy_(saved)


class TimedReplay:
    """Own a captured event/kernel/event sequence and its per-kernel divisor."""
    def __init__(self, graph, start, end, count):
        self.graph, self.start, self.end, self.count = graph, start, end, count

    def replay(self):
        self.graph.replay()

    def elapsed_ms(self):
        self.end.synchronize()
        return self.start.elapsed_time(self.end) / self.count


def build_replays(launches, iterations):
    """Capture after symbol resolution/warmup; no allocations in kernel bodies."""
    replays, graphs = {}, []
    capture_stream = torch.cuda.Stream()
    for name, call in launches.items():
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        per_mode = {}
        for mode, count in (("warm", iterations), ("cold", 1)):
            graph = torch.cuda.CUDAGraph()
            # external=True creates actual graph event-record nodes instead of
            # capture-internal dependency markers. Their timing span contains
            # only GPU-scheduled graph work, not Python's replay submission.
            start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            with torch.cuda.graph(graph, stream=capture_stream):
                start.record()
                for _ in range(count):
                    call()
                end.record()
            graph.replay()
            end.synchronize()
            graphs.append(graph)
            per_mode[mode] = TimedReplay(graph, start, end, count)
        replays[name] = per_mode
    return replays, graphs


def measured_block(replay, mode, eviction, iterations, cold_iterations):
    assert replay.count == (iterations if mode == "warm" else 1)
    if mode == "warm":
        replay.replay()
    samples = []
    for _ in range(cold_iterations if mode == "cold" else 1):
        if mode == "cold":
            # Same stream: eviction finishes before the graph's start node.
            # Neither the eviction nor the host's graph submission is timed.
            eviction.add_(1)
        replay.replay()
        # Read this invocation before the same graph event handles are reused.
        samples.append(replay.elapsed_ms())
    return samples


def paired_benchmark(replays, variant, mode, args, eviction):
    """ABBA and BAAB pairs; controls use the very same baseline graph twice."""
    target = "baseline_v5" if variant == "AA_control" else variant
    raw, speedups = [], []
    for pair in range(args.pairs):
        order = ("A", "B", "B", "A") if pair % 2 == 0 else ("B", "A", "A", "B")
        blocks, grouped = [], {"A": [], "B": []}
        for label in order:
            name = "baseline_v5" if label == "A" else target
            samples = measured_block(replays[name][mode], mode, eviction, args.iters, args.cold_iters)
            median = statistics.median(samples)
            grouped[label].append(median)
            blocks.append({"label": label, "kernel": name, "samples_ms": samples, "median_ms": median})
        a, b = (statistics.mean(grouped[label]) for label in ("A", "B"))
        assert a > 0 and b > 0, "Invalid CUDA event interval"
        speedups.append(a / b)
        raw.append({"pair": pair, "order": "".join(order), "blocks": blocks,
                    "baseline_ms": a, "candidate_ms": b, "speedup": a / b})
    a_median = statistics.median(p["baseline_ms"] for p in raw)
    b_median = statistics.median(p["candidate_ms"] for p in raw)
    return {"samples": raw, "median_baseline_ms": a_median, "median_candidate_ms": b_median,
            "ratio_of_medians": a_median / b_median,
            "median_paired_speedup": statistics.median(speedups),
            "min_paired_speedup": min(speedups), "max_paired_speedup": max(speedups)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true", help="CPU-only ABI/layout tests; no torch import")
    parser.add_argument("--mode", choices=("all", "correctness", "bench"), default="all")
    parser.add_argument("--rows", default="1,2,4,8,16")
    parser.add_argument("--projections", default="qkv,o")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="Comma-separated variants; s4w4 control included, or none for existing split kernels only")
    parser.add_argument("--include-v5-split", action="store_true",
                        help="Also test O with immutable-v5 SK4 noncluster projection + ordered FP32 reduction")
    parser.add_argument("--include-candidate-split", action="store_true",
                        help="Also test O with candidate-build ordinary SK4 noncluster projection + FP32 reduction")
    parser.add_argument("--cache", default="warm,cold")
    parser.add_argument("--iters", type=int, default=30, help="Sequential graph kernels per warm timed block")
    parser.add_argument("--pairs", type=int, default=7, help="Alternating ABBA/BAAB pairs per comparison")
    parser.add_argument("--cold-iters", type=int, default=5, help="Individually evicted samples per cold block")
    parser.add_argument("--eviction-mib", type=int, default=64)
    parser.add_argument("--reference-columns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=72519)
    parser.add_argument("--expect-baseline-sha256", help="Fail before testing if immutable baseline hash differs")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate fatbin paths are required except with --self-test")
    try:
        args.rows = list(dict.fromkeys(int(v) for v in args.rows.split(",")))
        assert args.rows and all(1 <= v <= 16 for v in args.rows)
        args.variants = [] if args.variants == "none" else list(dict.fromkeys(["s4w4"] + args.variants.split(",")))
        assert all(v in VARIANTS for v in args.variants)
        args.projections = list(dict.fromkeys(args.projections.split(",")))
        assert all(v in PROJECTIONS for v in args.projections)
        args.cache = list(dict.fromkeys(args.cache.split(",")))
        assert all(v in ("warm", "cold") for v in args.cache)
        assert min(args.iters, args.pairs, args.cold_iters, args.reference_columns) > 0
        assert args.eviction_mib >= 64
    except (ValueError, AssertionError):
        parser.error("invalid row/variant/projection/cache list or non-positive counts; eviction must be >=64 MiB")
    return args


def main():
    global torch
    args = arguments()
    cpu_self_test()
    if args.self_test:
        return
    import torch as torch_module
    torch = torch_module
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.init()
    # init() may not leave a current driver context on this host thread.
    # Keep an allocation alive, as the existing bench_kda_direct.py does.
    torch.cuda.set_device(0)
    context_anchor = torch.empty(1, device="cuda")
    torch.cuda.current_stream()
    capability = torch.cuda.get_device_capability()
    assert capability == (12, 1), f"This qualification is GB10-only, got SM{capability}"
    base, candidate = Native(args.baseline, "baseline_v5"), Native(args.candidate, "candidate")
    try:
        if args.expect_baseline_sha256:
            assert base.sha256 == args.expect_baseline_sha256.lower(), "Immutable baseline SHA mismatch"
        props = torch.cuda.get_device_properties(0)
        report(kind="provenance", baseline_path=str(base.path.resolve()), baseline_sha256=base.sha256,
               candidate_path=str(candidate.path.resolve()), candidate_sha256=candidate.sha256,
               gpu=torch.cuda.get_device_name(), capability=capability, torch=torch.__version__,
               cuda=torch.version.cuda, seed=args.seed, mode=args.mode, variants=args.variants,
               include_v5_split=args.include_v5_split,
               include_candidate_split=args.include_candidate_split,
               l2_cache_bytes=getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", None)),
               eviction_bytes=args.eviction_mib * 1024 ** 2, pdl=False, tensor_scale=1.0,
               correctness_gate_run=args.mode != "bench", graph_timing=True,
               timing_events="captured_external_record_nodes", timing_revision=2,
               qualification="immutable-v5 same-bit replacement; microbench only, not end-to-end")
        eviction = torch.zeros(args.eviction_mib * 1024 ** 2 // 4, dtype=torch.int32, device="cuda")
        for projection in args.projections:
            config = PROJECTIONS[projection]
            n, sk = config["n"], config["sk"]
            codes_cpu = torch.randint(0, 256, (n, K), dtype=torch.uint8)
            codes_cpu.masked_fill_((codes_cpu & 127) == 127, 126)
            scales_cpu = torch.randint(119, 132, (n, K // 32), dtype=torch.uint8)
            packed_codes, packed_scales = pack_codes(codes_cpu), pack_scales(scales_cpu)
            validate_tensor_packing(codes_cpu, scales_cpu, packed_codes, packed_scales)
            codes, scales = packed_codes.cuda(), packed_scales.cuda()
            immutable_weights = [digest(t) for t in (codes, scales)]
            report(kind="shape", projection=projection, n=n, k=K, split_k=sk,
                   cluster=[1, 1, sk] if sk > 1 else None, output="fp32" if config["wide"] else "bf16",
                   codes_sha256=immutable_weights[0], scales_sha256=immutable_weights[1],
                   code_layout="[Npad/64][K/64][8][32][2][8]",
                   scale_layout="[Npad/64][K/64][64][2]")
            for rows in args.rows:
                x_storage = torch.full((GUARD + rows * (K + 16) + GUARD,), SENTINEL,
                                       dtype=torch.bfloat16, device="cuda")
                x = x_storage[GUARD:-GUARD].view(rows, K + 16)[:, :K]
                saved = torch.randn((rows, K), dtype=torch.bfloat16).cuda()
                x.copy_(saved)
                storage = torch.full((GUARD + rows * n + GUARD,), SENTINEL,
                                     dtype=torch.float32 if config["wide"] else torch.bfloat16, device="cuda")
                out = storage[GUARD:-GUARD].view(rows, n)
                partial_storage = None
                launches = {"baseline_v5": lambda: base.projection("baseline_v5", projection, x, codes, scales, out, rows)}
                if (args.include_v5_split or args.include_candidate_split) and projection == "o":
                    partial_storage = torch.full((GUARD + sk * rows * n + GUARD,), SENTINEL,
                                                 dtype=torch.float32, device="cuda")
                    partial = partial_storage[GUARD:-GUARD].view(sk, rows, n)
                    if args.include_v5_split:
                        launches["baseline_v5_split"] = lambda: base.projection(
                            "baseline_v5_split", projection, x, codes, scales, out, rows, partial)
                    if args.include_candidate_split:
                        # Same ordinary template symbols, but from the new build:
                        # do not accidentally qualify only the old fatbin's SASS.
                        launches["candidate_build_split"] = lambda: candidate.projection(
                            "baseline_v5_split", projection, x, codes, scales, out, rows, partial)
                for variant in args.variants:
                    launches[variant] = lambda variant=variant: candidate.projection(variant, projection, x, codes, scales, out, rows)
                if args.mode != "bench":
                    correctness(launches, projection, rows, x, saved, x_storage, storage, out,
                                codes_cpu, scales_cpu, args.reference_columns, partial_storage)
                if args.mode != "correctness":
                    before = digest(x_storage)
                    replays, graphs = build_replays(launches, args.iters)
                    for mode in args.cache:
                        for variant in ["AA_control"] + list(launches)[1:]:
                            results = paired_benchmark(replays, variant, mode, args, eviction)
                            report(kind="timing", projection=projection, rows=rows, variant=variant,
                                   cache=mode, warm_graph_kernels=args.iters * (2 if variant in SPLIT_VARIANTS else 1),
                                   warm_graph_projections=args.iters,
                                   kernels_per_projection=2 if variant in SPLIT_VARIANTS else 1,
                                   cold_samples_per_block=args.cold_iters,
                                   pairs=args.pairs, **results)
                    torch.cuda.synchronize()
                    assert digest(x_storage) == before, "Benchmark changed activation/guards"
                    assert bool((storage[:GUARD] == SENTINEL).all() and
                                (storage[-GUARD:] == SENTINEL).all()), "Benchmark changed output guards"
                    if partial_storage is not None:
                        assert bool((partial_storage[:GUARD] == SENTINEL).all() and
                                    (partial_storage[-GUARD:] == SENTINEL).all()), "Benchmark changed partial guards"
                    # Replay bound methods retain graphs; release them before the buffers change.
                    del replays, graphs
            assert immutable_weights == [digest(t) for t in (codes, scales)], "Weight bytes changed"
        report(kind="summary", status="PASS", mode=args.mode, immutable_weights=True,
               limits="Rows<=16 synthetic operator tests; sampled FP64 reference; native and full-model tests still required")
    finally:
        torch.cuda.synchronize()
        candidate.close()
        base.close()
        del context_anchor


if __name__ == "__main__":
    main()
