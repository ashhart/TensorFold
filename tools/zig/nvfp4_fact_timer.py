"""Times the NVFP4 fact prompt tiles of two or more nvfp4_experts cubins on a synthetic Flash Next expert layer."""
import argparse
import ctypes
import json
import statistics
import sys

import torch

USAGE = """Kernels: tf_nvfp4_fact_swiglu and tf_nvfp4_fact_down_f32 (zig/kernels/cuda/nvfp4_experts.cu).

Layer: D 2560, intermediate 640, 512 experts, top-10 (Flash Next's routed experts). Random NVFP4 weights (finite
e4m3 scales), random bf16 inputs, uniform random routing (seeded), items of up to 64 pairs per expert as the
engine's grouped plan makes them. Weights stream from memory on every call (about 1 GB of experts, larger than L2).
Timing: each kernel captured in a CUDA graph and replayed (median of 30 replays, best of 3); modules alternate
order round by round inside one process. Synthetic kernel timings only, not served TTFT. Every module's output
bytes are checked against the first module's.

Requires an NVIDIA GPU, torch with CUDA, and nvcc to build the cubins; the CUDA driver is called through ctypes.
Build one cubin per checkout (ARCH is the GPU's, e.g. sm_121a, sm_120a, sm_90a, sm_89):

    FL="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \\
        -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3"
    (cd BASE && nvcc -cubin $FL -arch=$ARCH -I zig/kernels/cuda -o /tmp/base.cubin zig/kernels/cuda/nvfp4_experts.cu)
    (cd HEAD && nvcc -cubin $FL -arch=$ARCH -I zig/kernels/cuda -o /tmp/head.cubin zig/kernels/cuda/nvfp4_experts.cu)
    python3 tools/zig/nvfp4_fact_timer.py --mods base=/tmp/base.cubin,head=/tmp/head.cubin --T 512,1536,2048,8192

Cold processes in alternating order: run with --rounds 1 several times, swapping the --mods order between runs.

Each output row: T, kernel, items, items with 49-64 pairs (the 4-tile items), <mod>_us per mod, <mod>_equal
(output bytes equal to the first mod's) and <mod>_vs_<first>. Exit status 1 if any output differs."""

# ---------------------------------------------------------------- CUDA driver through ctypes
_cu = ctypes.CDLL("libcuda.so.1")
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def _chk(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        _cu.cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what}: CUDA driver error {rc} ({name.value.decode() if name.value else '?'})")


class Module:
    """A cubin loaded into torch's (primary) context."""

    def __init__(self, path):
        torch.zeros(1, device="cuda")  # torch makes its context current on this thread
        self.data = open(path, "rb").read()
        self.mod = ctypes.c_void_p()
        _chk(_cu.cuModuleLoadData(ctypes.byref(self.mod), ctypes.c_char_p(self.data)), f"load {path}")

    def fn(self, name, dyn_smem):
        f = ctypes.c_void_p()
        _chk(_cu.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), f"function {name}")
        _chk(_cu.cuFuncSetAttribute(f, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ctypes.c_int(dyn_smem)), "smem")
        return f


def launch(f, grid, block, smem, args):
    """args: ctypes values in the kernel's parameter order; launched on torch's current stream (graph capture works)."""
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _chk(_cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, params, None), "launch")


def graph_time(fn, replays=30, warm=10):
    """Median microseconds per call of fn, captured once in a CUDA graph and replayed."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(replays):
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


# ---------------------------------------------------------------- the synthetic layer
D, NI, E, K = 2560, 640, 512, 10
TILE = 64  # pairs an item holds at most
# kernel -> (M weight matrices, WN warps a CTA, input side, output bytes)
KINDS = {"fact_swiglu": (2, 4, "up", 2), "fact_down_f32": (1, 8, "down", 4)}


def smem(parts, wn):
    st = (64 * 8 + wn * 2 * parts * 36) * 16
    return (4 if st * 4 <= 49152 else 3 if st * 3 <= 49152 else 2) * st


def nvfp4(blocks, g):
    """32x32 NVFP4 blocks of 576 bytes: 512 code bytes, 64 e4m3 scale bytes in 0x28..0x3F (finite, ~0.03-0.94)."""
    w = torch.randint(0, 256, (blocks, 576), dtype=torch.uint8, device="cuda", generator=g)
    w[:, 512:] = torch.randint(0x28, 0x40, (blocks, 64), dtype=torch.uint8, device="cuda", generator=g)
    return w.flatten()


def plan(T):
    """Uniform random top-K routing of T rows; items of up to 64 pairs per expert, as (expert, first, count)."""
    ex = torch.stack([torch.randperm(E, generator=torch.Generator().manual_seed(t))[:K] for t in range(T)]).flatten()
    order = torch.argsort(ex, stable=True)
    cnt = torch.bincount(ex, minlength=E).tolist()
    items, first = [], 0
    for e in range(E):
        for o in range(0, cnt[e], TILE):
            items += [e, first + o, min(TILE, cnt[e] - o)]
        first += cnt[e]
    t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")
    return t(items), t([len(items) // 3]), order.to(torch.int32).cuda(), items


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mods", required=True, help="name=path.cubin,... (the first is the byte reference)")
    ap.add_argument("--T", default="512,1536,2048,8192", help="prompt rows per pass")
    ap.add_argument("--kinds", default="fact_swiglu,fact_down_f32")
    ap.add_argument("--rounds", type=int, default=4, help="alternating-order rounds in this process")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mods = dict(s.split("=", 1) for s in a.mods.split(","))
    names = list(mods)
    loaded = {n: Module(p) for n, p in mods.items()}
    kinds = a.kinds.split(",")
    print(f"# {torch.cuda.get_device_name()}, torch {torch.__version__}, mods {names}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(7)
    W, S = {}, {}
    for kind in kinds:
        M, WN, io, ob = KINDS[kind]
        kk, nn = (D, NI) if io == "up" else (NI, D)
        W[kind] = nvfp4(E * (nn // 32) * (kk // 32) * M, g)
        S[kind] = torch.rand(E * M, device="cuda", generator=g) * 0.01 + 0.001
    fns = {(n, k): loaded[n].fn(f"tf_nvfp4_{k}", smem(KINDS[k][0], KINDS[k][1])) for n in names for k in kinds}
    rows, differ = [], False
    for T in [int(v) for v in a.T.split(",")]:
        items, counts, members, il = plan(T)
        ni = len(il) // 3
        x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda", generator=g)
        h = torch.randn(T * K, NI, dtype=torch.bfloat16, device="cuda", generator=g)
        for kind in kinds:
            M, WN, io, ob = KINDS[kind]
            up = io == "up"
            kk, nn = (D, NI) if up else (NI, D)
            KG, NB = kk // 32, nn // 32
            grid = ni * ((NB + WN - 1) // WN)
            xin, stride, slots = (x, D, K) if up else (h, NI, 0)
            dt = torch.float32 if ob == 4 else torch.bfloat16
            outs = {n: torch.zeros(T * K * nn, dtype=dt, device="cuda") for n in names}

            def call(n, out):
                args = [ctypes.c_void_p(xin.data_ptr()), ctypes.c_int(stride), ctypes.c_int(slots),
                        ctypes.c_void_p(W[kind].data_ptr()), ctypes.c_void_p(S[kind].data_ptr()), ctypes.c_int(KG),
                        ctypes.c_int(NB), ctypes.c_void_p(items.data_ptr()), ctypes.c_void_p(counts.data_ptr()),
                        ctypes.c_void_p(members.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int(nn),
                        ctypes.c_float(0.0), ctypes.c_int(-1)]
                launch(fns[(n, kind)], grid, WN * 32, smem(M, WN), args)

            for n in names:
                call(n, outs[n])
            torch.cuda.synchronize()
            iv = torch.int32 if ob == 4 else torch.int16
            row = {"T": T, "kernel": kind, "items": ni, "full_items": sum(1 for c in il[2::3] if c > 48)}
            for n in names[1:]:
                row[f"{n}_equal"] = bool(torch.equal(outs[names[0]].view(iv), outs[n].view(iv)))
                differ |= not row[f"{n}_equal"]
            times = {n: [] for n in names}
            for r in range(a.rounds):
                for n in (names if r % 2 == 0 else names[::-1]):
                    times[n].append(min(graph_time(lambda n=n: call(n, outs[n])) for _ in range(3)))
            for n in names:
                row[f"{n}_us"] = round(statistics.median(times[n]), 1)
            for n in names[1:]:
                row[f"{n}_vs_{names[0]}"] = f"{100 * (row[f'{n}_us'] / row[f'{names[0]}_us'] - 1):+.2f}%"
            print(json.dumps(row), flush=True)
            rows.append(row)
            if a.out:
                json.dump(rows, open(a.out, "w"), indent=1)
            del outs
        del x, h
        torch.cuda.empty_cache()
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
