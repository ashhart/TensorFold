"""Decode-window timing of the EXL3 dense linear (``cuda/exl3/linear.cu``) on full GLM-5.3's per-rank (TP=4) non-expert
shapes, 5-bit mul1 synthetic weights: per shape and window of R rows, microseconds a call and weight GB/s.

Every call is a CUDA graph node (no Python launch cost), and each shape rotates over enough copies of the layer
(>= 96 MB) that the words come from DRAM, not L2. Repeats 3 times, prints the median.

    python tests/cuda/bench_exl3_decode_linear.py [--tiles plan|tuned|sweep] [--loads 0 1] [--rows 1 3 4 8 16]
"""

from __future__ import annotations

import argparse
import statistics

import torch

from tensorfold.cuda.exl3 import linear

# q_a, q_b (and indexer wq_b), kv_a (640 stored), o_proj, shared expert gate/up and down, layers 0-2 MLP
SHAPES = [(6144, 2048), (2048, 4096), (6144, 640), (4096, 6144), (6144, 512), (512, 6144), (6144, 3072),
          (3072, 6144)]
BITS, CODEBOOK = 5, "mul1"
POOL_BYTES = 96 << 20


def make(k: int, n: int, copies: int) -> list:
    out = []
    g = torch.Generator(device="cuda").manual_seed(k * 7 + n)
    for _ in range(copies):
        tr = torch.randint(-2**15, 2**15, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device="cuda",
                           generator=g)
        suh = (torch.randint(0, 2, (k,), device="cuda", generator=g) * 2 - 1).half()
        svh = (torch.randint(0, 2, (n,), device="cuda", generator=g) * 2 - 1).half()
        out.append(linear.Exl3Linear.from_tensors(tr, suh, svh, CODEBOOK))
    return out


ROT = False          # --rot: each call also rotates its bf16 input (rot_in), as Exl3Linear.__call__ does


def call(lin, xh, out, z, loads: int, x=None) -> None:
    sk, wk = lin.split
    loads = linear.LOADS if loads is None else loads
    if x is not None:
        linear._ext().rot_in(x, lin.suh, xh, (loads >> 5) & 1)
    a, b = lin.strides
    args = [xh, lin.words, a, b, lin.svh, lin.bias, out, z if sk > 1 else None, lin.counters, lin.k2,
            linear.CODEBOOK_IDS[lin.codebook], sk, wk]
    linear._ext().linear(*args, loads)


def build(lins, rows: int, loads):
    """A CUDA graph of one call to each copy (and the buffers it holds on to)."""

    k, n = lins[0].k, lins[0].n
    sk = lins[0].split[0]
    xh = (torch.randn(rows, k, device="cuda") * 0.1).half()
    x = xh.bfloat16() if ROT else None
    outs = [torch.empty(rows, n, device="cuda", dtype=torch.bfloat16) for _ in lins]
    z = torch.empty(sk * rows * n, device="cuda", dtype=torch.float32)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for lin, o in zip(lins, outs):
            call(lin, xh, o, z, loads, x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for lin, o in zip(lins, outs):
                call(lin, xh, o, z, loads, x)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g, (xh, x, outs, z)


def time_one(g, calls: int, replays: int) -> float:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(replays):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (replays * calls)


def time_graphs(lins, rows: int, loads_list, reps: int = 3, replays: int = 20) -> list[float]:
    """Median microseconds a call for each load path, the paths' timings interleaved (drift hits them alike)."""

    graphs = [build(lins, rows, ld) for ld in loads_list]
    ts = [[] for _ in loads_list]
    for _ in range(reps):
        for i, (g, _) in enumerate(graphs):
            ts[i].append(time_one(g, len(lins), replays))
    return [statistics.median(t) for t in ts]


def time_graph(lins, rows: int, loads, reps: int = 3, replays: int = 20) -> float:
    return time_graphs(lins, rows, [loads], reps, replays)[0]


def candidates(k: int):
    kt = k // 16
    for sk in (1, 2, 4, 8, 16, 32, 64):
        for wk in (2, 4, 8):
            if kt % (sk * wk) == 0 and kt // (sk * wk) >= 2:
                yield sk, wk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tuned", choices=["plan", "tuned", "sweep"])
    ap.add_argument("--loads", type=int, nargs="*", default=None, help="kernel load paths to compare (linear.LOADS by default)")
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 3, 4, 8, 16])
    ap.add_argument("--shapes", nargs="*", default=None, help="KxN, e.g. 6144x2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rot", action="store_true", help="time rot_in + linear, the production pair")
    a = ap.parse_args()
    global ROT
    ROT = a.rot
    shapes = [tuple(map(int, s.split("x"))) for s in a.shapes] if a.shapes else SHAPES
    loads_list = a.loads if a.loads else [None]
    print(f"{'shape':>11} {'MB':>6} {'tiles':>8} {'ld':>2} " + " ".join(f"{'R=' + str(r):>15}" for r in a.rows))
    for k, n in shapes:
        nbytes = k * n * BITS // 8
        copies = max(2, -(-POOL_BYTES // nbytes))
        lins = make(k, n, copies)
        if a.tiles == "plan":
            tiles = [lins[0].split]
        elif a.tiles == "tuned":
            tiles = [time_tune(lins, loads_list[0])]
        else:
            tiles = list(candidates(k))
        for t in tiles:
            for lin in lins:
                lin.split = t
            res = {r: time_graphs(lins, r, loads_list, reps=a.reps) for r in a.rows}
            for i, ld in enumerate(loads_list):
                cells = [f"{res[r][i]:7.1f}us {nbytes / res[r][i] / 1e3:4.0f}" for r in a.rows]
                print(f"{k:>5}x{n:<5} {nbytes / 2**20:6.2f} {str(t):>8} {'-' if ld is None else ld:>2} "
                      + " ".join(f"{c:>15}" for c in cells), flush=True)
        del lins
        torch.cuda.empty_cache()


def time_tune(lins, loads, rows: int = 3) -> tuple[int, int]:
    """The (K splits, warps) the engine's start-up tuner would pick (fused.tune_linears), timed with graphs here."""

    best, best_t = None, None
    for t in candidates(lins[0].k):
        for lin in lins:
            lin.split = t
        us = time_graph(lins, rows, loads, reps=1, replays=5)
        if best_t is None or us < best_t:
            best, best_t = t, us
    return best


if __name__ == "__main__":
    main()
