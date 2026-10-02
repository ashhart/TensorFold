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
    loads = linear.LOADS if loads is None else loads
    if x is not None:
        linear._ext().rot_in(x, [lin.suh], [xh], (loads >> 5) & 1)
    none = linear._none(xh.device)
    linear._ext().linear([xh], [lin.words], [lin.strides[0]], [lin.strides[1]], [lin.svh], [none], [out],
                         [z if lin.split[0] > 1 else none], [lin.counters], lin.k2, linear.CODEBOOK_IDS[lin.codebook],
                         [lin.split[0]], lin.split[1], loads)


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


GROUPS = [((6144, 2048), (6144, 640)), ((6144, 512), (6144, 512)), ((2048, 4096), (2048, 4096))]


def group_bench(rows: list[int], loads, reps: int) -> None:
    """Layers of one input (q_a + kv_a, gate + up, q_b + indexer wq_b): one call each vs one launch (linear.group),
    rot_in included; each layer's tiles the best for it alone at each warps-a-block, the best warps for each way."""

    for shapes in GROUPS:
        sets = [make(k, n, max(2, -(-POOL_BYTES // (k * n * BITS // 8 * len(shapes))))) for k, n in shapes]
        copies = min(len(s_) for s_ in sets)
        sets = [s_[:copies] for s_ in sets]
        nbytes = sum(k * n * BITS // 8 for k, n in shapes)
        best_seq, best_grp = {}, {}
        for wk in (2, 4, 8):
            tiles = []
            for s_ in sets:
                k = s_[0].k
                cand = [(sk, w) for sk, w in candidates(k) if w == wk]
                if not cand:
                    break
                bt, bu = None, None
                for t in cand:
                    for lin in s_:
                        lin.split = t
                    us = time_graph(s_, 3, loads, reps=1, replays=5)
                    if bu is None or us < bu:
                        bt, bu = t, us
                tiles.append(bt)
            if len(tiles) != len(sets):
                continue
            for s_, t in zip(sets, tiles):
                for lin in s_:
                    lin.split = t
            for r in rows:
                seq, grp = time_pair(sets, r, loads, reps)
                if r not in best_seq or seq < best_seq[r][0]:
                    best_seq[r] = (seq, tuple(tiles))
                if r not in best_grp or grp < best_grp[r][0]:
                    best_grp[r] = (grp, tuple(tiles))
        name = " + ".join(f"{k}x{n}" for k, n in shapes)
        for r in rows:
            (sq, st), (gp, gt) = best_seq[r], best_grp[r]
            print(f"{name:>22} R={r:<2} {nbytes / 2**20:5.2f} MB  one by one {sq:6.1f}us ({nbytes / sq / 1e3:3.0f} GB/s) "
                  f"{st}   one launch {gp:6.1f}us ({nbytes / gp / 1e3:3.0f} GB/s) {gt}", flush=True)


def time_pair(sets, rows: int, loads, reps: int) -> tuple[float, float]:
    """Microseconds a group: its layers one call each, and as one launch (graphs over the copies, interleaved)."""

    k = sets[0][0].k
    x = (torch.randn(rows, k, device="cuda") * 0.1).bfloat16()
    outs = [[torch.empty(rows, lin.n, device="cuda", dtype=torch.bfloat16) for lin in s_] for s_ in sets]
    graphs = []
    for grouped in (False, True):
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            def run():
                for c in range(len(sets[0])):
                    layers = [s_[c] for s_ in sets]
                    for lin in layers:
                        lin.loads = loads
                    if grouped:
                        linear.group(layers, x, [o[c] for o in outs])
                    else:
                        for lin, o in zip(layers, outs):
                            lin(x, out=o[c])
            run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                run()
        g.replay()
        torch.cuda.synchronize()
        graphs.append(g)
    ts = [[], []]
    for _ in range(reps):
        for i, g in enumerate(graphs):
            ts[i].append(time_one(g, len(sets[0]), 20))
    return statistics.median(ts[0]), statistics.median(ts[1])


# a MoE layer's EXL3 linears at TP=4 (rank 0) in decode order: (input group) -> outputs
LAYER = [("q_a", 6144, 2048), ("kv_a", 6144, 640), ("wq_b", 2048, 4096), ("q_b", 2048, 4096), ("o_proj", 4096, 6144),
         ("gate", 6144, 512), ("up", 6144, 512), ("down", 512, 6144)]
LAYER_GROUPS = [("q_a", "kv_a"), ("wq_b", "q_b"), ("o_proj",), ("gate", "up"), ("down",)]


def layer_bench(rows: list[int], reps: int, copies: int = 8) -> None:
    """A MoE layer's linears (rot_in included) over ``copies`` layers as one graph: the original kernel (loads 0)
    one call each, tiles from the engine's tuner, vs the default path with the engine's group tuner and ``lins``.
    A small elementwise kernel stands for the glue between the groups (it breaks the PDL chain, as in the model)."""

    from tensorfold.families.glm_moe_dsa.cuda import fused

    layers = []
    for c in range(copies):
        L = {}
        for name, k, n in LAYER:
            L[name] = make(k, n, 1)[0]
        layers.append(L)
    xs = {k: (torch.randn(max(rows), k, device="cuda") * 0.1).bfloat16() for _, k, _ in LAYER}
    outs = [{name: torch.empty(max(rows), n, device="cuda", dtype=torch.bfloat16) for name, _, n in LAYER}
            for _ in layers]
    every = [lin for L in layers for lin in L.values()]
    glue = torch.zeros(4096, device="cuda")
    res = {}
    for mode in ("before", "after"):
        for lin in every:
            lin.loads = 0 if mode == "before" else None
            lin.split = None
            lin.__post_init__()
        fused.tune_linears(every)
        if mode == "after":
            print("groups:", fused.tune_groups([[L[n] for n in grp] for L in layers for grp in LAYER_GROUPS
                                                if len(grp) > 1]), flush=True)
        for r in rows:
            def run():
                for L, o in zip(layers, outs):
                    for grp in LAYER_GROUPS:
                        x = xs[L[grp[0]].k][:r]
                        if mode == "after":
                            fused.lins([L[n] for n in grp], None, x, [o[n][:r] for n in grp])
                        else:
                            for n in grp:
                                L[n](x, out=o[n][:r])
                        glue.add_(1.0)                  # the norms, attention, ... between them (no PDL)
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                run()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    run()
            g.replay()
            torch.cuda.synchronize()
            res[mode, r] = g
    nbytes = sum(k * n * BITS // 8 for _, k, n in LAYER)
    for r in rows:
        ts = {"before": [], "after": []}
        for _ in range(reps):
            for mode in ts:
                ts[mode].append(time_one(res[mode, r], copies, 20))
        b, a = statistics.median(ts["before"]), statistics.median(ts["after"])
        print(f"MoE layer linears R={r}: {nbytes / 2**20:.1f} MB  before {b:6.1f}us ({nbytes / b / 1e3:3.0f} GB/s)  "
              f"after {a:6.1f}us ({nbytes / a / 1e3:3.0f} GB/s)  saving {b - a:5.1f}us a layer", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tuned", choices=["plan", "tuned", "sweep"])
    ap.add_argument("--loads", type=int, nargs="*", default=None, help="kernel load paths to compare (linear.LOADS by default)")
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 3, 4, 8, 16])
    ap.add_argument("--shapes", nargs="*", default=None, help="KxN, e.g. 6144x2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rot", action="store_true", help="time rot_in + linear, the production pair")
    ap.add_argument("--layer", action="store_true", help="a MoE layer's linears, before vs after")
    ap.add_argument("--groups", action="store_true", help="layers of one input: one by one vs one launch")
    a = ap.parse_args()
    global ROT
    ROT = a.rot
    if a.layer:
        layer_bench(a.rows, a.reps)
        return
    if a.groups:
        for ld in (a.loads or [None]):
            group_bench(a.rows, ld, a.reps)
        return
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
