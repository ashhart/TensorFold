"""DeepSeek-V4.1-Flash EXL3 weight-read timing on one rank of a TP=2 split: the go/no-go floor for decode.

Loads one rank's half of several layers (routed experts split along the expert width, attention split by heads,
shared expert split like a routed one) and the head's half, then times CUDA graphs of back-to-back layers with fresh
random expert picks each replay, so the reads come from DRAM, not L2. Attention math, norms, routers, Engram and the
collectives are not included: this is the weight-streaming part of a decode step.

    python tools/dsv41_bench_layer.py /models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw --rank 0 --layers 3,9,15,21,27,33
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.cuda.exl3.linear import Exl3Linear


class Shards:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.open: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        f = self.index[name]
        if f not in self.open:
            self.open[f] = safe_open(str(self.root / f), framework="pt")
        return self.open[f].get_tensor(name)

    def group(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get(prefix + ".trellis"), self.get(prefix + ".suh"), self.get(prefix + ".svh")


def split_n(t, suh, svh, rank, world=2):
    """Output columns of this rank: whole tile columns and their svh."""

    step = t.shape[1] // world
    return t[:, rank * step:(rank + 1) * step].contiguous(), suh, svh[rank * step * 16:(rank + 1) * step * 16]


def split_k(t, suh, svh, rank, world=2):
    """Input rows of this rank: whole tile rows and their suh (partial sums add across ranks)."""

    step = t.shape[0] // world
    return t[rank * step:(rank + 1) * step].contiguous(), suh[rank * step * 16:(rank + 1) * step * 16], svh


def linear(group, dev="cuda") -> Exl3Linear:
    t, suh, svh = group
    return Exl3Linear.from_tensors(t, suh, svh, "mul1", device=dev)


def cuda_group(group, dev="cuda"):
    t, suh, svh = group
    return t.to(dev).contiguous(), suh.to(dev), svh.to(dev)


class Layer:
    def __init__(self, sh: Shards, L: int, rank: int, experts: int, rows: int) -> None:
        p = f"layers.{L}"
        self.attn = [linear(sh.group(f"{p}.attn.wq_a")), linear(sh.group(f"{p}.attn.wkv")),
                     linear(split_n(*sh.group(f"{p}.attn.wq_b"), rank))]
        groups = 8
        self.wo_a = [linear(sh.group(f"{p}.attn.wo_a.slice.{g}")) for g in range(rank * groups // 2,
                                                                                 (rank + 1) * groups // 2)]
        self.wo_b = linear(split_k(*sh.group(f"{p}.attn.wo_b"), rank))
        self.shared = [linear(split_n(*sh.group(f"{p}.ffn.shared_experts.w1"), rank)),
                       linear(split_n(*sh.group(f"{p}.ffn.shared_experts.w3"), rank)),
                       linear(split_k(*sh.group(f"{p}.ffn.shared_experts.w2"), rank))]
        gate, up, down = [], [], []
        for e in range(experts):
            q = f"{p}.ffn.experts.{e}"
            gate.append(cuda_group(split_n(*sh.group(q + ".w1"), rank)))
            up.append(cuda_group(split_n(*sh.group(q + ".w3"), rank)))
            down.append(cuda_group(split_k(*sh.group(q + ".w2"), rank)))
        self.ex = ex3.prepare(gate, up, down, "mul1")
        self.scratch = ex3.Scratch(self.ex, rows, 6)
        self.pick = torch.zeros((rows, 6), dtype=torch.int32, device="cuda")
        self.wts = torch.full((rows, 6), 1 / 6, dtype=torch.float32, device="cuda")
        self.out = torch.zeros((rows, 5120), dtype=torch.float32, device="cuda")

    def bytes_dense(self) -> int:
        mods = self.attn + self.wo_a + [self.wo_b] + self.shared
        return sum(m.words.numel() * 4 for m in mods)

    def step(self, x: torch.Tensor, R: int, parts: str) -> None:
        if "a" in parts:
            q = self.attn[0](x[:R])
            self.attn[1](x[:R])
            self.attn[2](q.to(x.dtype))
            o = x.new_zeros((R, 4096))
            lat = torch.cat([m(o) for m in self.wo_a], dim=1)
            self.wo_b(lat)
        if "s" in parts:
            g = self.shared[0](x[:R])
            self.shared[1](x[:R])
            self.shared[2](g)
        if "e" in parts:
            ex3.routed(x, self.pick, self.wts, self.ex, self.scratch, self.out, R, limit=10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", default="3,9,15,21,27,33")
    ap.add_argument("--rows", default="1,4,8")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(0)
    sh = Shards(Path(args.model))
    rows_list = [int(r) for r in args.rows.split(",")]
    maxr = max(rows_list)
    t0 = time.time()
    layers = [Layer(sh, int(L), args.rank, args.experts, maxr) for L in args.layers.split(",")]
    head = linear(split_n(*sh.group("head"), args.rank))
    torch.cuda.synchronize()
    print(f"loaded {len(layers)} layers + head in {time.time() - t0:.0f} s; "
          f"torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    x = torch.randn((maxr, 5120), dtype=torch.float16, device="cuda") * 0.1
    per_expert = layers[0].ex.nbytes_read([0])
    print(f"rank {args.rank}: dense bytes/layer {layers[0].bytes_dense() / 1e6:.1f} MB, "
          f"one routed expert {per_expert / 1e6:.2f} MB, head {head.words.numel() * 4 / 1e6:.0f} MB")

    for R in rows_list:
        for parts in ("a", "s", "e", "ase"):
            def body(R=R, parts=parts):
                for lay in layers:
                    lay.step(x, R, parts)
            for lay in layers:
                lay.pick[:R] = torch.stack([torch.randperm(args.experts, device="cuda")[:6] for _ in range(R)]).int()
            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            times = []
            for _ in range(args.reps):
                for lay in layers:
                    lay.pick[:R] = torch.stack([torch.randperm(args.experts, device="cuda")[:6]
                                                for _ in range(R)]).int()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) / len(layers))
            times.sort()
            print(f"R={R:2d} {parts:4s} median {times[len(times) // 2] * 1000:7.1f} us/layer "
                  f"(min {times[0] * 1000:.1f})", flush=True)
        hs = []
        hx = x[:R]
        head(hx)
        for _ in range(9):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            head(hx)
            e.record()
            torch.cuda.synchronize()
            hs.append(s.elapsed_time(e))
        hs.sort()
        print(f"R={R:2d} head median {hs[4] * 1000:7.1f} us", flush=True)


if __name__ == "__main__":
    main()
