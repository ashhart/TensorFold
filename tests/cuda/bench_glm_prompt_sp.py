"""Sequence-parallel prompt chunks (fused.compute_prompt_sp) vs today's prompt path, on real weights (TF_GLM53_CKPT,
layers 0-3), four ranks as threads of one GPU.

    PYTHONPATH=src:tests/cuda python tests/cuda/bench_glm_prompt_sp.py [--rows 2048] [--prompts 1000,3000,4500]
        [--iters 3] [--solo]

Accuracy: every configuration prefills each prompt in ``--rows`` chunks; the final hidden rows and the last row's
logits are compared with the exact reference (fused.compute, rank-order reduce, no overlap); greedy first tokens must
match; SP exact runs twice and must be bit-identical. Timing: per-chunk wall of the four rank threads on one GPU
(every rank's work serialized: a sum, not one rank's time), and with ``--solo`` one rank of four alone with a local
stand-in for the collectives (one rank's GPU time, comm excluded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from threadcomm import RingThreadComm, run_ranks  # noqa: E402

CKPT = os.environ.get("TF_GLM53_CKPT", "")
LAYERS = (0, 1, 2, 3)

# name -> (PROMPT_SP, PREFILL_REDUCE, SP_SELECT, path)
CONFIGS = {
    "ref-exact": (False, "rs", True, "compute"),
    "ar-ring": (False, "ring", True, "prompt"),
    "sp-ring": (True, "ring", True, "sp"),
    "sp-exact": (True, "rs", True, "sp"),
    "sp-exact-again": (True, "rs", True, "sp"),
    "sp-exact-repsel": (True, "rs", False, "sp"),
}


def _weights(rank, world, comm):
    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layer

    cfg = Config.from_dict(json.loads((Path(CKPT) / "config.json").read_text()))
    r = RankReader(CKPT, rank, world)
    embed, norm, head = (r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight",
                                                     "lm_head.weight"))
    layers = [load_layer(r, cfg, i) for i in LAYERS]
    return fused.Weights(cfg, rank, world, comm, embed, norm, head, layers, None)


def _set(cfg):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    fused.PROMPT_SP, fused.PREFILL_REDUCE, fused.SP_SELECT = cfg[0], cfg[1], cfg[2]


def _prefill(w, st, b0, b1, pos1, cs, toks, rows, path, times=None):
    """Chunks of ``rows``; returns (final hidden of every row, last row's logits)."""
    from tensorfold.families.glm_moe_dsa.cuda import fused

    L0 = len(toks)
    hid = []
    for a in range(0, L0, rows):
        e = min(a + rows, L0)
        R = e - a
        T = e if e > w.cfg.index_topk else None
        b0.ids[:R].copy_(toks[a:e])
        st.pos.fill_(a)
        pos1.fill_(a + R // 2)
        last = "last" if e == L0 else "none"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if path == "compute":
            fused.compute(w, st, b0, R, T, logits=last)
        elif path == "prompt":
            fused.compute_prompt(w, st, b0, b1, R, T, pos1, cs, logits=last)
        else:
            fused.compute_prompt_sp(w, st, b0, b1, R, T, pos1, cs, logits=last)
        torch.cuda.synchronize()
        if times is not None:
            times.append((R, T is not None, time.perf_counter() - t0))
        hid.append(b0.hidden[:R].clone())
    return torch.cat(hid), b0.logits[:1].clone()


def accuracy(rows, prompts, iters):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    cap = max(prompts) + 8
    g = torch.Generator().manual_seed(1234)
    prompt_toks = [torch.randint(0, 150000, (n,), generator=g) for n in prompts]

    def run(rank, comm):
        w = _weights(rank, 4, comm)
        st = fused.State(w, cap)
        b0 = fused.Buffers(w, rows, cap)
        b1 = fused.Buffers(w, rows - rows // 2, cap)
        pos1 = torch.zeros((1,), dtype=torch.int32, device="cuda")
        cs = torch.cuda.Stream()
        out, times = {}, {}
        for name, cfg in CONFIGS.items():
            comm.barrier()
            _set(cfg)
            comm.barrier()
            out[name] = [_prefill(w, st, b0, b1, pos1, cs, t.cuda(), rows, cfg[3]) for t in prompt_toks]
            tt = []
            for _ in range(iters):               # timing: the longest prompt again
                _prefill(w, st, b0, b1, pos1, cs, prompt_toks[-1].cuda(), rows, cfg[3], tt)
            times[name] = tt
        return out, times

    res = run_ranks(run, 4, comm_cls=RingThreadComm)
    out, times = res[0]
    ref = out["ref-exact"]
    print(f"accuracy vs ref-exact (fused.compute, rank-order reduce), {rows}-row chunks, prompts {prompts}")
    for name in CONFIGS:
        if name == "ref-exact":
            continue
        for k, n in enumerate(prompts):
            h, lg = out[name][k]
            rh, rl = ref[k]
            dh = (h.float() - rh.float()).abs()
            rel_h_max = float((dh.max(dim=-1).values / rh.float().abs().max(dim=-1).values).max())
            rel_h = float(dh.norm() / rh.float().norm())
            dl = (lg - rl).abs()
            rel_l_max = float(dl.max() / rl.abs().max())
            rel_l = float(dl.norm() / rl.norm())
            same = int(lg.argmax()) == int(rl.argmax())
            print(f"  {name:16s} n={n:5d}  hidden rel(norm) {rel_h:.2e} rowmax {rel_h_max:.2e}  "
                  f"logits rel(norm) {rel_l:.2e} max {rel_l_max:.2e}  greedy {'==' if same else '!='} "
                  f"({int(lg.argmax())} vs {int(rl.argmax())})")
    a, b = out["sp-exact"], out["sp-exact-again"]
    rep = all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))
    print(f"  sp-exact run-to-run bit-identical: {rep}")
    # every rank alike?
    for name in ("sp-ring", "sp-exact"):
        alike = all(torch.equal(res[0][0][name][k][0].cpu(), res[r][0][name][k][0].cpu())
                    for r in range(1, 4) for k in range(len(prompts)))
        print(f"  {name}: final hidden identical on all 4 ranks: {alike}")
    print(f"threadcomm wall per chunk (4 rank threads on 1 GPU; prompt {prompts[-1]}):")
    for name, tt in times.items():
        _report(name, tt, -(-prompts[-1] // rows))


def _report(name, tt, chunks):
    """Per prefill (``chunks`` chunks each): best total, and best first / last chunk (shortest / longest context)."""
    runs = [tt[i:i + chunks] for i in range(0, len(tt), chunks)]
    best = min(runs, key=lambda r: sum(t for _, _, t in r))
    first = min(r[0][2] for r in runs)
    last = min(r[-1][2] for r in runs)
    print(f"  {name:16s} total {1e3 * sum(t for _, _, t in best):8.1f} ms  ({len(best)} chunks: first {1e3 * first:.1f} ms, "
          f"last {1e3 * last:.1f} ms)")


class SoloComm:
    """One rank of ``world`` alone: collectives become local copies of the right shapes (timing only)."""

    def __init__(self, world):
        self.rank, self.world = 0, world

    def all_gather(self, send, recv):
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            if flat[r * n:(r + 1) * n].data_ptr() != send.data_ptr():
                flat[r * n:(r + 1) * n].copy_(send.reshape(-1))

    def all_reduce(self, send, recv):
        recv.copy_(send)

    def reduce_scatter(self, send, recv):
        recv.view(-1).copy_(send.reshape(-1)[:recv.numel()])

    def all_to_all(self, send, recv):
        recv.copy_(send)

    def barrier(self):
        pass


def solo(rows, prompts, iters):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    n = max(prompts)
    cap = n + 8
    toks = torch.randint(0, 150000, (n,), generator=torch.Generator().manual_seed(7)).cuda()
    with torch.no_grad():
        w = _weights(0, 4, SoloComm(4))
        st = fused.State(w, cap)
        b0 = fused.Buffers(w, rows, cap)
        b1 = fused.Buffers(w, rows - rows // 2, cap)
        pos1 = torch.zeros((1,), dtype=torch.int32, device="cuda")
        cs = torch.cuda.Stream()
        print(f"solo rank 0 of 4 (collectives = local copies), {rows}-row chunks, prompt {n}:")
        for name in ("ar-ring", "sp-ring", "sp-exact", "sp-exact-repsel", "ar-ring", "sp-ring"):
            cfg = CONFIGS[name]
            _set(cfg)
            tt = []
            _prefill(w, st, b0, b1, pos1, cs, toks, rows, cfg[3])      # warm
            for _ in range(iters):
                _prefill(w, st, b0, b1, pos1, cs, toks, rows, cfg[3], tt)
            _report(name, tt, -(-n // rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--prompts", default="1000,3000,4500")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--solo", action="store_true")
    ap.add_argument("--layers", default="0,1,2,3")
    a = ap.parse_args()
    LAYERS = tuple(int(v) for v in a.layers.split(","))
    ps = [int(v) for v in a.prompts.split(",")]
    with torch.no_grad():
        (solo if a.solo else accuracy)(a.rows, ps, a.iters)
