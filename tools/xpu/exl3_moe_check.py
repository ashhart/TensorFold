# SPDX-License-Identifier: Apache-2.0
"""Fused routed MLP (exl3_moe_gate_up, _act, _down) through pyopencl: check against the float64 reference,
then time it (device time a kernel, median). One Sage layer: top-k of the routed experts plus the shared expert,
every slot y = down(silu(gate x) * up x).

Usage: exl3_moe_check.py <repo> <EXL3 checkpoint> [reps]   (one GPU job at a time)
Env: MOE_P, MOE_PD, MOE_DS (gate|up K splits, down K splits, down strips a sub-group; default 4, 2, 2 as in
exl3_variants.moe), MOE_TOKENS (check sizes, default 1,4; empty: none), MOE_IDSETS (fresh picks cycled while timing,
default 32; 1 reads the same experts every call, so some stay in L2), EXL3_OPTS (more build options)."""
import os, sys
import numpy as np
import pyopencl as cl
sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref
from exl3_check import Exl3, Arena, DESC, to_bf16, from_bf16

LAYER = "model.language_model.layers.{}.mlp"
TILE_M = int(os.environ.get("TILE_M", "16"))    # exl3.cl TILE_M and TILE_NS (pass the same in EXL3_OPTS)
TILE_NS = int(os.environ.get("TILE_NS", "2"))


def table(ar, mats):
    d = np.zeros(len(mats), DESC)
    for i, (tr, suh, svh, _) in enumerate(mats):
        kt, nt, w = tr.shape
        d[i]["trellis"], d[i]["suh"], d[i]["svh"] = ar.add(tr), ar.add(suh), ar.add(svh)
        d[i]["k"], d[i]["n"], d[i]["halves"], d[i]["rows"] = kt * 16, nt * 16, w // 8, 1
    return d


class Moe:
    def __init__(self, e, P, PD, DS=1):
        self.e, self.P, self.PD, self.DS = e, P, PD, DS
        self.p = cl.Program(e.ctx, e.src).build(f"-cl-std=CL3.0 -DBITS=4 -DNS=1 -DROWS=1 -DMOE_P={P} -DMOE_PD={PD} -DMOE_DS={DS} " + os.environ.get("EXL3_OPTS", ""))
        self.gu, self.dn = self.p.exl3_moe_gate_up, self.p.exl3_moe_down

    def setup(self, experts, shared, x, ids_sets, topk):
        """experts: list of (gate, up, down) groups; shared: one (gate, up, down); x bf16 [tokens, H];
        ids_sets: list of u32 [tokens, topk] (one a call, cycled)."""
        e = self.e
        ar = Arena(e.ctx)
        mats = experts + [shared]
        self.tabs = [e.buf(table(ar, [m[j] for m in mats])) for j in range(3)]
        self.H, self.I = mats[0][0][0].shape[0] * 16, mats[0][0][0].shape[1] * 16
        self.T, self.topk = x.shape[0], topk
        self.pairs = self.T * topk
        self.slots = self.pairs + self.T
        self.x = ar.add(x)
        self.gu = ar.add(nbytes=self.slots * self.P * 2 * self.I * 4)
        self.act = ar.add(nbytes=self.slots * self.I * 4)
        self.y = ar.add(nbytes=self.slots * self.H * 4)
        self.rot = ar.add(nbytes=2 * self.slots * self.H * 4)  # prompt tiles: gate|up rotated input, then down sums
        self.dns = ar.add(nbytes=self.slots * self.H * 4)
        self.ar, self.base = ar, ar.build(e.q)
        self.ids = [e.buf(np.ascontiguousarray(i, np.uint32)) for i in ids_sets]
        self.shared_id = len(experts)

    def launch(self, i=0):
        u32, u64 = np.uint32, np.uint64
        ids = self.ids[i % len(self.ids)]
        g, u, d = self.tabs
        ev = [self.p.exl3_moe_gate_up(self.e.q, (self.I // 128 * self.P * 128, self.slots), (128, 1), self.base, g, u, ids,
                                      u32(self.topk), u32(self.pairs), u32(self.shared_id), u64(self.x), u32(self.H), u64(self.gu), u32(0)),
              self.p.exl3_moe_act(self.e.q, (self.I // 128 * 32, self.slots), (32, 1), self.base, g, u, d, ids, u32(self.pairs),
                                  u32(self.shared_id), u64(self.gu), u64(self.act), u32(self.P)),
              self.p.exl3_moe_down(self.e.q, (self.H // 128 * 128 // self.DS * self.PD, self.slots), (128 // self.DS * self.PD, 1),
                                   self.base, d, ids, u32(self.pairs), u32(self.shared_id), u64(self.act), u64(self.y), u32(self.H), u32(1))]
        return ev

    def prompt(self, ids):
        """The prompt path (expert tiles on XMX): the runner's sort on the host, then exl3_moe_rot, exl3_moe_tile
        (gate|up), exl3_moe_act (one split), exl3_moe_tile (down), exl3_moe_out_rows."""
        e, u32, u64 = self.e, np.uint32, np.uint64
        ne = self.shared_id
        H, I, S = self.H, self.I, self.slots
        nt = S // TILE_M + ne + 1  # tile bound
        ob = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, S * 4)
        tb = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, nt * 16)
        nb = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, 4)
        ib = e.buf(np.ascontiguousarray(ids, np.uint32))
        g, u, d = self.tabs
        ev = [self.p.exl3_moe_sort(e.q, (256,), (256,), ib, ob, tb, nb, u32(ne), u32(self.pairs), u32(self.T)),
              self.p.exl3_moe_rot(e.q, (H // 128 * 16, S, 2), (16, 1, 1), self.base, g, u, ib, u32(self.topk), u32(self.pairs),
                                  u32(self.shared_id), u64(self.x), u32(H), u32(0), u64(self.rot), u32(S)),
              self.p.exl3_moe_tile(e.q, ((I // 16 // TILE_NS * 16 + 127) // 128 * 128, nt, 2), (128, 1, 1), self.base, g, u, ob, tb, nb,
                                   u64(self.rot), u64(S * H), u64(self.gu), u32(2 * I), u32(I)),
              self.p.exl3_moe_act(e.q, (I // 128 * 32, S), (32, 1), self.base, g, u, d, ib, u32(self.pairs), u32(self.shared_id),
                                  u64(self.gu), u64(self.act), u32(1)),
              self.p.exl3_moe_tile(e.q, ((H // 16 // TILE_NS * 16 + 127) // 128 * 128, nt, 1), (128, 1, 1), self.base, d, d, ob, tb, nb,
                                   u64(self.act), u64(0), u64(self.dns), u32(H), u32(0)),
              self.p.exl3_moe_out_rows(e.q, (H // 128 * 16, S), (16, 1), self.base, d, ib, u32(self.pairs), u32(self.shared_id),
                                       u64(self.dns), u64(self.y), u32(H), u32(1))]
        return ev

    def y_out(self):
        self.e.q.finish()
        return self.ar.read(self.e.q, self.y, (self.slots, self.H), np.float32)


def dq(m):
    tr, suh, svh, _ = m
    return ref.dequantize(tr.shape[2] // 8, 2, tr, suh, svh)


def main():
    repo, sage = sys.argv[1:3]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    P, PD, DS = (int(os.environ.get(k, d)) for k, d in (("MOE_P", "4"), ("MOE_PD", "2"), ("MOE_DS", "2")))
    e = Exl3(repo)
    m = Moe(e, P, PD, DS)
    st = ref.SafeTensors(sage)
    rng = np.random.default_rng(5)
    L = LAYER.format(0)
    grp = lambda p: tuple(st.group(f"{p}.{j}_proj") for j in ("gate", "up", "down"))
    # check: 24 experts, ids among them, float64 reference
    sub = [grp(f"{L}.experts.{i}") for i in range(24)]
    shared = grp(f"{L}.shared_expert")
    worst = 0.0
    for T in [int(t) for t in os.environ.get("MOE_TOKENS", "1,4").split(",") if t]:
        x = to_bf16(rng.standard_normal((T, 2048)).astype(np.float32))
        ids = np.stack([rng.permutation(24)[:8] for _ in range(T)]).astype(np.uint32)
        m.setup(sub, shared, x, [ids], 8)
        m.launch()
        y = m.y_out()
        xf = from_bf16(x).astype(np.float64)
        for s in range(m.slots):
            ex = sub[ids.flat[s]] if s < m.pairs else shared
            t = s // 8 if s < m.pairs else s - m.pairs
            g, u = xf[t] @ dq(ex[0]), xf[t] @ dq(ex[1])
            yr = (g / (1 + np.exp(-g)) * u) @ dq(ex[2])
            worst = max(worst, np.abs(y[s] - yr).max() / np.abs(yr).max())
        print(f"moe tokens={T} slots={m.slots}: worst max|y-ref|/max|ref| = {worst:.2e}", flush=True)
    assert worst <= 1e-4, worst
    # prompt tiles: MOE_PROMPT tokens (default 64), ids among the 24 experts
    for T in [int(t) for t in os.environ.get("MOE_PROMPT", "64").split(",") if t]:
        x = to_bf16(rng.standard_normal((T, 2048)).astype(np.float32))
        ids = np.stack([rng.permutation(24)[:8] for _ in range(T)]).astype(np.uint32)
        m.setup(sub, shared, x, [ids], 8)
        ev = m.prompt(ids)
        y = m.y_out()
        xf = from_bf16(x).astype(np.float64)
        cache, pw = {}, 0.0
        for s in range(m.slots):
            ei = int(ids.flat[s]) if s < m.pairs else -1
            ex = sub[ei] if ei >= 0 else shared
            if ei not in cache:
                cache[ei] = tuple(dq(ex[j]) for j in range(3))
            G, U, D = cache[ei]
            t = s // 8 if s < m.pairs else s - m.pairs
            gg, uu = xf[t] @ G, xf[t] @ U
            yr = (gg / (1 + np.exp(-gg)) * uu) @ D
            pw = max(pw, np.abs(y[s] - yr).max() / np.abs(yr).max())
        tt = [(v.profile.end - v.profile.start) * 1e-3 for v in ev]
        print(f"moe prompt tiles tokens={T} slots={m.slots}: worst max|y-ref|/max|ref| = {pw:.2e}; us sort/rot/gu/act/down/out " + " ".join(f"{v:.0f}" for v in tt), flush=True)
        assert pw <= 1e-4, pw
    # time: all 256 experts resident, fresh picks each call (see MOE_IDSETS)
    full = [grp(f"{L}.experts.{i}") for i in range(256)]
    for T in [int(t) for t in os.environ.get("MOE_PROMPT_TIME", "512").split(",") if t]:  # a prompt chunk, timing only
        x = to_bf16(rng.standard_normal((T, 2048)).astype(np.float32))
        ids = np.stack([rng.permutation(256)[:8] for _ in range(T)]).astype(np.uint32)
        m.setup(full, shared, x, [ids], 8)
        tt = []
        for _ in range(max(3, reps // 10)):
            ev = m.prompt(ids)
            e.q.finish()
            tt.append([(v.profile.end - v.profile.start) * 1e-3 for v in ev])
        tt = np.median(np.array(tt), axis=0)
        print(f"moe prompt tiles tokens={T} slots={m.slots}, 256 experts: us sort/rot/gu/act/down/out " + " ".join(f"{v:.0f}" for v in tt) + f", sum {tt.sum():.0f} us", flush=True)
    x = to_bf16(rng.standard_normal((1, 2048)).astype(np.float32))
    m.setup(full, shared, x, [rng.permutation(256)[:8].reshape(1, 8) for _ in range(int(os.environ.get("MOE_IDSETS", "32")))], 8)
    t = []
    for i in range(reps):
        ev = m.launch(i)
        e.q.finish()
        t.append([(v.profile.end - v.profile.start) * 1e-9 for v in ev])
    t = np.median(np.array(t), axis=0)
    gu_bytes = sum(a.nbytes for j in (0, 1) for a in full[0][j][:3]) * 9
    dn_bytes = sum(a.nbytes for a in full[0][2][:3]) * 9
    print(f"P={P} PD={PD} DS={DS} 1 token, 8 picks + shared: gate_up {t[0]*1e6:.1f} us {gu_bytes/t[0]/1e9:.0f} GB/s, act {t[1]*1e6:.1f} us, "
          f"down {t[2]*1e6:.1f} us {dn_bytes/t[2]/1e9:.0f} GB/s, sum {sum(t)*1e6:.1f} us {(gu_bytes+dn_bytes)/sum(t)/1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
