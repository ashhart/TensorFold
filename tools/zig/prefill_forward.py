"""Nemotron prompt chunks through our kernels only: mlx_lm's backbone op for op, with the engine's state layout."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

import prefill_glue as pg
import prefill_launch as pl
from check_mlx_ops import op_file
from prefill_kernels import Runner

LANE_ROWS = 128                       # lane_qmm's rows; wider chunks take MLX's 4-bit matmuls (ours here)
STEP, KV_STEP, EPS = 256, 256, 1e-5   # mlx_lm's SSD step and KVCache step; RMS eps


@dataclass
class State:
    """A layer's prompt state: conv rows [3, C] and SSM [1, H, DH, N] (Mamba), or a KV buffer pair and offset."""

    conv: mx.array | None = None
    ssm: mx.array | None = None
    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0
    extra: dict = field(default_factory=dict)


class Prefill:
    """The backbone of an engine-loaded NemotronH, its weights in MLX's layout beside lane_qmm's for wide chunks."""

    def __init__(self, backbone, args) -> None:
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        self.b, self.args, self.run = backbone, args, Runner(mx)
        self.untiled = {}
        for layer in backbone.layers:
            for m in self.projections(layer):
                w = lane_qmm.untile_weight(m["weight"], m._lane_nt, m.group_size, bits=m.bits) \
                    if getattr(m, "_lane_tiled", False) else m["weight"]
                self.untiled[id(m)] = w
        mx.eval(list(self.untiled.values()))
        self.ones = {n: mx.ones((n,), dtype=mx.bfloat16) for n in (args.hidden_size, self.group_width())}
        self.eps, self.dims = mx.array([EPS], dtype=mx.float32), {}
        mx.eval(list(self.ones.values()))       # load-time constants: no MLX fill may run inside a prefill
        self.a = {}

    def group_width(self) -> int:
        return self.args.mamba_num_heads * self.args.mamba_head_dim // self.args.n_groups

    @staticmethod
    def projections(layer):
        m = layer.mixer
        names = {"M": ("in_proj", "out_proj"), "*": ("q_proj", "k_proj", "v_proj", "o_proj")}.get(layer.block_type, ())
        found = [getattr(m, n) for n in names]
        if layer.block_type == "E":
            found += [m.shared_experts.up_proj, m.shared_experts.down_proj]
        return found

    def outs(self, launch) -> list:
        v, inputs, grid, group, shapes, _ = launch
        return self.run(v, inputs, grid, group, shapes, verify=False)

    def op(self, stem: str, name: str, inputs: list, outputs: list, grid: tuple, group: tuple) -> list:
        return op_file(stem).run(name, inputs, outputs, grid, group)

    def rms(self, x, w, width: int):
        rows = x.size // width
        threads = 32 * -(-(-(-width // 4)) // 32)
        dim = self.dims.setdefault(width, mx.array([width], dtype=mx.int32))
        return self.op("embed_norm", "tf_rms_norm_bf16", [x, w, self.eps, dim], [((rows, width), mx.bfloat16)],
                       (threads * rows, 1, 1), (threads, 1, 1))[0]

    def add(self, a, b):
        n = a.size
        return self.op("elementwise", "tf_add_bf16", [a, b, mx.array([n], dtype=mx.int32)],
                       [(a.shape, mx.bfloat16)], (n, 1, 1), (256, 1, 1))[0]

    def relu2(self, x):
        n = x.size
        return self.op("elementwise", "tf_relu2_bf16", [x, mx.array([n], dtype=mx.int32)], [(x.shape, mx.bfloat16)],
                       (n, 1, 1), (256, 1, 1))[0]

    def linear(self, x, m):
        """A 4-bit projection as the engine runs it: lane_qmm up to 128 rows, else MLX's split-K or NAX qmm (ours)."""

        rows, N, K = x.size // x.shape[-1], int(m["scales"].shape[0]), x.shape[-1]
        if rows <= LANE_ROWS:
            return m(x.reshape(1, rows, K)).reshape(rows, N)
        w, s, b = self.untiled[id(m)], m["scales"], m["biases"]
        parts = pg.splitk_parts(rows, N, K)
        launches = pg.qmm_splitk(x, w, s, b, parts)[1] if parts > 1 else pl.qmm(x, w, s, b)[1]
        return pl.run(self.run, launches, verify=False).reshape(rows, N)

    def embed(self, ids: list[int]):
        e = self.b.embeddings
        rows, D = len(ids), self.args.hidden_size
        idx = mx.array(list(ids) + [0] * max(0, 8 - rows), dtype=mx.uint32)
        return self.op("embed_norm", "tf_embed_b4_g64", [idx, e["weight"].view(mx.uint8), e["scales"], e["biases"],
                                                         mx.array([D], dtype=mx.int32)],
                       [((rows, D), mx.bfloat16)], (D // 2, rows, 1), (256, 1, 1))[0]

    def forward(self, ids: list[int], states: list[State]) -> mx.array:
        """One prompt chunk: the final-normed rows [L, D]; states (one per Mamba or attention layer) advance."""

        h = self.embed(ids)
        at = 0
        for layer in self.b.layers:
            hn = self.rms(h, layer.norm.weight, self.args.hidden_size)
            if layer.block_type == "M":
                out = self.mamba(layer.mixer, hn, states[at])
                at += 1
            elif layer.block_type == "*":
                out = self.attention(layer.mixer, hn, states[at])
                at += 1
            else:
                out = self.moe(layer.mixer, hn)
            h = self.add(h, out)
            mx.eval(h, *[a for s in states for a in (s.conv, s.ssm, s.keys, s.values) if a is not None])
        return self.rms(h, self.b.norm_f.weight, self.args.hidden_size)

    def mamba(self, mx_mod, x, st: State):
        a = self.args
        L, H, DH, G, N = x.shape[0], a.mamba_num_heads, a.mamba_head_dim, a.n_groups, a.ssm_state_size
        inner, C = H * DH, H * DH + 2 * G * N
        proj = self.linear(x, mx_mod.in_proj)
        P = proj.shape[-1]
        pad = self.outs(pg.launch(pg.CONV_PACK, [proj, st.conv if st.conv is not None else proj],
                                  [((L + 3, C), mx.bfloat16)], (C, L + 3, 1), (256, 1, 1),
                                  [L, C, P, inner, int(st.conv is not None)]))[0]
        conv = pl.run(self.run, pl.conv(pad, mx_mod.conv1d.weight, L, C, 4)[1], verify=False).reshape(L, C)
        act = self.outs(pg.launch(pg.CONV_ACT, [conv, mx_mod.conv1d.bias], [((L, C), mx.bfloat16)], (L * C, 1, 1),
                                  (256, 1, 1), [L * C, C]))[0]
        st.conv = self.outs(pg.launch(pg.CONV_KEEP, [pad], [((3, C), mx.bfloat16)], (C, 3, 1), (256, 1, 1),
                                      [L, C]))[0]
        if id(mx_mod) not in self.a:
            self.a[id(mx_mod)] = self.outs(pg.launch(pg.MAMBA_A, [mx_mod.A_log], [((H,), mx.float32)], (H, 1, 1),
                                                     (H, 1, 1), [H]))[0]
        dt, dta, dtx = self.outs(pg.launch(pg.MAMBA_DT, [proj, mx_mod.dt_bias, self.a[id(mx_mod)], act],
                                           [((L, H), mx.float32), ((L, H), mx.float32), ((L, H * DH), mx.float32)],
                                           (H * DH, L, 1), (256, 1, 1), [L, P, inner + C, C, H, DH]))
        ycat = None
        for i0 in range(0, L, STEP):
            s = min(STEP, L - i0)
            cb = pl.run(self.run, pl.gemm(act, act, s, s, N, C, C, False, True, G, (N, N),
                                          (i0 * C + inner + G * N, i0 * C + inner))[1], verify=False)
            seg_in = self.outs(pg.launch(pg.SEGSUM_IN, [dta], [((H, s, s), mx.float32)], (s, s, H), (32, 8, 1),
                                         [i0, s, H]))[0]
            seg = pl.run(self.run, pl.scan(seg_in, H, s, s)[1], verify=False)
            sur, last = self.outs(pg.launch(pg.SSD_DECAY, [seg, cb], [((H, s, s), mx.float32), ((H, s), mx.float32)],
                                            (s, s, H), (32, 8, 1), [s, H, G]))
            y = pl.run(self.run, pl.gemm(sur, dtx, s, DH, s, s, H * DH, False, False, H, (s * s, DH),
                                         (0, i0 * H * DH))[1], verify=False)
            bh = self.outs(pg.launch(pg.SSD_B, [act], [((H, N, s), mx.float32)], (s, N, H), (32, 8, 1),
                                     [i0, s, C, inner, H, G, N]))[0]
            dd = self.outs(pg.launch(pg.SSD_DD, [dtx, last], [((s, H * DH), mx.float32)], (H * DH, s, 1), (256, 1, 1),
                                     [i0, s, H, DH]))[0]
            nxt = pl.run(self.run, pl.gemm(dd, bh, DH, N, s, H * DH, s, True, True, H, (DH, N * s))[1], verify=False)
            carry = st.ssm is not None
            e = prev = y
            if carry:
                cs = pl.run(self.run, pl.scan(dta, 1, s, H, offset=i0 * H)[1], verify=False)
                e = self.outs(pg.launch(pg.EXP, [cs], [((s, H), mx.float32)], (s * H, 1, 1), (256, 1, 1), [s * H]))[0]
                old = st.ssm.reshape(H, DH, N)
                cf = self.outs(pg.launch(pg.SSD_C, [act], [((s, G * N), mx.float32)], (G * N, s, 1), (256, 1, 1),
                                         [i0, s, C, inner + G * N, G * N]))[0]
                prev = pl.run(self.run, pl.gemv(old, cf, s, G, H, DH, N)[1], verify=False)
                nxt = self.outs(pg.launch(pg.STATE_CARRY, [nxt, e, old], [((H, DH, N), mx.float32)],
                                          (H * DH * N, 1, 1), (256, 1, 1), [s, H, DH * N]))[0]
            keep = int(ycat is not None)
            ycat = self.outs(pg.launch(pg.Y_OUT, [y, e, prev, ycat if keep else proj], [((L, inner), mx.bfloat16)],
                                       (inner, L if keep else s, 1), (256, 1, 1), [i0, s, H, DH, int(carry), L, keep]))[0]
            st.ssm = nxt.reshape(1, H, DH, N)
        yo = self.outs(pg.launch(pg.SKIP, [ycat, act, mx_mod.D], [((L, inner), mx.bfloat16)], (inner, L, 1),
                                 (256, 1, 1), [L, C, H, DH]))[0]
        g = self.outs(pg.launch(pg.GATE, [proj, yo], [((L, inner), mx.bfloat16)], (inner, L, 1), (256, 1, 1),
                                [L, inner, P]))[0]
        width = self.group_width()
        nrm = self.rms(g, self.ones[width], width)
        gn = self.outs(pg.launch(pg.SCALE, [nrm, mx_mod.norm.weight], [((L, inner), mx.bfloat16)], (L * inner, 1, 1),
                                 (256, 1, 1), [L * inner, inner]))[0]
        return self.linear(gn, mx_mod.out_proj)

    def attention(self, m, x, st: State):
        a = self.args
        L, H, HK, D = x.shape[0], a.num_attention_heads, a.num_key_value_heads, a.head_dim
        q, k, v = self.linear(x, m.q_proj), self.linear(x, m.k_proj), self.linear(x, m.v_proj)
        prev, old_cap = st.offset, (st.keys.shape[2] if st.keys is not None else 0)
        cap = old_cap
        if st.keys is None or prev + L > old_cap:
            base = 0 if st.keys is None else (prev if prev % KV_STEP else old_cap)
            cap = base + -(-L // KV_STEP) * KV_STEP
        bufs = []
        for new, old in ((k, st.keys), (v, st.values)):
            params = [L, HK, D, cap, prev, max(old_cap, 1), 1]
            bufs.append(self.outs(pg.launch(pg.KV_PUT, [new, old if old is not None else new],
                                            [((1, HK, cap, D), mx.bfloat16)], (D, prev + L, HK), (128, 2, 1),
                                            params))[0])
        st.keys, st.values, st.offset = bufs[0], bufs[1], prev + L
        kL, row, kvh = prev + L, H * D, HK * cap * D
        strides = (L * row, D, row, kvh, cap * D, D, kvh, cap * D, D, L * row, D, row)
        o = pl.run(self.run, pl.attention(q, st.keys, st.values, L, kL, strides, D ** -0.5)[1], verify=False)
        return self.linear(o.reshape(L, row), m.o_proj)

    def moe(self, m, x):
        a = self.args
        L, D, k, E = x.shape[0], a.hidden_size, a.num_experts_per_tok, a.n_routed_experts
        logits = pl.run(self.run, pl.matmul_nt(x, m.gate.weight)[1], verify=False).reshape(L, E)
        ids, wt = self.op("route", "tf_route_topk", [logits, m.gate.e_score_correction_bias,
                                                     mx.array([1e-20, a.routed_scaling_factor], dtype=mx.float32),
                                                     mx.array([E, k], dtype=mx.int32)],
                          [((L, k), mx.uint32), ((L, k), mx.float32)], (128, L, 1), (128, 1, 1))
        n = L * k
        flat = ids.reshape(n)
        counts = self.outs(pg.sort_count(flat, E))[0]
        starts, offsets = self.outs(pg.sort_starts(counts, n, E))
        order, sorted_ids = self.outs(pg.sort_place(flat, starts, n, E))
        fc1, fc2 = m.switch_mlp.fc1, m.switch_mlp.fc2
        if n >= 16 and n // E >= 4:
            xs = self.outs(pg.launch(pg.ROWS_TAKE, [x, order], [((n, D), mx.bfloat16)], (D, n, 1), (256, 1, 1),
                                     [n, D, k]))[0]
            y1 = self.rhs(xs, fc1, offsets)
            y2 = self.rhs(self.relu2(y1), fc2, offsets)
        else:
            xids = self.outs(pg.launch(pg.ROWS_OF, [order], [((n,), mx.uint32)], (n, 1, 1), (256, 1, 1), [n, k]))[0]
            rows = self.outs(pg.launch(pg.ROWS_OF, [order], [((n,), mx.uint32)], (n, 1, 1), (256, 1, 1), [n, 0]))[0]
            y1 = self.gather_qmv(x, fc1, sorted_ids, xids)
            y2 = self.gather_qmv(self.relu2(y1), fc2, sorted_ids, rows)
        inv = self.outs(pg.launch(pg.SORT_INV, [order], [((n,), mx.uint32)], (n, 1, 1), (256, 1, 1), [n]))[0]
        yu = self.outs(pg.launch(pg.ROWS_TAKE, [y2, inv], [((n, D), mx.bfloat16)], (D, n, 1), (256, 1, 1),
                                 [n, D, 1]))[0]
        comb = self.op("elementwise", "tf_moe_combine_rows", [yu, wt.reshape(n), mx.array([k, D], dtype=mx.int32)],
                       [((L, D), mx.bfloat16)], (D, L, 1), (256, 1, 1))[0]
        shared = self.linear(self.relu2(self.linear(x, m.shared_experts.up_proj)), m.shared_experts.down_proj)
        return self.add(comb, shared)

    def rhs(self, xs, fc, offsets):
        n, K, (E, N) = xs.shape[0], xs.shape[-1], fc["weight"].shape[:2]
        bm = 32 if n // E < 64 else 64
        v = pl.Variant(pl.RHS, ("bfloat16", bm), ("bfloat16", "uint32", "bfloat16", "bfloat16", "int32", "int32",
                                                  "bfloat16"))
        params = [n, N, K, E]
        launch = (v, [xs, fc["weight"], fc["scales"], fc["biases"], offsets, pl.p_array(params)],
                  (-(-N // 64) * 32, min(n, -(-n // bm) + E - 1) * 2, 2), (32, 2, 2), [(n, N)], params)
        return self.outs(launch)[0]

    def gather_qmv(self, x, fc, ids, xids):
        n, (N, K) = ids.size, (fc["weight"].shape[1], x.shape[-1])
        return self.op("qmv", "tf_gather_qmv_b4_g64", [fc["weight"], fc["scales"], fc["biases"], x, ids, xids,
                                                        mx.array([K, N], dtype=mx.int32)],
                       [((n, N), mx.bfloat16)], (32, N // 8 * 2, n), (32, 2, 1))[0]
