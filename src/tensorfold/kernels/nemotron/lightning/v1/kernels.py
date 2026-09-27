"""Nemotron-H decode glue in a few kernels (Nemotron 3.5 Lightning 30B-A3B). Between the matmuls:

    add_norm     residual add (with the MoE combine) + the next block's RMSNorm
    route        sigmoid scores + correction bias -> top 6 experts and their weights
    mamba_scan   conv + SiLU over every row at once, then dt + SSM state update + D skip + SiLU(z) gate in order
    group_norm   the Mamba output's RMSNorm over groups of 512

Every kernel takes R rows of consecutive tokens and treats them in order, and
a row's value depends only on its own inputs and the rows before it, so the
bits of a row do not depend on how many rows ride with it. The arithmetic
follows mlx_lm's (fp32 math, bf16 where mlx_lm stores bf16) but is its own:
serial decoding goes through these kernels too, so they define the reference.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from tensorfold.kernels.inputs import ints, padded
from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels

_ADD_NORM = r"""
  // one threadgroup of T threads per row; thread t owns elements t, t + T, t + 2T, ...
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float partial[T / 32];
  float hv[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const int at = int(r) * D + c;
    float delta;
    MIX
    const bfloat hn = bfloat(float(H[at]) + delta);
    HN[at] = hn;
    hv[i] = float(hn);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    OUT[int(r) * D + c] = bfloat(float(W[c]) * (hv[i] * scale));
  }
"""


def _with_group_sums(source: str) -> str:
    """``_ADD_NORM`` that also writes the lane matmul's input sums of OUT: XS [D / 64, MP] for rows padded to MP
    (dims = (R, MP)), each group's 64 bf16 values summed in order in fp32 as lane_qmm's XSUM kernel does (same
    bits), rows past R zero. The next dense projection then skips its XSUM dispatch."""

    head = "  const uint r = threadgroup_position_in_grid.x;\n"
    store = "    OUT[int(r) * D + c] = bfloat(float(W[c]) * (hv[i] * scale));\n"
    assert head in source and store in source
    source = source.replace(head, head + """  threadgroup bfloat xb[D];
  if (int(r) >= dims[0]) {
    for (int g = int(t); g < D / 64; g += T) XS[g * dims[1] + int(r)] = 0.0f;
    return;
  }
""")
    source = source.replace(store, """    const bfloat x = bfloat(float(W[c]) * (hv[i] * scale));
    OUT[int(r) * D + c] = x;
    xb[c] = x;
""")
    return source + """  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int g = int(t); g < D / 64; g += T) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(xb[g * 64 + i]);
    XS[g * dims[1] + int(r)] = acc;
  }
"""


# plain residual: the block's output
_MIX_PLAIN = "delta = float(X[at]);"
# MoE: sum_e w_e y_e (fp32, experts in order) rounded to bf16, plus the shared expert (bf16 add), as mlx_lm
_MIX_MOE = r"""{
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(float(bfloat(routed)) + float(SH[at])));
    }"""

_ROUTE = r"""
  // one simdgroup per row: lane l holds experts l, l + 32, l + 64, l + 96
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sel[NE / 32], prob[NE / 32];
  for (int j = 0; j < NE / 32; j++) {
    const int e = int(lane) + 32 * j;
    const float g = float(G[int(r) * NE + e]);
    prob[j] = 1.0f / (1.0f + metal::exp(-g));
    sel[j] = prob[j] + bias[e];
  }
  float total = 0.0f;
  float picked[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      const int e = int(lane) + 32 * j;
      if (sel[j] > best) { best = sel[j]; best_e = e; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));   // ties: the lowest expert id
    float p = 0.0f;
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
    }
    p = simd_sum(p);
    picked[k] = p;
    total += p;
    if (lane == 0) IDX[int(r) * K + k] = uint(winner);
  }
  if (lane == 0) {
    const float denominator = total + 1e-20f;
    for (int k = 0; k < K; k++) WT[int(r) * K + k] = picked[k] / denominator * scaling[0];
  }
"""

_MAMBA_CONV = r"""
  // grid (CD, R): channel ch of row rr. Rows come in segments, one per stream (SEG: a row's segment, START: a
  // segment's first row); a segment's taps before its first row come from its conv state, row SLOT[s] of CS_IN.
  // Writes the conv output bf16(silu(bf16(conv))) as mlx_lm rounds it, and the row's conv state (its segment's
  // last KC-1 inputs).
  constexpr int CD = XD + 2 * NG * DS;
  const int ch = int(thread_position_in_grid.x);
  const int rr = int(thread_position_in_grid.y);
  const int s = SEG[rr];
  const int b = START[s];
  const int loc = rr - b;
  const int slot = SLOT[s];
  #define TAP(lp) ((lp) < 0 ? CS_IN[(slot * (KC - 1) + (lp) + KC - 1) * CD + ch] : P[(b + (lp)) * PROJ + XOFF + ch])
  float a = float(CB[ch]);
  for (int k = 0; k < KC; k++) a = fma(CW[k * CD + ch], float(TAP(loc - (KC - 1) + k)), a);
  const float cv = float(bfloat(a));
  XBC[rr * CD + ch] = bfloat(cv / (1.0f + metal::exp(-cv)));
  for (int k = 0; k < KC - 1; k++) CS_OUT[(rr * (KC - 1) + k) * CD + ch] = TAP(loc - (KC - 2) + k);
  #undef TAP
"""

_MAMBA_SCAN = r"""
  // grid (32, DH, H): lane = NS state elements of channel d of head h. Rows in order; segment s starts from its
  // stream's SSM state, row SLOT[s] of S_IN. A row's arithmetic depends only on its own inputs and the state
  // before it.
  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int R = dims[0];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  const int sbase = cx * DS + int(lane) * NS;
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);
  float st[NS];
  int cur = -1;
  for (int rr = 0; rr < R; rr++) {
    const int s = SEG[rr];
    if (s != cur) {
      cur = s;
      for (int i = 0; i < NS; i++) st[i] = float(S_IN[size_t(SLOT[s]) * SSZ + sbase + i]);
    }
    const float xv = float(XBC[rr * CD + cx]);
    float dt = float(P[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));   // softplus (logaddexp(x, 0))
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float sv = dA * st[i] + xdt * float(XBC[rr * CD + cb + i]);
      st[i] = sv;
      acc += sv * float(XBC[rr * CD + cc + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
    }
    // the SSM state after this row (a verify window keeps the state of its last accepted row)
    for (int i = 0; i < NS; i++) S_OUT[size_t(rr) * SSZ + sbase + i] = st[i];
  }
"""

_GROUP_NORM = r"""
  // one threadgroup of GS / 4 threads per (row, group): thread t owns 4 consecutive elements
  const uint t = thread_position_in_threadgroup.x;
  const uint grp = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int T = GS / 4;
  threadgroup float partial[T / 32];
  const int base = int(r) * XD + int(grp) * GS + int(t) * 4;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = fma(v[i], v[i], ss); }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(GS) + eps[0]);
  for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    OUT[base + i] = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));
  }
"""


_ROUTER = r"""
  // bf16 router logits for R rows: one threadgroup of SG simdgroups per (expert, block of MAXR rows). Simdgroup g
  // sums its D / SG inputs (lane l: 4 consecutive inputs at a time, 128 apart), then simd_sum; the simdgroups'
  // sums are added in order. A row's logits have the same bits at any row count (the row count is a runtime value).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * MAXR;
  const int R = min(rows[0] - rb, MAXR);
  constexpr int PART = D / SG;
  threadgroup float part[MAXR][SG];
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  const int begin = int(g) * PART;
  for (int c = begin + 4 * int(lane); c < begin + PART; c += 128) {
    const float w0 = float(GW[size_t(e) * D + c]), w1 = float(GW[size_t(e) * D + c + 1]);
    const float w2 = float(GW[size_t(e) * D + c + 2]), w3 = float(GW[size_t(e) * D + c + 3]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + (rb + r) * D + c;
      acc[r] = fma(float(xr[3]), w3, fma(float(xr[2]), w2, fma(float(xr[1]), w1, fma(float(xr[0]), w0, acc[r]))));
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) part[r][g] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && int(lane) < R) {
    float total = 0.0f;
    for (int k = 0; k < SG; k++) total += part[lane][k];
    OUT[(rb + int(lane)) * NE + e] = bfloat(total);
  }
"""

_kernels: dict[str, Any] = {}


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    arch = str(info.get("architecture", ""))
    digits = "".join(ch for ch in arch.removeprefix("applegpu_g") if ch.isdigit())
    return bool(digits) and int(digits) >= 17


def _named(base: str, source: str) -> str:
    return f"{base}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str]) -> Any:
    key = _named(name, source)
    kernel = _kernels.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(name=key, input_names=inputs, output_names=outputs, source=source)
        _kernels[key] = kernel
    return kernel


_NORM_DIMS: dict[int, mx.array] = {}


def _norm_call(name: str, mix: str, names: list[str], inputs: list[mx.array], template: list, rows: int, dims: int,
               group_sums: bool) -> tuple[mx.array, ...]:
    threads = 896 if dims % 896 == 0 else 256
    source = _ADD_NORM.replace("MIX", mix)
    outputs, shapes, dtypes = ["HN", "OUT"], [(rows, dims), (rows, dims)], [mx.bfloat16, mx.bfloat16]
    grid = rows
    if group_sums:
        grid = 16 * -(-rows // 16)                  # the lane matmul's padded rows
        if rows not in _NORM_DIMS:
            _NORM_DIMS[rows] = ints((rows, grid))
        source, name = _with_group_sums(source), name + "_xs"
        names, inputs = names + ["dims"], inputs + [_NORM_DIMS[rows]]
        outputs, shapes, dtypes = outputs + ["XS"], shapes + [(dims // 64, grid)], dtypes + [mx.float32]
    kernel = _kernel(name, source, names, outputs)
    return tuple(kernel(inputs=inputs, template=[("D", dims), ("T", threads), *template],
                        grid=(threads * grid, 1, 1), threadgroup=(threads, 1, 1),
                        output_shapes=shapes, output_dtypes=dtypes))


def add_norm(h: mx.array, delta: mx.array, weight: mx.array, eps: mx.array, *, group_sums: bool = False
             ) -> tuple[mx.array, ...]:
    """(h + delta, RMSNorm(h + delta) * weight) for rows [R, D], both bf16; with ``group_sums`` also the lane
    matmul's input sums of the second (``_with_group_sums``)."""

    rows, dims = h.shape[0], h.shape[-1]
    return _norm_call("nemotron_add_norm", _MIX_PLAIN, ["H", "X", "W", "eps"], [h, delta, weight, eps], [],
                      rows, dims, group_sums)


def add_norm_moe(h: mx.array, routed: mx.array, weights: mx.array, shared: mx.array, weight: mx.array,
                 eps: mx.array, *, group_sums: bool = False) -> tuple[mx.array, ...]:
    """As ``add_norm`` with delta = bf16(bf16(sum_e w_e y_e) + shared): routed [R, E, D], weights [R, E]."""

    rows, experts, dims = routed.shape
    return _norm_call("nemotron_add_norm_moe", _MIX_MOE, ["H", "Y", "WE", "SH", "W", "eps"],
                      [h, routed, padded(weights), shared, weight, eps], [("E", experts)], rows, dims, group_sums)


_ROUTER_ROWS: dict[int, mx.array] = {}


def router_logits(x: mx.array, gate_w: mx.array, *, simdgroups: int = 8) -> mx.array:
    """x [R, D] @ gate_w.T [D, E] -> [R, E] bf16, each row with the same bits at any R (blocks of 16 rows)."""

    rows, dims = x.shape
    experts = gate_w.shape[0]
    count = _ROUTER_ROWS.get(rows)
    if count is None:
        count = mx.array([rows], dtype=mx.int32)
        _ROUTER_ROWS[rows] = count
    if dims % (simdgroups * 4):
        raise ValueError("router_logits: D must split into simdgroups of 4-wide steps")
    kernel = _kernel("nemotron_router", _ROUTER, ["X", "GW", "rows"], ["OUT"])
    return kernel(inputs=[x, gate_w, count], template=[("D", dims), ("NE", experts), ("SG", simdgroups), ("MAXR", 16)],
                  grid=(32 * simdgroups, experts, -(-rows // 16)), threadgroup=(32 * simdgroups, 1, 1),
                  output_shapes=[(rows, experts)], output_dtypes=[mx.bfloat16])[0]


def _stack_linears(linears: list[Any]) -> tuple[Any, list[int]]:
    """One quantized linear for projections that read the same input; returns it and the split points."""

    import mlx.nn as nn

    first = linears[0]
    stacked = nn.QuantizedLinear(first.weight.shape[1] * 32 // first.bits, 1, bias=False,
                                 group_size=first.group_size, bits=first.bits)
    stacked.weight = mx.concatenate([l.weight for l in linears], axis=0)
    stacked.scales = mx.concatenate([l.scales for l in linears], axis=0)
    stacked.biases = mx.concatenate([l.biases for l in linears], axis=0)
    mx.eval(stacked.parameters())
    cuts, total = [], 0
    for l in linears[:-1]:
        total += l.weight.shape[0]
        cuts.append(total)
    return stacked, cuts


def route(logits: mx.array, bias: mx.array, top_k: int, scaling: mx.array) -> tuple[mx.array, mx.array]:
    """Expert ids [R, K] (best first; ties to the lower id) and weights [R, K] (fp32) from gate logits [R, E]."""

    rows, experts = logits.shape
    kernel = _kernel("nemotron_route", _ROUTE, ["G", "bias", "scaling"], ["IDX", "WT"])
    return kernel(inputs=[logits, bias, scaling], template=[("NE", experts), ("K", top_k)],
                  grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, top_k), (rows, top_k)], output_dtypes=[mx.uint32, mx.float32])


_TABLES: dict[tuple[str, tuple[int, ...]], Any] = {}      # a call's small index tables, by the call's layout


def _table(kind: str, key: tuple[int, ...], build: Any) -> Any:
    found = _TABLES.get((kind, key))
    if found is None:
        if len(_TABLES) >= 1024:                             # many streams give many layouts: keep the recent
            _TABLES.clear()
        found = _TABLES[(kind, key)] = build()
    return found


def _segments(lengths: tuple[int, ...]) -> tuple[mx.array, mx.array, mx.array]:
    """(row count, each row's segment, each segment's first row) for segments of ``lengths`` rows."""

    def build() -> tuple[mx.array, mx.array, mx.array]:
        seg = [i for i, n in enumerate(lengths) for _ in range(n)]
        starts, at = [], 0
        for n in lengths:
            starts.append(at)
            at += n
        return mx.array([at], dtype=mx.int32), ints(seg), ints(starts)

    return _table("segments", lengths, build)


def mamba_scan(proj: mx.array, conv_states: mx.array, ssm_states: mx.array, lengths: tuple[int, ...],
               conv_w: mx.array, conv_b: mx.array, a_log: mx.array, d_skip: mx.array, dt_bias: mx.array,
               limits: mx.array, *, heads: int, head_dim: int, groups: int, state_dim: int,
               slots: tuple[int, ...] | None = None) -> tuple[mx.array, mx.array, mx.array]:
    """Rows of several streams through one Mamba-2 mixer's conv and scan: segment i is ``lengths[i]`` consecutive
    tokens of stream i, from its conv state ``conv_states[slots[i]]`` [KC-1, CD] and SSM state
    ``ssm_states[slots[i]]`` [H, DH, DS] (slot i when ``slots`` is None). Returns gated y [R, XD] and every row's
    conv and SSM states after it: [R, KC-1, CD] and [R, H, DH, DS]. A row's arithmetic is the same whatever the
    other rows and segments (one code path)."""

    rows, width = proj.shape
    slots = tuple(range(len(lengths))) if slots is None else tuple(int(s) for s in slots)
    held = min(int(conv_states.shape[0]), int(ssm_states.shape[0]))
    if sum(lengths) != rows or len(slots) != len(lengths) or not all(0 <= s < held for s in slots):
        raise ValueError("mamba_scan: lengths must cover the rows, a state slot per segment")
    slot = _table("slots", slots, lambda: ints(slots))
    xd = heads * head_dim
    conv_dim = xd + 2 * groups * state_dim
    kc = conv_w.shape[0]
    dims, seg, starts = _segments(tuple(int(n) for n in lengths))
    conv = _kernel("nemotron_mamba_conv", _MAMBA_CONV, ["P", "CS_IN", "CW", "CB", "SEG", "START", "SLOT"],
                   ["XBC", "CS_OUT"])
    xbc, conv_rows = conv(
        inputs=[proj, conv_states, conv_w, conv_b, seg, starts, slot],
        template=[("XD", xd), ("NG", groups), ("DS", state_dim), ("KC", kc), ("PROJ", width), ("XOFF", xd)],
        grid=(conv_dim, rows, 1), threadgroup=(min(256, conv_dim), 1, 1),
        output_shapes=[(rows, conv_dim), (rows, kc - 1, conv_dim)], output_dtypes=[mx.bfloat16, conv_states.dtype])
    scan = _kernel("nemotron_mamba_scan", _MAMBA_SCAN,
                   ["P", "XBC", "S_IN", "A_LOG", "DSKIP", "DT_BIAS", "limits", "dims", "SEG", "SLOT"], ["Y", "S_OUT"])
    y, ssm_rows = scan(
        inputs=[proj, xbc, ssm_states, a_log, d_skip, dt_bias, limits, dims, seg, slot],
        template=[("H", heads), ("DH", head_dim), ("NG", groups), ("DS", state_dim), ("XD", xd), ("PROJ", width),
                  ("DTOFF", xd + conv_dim), ("SSZ", heads * head_dim * state_dim)],
        grid=(32, head_dim, heads), threadgroup=(32, 8, 1),
        output_shapes=[(rows, xd), (rows, heads, head_dim, state_dim)], output_dtypes=[mx.bfloat16, ssm_states.dtype])
    return y, conv_rows, ssm_rows


def mamba_step(proj: mx.array, conv_state: mx.array, ssm_state: mx.array, conv_w: mx.array, conv_b: mx.array,
               a_log: mx.array, d_skip: mx.array, dt_bias: mx.array, limits: mx.array, *, heads: int,
               head_dim: int, groups: int, state_dim: int) -> tuple[mx.array, mx.array, mx.array]:
    """R consecutive tokens of one stream through one Mamba-2 mixer's conv and scan (``mamba_scan`` with one
    segment): gated y [R, XD] and the conv and SSM states after every row, [R, KC-1, CD] and [R, H, DH, DS] (row
    r's states are the cache after the first r+1 tokens)."""

    return mamba_scan(proj, conv_state, ssm_state, (int(proj.shape[0]),), conv_w, conv_b, a_log, d_skip, dt_bias,
                      limits, heads=heads, head_dim=head_dim, groups=groups, state_dim=state_dim)


def group_norm(x: mx.array, weight: mx.array, eps: mx.array, group: int) -> mx.array:
    rows, dims = x.shape
    kernel = _kernel("nemotron_group_norm", _GROUP_NORM, ["X", "W", "eps"], ["OUT"])
    return kernel(inputs=[x, weight, eps], template=[("XD", dims), ("GS", group)],
                  grid=((group // 4) * (dims // group), rows, 1), threadgroup=(group // 4, 1, 1),
                  output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


class FusedDecode:
    """Nemotron-H decode (one or more consecutive rows) through the kernels above and MLX's matmuls."""

    # layers per slice handed to the GPU while the rest of the forward is built (0: the caller evaluates)
    eval_every = 8

    def __init__(self, model: Any) -> None:
        args = model.args
        self.model = model
        self.backbone = model.backbone
        self.layers = model.backbone.layers
        # lane attention needs the M5's tensor units (its fragment layout is theirs; an M3 gets wrong values)
        self.lane_attention = tensor_units()
        self.lane_attention_from = 10_000
        self.eps_value = float(args.layer_norm_epsilon)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.limits = mx.array([float(args.time_step_limit[0]), float(args.time_step_limit[1])], dtype=mx.float32)
        self.scaling = mx.array([float(args.routed_scaling_factor or 1.0)], dtype=mx.float32)
        self.top_k = int(args.num_experts_per_tok)
        self.heads, self.head_dim = int(args.mamba_num_heads), int(args.mamba_head_dim)
        self.groups, self.state_dim = int(args.n_groups), int(args.ssm_state_size)
        self.mamba: dict[int, tuple[mx.array, ...]] = {}
        # the last call's Mamba states after each of its rows, by layer (for keeping a prefix of a window)
        self.row_states: dict[int, tuple[mx.array, mx.array]] = {}
        self._compiled_blocks: dict[int, Any] = {}
        self.mamba_conv_dim = int(args.mamba_num_heads * args.mamba_head_dim + 2 * args.n_groups * args.ssm_state_size)
        for i, layer in enumerate(self.layers):
            if layer.block_type == "M":
                m = layer.mixer
                conv_w = m.conv1d.weight[:, :, 0].T.astype(mx.float32)          # [KC, CD]
                conv_b = (m.conv1d.bias if "bias" in m.conv1d else mx.zeros((m.conv_dim,))).astype(mx.float32)
                self.mamba[i] = (conv_w, conv_b, m.A_log.astype(mx.float32), m.D.astype(mx.float32),
                                 m.dt_bias.astype(mx.float32))
        mx.eval(list(self.mamba.values()))
        self.gate_bias = {i: layer.mixer.gate.e_score_correction_bias.astype(mx.float32)
                          for i, layer in enumerate(self.layers) if layer.block_type == "E"}
        mx.eval(list(self.gate_bias.values()))
        self.qkv: dict[int, tuple[Any, list[int]]] = {}
        for i, layer in enumerate(self.layers):
            if layer.block_type == "*":
                self.qkv[i] = _stack_linears([layer.mixer.q_proj, layer.mixer.k_proj, layer.mixer.v_proj])
        # with the lane matmul (set when it is installed): each norm kernel also writes the next projection's
        # 64-group input sums, handed to it (in_proj, q|k|v, shared up, head) instead of its own XSUM dispatch;
        # ``_no_xs`` stands in where no sums exist (the first layer's input, or no lane matmul)
        self.lane_xs = False
        self._no_xs = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, inputs: mx.array, cache: list[Any]) -> mx.array:
        """Hidden states after the final norm, [1, R, D], for R consecutive tokens (batch 1)."""

        tokens = inputs.reshape(-1)
        rows = tokens.shape[0]
        h = self.backbone.embeddings(tokens)                                     # [R, D]
        normed = mx.fast.rms_norm(h, self.layers[0].norm.weight, self.eps_value)
        xs = self._no_xs
        cache_at = 0
        for i, layer in enumerate(self.layers):
            kind = layer.block_type
            nxt = self.layers[i + 1].norm.weight if i + 1 < len(self.layers) else self.backbone.norm_f.weight
            if kind == "M":
                c = cache[cache_at]
                cache_at += 1
                conv_state, ssm_state = self._mamba_states(c, normed.dtype)
                block = self._block(i, "M", nxt)
                h, normed, xs, conv_rows, ssm_rows = block(normed, xs, h, conv_state, ssm_state)
                self._hold(c, conv_rows, ssm_rows, rows - 1)
                self.row_states[i] = (conv_rows, ssm_rows)
                c.advance(rows)
            elif kind == "*":
                c = cache[cache_at]
                cache_at += 1
                self._use_sums(normed, xs)
                delta = self._attention(layer.mixer, normed, c, i)
                h, normed, xs = self._add_norm(h, delta, nxt, xs)
            else:
                block = self._block(i, "E", nxt)
                h, normed, xs = block(normed, xs, h)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return self._use_sums(normed.reshape(1, rows, -1), xs)

    def run_streams(self, tokens: mx.array, lengths: tuple[int, ...], caches: list[list[Any]]) -> mx.array:
        """Hidden states after the final norm, [1, N, D], for several streams' consecutive tokens in one forward:
        rows laid out stream by stream (``lengths``), stream i's rows advancing only ``caches[i]``. A row gets the
        bits it gets in a call of its stream alone: every kernel treats rows on their own, the scan carries each
        stream's state through its own segment, and attention reads each stream's own keys."""

        lengths = tuple(int(n) for n in lengths)
        if len(lengths) == 1:
            return self(tokens, caches[0])
        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        h = self.backbone.embeddings(tokens)                                     # [N, D]
        normed = mx.fast.rms_norm(h, self.layers[0].norm.weight, self.eps_value)
        xs = self._no_xs
        cache_at = 0
        for i, layer in enumerate(self.layers):
            kind = layer.block_type
            nxt = self.layers[i + 1].norm.weight if i + 1 < len(self.layers) else self.backbone.norm_f.weight
            if kind == "M":
                layer_caches = [c[cache_at] for c in caches]
                cache_at += 1
                conv_in, ssm_in, slots = self._states_in(layer_caches, normed.dtype)
                mixer = layer.mixer
                conv_w, conv_b, a_log, d_skip, dt_bias = self.mamba[i]
                y, conv_rows, ssm_rows = mamba_scan(
                    mixer.in_proj(self._use_sums(normed, xs)), conv_in, ssm_in, lengths, conv_w, conv_b, a_log,
                    d_skip, dt_bias,
                    self.limits, heads=self.heads, head_dim=self.head_dim, groups=self.groups,
                    state_dim=self.state_dim, slots=slots)
                y = group_norm(y, mixer.norm.weight, self.eps, mixer.norm.group_size)
                h, normed, xs = self._add_norm(h, mixer.out_proj(y), nxt, xs)
                for c, at, n in zip(layer_caches, offsets, lengths):
                    self._hold(c, conv_rows, ssm_rows, at + n - 1)
                    c.advance(n)
                self.row_states[i] = (conv_rows, ssm_rows)
            elif kind == "*":
                layer_caches = [c[cache_at] for c in caches]
                cache_at += 1
                self._use_sums(normed, xs)
                delta = self._attention_streams(layer.mixer, normed, layer_caches, lengths, i)
                h, normed, xs = self._add_norm(h, delta, nxt, xs)
            else:
                block = self._block(i, "E", nxt)
                h, normed, xs = block(normed, xs, h)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return self._use_sums(normed.reshape(1, rows, -1), xs)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: tuple[int, ...], keeps: tuple[int, ...]) -> None:
        """After ``run_streams``: stream i keeps the first ``keeps[i]`` of its ``lengths[i]`` rows."""

        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        cache_at = 0
        for i, layer in enumerate(self.layers):
            if layer.block_type not in "M*":
                continue
            for cache, at, n, keep in zip(caches, offsets, lengths, keeps):
                if keep == n:
                    continue
                c = cache[cache_at]
                if layer.block_type == "M":
                    conv_rows, ssm_rows = self.row_states[i]
                    self._hold(c, conv_rows, ssm_rows, at + keep - 1)
                else:
                    c.trim(n - keep)
            cache_at += 1

    @staticmethod
    def _hold(cache: Any, conv_rows: mx.array, ssm_rows: mx.array, row: int) -> None:
        """The layer's states after row ``row`` of a call: kept a row of the call's states when the cache can."""

        point = getattr(cache, "point", None)
        if point is not None:
            point(conv_rows, ssm_rows, row)
        else:
            cache[0], cache[1] = conv_rows[row:row + 1], ssm_rows[row:row + 1]

    def _states_in(self, caches: list[Any], dtype: Any) -> tuple[mx.array, mx.array, tuple[int, ...] | None]:
        """Every stream's (conv, SSM) state for one scan: rows of one earlier call's states when every stream's
        still is (read through slots, no copy), else the states stacked."""

        refs = [getattr(c, "ref", None) for c in caches]
        if refs[0] is not None and all(r is not None and r[1] is refs[0][1] for r in refs):
            return refs[0][0], refs[0][1], tuple(r[2] for r in refs)
        states = [self._mamba_states(c, dtype) for c in caches]
        return mx.concatenate([s[0] for s in states]), mx.concatenate([s[1] for s in states]), None

    def _add_norm(self, h: mx.array, delta: mx.array, weight: mx.array, xs: mx.array
                  ) -> tuple[mx.array, mx.array, mx.array]:
        """add_norm, and the output's group sums with the lane matmul (else ``xs`` passes through)."""

        if not self.lane_xs:
            return (*add_norm(h, delta, weight, self.eps), xs)
        return add_norm(h, delta, weight, self.eps, group_sums=True)

    def _use_sums(self, x: mx.array, xs: mx.array) -> mx.array:
        """``x``, its group sums handed to the lane matmul's next call on it (when ``xs`` holds them: not
        ``_no_xs``). Inside a compiled block this is decided per trace: xs has its own shape there."""

        if self.lane_xs and xs.ndim == 2:
            from tensorfold.kernels.qwen.dense.v1 import lane_glue

            lane_glue.remember(x, xs)
        return x

    def _mamba_states(self, cache: Any, dtype: Any) -> tuple[mx.array, mx.array]:
        conv_state, ssm_state = cache[0], cache[1]
        if conv_state is None:
            conv_state = mx.zeros((1, 3, self.mamba_conv_dim), dtype=dtype)
        if ssm_state is None:
            ssm_state = mx.zeros((1, self.heads, self.head_dim, self.state_dim), dtype=mx.float32)
        return conv_state, ssm_state

    def _block(self, index: int, kind: str, nxt: mx.array) -> Any:
        """The layer's work between its input norm and the next layer's, compiled (a traced graph per row
        count replaces ~8-12 Python-built ops."""

        fn = self._compiled_blocks.get(index)
        if fn is None:
            fn = mx.compile(self._mamba_block(index, nxt) if kind == "M" else self._moe_block(index, nxt))
            self._compiled_blocks[index] = fn
        return fn

    def _mamba_block(self, index: int, nxt: mx.array) -> Any:
        mixer = self.layers[index].mixer
        conv_w, conv_b, a_log, d_skip, dt_bias = self.mamba[index]

        def block(x: mx.array, xs: mx.array, h: mx.array, conv_state: mx.array, ssm_state: mx.array
                  ) -> tuple[mx.array, ...]:
            proj = mixer.in_proj(self._use_sums(x, xs))
            y, conv_rows, ssm_rows = mamba_step(proj, conv_state, ssm_state, conv_w, conv_b, a_log, d_skip,
                                                dt_bias, self.limits, heads=self.heads, head_dim=self.head_dim,
                                                groups=self.groups, state_dim=self.state_dim)
            y = group_norm(y, mixer.norm.weight, self.eps, mixer.norm.group_size)
            hn, xn, xsn = self._add_norm(h, mixer.out_proj(y), nxt, xs)
            return hn, xn, xsn, conv_rows, ssm_rows

        return block

    def _moe_block(self, index: int, nxt: mx.array) -> Any:
        mixer = self.layers[index].mixer

        def block(x: mx.array, xs: mx.array, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
            routed, weights, shared = self._moe(index, mixer, self._use_sums(x, xs))
            if not self.lane_xs:
                return (*add_norm_moe(h, routed, weights, shared, nxt, self.eps), xs)
            return add_norm_moe(h, routed, weights, shared, nxt, self.eps, group_sums=True)

        return block

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call on ``rows`` rows, make ``cache`` hold only its first ``keep`` rows."""

        if keep == rows:
            return
        drop = rows - keep
        cache_at = 0
        for i, layer in enumerate(self.layers):
            if layer.block_type not in "M*":
                continue
            c = cache[cache_at]
            cache_at += 1
            if layer.block_type == "M":
                conv_rows, ssm_rows = self.row_states[i]
                self._hold(c, conv_rows, ssm_rows, keep - 1)
            else:
                c.trim(drop)

    def _attention(self, mixer: Any, x: mx.array, cache: Any, index: int | None = None) -> mx.array:
        return self._attention_streams(mixer, x, [cache], (int(x.shape[0]),), index)

    def _attention_streams(self, mixer: Any, x: mx.array, caches: list[Any], lengths: tuple[int, ...],
                           index: int | None = None) -> mx.array:
        """q/k/v for all rows, then each stream's rows against its own cache (``lengths``: rows a stream)."""

        rows = x.shape[0]
        if index in self.qkv:
            stacked, cuts = self.qkv[index]
            q, k, v = mx.split(stacked(x), cuts, axis=-1)
        else:
            q, k, v = mixer.q_proj(x), mixer.k_proj(x), mixer.v_proj(x)
        q = q.reshape(1, rows, mixer.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(1, rows, mixer.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(1, rows, mixer.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if len(lengths) == 1:
            out = self._attend_stream(mixer, q, k, v, caches[0], rows)
        else:
            outs, at = [], 0
            for cache, n in zip(caches, lengths):
                outs.append(self._attend_stream(mixer, q[:, :, at:at + n], k[:, :, at:at + n], v[:, :, at:at + n],
                                                cache, n))
                at += n
            out = mx.concatenate(outs, axis=2)
        return mixer.o_proj(out.transpose(0, 2, 1, 3).reshape(rows, -1))

    def _attend_stream(self, mixer: Any, q: mx.array, k: mx.array, v: mx.array, cache: Any, rows: int) -> mx.array:
        keys, values = cache.update_and_fetch(k, v)
        # the kernel is chosen by each row's own key count, so a row gets the bits serial decoding gives it:
        # a window straddling the switch attends row by row
        first = cache.offset - rows + 1
        lane_rows = [self.lane_attention and first + r >= self.lane_attention_from for r in range(rows)]
        if rows > 1 and not all(lane_rows):
            # row by row (each with its own keys): MLX's attention picks its kernel by query count, and a row
            # must get serial decoding's bits; the lane kernel's rows are independent by construction
            outs = [self._attend(q[:, :, r:r + 1], keys[:, :, :first + r], values[:, :, :first + r], mixer.scale,
                                 lane_rows[r], 1) for r in range(rows)]
            return mx.concatenate(outs, axis=2)
        return self._attend(q, keys, values, mixer.scale, lane_rows[0], rows)

    @staticmethod
    def _attend(q: mx.array, keys: mx.array, values: mx.array, scale: float, lane: bool, rows: int) -> mx.array:
        if lane:
            # the 16 query heads of a KV head are one 16-row tile of the tensor-unit kernel: each key read once
            # (its extra launches cost more than it saves below ``lane_attention_from`` keys)
            from tensorfold.kernels.qwen.dense.v1.lane_attention import lane_sdpa

            return lane_sdpa(q, keys, values, scale)
        return mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask="causal" if rows > 1 else None)

    def _moe(self, index: int, mixer: Any, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # our own router matvec, row-invariant by construction: MLX's bf16 matmul sums 2 rows in another order
        # than 1; then the row-exact expert kernels
        logits = router_logits(x, mixer.gate.weight)
        experts, weights = route(logits, self.gate_bias[index], self.top_k, self.scaling)
        return row_kernels.experts(mixer.switch_mlp, x, experts), weights, mixer.shared_experts(x)
