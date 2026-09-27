"""Gemma 4 decode glue in a few kernels (Gemma 4 26B-A4B).

mlx_lm's one-token step runs ~40 small kernels a layer (1,200 a token over 30 layers) around the matmuls. Here
each layer is:

    qkv          one quantized matmul for the q, k (and v) projections, which read the same input
    qkv_norm     q_norm, k_norm and v_norm of every head in one kernel
    rope, cache, attention, o_proj        mlx_lm's own
    attn_tail    post-attention norm + residual add + the three norms that read the result
                 (dense MLP input, experts input, router input)
    gate_up      one quantized matmul for the dense MLP's gate and up projections, GeGLU, down
    route        router logits -> top 8 of 128 experts, softmax over the 8, per-expert scale
    expert_gateup   the 8 experts' gate and up projections + GeGLU, one pass over the input (MLX's gathered
                 matmuls read it once a projection and launch the activation apart)
    expert_down  the 8 experts' down projections + their weighted sum
    moe_tail     post-feedforward norms 1, 2 and the joint one + residual add + layer scalar + the next layer's
                 input norm (the final norm after the last layer)

The arithmetic follows mlx_lm's (fp32 sums, bf16 wherever mlx_lm stores bf16) but the sum orders are these
kernels' own, so the fused step is its own reference. Measured against the same 4-bit weights run with fp32
activations (tools/gemma4_truth_eval.py), it lands as close as mlx_lm's own bf16 decode.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

_kernels: dict[str, Any] = {}

# one simdgroup per (row, head): lane l owns elements l, l + 32, ...
_QKV_NORM = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint slot = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int PER = DH / 32;
  int src;
  if (slot < NQ) src = int(slot) * DH;
  else if (slot < NQ + NK) src = NQ * DH + (int(slot) - NQ) * DH;
  else src = (VK ? NQ * DH : (NQ + NK) * DH) + (int(slot) - NQ - NK) * DH;   // VK: values are the raw keys
  float xv[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) {
    xv[i] = float(QKV[int(r) * W + src + int(lane) + 32 * i]);
    ss = fma(xv[i], xv[i], ss);
  }
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(DH) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(lane) + 32 * i;
    const bfloat n = bfloat(xv[i] * inv);
    if (slot < NQ) Q[(int(r) * NQ + int(slot)) * DH + c] = bfloat(float(QW[c]) * float(n));
    else if (slot < NQ + NK) K[(int(r) * NK + int(slot) - NQ) * DH + c] = bfloat(float(KW[c]) * float(n));
    else V[(int(r) * NK + int(slot) - NQ - NK) * DH + c] = n;
  }
"""

# sum of squares over the threadgroup's T threads into `total` (P: a threadgroup array of T / 32 floats)
_REDUCE = r"""
  {
    float s = simd_sum(ACC);
    if (thread_index_in_simdgroup == 0) P[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    OUTV = 0.0f;
    for (int q = 0; q < T / 32; q++) OUTV += P[q];
  }
"""


def _reduce(acc: str, partial: str, out: str) -> str:
    return _REDUCE.replace("ACC", acc).replace("P[", f"{partial}[").replace("OUTV", out)


_ATTN_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float ov[PER], hin[PER], wa[PER], w1[PER], w2[PER], w3[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    ov[i] = float(O[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wa[i] = float(WA[c]); w1[i] = float(W1[c]); w2[i] = float(W2[c]); w3[i] = float(W3[c]);
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(ov[i], ov[i], ss);
  float total1;
  REDUCE1
  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float a = float(bfloat(wa[i] * float(bfloat(ov[i] * inv1))));
    const bfloat hn = bfloat(hin[i] + a);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  float total2;
  REDUCE2
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float n = float(bfloat(hv[i] * inv2));
    N1[int(r) * D + c] = bfloat(w1[i] * n);
    N2[int(r) * D + c] = bfloat(w2[i] * n);
    N3[int(r) * D + c] = bfloat(w3[i] * n);
  }
""".replace("REDUCE1", _reduce("ss", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2"))

# one simdgroup per row: lane l holds experts l, l + 32, ...; top K by score (ties to the lower id), softmax over
# the K scores, times the expert's scale, both rounded to bf16 as mlx_lm's bf16 ops round them
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sc[NE / 32];
  for (int j = 0; j < NE / 32; j++) sc[j] = float(G[int(r) * NE + int(lane) + 32 * j]);
  float picked[K];
  int ids[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      if (sc[j] > best) { best = sc[j]; best_e = int(lane) + 32 * j; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) sc[j] = -INFINITY;
    }
    picked[k] = top;
    ids[k] = winner;
  }
  if (lane == 0) {
    float total = 0.0f;
    for (int k = 0; k < K; k++) total += metal::exp(picked[k] - picked[0]);
    for (int k = 0; k < K; k++) {
      const float p = float(bfloat(metal::exp(picked[k] - picked[0]) / total));
      IDX[int(r) * K + k] = uint(ids[k]);
      WT[int(r) * K + k] = bfloat(p * float(PES[ids[k]]));
    }
  }
"""

_MOE_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32], p3[T / 32], p4[T / 32];
  // every load before the first reduction, as in attn_tail
  float y1v[PER], h2v[PER], hin[PER], w1[PER], w2[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    y1v[i] = float(Y1[int(r) * D + c]); h2v[i] = float(Y2[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    w1[i] = float(W1[c]); w2[i] = float(W2[c]); wp[i] = float(WP[c]); wn[i] = float(WN[c]);
  }
  const float sc = float(SC[0]);
  float ss1 = 0.0f, ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    ss1 = fma(y1v[i], y1v[i], ss1);
    ss2 = fma(h2v[i], h2v[i], ss2);
  }
  float total1, total2;
  REDUCE1
  REDUCE2
  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  float sv[PER];
  float ss3 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const float a1 = float(bfloat(w1[i] * float(bfloat(y1v[i] * inv1))));
    const float a2 = float(bfloat(w2[i] * float(bfloat(h2v[i] * inv2))));
    sv[i] = float(bfloat(a1 + a2));
    ss3 = fma(sv[i], sv[i], ss3);
  }
  float total3;
  REDUCE3
  const float inv3 = metal::precise::rsqrt(total3 / float(D) + eps[0]);
  float hv[PER];
  float ss4 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float b = float(bfloat(wp[i] * float(bfloat(sv[i] * inv3))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss4 = fma(hv[i], hv[i], ss4);
  }
  float total4;
  REDUCE4
  const float inv4 = metal::precise::rsqrt(total4 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    NEXT[int(r) * D + c] = bfloat(wn[i] * float(bfloat(hv[i] * inv4)));
  }
""".replace("REDUCE1", _reduce("ss1", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2")) \
   .replace("REDUCE3", _reduce("ss3", "p3", "total3")).replace("REDUCE4", _reduce("ss4", "p4", "total4"))


# MLX's 4-bit qmv inner loop (as in the Flash Next kernels): 16 inputs a lane, pre-divided by 1, 16, 256, 4096 so
# the masked nibbles need no shift; w = scale * q + bias gives scale * dot(q, x) + bias * sum(x)
_QDOT_HEADER = r"""
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
// mlx.nn.gelu_approx in fp32
inline float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + metal::precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
"""

# Threadgroup (b, p): SG simdgroups, rows RPS g .. of the gate and up projections of slot p (p = row * TOPK + k:
# the row's k-th expert from route), over K in steps of 512 (the last step partial when K % 512 != 0); then
# bf16(gelu(bf16(gate)) * bf16(up)). Quantization groups of GS inputs (a 16-input chunk c uses group c / (GS / 16)).
_EXPERT_GATEUP = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const int r = p / TOPK;
  const size_t e = size_t(IDX[p]);
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* gw = (const device uint8_t*)GW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UW + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* gb = GBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* us = USC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* ub = UBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    if (k0 + int(lane) * 16 < K) {
      const float sum = load16(x, xt);
      for (int row = 0; row < RPS; row++) {
        ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
        au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
      }
    }
    gw += 256; uw += 256; gs += 512 / GS; gb += 512 / GS; us += 512 / GS; ub += 512 / GS; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(gelu_tanh(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

# Threadgroup (b, r): TOPK simdgroups, simdgroup k takes the row's k-th expert for model dims 8 b .. 8 b + 7: lane l
# reads 16-input chunks l and (l < NC - 32) 32 + l of its activation; y_k = bf16(sum). Then
# out = bf16(sum_k bf16(w_k y_k)) (fp32 sum, slots in order): mlx_lm's (w * y).sum(-2) on bf16.
_EXPERT_DOWN = r"""
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.z);
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / GS;
  constexpr int NC = NI / 16;
  threadgroup float ys[TOPK][8];
  const size_t e = size_t(IDX[r * TOPK + k]);
  const device bfloat* x = ACT + (r * TOPK + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  // the 8 rows' loads before the first simd_sum, so their reads are in flight together
  float acc[8];
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DW + at * KB;
    acc[row] = qdot16(w + lane * 8, xa, float(DSC[at * KG + lane / (GS / 16)]), float(DBI[at * KG + lane / (GS / 16)]), sa);
  }
  if (second) {
    #pragma unroll
    for (int row = 0; row < 8; row++) {
      const size_t at = e * D + d0 + row;
      const device uint8_t* w = (const device uint8_t*)DW + at * KB;
      acc[row] += qdot16(w + (32 + lane) * 8, xb, float(DSC[at * KG + (32 + lane) / (GS / 16)]),
                         float(DBI[at * KG + (32 + lane) / (GS / 16)]), sb);
    }
  }
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const float s = simd_sum(acc[row]);
    if (lane == 0) ys[k][row] = float(bfloat(s));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < 8) {
    float routed = 0.0f;
    for (int kk = 0; kk < TOPK; kk++) routed += float(bfloat(ys[kk][lane] * float(WT[r * TOPK + kk])));
    OUT[r * D + d0 + int(lane)] = bfloat(routed);
  }
"""


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str], header: str = "") -> Any:
    key = f"{name}_{hashlib.sha256((header + source).encode()).hexdigest()[:16]}"
    kernel = _kernels.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(name=key, input_names=inputs, output_names=outputs, source=source,
                                      header=header)
        _kernels[key] = kernel
    return kernel


def _check_q4(linear: Any) -> None:
    if getattr(linear, "bits", None) != 4 or getattr(linear, "group_size", None) not in (32, 64, 128):
        raise ValueError("the Gemma expert kernels read 4-bit weights in groups of 32, 64 or 128")


def expert_gateup(x: mx.array, ids: mx.array, gate: Any, up: Any, *, simdgroups: int = 2,
                  rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K], expert ids [R, TOPK] -> bf16(gelu(x W_gate^T) * (x W_up^T)) for every (row, slot): [R * TOPK, N]."""

    _check_q4(gate)
    rows, dims = x.shape
    top_k = ids.shape[-1]
    width = gate.weight.shape[1]
    per_tg = simdgroups * rows_per_simdgroup
    if width % per_tg or dims % 16:
        raise ValueError("expert_gateup: expert width must split into threadgroups, inputs into chunks of 16")
    kernel = _kernel("gemma_expert_gateup", _EXPERT_GATEUP,
                     ["X", "IDX", "GW", "GSC", "GBI", "UW", "USC", "UBI"], ["ACT"], header=_QDOT_HEADER)
    return kernel(inputs=[x, ids, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases],
                  template=[("K", dims), ("N", width), ("TOPK", top_k), ("GS", gate.group_size), ("SG", simdgroups),
                            ("RPS", rows_per_simdgroup)],
                  grid=(32 * simdgroups, width // per_tg, rows * top_k), threadgroup=(32 * simdgroups, 1, 1),
                  output_shapes=[(rows * top_k, width)], output_dtypes=[mx.bfloat16])[0]


def expert_down(act: mx.array, ids: mx.array, weights: mx.array, down: Any) -> mx.array:
    """act [R * TOPK, NI] -> bf16(sum_k w_k * (act_k W_down^T)): [R, D]."""

    _check_q4(down)
    rows, top_k = ids.shape
    inner = act.shape[-1]
    dims = down.weight.shape[1]
    if inner % 16 or not 32 <= inner // 16 <= 64 or dims % 8:
        raise ValueError("expert_down: the expert width must be 512 to 1,024 in chunks of 16")
    kernel = _kernel("gemma_expert_down", _EXPERT_DOWN, ["ACT", "IDX", "WT", "DW", "DSC", "DBI"], ["OUT"],
                     header=_QDOT_HEADER)
    return kernel(inputs=[act, ids, weights, down.weight, down.scales, down.biases],
                  template=[("NI", inner), ("D", dims), ("TOPK", top_k), ("GS", down.group_size)],
                  grid=(32 * top_k, dims // 8, rows), threadgroup=(32 * top_k, 1, 1),
                  output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


def _threads(dims: int) -> int:
    for threads in (256, 128, 64, 32):
        if dims % threads == 0:
            return threads
    raise ValueError(f"hidden size {dims} does not split into threadgroups of 32")


def qkv_norm(qkv: mx.array, q_w: mx.array, k_w: mx.array, eps: mx.array, *, heads: int, kv_heads: int,
             head_dim: int, values_are_keys: bool) -> tuple[mx.array, mx.array, mx.array]:
    """From the stacked projection [R, W]: q_norm(q) [R, H, Dh], k_norm(k) [R, Hk, Dh] and the scale-free
    v_norm of v (or of the raw keys when the layer's values are its keys) [R, Hk, Dh], all bf16."""

    rows, width = qkv.shape
    if head_dim % 32:
        raise ValueError("qkv_norm: head_dim must be a multiple of 32")
    kernel = _kernel("gemma_qkv_norm", _QKV_NORM, ["QKV", "QW", "KW", "eps"], ["Q", "K", "V"])
    return kernel(inputs=[qkv, q_w, k_w, eps],
                  template=[("DH", head_dim), ("NQ", heads), ("NK", kv_heads), ("W", width),
                            ("VK", int(values_are_keys))],
                  grid=(32 * (heads + 2 * kv_heads), rows, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, heads, head_dim), (rows, kv_heads, head_dim), (rows, kv_heads, head_dim)],
                  output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16])


def attn_tail(h: mx.array, o: mx.array, w_attn: mx.array, w1: mx.array, w2: mx.array, w3: mx.array,
              eps: mx.array) -> tuple[mx.array, ...]:
    """hn = h + RMSNorm(o) * w_attn, then RMSNorm(hn) times w1, w2 and w3: (hn, n1, n2, n3), [R, D] bf16."""

    rows, dims = h.shape
    threads = _threads(dims)
    kernel = _kernel("gemma_attn_tail", _ATTN_TAIL, ["H", "O", "WA", "W1", "W2", "W3", "eps"],
                     ["HN", "N1", "N2", "N3"])
    return kernel(inputs=[h, o, w_attn, w1, w2, w3, eps], template=[("D", dims), ("T", threads)],
                  grid=(threads * rows, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, dims)] * 4, output_dtypes=[mx.bfloat16] * 4)


def route(scores: mx.array, expert_scale: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    """Router scores [R, E] -> expert ids [R, K] (best first) and weights [R, K] bf16."""

    rows, experts = scores.shape
    if experts % 32:
        raise ValueError("route: the expert count must be a multiple of 32")
    kernel = _kernel("gemma_route", _ROUTE, ["G", "PES"], ["IDX", "WT"])
    return kernel(inputs=[scores, expert_scale], template=[("NE", experts), ("K", top_k)],
                  grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, top_k), (rows, top_k)], output_dtypes=[mx.uint32, mx.bfloat16])


def moe_tail(h: mx.array, y1: mx.array, y2: mx.array, w1: mx.array, w2: mx.array, w_post: mx.array,
             scalar: mx.array, w_next: mx.array, eps: mx.array) -> tuple[mx.array, mx.array]:
    """hn = (h + RMSNorm(RMSNorm(y1) w1 + RMSNorm(y2) w2) w_post) * scalar and RMSNorm(hn) * w_next, for the dense
    MLP's y1 and the experts' weighted sum y2, [R, D]: (hn, next) [R, D] bf16."""

    rows, dims = h.shape
    threads = _threads(dims)
    kernel = _kernel("gemma_moe_tail", _MOE_TAIL, ["H", "Y1", "Y2", "W1", "W2", "WP", "SC", "WN", "eps"],
                     ["HN", "NEXT"])
    return kernel(inputs=[h, y1, y2, w1, w2, w_post, scalar, w_next, eps],
                  template=[("D", dims), ("T", threads)],
                  grid=(threads * rows, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, dims), (rows, dims)], output_dtypes=[mx.bfloat16, mx.bfloat16])


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


class FusedDecode:
    """Gemma 4 one-row decode through the kernels above and MLX's matmuls, attention and RoPE."""

    # layers per slice handed to the GPU while the rest of the step is built (0: the caller evaluates)
    eval_every = 8

    def __init__(self, text_model: Any) -> None:
        args = text_model.args
        if getattr(args, "hidden_size_per_layer_input", 0) or getattr(args, "num_kv_shared_layers", 0):
            raise ValueError("FusedDecode covers Gemma 4 without per-layer inputs or shared KV layers (26B-A4B)")
        if not getattr(args, "enable_moe_block", False):
            raise ValueError("FusedDecode covers Gemma 4's MoE layers (26B-A4B)")
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            raise ValueError("the fused kernels run on the GPU")
        self.backbone = text_model.model
        self.layers = self.backbone.layers
        for layer in self.layers:
            attn, experts = layer.self_attn, layer.experts.switch_glu
            linears = [attn.q_proj, attn.k_proj, layer.mlp.gate_proj, layer.mlp.up_proj, experts.gate_proj,
                       experts.up_proj, experts.down_proj]
            if any(getattr(l, "bits", None) != 4 or getattr(l, "group_size", None) not in (32, 64, 128)
                   for l in linears):
                raise ValueError("the fused kernels read 4-bit weights in groups of 32, 64 or 128")
            inner = experts.down_proj.weight.shape[-1] * 8
            if attn.head_dim % 32 or args.hidden_size % 32 or inner % 16 or not 32 <= inner // 16 <= 64:
                raise ValueError("the fused kernels need head dims in 32s, hidden size in 32s and an expert "
                                 "width of 512 to 1,024")
        self.eps = mx.array([float(args.rms_norm_eps)], dtype=mx.float32)
        self.eps_value = float(args.rms_norm_eps)
        self.top_k = int(args.top_k_experts)
        self.qkv: dict[int, tuple[Any, list[int]]] = {}
        self.gate_up: dict[int, tuple[Any, list[int]]] = {}
        self.router_norm: dict[int, mx.array] = {}
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            projs = [attn.q_proj, attn.k_proj] + ([] if attn.use_k_eq_v else [attn.v_proj])
            self.qkv[i] = _stack_linears(projs)
            self.gate_up[i] = _stack_linears([layer.mlp.gate_proj, layer.mlp.up_proj])
            # mlx_lm normalizes the router input with weight scale * hidden**-0.5, computed in the scale's dtype
            self.router_norm[i] = layer.router.scale * layer.router._root_size
        mx.eval(list(self.router_norm.values()))
        self._fronts: dict[int, Any] = {}
        self._backs: dict[int, Any] = {}

    def __call__(self, inputs: mx.array, cache: list[Any]) -> mx.array:
        """Hidden states after the final norm, [1, 1, D], for one token (batch 1)."""

        from mlx_lm.models.gemma4_text import scaled_dot_product_attention

        tokens = inputs.reshape(-1)
        rows = tokens.shape[0]
        if rows != 1:
            raise ValueError("FusedDecode takes one row a call")
        h = self.backbone.embed_tokens(tokens) * self.backbone.embed_scale          # [1, D]
        normed = mx.fast.rms_norm(h, self.layers[0].input_layernorm.weight, self.eps_value)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            c = cache[i]
            q, k, v = self._front(i)(normed)
            q = q.reshape(1, rows, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(1, rows, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(1, rows, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            offset = mx.array(c.offset)
            k = attn.rope(k, offset=offset)
            q = attn.rope(q, offset=offset)
            keys, values = c.update_and_fetch(k, v)
            out = scaled_dot_product_attention(q, keys, values, cache=c, scale=attn.scale, mask=None)
            out = out.transpose(0, 2, 1, 3).reshape(rows, -1)
            h, normed = self._back(i)(out, h)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return normed.reshape(1, rows, -1)

    def _front(self, index: int) -> Any:
        """The layer's q, k, v projections and head norms, compiled."""

        fn = self._fronts.get(index)
        if fn is None:
            attn = self.layers[index].self_attn
            stacked, _ = self.qkv[index]

            def front(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                return qkv_norm(stacked(x), attn.q_norm.weight, attn.k_norm.weight, self.eps, heads=attn.n_heads,
                                kv_heads=attn.n_kv_heads, head_dim=attn.head_dim, values_are_keys=attn.use_k_eq_v)

            fn = mx.compile(front)
            self._fronts[index] = fn
        return fn

    def _back(self, index: int) -> Any:
        """The layer from the attention output to the next layer's normed input, compiled."""

        fn = self._backs.get(index)
        if fn is None:
            from mlx_lm.models.gemma4_text import geglu

            layer = self.layers[index]
            nxt = (self.layers[index + 1].input_layernorm.weight if index + 1 < len(self.layers)
                   else self.backbone.norm.weight)
            gate_up, cuts = self.gate_up[index]
            router_w = self.router_norm[index]

            def back(out: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
                o = layer.self_attn.o_proj(out)
                hn, n_mlp, n_exp, n_router = attn_tail(h, o, layer.post_attention_layernorm.weight,
                                                       layer.pre_feedforward_layernorm.weight,
                                                       layer.pre_feedforward_layernorm_2.weight, router_w, self.eps)
                gate, up = mx.split(gate_up(n_mlp), cuts, axis=-1)
                y1 = layer.mlp.down_proj(geglu(gate, up))
                ids, weights = route(layer.router.proj(n_router), layer.router.per_expert_scale, self.top_k)
                experts = layer.experts.switch_glu
                act = expert_gateup(n_exp, ids, experts.gate_proj, experts.up_proj)
                y2 = expert_down(act, ids, weights, experts.down_proj)
                return moe_tail(hn, y1, y2, layer.post_feedforward_layernorm_1.weight,
                                layer.post_feedforward_layernorm_2.weight, layer.post_feedforward_layernorm.weight,
                                layer.layer_scalar, nxt, self.eps)

            fn = mx.compile(back)
            self._backs[index] = fn
        return fn
