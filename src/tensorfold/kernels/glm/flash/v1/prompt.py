"""A prompt chunk's KDA glue and scan, SwiGLU and MoE combine in few kernels, each with its MLX ops' bits."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1.fused import SQ_FMA, _kernel, metal

# simdgroup (row, head); lane l < D / 4 holds dims 4l..4l+3, as MLX's rms_norm reads a row of D <= 128 (4 a thread)
_KDA_PRE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint h = threadgroup_position_in_grid.y * HPT + simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const bool on = 4u * lane < uint(D);
  constexpr uint W = uint(H * D);
  constexpr uint C3 = 3u * W;
  const uint PS = uint(P_shape[1]);
  // the causal conv over [window ; rows]: fp32 products in tap order, each rounded before its add; bf16, then silu
  float qk[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
  for (int part = 0; part < 3 && on; ++part) {
    for (int i = 0; i < 4; ++i) {
      const uint c = uint(part) * W + h * uint(D) + 4u * lane + uint(i);
      float acc = 0.0f;
      for (int j = 0; j < TAPS; ++j) {
        const int e = r + j;
        const bfloat xv = e < TAPS - 1 ? CS[size_t(e) * C3 + c] : P[size_t(e - (TAPS - 1)) * PS + c];
        acc = j == 0 ? float(xv) * CW[size_t(j) * C3 + c] : mul_add(acc, float(xv), CW[size_t(j) * C3 + c]);
      }
      const bfloat xb = bfloat(acc);
      const bfloat sl = xb * sigmoid_precise(xb);
      if (part == 2) V[(size_t(r) * H + h) * D + 4u * lane + uint(i)] = sl;
      else qk[part][i] = float(sl);
    }
  }
  // l2 norms as rms_norm over the head (eps / D), then q * 1/D and k * D^-1/2, to bf16
  float sq = 0.0f, sk = 0.0f;
  for (int i = 0; i < 4 && on; ++i) { sq = sq_acc<SQ_FMA>(sq, qk[0][i]); sk = sq_acc<SQ_FMA>(sk, qk[1][i]); }
  sq = simd_sum(sq);
  sk = simd_sum(sk);
  const float iq = metal::precise::rsqrt(sq / float(D) + NEPS[0]);
  const float ik = metal::precise::rsqrt(sk / float(D) + NEPS[0]);
  for (int i = 0; i < 4 && on; ++i) {
    const size_t o = (size_t(r) * H + h) * D + 4u * lane + uint(i);
    Q[o] = bfloat((qk[0][i] * iq) * SCL[0]);
    K[o] = bfloat((qk[1][i] * ik) * SCL[1]);
    // the decays exp(lb * sigmoid(A * (a + dt_bias))) in fp32
    const uint d = h * uint(D) + 4u * lane + uint(i);
    const float av = add_nc(float(AIN[size_t(r) * W + d]), DTB[d]);
    G[o] = metal::precise::exp(LB[0] * sigmoid_precise(AH[h] * av));
  }
  if (lane == 0u) BETA[size_t(r) * H + h] = sigmoid_precise(P[size_t(r) * PS + C3 + 2u * uint(D) + h]);
"""

# rms_norm of a head's scan output with o_norm (fp32), times sigmoid(gate), to bf16
_KDA_POST = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint h = threadgroup_position_in_grid.y * HPT + simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const bool on = 4u * lane < uint(D);
  constexpr uint W = uint(H * D);
  float yv[4];
  float s = 0.0f;
  for (int i = 0; i < 4 && on; ++i) {
    yv[i] = float(Y[(size_t(r) * H + h) * D + 4u * lane + uint(i)]);
    s = sq_acc<SQ_FMA>(s, yv[i]);
  }
  s = simd_sum(s);
  const float inv = metal::precise::rsqrt(s / float(D) + EPS[0]);
  for (int i = 0; i < 4 && on; ++i) {
    const uint d = 4u * lane + uint(i);
    const size_t o = size_t(r) * W + h * uint(D) + d;
    OUT[o] = bfloat((ONW[d] * (yv[i] * inv)) * sigmoid_precise(float(GATE[o])));
  }
"""

# mlx-lm's gated delta kernel (vector gates), each column's expressions as there; CPT columns stage TS steps at once
_SCAN = r"""
  constexpr int NPT = Dk / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_index_in_threadgroup;
  const uint n = threadgroup_position_in_grid.z;
  const uint b_idx = n / Hv, hv_idx = n % Hv, hk_idx = hv_idx / (Hv / Hk);
  const uint dv0 = threadgroup_position_in_grid.y * CPT;
  const uint dv_idx = dv0 + sg;
  const int T = int(q_shape[1]);
  threadgroup float tq[TS][Dk];
  threadgroup float tk[TS][Dk];
  threadgroup float tg[TS][Dk];
  threadgroup float tv[TS][CPT];
  threadgroup float tb[TS];
  device const InT* q_ = q + (size_t(b_idx) * T * Hk + hk_idx) * Dk;
  device const InT* k_ = k + (size_t(b_idx) * T * Hk + hk_idx) * Dk;
  device const float* g_ = g + (size_t(b_idx) * T * Hv + hv_idx) * Dk;
  device const InT* v_ = v + (size_t(b_idx) * T * Hv + hv_idx) * Dv;
  device const InT* beta_ = beta + size_t(b_idx) * T * Hv + hv_idx;
  device InT* y_ = y + (size_t(b_idx) * T * Hv + hv_idx) * Dv + dv_idx;
  device const float* i_state = state_in + (size_t(n) * Dv + dv_idx) * Dk;
  device float* o_state = state_out + (size_t(n) * Dv + dv_idx) * Dk;
  float state[NPT];
  for (int i = 0; i < NPT; ++i) state[i] = static_cast<float>(i_state[NPT * lane + i]);
  for (int t0 = 0; t0 < T; t0 += TS) {
    const int ts = min(TS, T - t0);
    for (uint e = tid; e < uint(ts * Dk); e += CPT * 32) {
      const uint tt = e / Dk, d = e % Dk;
      tq[tt][d] = float(q_[size_t(t0 + tt) * Hk * Dk + d]);
      tk[tt][d] = float(k_[size_t(t0 + tt) * Hk * Dk + d]);
      tg[tt][d] = g_[size_t(t0 + tt) * Hv * Dk + d];
    }
    for (uint e = tid; e < uint(ts * CPT); e += CPT * 32) {
      tv[e / CPT][e % CPT] = float(v_[size_t(t0 + e / CPT) * Hv * Dv + dv0 + e % CPT]);
    }
    if (tid < uint(ts)) tb[tid] = float(beta_[size_t(t0 + tid) * Hv]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int tt = 0; tt < ts; ++tt) {
      float kv_mem = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        auto s_idx = NPT * lane + i;
        state[i] = state[i] * tg[tt][s_idx];
        kv_mem += state[i] * tk[tt][s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (tv[tt][sg] - kv_mem) * tb[tt];
      float out = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        auto s_idx = NPT * lane + i;
        state[i] = state[i] + tk[tt][s_idx] * delta;
        out += state[i] * tq[tt][s_idx];
      }
      out = simd_sum(out);
      if (lane == 0) y_[size_t(t0 + tt) * Hv * Dv] = static_cast<InT>(out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < NPT; ++i) o_state[NPT * lane + i] = static_cast<float>(state[i]);
"""

SCAN_STEPS, SCAN_COLUMNS = 16, 32         # steps a threadgroup loads at once; value columns (simdgroups) it runs


# DSA index scores sum_h w_h relu(q_h . pool): MLX's steel GEMM, bf16 scores, relu, weight product, then MLX's head sum
_INDEX = r"""
  using namespace mlx::steel;
  constexpr int BM = 64, BN = 64, BK = 16, WM = 1, WN = 2;
  using gemm_kernel = GEMMKernel<bfloat16_t, bfloat16_t, BM, BN, BK, WM, WN, false, true, true, true, float>;
  using mma_t = typename gemm_kernel::mma_t;
  threadgroup bfloat16_t As[gemm_kernel::tgp_mem_size_a];
  threadgroup bfloat16_t Bs[gemm_kernel::tgp_mem_size_b];
  threadgroup bfloat16_t tile[BM * BN];
  const int M = int(Q_shape[0]) * HEADS;
  const int N = int(POOL_shape[0]);
  const int c_row = int(threadgroup_position_in_grid.y) * BM;
  const int c_col = int(threadgroup_position_in_grid.x) * BN;
  const ushort sg = ushort(simdgroup_index_in_threadgroup), lane = ushort(thread_index_in_simdgroup);
  thread mma_t mma_op(sg, lane);
  thread typename gemm_kernel::loader_a_t loader_a(Q + size_t(c_row) * DIM, DIM, As, sg, lane);
  thread typename gemm_kernel::loader_b_t loader_b(POOL + size_t(c_col) * DIM, DIM, Bs, sg, lane);
  const short tgp_bm = short(min(BM, M - c_row)), tgp_bn = short(min(BN, N - c_col)), lbk = 0;
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_kernel::gemm_loop(As, Bs, DIM / BK, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk,
                           LoopAlignment<true, true, true>{});
  } else if (tgp_bn == BN) {
    gemm_kernel::gemm_loop(As, Bs, DIM / BK, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk,
                           LoopAlignment<false, true, true>{});
  } else if (tgp_bm == BM) {
    gemm_kernel::gemm_loop(As, Bs, DIM / BK, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk,
                           LoopAlignment<true, false, true>{});
  } else {
    gemm_kernel::gemm_loop(As, Bs, DIM / BK, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk,
                           LoopAlignment<false, false, true>{});
  }
  // the scores as MLX stores them (bf16), relu as mx.maximum takes it, times the row's head weight in bf16
  const bfloat16_t zero = bfloat16_t(0.0f);
  for (short i = 0; i < mma_t::TM; i++) {
    for (short j = 0; j < mma_t::TN; j++) {
      for (short e = 0; e < 2; e++) {
        const short row = mma_op.sm + i * mma_t::TM_stride, col = mma_op.sn + j * mma_t::TN_stride + e;
        const bfloat16_t sv = static_cast<bfloat16_t>(mma_op.Ctile.frag_at(i, j)[e]);
        const bfloat16_t rl = metal::isnan(sv) ? sv : (sv > zero ? sv : zero);
        const bfloat16_t w = row < tgp_bm ? IW[size_t(c_row + row)] : zero;
        tile[row * BN + col] = w * rl;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // a thread a (query, block): the fp32 head sum as an adjacent pairwise tree (1, 2, 4, 8, 16 apart)
  const int queries = tgp_bm / HEADS;
  for (int o = int(sg) * 32 + int(lane); o < queries * BN; o += WM * WN * 32) {
    const int q = o / BN, col = o % BN;
    float v[HEADS];
    for (int h = 0; h < HEADS; ++h) v[h] = float(tile[(q * HEADS + h) * BN + col]);
    for (int w = 1; w < HEADS; w *= 2)
      for (int h = 0; h < HEADS; h += 2 * w) v[h] = v[h] + v[h + w];
    if (c_col + col < N) OUT[size_t(c_row / HEADS + q) * N + c_col + col] = static_cast<bfloat16_t>(v[0]);
  }
"""


# SwiGLU with the limit: gate row r at column c, up at column UO + c of rows GS wide (a stacked gate | up, or apart)
_SWIGLU = r"""
  const uint i = thread_position_in_grid.x;
  const uint r = i / uint(N), c = i % uint(N);
  const float lim = float(bfloat(LIM[0]));
  const bfloat gt = bfloat(metal::min(float(GATE[size_t(r) * GS + c]), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(UP[size_t(r) * GS + UO + c]), -lim), lim));
  ACT[i] = (gt * sigmoid_precise(gt)) * up;
"""

# out[r] = bf16(fp32 sum over slots of w * y, each product rounded before its add) + shared, y read through the unsort
_COMBINE = r"""
  const uint gid = thread_position_in_grid.x;
  const uint r = gid / uint(D), d = gid % uint(D);
  const device bfloat* y = Y + d;
  float acc = WTS[r * TOPK] * float(y[size_t(INV[r * TOPK]) * D]);
  for (int k = 1; k < TOPK; k++) acc = mul_add(acc, WTS[r * TOPK + k], float(y[size_t(INV[r * TOPK + k]) * D]));
  OUT[size_t(r) * D + d] = bfloat(acc) + SH[size_t(r) * D + d];
"""

def proven() -> bool:
    """Whether the prompt kernels serve this chip: Metal without tensor units (M1-M4) until an M5 run proves them."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    return metal() and not PM._tensor_units()


def kda_fits(kda: Any) -> bool:
    """The KDA kernels' shapes: heads of at most 128 dims (4 a lane), the stacked projection's order, on Metal."""

    return (metal() and kda.dim % 4 == 0 and kda.dim <= 128 and kda.cuts[2] == 3 * kda.width
            and kda.cuts[3] - kda.cuts[2] == kda.dim and kda.cuts[4] - kda.cuts[3] == kda.dim
            and kda.in_proj.outs - kda.cuts[4] == kda.heads)


def _heads(kda: Any) -> int:
    """Heads (simdgroups) a threadgroup: up to 8, dividing the head count."""

    return next(n for n in (8, 4, 2, 1) if kda.heads % n == 0)


def kda_pre(kda: Any, proj: mx.array, conv: mx.array, a: mx.array) -> tuple[mx.array, ...]:
    """A chunk's q, k, v, decays and beta from its stacked projection, conv window and f_b output."""

    rows, h, d = int(proj.shape[0]), kda.heads, kda.dim
    kernel = _kernel("kda_prompt_pre", _KDA_PRE, ["P", "CS", "CW", "AIN", "AH", "DTB", "LB", "NEPS", "SCL"],
                     ["Q", "K", "V", "G", "BETA"])
    shape = (1, rows, h, d)
    return tuple(kernel(
        inputs=[proj, conv, kda.conv_w, a, kda.A_flat, kda.dt_bias_flat, kda.lb_array, kda.l2_eps, kda.qk_scale],
        template=[("H", h), ("D", d), ("TAPS", kda.taps), ("HPT", _heads(kda)), ("SQ_FMA", SQ_FMA)],
        grid=(32 * rows, h, 1), threadgroup=(32, _heads(kda), 1),
        output_shapes=[shape, shape, shape, shape, (1, rows, h)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.bfloat16]))


def kda_post(kda: Any, y: mx.array, gate: mx.array) -> mx.array:
    """The gated RMSNorm of a chunk's scan output [1, R, H, D] with its g_b output [R, H D]: bf16 [R, H D]."""

    rows, h, d = int(y.shape[1]), kda.heads, kda.dim
    kernel = _kernel("kda_prompt_post", _KDA_POST, ["Y", "GATE", "ONW", "EPS"], ["OUT"])
    return kernel(inputs=[y, gate, kda.o_norm, kda.eps_array],
                  template=[("H", h), ("D", d), ("HPT", _heads(kda)), ("SQ_FMA", SQ_FMA)],
                  grid=(32 * rows, h, 1), threadgroup=(32, _heads(kda), 1),
                  output_shapes=[(rows, h * d)], output_dtypes=[mx.bfloat16])[0]


def swiglu(gate: mx.array, up: mx.array | None, limit: mx.array, width: int) -> mx.array:
    """SwiGLU of ``width`` columns a row: gate and up apart, or ``up`` None and ``gate`` the stacked [gate | up]."""

    rows = gate.size // int(gate.shape[-1])
    stride = int(gate.shape[-1])
    kernel = _kernel("prompt_swiglu", _SWIGLU, ["GATE", "UP", "LIM"], ["ACT"])
    out = kernel(inputs=[gate, gate if up is None else up, limit],
                 template=[("N", width), ("GS", stride), ("UO", width if up is None else 0)],
                 grid=(rows * width, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(rows, width)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*gate.shape[:-1], width)


def combine(y: mx.array, inv: mx.array, weights: mx.array, shared: mx.array) -> mx.array:
    """Routed outputs sorted by expert [R k, D] read back through ``inv``, weighted, summed, plus the shared expert."""

    rows, top = int(weights.shape[0]), int(weights.shape[1])
    dims = int(y.shape[-1])
    kernel = _kernel("prompt_moe_combine", _COMBINE, ["Y", "INV", "WTS", "SH"], ["OUT"])
    return kernel(inputs=[y.reshape(-1, dims), inv, weights, shared], template=[("D", dims), ("TOPK", top)],
                  grid=(rows * dims, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]

def scan(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
         state: mx.array) -> tuple[mx.array, mx.array]:
    """mlx-lm's gated_delta_kernel with vector gates, its bits: (y [B, T, Hv, Dv], the fp32 state after the steps)."""

    batch, _, hk, dk = k.shape
    hv, dv = int(v.shape[2]), int(v.shape[3])
    cols = next(c for c in (SCAN_COLUMNS, 16, 8, 4, 2, 1) if dv % c == 0)
    kernel = _kernel("prompt_scan", _SCAN, ["q", "k", "v", "g", "beta", "state_in"], ["y", "state_out"])
    return tuple(kernel(inputs=[q, k, v, g, beta, state],
                        template=[("InT", q.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv),
                                  ("TS", SCAN_STEPS), ("CPT", cols)],
                        grid=(32, dv, batch * hv), threadgroup=(32, cols, 1),
                        output_shapes=[(batch, int(q.shape[1]), hv, dv), tuple(state.shape)],
                        output_dtypes=[q.dtype, mx.float32]))


def scan_fits(q: mx.array, g: mx.array) -> bool:
    """The staged scan's shapes: vector gates, key dims a multiple of 32 up to 128, bf16 or fp32 inputs."""

    dk = int(q.shape[-1])
    return metal() and g.ndim == 4 and dk % 32 == 0 and dk <= 128 and q.dtype in (mx.bfloat16, mx.float32)

def index_fits(iq: mx.array, pool: mx.array) -> bool:
    """The fused index scores' shapes: 32 heads (one a lane), a head dim of 16s, bf16, M1-M4 (MLX's steel GEMM)."""

    return (proven() and iq.ndim == 3 and int(iq.shape[1]) == 32 and int(iq.shape[2]) % 16 == 0
            and iq.dtype == pool.dtype == mx.bfloat16)


def index_scores(iq: mx.array, iw: mx.array, pool: mx.array) -> mx.array:
    """sum over heads of w_h relu(q_h . pool) [n, P] for iq [n, 32, D], iw [n, 32], pool [P, D]: the three ops' bits."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    rows, heads, dim = (int(v) for v in iq.shape)
    blocks = int(pool.shape[0])
    kernel = PM._k("tf_glm5_index_scores", _INDEX, ["Q", "IW", "POOL"], ["OUT"], PM._header())
    return kernel(inputs=[iq, iw, pool], template=[("HEADS", heads), ("DIM", dim)],
                  grid=(-(-blocks // 64) * 64, -(-rows * heads // 64), 1), threadgroup=(64, 1, 1),
                  output_shapes=[(rows, blocks)], output_dtypes=[mx.bfloat16])[0]
