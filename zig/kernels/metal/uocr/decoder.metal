// Unlimited-OCR-2's decoder steps on Metal: norms, rotary, the block router, attention and the experts' glue.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint UD_H = 1280, UD_HD = 128, UD_HEADS = 10, UD_BLK = 32, UD_LOCAL = 128, UD_RD = 128;
constant constexpr uint UD_READ = 12, UD_TOP = 16, UD_EXPERTS = 64, UD_PICK = 6, UD_PROJ = 4096;
// The router's last block a query may read sits this many blocks before its own (the local window's blocks).
constant constexpr uint UD_LOCAL_BLOCKS = 4;
// Scores past these clamp, and the release's floor under a block's entmax weight.
constant constexpr float UD_CLAMP = 1.0e4f, UD_FLOOR = 1.0e-4f;
// 1 / sqrt(128) and 128 / sqrt(128) rounded to fp32: the scores' scale and the router's.
constant constexpr float UD_INV_SQRT_HD = 0.0883883476483184405f, UD_ROUTER_SCALE = 11.3137084989847604f;

inline float ud_bf(float x) { return float(bfloat(x)); }

// The release's clamp at +-1e4: NaN stays NaN.
inline float ud_clamp(float x) { return isnan(x) ? x : clamp(x, -UD_CLAMP, UD_CLAMP); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float ud_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// Gathered rows: y[i] = x[rows[i]] (1280 wide).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_gather(
    const device bfloat* x [[buffer(0)]], const device uint* rows [[buffer(1)]], device bfloat* y [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= UD_H) return;
  y[ulong(at.y) * UD_H + at.x] = x[ulong(rows[at.y]) * UD_H + at.x];
}

struct UdMv {
  uint n;  // output rows
  uint k;  // input width, a multiple of 8
};

// One token's projection y[n] = bf16(W[n, :] . x) with fp32 sums: 4 rows a simdgroup, 8 simdgroups a group.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_matvec(
    const device bfloat* x [[buffer(0)]], const device bfloat* W [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant UdMv& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg * 8 + sg) * 4;
  if (row0 >= p.n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane * 8; k < p.k; k += 256) {
    const float4 x0 = float4(*(const device bfloat4*)(x + k)), x1 = float4(*(const device bfloat4*)(x + k + 4));
    for (uint r = 0; r < 4; r++) {
      if (row0 + r >= p.n) break;
      const device bfloat* w = W + ulong(row0 + r) * p.k + k;
      acc[r] += dot(x0, float4(*(const device bfloat4*)w)) + dot(x1, float4(*(const device bfloat4*)(w + 4)));
    }
  }
  for (uint r = 0; r < 4; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < p.n) y[row0 + r] = bfloat(s);
  }
}

struct UdNorm {
  uint add;  // 1: the residual takes the added rows first
  float eps;
};

// A row a group of 256: with `add`, R = bf16(R + A) first; then Y = bf16(w * bf16(r * rsqrt(mean(r^2) + eps))).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_norm(
    device bfloat* R [[buffer(0)]], const device bfloat* A [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* Y [[buffer(3)]], constant UdNorm& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[33];
  device bfloat* r = R + ulong(row) * UD_H;
  float v[5];
  float ss = 0.0f;
  for (uint i = 0; i < 5; i++) {
    const uint c = t + 256 * i;
    float x = float(r[c]);
    if (p.add != 0) {
      x = ud_bf(x + float(A[ulong(row) * UD_H + c]));
      r[c] = bfloat(x);
    }
    v[i] = x;
    ss += x * x;
  }
  ss = ud_sum(ss, part, 256, lane, sg);
  const float inv = precise::rsqrt(ss / float(UD_H) + p.eps);
  for (uint i = 0; i < 5; i++) {
    const uint c = t + 256 * i;
    Y[ulong(row) * UD_H + c] = bfloat(float(w[c]) * ud_bf(v[i] * inv));
  }
}

struct UdQk {
  uint pos0;  // the cache position of row 0
  float eps;
};

// Rotary on one value of a 128-wide head in torch's bf16 steps: bf16(bf16(x c) + bf16(rot(x) s)).
inline float ud_rope(float x, float partner, uint d, const device bfloat* cs, const device bfloat* sn) {
  const float c = float(cs[d % 64]), s = float(sn[d % 64]);
  const float rot = d < 64 ? -partner : partner;
  return ud_bf(ud_bf(x * c) + ud_bf(rot * s));
}

// The router's norm of a 128-wide row in fp32, (1 + w) scaled, clamped at +-1e4 before and after the scale; bf16 out.
inline float ud_router_norm(float x, const device bfloat* w, uint d, float eps, threadgroup float* part, uint lane,
                            uint sg) {
  const float ss = ud_sum(x * x, part, UD_RD, lane, sg);
  const float n = ud_clamp(x * precise::rsqrt(ss / float(UD_RD) + eps));
  return ud_bf(ud_clamp(n * (1.0f + float(w[d]))));
}

// A row's projections P (q | k | v | router q | router k) into rotated q, the caches at pos0 + row and the router's q.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ud_qk(
    const device bfloat* P [[buffer(0)]], const device bfloat* COS [[buffer(1)]], const device bfloat* SIN [[buffer(2)]],
    const device bfloat* qn [[buffer(3)]], const device bfloat* kn [[buffer(4)]], device bfloat* Q [[buffer(5)]],
    device bfloat* K [[buffer(6)]], device bfloat* V [[buffer(7)]], device bfloat* RQ [[buffer(8)]],
    device bfloat* RK [[buffer(9)]], constant UdQk& p [[buffer(10)]], uint row [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[33];
  threadgroup float rr[2][UD_RD];
  const device bfloat* x = P + ulong(row) * UD_PROJ;
  const device bfloat* cs = COS + ulong(row) * 64;
  const device bfloat* sn = SIN + ulong(row) * 64;
  const ulong at = ulong(p.pos0 + row) * UD_H;
  const uint pd = d ^ 64;
  for (uint h = 0; h < UD_HEADS; h++) {
    const uint o = h * UD_HD;
    Q[ulong(row) * UD_H + o + d] = bfloat(ud_rope(float(x[o + d]), float(x[o + pd]), d, cs, sn));
    K[at + o + d] = bfloat(ud_rope(float(x[UD_H + o + d]), float(x[UD_H + o + pd]), d, cs, sn));
    V[at + o + d] = x[2 * UD_H + o + d];
  }
  rr[0][d] = ud_router_norm(float(x[3 * UD_H + d]), qn, d, p.eps, part, lane, sg);
  rr[1][d] = ud_router_norm(float(x[3 * UD_H + UD_RD + d]), kn, d, p.eps, part, lane, sg);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  RQ[ulong(row) * UD_RD + d] = bfloat(ud_rope(rr[0][d], rr[0][pd], d, cs, sn));
  RK[ulong(p.pos0 + row) * UD_RD + d] = bfloat(ud_rope(rr[1][d], rr[1][pd], d, cs, sn));
}

// Block c0 + b's summary key (fp32): its 32 router keys weighted by a softmax of their dot with head_cls / sqrt(128).
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void ud_block_keys(
    const device bfloat* RK [[buffer(0)]], const device bfloat* hc [[buffer(1)]], device float* KB [[buffer(2)]],
    constant uint& c0 [[buffer(3)]], uint b [[threadgroup_position_in_grid]], uint j [[thread_index_in_simdgroup]]) {
  const ulong c = c0 + b;
  const device bfloat* k = RK + (c * UD_BLK + j) * UD_RD;
  float aw = 0.0f;
  for (uint d = 0; d < UD_RD; d++) aw += float(k[d]) * float(hc[d]);
  aw = ud_clamp(aw * UD_INV_SQRT_HD);
  const float e = precise::exp(aw - simd_max(aw));
  const float w = precise::divide(e, max(simd_sum(e), 1.0e-6f));
  for (uint d = j; d < UD_RD; d += 32) {
    float acc = 0.0f;
    for (uint i = 0; i < UD_BLK; i++) acc += simd_shuffle(w, i) * float(RK[(c * UD_BLK + i) * UD_RD + d]);
    KB[c * UD_RD + d] = acc;
  }
}

struct UdRoute {
  uint pos0;   // the position of row 0
  uint words;  // u32 words a candidate mask row holds
};

// A sum kept as two floats (hi + lo exact), so a running total rounds to fp32 once at each read, as a double would.
struct UdTwo {
  float hi, lo;
};

inline UdTwo ud_two_add(UdTwo a, float b) {
  const float s = a.hi + b;
  const float bb = s - a.hi;
  const float err = (a.hi - (s - bb)) + (b - bb);
  const float lo = a.lo + err;
  const float hi = s + lo;
  return {hi, lo - (hi - s)};
}

// One query row's router: remote block scores, page candidates, entmax-1.5 over the top 16; SEL/BIAS the top 12.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_route(
    const device bfloat* RQ [[buffer(0)]], const device float* KB [[buffer(1)]], const device uint* masks [[buffer(2)]],
    const device int* qmask [[buffer(3)]], device uint* SEL [[buffer(4)]], device float* BIAS [[buffer(5)]],
    constant UdRoute& p [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[33];
  threadgroup float q[UD_RD];
  threadgroup float score[4096];
  threadgroup float best_v[8];
  threadgroup uint best_i[8];
  threadgroup float top_v[UD_TOP];
  threadgroup uint top_i[UD_TOP];
  const uint pos = p.pos0 + row;
  const uint nrem = pos / UD_BLK >= UD_LOCAL_BLOCKS ? pos / UD_BLK - UD_LOCAL_BLOCKS + 1 : 0;
  if (t < UD_RD) q[t] = float(RQ[ulong(row) * UD_RD + t]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int m = qmask[row];
  float own = 0.0f;
  for (uint c = t; c < nrem; c += 256) {
    const device float* kb = KB + ulong(c) * UD_RD;
    float z = 0.0f;
    for (uint d = 0; d < UD_RD; d++) z += q[d] * kb[d];
    z = ud_clamp(z * UD_INV_SQRT_HD);
    const bool cand = m < 0 || (c / 32 < p.words && ((masks[ulong(m) * p.words + c / 32] >> (c % 32)) & 1u) != 0);
    score[c] = cand ? z * UD_ROUTER_SCALE : -INFINITY;
    own += cand ? 1.0f : 0.0f;
  }
  // no candidate on the query's page among the remote blocks: every remote block is one
  if (ud_sum(own, part, 256, lane, sg) == 0.0f) {
    for (uint c = t; c < nrem; c += 256) {
      const device float* kb = KB + ulong(c) * UD_RD;
      float z = 0.0f;
      for (uint d = 0; d < UD_RD; d++) z += q[d] * kb[d];
      z = ud_clamp(z * UD_INV_SQRT_HD);
      score[c] = z * UD_ROUTER_SCALE;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the 16 best in order, ties to the lower block
  uint n = 0;
  for (; n < UD_TOP; n++) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint c = t; c < nrem; c += 256) {
      const float v = score[c];
      if (v > bv || (v == bv && v != -INFINITY && c < bi)) {
        bv = v;
        bi = c;
      }
    }
    for (uint off = 16; off > 0; off /= 2) {
      const float ov = simd_shuffle_down(bv, off);
      const uint oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      best_v[sg] = bv;
      best_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float v = best_v[0];
      uint i = best_i[0];
      for (uint s = 1; s < 8; s++)
        if (best_v[s] > v || (best_v[s] == v && best_i[s] < i)) {
          v = best_v[s];
          i = best_i[s];
        }
      top_v[n] = v;
      top_i[n] = i;
      if (i != 0xffffffffu) score[i] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (top_i[n] == 0xffffffffu) break;
  }
  if (t != 0) return;
  device uint* sel = SEL + ulong(row) * UD_READ;
  device float* bias = BIAS + ulong(row) * UD_READ;
  const uint keep = min(n, UD_READ);
  // entmax-1.5 over the n sorted scores: X = (x - max) / 2, the threshold from running means, Y = max(X - tau, 0)^2
  float X[UD_TOP], tau[UD_TOP];
  UdTwo s1 = {0.0f, 0.0f}, s2 = {0.0f, 0.0f};
  uint support = 0;
  for (uint i = 0; i < n; i++) {
    X[i] = (top_v[i] - top_v[0]) / 2.0f;
    s1 = ud_two_add(s1, X[i]);
    s2 = ud_two_add(s2, X[i] * X[i]);
    const float rho = float(i + 1);
    const float mean = precise::divide(s1.hi + s1.lo, rho), msq = precise::divide(s2.hi + s2.lo, rho);
    const float ss = rho * (msq - mean * mean);
    const float delta = precise::divide(1.0f - ss, rho);
    tau[i] = mean - precise::sqrt(max(delta, 0.0f));
    support += tau[i] <= X[i] ? 1 : 0;
  }
  const float ts = n > 0 ? tau[max(support, 1u) - 1] : 0.0f;
  float logp[UD_READ];
  UdTwo total = {0.0f, 0.0f};
  for (uint i = 0; i < keep; i++) {
    const float y = max(X[i] - ts, 0.0f);
    logp[i] = precise::log(max(y * y, UD_FLOOR));
    total = ud_two_add(total, logp[i]);
  }
  const float mu = precise::divide(total.hi + total.lo, float(max(keep, 1u)));
  for (uint i = 0; i < UD_READ; i++) {
    sel[i] = i < keep ? top_i[i] : 0xffffffffu;
    bias[i] = i < keep ? logp[i] - mu : 0.0f;
  }
}

struct UdAttend {
  uint pos0;       // the position of row 0
  uint bf16_bias;  // 1: the blocks' biases round to bf16 first (the prompt's mask is bf16)
};

// The largest of every thread's `v` in a threadgroup, returned to all of them.
inline float ud_max(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_max(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : -INFINITY;
    s = simd_max(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// One (head, row) of fp32 attention over the last 128 keys and the read blocks (biased); simdgroups stride keys.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_attend(
    const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    const device uint* SEL [[buffer(3)]], const device float* BIAS [[buffer(4)]], device bfloat* O [[buffer(5)]],
    constant UdAttend& p [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint MAXK = UD_LOCAL + UD_READ * UD_BLK;
  threadgroup uint keys[MAXK];
  threadgroup float kbias[MAXK];
  threadgroup float sc[MAXK];
  threadgroup float4 pv[8][32];
  threadgroup float part[33];
  threadgroup uint start[UD_READ + 1];
  const uint h = g.x, row = g.y;
  const uint pos = p.pos0 + row;
  const uint lo = pos >= UD_LOCAL - 1 ? pos - (UD_LOCAL - 1) : 0;
  const uint nloc = pos - lo + 1;
  const device uint* sel = SEL + ulong(row) * UD_READ;
  const device float* bias = BIAS + ulong(row) * UD_READ;
  if (t == 0) {
    uint n = nloc;
    for (uint s = 0; s < UD_READ; s++) {
      start[s] = n;
      const uint c = sel[s];
      if (c != 0xffffffffu && c * UD_BLK < lo) n += min(UD_BLK, lo - c * UD_BLK);
    }
    start[UD_READ] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total = start[UD_READ];
  for (uint i = t; i < total; i += 256) {
    if (i < nloc) {
      keys[i] = lo + i;
      kbias[i] = 0.0f;
    } else {
      uint s = 0;
      while (start[s + 1] <= i) s++;
      keys[i] = sel[s] * UD_BLK + (i - start[s]);
      kbias[i] = p.bf16_bias != 0 ? ud_bf(bias[s]) : bias[s];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint o = h * UD_HD + 4 * lane;
  const float4 q = float4(*(const device bfloat4*)(Q + ulong(row) * UD_H + o));
  for (uint i = sg; i < total; i += 8) {
    const float4 k = float4(*(const device bfloat4*)(K + ulong(keys[i]) * UD_H + o));
    const float s = simd_sum(dot(q, k));
    if (lane == 0) sc[i] = s * UD_INV_SQRT_HD + kbias[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = -INFINITY;
  for (uint i = t; i < total; i += 256) top = max(top, sc[i]);
  top = ud_max(top, part, 256, lane, sg);
  float mass = 0.0f;
  for (uint i = t; i < total; i += 256) {
    const float e = precise::exp(sc[i] - top);
    sc[i] = e;
    mass += e;
  }
  mass = ud_sum(mass, part, 256, lane, sg);
  for (uint i = t; i < total; i += 256) sc[i] = precise::divide(sc[i], mass);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc = 0.0f;
  for (uint i = sg; i < total; i += 8) acc += sc[i] * float4(*(const device bfloat4*)(V + ulong(keys[i]) * UD_H + o));
  pv[sg][lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (uint s = 1; s < 8; s++) acc += pv[s][lane];
  *(device bfloat4*)(O + ulong(row) * UD_H + o) = bfloat4(acc);
}

// A row's fp32 logits against the gate's 64 rows, their softmax, the 6 largest in order (ties to the lower expert).
[[max_total_threads_per_threadgroup(1024)]] [[kernel]] void ud_moe_gate(
    const device bfloat* X [[buffer(0)]], const device bfloat* G [[buffer(1)]], device uint* IDS [[buffer(2)]],
    device float* WTS [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float logit[UD_EXPERTS];
  const device bfloat* x = X + ulong(row) * UD_H;
  float acc[2] = {0.0f, 0.0f};
  for (uint k = lane * 8; k < UD_H; k += 256) {
    const float4 x0 = float4(*(const device bfloat4*)(x + k)), x1 = float4(*(const device bfloat4*)(x + k + 4));
    for (uint r = 0; r < 2; r++) {
      const device bfloat* g = G + ulong(sg * 2 + r) * UD_H + k;
      acc[r] += dot(x0, float4(*(const device bfloat4*)g)) + dot(x1, float4(*(const device bfloat4*)(g + 4)));
    }
  }
  for (uint r = 0; r < 2; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0) logit[sg * 2 + r] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float v[2] = {logit[lane], logit[lane + 32]};
  const float top = simd_max(max(v[0], v[1]));
  v[0] = precise::exp(v[0] - top);
  v[1] = precise::exp(v[1] - top);
  const float mass = simd_sum(v[0] + v[1]);
  v[0] = precise::divide(v[0], mass);
  v[1] = precise::divide(v[1], mass);
  // a NaN probability ranks below every number and above a taken expert, so a row of NaN picks experts 0 to 5
  bool taken[2] = {false, false};
  for (uint k = 0; k < UD_PICK; k++) {
    const float a = taken[0] ? -2.0f : (isnan(v[0]) ? -1.0f : v[0]);
    const float b = taken[1] ? -2.0f : (isnan(v[1]) ? -1.0f : v[1]);
    const float best = simd_max(max(a, b));
    const uint e = simd_min(a == best ? lane : (b == best ? lane + 32 : UD_EXPERTS));
    const float wt = simd_shuffle(e < 32 ? v[0] : v[1], e % 32);
    if (e == lane) taken[0] = true;
    if (e == lane + 32) taken[1] = true;
    if (lane == 0) {
      IDS[ulong(row) * UD_PICK + k] = e;
      WTS[ulong(row) * UD_PICK + k] = wt;
    }
  }
}

struct UdAct {
  uint width;     // values a row
  uint gstride;   // the gate's row stride
  uint ustride;   // the up's row stride
};

// SwiGLU in torch's bf16 steps: Y = bf16(bf16(silu(g)) * u), silu in fp32.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_silu_mul(
    const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant UdAct& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.width) return;
  const float g = float(G[ulong(at.y) * p.gstride + at.x]);
  const float a = ud_bf(precise::divide(g, 1.0f + precise::exp(-g)));
  Y[ulong(at.y) * p.width + at.x] = bfloat(a * float(U[ulong(at.y) * p.ustride + at.x]));
}

// A routed expert's three projections, by their GPU addresses.
struct UdExpert {
  device const bfloat* gate;
  device const bfloat* up;
  device const bfloat* down;
};

struct UdExp {
  uint width;  // the expert's hidden width (896)
};

// One token's picked experts' gate and up rows: GU[j] = [gate . x | up . x] for each of its 6 experts, fp32 sums.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_expert_gu(
    const device bfloat* x [[buffer(0)]], const device UdExpert* E [[buffer(1)]], const device uint* ids [[buffer(2)]],
    device bfloat* GU [[buffer(3)]], constant UdExp& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg * 8 + sg) * 4;
  if (row0 >= UD_PICK * 2 * p.width) return;
  const uint j = row0 / (2 * p.width), r0 = row0 % (2 * p.width);
  const UdExpert ex = E[ids[j]];
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane * 8; k < UD_H; k += 256) {
    const float4 x0 = float4(*(const device bfloat4*)(x + k)), x1 = float4(*(const device bfloat4*)(x + k + 4));
    for (uint r = 0; r < 4; r++) {
      const uint rr = r0 + r;
      const device bfloat* w = (rr < p.width ? ex.gate + ulong(rr) * UD_H : ex.up + ulong(rr - p.width) * UD_H) + k;
      acc[r] += dot(x0, float4(*(const device bfloat4*)w)) + dot(x1, float4(*(const device bfloat4*)(w + 4)));
    }
  }
  for (uint r = 0; r < 4; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0) GU[ulong(row0) + r] = bfloat(s);
  }
}

// One token's picked experts' down rows: D[j] = down_j . act_j (1280 each), fp32 sums.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_expert_down(
    const device bfloat* A [[buffer(0)]], const device UdExpert* E [[buffer(1)]], const device uint* ids [[buffer(2)]],
    device bfloat* D [[buffer(3)]], constant UdExp& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg * 8 + sg) * 4;
  if (row0 >= UD_PICK * UD_H) return;
  const uint j = row0 / UD_H, r0 = row0 % UD_H;
  const device bfloat* x = A + ulong(j) * p.width;
  const device bfloat* W = E[ids[j]].down;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane * 8; k < p.width; k += 256) {
    const float4 x0 = float4(*(const device bfloat4*)(x + k)), x1 = float4(*(const device bfloat4*)(x + k + 4));
    for (uint r = 0; r < 4; r++) {
      const device bfloat* w = W + ulong(r0 + r) * p.width + k;
      acc[r] += dot(x0, float4(*(const device bfloat4*)w)) + dot(x1, float4(*(const device bfloat4*)(w + 4)));
    }
  }
  for (uint r = 0; r < 4; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0) D[ulong(row0) + r] = bfloat(s);
  }
}

// The experts' sum for a row: bf16(sum_j D[slot[r, j]] w[r, j]) in fp32, then the shared expert's rows added in bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_combine(
    const device bfloat* D [[buffer(0)]], const device uint* slot [[buffer(1)]], const device float* WTS [[buffer(2)]],
    const device bfloat* S [[buffer(3)]], device bfloat* Y [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= UD_H) return;
  float acc = 0.0f;
  for (uint j = 0; j < UD_PICK; j++) {
    const ulong i = ulong(at.y) * UD_PICK + j;
    acc += float(D[ulong(slot[i]) * UD_H + at.x]) * WTS[i];
  }
  const ulong o = ulong(at.y) * UD_H + at.x;
  Y[o] = bfloat(ud_bf(acc) + float(S[o]));
}
