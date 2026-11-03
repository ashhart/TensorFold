// Unlimited-OCR-2's decoder steps: page2d rotary, the DASH router (block keys, ranking, entmax bias), its sparse
// attention, and the MoE's gate and combine, each rounded where the publisher's torch code rounds; the projections
// run on gemm_nax beside these.

#include <metal_stdlib>
using namespace metal;

#define UD_NEG -1.0e9f
#define UD_ENTMAX_NEG -1.0e4f
#define UD_CLAMP 1.0e4f

inline float ud_bf(float x) { return float(bfloat(x)); }

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

struct UdNorm {
  uint width;
  float eps;
};

// DeepseekV2RMSNorm on bf16 rows: x * rsqrt(mean(x^2) + eps) in fp32, rounded, then times the weight in bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_rms_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant UdNorm& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float v = float(xr[i]);
    s += v * v;
  }
  s = ud_sum(s, part, n, lane, sg);
  const float r = metal::precise::rsqrt(s / float(p.width) + p.eps);
  device bfloat* yr = y + ulong(row) * p.width;
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(float(w[i]) * ud_bf(float(xr[i]) * r));
}

// The page2d angles for one position: 63 token frequencies, then the page parity (page mod 2) at frequency 1.
inline float ud_angle(constant float* inv_tok, uint tok, uint page, uint i) {
  const uint j = i % 64;
  return j < 63 ? float(tok) * inv_tok[j] : float(page % 2);
}

// x * cos + rotate_half(x) * sin over 128-wide heads, every product and the sum rounded to bf16 (torch's bf16 ops);
// cos and sin are bf16 (the rotary's output dtype).
inline bfloat ud_rotate(const device bfloat* h, uint d, float ang) {
  const float c = ud_bf(metal::precise::cos(ang)), s = ud_bf(metal::precise::sin(ang));
  const float rh = d < 64 ? -float(h[d + 64]) : float(h[d - 64]);
  return bfloat(ud_bf(float(h[d]) * c) + ud_bf(rh * s));
}

struct UdRope {
  uint heads;   // 128-wide heads in a row
  uint stride;  // elements between rows
};

// The main branch's page2d rotary in place on rows of `heads` heads: row r at (tok[r], page[r]).
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ud_rope(
    device bfloat* x [[buffer(0)]], const device uint* tok [[buffer(1)]], const device uint* page [[buffer(2)]],
    constant float* inv_tok [[buffer(3)]], constant UdRope& p [[buffer(4)]], uint3 at [[thread_position_in_grid]]) {
  const uint d = at.x, head = at.y, row = at.z;
  device bfloat* h = x + ulong(row) * p.stride + head * 128;
  const bfloat out = ud_rotate(h, d, ud_angle(inv_tok, tok[row], page[row], d));
  threadgroup_barrier(mem_flags::mem_device);
  h[d] = out;
}

// The router's index q or k: Gemma RMSNorm (fp32, clamped, times 1 + w, clamped, bf16), the same rotary in bf16,
// then fp32. One 128-thread group a row; y rows are 128 floats.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void ud_router_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device uint* tok [[buffer(2)]],
    const device uint* page [[buffer(3)]], constant float* inv_tok [[buffer(4)]], device float* y [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup bfloat normed[128];
  float v = float(x[ulong(row) * 128 + d]);
  if (!isfinite(v)) v = 0.0f;
  const float s = ud_sum(v * v, part, 128, lane, sg);
  float n = clamp(v * metal::precise::rsqrt(s / 128.0f + 1e-6f), -UD_CLAMP, UD_CLAMP);
  n = clamp(n * (1.0f + float(w[d])), -UD_CLAMP, UD_CLAMP);
  normed[d] = bfloat(n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ang = ud_angle(inv_tok, tok[row], page[row], d);
  const float c = ud_bf(metal::precise::cos(ang)), sn = ud_bf(metal::precise::sin(ang));
  const float rh = d < 64 ? -float(normed[d + 64]) : float(normed[d - 64]);
  float out = float(bfloat(ud_bf(float(normed[d]) * c) + ud_bf(rh * sn)));
  if (!isfinite(out)) out = 0.0f;
  y[ulong(row) * 128 + d] = out;
}

struct UdBlocks {
  uint first;  // the first block to summarize
  float scale; // 1 / sqrt(128)
};

// A complete block's summary key: softmax over its 32 keys of clamp(k . head_cls / sqrt(128)), then the weighted
// sum of the keys, in fp32 (one 32-thread group a block, one key a thread).
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void ud_block_keys(
    const device float* idx_k [[buffer(0)]], const device bfloat* head_cls [[buffer(1)]], device float* kbar [[buffer(2)]],
    constant UdBlocks& p [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]]) {
  threadgroup float wts[32];
  const uint c = p.first + g;
  const device float* k = idx_k + (ulong(c) * 32 + j) * 128;
  float a = 0.0f;
  for (uint d = 0; d < 128; d++) a += k[d] * float(head_cls[d]);
  a = clamp(a * p.scale, -UD_CLAMP, UD_CLAMP);
  const float m = simd_max(a);
  const float e = metal::precise::exp(a - m);
  const float total = max(simd_sum(e), 1e-6f);
  wts[j] = e / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = j; d < 128; d += 32) {
    float s = 0.0f;
    for (uint t = 0; t < 32; t++) s += wts[t] * idx_k[(ulong(c) * 32 + t) * 128 + d];
    kbar[ulong(c) * 128 + d] = isfinite(s) ? s : 0.0f;
  }
}

struct UdRoute {
  uint nblk;        // blocks in the key range (prefill: the whole prompt's; decode: the cache's)
  uint complete;    // blocks with a summary key (every remote block is complete)
  float scale;      // 1 / sqrt(128)
  float idx_scale;  // 128 / sqrt(128)
  float floor_;     // the prior's floor before its log
  uint round_bias;  // 1: the bias as the prefill's bf16 mask carries it
};

// One query's remote reads (a simdgroup a query): block scores against the summary keys, the page-aware candidate
// blocks (a block holding an image token of the query's page, any block when the query has no page or none
// holds one; remote blocks only, all of them when that leaves none), the top 16 by score, entmax-1.5 over them, and
// the first 12 with bias log(max(prior, floor)) - their mean. sel[q][s] = -1 marks an unread slot.
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void ud_route(
    const device float* idx_q [[buffer(0)]], const device float* kbar [[buffer(1)]], const device int* bmin [[buffer(2)]],
    const device int* bmax [[buffer(3)]], const device int* qpage [[buffer(4)]], const device uint* qpos [[buffer(5)]],
    device int* sel [[buffer(6)]], device float* bias [[buffer(7)]], constant UdRoute& p [[buffer(8)]],
    uint q [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float x[4096];
  threadgroup uchar cand[4096];
  const device float* iq = idx_q + ulong(q) * 128;
  const int qp = qpage[q];
  const int last_remote = int(qpos[q] / 32) - 4;
  // the page's own image blocks anywhere in the key range, or every block
  bool any_page = false;
  if (qp >= 0) {
    for (uint c = lane; c < p.nblk; c += 32) any_page = any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
    any_page = simd_any(any_page);
  }
  bool any_cand = false;
  for (uint c = lane; c < p.nblk; c += 32) {
    const bool remote = int(c) <= last_remote && c < p.complete;
    const bool allow = !any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
    float z = UD_NEG;
    if (remote) {
      const device float* kb = kbar + ulong(c) * 128;
      float s = 0.0f;
      for (uint d = 0; d < 128; d++) s += iq[d] * kb[d];
      z = clamp(s * p.scale, -UD_CLAMP, UD_CLAMP);
    }
    x[c] = z;
    cand[c] = uchar(remote ? (allow ? 2 : 1) : 0);
    any_cand = any_cand || (remote && allow);
  }
  any_cand = simd_any(any_cand);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = lane; c < p.nblk; c += 32) {
    const bool is = any_cand ? cand[c] == 2 : cand[c] >= 1;
    cand[c] = uchar(is);
    float v = is ? x[c] * p.idx_scale : UD_ENTMAX_NEG;
    if (!isfinite(v)) v = UD_ENTMAX_NEG;
    x[c] = v;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // the top M by value, ties to the lower block
  const uint M = max(1u, min(16u, p.nblk));
  float xv[16];
  int xi[16];
  for (uint r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (uint c = lane; c < p.nblk; c += 32) {
      bool taken = false;
      for (uint t = 0; t < r; t++) taken = taken || xi[t] == int(c);
      if (!taken && (x[c] > best || (x[c] == best && int(c) < at))) {
        best = x[c];
        at = int(c);
      }
    }
    const float top = simd_max(best);
    const int idx = simd_min(best == top ? at : 0x7fffffff);
    xv[r] = top;
    xi[r] = idx;
  }
  if (lane != 0) return;
  // entmax-1.5 over the M values (already in descending order)
  const float mx = xv[0];
  float X[16], tau[16];
  float cs = 0.0f, cs2 = 0.0f;
  uint support = 0;
  for (uint k = 0; k < M; k++) {
    X[k] = (xv[k] - mx) / 2.0f;
    cs += X[k];
    cs2 += X[k] * X[k];
    const float rho = float(k + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = max((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - metal::precise::sqrt(delta);
    support += tau[k] <= X[k] ? 1 : 0;
  }
  const float tau_star = tau[max(support, 1u) - 1];
  const uint K = max(1u, min(12u, M));
  float logp[12];
  float total = 0.0f;
  uint count = 0;
  for (uint s = 0; s < K; s++) {
    const float d = max(X[s] - tau_star, 0.0f);
    logp[s] = metal::precise::log(max(d * d, p.floor_));
    if (cand[xi[s]]) {
      total += logp[s];
      count++;
    }
  }
  const float mu = total / float(max(count, 1u));
  for (uint s = 0; s < 12; s++) {
    const bool valid = s < K && cand[xi[s]];
    sel[q * 12 + s] = valid ? xi[s] : -1;
    const float b = valid ? logp[s] - mu : 0.0f;
    bias[q * 12 + s] = p.round_bias ? ud_bf(b) : b;
  }
}

struct UdAttend {
  uint width;    // heads x 128 in a q, k or v row
  float scale;   // 1 / sqrt(128)
  uint local;    // the local window (128 keys)
};

// One query row and head (a simdgroup): softmax over the local keys p-127..p (no bias) and the selected blocks' keys
// before the window (their bias), fp32 scores and weights, out = sum(p v) / sum(p) rounded to bf16.
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void ud_attend(
    const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]], const device bfloat* vc [[buffer(2)]],
    const device int* sel [[buffer(3)]], const device float* bias [[buffer(4)]], const device uint* qpos [[buffer(5)]],
    device bfloat* o [[buffer(6)]], constant UdAttend& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float score[512];
  threadgroup uint key[512];
  const uint row = tg.x, h = tg.y;
  const uint pos = qpos[row];
  const uint lo = pos + 1 > p.local ? pos + 1 - p.local : 0;
  const device bfloat* qr = q + ulong(row) * p.width + h * 128;
  const float4 qv = float4(qr[lane * 4], qr[lane * 4 + 1], qr[lane * 4 + 2], qr[lane * 4 + 3]);
  uint n = 0;
  float m = -INFINITY;
  // remote blocks first, then the window; a block key inside the window is the window's
  for (uint s = 0; s <= 12; s++) {
    uint first, last;
    float b = 0.0f;
    if (s < 12) {
      const int c = sel[row * 12 + s];
      if (c < 0) continue;
      first = uint(c) * 32;
      last = min(first + 32, lo);
      b = bias[row * 12 + s];
    } else {
      first = lo;
      last = pos + 1;
    }
    for (uint j = first; j < last; j++) {
      const device bfloat* kr = kc + ulong(j) * p.width + h * 128 + lane * 4;
      const float part = qv.x * float(kr[0]) + qv.y * float(kr[1]) + qv.z * float(kr[2]) + qv.w * float(kr[3]);
      const float sc = simd_sum(part) * p.scale + b;
      if (lane == 0) {
        score[n] = sc;
        key[n] = j;
      }
      m = max(m, sc);
      n++;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc = 0.0f;
  float total = 0.0f;
  for (uint i = 0; i < n; i++) {
    const float e = metal::precise::exp(score[i] - m);
    total += e;
    const device bfloat* vr = vc + ulong(key[i]) * p.width + h * 128 + lane * 4;
    acc += e * float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
  }
  device bfloat* out = o + ulong(row) * p.width + h * 128 + lane * 4;
  for (uint d = 0; d < 4; d++) out[d] = bfloat(acc[d] / total);
}

// y = bf16(bf16(silu(g)) * u): the MLP's act_fn(gate) * up in bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_silu_mul(
    const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = float(g[i]);
  y[i] = bfloat(ud_bf(v / (1.0f + metal::precise::exp(-v))) * float(u[i]));
}

// y = bf16(a + b) over `n` values.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_add(
    const device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = bfloat(float(a[i]) + float(b[i]));
}

// The MoE gate for one row (a 64-thread group): fp32 logits of x against the gate's 64 rows, softmax, and the 6
// largest scores (descending; ties to the lower expert) as ids and weights, not renormalized.
[[max_total_threads_per_threadgroup(64)]] [[kernel]] void ud_moe_gate(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device uint* ids [[buffer(2)]],
    device float* wts [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint e [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup float score[64];
  const device bfloat* xr = x + ulong(row) * 1280;
  const device bfloat* wr = w + ulong(e) * 1280;
  float s = 0.0f;
  for (uint d = 0; d < 1280; d++) s += float(xr[d]) * float(wr[d]);
  // softmax over the 64 logits
  float m = simd_max(s);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(part[0], part[1]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ex = metal::precise::exp(s - m);
  const float total = ud_sum(ex, part, 64, lane, sg);
  score[e] = ex / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e != 0) return;
  bool taken[64];
  for (uint i = 0; i < 64; i++) taken[i] = false;
  for (uint k = 0; k < 6; k++) {
    uint best = 0;
    float bv = -INFINITY;
    for (uint i = 0; i < 64; i++) {
      if (!taken[i] && score[i] > bv) {
        bv = score[i];
        best = i;
      }
    }
    taken[best] = true;
    ids[row * 6 + k] = best;
    wts[row * 6 + k] = bv;
  }
}

// Gathered rows: y[i] = x[rows[i]] (1280 wide).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_gather(
    const device bfloat* x [[buffer(0)]], const device uint* rows [[buffer(1)]], device bfloat* y [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= 1280) return;
  y[ulong(at.y) * 1280 + at.x] = x[ulong(rows[at.y]) * 1280 + at.x];
}

// An expert's rows back to their slots: slots[dest[i]] = y[i].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_scatter(
    const device bfloat* y [[buffer(0)]], const device uint* dest [[buffer(1)]], device bfloat* slots [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= 1280) return;
  slots[ulong(dest[at.y]) * 1280 + at.x] = y[ulong(at.y) * 1280 + at.x];
}

// The routed sum and the shared expert: y = bf16(bf16(sum_s w_s * slot_s in fp32) + shared), slots in gate order.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void ud_moe_combine(
    const device bfloat* slots [[buffer(0)]], const device float* wts [[buffer(1)]], const device bfloat* shared [[buffer(2)]],
    device bfloat* y [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= 1280) return;
  const uint row = at.y;
  float s = 0.0f;
  for (uint k = 0; k < 6; k++) s += float(slots[(ulong(row) * 6 + k) * 1280 + at.x]) * wts[row * 6 + k];
  const ulong i = ulong(row) * 1280 + at.x;
  y[i] = bfloat(ud_bf(s) + float(shared[i]));
}
