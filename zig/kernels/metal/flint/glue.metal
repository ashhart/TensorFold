// Flint's glue at the reference's rounding points (RecursiveSI's rec_mlx on MLX): bf16 residual, MLX rms_norm and rope.
#include <metal_stdlib>
using namespace metal;

struct fl_args { uint rows, width, heads, dim, pos, stride, bits, group, ring, rope; float eps, theta; };

// Row r's slot: (pos + r) % ring in a ring, else r.
inline size_t fl_slot(constant fl_args& a, uint row) { return a.ring ? (a.pos + row) % a.ring : row; }

// Row r of residual R = dequantized affine embedding row ids[r]: bf16(scale * q + bias), held in fp32.
[[kernel]] void fl_embed(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant fl_args& a [[buffer(4)]],
    device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t row = ids[p.y];
  const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;
  const uint word = W[row * (a.width / per) + p.x / per];
  const float q = float((word >> ((p.x % per) * a.bits)) & mask);
  const size_t g = row * (a.width / a.group) + p.x / a.group;
  R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));
}

// MLX's rms_norm of the residual's rows (bf16 values): out = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))).
[[kernel]] void fl_norm(const device float* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant fl_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) sum = fma(x[i], x[i], sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv)));
}

// A = silu(gate) * up as MLX rounds it: s = bf16(sigmoid(g)), bf16(bf16(g * s) * u), over rows x width.
[[kernel]] void fl_swiglu(const device bfloat* Gt [[buffer(0)]], const device bfloat* U [[buffer(1)]],
    constant fl_args& a [[buffer(2)]], device bfloat* A [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  const float g = float(Gt[i]), u = float(U[i]);
  const float e = precise::exp(-fabs(g)), y = 1.0f / (1.0f + e);
  const float s = float(bfloat(g < 0.0f ? 1.0f - y : y));
  A[i] = bfloat(float(bfloat(g * s)) * u);
}

// R = bf16(R + X): a sublayer's bf16 output into the residual.
[[kernel]] void fl_add(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant fl_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] = float(bfloat(R[i] + float(X[i])));
}

// routed = bf16(sum_k bf16(Y[r, k] * bf16(wts[r, k]))) in pick order; R = bf16(R + bf16(routed + shared)).
[[kernel]] void fl_moe_add(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device float* wts [[buffer(2)]], const device bfloat* shared [[buffer(3)]],
    constant fl_args& a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint k = a.heads; // picks a row
  float sum = 0.0f;
  for (uint j = 0; j < k; ++j) {
    const float w = float(bfloat(wts[p.y * k + j]));
    sum += float(bfloat(w * float(Y[((size_t)p.y * k + j) * a.width + p.x])));
  }
  const size_t i = (size_t)p.y * a.width + p.x;
  const float mlp = float(bfloat(float(bfloat(sum)) + float(shared[i])));
  R[i] = float(bfloat(R[i] + mlp));
}

// A chunk's a.rows picks by expert (a.width of them): offsets[e] its first pair, order the pairs, each in pick order.
[[kernel]] void fl_sort(const device int* pick [[buffer(0)]], constant fl_args& a [[buffer(1)]],
    device int* offsets [[buffer(2)]], device int* order [[buffer(3)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup int counts[1024];
  int n = 0;
  if (t < a.width)
    for (uint i = 0; i < a.rows; ++i) n += pick[i] == int(t) ? 1 : 0;
  counts[t] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    int sum = 0;
    for (uint e = 0; e < a.width; ++e) { const int c = counts[e]; counts[e] = sum; sum += c; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t >= a.width) return;
  int at = counts[t];
  offsets[t] = at;
  for (uint i = 0; i < a.rows; ++i) if (pick[i] == int(t)) order[at++] = int(i);
}

// Y[i] = X[order[i] / a.heads]: each sorted pair's input row (a.width wide).
[[kernel]] void fl_rows_by_order(const device bfloat* X [[buffer(0)]], const device int* order [[buffer(1)]],
    constant fl_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) Y[(size_t)p.y * a.width + p.x] = X[(size_t)(order[p.y] / int(a.heads)) * a.width + p.x];
}

// A simdgroup a row: the core select's top-k (sigmoid, ties to the lower id) and w / sum(w), no per-Mac lists.
[[kernel]] void fl_route_pick(const device float* L [[buffer(0)]], constant fl_args& a [[buffer(1)]],
    device int* PICK [[buffer(2)]], device float* W [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  constexpr uint PER = 8; // experts a lane: a.width <= 256
  float sc[PER];
  uint taken = 0;
  for (uint j = 0; j < PER; ++j) {
    const uint id = j * 32 + lane;
    if (id < a.width) {
      const float x = L[(size_t)row * a.width + id];
      const float y = 1.0f / (1.0f + precise::exp(fabs(x)));
      sc[j] = x < 0 ? y : 1.0f - y;
    } else {
      sc[j] = 0.0f;
      taken |= 1u << j;
    }
  }
  float w[16];
  for (uint k = 0; k < a.heads; ++k) {
    float best = -INFINITY;
    uint bid = a.width;
    for (uint j = 0; j < PER; ++j) {
      const uint id = j * 32 + lane;
      if (!(taken & (1u << j)) && (sc[j] > best || (sc[j] == best && id < bid))) { best = sc[j]; bid = id; }
    }
    for (ushort off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const uint oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    w[k] = best;
    if (bid % 32 == lane) taken |= 1u << (bid / 32);
    if (lane == 0) PICK[row * a.heads + k] = int(bid);
  }
  if (lane != 0) return;
  float total = w[0];
  for (uint k = 1; k < a.heads; ++k) total = total + w[k];
  for (uint k = 0; k < a.heads; ++k) W[row * a.heads + k] = (w[k] / total) * 1.0f;
}

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct fl_vec_args { uint rows, pos, heads, groups, ring, window, blocks, chunk, first; float scale; };

// Decode attention partials (sdpa_vector lanes): one (key head, row, absolute key block) a threadgroup, head a sg.
[[kernel]] void fl_decode_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant fl_vec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, block = a.first + tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const uint p = a.pos + row;
  const uint lo = a.window > 0 && p + 1 > a.window ? p + 1 - a.window : 0;
  const uint from = max(lo, block * a.chunk), to = min(p + 1, (block + 1) * a.chunk);
  const size_t ld = (size_t)a.groups * 128;
  const float4 qv = a.scale * float4(*(const device bfloat4*)(Q + ((size_t)row * a.heads + head) * 128 + 4 * lane));
  float4 acc = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 vr[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to; // past the block's end: a masked key (read at the block's first, weight 0)
      const uint i = live ? i0 + u : from;
      const size_t at = (size_t)(a.ring ? i % a.ring : i) * ld + kv * 128 + 4 * lane;
      vr[u] = float4(*(const device bfloat4*)(V + at));
      const float sc = simd_sum(dot(qv, float4(*(const device bfloat4*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc *= rescale;
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc += w * vr[u];
    }
    top = peak;
  }
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + tg.z;
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * 128 + 4 * lane) = acc;
}

// MLX's sdpa_vector_2pass_2 with the sink in max and denominator: 32 simdgroups a (head, row) fold the blocks.
[[kernel]] void fl_merge_vec(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant fl_vec_args& a [[buffer(3)]], const device bfloat* sinks [[buffer(4)]],
    device bfloat* O [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float merged[32 * 32];
  const uint head = tg.x, row = tg.y;
  const size_t at = (size_t)head * a.rows + row;
  const device float* ms = PM + at * a.blocks;
  const device float* ls = PL + at * a.blocks;
  const float sink = float(sinks[head]);
  // lanes and simdgroups take blocks by absolute index (first + b): a row's sums never move with the window's first row
  const uint lane0 = (lane + 32 - a.first % 32) % 32, sg0 = (sg + 32 - a.first % 32) % 32;
  float top = sink;
  for (uint b = lane0; b < a.blocks; b += 32) top = max(top, ms[b]);
  top = simd_max(top);
  float total = 0.0f;
  for (uint b = lane0; b < a.blocks; b += 32) total += ls[b] > 0.0f ? ls[b] * fast::exp(ms[b] - top) : 0.0f;
  total = simd_sum(total) + fast::exp(sink - top);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint b = sg0; b < a.blocks; b += 32) {
    if (ls[b] <= 0.0f) continue;
    const float f = fast::exp(ms[b] - top);
    for (uint j = 0; j < 4; ++j) acc[j] += f * PO[(at * a.blocks + b) * 128 + 4 * lane + j];
  }
  for (uint j = 0; j < 4; ++j) {
    merged[lane * 32 + sg] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc[j] = simd_sum(merged[sg * 32 + lane]) / total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    for (uint j = 0; j < 4; ++j) O[(row * a.heads + head) * 128 + 4 * sg + j] = bfloat(acc[j]);
}

// fl_add then fl_norm on a row, one threadgroup of 256: R = bf16(R + X), out = rms_norm(R) * w, the same sums.
[[kernel]] void fl_add_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant fl_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    const float v = float(bfloat(x[i] + float(X[(size_t)row * a.width + i])));
    x[i] = v;
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv)));
}

// fl_moe_add then fl_norm on a row (the next layer's input norm), one threadgroup of 256, the same sums.
[[kernel]] void fl_moe_add_norm(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device float* wts [[buffer(2)]], const device bfloat* shared [[buffer(3)]], const device bfloat* w [[buffer(4)]],
    constant fl_args& a [[buffer(5)]], device bfloat* out [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  const uint k = a.heads;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) {
    float routed = 0.0f;
    for (uint j = 0; j < k; ++j) {
      const float wj = float(bfloat(wts[row * k + j]));
      routed += float(bfloat(wj * float(Y[((size_t)row * k + j) * a.width + i])));
    }
    const float mlp = float(bfloat(float(bfloat(routed)) + float(shared[(size_t)row * a.width + i])));
    const float v = float(bfloat(x[i] + mlp));
    x[i] = v;
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv)));
}

struct fl_prep_args { uint rows, heads, kv_heads, dim, pos, ring, rope, stride; float eps, theta; };

// One launch: q normed and rotated to QN; k normed, rotated into the cache and v beside it (rows a.stride apart).
[[kernel]] void fl_prep(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant fl_prep_args& a [[buffer(3)]], device bfloat* QN [[buffer(4)]],
    device bfloat* KC [[buffer(5)]], device bfloat* VC [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? 0 : a.heads * a.dim) + (size_t)head * a.dim;
  const device bfloat* w = is_q ? qw : kw;
  const size_t slot = a.ring ? (a.pos + row) % a.ring : a.pos + row;
  const size_t kvw = (size_t)a.kv_heads * a.dim;
  device bfloat* y = is_q ? QN + ((size_t)row * a.heads + head) * a.dim : KC + slot * kvw + (size_t)head * a.dim;
  const uint per = a.dim / 32;
  float v[8];
  float sum = 0.0f;
  for (uint j = 0; j < per; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / float(a.dim) + a.eps);
  threadgroup float nrm[256];
  for (uint j = 0; j < per; ++j) nrm[lane + 32 * j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_dim = a.dim / 2;
  const float position = float(a.pos + row);
  const float base = precise::log2(a.theta);
  for (uint j = 0; j < per; ++j) {
    const uint d = lane + 32 * j;
    if (a.rope == 0) { y[d] = bfloat(nrm[d]); continue; }
    const uint i = d % half_dim;
    const float freq = precise::exp2(-(float(i) / float(half_dim)) * base);
    const float angle = position * freq;
    const float c = precise::cos(angle), s = precise::sin(angle);
    const float re = nrm[i], im = nrm[i + half_dim];
    y[d] = bfloat(d < half_dim ? re * c - im * s : re * s + im * c);
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + (a.heads + a.kv_heads) * a.dim + (size_t)head * a.dim;
  device bfloat* vd = VC + slot * kvw + (size_t)head * a.dim;
  for (uint j = 0; j < per; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// fl_swiglu on a stacked gate|up row (gate at 0, up at a.width): A = bf16(bf16(g * bf16(sigmoid(g))) * u).
[[kernel]] void fl_swiglu_gu(const device bfloat* GU [[buffer(0)]], constant fl_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  const float g = float(GU[at]), u = float(GU[at + a.width]);
  const float e = precise::exp(-fabs(g)), y = 1.0f / (1.0f + e);
  const float s = float(bfloat(g < 0.0f ? 1.0f - y : y));
  A[(size_t)p.y * a.width + p.x] = bfloat(float(bfloat(g * s)) * u);
}

inline bool fl_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// The greedy draw (nemotron_glue's tf_argmax_bf16): a bf16 logits row's argmax, ties to the lower id, into ids[0].
[[kernel]] void fl_argmax(const device bfloat* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
    constant uint& vocab [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint idxs[32];
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(logits[i]);
    if (fl_better(v, i, best, at)) { best = v; at = i; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (fl_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vals[lane];
  at = idxs[lane];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (fl_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) ids[0] = at;
}
