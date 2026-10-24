// MiniMax-M3.1-Flash's glue at transformers' rounding points (modeling_minimax_m3_vl.py, bf16 activations).
#include <metal_stdlib>
using namespace metal;

struct mm_args { uint rows, width, heads, dim, pos, stride, bits, group; float eps, alpha, limit, scale; };

// Row r of residual R = dequantized affine embedding row ids[r]: bf16(scale * q + bias), held in fp32.
[[kernel]] void mm_embed(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant mm_args& a [[buffer(4)]],
    device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t row = ids[p.y];
  const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;
  const uint word = W[row * (a.width / per) + p.x / per];
  const float q = float((word >> ((p.x % per) * a.bits)) & mask);
  const size_t g = row * (a.width / a.group) + p.x / a.group;
  R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));
}

// Gemma's rms norm of fp32 values: bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)), one rounding.
inline void mm_norm_row(const device float* x, const device bfloat* w, uint width, float eps, device bfloat* out,
    uint t, uint n, threadgroup float* part) {
  float sum = 0.0f;
  for (uint i = t; i < width; i += n) sum = fma(x[i], x[i], sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(width) + eps);
  for (uint i = t; i < width; i += n) out[i] = bfloat(x[i] * inv * (1.0f + float(w[i])));
}

[[kernel]] void mm_norm(const device float* R [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  mm_norm_row(R + (size_t)row * a.width, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part);
}

// R = bf16(R + X), then the next norm of R: a sublayer's output into the residual.
[[kernel]] void mm_add_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant mm_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  for (uint i = t; i < a.width; i += n) x[i] = float(bfloat(x[i] + float(X[(size_t)row * a.width + i])));
  threadgroup_barrier(mem_flags::mem_device);
  mm_norm_row(x, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part);
}

// R = bf16(R + X) alone (the last layer's output; the final norm runs on chosen rows).
[[kernel]] void mm_add(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] = float(bfloat(R[i] + float(X[i])));
}

// torch's sigmoid of an fp32 value (precise exp of -|x|, the same on either side).
inline float mm_sigmoid(float x) {
  const float e = precise::exp(-fabs(x)), y = 1.0f / (1.0f + e);
  return x < 0.0f ? 1.0f - y : y;
}

// SwiGLU-OAI of a gate|up row in bf16 steps: (clamp(up) + 1) * g * sigmoid(g * alpha), g = min(gate, limit).
[[kernel]] void mm_swiglu(const device bfloat* GU [[buffer(0)]], constant mm_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * a.stride + p.x;
  const float g = min(float(GU[at]), a.limit);
  const float u = clamp(float(GU[at + a.width]), -a.limit, a.limit);
  const float s = float(bfloat(mm_sigmoid(float(bfloat(g * a.alpha)))));
  const float glu = float(bfloat(g * s));
  A[(size_t)p.y * a.width + p.x] = bfloat(float(bfloat(u + 1.0f)) * glu);
}

// The router's fp32 logits: L[r, e] = sum_d float(X[r, d]) * W[e, d], 16 rows a threadgroup, an expert a threadgroup.
[[kernel]] void mm_router(const device bfloat* X [[buffer(0)]], const device float* W [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], device float* L [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint RB = 16;
  threadgroup float part[RB][8];
  const uint e = tg.x, r0 = tg.y * RB, nr = min(RB, a.rows - r0);
  const device float* w = W + (size_t)e * a.width;
  float acc[RB];
  for (uint r = 0; r < RB; ++r) acc[r] = 0.0f;
  for (uint d = 4 * t; d < a.width; d += 4 * 256) {
    const float4 wv = *(const device float4*)(w + d);
    for (uint r = 0; r < nr; ++r) acc[r] += dot(wv, float4(*(const device bfloat4*)(X + (size_t)(r0 + r) * a.width + d)));
  }
  for (uint r = 0; r < nr; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0) part[r][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < nr) {
    float s = 0.0f;
    for (uint j = 0; j < 8; ++j) s += part[t][j];
    L[(size_t)(r0 + t) * a.heads + e] = s;
  }
}

// A row's pick: sigmoid scores, top-k of score + bias (ties to the lower id), weights = score / their sum.
inline void mm_pick_row(const device float* L, const device float* bias, constant mm_args& a, device int* PICK,
    device float* WTS, uint row, uint lane);

// A simdgroup a row (a threadgroup each).
[[kernel]] void mm_route_pick(const device float* L [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], device int* PICK [[buffer(3)]], device float* WTS [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  mm_pick_row(L, bias, a, PICK, WTS, row, lane);
}

// A decode window's picks, a simdgroup a row, then its slots by expert as experts.metal reads them (G, M).
[[kernel]] void mm_route_groups(const device float* L [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], device int* PICK [[buffer(3)]], device float* WTS [[buffer(4)]],
    device uint* G [[buffer(5)]], device uint* M [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  if (sg < a.rows) mm_pick_row(L, bias, a, PICK, WTS, sg, lane);
  threadgroup_barrier(mem_flags::mem_device);
  if (t != 0) return;
  const uint slots = a.rows * a.heads;
  uint n = 0, at = 0;
  int last = -1;
  while (true) {
    int next = 0x7fffffff;
    for (uint s = 0; s < slots; ++s) if (PICK[s] > last && PICK[s] < next) next = PICK[s];
    if (next == 0x7fffffff) break;
    G[1 + 2 * n] = uint(next);
    G[2 + 2 * n] = at;
    for (uint s = 0; s < slots; ++s) if (PICK[s] == next) M[at++] = s;
    last = next;
    n += 1;
  }
  G[0] = n;
}

inline void mm_pick_row(const device float* L, const device float* bias, constant mm_args& a, device int* PICK,
    device float* WTS, uint row, uint lane) {
  constexpr uint PER = 8; // experts a lane: a.width <= 256
  float sc[PER], ch[PER];
  uint taken = 0;
  for (uint j = 0; j < PER; ++j) {
    const uint id = j * 32 + lane;
    if (id < a.width) {
      sc[j] = mm_sigmoid(L[(size_t)row * a.width + id]);
      ch[j] = sc[j] + bias[id];
    } else {
      sc[j] = ch[j] = 0.0f;
      taken |= 1u << j;
    }
  }
  float w[8];
  for (uint k = 0; k < a.heads; ++k) {
    float best = -INFINITY, bw = 0.0f;
    uint bid = a.width;
    for (uint j = 0; j < PER; ++j) {
      const uint id = j * 32 + lane;
      if (!(taken & (1u << j)) && (ch[j] > best || (ch[j] == best && id < bid))) { best = ch[j]; bw = sc[j]; bid = id; }
    }
    for (ushort off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off), ow = simd_shuffle_xor(bw, off);
      const uint oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bw = ow; bid = oi; }
    }
    w[k] = bw;
    if (bid % 32 == lane) taken |= 1u << (bid / 32);
    if (lane == 0) PICK[row * a.heads + k] = int(bid);
  }
  if (lane != 0) return;
  float total = 0.0f;
  for (uint k = 0; k < a.heads; ++k) total += w[k];
  // written in ascending expert id (the combine's order), each weight its own pick's
  uint ids[8];
  for (uint k = 0; k < a.heads; ++k) ids[k] = uint(PICK[row * a.heads + k]);
  for (uint k = 0; k < a.heads; ++k) w[k] = w[k] / total;
  for (uint j = 1; j < a.heads; ++j)
    for (uint m = j; m > 0 && ids[m] < ids[m - 1]; --m) {
      const uint ti = ids[m]; ids[m] = ids[m - 1]; ids[m - 1] = ti;
      const float tw = w[m]; w[m] = w[m - 1]; w[m - 1] = tw;
    }
  for (uint k = 0; k < a.heads; ++k) {
    PICK[row * a.heads + k] = int(ids[k]);
    WTS[row * a.heads + k] = w[k];
  }
}

// transformers' expert loop: picks by id, acc = bf16(acc + bf16(y * w)), then bf16(bf16(acc * scale) + shared).
inline float mm_moe_out(const device bfloat* Y, const device int* pick, const device float* wts,
    const device bfloat* shared, constant mm_args& a, uint row, uint i) {
  const uint k = a.heads; // the picks are in ascending expert id (mm_pick_row writes them so)
  (void)pick;
  float acc = 0.0f;
  for (uint j = 0; j < k; ++j) {
    const uint s = row * k + j;
    acc = float(bfloat(acc + float(bfloat(float(Y[(size_t)s * a.width + i]) * wts[s]))));
  }
  return float(bfloat(float(bfloat(acc * a.scale)) + float(shared[(size_t)row * a.width + i])));
}

// R = bf16(R + moe), then the next layer's input norm of R.
[[kernel]] void mm_moe_add_norm(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device int* pick [[buffer(2)]], const device float* wts [[buffer(3)]], const device bfloat* shared [[buffer(4)]],
    const device bfloat* w [[buffer(5)]], constant mm_args& a [[buffer(6)]], device bfloat* out [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  for (uint i = t; i < a.width; i += n) x[i] = float(bfloat(x[i] + mm_moe_out(Y, pick, wts, shared, a, row, i)));
  threadgroup_barrier(mem_flags::mem_device);
  mm_norm_row(x, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part);
}

[[kernel]] void mm_moe_add(device float* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device int* pick [[buffer(2)]], const device float* wts [[buffer(3)]], const device bfloat* shared [[buffer(4)]],
    constant mm_args& a [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  R[i] = float(bfloat(R[i] + mm_moe_out(Y, pick, wts, shared, a, p.y, p.x)));
}

// A chunk's a.rows picks by expert (a.width of them): offsets[e] its first pair, order the pairs, each in pick order.
[[kernel]] void mm_sort(const device int* pick [[buffer(0)]], constant mm_args& a [[buffer(1)]],
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
[[kernel]] void mm_rows_by_order(const device bfloat* X [[buffer(0)]], const device int* order [[buffer(1)]],
    constant mm_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x < a.width && p.y < a.rows) Y[(size_t)p.y * a.width + p.x] = X[(size_t)(order[p.y] / int(a.heads)) * a.width + p.x];
}

struct mm_prep_args { uint rows, heads, kv_heads, index_heads, dim, pos, rotary, stride; float eps, theta; };

// An int8 cache row: a head's 128 values (four a lane) as round(x / s), s = max |x| / 127 (an all-zero row s = 0).
inline void mm_put8(thread const float* v, device char* q, device float* scale, uint lane) {
  float m = 0.0f;
  for (uint j = 0; j < 4; ++j) m = max(m, fabs(v[j]));
  m = simd_max(m);
  const float s = m / 127.0f, inv = m > 0.0f ? 127.0f / m : 0.0f;
  for (uint j = 0; j < 4; ++j) q[lane + 32 * j] = char(rint(v[j] * inv));
  if (lane == 0) *scale = s;
}

// KV8: keys and values to int8 rows with a scale a (slot, head) in KS and VS; else bf16 rows.
template <bool KV8>
[[kernel]] void mm_prep(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], const device bfloat* iqw [[buffer(3)]], const device bfloat* ikw [[buffer(4)]],
    constant mm_prep_args& a [[buffer(5)]], device bfloat* QN [[buffer(6)]], device bfloat* IQ [[buffer(7)]],
    device uchar* KC [[buffer(8)]], device uchar* VC [[buffer(9)]], device bfloat* IK [[buffer(10)]],
    device float* KS [[buffer(11)]], device float* VS [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  const uint nq = a.heads, nk = a.kv_heads, ni = a.index_heads;
  if (unit >= nq + nk + ni + 1 || row >= a.rows) return;
  const uint D = a.dim; // 128: four values a lane
  const device bfloat* src = X + (size_t)row * a.stride;
  const device bfloat* w;
  device bfloat* y = nullptr; // a bf16 row's place; an int8 key's is KC's
  const size_t slot = a.pos + row;
  const bool key = unit >= nq && unit < nq + nk;
  uint head;
  if (unit < nq) {
    head = unit;
    src += (size_t)head * D;
    w = qw;
    y = QN + ((size_t)row * nq + head) * D;
  } else if (key) {
    head = unit - nq;
    src += (size_t)(nq + head) * D;
    w = kw;
    if (!KV8) y = (device bfloat*)KC + (slot * nk + head) * D;
  } else if (unit < nq + nk + ni) {
    head = unit - nq - nk;
    src += (size_t)(nq + 2 * nk + head) * D;
    w = iqw;
    y = IQ + ((size_t)row * ni + head) * D;
  } else {
    head = 0;
    src += (size_t)(nq + 2 * nk + ni) * D;
    w = ikw;
    y = IK + slot * D;
  }
  float v[4];
  float sum = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / float(D) + a.eps);
  threadgroup float nrm[128];
  for (uint j = 0; j < 4; ++j) nrm[lane + 32 * j] = float(bfloat(v[j] * inv * (1.0f + float(w[lane + 32 * j]))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_r = a.rotary / 2;
  const float position = float(a.pos + row);
  float out[4];
  for (uint j = 0; j < 4; ++j) {
    const uint d = lane + 32 * j;
    if (d >= a.rotary) { out[j] = float(bfloat(nrm[d])); continue; }
    const uint i = d % half_r;
    const float freq = 1.0f / precise::pow(a.theta, float(2 * i) / float(a.rotary));
    const float angle = position * freq;
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float rot = d < half_r ? -nrm[d + half_r] : nrm[d - half_r];
    out[j] = float(bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s))));
  }
  if (KV8 && key) mm_put8(out, (device char*)KC + (slot * nk + head) * D, KS + slot * nk + head, lane);
  else for (uint j = 0; j < 4; ++j) y[lane + 32 * j] = bfloat(out[j]);
  if (!key) return;
  const device bfloat* vs = X + (size_t)row * a.stride + (size_t)(nq + nk + head) * D;
  if (KV8) {
    float vv[4];
    for (uint j = 0; j < 4; ++j) vv[j] = float(vs[lane + 32 * j]);
    mm_put8(vv, (device char*)VC + (slot * nk + head) * D, VS + slot * nk + head, lane);
    return;
  }
  device bfloat* vd = (device bfloat*)VC + (slot * nk + head) * D;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

template [[host_name("mm_prep")]] [[kernel]] decltype(mm_prep<false>) mm_prep<false>;
template [[host_name("mm_prep8")]] [[kernel]] decltype(mm_prep<true>) mm_prep<true>;

struct mm_index_args { uint rows, pos, heads, blocks, first_row; };

// Indexer scores: S[row, h, j] = max over block j's keys up to the row of IQ[row, h] . IK[k]; 128 keys by 16 rows.
[[kernel]] void mm_index_scores(const device bfloat* IQ [[buffer(0)]], const device bfloat* IK [[buffer(1)]],
    constant mm_index_args& a [[buffer(2)]], device float* S [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint RB = 16, H = 4, D = 128;
  threadgroup bfloat q[RB * H * D];
  threadgroup float part[4][RB * H];
  const uint j = tg.x, r0 = tg.y * RB, nr = min(RB, a.rows - r0);
  for (uint i = t; i < nr * H * D; i += 128) q[i] = IQ[(size_t)r0 * H * D + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint key = j * 128 + t;
  const device bfloat* k = IK + (size_t)key * D;
  float acc[RB * H];
  for (uint i = 0; i < RB * H; ++i) acc[i] = 0.0f;
  const bool any = key <= a.pos + r0 + nr - 1;
  if (any) {
    for (uint d = 0; d < D; d += 8) {
      const float4 k0 = float4(*(const device bfloat4*)(k + d)), k1 = float4(*(const device bfloat4*)(k + d + 4));
      for (uint r = 0; r < nr; ++r)
        for (uint h = 0; h < H; ++h) {
          const threadgroup bfloat* qq = q + (r * H + h) * D + d;
          acc[r * H + h] += dot(k0, float4(*(const threadgroup bfloat4*)qq)) + dot(k1, float4(*(const threadgroup bfloat4*)(qq + 4)));
        }
    }
  }
  for (uint r = 0; r < nr; ++r)
    for (uint h = 0; h < H; ++h) {
      const float s = simd_max(key <= a.pos + r0 + r ? acc[r * H + h] : -INFINITY);
      if (lane == 0) part[sg][r * H + h] = s;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < nr * H) {
    const float s = max(max(part[0][t], part[1][t]), max(part[2][t], part[3][t]));
    S[((size_t)(r0 + t / H) * a.heads + t % H) * a.blocks + j] = s;
  }
}

// A (row, key head)'s kept blocks ascending: all if at most topk, else its own and the best (ties low); and a bitmap.
[[kernel]] void mm_index_select(const device float* S [[buffer(0)]], constant mm_index_args& a [[buffer(1)]],
    constant uint& topk [[buffer(2)]], device int* SEL [[buffer(3)]], device uint* BITS [[buffer(4)]],
    constant uint& words [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.y, h = tg.x;
  if (row >= a.rows) return;
  const uint nb = (a.pos + row) / 128 + 1; // blocks the row sees; the last is its own
  const device float* s = S + ((size_t)row * a.heads + h) * a.blocks;
  device int* out = SEL + ((size_t)row * a.heads + h) * topk;
  int list[32];
  uint count = 0;
  if (nb <= topk) {
    for (uint b = 0; b < nb; ++b) list[count++] = int(b);
  } else {
    list[count++] = int(nb - 1);
    float last_best = INFINITY;
    uint last_id = nb - 1;
    for (uint k = 1; k < topk; ++k) {
      // the best (score, id) strictly after (last_best, last_id) in the order score desc, id asc
      float best = -INFINITY;
      uint bid = 0xffffffffu;
      for (uint b = lane; b < nb - 1; b += 32) {
        const float v = s[b];
        const bool after = v < last_best || (v == last_best && b > last_id);
        if (after && (v > best || (v == best && b < bid))) { best = v; bid = b; }
      }
      for (ushort off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const uint oi = simd_shuffle_xor(bid, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
      }
      if (bid == 0xffffffffu || best == -INFINITY) break;
      list[count++] = int(bid);
      last_best = best;
      last_id = bid;
    }
  }
  if (lane == 0) {
    for (uint i = 1; i < count; ++i)
      for (uint m = i; m > 0 && list[m] < list[m - 1]; --m) { const int t = list[m]; list[m] = list[m - 1]; list[m - 1] = t; }
    for (uint i = 0; i < topk; ++i) out[i] = i < count ? list[i] : -1;
  }
  if (words == 0) return;
  device uint* bits = BITS + ((size_t)row * a.heads + h) * words;
  for (uint wd = lane; wd < words; wd += 32) {
    uint m = 0;
    for (uint i = 0; i < count; ++i) if (uint(list[i]) / 32 == wd) m |= 1u << (uint(list[i]) % 32);
    bits[wd] = m;
  }
}

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct mm_vec_args { uint rows, pos, heads, groups, topk; float scale; uint sel, all; };

// Decode attention partials: a threadgroup a (key head, row, 32-key quarter of a kept block), a simdgroup a query head.
template <bool KV8>
[[kernel]] void mm_decode_vec(const device bfloat* Q [[buffer(0)]], const device uchar* K [[buffer(1)]],
    const device uchar* V [[buffer(2)]], constant mm_vec_args& a [[buffer(3)]], const device int* SEL [[buffer(4)]],
    device float* PM [[buffer(5)]], device float* PL [[buffer(6)]], device float* PO [[buffer(7)]],
    const device float* KS [[buffer(8)]], const device float* VS [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, z = tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const size_t slot = ((size_t)head * a.rows + row) * a.topk + z;
  const uint p = a.pos + row;
  const uint pick = z / 4, quarter = z % 4;
  const int block = a.all ? (pick <= p / 128 ? int(pick) : -1) : SEL[((size_t)row * a.groups + kv) * a.sel + pick];
  const uint from = block < 0 ? 0 : uint(block) * 128 + quarter * 32, to = block < 0 ? 0 : min(p + 1, from + 32);
  if (from >= to) {
    if (lane == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    return;
  }
  const size_t ld = (size_t)a.groups * 128;
  const float4 qv = a.scale * float4(*(const device bfloat4*)(Q + ((size_t)row * a.heads + head) * 128 + 4 * lane));
  float4 acc = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 vr[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to; // past the row's position: a masked key (read at the block's first, weight 0)
      const size_t key = (size_t)(live ? i0 + u : from), at = key * ld + kv * 128 + 4 * lane;
      float sc;
      if (KV8) { // int8 rows: q . (s k8) = s (q . k8), and s v8
        vr[u] = VS[key * a.groups + kv] * float4(*(const device char4*)(V + at));
        sc = KS[key * a.groups + kv] * simd_sum(dot(qv, float4(*(const device char4*)(K + at))));
      } else {
        vr[u] = float4(*(const device bfloat4*)((const device bfloat*)V + at));
        sc = simd_sum(dot(qv, float4(*(const device bfloat4*)((const device bfloat*)K + at))));
      }
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
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * 128 + 4 * lane) = acc;
}

template [[host_name("mm_decode_vec")]] [[kernel]] decltype(mm_decode_vec<false>) mm_decode_vec<false>;
template [[host_name("mm_decode_vec8")]] [[kernel]] decltype(mm_decode_vec<true>) mm_decode_vec<true>;

// The partials' merge in block order: a simdgroup a (head, row), the row's max first, then its mass and outputs.
[[kernel]] void mm_merge_vec(const device float* PM [[buffer(0)]], const device float* PL [[buffer(1)]],
    const device float* PO [[buffer(2)]], constant mm_vec_args& a [[buffer(3)]], device bfloat* O [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const size_t at = (size_t)head * a.rows + row;
  const device float* ms = PM + at * a.topk;
  const device float* ls = PL + at * a.topk;
  float top = -INFINITY;
  for (uint b = lane; b < a.topk; b += 32) top = max(top, ms[b]);
  top = simd_max(top);
  float total = 0.0f;
  float4 acc = 0.0f;
  for (uint b = 0; b < a.topk; ++b) {
    if (ls[b] <= 0.0f) continue;
    const float f = fast::exp(ms[b] - top);
    total += ls[b] * f;
    acc += f * *(const device float4*)(PO + (at * a.topk + b) * 128 + 4 * lane);
  }
  acc /= total;
  *(device bfloat4*)(O + (row * a.heads + head) * 128 + 4 * lane) = bfloat4(acc);
}

inline bool mm_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// The greedy draw: a bf16 logits row's argmax, ties to the lower id, into ids[0].
[[kernel]] void mm_argmax(const device bfloat* logits [[buffer(0)]], device uint* ids [[buffer(1)]],
    constant uint& vocab [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint idxs[32];
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(logits[i]);
    if (mm_better(v, i, best, at)) { best = v; at = i; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (mm_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  best = vals[lane];
  at = idxs[lane];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (mm_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) ids[0] = at;
}


// bf16 rows into fp32 ones (n values): an image's rows over their placeholders' embeddings, the tower's pre-norm out.
[[kernel]] void mm_widen(const device bfloat* X [[buffer(0)]], device float* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) Y[i] = float(X[i]);
}

// fp32 rows to bf16 ones (n values): the tower's last residual for the projector.
[[kernel]] void mm_narrow(const device float* X [[buffer(0)]], device bfloat* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) Y[i] = bfloat(X[i]);
}
