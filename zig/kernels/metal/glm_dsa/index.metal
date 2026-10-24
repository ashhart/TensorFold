// GLM-5.3's DSA indexer for lane rows: bf16 index keys, every visible key's score, each row's top-k by position.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint IH = 32, ID = 128, IROPE = 64, TOPK = 2048;

// A row's cache slot, position and key list (the top-k writes `nkeys`; shared layers read it).
struct IndexRow {
  uint slot, pos, list, nkeys;
};
struct IndexArgs {
  uint rows, slot_keys, proj_stride, k_col;
  uint w_col, q_stride, score_stride, pad0;
  float eps, key_scale, head_scale, pad1;
};

inline float glm_index_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

inline float2 glm_index_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

// One row a simdgroup: the key = bf16(rope_64(LayerNorm(wk x))) into the index cache at the row's position.
kernel void glm_index_key(const device float* proj [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                          const device bfloat* b [[buffer(2)]], const device float* inv [[buffer(3)]],
                          const device IndexRow* rows [[buffer(4)]], constant IndexArgs& a [[buffer(5)]],
                          device bfloat* cache [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const device float* x = proj + ulong(row) * a.proj_stride + a.k_col;
  float v[4], s = 0.0f;
  for (uint q = 0; q < 4; ++q) v[q] = x[lane + 32 * q], s += v[q];
  const float mean = glm_index_lanes(s) / float(ID);
  float d = 0.0f;
  for (uint q = 0; q < 4; ++q) d = fma(v[q] - mean, v[q] - mean, d);
  const float rstd = precise::rsqrt(glm_index_lanes(d) / float(ID) + a.eps);
  threadgroup float y[ID];
  for (uint q = 0; q < 4; ++q) {
    const uint i = lane + 32 * q;
    y[i] = fma((v[q] - mean) * rstd, float(w[i]), float(b[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const IndexRow r = rows[row];
  device bfloat* key = cache + (ulong(r.slot) * a.slot_keys + r.pos) * ID;
  const float2 o = glm_index_rope(y[2 * lane], y[2 * lane + 1], r.pos, inv, lane);
  key[lane] = bfloat(o.x);
  key[IROPE / 2 + lane] = bfloat(o.y);
  key[IROPE + lane] = bfloat(y[IROPE + lane]);
  key[IROPE + 32 + lane] = bfloat(y[IROPE + 32 + lane]);
}

// Row r, keys 256 tg.x ..: score = sum_h (w_h / sqrt 32) relu(q_h . k / sqrt 128), heads in order, q roped in place.
kernel void glm_index_scores(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                             const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                             constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                             const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
  threadgroup float qs[IH * ID];
  threadgroup float ws[IH];
  const uint r = tg.y;
  const IndexRow row = rows[r];
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint i = t; i < IH * ID; i += 256) {
    const uint h = i / ID, c = i % ID;
    if (c >= IROPE) {
      qs[i] = qr[i];
    } else if (c < IROPE / 2) {
      const float2 o = glm_index_rope(qr[h * ID + 2 * c], qr[h * ID + 2 * c + 1], row.pos, inv, c);
      qs[h * ID + c] = o.x;
      qs[h * ID + IROPE / 2 + c] = o.y;
    }
  }
  if (t < IH) ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = tg.x * 256 + t;
  if (k > row.pos) return;
  const device bfloat* key = cache + (ulong(row.slot) * a.slot_keys + k) * ID;
  float kv[ID];
  for (uint i = 0; i < ID; ++i) kv[i] = float(key[i]);
  float s = 0.0f;
  for (uint h = 0; h < IH; ++h) {
    float d = 0.0f;
    for (uint i = 0; i < ID; ++i) d = fma(qs[h * ID + i], kv[i], d);
    s = fma(ws[h], max(d * a.key_scale, 0.0f), s);
  }
  score[ulong(r) * a.score_stride + k] = s;
}

// The keys a thread scores at once: the query is read once for the block, not once a key.
constant constexpr uint BLOCK = 2;
constant constexpr uint THREADS = 128;

// Row r's score for each of the thread's keys: the head, dimension and sum order are the row kernel's.
kernel void glm_index_scores_block(const device float* q [[buffer(0)]], const device float* proj [[buffer(1)]],
                                   const device bfloat* cache [[buffer(2)]], const device IndexRow* rows [[buffer(3)]],
                                   constant IndexArgs& a [[buffer(4)]], device float* score [[buffer(5)]],
                                   const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
  threadgroup float qs[IH * ID];
  threadgroup float ws[IH];
  const uint r = tg.y;
  const IndexRow row = rows[r];
  const device float* qr = q + ulong(r) * a.q_stride;
  for (uint i = t; i < IH * ID; i += THREADS) {
    const uint h = i / ID, c = i % ID;
    if (c >= IROPE) {
      qs[i] = qr[i];
    } else if (c < IROPE / 2) {
      const float2 o = glm_index_rope(qr[h * ID + 2 * c], qr[h * ID + 2 * c + 1], row.pos, inv, c);
      qs[h * ID + c] = o.x;
      qs[h * ID + IROPE / 2 + c] = o.y;
    }
  }
  if (t < IH) ws[t] = proj[ulong(r) * a.proj_stride + a.w_col + t] * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kbase = tg.x * (THREADS * BLOCK) + t;
  float d[BLOCK][IH];
  for (uint b = 0; b < BLOCK; ++b)
    for (uint h = 0; h < IH; ++h) d[b][h] = 0.0f;
  float kv[BLOCK];
  const device bfloat* kb[BLOCK];
  uint kk[BLOCK];
  for (uint b = 0; b < BLOCK; ++b) {
    kk[b] = kbase + b * THREADS;
    kb[b] = cache + (ulong(row.slot) * a.slot_keys + kk[b]) * ID;
  }
  for (uint i = 0; i < ID; ++i) {
    for (uint b = 0; b < BLOCK; ++b) kv[b] = kk[b] <= row.pos ? float(kb[b][i]) : 0.0f;
    for (uint h = 0; h < IH; ++h) {
      const float qv = qs[h * ID + i];
      for (uint b = 0; b < BLOCK; ++b) d[b][h] = fma(qv, kv[b], d[b][h]);
    }
  }
  for (uint b = 0; b < BLOCK; ++b) {
    const uint k = kk[b];
    if (k > row.pos) continue;
    float s = 0.0f;
    for (uint h = 0; h < IH; ++h) s = fma(ws[h], max(d[b][h] * a.key_scale, 0.0f), s);
    score[ulong(r) * a.score_stride + k] = s;
  }
}

// fp32 as an unsigned key whose order is the floats' order.
inline uint glm_order(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One row a threadgroup of 1024: its top 2048 visible keys (ties to the lower position), listed by position.
kernel void glm_index_topk(const device float* score [[buffer(0)]], device IndexRow* rows [[buffer(1)]],
                           constant IndexArgs& a [[buffer(2)]], device uint* list [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint found[2];
  threadgroup uint sums[2][32];
  const IndexRow row = rows[r];
  const uint n = row.pos + 1;
  device uint* out = list + row.list;
  const device float* sc = score + ulong(r) * a.score_stride;
  if (n <= TOPK) {
    for (uint i = t; i < n; i += 1024) out[i] = i;
    if (t == 0) rows[r].nkeys = n;
    return;
  }
  uint prefix = 0, mask = 0, need = TOPK;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 1024) {
      const uint k = glm_order(sc[i]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint above = 0, b = 255;
      for (;; --b) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (above + c >= need || b == 0) break;
        above += c;
      }
      found[0] = b;
      found[1] = need - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= found[0] << shift;
    mask |= 255u << shift;
    need = found[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // keys above the threshold all go; `need` of those equal to it, the lowest positions first
  uint gt_seen = 0, eq_seen = 0;
  for (uint base = 0; base < n; base += 1024) {
    const uint i = base + t;
    const uint k = i < n ? glm_order(sc[i]) : 0u;
    const uint gt = (i < n && k > prefix) ? 1u : 0u, eq = (i < n && k == prefix) ? 1u : 0u;
    const uint gx = simd_prefix_exclusive_sum(gt), ex = simd_prefix_exclusive_sum(eq);
    if (lane == 31) sums[0][sg] = gx + gt, sums[1][sg] = ex + eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gb = 0, eb = 0, gall = 0, eall = 0;
    for (uint g = 0; g < 32; ++g) {
      if (g < sg) gb += sums[0][g], eb += sums[1][g];
      gall += sums[0][g], eall += sums[1][g];
    }
    const uint eq_before = eq_seen + eb + ex;
    const uint take = gt | ((eq && eq_before < need) ? 1u : 0u);
    if (take) out[gt_seen + gb + gx + min(eq_before, need)] = i;
    gt_seen += gall;
    eq_seen += eall;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) rows[r].nkeys = TOPK;
}
