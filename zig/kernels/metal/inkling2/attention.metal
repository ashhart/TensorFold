// Inkling 2 flash attention fp32: logit (q.k)*scale + bf16(rel[p].proj[:,p-k]); sink dropped; merge inkling2_attn_merge

constant constexpr uint I2_DIM = 128;
constant constexpr uint I2_DREL = 16;
constant constexpr uint I2_CHUNK = 2048;
constant constexpr uint I2_PART = 132; // floats of a chunk's partial: m, l, two unused, acc[128]
constant constexpr uint I2_ATTN_SGS = 4;

struct I2AttnArgs {
  uint rows, pos0, heads, kv_heads;
  uint extent, window, ring, chunks;
  float scale;
  uint pad0, pad1, pad2;
};

inline ulong i2_attn_lo(constant I2AttnArgs& a, ulong p) { return (a.window != 0 && p + 1 > a.window) ? p + 1 - a.window : 0; }

inline ulong i2_attn_slot(constant I2AttnArgs& a, ulong key) { return a.ring == 0 ? key : key % a.ring; }

// m, l, A of one (row, head) over its chunks to the bf16 gated output.
inline bfloat i2_attn_finish(float M, float L, float A, float gate) {
  const float o = (isnan(M) || M == -INFINITY) ? NAN : A / L;
  return bfloat(float(bfloat(o)) * i2_sigmoid(gate));
}

template <uint G, bool FINAL>
inline void i2_attn_chunk(const device bfloat* q, const device bfloat* rel, const device bfloat* sink,
                          const device bfloat* gate, const device bfloat* kcache, const device bfloat* vcache,
                          const device bfloat* proj, device bfloat* out, device float* scratch,
                          constant I2AttnArgs& a, uint3 tg, uint tid, uint sg, uint lane, threadgroup float* q_tg,
                          threadgroup float* rel_tg, threadgroup float* mrg_m, threadgroup float* mrg_l,
                          threadgroup float* mrg_acc) {

  const uint chunk = tg.x, g = tg.y, r = tg.z;
  const ulong p = ulong(a.pos0) + r;
  const ulong lo = i2_attn_lo(a, p);
  const ulong nch = (p - lo + I2_CHUNK) / I2_CHUNK;
  if (chunk >= nch) return;
  const ulong ks = lo + ulong(chunk) * I2_CHUNK;
  const ulong ke = min(p + 1, ks + I2_CHUNK);
  const ulong head0 = ulong(r) * a.heads + ulong(g) * G;

  for (uint i = tid; i < G * I2_DIM; i += 128) q_tg[i] = float(q[head0 * I2_DIM + i]);
  for (uint i = tid; i < G * I2_DREL; i += 128) rel_tg[i] = float(rel[head0 * I2_DREL + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m[G], l[G];
  float4 acc[G];
  for (uint h = 0; h < G; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.0f;
    acc[h] = float4(0.0f);
    if (chunk == 0 && sg == 0) {
      m[h] = float(sink[head0 + h]);
      l[h] = m[h] == -INFINITY ? 0.0f : 1.0f;
    }
  }

  const ulong kv_row = ulong(a.kv_heads) * I2_DIM;
  for (ulong block = sg; block * 32 < ke - ks; block += I2_ATTN_SGS) {
    const ulong kk0 = ks + block * 32;
    const uint nvalid = uint(min(ulong(32), ke - kk0));
    const bool valid = lane < nvalid;
    const ulong key = kk0 + lane;
    float score[G];
    for (uint h = 0; h < G; ++h) score[h] = 0.0f;
    if (valid) {
      const device bfloat4* kp = (const device bfloat4*)(kcache + i2_attn_slot(a, key) * kv_row + ulong(g) * I2_DIM);
      for (uint d4 = 0; d4 < I2_DIM / 4; ++d4) {
        const float4 kf = float4(kp[d4]);
        for (uint h = 0; h < G; ++h) {
          const threadgroup float* qh = q_tg + h * I2_DIM + d4 * 4;
          score[h] = fma(qh[0], kf.x, score[h]);
          score[h] = fma(qh[1], kf.y, score[h]);
          score[h] = fma(qh[2], kf.z, score[h]);
          score[h] = fma(qh[3], kf.w, score[h]);
        }
      }
      const ulong dist = p - key;
      // Separate in-place scale and bias loops miscompile G=5 on M4. Keep the multiply and bias in one store.
      if (dist < a.extent) {
        float pr[I2_DREL];
        for (uint i = 0; i < I2_DREL; ++i) pr[i] = float(proj[ulong(i) * a.extent + dist]);
        for (uint h = 0; h < G; ++h) {
          float b = 0.0f;
          for (uint i = 0; i < I2_DREL; ++i) b = fma(rel_tg[h * I2_DREL + i], pr[i], b);
          score[h] = score[h] * a.scale + float(bfloat(b));
        }
      } else {
        for (uint h = 0; h < G; ++h) score[h] = score[h] * a.scale;
      }
    }
    float pw[G];
    for (uint h = 0; h < G; ++h) {
      const bool bad = simd_any(valid && isnan(score[h]));
      const float top = simd_max((valid && !isnan(score[h])) ? score[h] : -INFINITY);
      const float m_new = bad ? NAN : i2_max(m[h], top);
      const float alpha = m[h] == -INFINITY ? 0.0f : metal::precise::exp(m[h] - m_new);
      pw[h] = !valid ? 0.0f : (score[h] == -INFINITY ? 0.0f : metal::precise::exp(score[h] - m_new));
      l[h] = l[h] * alpha + simd_sum(pw[h]);
      acc[h] = acc[h] * alpha;
      m[h] = m_new;
    }
    for (uint j = 0; j < nvalid; ++j) {
      const device bfloat4* vp = (const device bfloat4*)(vcache + i2_attn_slot(a, kk0 + j) * kv_row + ulong(g) * I2_DIM);
      const float4 vf = float4(vp[lane]);
      for (uint h = 0; h < G; ++h) acc[h] = fma(float4(simd_shuffle(pw[h], ushort(j))), vf, acc[h]);
    }
  }

  for (uint h = 0; h < G; ++h) {
    if (lane == 0) {
      mrg_m[sg * G + h] = m[h];
      mrg_l[sg * G + h] = l[h];
    }
    for (uint c = 0; c < 4; ++c) mrg_acc[(sg * G + h) * I2_DIM + lane * 4 + c] = acc[h][c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint d = tid;
  for (uint h = 0; h < G; ++h) {
    float M = -INFINITY;
    for (uint s = 0; s < I2_ATTN_SGS; ++s) M = i2_max(M, mrg_m[s * G + h]);
    float L = 0.0f, A = 0.0f;
    for (uint s = 0; s < I2_ATTN_SGS; ++s) {
      const float ms = mrg_m[s * G + h];
      const float e = ms == -INFINITY ? 0.0f : metal::precise::exp(ms - M);
      L += mrg_l[s * G + h] * e;
      A += mrg_acc[(s * G + h) * I2_DIM + d] * e;
    }
    if (FINAL) {
      const ulong at = (head0 + h) * I2_DIM + d;
      out[at] = i2_attn_finish(M, L, A, float(gate[at]));
    } else {
      device float* rec = scratch + ((head0 + h) * a.chunks + chunk) * I2_PART;
      rec[4 + d] = A;
      if (d == 0) {
        rec[0] = M;
        rec[1] = L;
      }
    }
  }
}

#define I2_ATTN_KERNEL(NAME, G, FINAL)                                                                              \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                      \
      const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],                                 \
      const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]],                              \
      const device bfloat* kcache [[buffer(4)]], const device bfloat* vcache [[buffer(5)]],                          \
      const device bfloat* proj [[buffer(6)]], device bfloat* out [[buffer(7)]],                                     \
      device float* scratch [[buffer(8)]], constant I2AttnArgs& a [[buffer(9)]],                                     \
      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                          \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                         \
    threadgroup float q_tg[G * I2_DIM];                                                                              \
    threadgroup float rel_tg[G * I2_DREL];                                                                           \
    threadgroup float mrg_m[I2_ATTN_SGS * G], mrg_l[I2_ATTN_SGS * G];                                                \
    threadgroup float mrg_acc[I2_ATTN_SGS * G * I2_DIM];                                                             \
    i2_attn_chunk<G, FINAL>(q, rel, sink, gate, kcache, vcache, proj, out, scratch, a, tg, tid, sg, lane, q_tg,     \
                            rel_tg, mrg_m, mrg_l, mrg_acc);                                                          \
  }

I2_ATTN_KERNEL(inkling2_attn_partial_g1, 1, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g2, 2, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g3, 3, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g4, 4, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g5, 5, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g6, 6, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g7, 7, false)
I2_ATTN_KERNEL(inkling2_attn_partial_g8, 8, false)
I2_ATTN_KERNEL(inkling2_attn_final_g1, 1, true)
I2_ATTN_KERNEL(inkling2_attn_final_g2, 2, true)
I2_ATTN_KERNEL(inkling2_attn_final_g3, 3, true)
I2_ATTN_KERNEL(inkling2_attn_final_g4, 4, true)
I2_ATTN_KERNEL(inkling2_attn_final_g5, 5, true)
I2_ATTN_KERNEL(inkling2_attn_final_g6, 6, true)
I2_ATTN_KERNEL(inkling2_attn_final_g7, 7, true)
I2_ATTN_KERNEL(inkling2_attn_final_g8, 8, true)

// One threadgroup a (head, row), a thread an output dimension: the chunks' partials in order.
[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_attn_merge(
    const device bfloat* gate [[buffer(0)]], device bfloat* out [[buffer(1)]], const device float* scratch [[buffer(2)]],
    constant I2AttnArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]]) {
  const uint head = tg.x, r = tg.y;
  const ulong p = ulong(a.pos0) + r;
  const ulong lo = i2_attn_lo(a, p);
  const ulong nch = (p - lo + I2_CHUNK) / I2_CHUNK;
  const ulong row_head = ulong(r) * a.heads + head;
  const device float* recs = scratch + row_head * a.chunks * I2_PART;
  float M = -INFINITY;
  for (ulong c = 0; c < nch; ++c) M = i2_max(M, recs[c * I2_PART]);
  float L = 0.0f, A = 0.0f;
  for (ulong c = 0; c < nch; ++c) {
    const float mc = recs[c * I2_PART];
    const float e = mc == -INFINITY ? 0.0f : metal::precise::exp(mc - M);
    L += recs[c * I2_PART + 1] * e;
    A += recs[c * I2_PART + 4 + d] * e;
  }
  const ulong at = row_head * I2_DIM + d;
  out[at] = i2_attn_finish(M, L, A, float(gate[at]));
}
