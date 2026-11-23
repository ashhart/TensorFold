// Inkling 2 full-attention prompt rows: eight rows of a head share each key block on 8x8 matrices

// Bit-equal to inkling2_attn_*_g*: block, m, l, acc and merge orders kept; an 8x8x8 op adds in k order
constant constexpr uint I2_RB = 8; // rows of a tile
constant constexpr uint I2_ROWS_TG_FLOATS = 2048; // 4 simdgroups x (a block's scores and probabilities, a staging)

// A row's blocks from `first` on in one simdgroup's stream, the scalar kernel's arithmetic (G = 1, no ring), in place.
inline void i2_rows_scalar(const device bfloat* kcache, const device bfloat* vcache, const device bfloat* proj,
                           constant I2AttnArgs& a, threadgroup const float* qrow, threadgroup const float* relrow,
                           ulong p, ulong ks, ulong first, ulong kv_off, uint lane, thread float& mi, thread float& li,
                           thread float4& ac) {
  const ulong ke = min(p + 1, ks + I2_CHUNK);
  const ulong kv_row = ulong(a.kv_heads) * I2_DIM;
  for (ulong block = first; block * 32 < ke - ks; block += I2_ATTN_SGS) {
    const ulong kk0 = ks + block * 32;
    const uint nvalid = uint(min(ulong(32), ke - kk0));
    const bool valid = lane < nvalid;
    const ulong key = kk0 + lane;
    float score = 0.0f;
    if (valid) {
      const device uint4* kp = (const device uint4*)(kcache + key * kv_row + kv_off);
      for (uint d8 = 0; d8 < I2_DIM / 8; ++d8) {
        const uint4 kw = kp[d8];
        const float4 kf2[2] = {i2_bf4(kw.xy), i2_bf4(kw.zw)};
        for (uint half4 = 0; half4 < 2; ++half4) {
          const float4 kf = kf2[half4];
          const threadgroup float* qh = qrow + d8 * 8 + half4 * 4;
          score = fma(qh[0], kf.x, score);
          score = fma(qh[1], kf.y, score);
          score = fma(qh[2], kf.z, score);
          score = fma(qh[3], kf.w, score);
        }
      }
      const ulong back = p - key;
      if (back < a.extent) {
        float pr[I2_DREL];
        for (uint j = 0; j < I2_DREL; ++j) pr[j] = float(proj[ulong(j) * a.extent + back]);
        float bb = 0.0f;
        for (uint j = 0; j < I2_DREL; ++j) bb = fma(relrow[j], pr[j], bb);
        score = score * a.scale + float(bfloat(bb));
      } else {
        score = score * a.scale;
      }
    }
    const bool bad = simd_any(valid && isnan(score));
    const float top = simd_max((valid && !isnan(score)) ? score : -INFINITY);
    const float m_new = bad ? NAN : i2_max(mi, top);
    const float alpha = mi == -INFINITY ? 0.0f : metal::precise::exp(mi - m_new);
    const float pw = !valid ? 0.0f : (score == -INFINITY ? 0.0f : metal::precise::exp(score - m_new));
    li = li * alpha + simd_sum(pw);
    ac = ac * alpha;
    mi = m_new;
    for (uint j = 0; j < nvalid; ++j) {
      const device bfloat4* vp = (const device bfloat4*)(vcache + (kk0 + j) * kv_row + kv_off);
      ac = fma(float4(simd_shuffle(pw, ushort(j))), float4(vp[lane]), ac);
    }
  }
}

template <bool FINAL>
inline void i2_attn_rows(const device bfloat* q, const device bfloat* rel, const device bfloat* sink,
                         const device bfloat* gate, const device bfloat* kcache, const device bfloat* vcache,
                         const device bfloat* proj, device bfloat* out, device float* scratch, constant I2AttnArgs& a,
                         uint3 tg, uint tid, uint sg, uint lane, threadgroup float* q_tg, threadgroup float* rel_tg,
                         threadgroup float* big, threadgroup float* al_tg, threadgroup float* mrg_m,
                         threadgroup float* mrg_l) {
  const uint chunk = tg.x, head = tg.y, r0 = tg.z * I2_RB;
  const uint nrows = min(I2_RB, a.rows - r0);
  const ulong pf = ulong(a.pos0) + r0; // the first row's position, and the last's
  const ulong pl = pf + nrows - 1;
  const ulong ks = ulong(chunk) * I2_CHUNK;
  if (ks > pl) return; // no row of the tile has this chunk
  const ulong kv_off = ulong(head / (a.heads / a.kv_heads)) * I2_DIM;
  const ulong kv_row = ulong(a.kv_heads) * I2_DIM;

  for (uint i = tid; i < I2_RB * I2_DIM; i += 128) {
    const uint r = i / I2_DIM;
    q_tg[i] = r < nrows ? float(q[(ulong(r0 + r) * a.heads + head) * I2_DIM + i % I2_DIM]) : 0.0f;
  }
  for (uint i = tid; i < I2_RB * I2_DREL; i += 128) {
    const uint r = i / I2_DREL;
    rel_tg[i] = r < nrows ? float(rel[(ulong(r0 + r) * a.heads + head) * I2_DREL + i % I2_DREL]) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // whole blocks of keys every row sees: those ending at or before the first row's position
  const ulong shared_end = ks <= pf ? min(pf + 1, ks + I2_CHUNK) : ks;
  const ulong nb_c = (shared_end - ks) / 32;

  float m[I2_RB], l[I2_RB];
  for (uint i = 0; i < I2_RB; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f;
    if (chunk == 0 && sg == 0 && i < nrows) {
      m[i] = float(sink[ulong(r0 + i) * a.heads + head]);
      l[i] = m[i] == -INFINITY ? 0.0f : 1.0f;
    }
  }

  simdgroup_float8x8 acc[16];
  for (uint n = 0; n < 16; ++n) acc[n] = simdgroup_float8x8(0.0f);
  threadgroup float* sc = big + sg * 512; // a block's scores, then its probabilities: [row][key]
  threadgroup float* stg = sc + 256;      // where a rescale stages two tiles
  const uint row_of_lane = ((lane >> 1) & 3) + ((lane >> 4) << 2);
  for (ulong b = sg; b < nb_c; b += I2_ATTN_SGS) {
    const ulong key0 = ks + b * 32;
    simdgroup_float8x8 s[4];
    for (uint kt = 0; kt < 4; ++kt) s[kt] = simdgroup_float8x8(0.0f);
    for (uint k = 0; k < 16; ++k) {
      simdgroup_float8x8 qa;
      simdgroup_load(qa, q_tg + k * 8, I2_DIM);
      for (uint kt = 0; kt < 4; ++kt) {
        simdgroup_bfloat8x8 kb;
        simdgroup_load(kb, kcache + (key0 + kt * 8) * kv_row + kv_off + k * 8, kv_row, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[kt], qa, kb, s[kt]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kt = 0; kt < 4; ++kt) simdgroup_store(s[kt], sc + kt * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const ulong key = key0 + lane;
    for (uint i = 0; i < I2_RB; ++i) {
      float score = sc[i * 32 + lane];
      const ulong back = pf + i - key;
      if (back < a.extent) {
        float pr[I2_DREL];
        for (uint j = 0; j < I2_DREL; ++j) pr[j] = float(proj[ulong(j) * a.extent + back]);
        float bb = 0.0f;
        for (uint j = 0; j < I2_DREL; ++j) bb = fma(rel_tg[i * I2_DREL + j], pr[j], bb);
        score = score * a.scale + float(bfloat(bb));
      } else {
        score = score * a.scale;
      }
      const bool bad = simd_any(isnan(score));
      const float top = simd_max(isnan(score) ? -INFINITY : score);
      const float m_new = bad ? NAN : i2_max(m[i], top);
      const float alpha = m[i] == -INFINITY ? 0.0f : metal::precise::exp(m[i] - m_new);
      const float pw = score == -INFINITY ? 0.0f : metal::precise::exp(score - m_new);
      l[i] = l[i] * alpha + simd_sum(pw);
      m[i] = m_new;
      sc[i * 32 + lane] = pw;
      if (lane == 0) al_tg[sg * I2_RB + i] = alpha;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // acc * alpha only when some alpha is not 1 (x * 1 is x), via threadgroup memory: the matrices stay in registers
    if (!simd_all(al_tg[sg * I2_RB + row_of_lane] == 1.0f)) {
      for (uint n = 0; n < 16; n += 2) {
        simdgroup_store(acc[n], stg, 16);
        simdgroup_store(acc[n + 1], stg + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float4* rp = (threadgroup float4*)(stg + (lane >> 2) * 16 + (lane & 3) * 4);
        rp[0] = rp[0] * al_tg[sg * I2_RB + (lane >> 2)];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(acc[n], stg, 16);
        simdgroup_load(acc[n + 1], stg + 8, 16);
      }
    }
    for (uint kt = 0; kt < 4; ++kt) {
      simdgroup_float8x8 pa;
      simdgroup_load(pa, sc + kt * 8, 32);
      simdgroup_bfloat8x8 vb[16];
      _Pragma("unroll")
      for (uint n = 0; n < 16; ++n) simdgroup_load(vb[n], vcache + (key0 + kt * 8) * kv_row + kv_off + n * 8, kv_row);
      _Pragma("unroll")
      for (uint n = 0; n < 16; ++n) simdgroup_multiply_accumulate(acc[n], pa, vb[n], acc[n]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float* tile = big; // one stream's 8 x 128 accumulators at a time
  if (lane == 0) {
    for (uint i = 0; i < I2_RB; ++i) {
      mrg_m[sg * I2_RB + i] = m[i];
      mrg_l[sg * I2_RB + i] = l[i];
    }
  }
  // Blocks past the shared ones, in the chunk holding the rows' positions: all streams' final m precede the merge's max
  const ulong first = nb_c + ((sg + I2_ATTN_SGS - nb_c % I2_ATTN_SGS) % I2_ATTN_SGS);
  if (nb_c * 32 < min(pl + 1, ks + I2_CHUNK) - ks) {
    for (uint s = 0; s < I2_ATTN_SGS; ++s) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg != s) continue;
      for (uint n = 0; n < 16; ++n) simdgroup_store(acc[n], tile + n * 8, I2_DIM);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < nrows; ++i) {
        const ulong p = pf + i;
        if (ks > p || first * 32 >= min(p + 1, ks + I2_CHUNK) - ks) continue;
        float mi = mrg_m[sg * I2_RB + i], li = mrg_l[sg * I2_RB + i];
        threadgroup float4* row = (threadgroup float4*)(tile + i * I2_DIM + lane * 4);
        float4 ac = *row;
        i2_rows_scalar(kcache, vcache, proj, a, q_tg + i * I2_DIM, rel_tg + i * I2_DREL, p, ks, first, kv_off, lane, mi,
                       li, ac);
        *row = ac;
        if (lane == 0) {
          mrg_m[sg * I2_RB + i] = mi;
          mrg_l[sg * I2_RB + i] = li;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint n = 0; n < 16; ++n) simdgroup_load(acc[n], tile + n * 8, I2_DIM);
    }
  }
  // The four streams merged in order, a stream's tile at a time, a thread an output dimension
  float M[I2_RB], L[I2_RB], A[I2_RB];
  for (uint s = 0; s < I2_ATTN_SGS; ++s) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == s) {
      for (uint n = 0; n < 16; ++n) simdgroup_store(acc[n], tile + n * 8, I2_DIM);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < I2_RB; ++i) {
      if (s == 0) {
        M[i] = -INFINITY;
        for (uint t = 0; t < I2_ATTN_SGS; ++t) M[i] = i2_max(M[i], mrg_m[t * I2_RB + i]);
        L[i] = 0.0f;
        A[i] = 0.0f;
      }
      const float ms = mrg_m[s * I2_RB + i];
      const float e = ms == -INFINITY ? 0.0f : metal::precise::exp(ms - M[i]);
      L[i] += mrg_l[s * I2_RB + i] * e;
      A[i] += tile[i * I2_DIM + tid] * e;
    }
  }
  for (uint i = 0; i < nrows; ++i) {
    if (ks > pf + i) continue;
    const ulong rh = ulong(r0 + i) * a.heads + head;
    if (FINAL) {
      const ulong at = rh * I2_DIM + tid;
      out[at] = i2_attn_finish(M[i], L[i], A[i], float(gate[at]));
    } else {
      device float* rec = scratch + (rh * a.chunks + chunk) * I2_PART;
      rec[4 + tid] = A[i];
      if (tid == 0) {
        rec[0] = M[i];
        rec[1] = L[i];
      }
    }
  }
}

#define I2_ROWS_KERNEL(NAME, FINAL)                                                                                 \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                      \
      const device bfloat* q [[buffer(0)]], const device bfloat* rel [[buffer(1)]],                                 \
      const device bfloat* sink [[buffer(2)]], const device bfloat* gate [[buffer(3)]],                              \
      const device bfloat* kcache [[buffer(4)]], const device bfloat* vcache [[buffer(5)]],                          \
      const device bfloat* proj [[buffer(6)]], device bfloat* out [[buffer(7)]],                                     \
      device float* scratch [[buffer(8)]], constant I2AttnArgs& a [[buffer(9)]],                                     \
      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                          \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                         \
    threadgroup float q_tg[I2_RB * I2_DIM];                                                                          \
    threadgroup float rel_tg[I2_RB * I2_DREL];                                                                       \
    threadgroup float4 big4[I2_ROWS_TG_FLOATS / 4];                                                                  \
    threadgroup float al_tg[I2_ATTN_SGS * I2_RB];                                                                    \
    threadgroup float mrg_m[I2_ATTN_SGS * I2_RB], mrg_l[I2_ATTN_SGS * I2_RB];                                        \
    i2_attn_rows<FINAL>(q, rel, sink, gate, kcache, vcache, proj, out, scratch, a, tg, tid, sg, lane, q_tg, rel_tg,  \
                        (threadgroup float*)big4, al_tg, mrg_m, mrg_l);                                              \
  }

I2_ROWS_KERNEL(inkling2_attn_rows_partial, false)
I2_ROWS_KERNEL(inkling2_attn_rows_final, true)
