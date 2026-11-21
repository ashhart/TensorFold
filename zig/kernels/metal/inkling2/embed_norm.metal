// Inkling 2 normed embedding, RMS norm and strided row copy: x * rsqrt(mean(x^2) + eps) * weight, rounded once to bf16.

struct I2NormArgs {
  uint hidden, rows, vocab, x_pitch;
  float eps;
};

// One row of a norm: x[hidden] (bf16) to out[hidden] (bf16).
inline void i2_norm_row(const device bfloat* x, const device bfloat* w, device bfloat* out, uint hidden, float eps,
                        threadgroup float* scratch, uint tid, uint sg, uint lane) {
  float ss = 0.0f;
  for (uint i = tid; i < hidden; i += I2_TG) {
    const float v = float(x[i]);
    ss = fma(v, v, ss);
  }
  const float inv = i2_rsqrt_mean(i2_tg_sum(ss, scratch, sg, lane), hidden, eps);
  for (uint i = tid; i < hidden; i += I2_TG) out[i] = bfloat(float(w[i]) * (float(x[i]) * inv));
}

// out[row] = norm(table[ids[row]]); an id past the vocabulary gives a NaN row, never a read past the table.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_embed(
    const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], device bfloat* out [[buffer(3)]], constant I2NormArgs& a [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  const uint id = ids[row];
  device bfloat* dst = out + ulong(row) * a.hidden;
  if (id >= a.vocab) {
    for (uint i = tid; i < a.hidden; i += I2_TG) dst[i] = bfloat(NAN);
    return;
  }
  i2_norm_row(table + ulong(id) * a.hidden, w, dst, a.hidden, a.eps, scratch, tid, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_rms_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant I2NormArgs& a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[I2_SGS];
  i2_norm_row(x + ulong(row) * a.x_pitch, w, out + ulong(row) * a.hidden, a.hidden, a.eps, scratch, tid, sg, lane);
}

struct I2CopyArgs {
  uint rows, cols, src_ld, dst_ld;
};

// dst[r][c] = src[r][c] over a rows x cols block of bf16 words (pitches in elements); src and dst must not overlap.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_copy_rows(
    const device bfloat* src [[buffer(0)]], device bfloat* dst [[buffer(1)]], constant I2CopyArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.cols || at.y >= a.rows) return;
  dst[ulong(at.y) * a.dst_ld + at.x] = src[ulong(at.y) * a.src_ld + at.x];
}
