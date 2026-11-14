// The register-only decode, T column tiles and R rows a threadgroup, the mul1 value by one exact half FMA (no table).

// A state's value: the table entry fma(half(1024 + sum), 0x1EEE, 0xC931), by 2 * sum as subnormal half bits, exactly.
inline float tf_exl3_t_value(ushort s) {
  const uint x = uint(s) * 0x83DCD12Du;
  const ushort2 u = as_type<ushort2>(((x >> 7u) & 0x01FE01FEu) + ((x & 0x00FF00FFu) << 1u));
  const half sum = as_type<half>(ushort(u.x + u.y));
  return float(fma(sum, as_type<half>(ushort(0x7AEEu)), as_type<half>(ushort(0xC2E8u))));
}

// The value of a state of any codebook: mul1 by arithmetic, the others from the resident table.
template <bool Mul1>
inline float tf_exl3_t_load(ushort state, const device half* table, uint codebook) {
  if (Mul1) return tf_exl3_t_value(state);
  return float(table[codebook * 65536u + state]);
}

// State J of a lane's run: its sixteen bits are the top half of a shifted word, read free as the high half.
template <uint S, uint J>
inline ushort tf_exl3_t_state(uint c0, uint c1, uint mid) {
  constexpr uint bit = S * J;
  constexpr uint k = bit / 16u;
  constexpr uint r = bit % 16u;
  static_assert(k <= 2u, "the run is at most three half-word windows");
  const uint base = k == 0u ? c0 : (k == 1u ? mid : c1);
  return as_type<ushort2>(base << r).y;
}

// The decode of one block's eight weights of a lane, then each row's eight products.
template <uint S, uint R, bool Mul1>
inline void tf_exl3_t_block(thread float (*acc)[8], uint c0, uint c1, const device half* x, const thread uint* xo,
                             const device half* table, uint codebook) {
  const uint mid = (c0 << 16) | (c1 >> 16);
  float w[8];
  w[0] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 0>(c0, c1, mid), table, codebook);
  w[1] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 1>(c0, c1, mid), table, codebook);
  w[2] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 2>(c0, c1, mid), table, codebook);
  w[3] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 3>(c0, c1, mid), table, codebook);
  w[4] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 4>(c0, c1, mid), table, codebook);
  w[5] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 5>(c0, c1, mid), table, codebook);
  w[6] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 6>(c0, c1, mid), table, codebook);
  w[7] = tf_exl3_t_load<Mul1>(tf_exl3_t_state<S, 7>(c0, c1, mid), table, codebook);
  for (uint r = 0; r < R; ++r) {
    // The lane's four k rows of this block, read as two halves each (the rows are adjacent in pairs).
    const half2 xa = *(const device half2*)(x + xo[r]);
    const half2 xb = *(const device half2*)(x + xo[r] + 8);
    const float x0 = float(xa.x), x1 = float(xa.y), x2 = float(xb.x), x3 = float(xb.y);
    tf_exl3_fma_step_w<0>(&acc[r][0], w[0], x0, x1, x2, x3);
    tf_exl3_fma_step_w<1>(&acc[r][1], w[1], x0, x1, x2, x3);
    tf_exl3_fma_step_w<2>(&acc[r][2], w[2], x0, x1, x2, x3);
    tf_exl3_fma_step_w<3>(&acc[r][3], w[3], x0, x1, x2, x3);
    tf_exl3_fma_step_w<4>(&acc[r][4], w[4], x0, x1, x2, x3);
    tf_exl3_fma_step_w<5>(&acc[r][5], w[5], x0, x1, x2, x3);
    tf_exl3_fma_step_w<6>(&acc[r][6], w[6], x0, x1, x2, x3);
    tf_exl3_fma_step_w<7>(&acc[r][7], w[7], x0, x1, x2, x3);
  }
}

// A lane's run in one tile: the stream words it spans, then the two 32-bit windows its eight states are cut from.
template <bool Two>
inline void tf_exl3_t_read(const device ushort* tile, const TfExl3RowWalk walk, thread uint* p) {
  p[0] = tf_exl3_word(tile, walk.first_word);
  p[1] = tf_exl3_word(tile, walk.next1);
  p[2] = Two ? 0u : tf_exl3_word(tile, walk.next2);
}

template <bool Two>
inline uint2 tf_exl3_t_funnel(const thread uint* p, uint first_bit) {
  const uint down = 32u - first_bit;
  const uint c0 = uint(((ulong(p[0]) << 32) | p[1]) >> down);
  return uint2(c0, Two ? (p[1] << first_bit) : uint(((ulong(p[1]) << 32) | p[2]) >> down));
}

// One threadgroup: T tiles of sixteen columns of up to R rows; k is cut into sixteen ranges whatever the warp count.
constant constexpr uint TF_EXL3_T_RANGES = 16u;

template <uint S, uint T, uint R, bool Mul1>
inline void tf_exl3_t_body(const device half* x, const TfExl3DecodeJob job, device float* sums, constant TfExl3Args& a,
                           const device half* table, uint row0, uint nb, uint lane, uint sg, threadgroup float* red) {
  constexpr bool Two = S <= 4u;
  constexpr uint W = TF_EXL3_T_RANGES;
  const uint blocks = a.k / 16;
  const TfExl3RowWalk walk = tf_exl3_row_walk(lane, job.halves);
  // A block row is a 32-bit step in ushorts (a pack is far below 8 GiB), so the pointers advance by one add.
  const uint block_step = uint(job.tile_k_stride >> 1);
  uint xrow[R];
  for (uint r = 0; r < R; ++r) xrow[r] = min(row0 + r, a.rows - 1u) * a.k + 2 * (lane & 3u);
  for (uint g = sg; g < W; g += TF_FMA_WARPS) {
    const uint b0 = blocks * g / W, b1 = blocks * (g + 1u) / W;
    const device ushort* tile[T];
    uint c0[T], c1[T];
    for (uint t = 0; t < T; ++t) {
      tile[t] = reinterpret_cast<const device ushort*>(job.trellis + ulong(b0) * job.tile_k_stride +
                                                       ulong(nb / 16 + t) * job.tile_n_stride);
      uint p[3];
      tf_exl3_t_read<Two>(tile[t], walk, p);
      const uint2 c = tf_exl3_t_funnel<Two>(p, walk.first_bit);
      c0[t] = c.x;
      c1[t] = c.y;
    }
    float acc[T][R][8];
    for (uint t = 0; t < T; ++t)
      for (uint r = 0; r < R; ++r)
        for (uint j = 0; j < 8u; ++j) acc[t][r][j] = 0.0f;
    uint xo[R];
    for (uint r = 0; r < R; ++r) xo[r] = xrow[r] + b0 * 16u;
    for (uint bb = b0; bb < b1; ++bb) {
      uint pre[T][3];
      const bool more = bb + 1 < b1;
      if (more) {
        for (uint t = 0; t < T; ++t) {
          tile[t] += block_step;
          tf_exl3_t_read<Two>(tile[t], walk, pre[t]);
        }
      }
      for (uint t = 0; t < T; ++t) tf_exl3_t_block<S, R, Mul1>(acc[t], c0[t], c1[t], x, xo, table, job.codebook);
      if (more) {
        for (uint t = 0; t < T; ++t) {
          const uint2 c = tf_exl3_t_funnel<Two>(pre[t], walk.first_bit);
          c0[t] = c.x;
          c1[t] = c.y;
        }
      }
      for (uint r = 0; r < R; ++r) xo[r] += 16u;
    }
    for (uint r = 0; r < R; ++r) {
      for (uint t = 0; t < T; ++t) {
        const uint slot = ((g * R + r) * T + t) * 64u;
        red[slot + lane] = (acc[t][r][0] + acc[t][r][1]) + (acc[t][r][2] + acc[t][r][3]);
        red[slot + 32u + lane] = (acc[t][r][4] + acc[t][r][5]) + (acc[t][r][6] + acc[t][r][7]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (uint r = 0; r < R; ++r) {
    if (row0 + r >= a.rows) break;
    for (uint t = 0; t < T; ++t) {
      float acc0 = 0.0f, acc1 = 0.0f;
      for (uint w = 0; w < W; ++w) {
        const uint slot = ((w * R + r) * T + t) * 64u;
        acc0 += red[slot + lane];
        acc1 += red[slot + 32u + lane];
      }
      acc0 += simd_shuffle_xor(acc0, 1);
      acc0 += simd_shuffle_xor(acc0, 2);
      acc1 += simd_shuffle_xor(acc1, 1);
      acc1 += simd_shuffle_xor(acc1, 2);
      if ((lane & 3u) == 0u) {
        sums[ulong(row0 + r) * a.n + nb + t * 16u + lane / 4] = acc0;
        sums[ulong(row0 + r) * a.n + nb + t * 16u + 8 + lane / 4] = acc1;
      }
    }
  }
}

template <uint T, uint R, bool Mul1>
inline void tf_exl3_t_halves(const device half* x, const TfExl3DecodeJob job, device float* sums,
                             constant TfExl3Args& a, const device half* table, uint row, uint nb, uint lane, uint sg,
                             threadgroup float* red) {
  switch (job.halves) {
    case 6u: tf_exl3_t_body<3, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    case 10u: tf_exl3_t_body<5, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    case 12u: tf_exl3_t_body<6, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    case 8u: tf_exl3_t_body<4, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    case 4u: tf_exl3_t_body<2, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    case 2u: tf_exl3_t_body<1, T, R, Mul1>(x, job, sums, a, table, row, nb, lane, sg, red); break;
    default: break;
  }
}

// The routed body: a row's expert names the pack slice; mul1 is computed, other codebooks read the table.
template <uint T>
inline void tf_exl3_t_routed_kernel(const device half* x, const device uchar* trellis, device float* sums,
                                    constant TfExl3Args& a, const device uint* routes,
                                    const device TfExl3Expert* descriptors, const device half* table, uint lane,
                                    uint sg, uint3 tg, threadgroup float* red) {
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const TfExl3DecodeJob job = tf_exl3_routed_decode_job(trellis, a, descriptors, expert);
  const uint nb = tg.x * 16u * T;
  if (nb >= job.width) return;
  if (job.codebook == 2u) tf_exl3_t_halves<T, 1, true>(x, job, sums, a, table, row, nb, lane, sg, red);
  else tf_exl3_t_halves<T, 1, false>(x, job, sums, a, table, row, nb, lane, sg, red);
}

// The dense body: one matrix, one row a threadgroup.
template <uint T>
inline void tf_exl3_t_dense_kernel(const device half* x, const device uchar* trellis, device float* sums,
                                   constant TfExl3Args& a, uint lane, uint sg, uint3 tg, threadgroup float* red) {
  const uint row = tg.y;
  if (row >= a.rows) return;
  const TfExl3DecodeJob job = tf_exl3_dense_decode_job(trellis, a);
  const uint nb = tg.x * 16u * T;
  if (nb >= job.width) return;
  tf_exl3_t_halves<T, 1, true>(x, job, sums, a, nullptr, row, nb, lane, sg, red);
}

// The multi-row dense body: R rows a threadgroup share one decode of the matrix (any codebook), one tile wide.
template <uint T, uint R>
inline void tf_exl3_t_multi_kernel(const device half* x, const device uchar* trellis, device float* sums,
                                   constant TfExl3Args& a, const device half* table, uint lane, uint sg, uint3 tg,
                                   threadgroup float* red) {
  const uint row0 = tg.y * R;
  if (row0 >= a.rows) return;
  const TfExl3DecodeJob job = tf_exl3_dense_decode_job(trellis, a);
  const uint nb = tg.x * 16u * T;
  if (nb >= job.width) return;
  if (job.codebook == 2u) tf_exl3_t_halves<T, R, true>(x, job, sums, a, table, row0, nb, lane, sg, red);
  else tf_exl3_t_halves<T, R, false>(x, job, sums, a, table, row0, nb, lane, sg, red);
}

#define TF_EXL3_T_MULTI(NAME, T, R)                                                                               \
  kernel void NAME(const device half* x [[buffer(0)]], const device uchar* trellis [[buffer(1)]],                 \
                   device float* sums [[buffer(2)]], constant TfExl3Args& a [[buffer(3)]],                        \
                   const device half* table [[buffer(7)]], uint lane [[thread_index_in_simdgroup]],               \
                   uint sg [[simdgroup_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {       \
    threadgroup float red[TF_EXL3_T_RANGES * 64 * T * R];                                                         \
    tf_exl3_t_multi_kernel<T, R>(x, trellis, sums, a, table, lane, sg, tg, red);                                  \
  }

TF_EXL3_T_MULTI(tf_exl3_matmul_decode_fma_dense_multi2, 1, 2)
TF_EXL3_T_MULTI(tf_exl3_matmul_decode_fma_dense_multi4, 1, 4)

#define TF_EXL3_T_KERNELS(T)                                                                                      \
  kernel void tf_exl3_matmul_decode_fma_t##T(                                                                     \
      const device half* x [[buffer(0)]], const device uchar* trellis [[buffer(1)]], device float* sums [[buffer(2)]], \
      constant TfExl3Args& a [[buffer(3)]], const device uint* routes [[buffer(4)]],                              \
      const device TfExl3Expert* descriptors [[buffer(5)]], const device half* table [[buffer(7)]],               \
      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],                         \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                \
    threadgroup float red[TF_EXL3_T_RANGES * 64 * T];                                                               \
    tf_exl3_t_routed_kernel<T>(x, trellis, sums, a, routes, descriptors, table, lane, sg, tg, red);              \
  }                                                                                                               \
  kernel void tf_exl3_matmul_decode_fma_t_dense##T(                                                               \
      const device half* x [[buffer(0)]], const device uchar* trellis [[buffer(1)]], device float* sums [[buffer(2)]], \
      constant TfExl3Args& a [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                    \
    threadgroup float red[TF_EXL3_T_RANGES * 64 * T];                                                               \
    tf_exl3_t_dense_kernel<T>(x, trellis, sums, a, lane, sg, tg, red);                                           \
  }

// More tiles a threadgroup share a block's x loads and conversions; fewer threadgroups fill the GPU.
TF_EXL3_T_KERNELS(1)
TF_EXL3_T_KERNELS(2)
TF_EXL3_T_KERNELS(4)
