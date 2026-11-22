// Mini gate-down for many rows on MMA, bit-equal to loop_gate_down: 256 chains, a tree a slice, slices added in order.

#define LD_UNR _Pragma("clang loop unroll(full)")

// Position of chain step e: the hidden half's steps first, then the message half's.
inline uint ld_step(uint e, uint steps_half, uint hidden) {
  return e < steps_half ? 256u * e : hidden + 256u * (e - steps_half);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_loop_gate_down_mma(
    const device uint4* xn [[buffer(0)]], const device uint4* w0 [[buffer(1)]], const device uint4* w1 [[buffer(2)]],
    const device uint4* w2 [[buffer(3)]], device bfloat* down [[buffer(4)]], constant LpGateArgs& a [[buffer(5)]],
    uint3 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint g [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint tg_x[4][8][2][8][4];
  threadgroup uint tg_w[4][8][2][8][4];
  threadgroup float part[4][2][2][8][8];
  const uint gate = grid.z, H = a.hidden, R = a.rank, rows = a.rows;
  const uint n0 = grid.x * 16u, r0 = grid.y * 16u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  const uint kv = H >> 2u;  // uint4 of a row: 2 H bf16
  const device uint4* X = xn + ulong(gate) * rows * kv;
  const device uint4* W = gate == 0 ? w0 : (gate == 1 ? w1 : w2);
  const uint half_steps = H >> 8u, steps = 2u * half_steps;
  const uint pp = lane & 3u;
  float total[2] = {0.0f, 0.0f};
  for (uint s = 0; s < 8u; ++s) {
    simdgroup_matrix<float, 8, 8> acc[2][2][8];
    LD_UNR for (uint i = 0; i < 2; ++i)
      LD_UNR for (uint j = 0; j < 2; ++j)
        LD_UNR for (uint c = 0; c < 8; ++c) acc[i][j][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    const uint slice = 32u * s + 8u * g;
    for (uint e0 = 0; e0 + 8u <= steps; e0 += 8u) {
      LD_UNR for (uint t = 0; t < 2; ++t) {
        const uint idx = lane + 32u * t;
        {
          const uint m = idx >> 2u;
          const uint row = r0 + m;
          uint4 v0 = uint4(0), v1 = uint4(0);
          if (row < rows) {
            const device uint4* p = X + ulong(row) * kv;
            v0 = p[(ld_step(e0 + 2u * pp, half_steps, H) + slice) >> 3u];
            v1 = p[(ld_step(e0 + 2u * pp + 1u, half_steps, H) + slice) >> 3u];
          }
          LD_UNR for (uint w = 0; w < 4; ++w) {
            tg_x[g][2 * w][m >> 3u][m & 7u][pp] = (v0[w] & 0xFFFFu) | (v1[w] << 16u);
            tg_x[g][2 * w + 1][m >> 3u][m & 7u][pp] = (v0[w] >> 16u) | (v1[w] & 0xFFFF0000u);
          }
        }
        {
          const uint p8 = (idx >> 2u) & 7u, nr = idx >> 5u;
          const uint col = n0 + nr * 8u + 2u * pp;
          uint4 v0 = uint4(0), v1 = uint4(0);
          const uint at = (ld_step(e0 + p8, half_steps, H) + slice) >> 3u;
          if (col < R) v0 = W[ulong(col) * kv + at];
          if (col + 1u < R) v1 = W[ulong(col + 1u) * kv + at];
          LD_UNR for (uint w = 0; w < 4; ++w) {
            tg_w[g][2 * w][nr][p8][pp] = (v0[w] & 0xFFFFu) | (v1[w] << 16u);
            tg_w[g][2 * w + 1][nr][p8][pp] = (v0[w] >> 16u) | (v1[w] & 0xFFFF0000u);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      LD_UNR for (uint c = 0; c < 8; ++c) {
        simdgroup_matrix<bfloat, 8, 8> xa[2], wb[2];
        LD_UNR for (uint i = 0; i < 2; ++i) {
          const uint xp = tg_x[g][c][i][fm][fn >> 1u];
          const uint wp = tg_w[g][c][i][fm][fn >> 1u];
          xa[i].thread_elements()[0] = as_type<bfloat>(ushort(xp));
          xa[i].thread_elements()[1] = as_type<bfloat>(ushort(xp >> 16u));
          wb[i].thread_elements()[0] = as_type<bfloat>(ushort(wp));
          wb[i].thread_elements()[1] = as_type<bfloat>(ushort(wp >> 16u));
        }
        LD_UNR for (uint i = 0; i < 2; ++i)
          LD_UNR for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j][c], xa[i], wb[j], acc[i][j][c]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The steps past the last full chunk, one fma each in chain order.
    for (uint e = steps & ~7u; e < steps; ++e) {
      const uint at = ld_step(e, half_steps, H) + slice;
      const device bfloat* xr[2];
      LD_UNR for (uint i = 0; i < 2; ++i) xr[i] = (const device bfloat*)(X + ulong(min(r0 + i * 8u + fm, rows - 1u)) * kv) + at;
      LD_UNR for (uint j = 0; j < 2; ++j)
        LD_UNR for (uint ee = 0; ee < 2; ++ee) {
          const device bfloat* wr = (const device bfloat*)(W + ulong(min(n0 + j * 8u + fn + ee, R - 1u)) * kv) + at;
          LD_UNR for (uint i = 0; i < 2; ++i)
            LD_UNR for (uint c = 0; c < 8; ++c)
              acc[i][j][c].thread_elements()[ee] = fma(float(wr[c]), float(xr[i][c]), acc[i][j][c].thread_elements()[ee]);
        }
    }
    LD_UNR for (uint i = 0; i < 2; ++i)
      LD_UNR for (uint j = 0; j < 2; ++j)
        LD_UNR for (uint e = 0; e < 2; ++e) {
          float v[8];
          LD_UNR for (uint c = 0; c < 8; ++c) v[c] = acc[i][j][c].thread_elements()[e];
          LD_UNR for (uint d = 1u; d < 8u; d *= 2u)
            LD_UNR for (uint c = 0; c < 8u; c += 2u * d) v[c] = v[c] + v[c + d];
          part[g][i][j][fm][fn + e] = v[0];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    LD_UNR for (uint t = 0; t < 2; ++t) {
      const uint o = g * 32u + lane + 128u * t;
      const uint i = o >> 7u, j = (o >> 6u) & 1u, m = (o >> 3u) & 7u, n = o & 7u;
      const float slice_sum = (part[0][i][j][m][n] + part[1][i][j][m][n]) + (part[2][i][j][m][n] + part[3][i][j][m][n]);
      total[t] = s == 0 ? slice_sum : total[t] + slice_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  LD_UNR for (uint t = 0; t < 2; ++t) {
    const uint o = g * 32u + lane + 128u * t;
    const uint i = o >> 7u, j = (o >> 6u) & 1u, m = (o >> 3u) & 7u, n = o & 7u;
    const uint row = r0 + i * 8u + m, col = n0 + j * 8u + n;
    if (row < rows && col < R) down[(ulong(gate) * rows + row) * R + col] = bfloat(total[t]);
  }
}
