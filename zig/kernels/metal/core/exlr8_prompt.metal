// EXLR8 trellis jobs for prompt rows on the tensor units (after exlr8_projection.metal): exact bf16 tiles, bf16 rows.
#include "../nax.h"

constant constexpr int TF_PR = 32;
constant constexpr int TF_PLD = 128 + 8;
constant constexpr int TF_WLD = 16 + 8;

// One job over up to 32 rows: simdgroup sg's outputs [16 sg, +16) a W^T tile a step; block scales on the fp32 sums.
kernel void tf_exlr8_prompt(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                            const device ulong* bases [[buffer(2)]], const device half* halves [[buffer(3)]],
                            const device char* signs [[buffer(4)]], const device TfExlr8Job* jobs [[buffer(5)]],
                            constant TfExlr8Rows& a [[buffer(6)]], device uchar* Y [[buffer(7)]],
                            const device uint* members [[buffer(8)]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]],
                            uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[TF_PR * TF_PLD];
  threadgroup bfloat ws[8 * 16 * TF_WLD];
  threadgroup float zs[TF_PR * 128];
  // member jobs map their rows; others take rows [32 tg.y, +32) of the window
  const TfExlr8Job j = jobs[tg.x];
  const uint row0 = j.count ? 0u : TF_PR * tg.y, rows = min(j.count ? j.count : a.rows - row0, uint(TF_PR));
  const short2 home = tfp::frag_home(ushort(lane));
  threadgroup bfloat* w16 = ws + sg * 16 * TF_WLD;
  tfp::frag<float> total[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
  for (uint ib = 0; ib < j.blocks; ++ib) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TF_PR * 128; i += 256) {
      const uint rr = i / 128;
      const ulong xr = j.count ? members[j.members + rr] >> 16 : row0 + rr;
      xs[rr * TF_PLD + i % 128] = rr < rows ? bfloat(X[xr * a.x_stride + j.x_col + ib * 128 + i % 128]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uchar* run = W + bases[j.base + ib];
    tfp::frag<float> part[2] = {tfp::frag<float>(0.0f), tfp::frag<float>(0.0f)};
    for (uint r = 0; r < 8; ++r) {
      half w[8];
      const device uint* tile = reinterpret_cast<const device uint*>(run + (8 * r + sg) * 32 * j.k);
      switch (j.k) {
        case 2: tf_exlr8_decode<2>(tile, lane, w); break;
        case 3: tf_exlr8_decode<3>(tile, lane, w); break;
        default: tf_exlr8_decode<4>(tile, lane, w); break;
      }
      // the decoded lane's inputs 2q, 2q + 1, 2q + 8, 2q + 9 and outputs c, c + 8 into W^T's 16 x 16 tile
      for (uint n = 0; n < 8; ++n) {
        const uint in = 2 * (lane % 4) + (n & 1) + 8 * ((n >> 1) & 1), out = lane / 4 + 8 * (n >> 2);
        w16[out * TF_WLD + in] = bfloat(float(w[n]));
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tfp::frag<bfloat> wt, x0, x1;
      tfp::frag_get(wt, w16, TF_WLD, 0, 0, home);
      tfp::frag_get_t(x0, xs, TF_PLD, 0, 16 * r, home);
      tfp::frag_get_t(x1, xs, TF_PLD, 16, 16 * r, home);
      tfp::mma_16x32<false, true>(part[0], part[1], wt, x0, x1);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float s = j.in_scale == TF_NONE ? 1.0f : float(halves[j.in_scale + ib]);
    for (uint h = 0; h < 2; ++h)
      for (short e = 0; e < 8; ++e) total[h][e] = fma(s, part[h][e], total[h][e]);
  }
  for (uint h = 0; h < 2; ++h)
    for (short e = 0; e < 8; ++e) {
      const uint out = 16 * sg + home.y + (e >> 2) * 8, row = 16 * h + home.x + TF_COL(e);
      zs[row * 128 + out] = total[h][e];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg; i < rows; i += 8) {
    float v[4];
    for (uint q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    const ulong row = ulong(j.count ? members[j.members + i] & 0xFFFFu : row0 + i) * a.y_stride + j.y_col;
    if (j.raw) {
      for (uint q = 0; q < 4; ++q) reinterpret_cast<device float*>(Y)[row + lane + 32 * q] = v[q];
      continue;
    }
    tf_exlr8_fwht(v, lane);
    const float block = j.out_block == TF_NONE ? 1.0f : float(halves[j.out_block]);
    for (uint q = 0; q < 4; ++q) {
      const uint c = lane + 32 * q;
      float scale = block * 0.0078125f;
      if (j.out_scale != TF_NONE) scale *= float(halves[j.out_scale + c]);
      if (j.out_sign != TF_NONE) scale *= float(signs[j.out_sign + c]);
      tf_exlr8_store(Y, row + c, v[q] * scale, a.y_f32);
    }
  }
}
