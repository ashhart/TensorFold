// Prompt index scores on the tensor units: ds_index_scores's bf16 head sum, the dot in the op's order
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

constant constexpr int IH = 32, ID = 128, PAD = ID + 8;

struct ScoreArgs {
  int rows;
  int groups;     // index keys in the cache
  int ratio;
  int first_pos;  // row r's position is first_pos + r
  int stride;     // score row stride
};

inline float ix_bf(float v) { return float(bfloat(v)); }

// A fragment of q at column c from a lane's two row starts (8-byte loads on M5).
inline frag<bfloat> ix_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  }
  return f;
#endif
}

// 64-key by 64-row tiles, 4 simdgroups of 32x32; keys past a row's visible ones score -inf, whole such tiles skipped
[[kernel]] void ds_index_scores_nax(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                    const device bfloat* K [[buffer(2)]], device bfloat* SCORE [[buffer(3)]],
                                    constant ScoreArgs& a [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  const int t0 = int(tg.x) * 64, r0 = int(tg.y) * 64;
  const int rows = min(64, a.rows - r0);
  const uint t = sg * 32 + lane;
  if (t0 >= (a.first_pos + r0 + rows) / a.ratio) {  // the tile's last row reads keys below (first_pos + r + 1) / ratio
    for (uint i = t; i < 64 * 64; i += 128) {
      const int r = int(i) / 64, k = t0 + int(i) % 64;
      if (r < rows && k < a.groups) SCORE[size_t(r0 + r) * a.stride + k] = bfloat(-INFINITY);
    }
    return;
  }
  for (uint i = t; i < 64 * (ID / 8); i += 128) {  // the 64 keys, 16 bytes a load (past the cache's keys: its last, never stored)
    const uint rr = i / (ID / 8), cc = (i % (ID / 8)) * 8;
    const int k = min(t0 + int(rr), a.groups - 1);
    *(threadgroup uint4*)(tile + rr * PAD + cc) = *(const device uint4*)(K + size_t(k) * ID + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2);
  const device bfloat* qr[2][2];
  const device bfloat* wr[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = tm + 16 * m + home.y + 8 * h;
      const long at = long(r0 + (r < rows ? r : 0));  // a dead row reads a live one; its scores are never stored
      qr[m][h] = Q + at * IH * ID;
      wr[m][h] = WT + at * IH;
    }
  }
  frag<float> total[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    total[m][0] = frag<float>(0);
    total[m][1] = frag<float>(0);
  }
  for (int hd = 0; hd < IH; hd++) {
    frag<float> s[2][2];
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      s[m][0] = frag<float>(0);
      s[m][1] = frag<float>(0);
    }
    TF_UNROLL
    for (short q = 0; q < ID / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, PAD, tn + 16, 16 * q, home);
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        const frag<bfloat> qa = ix_frag(qr[m][0] + hd * ID, qr[m][1] + hd * ID, 16 * q, home);
        mma_16x32<false, true>(s[m][0], s[m][1], qa, k0, k1);
      }
    }
    // the head's term as the lane loop makes it: bf16 dot, relu, times the weight, bf16; summed in fp32 in head order
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      const float w0 = float(wr[m][0][hd]), w1 = float(wr[m][1][hd]);
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          total[m][j][e] += ix_bf(metal::max(ix_bf(s[m][j][e]), 0.0f) * (e < 4 ? w0 : w1));
        }
      }
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int r = tm + 16 * m + home.y + (e >> 2) * 8;
        const int k = t0 + tn + 16 * j + home.x + TF_COL(e);
        if (r < rows && k < a.groups) SCORE[size_t(r0 + r) * a.stride + k] = bfloat(k < (a.first_pos + r0 + r + 1) / a.ratio ? total[m][j][e] : -INFINITY);
      }
    }
  }
}
