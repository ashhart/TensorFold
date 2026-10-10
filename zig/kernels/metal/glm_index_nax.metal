// GLM's index scores for prompt chunks on the tensor units: row r's fp32 score of pooled block b, the sum over heads in order of w_h relu(q_h . pool_b).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

constant constexpr int HI = 32, DI = 128, PAD = DI + 8;

struct GlmScoreArgs {
  uint p0, q_stride, w_stride, s_stride, rows;
};

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

// Grid (64-block tiles, 64-row tiles): 4 simdgroups of 32 rows x 32 blocks; only blocks a row reads are stored, whole tiles past them skipped.
[[kernel]] void glm_index_scores_nax(const device bfloat* iq [[buffer(0)]], const device bfloat* iw [[buffer(1)]],
                                     const device bfloat* pool [[buffer(2)]], device float* scores [[buffer(3)]],
                                     constant GlmScoreArgs& a [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  const int b0 = int(tg.x) * 64, r0 = int(tg.y) * 64;
  const int rows = min(64, int(a.rows) - r0);
  if (b0 >= (int(a.p0) + r0 + rows) / 4) return; // the tile's last row reads blocks below (p0 + row + 1) / 4
  const uint t = sg * 32 + lane;
  for (uint i = t; i < 64 * (DI / 8); i += 128) { // the 64 blocks' pooled keys, 16 bytes a load (past the pool: its last)
    const uint rr = i / (DI / 8), cc = (i % (DI / 8)) * 8;
    const int b = min(b0 + int(rr), int(a.s_stride) - 1);
    *(threadgroup uint4*)(tile + rr * PAD + cc) = *(const device uint4*)(pool + size_t(b) * DI + cc);
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
      const long at = long(r0 + (r < rows ? r : 0)); // a dead row reads a live one; its scores are never stored
      qr[m][h] = iq + at * a.q_stride;
      wr[m][h] = iw + at * a.w_stride;
    }
  }
  frag<float> total[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    total[m][0] = frag<float>(0);
    total[m][1] = frag<float>(0);
  }
  for (int hd = 0; hd < HI; hd++) {
    frag<float> s[2][2];
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      s[m][0] = frag<float>(0);
      s[m][1] = frag<float>(0);
    }
    TF_UNROLL
    for (short q = 0; q < DI / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, PAD, tn + 16, 16 * q, home);
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        const frag<bfloat> qa = ix_frag(qr[m][0] + hd * DI, qr[m][1] + hd * DI, 16 * q, home);
        mma_16x32<false, true>(s[m][0], s[m][1], qa, k0, k1);
      }
    }
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      const float w0 = float(wr[m][0][hd]), w1 = float(wr[m][1][hd]);
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          total[m][j][e] = fma(e < 4 ? w0 : w1, metal::max(s[m][j][e], 0.0f), total[m][j][e]);
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
        const int b = b0 + tn + 16 * j + home.x + TF_COL(e);
        if (r < rows && b < (int(a.p0) + r0 + r + 1) / 4) scores[size_t(r0 + r) * a.s_stride + b] = total[m][j][e];
      }
    }
  }
}

// A transposed right operand's element e: its key (a block, the op's column) and its dim (on M5 the op loads it as any operand).
#ifndef TF_SIMD_FRAGS
inline short ix_key(short e, short2 home) { return home.y + (e >> 2) * 8; }
inline short ix_dim(short e, short2 home) { return home.x + (e & 3); }
#else
inline short ix_key(short e, short2 home) { return home.x + TF_COL(e); }
inline short ix_dim(short e, short2 home) { return home.y + (e >> 2) * 8; }
#endif

// Decode rows' index scores: a row's 32 heads are the op's rows over 32 blocks a simdgroup, fp32 dots of exact bf16 products,
// then relu, each head's weight and the sum over heads in a fixed tree (four in a lane, then lanes xor 2, 4, 16).
// Grid (128-block tiles, rows) of 4 simdgroups.
[[kernel]] void glm_index_decode(const device bfloat* iq [[buffer(0)]], const device bfloat* iw [[buffer(1)]],
                                 const device bfloat* pool [[buffer(2)]], device float* scores [[buffer(3)]],
                                 constant GlmScoreArgs& a [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int row = int(tg.y), b0 = int(tg.x) * 128 + 32 * int(sg);
  const int blocks = (int(a.p0) + row + 1) / 4;
  if (b0 >= blocks) return;
  const short2 home = frag_home(ushort(lane));
  const device bfloat* q = iq + long(row) * a.q_stride;
  frag<float> s[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    s[m][0] = frag<float>(0);
    s[m][1] = frag<float>(0);
  }
  TF_UNROLL
  for (short kq = 0; kq < DI / 16; kq++) {
    frag<bfloat> k0, k1;
#ifndef TF_SIMD_FRAGS
    k0 = frag<bfloat>(*(const device bfloat4*)(pool + long(min(b0 + int(home.y), blocks - 1)) * DI + 16 * kq + home.x),
                      *(const device bfloat4*)(pool + long(min(b0 + int(home.y) + 8, blocks - 1)) * DI + 16 * kq + home.x));
    k1 = frag<bfloat>(*(const device bfloat4*)(pool + long(min(b0 + 16 + int(home.y), blocks - 1)) * DI + 16 * kq + home.x),
                      *(const device bfloat4*)(pool + long(min(b0 + 24 + int(home.y), blocks - 1)) * DI + 16 * kq + home.x));
#else
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      k0[e] = pool[long(min(b0 + int(ix_key(e, home)), blocks - 1)) * DI + 16 * kq + ix_dim(e, home)];
      k1[e] = pool[long(min(b0 + 16 + int(ix_key(e, home)), blocks - 1)) * DI + 16 * kq + ix_dim(e, home)];
    }
#endif
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      const frag<bfloat> qa = ix_frag(q + (16 * m + home.y) * DI, q + (16 * m + home.y + 8) * DI, 16 * kq, home);
      mma_16x32<false, true>(s[m][0], s[m][1], qa, k0, k1);
    }
  }
  float w[2][2]; // the weights of this lane's heads: 16 m + home.y (+ 8)
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    w[m][0] = float(iw[long(row) * a.w_stride + 16 * m + home.y]);
    w[m][1] = float(iw[long(row) * a.w_stride + 16 * m + home.y + 8]);
  }
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    TF_UNROLL
    for (short c = 0; c < 4; c++) {
      float t = w[0][0] * metal::max(s[0][j][c], 0.0f);
      t = fma(w[0][1], metal::max(s[0][j][4 + c], 0.0f), t);
      t = fma(w[1][0], metal::max(s[1][j][c], 0.0f), t);
      t = fma(w[1][1], metal::max(s[1][j][4 + c], 0.0f), t);
      t += simd_shuffle_xor(t, ushort(2));
      t += simd_shuffle_xor(t, ushort(4));
      t += simd_shuffle_xor(t, ushort(16));
      const int b = b0 + 16 * j + home.x + TF_COL(c);
      if (home.y == 0 && b < blocks) scores[long(row) * a.s_stride + b] = t;
    }
  }
}
