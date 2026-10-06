// GLM's MLA absorb for prompt chunks on the tensor units: each fp32 weight as three exact bf16 parts, so every product is exact.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

constant constexpr int HEADS = 64, NOPE = 256, LATENT = 512, PER_HEAD = 512; // kv_b rows a head: key half, value half
constant constexpr int PAD = 64 + 8;

// ql [rows, 64, 512] = q_nope (row r's head h at r * q_stride + h * 256) times W_k[h]; grid (8 columns, rows / 64, 64 heads).
[[kernel]] void glm_absorb_nax(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                               const device bfloat* B [[buffer(2)]], const device bfloat* qp [[buffer(3)]],
                               device bfloat* ql [[buffer(4)]], constant int2& a [[buffer(5)]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat parts[3][32 * PAD]; // a step's 32 weight rows (k) by 64 outputs (n), three bf16 parts each
  const int rows = a.x, q_stride = a.y;
  const int col = int(tg.x) * 64, row = int(tg.y) * 64, h = int(tg.z);
  const int t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - row - tm, 0, 32);
  const short2 home = frag_home(ushort(lane));
  const device bfloat* x = qp + long(row + tm) * q_stride + h * NOPE;
  // thread t dequantizes weight row k = t / 4 of the step, outputs 16 (t % 4) .. +16 of this block (one scale group)
  const int wk = t / 4, wn = 16 * (t % 4);
  frag<float> acc[2][2];
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k0 = 0; k0 < NOPE; k0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const long wrow = long(h) * PER_HEAD + k0 + wk;
      const device uchar* wb = (const device uchar*)W + wrow * (LATENT / 2) + (col + wn) / 2;
      const float s = float(S[wrow * (LATENT / 64) + (col + wn) / 64]);
      const float b = float(B[wrow * (LATENT / 64) + (col + wn) / 64]);
      TF_UNROLL
      for (short i = 0; i < 8; i++) {
        const uchar q = wb[i];
        TF_UNROLL
        for (short hi = 0; hi < 2; hi++) {
          const float v = s * float(hi == 0 ? (q & 0x0f) : (q >> 4)) + b;
          const bfloat p1 = bfloat(v);
          const float r1 = v - float(p1);
          const bfloat p2 = bfloat(r1);
          const int at = wk * PAD + wn + 2 * i + hi;
          parts[0][at] = p1;
          parts[1][at] = p2;
          parts[2][at] = bfloat(r1 - float(p2));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live > 0) {
      TF_UNROLL
      for (short kk = 0; kk < 32; kk += 16) {
        frag<bfloat> xa[2];
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          if (live == 32) {
            frag_get(xa[i], x, q_stride, 16 * i, k0 + kk, home);
          } else {
            frag_get_in(xa[i], x, q_stride, 16 * i, k0 + kk, home, live, NOPE);
          }
        }
        TF_UNROLL
        for (short pp = 0; pp < 3; pp++) {
          frag<bfloat> b0, b1;
          frag_get(b0, (const threadgroup bfloat*)parts[pp], PAD, kk, tn, home);
          frag_get(b1, (const threadgroup bfloat*)parts[pp], PAD, kk, tn + 16, home);
          TF_UNROLL
          for (short i = 0; i < 2; i++) mma_16x32<false, false>(acc[i][0], acc[i][1], xa[i], b0, b1);
        }
      }
    }
  }
  if (live == 0) return;
  device bfloat* out = ql + (long(row + tm) * HEADS + h) * LATENT + col + tn;
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 32) {
        frag_put(acc[i][j], out, HEADS * LATENT, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], out, HEADS * LATENT, 16 * i, 16 * j, home, live, 32);
      }
    }
  }
}
