// DeepSeek V4.1 dense FP8 prompt rows on M5 tensor units: each E4M3 code times its E8M0 block scale, exact in bf16.

// The units sum the exact products over all of K into fp32, 16 deep a step.

// Each output depends only on its own row and K: rows a launch and the chunk change no bit.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

constant constexpr int FN_PAD = 64 + 8;

struct TfDsv41NaxArgs {
  uint K, N, R;
};

inline float fn_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

// A byte's E4M3 value times the block scale s (its value over 256 held exactly by half's bits, then 256 s), as bf16.
inline ushort fn_code(uint b, float s) {
  const float v = (b & 0x7Fu) == 0x7Fu && TF_CTL != 1 ? NAN : float(as_type<half>(ushort((b & 0x7Fu) << 7))) * 256.0f * s;
  return as_type<ushort>(bfloat((b & 0x80u) ? -v : v));
}

// Codes at bytes s and s + 1 of w as a bf16 pair.
inline uint fn_pair(uint w, uint s, float sc) {
  return uint(fn_code((w >> (8 * s)) & 0xFFu, sc)) | (uint(fn_code((w >> (8 * s + 8)) & 0xFFu, sc)) << 16);
}

// A thread's 32 codes of one weight row (one block's group, two 16-byte loads) times its scale into its tile row.
inline void fn_stage(uint4 lo, uint4 hi, float sc, threadgroup bfloat* out) {
  threadgroup uint4* o = (threadgroup uint4*)out;
  const uint w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  TF_UNROLL
  for (short i = 0; i < 4; i++) o[i] = uint4(fn_pair(w[2 * i], 0, sc), fn_pair(w[2 * i], 2, sc), fn_pair(w[2 * i + 1], 0, sc), fn_pair(w[2 * i + 1], 2, sc));
}

// 64 rows x 64 outputs a threadgroup; simdgroup s takes rows 32 (s / 2) .. + 31 by outputs 32 (s % 2) .. + 31.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_dsv41_fp8_nax(
    const device bfloat* X [[buffer(0)]], const device uchar* W [[buffer(1)]], const device uchar* S [[buffer(2)]],
    device T* Y [[buffer(3)]], constant TfDsv41NaxArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat tile[64 * FN_PAD];
  const int K = int(a.K), N = int(a.N), R = int(a.R), groups = K / 32;
  const int col = int(tg.x) * 64, row = int(tg.y) * 64;
  const int t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2);
  const int live = clamp(R - row - tm, 0, 32);
  const short2 home = frag_home(ushort(lane));
  const device bfloat* xr[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++)
    for (short h = 0; h < 2; h++) {
      const int i = row + tm + 16 * m + home.y + 8 * h;
      xr[m][h] = X + long(i < R ? i : row) * K;
    }
  frag<float> acc[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    acc[m][0] = frag<float>(0);
    acc[m][1] = frag<float>(0);
  }
  const device uchar* wr = W + long(col + t / 2) * K + 32 * (t % 2);
  threadgroup bfloat* mine = tile + (t / 2) * FN_PAD + 32 * (t % 2);
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = k / 32 + t % 2;
    const float s = TF_CTL == 2 && g == groups - 1 ? 0.0f : fn_e8m0(S[long((col + t / 2) / 32) * groups + g]);
    fn_stage(*(const device uint4*)(wr + k), *(const device uint4*)(wr + k + 16), s, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live == 0) continue;
    TF_UNROLL
    for (short q = 0; q < 4; q++) {
      frag<bfloat> b0, b1, x[2];
      frag_get_tg(b0, tile, FN_PAD, tn, 16 * q, home);
      frag_get_tg(b1, tile, FN_PAD, tn + 16, 16 * q, home);
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        const int c = k + 16 * q + home.x;
        x[m] = frag<bfloat>(*(const device bfloat4*)(xr[m][0] + c), *(const device bfloat4*)(xr[m][1] + c));
      }
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        mma_16x32<false, true>(acc[m][0], acc[m][1], x[m], b0, b1);
        if (TF_CTL == 3)
          for (short j = 0; j < 2; j++)
            for (short e = 0; e < 8; e++) acc[m][j][e] = float(half(acc[m][j][e]));
      }
    }
  }
  if (live == 0) return;
  device T* out = Y + long(row + tm) * N + col + tn;
  TF_UNROLL
  for (short m = 0; m < 2; m++)
    for (short j = 0; j < 2; j++) {
      if (live == 32) frag_put(acc[m][j], out, N, 16 * m, 16 * j, home);
      else frag_put_in(acc[m][j], out, N, 16 * m, 16 * j, home, live, 32);
    }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && t == 0) Y[long(R) * N] = T(1.0f);
}

template [[host_name("tf_dsv41_fp8_nax_f32")]] [[kernel]] decltype(tf_dsv41_fp8_nax<float>) tf_dsv41_fp8_nax<float>;
template [[host_name("tf_dsv41_fp8_nax_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_nax<bfloat>) tf_dsv41_fp8_nax<bfloat>;
