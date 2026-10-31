// Default-off M5 tiled SDPA experiment: strict QK, BF16 high+residual PV, FP32 accumulator; no bit-identity or fidelity claim before measured envelope gates.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
// M5 tensor-unit fragments shared by every NAX kernel (ops and prefill): layout, loads, stores and the 16x32x16 op.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace tfp {
using namespace metal;

// Full unroll: register arrays indexed by a loop counter stay in registers only when the loop unrolls.
#define TF_UNROLL _Pragma("clang loop unroll(full)")

// One simdgroup's 16x16 fragment: 8 values a lane.
template <typename T>
using frag = vec<T, 8>;

#ifndef TF_SIMD_FRAGS
// Lane l holds rows home.y and home.y + 8, columns home.x .. home.x + 3 of every fragment (the M5 operand layout).
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// The 16x16 block at (r, c) of a row-major matrix with leading dimension ld, in device or threadgroup memory.
template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + (e & 3)]);
  }
}

// frag_get with zeros outside rows < nr and columns < nc (both relative to p); nothing outside is read.
template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + (e & 3)] = O(f[e]);
  }
}

template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}

// (lo | hi) += a * (b0 | b1): one strict 16x32x16 multiply-accumulate on the tensor unit.
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// frag_get from a threadgroup block in two 8-byte loads: (c + home.x) * 2 and ld * 2 must be multiples of 8.
inline void frag_get_tg(thread frag<bfloat>& f, const threadgroup bfloat* p, int ld, int r, int c, short2 home) {
  const threadgroup bfloat* q = p + (r + home.y) * ld + (c + home.x);
  f = frag<bfloat>(*(const threadgroup bfloat4*)q, *(const threadgroup bfloat4*)(q + 8 * ld));
}

// Element e's column past home.x (a plain index: M5 kernels keep their exact text); its row is home.y + (e >> 2) * 8.
#define TF_COL(e) (e & 3)
// A transposed right operand's block (its rows are the op's columns): on M5 it loads like any other.
#define frag_get_t frag_get
#define frag_get_t_in frag_get_in
#else
// TF_SIMD_FRAGS (before M5, 8x8 simdgroup matrices): lane l holds rows home.y, home.y + 8, columns home.x + {0,1,8,9}.
inline short2 frag_home(ushort l) {
  return short2(short(((l & 8) >> 1) | ((l & 1) << 1)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

#define TF_COL(e) (((e) & 1) + 8 * (((e) >> 1) & 1))

template <typename T, typename P>
inline void frag_get(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  const P q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(q[(e >> 2) * 8 * ld + TF_COL(e)]);
  }
}

template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

// The block at (r, c) as a transposed right operand: its rows on the column pattern, its columns on the row pattern.
template <typename T, typename P>
inline void frag_get_t(thread frag<T>& f, P p, int ld, int r, int c, short2 home) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = T(p[(r + home.x + TF_COL(e)) * ld + c + home.y + (e >> 2) * 8]);
  }
}

template <typename T, typename S>
inline void frag_get_t_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.x + TF_COL(e), cc = c + home.y + (e >> 2) * 8;
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

template <typename O>
inline void frag_put(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home) {
  device O* q = p + (r + home.y) * ld + (c + home.x);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    q[(e >> 2) * 8 * ld + TF_COL(e)] = O(f[e]);
  }
}

template <typename O>
inline void frag_put_in(thread const frag<float>& f, device O* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + TF_COL(e);
    if (rr < nr && cc < nc) {
      p[rr * ld + cc] = O(f[e]);
    }
  }
}

// The op as on M5; right operand and accumulator hold a lane's first-half columns at (e >> 2) * 8 + (e & 3).
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const short j = (e >> 2) * 8 + (e & 3);
    left[e] = a[e];
    right[j] = b0[e];
    right[j + 4] = b1[e];
    acc[j] = lo[e];
    acc[j + 4] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const short j = (e >> 2) * 8 + (e & 3);
    lo[e] = acc[j];
    hi[e] = acc[j + 4];
  }
}
#endif

// frag_get from a block whose rows home.y and home.y + 8 start at r0 and r1; an unflagged row reads as zero.
template <typename T, typename S>
inline void frag_get_rows(thread frag<T>& f, const device S* r0, const device S* r1, int c, short2 home, bool lo,
                          bool hi) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = ((e >> 2) ? hi : lo) ? T(((e >> 2) ? r1 : r0)[c + home.x + TF_COL(e)]) : T(0);
  }
}

// frag_put to a block whose rows home.y and home.y + 8 start at r0 and r1; only flagged rows are written.
template <typename O>
inline void frag_put_rows(thread const frag<float>& f, device O* r0, device O* r1, int c, short2 home, bool lo,
                          bool hi) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    if ((e >> 2) ? hi : lo) {
      ((e >> 2) ? r1 : r0)[c + home.x + TF_COL(e)] = O(f[e]);
    }
  }
}

}  // namespace tfp


namespace tfp {

// MLX's Limits<float>::finite_min, the score of a masked key.
constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr int kAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void row_fold(thread const frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

// Convert canonical 16x16 fragment coordinates to the strict mixed-type MPP operand ABI.
inline void mma_pv_residual(thread frag<float>& lo,thread frag<float>& hi,const thread frag<float>& a,const thread frag<bfloat>& b0,const thread frag<bfloat>& b1){
 frag<bfloat> high,low;
 TF_UNROLL for(short e=0;e<8;e++){high[e]=bfloat(a[e]);low[e]=bfloat(a[e]-float(high[e]));}
 mma_16x32<false,false>(lo,hi,high,b0,b1);
 mma_16x32<false,false>(lo,hi,low,b0,b1);
}

// One 64-query block of one head, online softmax over 32-key blocks; P holds the ATTN_PARAMS fields, F[0] the scale.
template <typename T, typename Out, bool AQ, bool AK, bool CAUSAL>
inline void attention_nax(const device T* Q, const device T* K, const device T* V, device Out* O, const thread int* P,
                          const thread float* F, uint3 tg, uint sg, uint lane, thread uint& nonfinite_rows) {
  constexpr int BQ = 64, BK = 32;
  const int qs = P[14], ks = P[17], vs = P[20], os = P[23];
  const int kv = int(tg.y) / P[4];
  Q += long(tg.z) * P[12] + long(tg.y) * P[13] + long(tg.x) * BQ * qs;
  K += long(tg.z) * P[15] + long(kv) * P[16];
  V += long(tg.z) * P[18] + long(kv) * P[19];
  O += long(tg.z) * P[21] + long(tg.y) * P[22] + long(tg.x) * BQ * os;
  const float scale2 = F[0] * 1.44269504089f;
  const short tm = 16 * short(sg);
  Q += tm * qs;
  const short2 home = frag_home(ushort(lane));

  frag<float> acc[8];               // 16 query rows x 128 values
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    acc[i] = frag<float>(0);
  }
  float top[2] = {kMasked, kMasked}, total[2] = {0.0f, 0.0f};

  int blocks = P[6], first_masked = P[6];
  if (CAUSAL) {
    blocks = min(P[6], ((int(tg.x) + 1) * BQ + P[11] + BK - 1) / BK);
    first_masked = max(0, int(tg.x) * BQ + P[11]) / BK;
  }
  const bool last_q = int(tg.x) == P[7];
  const short q_rows = short(P[9] - tm), k_rows = short(P[10]);

  const int first_block = P[24] == 0 ? 0 : max(0, int(tg.x)*BQ + P[11] + 1 - P[24]) / BK;
  K += first_block*BK*ks; V += first_block*BK*vs;
  for (int kb = first_block; kb < blocks; kb++) {
    const bool last_k = kb == P[8];
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < 8; d++) {
      frag<T> q, k0, k1;
      if (!AQ && last_q) {
        frag_get_in(q, Q, qs, 0, 16 * d, home, q_rows, kAnyCols);
      } else {
        frag_get(q, Q, qs, 0, 16 * d, home);
      }
      if (!AK && last_k) {
        frag_get_t_in(k0, K, ks, 0, 16 * d, home, k_rows, kAnyCols);
        frag_get_t_in(k1, K, ks, 16, 16 * d, home, k_rows, kAnyCols);
      } else {
        frag_get_t(k0, K, ks, 0, 16 * d, home);
        frag_get_t(k1, K, ks, 16, 16 * d, home);
      }
      mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] *= scale2;
      }
    }
    if (!AK && last_k) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kMasked;
        }
      }
    }
    if (CAUSAL && kb >= first_masked) {
      const int r0 = int(tg.x) * BQ + P[11] + tm + home.y, c0 = kb * BK + home.x;
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          s[f][e] = (r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) ? kMasked : s[f][e];
        }
      }
    }

    if(P[24]>0){
      const int r0=int(tg.x)*BQ+P[11]+tm+home.y,c0=kb*BK+home.x;
      TF_UNROLL for(short f=0;f<2;f++) TF_UNROLL for(short e=0;e<8;e++)
        if(c0+16*f+TF_COL(e)<r0+(e>>2)*8+1-P[24])s[f][e]=kMasked;
    }
    float top_new[2] = {top[0], top[1]};
    row_fold<true>(s[0], top_new);
    row_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
      }
    }
    float scale_old[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      scale_old[h] = fast::exp2(top[h] - top_new[h]);
      top[h] = top_new[h];
      total[h] = total[h] * scale_old[h];
    }
    row_fold<false>(s[0], total);
    row_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[i][e] = acc[i][e] * scale_old[e >> 2];
      }
    }
    simdgroup_barrier(mem_flags::mem_none);

    TF_UNROLL
    for (short d = 0; d < 8; d += 2) {
      if (d == 4) {
        threadgroup_barrier(mem_flags::mem_none);
      }
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<T> v0, v1;
        if (!AK && last_k) {
          frag_get_in(v0, V, vs, 16 * k, 16 * d, home, k_rows, kAnyCols);
          frag_get_in(v1, V, vs, 16 * k, 16 * d + 16, home, k_rows, kAnyCols);
        } else {
          frag_get(v0, V, vs, 16 * k, 16 * d, home);
          frag_get(v1, V, vs, 16 * k, 16 * d + 16, home);
        }
        mma_pv_residual(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
    K += BK * ks;
    V += BK * vs;
  }

  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    inv[h] = 1.f / total[h];
  }
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      acc[i][e] = acc[i][e] * inv[e >> 2];
    }
  }
  // Each SIMD group owns 16 rows; every lane receives the same repair mask.
  uint local_nonfinite_rows = 0;
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      if (!isfinite(float(bfloat(acc[i][e]))))
        local_nonfinite_rows |= 1u << (home.y + (e >> 2) * 8);
    }
  }
  nonfinite_rows = simd_or(local_nonfinite_rows);
  O += tm * os;
  if (!AQ && last_q) {
    if (q_rows <= 0) {
      return;
    }
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      frag_put_in(acc[i], O, os, 0, 16 * i, home, q_rows, kAnyCols);
    }
  } else {
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      frag_put(acc[i], O, os, 0, 16 * i, home);
    }
  }
}

}  // namespace tfp

[[kernel,max_total_threads_per_threadgroup(128)]] void lfm3_attn_tiled_bf16(
 const device bfloat* q [[buffer(0)]],const device bfloat* k [[buffer(1)]],const device bfloat* v [[buffer(2)]],
 constant lfm3_attn_args& a [[buffer(3)]],device bfloat* out [[buffer(8)]],constant uint& rows [[buffer(9)]],
 uint3 tg [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 int P[25]={0};P[4]=4;P[6]=(a.len+rows+31)/32;P[7]=(rows-1)/64;P[8]=(a.len+rows-1)/32;P[9]=((rows-1)%64)+1;P[10]=((a.len+rows-1)%32)+1;P[11]=a.len;
 P[13]=128;P[14]=2048;P[16]=a.cap*128;P[17]=128;P[19]=a.cap*128;P[20]=128;P[22]=128;P[23]=2048;P[24]=a.window;
 float F[1]={a.scale};
#ifdef TF_TILED_BAD_SCALE
 F[0]*=0.5f;
#endif
 uint nonfinite_rows = 0;
 tfp::attention_nax<bfloat,bfloat,false,false,true>(q,k,v,out,P,F,tg,sg,lane,nonfinite_rows);
 // Fragment stores and replay stores assign columns to different lanes.
 if (nonfinite_rows != 0) simdgroup_barrier(mem_flags::mem_device);
 // Replay only affected visible rows using the established causal/window and extended-range path.
 for (uint i=0;i<16;++i) {
  const uint row=tg.x*64+sg*16+i;
  if (row>=rows || (nonfinite_rows & (1u<<i))==0) continue;
  l3_x repaired[4];
  l3_attention_replay(q,k,v,a,tg.y,row,lane,repaired);
  for(uint j=0;j<4;++j)
   reinterpret_cast<device ushort*>(out)[(size_t(row)*16+tg.y)*128+lane+32*j]=l3_x_pack(repaired[j]);
 }
}
