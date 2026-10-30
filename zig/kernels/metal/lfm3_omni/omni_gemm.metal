// LFM3-Omni's simdgroup products for chips without tensor units, the depth decoder's picks and the log-mel front end.
#include "om_common.h"

inline void better(thread float& best, thread int& at, float v, int i) {
  if (v > best || (v == best && i < at)) {
    best = v;
    at = i;
  }
}

// ---- C = A B (either transposed) on the simdgroup matrix units from fp32 tiles, fp32 sums; a batch on z

struct GemmArgs {
  int M, N, K, lda, ldb, ldc, ta, tb;
  long sa, sb, sc;
};

template <typename TI>
inline float ld_in(device const TI* p, long i);
template <>
inline float ld_in<bf>(device const bf* p, long i) {
  return f(p[i]);
}
template <>
inline float ld_in<float>(device const float* p, long i) {
  return p[i];
}

template <typename TO>
inline void st_out(device TO* p, long i, float v);
template <>
inline void st_out<bf>(device bf* p, long i, float v) {
  p[i] = b(v);
}
template <>
inline void st_out<float>(device float* p, long i, float v) {
  p[i] = v;
}

template <typename TI, typename TO>
inline void gemm_body(device const TI* A, device const TI* B, device TO* C, constant GemmArgs& a, uint3 tg, uint tid, uint sg,
                      uint lane, threadgroup float* As, threadgroup float* Bs, threadgroup float* Cs) {
  constexpr int BM = 64, BN = 64, BK = 16;
  A += (long)tg.z * a.sa;
  B += (long)tg.z * a.sb;
  C += (long)tg.z * a.sc;
  const int m0 = tg.y * BM, n0 = tg.x * BN;
  const int wm = (sg >> 1) * 32, wn = (sg & 1) * 32;
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (int k0 = 0; k0 < a.K; k0 += BK) {
    // A tile [BM][BK], B tile [BK][BN]: 1024 values each, 8 a thread
    for (int i = tid; i < BM * BK; i += 128) {
      int r, k;
      if (a.ta) {
        r = i % BM;
        k = i / BM;
      } else {
        r = i / BK;
        k = i % BK;
      }
      const int gm = m0 + r, gk = k0 + k;
      float v = 0.0f;
      if (gm < a.M && gk < a.K) v = ld_in<TI>(A, a.ta ? (long)gk * a.lda + gm : (long)gm * a.lda + gk);
      As[r * BK + k] = v;
    }
    for (int i = tid; i < BK * BN; i += 128) {
      int k, c;
      if (a.tb) {
        k = i % BK;
        c = i / BK;
      } else {
        k = i / BN;
        c = i % BN;
      }
      const int gk = k0 + k, gn = n0 + c;
      float v = 0.0f;
      if (gn < a.N && gk < a.K) v = ld_in<TI>(B, a.tb ? (long)gn * a.ldb + gk : (long)gk * a.ldb + gn);
      Bs[k * BN + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4], bm[4];
      for (int i = 0; i < 4; ++i) simdgroup_load(am[i], As + (wm + i * 8) * BK + kk, BK);
      for (int j = 0; j < 4; ++j) simdgroup_load(bm[j], Bs + kk * BN + wn + j * 8, BN);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* cs = Cs + sg * 64;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], cs, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = lane; e < 64; e += 32) {
        const int gm = m0 + wm + i * 8 + e / 8, gn = n0 + wn + j * 8 + e % 8;
        if (gm < a.M && gn < a.N) st_out<TO>(C, (long)gm * a.ldc + gn, cs[e]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define OM_GEMM(NAME, TI, TO)                                                                                                      \
  [[max_total_threads_per_threadgroup(128)]] kernel void NAME(device const TI* A [[buffer(0)]], device const TI* B [[buffer(1)]],  \
                                                             device TO* C [[buffer(2)]], constant GemmArgs& a [[buffer(3)]],       \
                                                             uint3 tg [[threadgroup_position_in_grid]],                             \
                                                             uint tid [[thread_index_in_threadgroup]],                              \
                                                             uint sg [[simdgroup_index_in_threadgroup]],                            \
                                                             uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float As[64 * 16];                                                                                                 \
    threadgroup float Bs[16 * 64];                                                                                                 \
    threadgroup float Cs[4 * 64];                                                                                                  \
    gemm_body<TI, TO>(A, B, C, a, tg, tid, sg, lane, As, Bs, Cs);                                                                  \
  }
OM_GEMM(om_gemm_bb, bf, bf)
OM_GEMM(om_gemm_bf, bf, float)
OM_GEMM(om_gemm_ff, float, float)

// ---- the depth decoder's picks on the device

inline ulong mix64(ulong z) {
  z += 0x9E3779B97F4A7C15ul;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
  return z ^ (z >> 31);
}

struct PickArgs {
  int p0, hn;
  float penalty, temp;
  ulong seed;
  int frame, NQ, size, pad;
};

// Codebook ORDER[p0 + group]'s code after HIST's repetition penalty: greedy (lower id on ties) or a hashed Gumbel max.
[[max_total_threads_per_threadgroup(256)]] kernel void om_depth_pick(device const bf* LOG [[buffer(0)]], device const int* ORDER [[buffer(1)]],
                                                                  device const int* HIST [[buffer(2)]], device int* CODES [[buffer(3)]],
                                                                  device int* MASKED [[buffer(4)]], constant PickArgs& a [[buffer(5)]],
                                                                  uint blk [[threadgroup_position_in_grid]],
                                                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uchar seen[2048];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const int q = ORDER[a.p0 + blk];
  for (int i = tid; i < a.size; i += 256) seen[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = tid; j < a.hn; j += 256) seen[HIST[j * a.NQ + q]] = 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = tid; i < a.size; i += 256) {
    float v = f(LOG[(long)q * a.size + i]);
    if (seen[i]) v = v < 0.0f ? v * a.penalty : v / a.penalty;
    if (a.temp > 0.0f) {
      const ulong h = mix64(a.seed ^ mix64(((ulong)a.frame << 32) ^ ((ulong)q << 16) ^ (ulong)i));
      const float u = (float(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
      v = v / a.temp - precise::log(-precise::log(u));
    }
    better(best, at, v, i);
  }
  for (ushort o = 16; o > 0; o >>= 1) better(best, at, simd_shuffle_xor(best, o), simd_shuffle_xor(at, o));
  if ((tid & 31) == 0) {
    bv[tid >> 5] = best;
    bi[tid >> 5] = at;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    best = bv[0];
    at = bi[0];
    for (int w = 1; w < 8; ++w) better(best, at, bv[w], bi[w]);
    CODES[q] = at;
    MASKED[q] = 0;
  }
}

struct FrameEndArgs {
  int NQ, eos, slot, push;
};

// A finished frame: any end-of-speech code makes it all EOS, else with push it joins HIST's ring at row `slot`.
kernel void om_frame_end(device int* CODES [[buffer(0)]], device int* HIST [[buffer(1)]], constant FrameEndArgs& a [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  bool any = false;
  for (int q = 0; q < a.NQ; ++q) any |= CODES[q] == a.eos;
  if (any)
    for (int q = 0; q < a.NQ; ++q) CODES[q] = a.eos;
  else if (a.push)
    for (int q = 0; q < a.NQ; ++q) HIST[a.slot * a.NQ + q] = CODES[q];
}

// ---- the log-mel front end: frames of the padded clip, |DFT|^2 by product, mel by product, log

struct MelArgs {
  int len, nf, nfft, hop;
};

kernel void om_mel_frames(device const float* X [[buffer(0)]], device const float* WIN [[buffer(1)]], device float* FR [[buffer(2)]],
                          constant MelArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.nfft) return;
  const int t = int(i / a.nfft), n = int(i % a.nfft);
  const int s = t * a.hop + n;
  FR[i] = s < a.len ? X[s] * WIN[n] : 0.0f;
}

struct PowerArgs {
  int nf, bins;
};

kernel void om_power(device const float* S [[buffer(0)]], device float* P [[buffer(1)]], constant PowerArgs& a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.bins) return;
  const long t = i / a.bins;
  const int k = int(i % a.bins);
  const float re = S[t * 2 * a.bins + k], im = S[t * 2 * a.bins + a.bins + k];
  const float m = precise::sqrt(re * re + im * im);
  P[i] = m * m;
}

struct LogmelArgs {
  int nf, mels;
  float floor_;
};

kernel void om_logmel(device const float* M [[buffer(0)]], device float* OUT [[buffer(1)]], constant LogmelArgs& a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.mels) return;
  const int t = int(i / a.mels), m = int(i % a.mels);
  OUT[(long)m * a.nf + t] = (max(precise::log10(M[i]), a.floor_) + 4.0f) / 4.0f;
}
