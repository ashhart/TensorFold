// The dense twin for windows (after tile_qmv.metal): 1-32 rows a threadgroup, codes unpacked in registers into MMAs.

constant constexpr int TR_ROWS = 32;  // rows a call at most: four 8-row MMA blocks
constant constexpr int TR_SG = 2;  // simdgroups a threadgroup, 8 columns each

// Y^T = W X^T in MMA blocks: a lane holds column n0 + fm's codes k = fn, fn + 1 and rows fn, fn + 1's inputs at k = fm.
struct TrLane { int fm, fn; };
inline TrLane tr_lane(uint lane) {
  const int l = int(lane), qid = l / 4;
  return { (qid & 4) + ((l / 2) % 4), (qid & 2) * 2 + (l % 2) * 2 };
}

inline void tr_set(thread simdgroup_matrix<bfloat, 8, 8>& m, uint pair) {
  m.thread_elements()[0] = as_type<bfloat>(ushort(pair & 0xFFFFu));
  m.thread_elements()[1] = as_type<bfloat>(ushort(pair >> 16));
}

// A column's group g as 2 BITS words (the lane's whole column group).
template <int BITS>
inline void tr_load(thread uint* w, const device uint8_t* col, int g) {
  const device uint4* src = (const device uint4*)(col + g * (8 * BITS));
  #pragma unroll
  for (int i = 0; i < BITS / 2; ++i) {
    const uint4 v = src[i];
    w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
  }
}

// Step kk's A elements: codes kk + fn and kk + fn + 1 of the lane's column as a bf16 pair.
template <int BITS>
inline uint tr_a(const thread uint* w, int kk, int fn) {
  uint lo, hi;
  if (BITS == 4) {
    const uint pair = (w[kk / 8] >> (4 * fn)) & 0xFFu;
    lo = pair & 15u; hi = pair >> 4;
  } else if (BITS == 6) {
    const int bit = 6 * kk, at = bit / 32;  // the two codes' 12 bits sit in words at and at + 1 (bit % 32 <= 16)
    const uint v = uint(((ulong(w[at + 1]) << 32) | w[at]) >> (bit % 32 + 6 * fn));
    lo = v & 63u; hi = (v >> 6) & 63u;
  } else {
    const uint v = (fn < 4 ? w[kk / 4] : w[kk / 4 + 1]) >> (8 * (fn % 4));
    lo = v & 255u; hi = (v >> 8) & 255u;
  }
  return (as_type<uint>(float(lo)) >> 16) | (as_type<uint>(float(hi)) & 0xFFFF0000u);
}

// Row r's sum over group g (x0 then + x1 ... + x63, as tf_tile_xsum).
inline float tr_sum(const device bfloat* x, int g, float one) {
  const device uint4* src = (const device uint4*)(x + size_t(g) * 64);
  float t = 0.0f;
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 v = src[kb];
    const uint w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = as_type<float>((j % 2) ? (w[j / 2] & 0xFFFF0000u) : (w[j / 2] << 16));
      t = kb == 0 && j == 0 ? f : fma(f, one, t);
    }
  }
  return t;
}

// OUT[r] = X[r] W^T on the simdgroup's 8 columns, GI groups at a time (4 MMA chains).
template <int BITS, int NB>
inline void tr_window(const device bfloat* X, int K, int rows, const device uint8_t* W, const device bfloat* SC,
                      const device bfloat* BI, int n0, int N, device bfloat* OUT, float one, threadgroup float* sums,
                      uint tid, uint lane) {
  constexpr int GI = NB == 4 ? 1 : NB == 2 ? 2 : 4;
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  const TrLane L = tr_lane(lane);
  const device ushort* xb[NB][2];
  #pragma unroll
  for (int rb = 0; rb < NB; ++rb)
    for (int e = 0; e < 2; ++e) {
      const int r = min(rb * 8 + L.fn + e, rows - 1);
      xb[rb][e] = (const device ushort*)(X + size_t(r) * K) + L.fm;
    }
  const device uint8_t* wc = W + size_t(n0 + L.fm) * stride;
  const device bfloat* sc = SC + size_t(n0 + L.fm) * G;
  const device bfloat* bi = BI + size_t(n0 + L.fm) * G;
  float acc[NB][2];
  #pragma unroll
  for (int rb = 0; rb < NB; ++rb) acc[rb][0] = acc[rb][1] = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {
    const int gn = min(32, G - g0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < rows * gn; i += TR_SG * 32) {
      const int r = i / gn, g = i % gn;
      sums[r * 32 + g] = tr_sum(X + size_t(r) * K, g0 + g, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = g0; g < g0 + gn; g += GI) {
      const int live = min(GI, g0 + gn - g);
      uint w[GI][2 * BITS];
      #pragma unroll
      for (int i = 0; i < GI; ++i)
        if (i < live) tr_load<BITS>(w[i], wc, g + i);
      simdgroup_matrix<float, 8, 8> P[GI][NB];
      #pragma unroll
      for (int i = 0; i < GI; ++i)
        for (int rb = 0; rb < NB; ++rb) P[i][rb] = simdgroup_matrix<float, 8, 8>(0.0f);
      #pragma unroll
      for (int kk = 0; kk < 64; kk += 8) {
        #pragma unroll
        for (int i = 0; i < GI; ++i)
          if (i < live) {
            simdgroup_matrix<bfloat, 8, 8> a;
            tr_set(a, tr_a<BITS>(w[i], kk, L.fn));
            #pragma unroll
            for (int rb = 0; rb < NB; ++rb) {
              simdgroup_matrix<bfloat, 8, 8> b;
              const int k = (g + i) * 64 + kk;
              tr_set(b, uint(xb[rb][0][k]) | (uint(xb[rb][1][k]) << 16));
              simdgroup_multiply_accumulate(P[i][rb], a, b, P[i][rb]);
            }
          }
      }
      #pragma unroll
      for (int i = 0; i < GI; ++i)
        if (i < live) {
          const float s = float(sc[g + i]), bb = float(bi[g + i]);
          #pragma unroll
          for (int rb = 0; rb < NB; ++rb)
            for (int e = 0; e < 2; ++e) {
              const float xs = sums[(rb * 8 + L.fn + e) * 32 + (g + i - g0)];
              acc[rb][e] = fma(bb, xs, fma(s, P[i][rb].thread_elements()[e], acc[rb][e]));
            }
        }
    }
  }
  #pragma unroll
  for (int rb = 0; rb < NB; ++rb)
    for (int e = 0; e < 2; ++e) {
      const int r = rb * 8 + L.fn + e;
      if (r < rows) OUT[size_t(r) * N + n0 + L.fm] = bfloat(acc[rb][e]);
    }
}

// Y [R, N] = X [R, K] W^T for R <= TR_ROWS rows: tg.x's 16 columns over every row, in the row blocks R takes.
template <int BITS>
[[kernel]] void tf_tile_qmv_rows(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                 const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                 constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sums[TR_ROWS * 32];
  const int n0 = (int(tg.x) * TR_SG + int(sg)) * 8;
  if (a.R <= 8) tr_window<BITS, 1>(X, a.K, a.R, W, SC, BI, n0, a.N, OUT, a.one, sums, tid, lane);
  else if (a.R <= 16) tr_window<BITS, 2>(X, a.K, a.R, W, SC, BI, n0, a.N, OUT, a.one, sums, tid, lane);
  else tr_window<BITS, 4>(X, a.K, a.R, W, SC, BI, n0, a.N, OUT, a.one, sums, tid, lane);
}

template [[host_name("tf_tile_qmv_rows_q4")]] [[kernel]] decltype(tf_tile_qmv_rows<4>) tf_tile_qmv_rows<4>;
template [[host_name("tf_tile_qmv_rows_q6")]] [[kernel]] decltype(tf_tile_qmv_rows<6>) tf_tile_qmv_rows<6>;
template [[host_name("tf_tile_qmv_rows_q8")]] [[kernel]] decltype(tf_tile_qmv_rows<8>) tf_tile_qmv_rows<8>;
