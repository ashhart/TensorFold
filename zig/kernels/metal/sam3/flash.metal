// SAM3 flash attention tiles on the tensor unit: 64 query rows of one head and batch item per threadgroup.
#include <metal_stdlib>
using namespace metal;

struct Sam3Flash {
  uint rows;
  uint keys;
  uint heads;
  uint batch;
};

// Simdgroup sg owns 16 query rows; softmax runs in base 2 with FP32 running maximum, sum and accumulator.
template <typename T, int D, int BK>
inline void sam3_flash_tile(const device T* Q, const device T* K, const device T* V, device T* O, constant Sam3Flash& p,
                            uint3 tg, uint sg, uint lane) {
  constexpr int F = D / 16;
  constexpr int KF = BK / 16;
  const int rows = int(p.rows), keys = int(p.keys), ld = int(p.heads) * D;
  const long head = long(tg.y) * D;
  const device T* q_base = Q + long(tg.z) * long(rows) * ld + head;
  const device T* k_base = K + long(tg.z) * long(keys) * ld + head;
  const device T* v_base = V + long(tg.z) * long(keys) * ld + head;
  device T* o_base = O + long(tg.z) * long(rows) * ld + head;
  const int r0 = int(tg.x) * 64 + 16 * int(sg);
  const short2 home = tfp::frag_home(ushort(lane));

  tfp::frag<T> q[F];
  TF_UNROLL
  for (short d = 0; d < F; d++) {
    tfp::frag_get_in(q[d], q_base, ld, r0, 16 * d, home, rows, D);
  }
  float m[2] = {tfp::kMasked, tfp::kMasked};
  float l[2] = {0.0f, 0.0f};
  tfp::frag<float> acc[F];
  TF_UNROLL
  for (short d = 0; d < F; d++) {
    acc[d] = tfp::frag<float>(0);
  }
  const float c = 1.44269504089f * metal::precise::rsqrt(float(D));

  for (int j0 = 0; j0 < keys; j0 += BK) {
    const bool full = j0 + BK <= keys;
    tfp::frag<float> s[KF];
    TF_UNROLL
    for (short f = 0; f < KF; f++) {
      s[f] = tfp::frag<float>(0);
    }
    TF_UNROLL
    for (short d = 0; d < F; d++) {
      TF_UNROLL
      for (short h = 0; h < KF / 2; h++) {
        tfp::frag<T> k0, k1;
        if (full) {
          tfp::frag_get(k0, k_base, ld, j0 + 32 * h, 16 * d, home);
          tfp::frag_get(k1, k_base, ld, j0 + 32 * h + 16, 16 * d, home);
        } else {
          tfp::frag_get_in(k0, k_base, ld, j0 + 32 * h, 16 * d, home, keys, D);
          tfp::frag_get_in(k1, k_base, ld, j0 + 32 * h + 16, 16 * d, home, keys, D);
        }
        tfp::mma_16x32<false, true>(s[2 * h], s[2 * h + 1], q[d], k0, k1);
      }
    }
    TF_UNROLL
    for (short f = 0; f < KF; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int key = j0 + 16 * f + home.x + TF_COL(e);
        s[f][e] = key < keys ? s[f][e] * c : tfp::kMasked;
      }
    }
    float top[2] = {m[0], m[1]};
    TF_UNROLL
    for (short f = 0; f < KF; f++) {
      tfp::row_fold<true>(s[f], top);
    }
    float alpha[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      alpha[h] = metal::fast::exp2(m[h] - top[h]);
      l[h] = l[h] * alpha[h];
      m[h] = top[h];
    }
    TF_UNROLL
    for (short f = 0; f < KF; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] = float(T(metal::fast::exp2(s[f][e] - top[e >> 2])));
      }
      tfp::row_fold<false>(s[f], l);
    }
    TF_UNROLL
    for (short d = 0; d < F; d++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        acc[d][e] = acc[d][e] * alpha[e >> 2];
      }
    }
    TF_UNROLL
    for (short f = 0; f < KF; f++) {
      const tfp::frag<T> probabilities = tfp::frag<T>(s[f]);
      TF_UNROLL
      for (short d = 0; d < F; d += 2) {
        tfp::frag<T> v0, v1;
        if (full) {
          tfp::frag_get(v0, v_base, ld, j0 + 16 * f, 16 * d, home);
          tfp::frag_get(v1, v_base, ld, j0 + 16 * f, 16 * d + 16, home);
        } else {
          tfp::frag_get_in(v0, v_base, ld, j0 + 16 * f, 16 * d, home, keys, D);
          tfp::frag_get_in(v1, v_base, ld, j0 + 16 * f, 16 * d + 16, home, keys, D);
        }
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], probabilities, v0, v1);
      }
    }
  }

  TF_UNROLL
  for (short d = 0; d < F; d++) {
    tfp::frag<float> out;
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      out[e] = acc[d][e] / l[e >> 2];
    }
    tfp::frag_put_in(out, o_base, ld, r0, 16 * d, home, rows, D);
  }
}

#define SAM3_FLASH_KERNEL(NAME, T, D, BK)                                                                          \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                                \
      const device T* q [[buffer(0)]], const device T* k [[buffer(1)]], const device T* v [[buffer(2)]],            \
      device T* out [[buffer(3)]], constant Sam3Flash& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                        \
    sam3_flash_tile<T, D, BK>(q, k, v, out, p, tg, sg, lane);                                                      \
  }

SAM3_FLASH_KERNEL(sam3_flash_f32_d32_k32, float, 32, 32)
SAM3_FLASH_KERNEL(sam3_flash_f32_d64_k32, float, 64, 32)
SAM3_FLASH_KERNEL(sam3_flash_f16_d32_k64, half, 32, 64)
SAM3_FLASH_KERNEL(sam3_flash_f16_d64_k32, half, 64, 32)
#undef SAM3_FLASH_KERNEL
