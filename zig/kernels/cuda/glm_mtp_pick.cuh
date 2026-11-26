// The host rank scan's exact float comparisons, expressed as integer tests so subnormals are never flushed.
#pragma once

#ifdef __CUDACC__
#define TF_GLM_MTP_INLINE __host__ __device__ __forceinline__
#else
#define TF_GLM_MTP_INLINE inline
#endif

TF_GLM_MTP_INLINE bool glm_mtp_nan(unsigned x) {
  return (x & 0x7fffffffu) > 0x7f800000u;
}

TF_GLM_MTP_INLINE unsigned glm_mtp_order(unsigned x) {
  return x ^ ((x >> 31) ? 0xffffffffu : 0x80000000u);
}

template <unsigned Control = 0>
TF_GLM_MTP_INLINE unsigned glm_mtp_pick_id(const unsigned* words, unsigned world) {
  unsigned best = 0;
  for (unsigned r = 1; r < world; ++r) {
    const unsigned a = words[4 * r], b = words[4 * best];
    const bool equal = a == b || ((a | b) & 0x7fffffffu) == 0;
    const bool tie = Control == 1 ? words[4 * r + 1] > words[4 * best + 1]
                                  : words[4 * r + 1] < words[4 * best + 1];
    const bool ordered = !glm_mtp_nan(a) && !glm_mtp_nan(b);
    if ((ordered && ((equal && tie) || (!equal && glm_mtp_order(a) > glm_mtp_order(b)))) ||
        (Control == 2 && glm_mtp_nan(b))) best = r;
  }
  return words[4 * best + 1];
}

#undef TF_GLM_MTP_INLINE
