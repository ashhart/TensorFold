// u32 copies, dst[i] = src[i]: TF_COPY_U32 for a grid of exactly n threads, TF_COPY_U32_N for any grid over n.
#define TF_COPY_U32(NAME)                                                                                          \
  kernel void NAME(const device uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],                           \
      uint i [[thread_position_in_grid]]) {                                                                        \
    dst[i] = src[i];                                                                                               \
  }
#define TF_COPY_U32_N(NAME)                                                                                        \
  kernel void NAME(const device uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],                           \
      constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {                                        \
    if (i < n) dst[i] = src[i];                                                                                    \
  }
