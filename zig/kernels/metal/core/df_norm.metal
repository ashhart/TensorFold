// DFlash2 drafters' RMS rows, bf16(w * bf16(x * rsqrt(mean(x^2) + eps))): TF_DF_NORM(P) = P##_args, P##_row, kernel P.
#define TF_DF_NORM(P) \
  struct P##_args { uint rows, width; float eps; uint add; }; \
  inline void P##_row(device float* x, const device bfloat* w, uint width, float eps, device bfloat* out, \
      uint t, uint n, threadgroup float* part) { \
    float sum = 0.0f; \
    for (uint i = t; i < width; i += n) sum = fma(x[i], x[i], sum); \
    sum = simd_sum(sum); \
    if (t % 32 == 0) part[t / 32] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float total = 0.0f; \
    for (uint s = 0; s < n / 32; ++s) total += part[s]; \
    const float inv = precise::rsqrt(total / float(width) + eps); \
    for (uint i = t; i < width; i += n) out[i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv))); \
  } \
  [[kernel]] void P(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]], \
      const device bfloat* w [[buffer(2)]], constant P##_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]], \
      uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], \
      uint n [[threads_per_threadgroup]]) { \
    threadgroup float part[32]; \
    device float* x = R + (size_t)row * a.width; \
    if (a.add) \
      for (uint i = t; i < a.width; i += n) x[i] = float(bfloat(x[i] + float(X[(size_t)row * a.width + i]))); \
    threadgroup_barrier(mem_flags::mem_device); \
    P##_row(x, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part); \
  }
// TF_DF_NORM16(P): kernel P##16, the same norm of bf16 rows (fc's output into the context's hidden_norm).
#define TF_DF_NORM16(P) \
  [[kernel]] void P##16(const device bfloat* X [[buffer(0)]], const device bfloat* w [[buffer(1)]], \
      constant P##_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]], \
      uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) { \
    threadgroup float part[32]; \
    const device bfloat* x = X + (size_t)row * a.width; \
    float sum = 0.0f; \
    for (uint i = t; i < a.width; i += n) sum = fma(float(x[i]), float(x[i]), sum); \
    sum = simd_sum(sum); \
    if (t % 32 == 0) part[t / 32] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float total = 0.0f; \
    for (uint s = 0; s < n / 32; ++s) total += part[s]; \
    const float inv = precise::rsqrt(total / float(a.width) + a.eps); \
    for (uint i = t; i < a.width; i += n) \
      out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv))); \
  }
