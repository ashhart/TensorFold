// Kernel NAME (ARGS: rows, width, bits, group): R row r = affine embedding row ids[r], bf16(scale * q + bias) in fp32.
#define TF_AFFINE_EMBED(NAME, ARGS)                                                                                \
  [[kernel]] void NAME(const device uint* ids [[buffer(0)]], const device uint* W [[buffer(1)]],                   \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant ARGS& a [[buffer(4)]],  \
      device float* R [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {                                        \
    if (p.x >= a.width || p.y >= a.rows) return;                                                                   \
    const size_t row = ids[p.y];                                                                                   \
    const uint per = 32 / a.bits, mask = (1u << a.bits) - 1;                                                       \
    const uint word = W[row * (a.width / per) + p.x / per];                                                        \
    const float q = float((word >> ((p.x % per) * a.bits)) & mask);                                                \
    const size_t g = row * (a.width / a.group) + p.x / a.group;                                                    \
    R[(size_t)p.y * a.width + p.x] = float(bfloat(float(S[g]) * q + float(B[g])));                                 \
  }
