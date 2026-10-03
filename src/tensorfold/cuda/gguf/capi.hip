// C entry points over Gufo's exact quantized GEMM (vendor/, MIT). Every row takes the decode GEMV's arithmetic order
// whatever the row count, so a drafted row's bits equal its serial bits.
#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/models/qwen/hip/ops/gemm.hpp"

extern "C" int tf_gguf_linear(int type, const void* w, const float* x, float* y, std::size_t rows, std::size_t m,
                              std::size_t k, void* stream) {
  gufo::hip::LaunchBatchedQuantGEMMFp32(static_cast<gufo::core::GgmlType>(type), w, x, y, rows, m, k,
                                        static_cast<hipStream_t>(stream));
  return static_cast<int>(hipGetLastError());
}

// One row on Gufo's decode GEMV (its LaunchGEMV kHipQuantDirect route for every quantized format).
extern "C" int tf_gguf_gemv(int type, const void* w, const float* x, float* y, std::size_t m, std::size_t k,
                            void* stream) {
  gufo::hip::LaunchQ8KBlockGEMV(w, static_cast<gufo::core::GgmlType>(type), x, y, m, k,
                                static_cast<hipStream_t>(stream));
  return static_cast<int>(hipGetLastError());
}

// Rows on Gufo's bf16-activation quant-direct GEMM (no activation quantization; its DFlash route past 8 rows).
extern "C" int tf_gguf_linear_bf16(int type, const void* w, const void* x, float* y, std::size_t rows, std::size_t m,
                                   std::size_t k, void* stream) {
  gufo::hip::LaunchBatchedQuantGEMMBf16(static_cast<gufo::core::GgmlType>(type), w, x, y, rows, m, k,
                                        static_cast<hipStream_t>(stream));
  return static_cast<int>(hipGetLastError());
}

extern "C" int tf_gguf_dequant_bf16(int type, const void* w, void* out, std::size_t n, void* stream) {
  gufo::hip::LaunchDequantizeToBf16(static_cast<gufo::core::GgmlType>(type), w, static_cast<hip_bfloat16*>(out), n,
                                    static_cast<hipStream_t>(stream));
  return static_cast<int>(hipGetLastError());
}

// Prompt rows on Gufo's blocked WMMA W8A8 GEMM (its kHipPrefillQuantDirect route): activations quantized to tiled
// Q8_1 (bf16 or fp32 in), then the prequantized GEMM. Gufo picks the tile by row count (<=8, <96, >=96); the caller
// pads every call to >= 96 rows so one weight always takes one kernel, and a row's bits never depend on the count.
extern "C" int tf_gguf_prefill_supported(int type) {
  const auto t = static_cast<gufo::core::GgmlType>(type);
  return t == gufo::core::GgmlType::kQ8_0 || gufo::hip::detail::IsNativeWmmaQuant(t);
}

// Tiled payload, per-token sum sidecar, then slack: Gufo's Q8_0 W8A8 kernel (prefill_quant_gemm.hip:500-503, taken
// when its wave64 route declines, e.g. m < 1024) stages all eight 16-token tiles of its 128-token block with no bound
// on the activation's tile count, so the last block reads up to 7 tiles past the payload (its 16/32/64-token
// small-row tiles read fewer). Gufo's engine sizes its
// scratch for the largest batch; an exact-size buffer faulted here. The tokens read are never stored.
extern "C" std::size_t tf_gguf_q8_1_bytes(std::size_t rows, std::size_t k) {
  constexpr std::size_t kSlackTiles = 8, kTileBytes = 576;    // a 128-token block, kQ8ActTileBytes
  return gufo::hip::QuantizedActivationBytes(rows, k) + kSlackTiles * (k / 32) * kTileBytes;
}

extern "C" int tf_gguf_prefill_linear(int type, const void* w, const void* x, int x_bf16, void* q8, float* y,
                                      std::size_t rows, std::size_t m, std::size_t k, void* stream) {
  if (!tf_gguf_prefill_supported(type)) return -1;
  auto s = static_cast<hipStream_t>(stream);
  if (x_bf16) {
    gufo::hip::LaunchQuantizeActivationQ8_1(x, q8, rows, k, s);
  } else {
    gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(static_cast<const float*>(x), q8, rows, k, s);
  }
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(static_cast<gufo::core::GgmlType>(type), w, q8, y, rows, m, k, s);
  return static_cast<int>(hipGetLastError());
}
