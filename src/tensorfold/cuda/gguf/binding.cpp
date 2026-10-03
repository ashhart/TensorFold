// torch entry points for libtfgguf (capi.hip): quantized rows in, fp32 rows out, on the caller's stream handle.
#include <torch/extension.h>

extern "C" int tf_gguf_linear(int, const void*, const float*, float*, size_t, size_t, size_t, void*);
extern "C" int tf_gguf_dequant_bf16(int, const void*, void*, size_t, void*);
extern "C" int tf_gguf_linear_bf16(int, const void*, const void*, float*, size_t, size_t, size_t, void*);
extern "C" int tf_gguf_gemv(int, const void*, const float*, float*, size_t, size_t, void*);
extern "C" int tf_gguf_prefill_supported(int);
extern "C" size_t tf_gguf_q8_1_bytes(size_t, size_t);
extern "C" int tf_gguf_prefill_linear(int, const void*, const void*, int, void*, float*, size_t, size_t, size_t, void*);

torch::Tensor linear(torch::Tensor x, torch::Tensor w, int64_t type, int64_t m, int64_t stream_handle) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2 && x.is_contiguous(), "x: fp32 (rows, K)");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w: packed uint8 blocks");
  auto y = torch::empty({x.size(0), m}, x.options());
  if (x.size(0) == 0) return y;
  auto stream = reinterpret_cast<void*>(stream_handle);
  int err = tf_gguf_linear(static_cast<int>(type), w.data_ptr(), x.data_ptr<float>(), y.data_ptr<float>(), x.size(0),
                           m, x.size(1), stream);
  TORCH_CHECK(err == 0, "tf_gguf_linear: HIP error ", err);
  return y;
}

torch::Tensor linear_bf16(torch::Tensor x, torch::Tensor w, int64_t type, int64_t m, int64_t stream_handle) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 (rows, K)");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w: packed uint8 blocks");
  auto y = torch::empty({x.size(0), m}, x.options().dtype(torch::kFloat32));
  if (x.size(0) == 0) return y;
  int err = tf_gguf_linear_bf16(static_cast<int>(type), w.data_ptr(), x.data_ptr(), y.data_ptr<float>(), x.size(0), m,
                                x.size(1), reinterpret_cast<void*>(stream_handle));
  TORCH_CHECK(err == 0, "tf_gguf_linear_bf16: HIP error ", err);
  return y;
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor w, int64_t type, int64_t m, int64_t stream_handle) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2 && x.size(0) == 1 && x.is_contiguous(),
              "x: fp32 (1, K)");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w: packed uint8 blocks");
  auto y = torch::empty({1, m}, x.options());
  int err = tf_gguf_gemv(static_cast<int>(type), w.data_ptr(), x.data_ptr<float>(), y.data_ptr<float>(), m, x.size(1),
                         reinterpret_cast<void*>(stream_handle));
  TORCH_CHECK(err == 0, "tf_gguf_gemv: HIP error ", err);
  return y;
}

torch::Tensor dequant_bf16(torch::Tensor w, int64_t type, int64_t n, int64_t stream_handle) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w: packed uint8 blocks");
  auto out = torch::empty({n}, w.options().dtype(torch::kBFloat16));
  auto stream = reinterpret_cast<void*>(stream_handle);
  int err = tf_gguf_dequant_bf16(static_cast<int>(type), w.data_ptr(), out.data_ptr(), n, stream);
  TORCH_CHECK(err == 0, "tf_gguf_dequant_bf16: HIP error ", err);
  return out;
}

// ``pad``: zero rows up to this count (96: Gufo's throughput tile, one kernel per weight at any row count; 0: none).
torch::Tensor prefill_linear(torch::Tensor x, torch::Tensor w, int64_t type, int64_t m, int64_t pad,
                             int64_t stream_handle) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x: contiguous (rows, K)");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32, "x: bf16 or fp32");
  TORCH_CHECK(x.size(1) % 32 == 0, "x: K a multiple of 32");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.is_contiguous(), "w: packed uint8 blocks");
  TORCH_CHECK(tf_gguf_prefill_supported(static_cast<int>(type)), "prefill_linear: no WMMA kernel for type ", type);
  const int64_t rows = x.size(0);
  if (rows == 0) return torch::empty({0, m}, x.options().dtype(torch::kFloat32));
  auto xp = x;
  if (rows < pad) {                                            // zero rows quantize to zero scales: no effect on others
    xp = torch::zeros({pad, x.size(1)}, x.options());
    xp.narrow(0, 0, rows).copy_(x);
  }
  const int64_t padded = xp.size(0), k = xp.size(1);
  auto q8 = torch::empty({static_cast<int64_t>(tf_gguf_q8_1_bytes(padded, k))}, w.options());
  auto y = torch::empty({padded, m}, x.options().dtype(torch::kFloat32));
  auto stream = reinterpret_cast<void*>(stream_handle);
  int err = tf_gguf_prefill_linear(static_cast<int>(type), w.data_ptr(), xp.data_ptr(),
                                   xp.scalar_type() == torch::kBFloat16, q8.data_ptr(), y.data_ptr<float>(), padded, m,
                                   k, stream);
  TORCH_CHECK(err == 0, "tf_gguf_prefill_linear: HIP error ", err);
  return padded == rows ? y : y.narrow(0, 0, rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear);
  m.def("gemv", &gemv);
  m.def("linear_bf16", &linear_bf16);
  m.def("dequant_bf16", &dequant_bf16);
  m.def("prefill_linear", &prefill_linear);
}
