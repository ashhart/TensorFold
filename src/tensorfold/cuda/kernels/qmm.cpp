#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void qmm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
              at::Tensor&, const at::Tensor&, int, int, int, int, bool, bool);
void qmm_prefill_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int,
                      int, bool, int);
void qmm_prefill8w_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int,
                        int, bool, int, bool);
void qmm_prefill8_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, at::Tensor&, int, int, bool, int);

// x (M, K) bf16 times a packed 4-bit weight; ``reduce`` false leaves the K slices unadded in ``part`` (SK, M, n).
void qmm(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
         const at::Tensor& biases, at::Tensor& out, const c10::optional<at::Tensor>& part, int64_t n, int64_t sk,
         int64_t gs, int64_t bm, bool f32, bool reduce) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64, "row tile 16, 32 or 64");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    const int64_t m = x.size(0), k = x.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0 && sk >= 1 && kg % sk == 0, "K splits into whole groups a slice");
    TORCH_CHECK(xs.is_cuda() && xs.is_contiguous() && xs.scalar_type() == at::kFloat && xs.size(0) == m &&
                xs.size(1) == kg, "xs: (M, K / gs) fp32");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.stride(1) == 1 && biases.stride(1) == 1 && scales.stride(0) >= npad &&
                biases.stride(0) == scales.stride(0) && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16, rows may be strided");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x.device());
    at::Tensor p;
    if (sk > 1 && (sk > 8 || !reduce)) {
        p = part.has_value() ? *part : at::empty({sk, m, n}, x.options().dtype(at::kFloat));
        TORCH_CHECK(p.is_cuda() && p.is_contiguous() && p.scalar_type() == at::kFloat && p.numel() >= sk * m * n,
                    "part: at least (SK, M, n) fp32");
    }
    qmm_cuda(x, xs, w, scales, biases, out, p, static_cast<int>(n), static_cast<int>(sk), static_cast<int>(gs),
             static_cast<int>(bm), f32, reduce);
}

// Prefill: x (M, K) bf16 times a packed 4-bit weight with each weight rounded once to bf16, one fp32 chain over K.
void qmm_prefill(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                 at::Tensor& out, int64_t n, int64_t gs, bool f32, int64_t tile) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(tile >= 0 && tile <= 11, "tile 0-11");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    const int64_t m = x.size(0), k = x.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0, "K splits into whole groups");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x.device());
    qmm_prefill_cuda(x, w, scales, biases, out, static_cast<int>(n), static_cast<int>(gs), f32, static_cast<int>(tile));
}

// Prefill in FP8: x8, xs and a from ``quantize_rows`` (e4m3 bytes, group sums / a, row scales) times a 4-bit weight.
void qmm_prefill8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
                  const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int64_t n, int64_t gs, bool f32,
                  int64_t tile) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(tile >= 0 && tile <= 2, "tile 0-2");
    TORCH_CHECK(x8.is_cuda() && x8.scalar_type() == at::kByte && x8.dim() == 2 && x8.is_contiguous() &&
                x8.size(0) >= 1, "x8: (M, K) e4m3 bytes, contiguous");
    const int64_t m = x8.size(0), k = x8.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0 && k % 32 == 0, "K splits into whole groups of 32 or 64 inputs");
    TORCH_CHECK(xs.is_cuda() && xs.is_contiguous() && xs.scalar_type() == at::kBFloat16 && xs.size(0) == m &&
                xs.size(1) == kg, "xs: (M, K / gs) bf16");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == at::kFloat && a.numel() == m, "a: (M,) fp32");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x8.device());
    qmm_prefill8_cuda(x8, xs, a, w, scales, biases, out, static_cast<int>(n), static_cast<int>(gs), f32,
                      static_cast<int>(tile));
}

// Prefill in FP8 over e4m3 weight bytes (fragment order): x8 and a from ``quantize_rows``, a bf16 scale per (gs, column).
void qmm_prefill8w(const at::Tensor& x8, const at::Tensor& a, const at::Tensor& w8, const at::Tensor& scales,
                   at::Tensor& out, int64_t n, int64_t gs, bool f32, int64_t tile, bool l64) {
    TORCH_CHECK(tile >= 0 && tile <= 2, "tile 0-2");
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64 inputs");
    TORCH_CHECK(!l64 || gs == 32, "the 64-input byte order serves groups of 32");
    TORCH_CHECK(x8.is_cuda() && x8.scalar_type() == at::kByte && x8.dim() == 2 && x8.is_contiguous() &&
                x8.size(0) >= 1, "x8: (M, K) e4m3 bytes, contiguous");
    const int64_t m = x8.size(0), k = x8.size(1);
    TORCH_CHECK(k % 64 == 0 && n % 128 == 0, "K a multiple of 64 and n of 128");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == at::kFloat && a.numel() == m, "a: (M,) fp32");
    TORCH_CHECK(w8.is_cuda() && w8.is_contiguous() && w8.scalar_type() == at::kByte && w8.numel() == n * k,
                "staged weight: n * K e4m3 bytes");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                scales.size(0) == k / gs && scales.size(1) == n, "scales: (K / gs, n) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x8.device());
    qmm_prefill8w_cuda(x8, a, w8, scales, out, static_cast<int>(n), static_cast<int>(gs), f32, static_cast<int>(tile),
                       l64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmm", &qmm);
    m.def("qmm_prefill", &qmm_prefill);
    m.def("qmm_prefill8", &qmm_prefill8);
    m.def("qmm_prefill8w", &qmm_prefill8w);
}
