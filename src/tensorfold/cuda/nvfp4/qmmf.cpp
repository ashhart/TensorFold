#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void qmmf_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double, at::Tensor&, const at::Tensor&, int64_t,
               int64_t, int64_t, int64_t, int64_t, int64_t, bool);
void stage_fp4_cuda(const at::Tensor&, const at::Tensor&, double, at::Tensor&, at::Tensor&, int64_t, int64_t);
void nvfp4_experts_cuda(int64_t, const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&, int64_t,
                        int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, double,
                        int64_t, int64_t);

// out (M, N) = x (M, K) bf16 @ W: mode 0 NVFP4 (tiled words, e4m3 block scales [npad/64, K/64, 64, 4]), 1 FP8
// (fragment-order bytes), 2 MXFP8 (those with e8m0 scales [npad/64, K/64, 64, 2]); ``scale`` the per-tensor factor.
void qmmf(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& bs, double scale, at::Tensor out,
          const c10::optional<at::Tensor>& part, int64_t mode, int64_t n, int64_t sk, int64_t npad, int64_t bm,
          bool f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x: (M, K) bf16");
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode 0-2");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % 64 == 0 && (k / 64) % sk == 0, "K in whole groups of 64, split evenly");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
                "weight bytes do not match n and K");
    TORCH_CHECK(mode == 1 || (bs.has_value() && bs->is_contiguous() &&
                              bs->numel() == (k / 64) * npad * (mode == 0 ? 4 : 2)), "block scales [npad/64, K/64, 64, 4|2]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    TORCH_CHECK(sk == 1 || (sk <= 8) || (part.has_value() && part->numel() >= sk * m * n), "part: (SK, M, n) fp32");
    c10::cuda::CUDAGuard guard(x.device());
    qmmf_cuda(x, w, bs.has_value() ? *bs : at::Tensor(), scale, out, part.has_value() ? *part : at::Tensor(), mode, n, k,
              sk, npad, bm, f32);
}

// NVFP4 tiled words and block scales -> e4m3 bytes in ``qmm_prefill8w``'s order and bf16 scales [K/64, npad].
void stage_fp4(const at::Tensor& words, const at::Tensor& bs, double global, at::Tensor w8, at::Tensor scales) {
    const int64_t kg = scales.size(0), npad = scales.size(1);
    TORCH_CHECK(words.is_cuda() && words.is_contiguous() && words.numel() * words.element_size() == npad * kg * 32,
                "words: NVFP4 tiled words");
    TORCH_CHECK(bs.is_contiguous() && bs.numel() == kg * npad * 4, "bs [npad/64, K/64, 64, 4]");
    TORCH_CHECK(w8.is_contiguous() && w8.numel() == npad * kg * 64, "w8: npad * K bytes");
    TORCH_CHECK(scales.is_contiguous() && scales.scalar_type() == at::kBFloat16, "scales [K/64, npad] bf16");
    c10::cuda::CUDAGuard guard(words.device());
    stage_fp4_cuda(words, bs, global, w8, scales, kg, npad);
}

// Grouped experts on a plan (``tensorfold.cuda.experts``): blocks [E, N/32, K/32, M, 144] int32, scales [E, M] fp32.
void experts(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
             const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
             const at::Tensor& members, at::Tensor out, int64_t n, double limit, int64_t skip, int64_t max_units) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x_stride % 4 == 0,
                "x: bf16 rows, 8-byte aligned");
    TORCH_CHECK(w.is_contiguous() && w.scalar_type() == at::kInt && w.size(-1) == 144, "w: [E, N/32, K/32, M, 144]");
    TORCH_CHECK(scale.is_contiguous() && scale.scalar_type() == at::kFloat && scale.size(0) == w.size(0),
                "scale: [E, M] fp32");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == (epi == 0 ? at::kFloat : at::kBFloat16), "out dtype");
    nvfp4_experts_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit, skip,
                       max_units);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmmf", &qmmf);
    m.def("stage_fp4", &stage_fp4);
    m.def("experts", &experts);
}
