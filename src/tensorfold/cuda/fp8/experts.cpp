#include <torch/extension.h>

void fp8_experts_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                      const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                      const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n, double limit,
                      int64_t skip, int64_t max_units);

void fp8_experts_prompt_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                             const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                             const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units);

// epi 2: out [pairs, n] bf16 SwiGLU of gate and up; 0: fp32 down; 3: bf16 down. w: [E, n/32, K/32, M, 32, 2] uint4.
void experts(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
             const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
             const at::Tensor& members, at::Tensor out, int64_t n, double limit, int64_t skip, int64_t max_units) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(-1) == 1, "x: bf16 rows");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt, "w: int32 blocks");
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() && scale.scalar_type() == at::kFloat, "scale: fp32");
    TORCH_CHECK(kg % 4 == 0 && (nb * 32) % 128 == 0, "K and n in 128 x 128 scale blocks");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out: contiguous");
    fp8_experts_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit, skip,
                     max_units);
}

// The same on prompt items of up to 64 pairs, each block staged once for the item (the same bits).
void prompt(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
            const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
            const at::Tensor& members, at::Tensor out, int64_t n, double limit, int64_t skip, int64_t max_units) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(-1) == 1, "x: bf16 rows");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt, "w: int32 blocks");
    TORCH_CHECK(kg % 4 == 0 && (nb * 32) % 128 == 0, "K and n in 128 x 128 scale blocks");
    fp8_experts_prompt_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit, skip,
                            max_units);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("experts", &experts);
    m.def("prompt", &prompt);
}
