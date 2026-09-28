#include <torch/extension.h>

// A plain fp16/bf16 linear, row-invariant by construction (one warp an output, fixed fp32 order); see b16.cu.
at::Tensor b16_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("b16_linear", &b16_linear, "plain fp16/bf16 linear (x, w [N, K], bias [N] or undefined)");
}
