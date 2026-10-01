#include <torch/extension.h>

// A plain fp16/bf16 linear, row-invariant by construction (one warp an output, fixed fp32 order); see b16.cu.
at::Tensor b16_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias);
at::Tensor b16_prompt(const at::Tensor& x, const at::Tensor& w, int64_t bm);
std::vector<at::Tensor> b16_linear_pair(const at::Tensor& x, const at::Tensor& w0, const at::Tensor& w1);
std::vector<at::Tensor> b16_prompt_pair(const at::Tensor& x, const at::Tensor& w0, const at::Tensor& w1, int64_t bm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("b16_linear", &b16_linear, "plain fp16/bf16 linear (x, w [N, K], bias [N] or undefined)");
    m.def("b16_prompt", &b16_prompt, "prompt rows on the bf16 mma: one fp32 chain over K a row");
    m.def("b16_linear_pair", &b16_linear_pair, "two weights of the same rows in one launch, each b16_linear's bits");
    m.def("b16_prompt_pair", &b16_prompt_pair, "two weights of the same prompt rows in one launch");
}
