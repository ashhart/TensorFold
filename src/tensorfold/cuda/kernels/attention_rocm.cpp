#include <torch/extension.h>

bool attention_supported(int heads, int kv_heads, int dim);
void prompt_attention(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, at::Tensor& out,
                      int p0, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supported", &attention_supported, "whether ROCm's prompt attention takes these heads and head size");
    m.def("attention", &prompt_attention, "ROCm prompt attention on WMMA, head size 256");
}
