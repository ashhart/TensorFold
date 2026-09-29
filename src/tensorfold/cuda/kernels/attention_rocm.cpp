#include <torch/extension.h>

bool attention_supported(int heads, int kv_heads, int dim);
void prompt_attention(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, at::Tensor& out,
                      int p0, double scale, int rb, bool pipe);
bool tree_supported(int heads, int kv_heads, int dim);
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale);
void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supported", &attention_supported, "whether ROCm's prompt attention takes these heads and head size");
    m.def("attention", &prompt_attention, "ROCm prompt attention on WMMA, head size 256");
    m.def("tree_supported", &tree_supported, "whether ROCm's tree attention takes these heads and head size");
    m.def("shared", &tree_shared, "committed key chunks' partials, each chunk read once for every pair");
    m.def("tail", &tree_tail, "tail chunks' partials: the last committed keys and each row's path");
}
