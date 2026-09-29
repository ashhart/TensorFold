#include <torch/extension.h>

bool tree_supported(int heads, int kv_heads, int dim);
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale);
void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supported", &tree_supported, "whether ROCm's tree attention takes these heads and head size");
    m.def("shared", &tree_shared, "committed key chunks' partials, each chunk read once for every pair");
    m.def("tail", &tree_tail, "tail chunks' partials: the last committed keys and each row's path");
}
