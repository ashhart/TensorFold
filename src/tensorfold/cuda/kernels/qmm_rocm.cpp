#include <torch/extension.h>

void gemv_groups(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& words, const at::Tensor& scales,
                 const at::Tensor& biases, int n, at::Tensor& out, at::Tensor& part, bool wmma, int fill,
                 at::Tensor& counts);
int wmma_slices(int kg, int n, int fill);
int gemv_slices(int kg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_groups", &gemv_groups, "ROCm decode matmul on group-major 4-bit words");
    m.def("gemv_slices", &gemv_slices, "K slices of a weight with kg groups");
    m.def("wmma_slices", &wmma_slices, "WMMA K slices of a (kg groups, n outputs) weight");
}
