// Binding of the two-rank all-gather over RoCE (gather.cu); protocol after b12x RoCEnante via MiaAI-Lab patch 0006,
// Apache-2.0; see THIRD_PARTY_NOTICES.md.
#include <torch/extension.h>

void rdma_gather(const at::Tensor& in, at::Tensor& out, int64_t region, int64_t flag_off, int64_t send_off,
                 int64_t recv_off, int64_t slot_bytes, at::Tensor& state, int64_t spin, int64_t rank);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gather", &rdma_gather, "two-rank all-gather over RoCE"); }
