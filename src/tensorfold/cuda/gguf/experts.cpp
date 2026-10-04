#include <torch/extension.h>

void gguf_pack_iq2_soa_cuda(const at::Tensor& raw, at::Tensor& out, int64_t nblk, int64_t dq_bytes);
void gguf_pack_q2_soa_cuda(const at::Tensor& raw, at::Tensor& out, int64_t nblk, int64_t dm_bytes, int64_t sc_bytes);
void gguf_iq2_soa_prefill_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                               int64_t dq_bytes, const at::Tensor& grid, const at::Tensor& items,
                               const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                               int64_t k, int64_t experts, int64_t pairs, int64_t max_items);
void gguf_iq2_soa_gate_up_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& gate,
                               const at::Tensor& up, int64_t dq_bytes, const at::Tensor& grid,
                               const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                               at::Tensor& out, int64_t n, int64_t k, int64_t experts, int64_t pairs,
                               int64_t max_items);
void gguf_q2_soa_prefill_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                              int64_t dm_bytes, int64_t sc_bytes, const at::Tensor& items, const at::Tensor& counts,
                              const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t experts,
                              int64_t pairs, int64_t max_items);

static void check_cuda(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, ": expected a CUDA tensor");
}

void pack_iq2_soa(const at::Tensor& raw, at::Tensor out, int64_t nblk, int64_t dq_bytes) {
  check_cuda(raw, "raw");
  check_cuda(out, "out");
  TORCH_CHECK(raw.scalar_type() == at::kByte && out.scalar_type() == at::kByte, "pack_iq2_soa: uint8");
  TORCH_CHECK(raw.is_contiguous() && out.is_contiguous(), "pack_iq2_soa: contiguous");
  TORCH_CHECK(nblk > 0 && dq_bytes >= nblk * 2 && (dq_bytes % 64) == 0, "pack_iq2_soa: geometry");
  TORCH_CHECK(raw.numel() == nblk * 66, "pack_iq2_soa: raw size");
  TORCH_CHECK(out.numel() == dq_bytes + nblk * 64, "pack_iq2_soa: out size");
  gguf_pack_iq2_soa_cuda(raw, out, nblk, dq_bytes);
}

void pack_q2_soa(const at::Tensor& raw, at::Tensor out, int64_t nblk, int64_t dm_bytes, int64_t sc_bytes) {
  check_cuda(raw, "raw");
  check_cuda(out, "out");
  TORCH_CHECK(raw.scalar_type() == at::kByte && out.scalar_type() == at::kByte, "pack_q2_soa: uint8");
  TORCH_CHECK(raw.is_contiguous() && out.is_contiguous(), "pack_q2_soa: contiguous");
  TORCH_CHECK(nblk > 0 && (dm_bytes % 64) == 0 && (sc_bytes % 64) == 0, "pack_q2_soa: geometry");
  TORCH_CHECK(raw.numel() == nblk * 84, "pack_q2_soa: raw size");
  TORCH_CHECK(out.numel() == dm_bytes + sc_bytes + nblk * 64, "pack_q2_soa: out size");
  gguf_pack_q2_soa_cuda(raw, out, nblk, dm_bytes, sc_bytes);
}

void iq2_soa_prefill(const at::Tensor& x, int64_t slots, const at::Tensor& w, int64_t dq_bytes,
                     const at::Tensor& grid, const at::Tensor& items, const at::Tensor& counts,
                     const at::Tensor& members, at::Tensor out, int64_t n, int64_t k, int64_t experts) {
  check_cuda(x, "x");
  check_cuda(w, "w");
  check_cuda(grid, "grid");
  check_cuda(out, "out");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "iq2 prefill: bf16");
  TORCH_CHECK(w.scalar_type() == at::kByte && grid.scalar_type() == at::kByte, "iq2 prefill: uint8 meta");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "iq2 prefill: x rank");
  const int64_t pairs = out.size(0) * (out.dim() == 3 ? out.size(1) : 1);
  const int64_t x_stride = x.stride(0);
  const int64_t max_items = items.size(0);
  auto y = out.dim() == 3 ? out.reshape({pairs, n}) : out;
  gguf_iq2_soa_prefill_cuda(x, x_stride, slots, w, dq_bytes, grid, items, counts, members, y, n, k, experts, pairs,
                            max_items);
}

void iq2_soa_gate_up(const at::Tensor& x, int64_t slots, const at::Tensor& gate, const at::Tensor& up,
                     int64_t dq_bytes, const at::Tensor& grid, const at::Tensor& items, const at::Tensor& counts,
                     const at::Tensor& members, at::Tensor out, int64_t n, int64_t k, int64_t experts) {
  check_cuda(x, "x");
  check_cuda(gate, "gate");
  check_cuda(up, "up");
  check_cuda(grid, "grid");
  check_cuda(out, "out");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "iq2 gate_up: bf16");
  TORCH_CHECK(gate.scalar_type() == at::kByte && up.scalar_type() == at::kByte, "iq2 gate_up: uint8 weights");
  const int64_t pairs = out.size(0) * (out.dim() == 3 ? out.size(1) : 1);
  const int64_t x_stride = x.stride(0);
  const int64_t max_items = items.size(0);
  auto y = out.dim() == 3 ? out.reshape({pairs, n}) : out;
  gguf_iq2_soa_gate_up_cuda(x, x_stride, slots, gate, up, dq_bytes, grid, items, counts, members, y, n, k, experts,
                            pairs, max_items);
}

void q2_soa_prefill(const at::Tensor& x, int64_t slots, const at::Tensor& w, int64_t dm_bytes, int64_t sc_bytes,
                    const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor out,
                    int64_t n, int64_t k, int64_t experts) {
  check_cuda(x, "x");
  check_cuda(w, "w");
  check_cuda(out, "out");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "q2 soa: bf16");
  TORCH_CHECK(w.scalar_type() == at::kByte, "q2 soa: uint8");
  const int64_t pairs = out.size(0) * (out.dim() == 3 ? out.size(1) : 1);
  const int64_t x_stride = x.stride(0);
  const int64_t max_items = items.size(0);
  auto y = out.dim() == 3 ? out.reshape({pairs, n}) : out;
  gguf_q2_soa_prefill_cuda(x, x_stride, slots, w, dm_bytes, sc_bytes, items, counts, members, y, n, k, experts, pairs,
                           max_items);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_iq2_soa", &pack_iq2_soa, "IQ2_XXS scale/code SoA pack");
  m.def("pack_q2_soa", &pack_q2_soa, "Q2_K dm/scale/code SoA pack");
  m.def("iq2_soa_prefill", &iq2_soa_prefill, "staged IQ2 SoA grouped prefill");
  m.def("iq2_soa_gate_up", &iq2_soa_gate_up, "fused IQ2 SoA gate+up+SwiGLU prefill");
  m.def("q2_soa_prefill", &q2_soa_prefill, "staged Q2_K SoA grouped prefill");
}
