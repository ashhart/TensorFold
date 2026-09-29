#include <torch/extension.h>

void experts_plan_cuda(const at::Tensor& picks, int64_t pairs, int64_t experts, int64_t tile, at::Tensor& members,
                       at::Tensor& items, at::Tensor& counts, at::Tensor& rank, at::Tensor& hist);
void experts_run_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                      const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
                      const at::Tensor& members, at::Tensor& out, int64_t n, double limit, int64_t max_units);
void experts_prefill_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                          const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items,
                          const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                          double limit, int64_t max_items);

void experts_pack_cuda(const at::Tensor& words, const at::Tensor& scales, const at::Tensor& biases, int64_t gs,
                       at::Tensor& out);

static void check(const at::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, ": expected a CUDA tensor of the right dtype");
}

// x rows of K inputs, weights, and out rows of n columns (fp32 for epilogue 0, else bf16)
static void check_call(const at::Tensor& x, int64_t k, const at::Tensor& w, const at::Tensor& out, int64_t n,
                       int64_t nb, int64_t epi) {
  check(x, "x", at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.size(1) == k,
              "x: rows of K = groups * group size inputs, 16-byte aligned");
  check(w, "w", at::kInt);
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(out.is_contiguous() && out.size(-1) == n, "out: contiguous rows of n columns");
  TORCH_CHECK(out.scalar_type() == (epi == 0 ? at::kFloat : at::kBFloat16), "out has the wrong dtype");
  TORCH_CHECK(n == nb * 32, "n must be the column blocks times 32");
}

void plan(const at::Tensor& picks, int64_t pairs, int64_t experts, int64_t tile, at::Tensor members,
          at::Tensor items, at::Tensor counts, at::Tensor rank, at::Tensor hist) {
  check(picks, "picks", at::kInt);
  TORCH_CHECK(picks.is_contiguous() && picks.numel() >= pairs, "picks: contiguous, one id a pair");
  for (const auto* t : {&members, &items, &counts, &rank, &hist}) check(*t, "plan buffer", at::kInt);
  TORCH_CHECK(members.numel() >= pairs && counts.numel() >= 2, "plan buffers too small");
  TORCH_CHECK(tile == 16 || tile == 64, "items hold 16 pairs (decode) or 64 (prefill)");
  experts_plan_cuda(picks, pairs, experts, tile, members, items, counts, rank, hist);
}

void run(int64_t gs, int64_t epi, const at::Tensor& x, int64_t slots, const at::Tensor& w, int64_t kg, int64_t nb,
         const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor out, int64_t n,
         double limit, int64_t max_units) {
  check_call(x, kg * gs, w, out, n, nb, epi);
  experts_run_cuda(gs, epi, x, x.stride(0), slots, w, kg, nb, items, counts, members, out, n, limit, max_units);
}

void prefill(int64_t gs, int64_t epi, const at::Tensor& x, int64_t slots, const at::Tensor& w, int64_t kg,
             int64_t nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor out,
             int64_t n, double limit, int64_t max_items) {
  check_call(x, kg * gs, w, out, n, nb, epi);
  experts_prefill_cuda(gs, epi, x, x.stride(0), slots, w, kg, nb, items, counts, members, out, n, limit, max_items);
}

// MLX words [E, N, K/8] (32-bit), scales and biases [E, N, K/gs] (2-byte) -> int32 blocks [E, N/32, K/gs, 32 * gs / 8 + 32]
void pack(const at::Tensor& words, const at::Tensor& scales, const at::Tensor& biases, int64_t gs, at::Tensor out) {
  TORCH_CHECK(gs == 32 || gs == 64, "pack: groups of 32 or 64 inputs");
  TORCH_CHECK(words.is_cuda() && words.element_size() == 4 && words.dim() == 3 && words.is_contiguous(),
              "pack: words must be contiguous 32-bit CUDA [E, N, K/8]");
  const int64_t e = words.size(0), n = words.size(1), k = words.size(2) * 8, kg = k / gs;
  TORCH_CHECK(n % 32 == 0 && k % gs == 0, "pack: N must be a multiple of 32 and K of the group size");
  for (const auto* t : {&scales, &biases})
    TORCH_CHECK(t->is_cuda() && t->element_size() == 2 && t->is_contiguous() && t->dim() == 3 && t->size(0) == e &&
                    t->size(1) == n && t->size(2) == kg,
                "pack: scales and biases must be contiguous 2-byte CUDA [E, N, K/gs]");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&scales, &biases, &out})
    TORCH_CHECK(t->device() == words.device(), "pack: words, scales, biases and out must be on one GPU");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.is_contiguous() &&
                  out.numel() == e * (n / 32) * kg * (32 * gs / 8 + 32),
              "pack: out must be contiguous int32 [E, N/32, K/gs, block]");
  TORCH_CHECK(e <= 65535 && n / 32 <= 65535, "pack: at most 65535 experts and 65535 column blocks");
  experts_pack_cuda(words, scales, biases, gs, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "repack MLX 4-bit experts into the grouped kernels' blocks, in one pass");
  m.def("plan", &plan, "group a layer's (row, slot) pairs by expert into items of at most tile pairs");
  m.def("run", &run, "grouped 4-bit expert matmul, decode form (epilogue 0: fp32, 1: relu^2, 2: SwiGLU)");
  m.def("prefill", &prefill, "grouped 4-bit expert matmul, prefill form (3: bf16 out)");
}
