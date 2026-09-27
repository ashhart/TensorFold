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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &plan, "group a layer's (row, slot) pairs by expert into items of at most tile pairs");
  m.def("run", &run, "grouped 4-bit expert matmul, decode form (epilogue 0: fp32, 1: relu^2, 2: SwiGLU)");
  m.def("prefill", &prefill, "grouped 4-bit expert matmul, prefill form (3: bf16 out)");
}
