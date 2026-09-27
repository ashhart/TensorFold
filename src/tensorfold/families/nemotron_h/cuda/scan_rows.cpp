#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void scan_rows_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, at::Tensor&, int, int, double, double);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == t, name, ": expected a contiguous CUDA tensor");
}

void scan_rows(const at::Tensor& proj, const at::Tensor& xc, at::Tensor state, const at::Tensor& a,
               const at::Tensor& dsk, const at::Tensor& dtb, at::Tensor y, int64_t dt_off, int64_t groups, double lo,
               double hi) {
    check(proj, at::kBFloat16, "projections");
    check(xc, at::kBFloat16, "conv outputs");
    check(state, at::kFloat, "state");
    check(a, at::kFloat, "A");
    check(dsk, at::kFloat, "D");
    check(dtb, at::kFloat, "dt bias");
    check(y, at::kBFloat16, "y");
    TORCH_CHECK(state.dim() == 3 && state.size(2) == 128 && state.size(1) % 32 == 0, "state (heads, 32k, 128)");
    TORCH_CHECK(state.size(0) % groups == 0 && proj.size(0) >= xc.size(0) && y.size(0) >= xc.size(0),
                "heads a multiple of groups; a projection and an output row a chunk row");
    c10::cuda::CUDAGuard guard(proj.device());
    scan_rows_cuda(proj, xc, state, a, dsk, dtb, y, static_cast<int>(dt_off), static_cast<int>(groups), lo, hi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scan_rows", &scan_rows); }
