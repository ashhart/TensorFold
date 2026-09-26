#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gdn_chain_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void gdn_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     int64_t, at::Tensor&);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == t, name, ": expected a contiguous CUDA tensor");
}

void chain(const at::Tensor& P, const at::Tensor& cs, const at::Tensor& cw, const at::Tensor& state_in,
           const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& norm_w, double eps, int64_t rows,
           at::Tensor out, at::Tensor xs, at::Tensor state_out, at::Tensor k_save, at::Tensor v_save,
           at::Tensor g_save, at::Tensor b_save) {
    check(P, at::kBFloat16, "P");
    check(cs, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(state_in, at::kFloat, "state");
    check(a_log, at::kFloat, "A_log");
    check(dt_bias, at::kFloat, "dt_bias");
    check(norm_w, at::kBFloat16, "norm");
    check(out, at::kBFloat16, "out");
    check(xs, at::kFloat, "group sums");
    const int64_t nv = a_log.numel(), nk = nv / 3;
    TORCH_CHECK(nv == 48 || nv == 24, "48 or 24 value heads");
    TORCH_CHECK(P.size(0) >= rows && P.size(1) == 2 * nk * 128 + 2 * nv * 128 + 2 * nv, "P width");
    TORCH_CHECK(state_in.numel() == nv * 128 * 128, "state must be [nv, 128, 128]");
    c10::cuda::CUDAGuard guard(P.device());
    gdn_chain_cuda(P, cs, cw, state_in, a_log, dt_bias, norm_w, eps, rows, out, xs, state_out, k_save, v_save,
                   g_save, b_save);
}

void replay(const at::Tensor& state_in, const at::Tensor& k_save, const at::Tensor& v_save, const at::Tensor& g_save,
            const at::Tensor& b_save, int64_t rows, at::Tensor state_out) {
    check(state_in, at::kFloat, "state");
    check(state_out, at::kFloat, "state out");
    c10::cuda::CUDAGuard guard(state_in.device());
    gdn_replay_cuda(state_in, k_save, v_save, g_save, b_save, rows, state_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain);
    m.def("replay", &replay);
}
