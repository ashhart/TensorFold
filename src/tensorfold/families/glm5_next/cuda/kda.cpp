#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void kda_chain_cuda(const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, const at::Tensor&, int64_t,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, double, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&);
void kda_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     int64_t, at::Tensor&);
void kda_replay_layers_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, at::Tensor&);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t, name, ": expected a CUDA tensor of the right dtype");
}

void chain(const at::Tensor& P, int64_t p_stride, int64_t b_off, const at::Tensor& A, int64_t a_stride,
           const at::Tensor& G, int64_t g_stride, const at::Tensor& cs, const at::Tensor& cw,
           const at::Tensor& state_in, const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& norm_w,
           double eps, double lower, int64_t rows, at::Tensor out, at::Tensor state_out, at::Tensor k_save,
           at::Tensor v_save, at::Tensor g_save, at::Tensor b_save) {
    check(P, at::kBFloat16, "P");
    check(A, at::kBFloat16, "A");
    check(G, at::kBFloat16, "G");
    check(cs, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(state_in, at::kFloat, "state");
    check(a_log, at::kFloat, "A_log");
    check(dt_bias, at::kFloat, "dt_bias");
    check(norm_w, at::kBFloat16, "norm");
    check(out, at::kBFloat16, "out");
    TORCH_CHECK(cs.is_contiguous() && cw.is_contiguous() && state_in.is_contiguous() && out.is_contiguous(),
                "conv state, conv weight, state and out must be contiguous");
    const int64_t H = a_log.numel();
    TORCH_CHECK(dt_bias.numel() == H * 128 && norm_w.numel() == 128, "dt_bias [H*128], norm [128]");
    TORCH_CHECK(cw.size(0) == 3 * H * 128 && cw.size(1) == 4, "conv weight [3*H*128, 4]");
    TORCH_CHECK(state_in.numel() == H * 128 * 128, "state must be [H, 128, 128]");
    TORCH_CHECK(rows >= 1 && P.size(0) >= rows && out.size(0) >= rows, "rows");
    c10::cuda::CUDAGuard guard(P.device());
    kda_chain_cuda(P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps,
                   lower, rows, out, state_out, k_save, v_save, g_save, b_save);
}

void replay(const at::Tensor& state_in, const at::Tensor& k_save, const at::Tensor& v_save, const at::Tensor& g_save,
            const at::Tensor& b_save, int64_t rows, at::Tensor state_out) {
    check(state_in, at::kFloat, "state");
    check(state_out, at::kFloat, "state out");
    TORCH_CHECK(state_in.is_contiguous() && state_out.is_contiguous(), "states must be contiguous");
    c10::cuda::CUDAGuard guard(state_in.device());
    kda_replay_cuda(state_in, k_save, v_save, g_save, b_save, rows, state_out);
}

void replay_layers(const at::Tensor& state_in, int64_t state_stride, const at::Tensor& k_save, const at::Tensor& v_save,
                   const at::Tensor& g_save, const at::Tensor& b_save, int64_t kv_stride, int64_t b_stride,
                   int64_t layers, int64_t heads, int64_t rows, at::Tensor state_out) {
    check(state_in, at::kFloat, "state");
    check(state_out, at::kFloat, "state out");
    c10::cuda::CUDAGuard guard(state_in.device());
    kda_replay_layers_cuda(state_in, state_stride, k_save, v_save, g_save, b_save, kv_stride, b_stride, layers, heads,
                           rows, state_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain);
    m.def("replay", &replay);
    m.def("replay_layers", &replay_layers);
}
