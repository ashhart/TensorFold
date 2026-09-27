#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t, int64_t, int64_t);
void exl3_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                      at::Tensor&, int64_t, int64_t, int64_t);
void exl3_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, double);
void exl3_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                             int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& T0, const at::Tensor& T1,
             const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor Z, int64_t mats,
             int64_t K, int64_t N, int64_t P, int64_t SK, int64_t max_items, int64_t nt, int64_t warps) {
    const int64_t E = T0.size(0);
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(T0, at::kInt, "T0");
    check(T1, at::kInt, "T1");
    check(items, at::kInt, "items");
    check(counts, at::kInt, "counts");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3_grouped_cuda(X0, X1, T0, T1, items, counts, members, Z, mats, K, N, P, SK, max_items, nt, warps, E);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x: bf16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, double limit) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, limit);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped);
    m.def("rot_in", &rot_in);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
}
