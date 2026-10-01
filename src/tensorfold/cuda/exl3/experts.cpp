#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3x_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t);
void exl3x_prompt_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t,
                       int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_prompt_down_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_dequant_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_group_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                double, int64_t);
void exl3x_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);
void exl3x_combine_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_down_combine_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                             at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
             const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo, int64_t hi) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt, warps,
                       pf, lo, hi);
}

void prompt(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
            const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& items, const at::Tensor& counts,
            const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t E,
            int64_t seg, int64_t wps, int64_t items_max, int64_t cb, int64_t nt, int64_t widths) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(items, at::kInt, "items");
    check(counts, at::kInt, "counts");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(items.numel() >= 3 * items_max, "items too small");
    TORCH_CHECK(Z.numel() >= mats * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    TORCH_CHECK(items_max <= 65535, "too many items for one launch");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_prompt_cuda(X0, X1, TP0, TP1, B0, B1, items, counts, members, Z, mats, K, N, P, E, seg, wps, items_max, cb,
                      nt, widths);
}

void prompt_down(const at::Tensor& X, const at::Tensor& TP, const at::Tensor& B, const at::Tensor& svh,
                 const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor Y, int64_t K,
                 int64_t N, int64_t P, int64_t E, int64_t seg, int64_t wps, int64_t items_max, int64_t cb,
                 int64_t widths) {
    check(X, at::kHalf, "X");
    check(TP, at::kLong, "TP");
    check(B, at::kInt, "B");
    check(svh, at::kHalf, "svh");
    check(items, at::kInt, "items");
    check(counts, at::kInt, "counts");
    check(members, at::kInt, "members");
    TORCH_CHECK(Y.is_cuda() && (Y.scalar_type() == at::kFloat || Y.scalar_type() == at::kBFloat16), "Y: fp32 or bf16");
    TORCH_CHECK(Y.dim() == 2 && Y.size(1) == N && Y.stride(1) == 1 && Y.stride(0) == N && Y.size(0) >= P,
                "Y: contiguous [>= P, N]");
    TORCH_CHECK(items.numel() >= 3 * items_max && items_max <= 65535, "items");
    TORCH_CHECK(X.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X.device());
    exl3x_prompt_down_cuda(X, TP, B, svh, items, counts, members, Y, K, N, P, E, seg, wps, items_max, cb, widths);
}

void dequant(const at::Tensor& T, at::Tensor out, int64_t k2, int64_t cb) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3,
                "T: int16 [K/16, N/16, 8 * k2]");
    TORCH_CHECK(T.size(2) == 8 * k2, "trellis last dim must be 8 * k2");
    check(out, at::kHalf, "out");
    const int64_t K = T.size(0) * 16, N = T.size(1) * 16;
    TORCH_CHECK(out.numel() == K * N, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    exl3x_dequant_cuda(T, out, K, N, k2, cb);
}

void group(const at::Tensor& pick, at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots,
           int64_t E) {
    check(pick, at::kInt, "pick");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_cuda(pick, uids, ucount, members, R, slots, E);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots,
            int64_t E) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots, E);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, int64_t E, double limit, int64_t act_mode) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act_mode);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots, E);
}

void combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t D, int64_t slots) {
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    c10::cuda::CUDAGuard guard(y.device());
    exl3x_combine_cuda(y, wts, out, rows, D, slots);
}

void down_combine(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                  const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots,
                  int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine_cuda(Z, pick, svh_d, y, wts, out, rows, P, D, SK, slots, E);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped);
    m.def("prompt", &prompt);
    m.def("prompt_down", &prompt_down);
    m.def("dequant", &dequant);
    m.def("group", &group);
    m.def("rot_in", &rot_in);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
    m.def("combine", &combine);
    m.def("down_combine", &down_combine);
}
