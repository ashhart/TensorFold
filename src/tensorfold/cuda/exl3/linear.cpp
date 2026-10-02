#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_rot_in_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&);
void exl3_linear_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                      const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&, at::Tensor&,
                      int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3_unpack_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3_unpack_fold_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3_unpack_fold2_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3_fdirect_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                       const c10::optional<at::Tensor>&, at::Tensor&, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

static void check_io(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat),
                name, ": expected a contiguous 2-d fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

// xh [M, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)); x [M, K] fp16, bf16 or fp32.
void rot_in(const at::Tensor& x, const at::Tensor& suh, at::Tensor xh) {
    check_io(x, "x");
    check(suh, at::kHalf, "suh");
    check(xh, at::kHalf, "xh");
    TORCH_CHECK(x.size(1) % 128 == 0 && suh.numel() == x.size(1) && xh.sizes() == x.sizes(),
                "x and xh must be [M, K], K a multiple of 128, suh [K]");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, suh, xh);
}

// y [M, N] = (xh @ W_q) @ H * svh + bias; Z [SK, M, N] fp32 when SK > 1; counters int32 [8 * N / 128], left zero.
// mode picks the kernel for 17-128 rows (0: linear_kernel; 6: the mid-M kernels; 7: linear_wc for 4- and 6-bit mul1,
// else 6); every mode gives the same bits.
void linear(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
            const c10::optional<at::Tensor>& bias, at::Tensor y, const c10::optional<at::Tensor>& Z,
            at::Tensor counters, int64_t K2, int64_t cb, int64_t SK, int64_t WK, int64_t mode) {
    check(xh, at::kHalf, "xh");
    check_io(y, "y");
    check(svh, at::kHalf, "svh");
    check(T, at::kInt, "T");
    check(counters, at::kInt, "counters");
    const int64_t M = xh.size(0), K = xh.size(1), N = y.size(1);
    TORCH_CHECK(xh.dim() == 2 && y.size(0) == M && M >= 1 && M <= 128, "xh and y must have the same 1 to 128 rows");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    TORCH_CHECK(svh.numel() == N, "svh must have N elements");
    TORCH_CHECK(T.numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "T must be 16-byte aligned");
    TORCH_CHECK(counters.numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
    if (bias) check(*bias, at::kHalf, "bias");
    if (SK > 1) {
        TORCH_CHECK(Z.has_value(), "Z is needed with more than one split");
        check(*Z, at::kFloat, "Z");
        TORCH_CHECK(Z->numel() >= SK * M * N, "Z too small");
    }
    c10::cuda::CUDAGuard guard(xh.device());
    exl3_linear_cuda(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, mode);
}

// W [K, N] fp16 = W_q, the trellis tiles decoded; tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb words.
void unpack(const at::Tensor& T, at::Tensor W, int64_t stride_k, int64_t stride_nb, int64_t K2, int64_t cb) {
    check(T, at::kInt, "T");
    check(W, at::kHalf, "W");
    TORCH_CHECK(W.dim() == 2 && W.size(0) % 128 == 0 && W.size(1) % 128 == 0, "W must be [K, N], multiples of 128");
    TORCH_CHECK(T.numel() == W.numel() * K2 / 64, "T must hold K * N * bits / 32 words");
    c10::cuda::CUDAGuard guard(T.device());
    exl3_unpack_cuda(T, W, stride_k, stride_nb, K2, cb);
}

static void check_fold(const at::Tensor& T, const at::Tensor& suh, const at::Tensor& W, int64_t K2) {
    check(T, at::kInt, "T");
    check(suh, at::kHalf, "suh");
    TORCH_CHECK(W.is_cuda() && W.is_contiguous() &&
                    (W.scalar_type() == at::kHalf || W.scalar_type() == at::kBFloat16),
                "W: contiguous fp16 or bf16");
    TORCH_CHECK(W.dim() == 2 && W.size(0) % 128 == 0 && W.size(1) % 128 == 0, "W must be [K, N], multiples of 128");
    TORCH_CHECK(suh.numel() == W.size(0), "suh must have K elements");
    TORCH_CHECK(T.numel() == W.numel() * K2 / 64, "T must hold K * N * bits / 32 words");
}

// W'' [K, N] fp16 or bf16 = diag(suh) Hk W_q Hn / 128: the prompt GEMM's weights with both rotations folded in.
void unpack_fold(const at::Tensor& T, const at::Tensor& suh, at::Tensor W, int64_t stride_k, int64_t stride_nb,
                 int64_t K2, int64_t cb) {
    check_fold(T, suh, W, K2);
    c10::cuda::CUDAGuard guard(T.device());
    exl3_unpack_fold_cuda(T, suh, W, stride_k, stride_nb, K2, cb);
}

// The same W'', bit for bit, from unpack_fold2_kernel (the N side in registers, two blocks an SM).
void unpack_fold2(const at::Tensor& T, const at::Tensor& suh, at::Tensor W, int64_t stride_k, int64_t stride_nb,
                  int64_t K2, int64_t cb) {
    check_fold(T, suh, W, K2);
    c10::cuda::CUDAGuard guard(T.device());
    exl3_unpack_fold2_cuda(T, suh, W, stride_k, stride_nb, K2, cb);
}

// out [M, N] bf16 = x @ W'' * svh + bias with W'' rebuilt in the kernel, never written: unpack_fold2 + _gemm_fold's bits.
void fdirect(const at::Tensor& x, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& suh,
             const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor out, int64_t K2, int64_t cb) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x: bf16 [M, K]");
    TORCH_CHECK(x.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0, "x alignment");
    check(T, at::kInt, "T");
    check(suh, at::kHalf, "suh");
    check(svh, at::kHalf, "svh");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.stride(1) == 1,
                "out: bf16 [M, N], unit column stride");
    TORCH_CHECK(out.size(0) == x.size(0) && suh.numel() == x.size(1) && svh.numel() == out.size(1), "shapes");
    TORCH_CHECK(x.size(1) % 128 == 0 && out.size(1) % 128 == 0, "K and N must be multiples of 128");
    if (bias) check(*bias, at::kHalf, "bias");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_fdirect_cuda(x, T, stride_k, stride_nb, suh, svh, bias, out, K2, cb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fdirect", &fdirect);
    m.def("unpack_fold", &unpack_fold);
    m.def("unpack_fold2", &unpack_fold2);
    m.def("rot_in", &rot_in);
    m.def("linear", &linear);
    m.def("unpack", &unpack);
}
