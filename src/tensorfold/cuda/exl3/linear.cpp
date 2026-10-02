#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_rot_in_cuda(const at::Tensor&, const std::vector<at::Tensor>&, std::vector<at::Tensor>&, int64_t);
void exl3_linear_cuda(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<int64_t>&,
                      const std::vector<int64_t>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                      std::vector<at::Tensor>&, const std::vector<at::Tensor>&, std::vector<at::Tensor>&, int64_t,
                      int64_t, const std::vector<int64_t>&, int64_t, int64_t);
void exl3_unpack_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);

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

// xh_j [M, K] fp16 = fp16(((x * suh_j) @ H) / sqrt(128)) for 1 to 3 layers of input x [M, K] fp16, bf16 or fp32.
void rot_in(const at::Tensor& x, const std::vector<at::Tensor>& suh, std::vector<at::Tensor> xh, int64_t pdl) {
    check_io(x, "x");
    TORCH_CHECK(x.size(1) % 128 == 0, "x must be [M, K], K a multiple of 128");
    TORCH_CHECK(!suh.empty() && suh.size() <= 3 && xh.size() == suh.size(), "rot_in: 1 to 3 (suh, xh) pairs");
    for (size_t i = 0; i < suh.size(); ++i) {
        check(suh[i], at::kHalf, "suh");
        check(xh[i], at::kHalf, "xh");
        TORCH_CHECK(suh[i].numel() == x.size(1) && xh[i].sizes() == x.sizes(), "suh [K] and xh [M, K]");
    }
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, suh, xh, pdl);
}

// y_j [M, N_j] = (xh_j @ W_q,j) @ H * svh_j + bias_j for 1 to 3 layers of one input (one launch; M, K, the width,
// codebook and WK shared, SK_j each); Z_j [SK_j, M, N_j] fp32 when SK_j > 1 (else empty), bias_j empty for none;
// counters_j int32 [8 * N_j / 128], left zero.
void linear(const std::vector<at::Tensor>& xh, const std::vector<at::Tensor>& T, const std::vector<int64_t>& stride_k,
            const std::vector<int64_t>& stride_nb, const std::vector<at::Tensor>& svh,
            const std::vector<at::Tensor>& bias, std::vector<at::Tensor> y, const std::vector<at::Tensor>& Z,
            std::vector<at::Tensor> counters, int64_t K2, int64_t cb, const std::vector<int64_t>& SK, int64_t WK,
            int64_t loads) {
    const size_t n = xh.size();
    TORCH_CHECK(n >= 1 && n <= 3, "linear: 1 to 3 layers a launch");
    for (const std::vector<at::Tensor>* v : {&T, &svh, &bias, &Z})
        TORCH_CHECK(v->size() == n, "linear: one entry a layer in every list");
    TORCH_CHECK(y.size() == n && counters.size() == n, "linear: one entry a layer in every list");
    TORCH_CHECK(stride_k.size() == n && stride_nb.size() == n && SK.size() == n, "linear: one entry a layer");
    const int64_t M = xh[0].size(0), K = xh[0].size(1);
    for (size_t i = 0; i < n; ++i) {
        check(xh[i], at::kHalf, "xh");
        check_io(y[i], "y");
        check(svh[i], at::kHalf, "svh");
        check(T[i], at::kInt, "T");
        check(counters[i], at::kInt, "counters");
        const int64_t N = y[i].size(1);
        TORCH_CHECK(xh[i].dim() == 2 && xh[i].size(0) == M && xh[i].size(1) == K && y[i].size(0) == M && M >= 1 &&
                        M <= 128,
                    "xh and y must have the same 1 to 128 rows (and the layers the same K)");
        TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
        TORCH_CHECK(svh[i].numel() == N, "svh must have N elements");
        TORCH_CHECK(T[i].numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(T[i].data_ptr()) % 16 == 0, "T must be 16-byte aligned");
        TORCH_CHECK(counters[i].numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
        if (bias[i].numel()) {
            check(bias[i], at::kHalf, "bias");
            TORCH_CHECK(bias[i].numel() == N, "bias must have N elements");
        }
        if (SK[i] > 1) {
            check(Z[i], at::kFloat, "Z");
            TORCH_CHECK(Z[i].numel() >= SK[i] * M * N, "Z too small");
        }
    }
    c10::cuda::CUDAGuard guard(xh[0].device());
    exl3_linear_cuda(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, loads);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rot_in", &rot_in);
    m.def("linear", &linear);
    m.def("unpack", &unpack);
}
