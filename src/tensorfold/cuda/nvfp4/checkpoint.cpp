#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void quant4_cuda(const at::Tensor&, double, at::Tensor&, at::Tensor&, int64_t, int64_t);
void quant8_cuda(const at::Tensor&, double, at::Tensor&, int64_t, int64_t);
void lane_cuda(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double, at::Tensor&,
               const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, bool);
void pack4_cuda(const at::Tensor&, int64_t, int64_t, at::Tensor&);
void gemm_cuda(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double, at::Tensor&,
               int64_t, int64_t, int64_t, int64_t, bool);
void gemm_gu_ck_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, double, double, int64_t, int64_t, int64_t, double, at::Tensor&, at::Tensor&,
                     bool);
void gemm_ws_cuda(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double,
                  at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

static void rows_in(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.size(1) % 64 == 0,
                "x: (M, K) bf16 rows, 16-byte aligned, K a multiple of 64");
}

// NVFP4 rows under the static global scale ``g`` (1 / input_scale): codes [M, K/2] and scales [K/64, mpad, 4], or
// with ``tb`` > 0 the prompt GEMM's tiles: codes [mpad / tb][K/64][tb][32], scales [mpad / tb, K/64, tb, 4].
void quant4(const at::Tensor& x, double g, at::Tensor codes, at::Tensor scales, int64_t tb) {
    rows_in(x);
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(tb == 0 || (tb % 16 == 0 && scales.dim() == 4 && scales.size(1) == k / 64 && scales.size(2) == tb &&
                            scales.size(3) == 4), "tb: 0, or a multiple of 16 with scales (mpad / tb, K/64, tb, 4)");
    TORCH_CHECK(tb || (scales.dim() == 3 && scales.size(0) == k / 64 && scales.size(2) == 4),
                "scales: (K/64, mpad, 4)");
    const int64_t mpad = tb ? scales.size(0) * tb : scales.size(1);
    TORCH_CHECK(scales.is_contiguous() && scales.scalar_type() == at::kByte && mpad >= m && mpad % 64 == 0,
                "scales: uint8, mpad a multiple of 64 holding every row");
    TORCH_CHECK(codes.is_contiguous() && codes.scalar_type() == at::kByte &&
                codes.numel() == (tb ? mpad : m) * k / 2, "codes: (M, K/2) uint8, (mpad, K/2) when tiled");
    c10::cuda::CUDAGuard guard(x.device());
    quant4_cuda(x, g, codes, scales, mpad, tb);
}

// FP8 rows under the static scale (``inv`` = 1 / input_scale), e4m3 bytes in the weights' fragment order; with
// ``tb`` > 0 the prompt GEMM's tiles [mpad / tb][K/64][tb][64] for out (mpad, K), rows past M zero.
void quant8(const at::Tensor& x, double inv, at::Tensor out, int64_t tb) {
    rows_in(x);
    const int64_t mpad = tb ? out.size(0) : x.size(0);
    TORCH_CHECK(tb == 0 || (tb % 16 == 0 && mpad % tb == 0 && mpad >= x.size(0)), "tb: 0 or a multiple of 16 "
                "dividing the padded rows");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kByte && out.size(0) == mpad &&
                out.size(1) == x.size(1), "out: (M, K) uint8, (mpad, K) when tiled");
    c10::cuda::CUDAGuard guard(x.device());
    quant8_cuda(x, inv, out, mpad, tb);
}

// out (M, n) = alpha * rows @ weight: mode 0 NVFP4 x NVFP4 (``quant4`` rows, ``pack4`` words, block scales
// [npad/64, K/64, 64, 4]), mode 1 FP8 x FP8 (``quant8`` rows, fragment-order bytes); part (SK, M, n) past 8 slices.
// ``tile`` (``checkpoint.lane_tile``) picks the block's shape, never the bits.
void lane(int64_t mode, const at::Tensor& x, const c10::optional<at::Tensor>& xs, const at::Tensor& w,
          const c10::optional<at::Tensor>& ws, double alpha, at::Tensor out, const c10::optional<at::Tensor>& part,
          int64_t n, int64_t k, int64_t sk, int64_t npad, int64_t tile, bool f32) {
    TORCH_CHECK(mode == 0 || mode == 1, "mode 0 (NVFP4) or 1 (FP8)");
    const int64_t m = out.size(0);
    TORCH_CHECK(k % 64 == 0 && (k / 64) % sk == 0, "K in whole steps of 64, split evenly");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kByte && x.size(0) == m &&
                x.size(1) == (mode == 0 ? k / 2 : k), "x: quantized rows (M, K/2 | K)");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
                "weight bytes do not match npad and K");
    int64_t mpad = 0;
    if (mode == 0) {
        TORCH_CHECK(xs.has_value() && xs->is_contiguous() && xs->size(0) == k / 64 && xs->size(1) >= m &&
                    xs->size(1) % 64 == 0, "xs: (K/64, mpad, 4) row scales");
        TORCH_CHECK(ws.has_value() && ws->is_contiguous() && ws->numel() == (k / 64) * npad * 4,
                    "ws: [npad/64, K/64, 64, 4] block scales");
        mpad = xs->size(1);
    }
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    TORCH_CHECK(sk <= 8 || (part.has_value() && part->numel() >= sk * m * n), "part: (SK, M, n) fp32");
    c10::cuda::CUDAGuard guard(x.device());
    lane_cuda(mode, x, xs.has_value() ? *xs : at::Tensor(), w, ws.has_value() ? *ws : at::Tensor(), alpha, out,
              part.has_value() ? *part : at::Tensor(), n, k, sk, mpad, tile, f32);
}

// Prompt rows: out (M, n) = alpha * rows @ weight in ``lane``'s layouts, one K chain a row (``tile`` picks the shape).
void gemm(int64_t mode, const at::Tensor& x, const c10::optional<at::Tensor>& xs, const at::Tensor& w,
          const c10::optional<at::Tensor>& ws, double alpha, at::Tensor out, int64_t n, int64_t k, int64_t npad,
          int64_t tile, bool f32) {
    TORCH_CHECK(mode == 0 || mode == 1, "mode 0 (NVFP4) or 1 (FP8)");
    const int64_t m = out.size(0);
    TORCH_CHECK(k % 64 == 0 && npad % 64 == 0 && n <= npad, "K in steps of 64, npad whole 64-column tiles");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kByte && x.size(0) == m &&
                x.size(1) == (mode == 0 ? k / 2 : k), "x: quantized rows (M, K/2 | K)");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
                "weight bytes do not match npad and K");
    if (mode == 0)
        TORCH_CHECK(xs.has_value() && xs->is_contiguous() && xs->size(0) == k / 64 && xs->size(1) >= m &&
                    xs->size(1) % 64 == 0 && ws.has_value() && ws->is_contiguous() &&
                    ws->numel() == (k / 64) * npad * 4, "xs: (K/64, mpad, 4) row scales; ws: [npad/64, K/64, 64, 4]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x.device());
    gemm_cuda(mode, x, xs.has_value() ? *xs : at::Tensor(), w, ws.has_value() ? *ws : at::Tensor(), alpha, out, n, k,
              npad, tile, f32);
}

// NVFP4 checkpoint bytes [N, K/2] -> ``lane``'s words [npad/64, K/64, 8, 32, 2] (int32).
void pack4(const at::Tensor& src, at::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.scalar_type() == at::kByte && src.dim() == 2,
                "src: (N, K/2) uint8");
    const int64_t n = src.size(0), k = src.size(1) * 2;
    TORCH_CHECK(k % 64 == 0 && dst.is_contiguous() && (dst.numel() * dst.element_size()) % (k / 2 * 64) == 0 &&
                dst.numel() * dst.element_size() >= n * k / 2, "dst: [npad/64, K/64, 8, 32, 2] words");
    c10::cuda::CUDAGuard guard(src.device());
    pack4_cuda(src, n, k, dst);
}

// Prompt rows on the warp-specialized GEMM: x the tiled rows (``quant4`` / ``quant8`` with tb 128), else as ``gemm``.
void gemm_ws(int64_t mode, const at::Tensor& x, const c10::optional<at::Tensor>& xs, const at::Tensor& w,
             const c10::optional<at::Tensor>& ws, double alpha, at::Tensor out, int64_t n, int64_t k, int64_t npad,
             int64_t tile, bool f32) {
    TORCH_CHECK(mode == 0 || mode == 1, "mode 0 (NVFP4) or 1 (FP8)");
    const int64_t m = out.size(0), mpad = x.size(0);
    TORCH_CHECK(k % 64 == 0 && npad % 64 == 0 && n <= npad, "K in steps of 64, npad whole 64-column tiles");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kByte && mpad % 128 == 0 && mpad >= m &&
                x.size(1) == (mode == 0 ? k / 2 : k), "x: tiled rows (mpad, K/2 | K), mpad a multiple of 128");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
                "weight bytes do not match npad and K");
    if (mode == 0)
        TORCH_CHECK(xs.has_value() && xs->is_contiguous() && xs->dim() == 4 && xs->size(0) * xs->size(2) == mpad &&
                    xs->size(1) == k / 64 && xs->size(2) == 128 && ws.has_value() && ws->is_contiguous() &&
                    ws->numel() == (k / 64) * npad * 4, "xs: (mpad / 128, K/64, 128, 4); ws: [npad/64, K/64, 64, 4]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x.device());
    gemm_ws_cuda(mode, x, xs.has_value() ? *xs : at::Tensor(), w, ws.has_value() ? *ws : at::Tensor(), alpha, out, n,
                 k, mpad, npad, tile, f32);
}

// gate|up of the same NVFP4 rows (``quant4``'s row-major layout) -> SiLU(gate) * up -> down's input as NVFP4 rows
// under down's global scale ``qg`` through the cp.async GEMM: codes (M, npad/2), scales (npad/64, mpad, 4).
void gemm_gu_ck(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& wg, const at::Tensor& wsg,
                const at::Tensor& wu, const at::Tensor& wsu, double alpha_g, double alpha_u, int64_t npad, int64_t k,
                double qg, at::Tensor codes, at::Tensor scales, bool fp32) {
    const int64_t m = x.size(0), mpad = xs.size(1);
    TORCH_CHECK(k % 128 == 0 && npad % 64 == 0, "K in steps of 128, npad whole 64-column tiles");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kByte && x.size(1) == k / 2,
                "x: NVFP4 rows (M, K/2)");
    TORCH_CHECK(xs.is_contiguous() && xs.dim() == 3 && xs.size(0) == k / 64 && mpad >= m && mpad % 64 == 0,
                "xs: (K/64, mpad, 4) row scales");
    for (const auto* w : {&wg, &wu})
        TORCH_CHECK(w->is_cuda() && w->is_contiguous() && w->numel() * w->element_size() == npad * k / 2,
                    "weight bytes do not match npad and K");
    for (const auto* ws : {&wsg, &wsu})
        TORCH_CHECK(ws->is_contiguous() && ws->numel() == (k / 64) * npad * 4, "ws: [npad/64, K/64, 64, 4]");
    TORCH_CHECK(codes.is_contiguous() && codes.scalar_type() == at::kByte && codes.size(0) == m &&
                codes.size(1) == npad / 2, "codes: (M, npad/2) uint8");
    TORCH_CHECK(scales.is_contiguous() && scales.scalar_type() == at::kByte && scales.dim() == 3 &&
                scales.size(0) == npad / 64 && scales.size(1) == mpad && scales.size(2) == 4,
                "scales: (npad/64, mpad, 4) uint8");
    c10::cuda::CUDAGuard guard(x.device());
    gemm_gu_ck_cuda(x, xs, wg, wsg, wu, wsu, alpha_g, alpha_u, m, npad, k, qg, codes, scales, fp32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant4", &quant4);
    m.def("quant8", &quant8);
    m.def("lane", &lane);
    m.def("pack4", &pack4);
    m.def("gemm", &gemm);
    m.def("gemm_ws", &gemm_ws);
    m.def("gemm_gu_ck", &gemm_gu_ck);
}
