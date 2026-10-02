#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void kda_chain_cuda(const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, const at::Tensor&, int64_t,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, double, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&);
void kda_chain_wide_cuda(const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, const at::Tensor&, int64_t,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, double, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                         at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void kda_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     int64_t, at::Tensor&);
void kda_replay_layers_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, at::Tensor&);

void kda_chain_wide_segments_cuda(const at::Tensor&, int64_t, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                                  int64_t, const at::Tensor&, int64_t, const at::Tensor&, int64_t, const at::Tensor&,
                                  at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, double, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                                  at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void kda_replay_layers_segments_cuda(const at::Tensor&, int64_t, at::Tensor&, int64_t, int64_t, int64_t,
                                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                     int64_t, int64_t, int64_t, int64_t);

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

void chain_wide(const at::Tensor& P, int64_t p_stride, int64_t b_off, const at::Tensor& A, int64_t a_stride,
                const at::Tensor& G, int64_t g_stride, const at::Tensor& cs, const at::Tensor& cw,
                const at::Tensor& state_in, const at::Tensor& a_log, const at::Tensor& dt_bias,
                const at::Tensor& norm_w, double eps, double lower, int64_t rows, at::Tensor out,
                at::Tensor state_out, at::Tensor k_save, at::Tensor v_save, at::Tensor g_save, at::Tensor b_save,
                at::Tensor q_tmp, at::Tensor y_tmp) {
    check(P, at::kBFloat16, "P");
    check(A, at::kBFloat16, "A");
    check(G, at::kBFloat16, "G");
    check(cs, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(state_in, at::kFloat, "state");
    check(out, at::kBFloat16, "out");
    check(k_save, at::kFloat, "k");
    check(v_save, at::kBFloat16, "v");
    check(g_save, at::kFloat, "g");
    check(b_save, at::kFloat, "beta");
    check(q_tmp, at::kFloat, "q scratch");
    check(y_tmp, at::kBFloat16, "read-out scratch");
    const int64_t H = a_log.numel();
    TORCH_CHECK(cs.is_contiguous() && cw.is_contiguous() && state_in.is_contiguous() && out.is_contiguous(),
                "conv state, conv weight, state and out must be contiguous");
    TORCH_CHECK(dt_bias.numel() == H * 128 && norm_w.numel() == 128, "dt_bias [H*128], norm [128]");
    TORCH_CHECK(cw.size(0) == 3 * H * 128 && cw.size(1) == 4, "conv weight [3*H*128, 4]");
    TORCH_CHECK(state_in.numel() == H * 128 * 128, "state must be [H, 128, 128]");
    TORCH_CHECK(rows >= 1 && P.size(0) >= rows && out.size(0) >= rows, "rows");
    TORCH_CHECK(k_save.numel() >= rows * H * 128 && q_tmp.numel() >= rows * H * 128 &&
                y_tmp.numel() >= rows * H * 128 && b_save.numel() >= rows * H, "scratch rows");
    c10::cuda::CUDAGuard guard(P.device());
    kda_chain_wide_cuda(P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps,
                        lower, rows, out, state_out, k_save, v_save, g_save, b_save, q_tmp, y_tmp);
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

static void check_segments(const at::Tensor& seg) {
    TORCH_CHECK(seg.is_cuda() && seg.scalar_type() == at::kInt && seg.dim() == 2 && seg.size(1) == 6 &&
                seg.is_contiguous(), "segments: a contiguous CUDA int32 [nseg, 6] table "
                "(first row, rows, state slot, parity, conv slot, keep)");
}

// rec [slots, 2, H, 128, 128] (this layer's states; each [H, 128, 128] contiguous), conv [slots, 3, 3 H 128]
// (this layer's conv windows; each [3, C] contiguous). Segment i runs rows row0 .. row0 + rows - 1 of P/A/G from
// rec[slot, parity] and conv[conv slot] into rec[slot, 1 - parity]; conv is read only.
void chain_wide_segments(const at::Tensor& seg, const at::Tensor& P, int64_t p_stride, int64_t b_off,
                         const at::Tensor& A, int64_t a_stride, const at::Tensor& G, int64_t g_stride,
                         const at::Tensor& conv, const at::Tensor& cw, at::Tensor rec, const at::Tensor& a_log,
                         const at::Tensor& dt_bias, const at::Tensor& norm_w, double eps, double lower, int64_t rows,
                         at::Tensor out, at::Tensor k_save, at::Tensor v_save, at::Tensor g_save, at::Tensor b_save,
                         at::Tensor q_tmp, at::Tensor y_tmp) {
    check_segments(seg);
    check(P, at::kBFloat16, "P");
    check(A, at::kBFloat16, "A");
    check(G, at::kBFloat16, "G");
    check(conv, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(rec, at::kFloat, "state");
    check(out, at::kBFloat16, "out");
    check(k_save, at::kFloat, "k");
    check(v_save, at::kBFloat16, "v");
    check(g_save, at::kFloat, "g");
    check(b_save, at::kFloat, "beta");
    check(q_tmp, at::kFloat, "q scratch");
    check(y_tmp, at::kBFloat16, "read-out scratch");
    const int64_t H = a_log.numel();
    TORCH_CHECK(cw.is_contiguous() && out.is_contiguous(), "conv weight and out must be contiguous");
    TORCH_CHECK(conv.dim() == 3 && conv.size(1) == 3 && conv.size(2) == 3 * H * 128 && conv.stride(2) == 1 &&
                conv.stride(1) == 3 * H * 128, "conv state [slots, 3, 3*H*128], each slot contiguous");
    TORCH_CHECK(rec.dim() == 5 && rec.size(1) == 2 && rec.size(2) == H && rec.size(3) == 128 && rec.size(4) == 128 &&
                rec.stride(4) == 1 && rec.stride(3) == 128 && rec.stride(2) == 128 * 128,
                "state [slots, 2, H, 128, 128], each [H, 128, 128] contiguous");
    TORCH_CHECK(dt_bias.numel() == H * 128 && norm_w.numel() == 128, "dt_bias [H*128], norm [128]");
    TORCH_CHECK(cw.size(0) == 3 * H * 128 && cw.size(1) == 4, "conv weight [3*H*128, 4]");
    TORCH_CHECK(rows >= 1 && P.size(0) >= rows && out.size(0) >= rows, "rows");
    TORCH_CHECK(k_save.numel() >= rows * H * 128 && q_tmp.numel() >= rows * H * 128 &&
                y_tmp.numel() >= rows * H * 128 && b_save.numel() >= rows * H, "scratch rows");
    c10::cuda::CUDAGuard guard(P.device());
    kda_chain_wide_segments_cuda(seg, seg.size(0), P, p_stride, b_off, A, a_stride, G, g_stride, conv,
                                 conv.stride(0), cw, rec, rec.stride(0), rec.stride(1), a_log, dt_bias, norm_w, eps,
                                 lower, rows, out, k_save, v_save, g_save, b_save, q_tmp, y_tmp);
}

// rec [slots, 2, layers, H, 128, 128] (each [H, 128, 128] contiguous); segment i with keep < rows replays its
// first ``keep`` saved rows of every layer from rec[slot, parity] into rec[slot, 1 - parity].
void replay_layers_segments(const at::Tensor& seg, at::Tensor rec, const at::Tensor& k_save, const at::Tensor& v_save,
                            const at::Tensor& g_save, const at::Tensor& b_save, int64_t kv_stride, int64_t b_stride,
                            int64_t layers, int64_t heads) {
    check_segments(seg);
    check(rec, at::kFloat, "state");
    check(k_save, at::kFloat, "k");
    check(v_save, at::kBFloat16, "v");
    check(g_save, at::kFloat, "g");
    check(b_save, at::kFloat, "beta");
    TORCH_CHECK(rec.dim() == 6 && rec.size(1) == 2 && rec.size(2) >= layers && rec.size(3) == heads &&
                rec.size(4) == 128 && rec.size(5) == 128 && rec.stride(5) == 1 && rec.stride(4) == 128 &&
                rec.stride(3) == 128 * 128, "state [slots, 2, layers, H, 128, 128], each [H, 128, 128] contiguous");
    c10::cuda::CUDAGuard guard(rec.device());
    kda_replay_layers_segments_cuda(seg, seg.size(0), rec, rec.stride(0), rec.stride(1), rec.stride(2), k_save,
                                    v_save, g_save, b_save, kv_stride, b_stride, layers, heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain);
    m.def("chain_wide", &chain_wide);
    m.def("replay", &replay);
    m.def("replay_layers", &replay_layers);
    m.def("chain_wide_segments", &chain_wide_segments);
    m.def("replay_layers_segments", &replay_layers_segments);
}
