#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t exl3p_item_rows();
void exl3p_route_cuda(const at::Tensor&, int64_t, int64_t, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t);
void exl3p_experts_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                        double, int64_t, int64_t, int64_t, int64_t, int64_t);

void exl3p_slot_sum_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);
void exl3p_fix_to_float_cuda(const at::Tensor&, at::Tensor&);
void exl3p_slot_sum16_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void route(const at::Tensor& pick, int64_t E, at::Tensor sorted, at::Tensor items, at::Tensor item_count,
           int64_t max_items, int64_t by_count) {
    check(pick, at::kInt, "pick");
    check(sorted, at::kInt, "sorted");
    check(items, at::kInt, "items");
    check(item_count, at::kInt, "item_count");
    TORCH_CHECK(sorted.numel() >= pick.numel() && items.numel() >= 3 * max_items, "route: buffers too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3p_route_cuda(pick, pick.numel(), E, sorted, items, item_count, max_items, by_count);
}

void experts(const at::Tensor& x, const at::Tensor& sorted, const at::Tensor& items, const at::Tensor& item_count,
             const at::Tensor& gate_ptr, const at::Tensor& up_ptr, const at::Tensor& down_ptr, const at::Tensor& gu_k2,
             const at::Tensor& d_k2, const at::Tensor& suh_g, const at::Tensor& suh_u, const at::Tensor& svh_g,
             const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d, const at::Tensor& wts,
             at::Tensor xd, at::Tensor out, int64_t D, int64_t I, int64_t NS, int64_t slots, int64_t max_items,
             double limit, int64_t act_mode, int64_t cb, int64_t ncb, int64_t which, int64_t f16) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                    x.size(1) == D,
                "x: bf16 [R, D], unit column stride, row stride a multiple of 8");
    check(sorted, at::kInt, "sorted");
    check(items, at::kInt, "items");
    check(item_count, at::kInt, "item_count");
    check(gate_ptr, at::kLong, "gate_ptr");
    check(up_ptr, at::kLong, "up_ptr");
    check(down_ptr, at::kLong, "down_ptr");
    check(gu_k2, at::kInt, "gu_k2");
    check(d_k2, at::kInt, "d_k2");
    for (auto* t : {&suh_g, &suh_u, &svh_g, &svh_u, &suh_d, &svh_d}) check(*t, at::kHalf, "suh/svh");
    check(wts, at::kFloat, "wts");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                    out.scalar_type() == (f16 == 1 || f16 == 4 ? at::kHalf : f16 == 3 ? at::kLong : at::kFloat),
                "out: contiguous fp32 (fp16 with f16 = 1, int64 fixed point with f16 = 3)");
    TORCH_CHECK(xd.numel() >= sorted.numel() * I, "xd too small");
    TORCH_CHECK(out.size(0) == (f16 == 2 || f16 == 4 ? sorted.numel() : x.size(0)) && out.size(1) == D,
                "out: [R, D] ([R * slots, D] pair rows with f16 = 2)");
    c10::cuda::CUDAGuard guard(x.device());
    exl3p_experts_cuda(x, sorted, items, item_count, gate_ptr, up_ptr, down_ptr, gu_k2, d_k2, suh_g, suh_u, svh_g,
                       svh_u, suh_d, svh_d, wts, xd, out, D, I, NS, slots, max_items, limit, act_mode, cb, ncb,
                       which, f16);
}

void slot_sum(const at::Tensor& pairs, const at::Tensor& pick, at::Tensor out, int64_t E) {
    check(pairs, at::kFloat, "pairs");
    check(pick, at::kInt, "pick");
    check(out, at::kFloat, "out");
    TORCH_CHECK(pick.dim() == 2 && pairs.size(0) == pick.numel() && pairs.size(1) == out.size(1) &&
                    out.size(0) == pick.size(0) && out.size(1) % 4 == 0,
                "slot_sum: pairs [R * S, D], pick [R, S], out [R, D]");
    c10::cuda::CUDAGuard guard(pairs.device());
    exl3p_slot_sum_cuda(pairs, pick, out, E);
}

void fix_to_float(const at::Tensor& in, at::Tensor out) {
    check(in, at::kLong, "in");
    check(out, at::kFloat, "out");
    TORCH_CHECK(in.numel() == out.numel(), "fix_to_float: same sizes");
    c10::cuda::CUDAGuard guard(in.device());
    exl3p_fix_to_float_cuda(in, out);
}

void slot_sum16(const at::Tensor& pairs, const at::Tensor& pick, at::Tensor out, int64_t E) {
    check(pairs, at::kHalf, "pairs");
    check(pick, at::kInt, "pick");
    check(out, at::kFloat, "out");
    TORCH_CHECK(pick.dim() == 2 && pairs.size(0) == pick.numel() && pairs.size(1) == out.size(1) &&
                    out.size(0) == pick.size(0) && out.size(1) % 4 == 0,
                "slot_sum16: pairs [R * S, D] fp16, pick [R, S], out [R, D] fp32");
    c10::cuda::CUDAGuard guard(pairs.device());
    exl3p_slot_sum16_cuda(pairs, pick, out, E);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("slot_sum16", &slot_sum16);
    m.def("fix_to_float", &fix_to_float);
    m.def("slot_sum", &slot_sum);
    m.def("item_rows", &exl3p_item_rows);
    m.def("route", &route);
    m.def("experts", &experts);
}
