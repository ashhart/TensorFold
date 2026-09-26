#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gdn_tree_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, at::Tensor&, bool);
void gdn_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, at::Tensor&);
void gdn_replay_many_cuda(const at::Tensor&, int, const at::Tensor&, const at::Tensor&, at::Tensor&, int, int, int);

static void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && state.is_cuda(),
                "GDN inputs must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                beta.is_contiguous() && state.is_contiguous(), "GDN inputs must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16, "q, k, v must be bf16");
    TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat &&
                state.scalar_type() == at::kFloat, "g, beta, state must be fp32");
    TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes() && v.dim() == 3 && g.dim() == 2 &&
                beta.sizes() == g.sizes() && state.dim() == 3, "invalid GDN ranks or paired shapes");
    TORCH_CHECK(q.size(0) > 0 && q.size(2) == 128 && v.size(0) == q.size(0) &&
                v.size(1) == g.size(1) && g.size(0) == q.size(0) &&
                state.size(0) == v.size(1) && state.size(1) == v.size(2) && state.size(2) == 128 &&
                v.size(1) % q.size(1) == 0, "invalid GDN dimensions");
    const auto device = q.device();
    TORCH_CHECK(k.device() == device && v.device() == device && g.device() == device &&
                beta.device() == device && state.device() == device, "GDN inputs must share a device");
}

at::Tensor tree(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state,
                const at::Tensor& parents, bool chain) {
    check_inputs(q, k, v, g, beta, state);
    TORCH_CHECK(parents.is_cuda() && parents.is_contiguous() && parents.scalar_type() == at::kInt &&
                parents.device() == q.device() && parents.numel() == q.size(0), "invalid parents");
    TORCH_CHECK(q.size(0) <= 128, "tree is limited to 128 nodes");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty({q.size(0), v.size(1), v.size(2)}, q.options());
    gdn_tree_cuda(q, k, v, g, beta, state, parents, out, chain);
    return out;
}

at::Tensor replay(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state,
                  const at::Tensor& rows, const at::Tensor& count) {
    check_inputs(q, k, v, g, beta, state);
    TORCH_CHECK(rows.is_cuda() && rows.is_contiguous() && rows.scalar_type() == at::kInt &&
                rows.device() == q.device() && rows.numel() >= q.size(0), "invalid rows");
    TORCH_CHECK(count.is_cuda() && count.is_contiguous() && count.scalar_type() == at::kInt &&
                count.device() == q.device() && count.numel() == 1, "invalid count");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty_like(state);
    gdn_replay_cuda(q, k, v, g, beta, state, rows, count, out);
    return out;
}

// Every GDN layer's replay in one launch. All layers share shapes; each layer's inputs are checked
// like replay's, and the kernel reads them through a table of device pointers.
at::Tensor replay_many(const std::vector<at::Tensor>& q, const std::vector<at::Tensor>& k,
                       const std::vector<at::Tensor>& v, const std::vector<at::Tensor>& g,
                       const std::vector<at::Tensor>& beta, const std::vector<at::Tensor>& state,
                       const at::Tensor& rows, const at::Tensor& count) {
    const int layers = static_cast<int>(q.size());
    TORCH_CHECK(layers > 0 && k.size() == q.size() && v.size() == q.size() && g.size() == q.size() &&
                beta.size() == q.size() && state.size() == q.size(), "one entry per layer in every list");
    for (int i = 0; i < layers; ++i) {
        check_inputs(q[i], k[i], v[i], g[i], beta[i], state[i]);
        TORCH_CHECK(k[i].sizes() == k[0].sizes() && v[i].sizes() == v[0].sizes() && g[i].sizes() == g[0].sizes() &&
                    state[i].sizes() == state[0].sizes() && k[i].device() == k[0].device(),
                    "every layer must have the same shapes and device");
    }
    TORCH_CHECK(rows.is_cuda() && rows.is_contiguous() && rows.scalar_type() == at::kInt &&
                rows.device() == q[0].device() && rows.numel() >= q[0].size(0), "invalid rows");
    TORCH_CHECK(count.is_cuda() && count.is_contiguous() && count.scalar_type() == at::kInt &&
                count.device() == q[0].device() && count.numel() == 1, "invalid count");
    c10::cuda::CUDAGuard guard(q[0].device());
    auto host = at::empty({5, layers}, at::TensorOptions().dtype(at::kLong));
    auto* t = host.data_ptr<int64_t>();
    for (int i = 0; i < layers; ++i) {
        t[0 * layers + i] = reinterpret_cast<int64_t>(k[i].data_ptr());
        t[1 * layers + i] = reinterpret_cast<int64_t>(v[i].data_ptr());
        t[2 * layers + i] = reinterpret_cast<int64_t>(g[i].data_ptr());
        t[3 * layers + i] = reinterpret_cast<int64_t>(beta[i].data_ptr());
        t[4 * layers + i] = reinterpret_cast<int64_t>(state[i].data_ptr());
    }
    auto table = host.to(q[0].device(), /*non_blocking=*/false);
    auto out = at::empty({layers, state[0].size(0), state[0].size(1), state[0].size(2)}, state[0].options());
    gdn_replay_many_cuda(table, layers, rows, count, out, static_cast<int>(q[0].size(1)),
                         static_cast<int>(v[0].size(1)), static_cast<int>(v[0].size(2)));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tree", &tree);
    m.def("replay", &replay);
    m.def("replay_many", &replay_many);
}
