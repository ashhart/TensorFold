#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
    return __shfl_sync(0xffffffff, x, 0);
}

// One thread builds a depth-first preorder from topologically ordered parents.
// This is small (at most 128 nodes) and runs once before all value-row warps.
// The caller guarantees one root and a maximum branching path of 32 nodes.
__global__ void preorder_kernel(const int* parents, int* order, int* depths, int nodes) {
    int at_depth[32], next_child[32];
    int depth = 0, emitted = 1;
    at_depth[0] = 0;
    next_child[0] = 1;
    order[0] = 0;
    depths[0] = 0;
    while (depth >= 0 && emitted < nodes) {
        const int parent = at_depth[depth];
        int child = -1;
        for (int i = next_child[depth]; i < nodes; ++i) {
            if (parents[i] == parent) {
                child = i;
                next_child[depth] = i + 1;
                break;
            }
        }
        if (child < 0) {
            --depth;
        } else {
            ++depth;
            // Inputs to this kernel must have root depth below 32.
            at_depth[depth] = child;
            next_child[depth] = child + 1;
            order[emitted++] = child;
            depths[child] = depth;
        }
    }
}

template <bool CHAIN, bool DFS, int SLOTS>
__global__ void tree_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                            const __nv_bfloat16* v, const float* g, const float* beta,
                            const float* state0, const int* parents,
                            const int* order, const int* depths, __nv_bfloat16* y,
                            int nodes, int hk, int hv, int dv) {
    const int value = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const int key_head = head / (hv / hk);
    const int state_base = (head * dv + value) * 128;
    float initial[4], states[SLOTS][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) initial[i] = state0[state_base + lane * 4 + i];
    for (int step_index = 0; step_index < nodes; ++step_index) {
        const int node = DFS ? order[step_index] : step_index;
        const int parent = parents[node];
        const int depth = DFS ? depths[node] : 0;
        const int source = CHAIN ? 0 : (DFS ? depth - 1 : parent);
        const int destination = CHAIN ? 0 : (DFS ? depth : node);
        const int key_base = (node * hk + key_head) * 128 + lane * 4;
        const int value_base = (node * hv + head) * dv + value;
        const float decay = g[node * hv + head];
        const float step = beta[node * hv + head];
        float s[4], qi[4], ki[4], mem = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] = (parent < 0 ? initial[i] : states[source][i]) * decay;
            qi[i] = __bfloat162float(q[key_base + i]);
            ki[i] = __bfloat162float(k[key_base + i]);
            mem += s[i] * ki[i];
        }
        const float delta = (__bfloat162float(v[value_base]) - warp_sum(mem)) * step;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] += ki[i] * delta;
            out += s[i] * qi[i];
            states[destination][i] = s[i];
        }
        out = warp_sum(out);
        if (lane == 0) y[value_base] = __float2bfloat16_rn(out);
    }
}

__global__ void replay_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                              const __nv_bfloat16* v, const float* g, const float* beta,
                              const float* state0, const int* rows, const int* count,
                              float* state_out, int hk, int hv, int dv) {
    const int value = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const int key_head = head / (hv / hk);
    const int state_base = (head * dv + value) * 128;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = state0[state_base + lane * 4 + i];
    for (int j = 0; j < count[0]; ++j) {
        const int node = rows[j];
        const int key_base = (node * hk + key_head) * 128 + lane * 4;
        const int value_base = (node * hv + head) * dv + value;
        const float decay = g[node * hv + head];
        float ki[4], mem = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] *= decay;
            ki[i] = __bfloat162float(k[key_base + i]);
            mem += s[i] * ki[i];
        }
        const float delta = (__bfloat162float(v[value_base]) - warp_sum(mem)) * beta[node * hv + head];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += ki[i] * delta;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) state_out[state_base + lane * 4 + i] = s[i];
}

// replay_kernel for many layers in one launch: blockIdx.z picks the layer, whose tensors come from
// a device table of pointers (rows k, v, g, beta, state; one column per layer). Same arithmetic as
// replay_kernel, element for element, so the states are bit-identical to one launch per layer.
__global__ void replay_many_kernel(const long long* table, int layers, const int* rows, const int* count,
                                   float* state_out, int hk, int hv, int dv) {
    const int value = blockIdx.x, head = blockIdx.y, layer = blockIdx.z, lane = threadIdx.x;
    const auto* k = reinterpret_cast<const __nv_bfloat16*>(table[0 * layers + layer]);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(table[1 * layers + layer]);
    const auto* g = reinterpret_cast<const float*>(table[2 * layers + layer]);
    const auto* beta = reinterpret_cast<const float*>(table[3 * layers + layer]);
    const auto* state0 = reinterpret_cast<const float*>(table[4 * layers + layer]);
    float* out = state_out + static_cast<long long>(layer) * hv * dv * 128;
    const int key_head = head / (hv / hk);
    const int state_base = (head * dv + value) * 128;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = state0[state_base + lane * 4 + i];
    for (int j = 0; j < count[0]; ++j) {
        const int node = rows[j];
        const int key_base = (node * hk + key_head) * 128 + lane * 4;
        const int value_base = (node * hv + head) * dv + value;
        const float decay = g[node * hv + head];
        float ki[4], mem = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] *= decay;
            ki[i] = __bfloat162float(k[key_base + i]);
            mem += s[i] * ki[i];
        }
        const float delta = (__bfloat162float(v[value_base]) - warp_sum(mem)) * beta[node * hv + head];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += ki[i] * delta;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) out[state_base + lane * 4 + i] = s[i];
}

} // namespace

void gdn_replay_many_cuda(const at::Tensor& table, int layers, const at::Tensor& rows, const at::Tensor& count,
                          at::Tensor& out, int hk, int hv, int dv) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(dv, hv, layers);
    replay_many_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const long long*>(table.data_ptr<int64_t>()), layers, rows.data_ptr<int>(),
        count.data_ptr<int>(), out.data_ptr<float>(), hk, hv, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_tree_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state,
                   const at::Tensor& parents, at::Tensor& out, bool chain) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(v.size(2), v.size(1));
    const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
    const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
#define TREE_ARGS qp, kp, vp, g.data_ptr<float>(), beta.data_ptr<float>(), state.data_ptr<float>(), parents.data_ptr<int>(), order_ptr, depth_ptr, yp, q.size(0), q.size(1), v.size(1), v.size(2)
    const int* order_ptr = nullptr;
    const int* depth_ptr = nullptr;
    at::Tensor order, depths;
    if (!chain && q.size(0) > 32) {
        order = at::empty({q.size(0)}, parents.options());
        depths = at::empty_like(order);
        preorder_kernel<<<1, 1, 0, stream>>>(parents.data_ptr<int>(), order.data_ptr<int>(),
                                               depths.data_ptr<int>(), q.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        order_ptr = order.data_ptr<int>();
        depth_ptr = depths.data_ptr<int>();
    }
    if (chain) tree_kernel<true, false, 1><<<grid, 32, 0, stream>>>(TREE_ARGS);
    else if (q.size(0) <= 16) tree_kernel<false, false, 16><<<grid, 32, 0, stream>>>(TREE_ARGS);
    else if (q.size(0) <= 32) tree_kernel<false, false, 32><<<grid, 32, 0, stream>>>(TREE_ARGS);
    else tree_kernel<false, true, 32><<<grid, 32, 0, stream>>>(TREE_ARGS);
#undef TREE_ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_replay_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state,
                     const at::Tensor& rows, const at::Tensor& count, at::Tensor& out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(v.size(2), v.size(1));
    replay_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        g.data_ptr<float>(), beta.data_ptr<float>(), state.data_ptr<float>(),
        rows.data_ptr<int>(), count.data_ptr<int>(), out.data_ptr<float>(),
        q.size(1), v.size(1), v.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
