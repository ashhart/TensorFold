// Gated DeltaNet: a wave per (batch, value head, value row) walks the sequence, lane sums as the Python reference.

#include "decode/plan.hpp"

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

__device__ float lane0_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int DK>
__global__ void __launch_bounds__(32)
gated_delta_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                   const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                   float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                   float* __restrict__ states) {
    constexpr int SPAN = (DK + 31) / 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (row >= dv) return;
    const int key_head = head / (value_heads / key_heads);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row =
        ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t = 0; t < length; ++t) {
        const size_t step = (static_cast<size_t>(batch) * length + t) * value_heads + head;
        const float decay = gate[step];
        const float step_beta = beta[step];
        const float value = v[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
                              static_cast<size_t>(head) * dv + row];
        const size_t key =
            ((static_cast<size_t>(batch) * length + t) * key_heads + key_head) * DK;

        float remembered = 0.f;
        float ks[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) {
                ks[i] = k[key + idx];
                s[i] = s[i] * decay;
                remembered = remembered + s[i] * ks[i];
            }
        }
        const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) s[i] = s[i] + ks[i] * delta;
        }
        if (states != nullptr) {
            const size_t snap = (((static_cast<size_t>(batch) * length + t) * value_heads + head) * dv + row) * DK;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) states[snap + idx] = s[i];
            }
        }

        float readout = 0.f;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) readout = readout + s[i] * q[key + idx];
        }
        // Every lane has to enter the shuffle. Lane 0 owns the stored sum.
        readout = lane0_sum(readout);
        if (lane == 0) {
            y[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
              static_cast<size_t>(head) * dv + row] = readout;
        }
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

// Value rows share a staged tile of k and q, in the wave kernel's order (-ffp-contract=off keeps it).
template <int DK, int ROWS, int STEPS>
__device__ __forceinline__ void
gated_delta_tile_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                      const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                      float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                      float* __restrict__ states, int batch, bool write_back = true) {
    constexpr int SPAN = (DK + 31) / 32;
    constexpr int WIDTH = 32 * ROWS;
    const int lane = threadIdx.x;
    const int local = threadIdx.y;
    const int row = blockIdx.x * ROWS + local;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const int tid = local * 32 + lane;

    __shared__ float ks[STEPS][DK];
    __shared__ float qs[STEPS][DK];
    __shared__ float gs[STEPS];
    __shared__ float bs[STEPS];
    __shared__ float vs[STEPS][ROWS];

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row = ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t0 = 0; t0 < length; t0 += STEPS) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
        for (int i = tid; i < nvec; i += WIDTH) {
            const int elem = i * 4;
            const int step = elem / DK;
            const int col = elem - step * DK;
            const size_t at =
                ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + col;
            const float4 key = *reinterpret_cast<const float4*>(k + at);
            ks[step][col] = key.x;
            ks[step][col + 1] = key.y;
            ks[step][col + 2] = key.z;
            ks[step][col + 3] = key.w;
            // a replay (no q) advances the state alone
            if (q == nullptr) continue;
            const float4 query = *reinterpret_cast<const float4*>(q + at);
            qs[step][col] = query.x;
            qs[step][col + 1] = query.y;
            qs[step][col + 2] = query.z;
            qs[step][col + 3] = query.w;
        }
        for (int i = tid; i < n; i += WIDTH) {
            const size_t step = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[step];
            bs[i] = beta[step];
        }
        for (int i = tid; i < n * ROWS; i += WIDTH) {
            const int step = i / ROWS;
            const int which = i - step * ROWS;
            vs[step][which] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * dv +
                                blockIdx.x * ROWS + which];
        }
        __syncthreads();

        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt];
            const float step_beta = bs[tt];
            const float value = vs[tt][local];
            float remembered = 0.f;
            float held[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) {
                    held[i] = ks[tt][idx];
                    s[i] = s[i] * decay;
                    remembered = remembered + s[i] * held[i];
                }
            }
            const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) s[i] = s[i] + held[i] * delta;
            }
            if (states != nullptr) {
                const size_t snap =
                    (((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row) * DK;
#pragma unroll
                for (int i = 0; i < SPAN; ++i) {
                    const int idx = lane + i * 32;
                    if (idx < DK) states[snap + idx] = s[i];
                }
            }
            if (q == nullptr) continue;
            float readout = 0.f;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) readout = readout + s[i] * qs[tt][idx];
            }
            readout = lane0_sum(readout);
            if (lane == 0) {
                y[((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row] = readout;
            }
        }
        __syncthreads();
    }

    if (!write_back) return;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

template <int DK, int ROWS, int STEPS>
__global__ void __launch_bounds__(32 * ROWS)
gated_delta_tile(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                 const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                 float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                 float* __restrict__ states) {
    gated_delta_tile_body<DK, ROWS, STEPS>(q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv, states,
                                           blockIdx.z);
}

// A round's windows in one launch, a grid plane a slot; rows, state and snapshots come from the plan, same scan.
template <int DK>
__device__ __forceinline__ void
gated_delta_plan_body(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                      int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    const int slot = blockIdx.z;
    const int length = plan.count[slot];
    if (length == 0) return;
    const long long row0 = plan.first[slot];
    const unsigned long long* desc = plan_desc(plan, slot);
    float* state = reinterpret_cast<float*>(desc[plan_second(layer)]);
    // a window of rows leaves the state as the round found it, for its keep to replay (padding rows keep nothing)
    const bool replayed = length > 1 && slot + 1 < static_cast<int>(gridDim.z);
    gated_delta_tile_body<DK, 8, 16>(q + row0 * key_heads * DK, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                     gate + row0 * value_heads, beta + row0 * value_heads, state,
                                     y + row0 * value_heads * dv, length, key_heads, value_heads, dv, nullptr, 0,
                                     !replayed);
}

// A keep: each slot's state run from the round's start through its kept rows, by the same scan on the inputs kept.
template <int DK>
__device__ __forceinline__ void gated_delta_replay_body(const int* keep, int rows, int key_heads, int value_heads, int dv,
                                                        PlanArgs plan, int slots) {
    const int slot = blockIdx.z % slots;
    const int layer = blockIdx.z / slots;
    const int n = keep[slot];
    if (n <= 0 || plan.count[slot] <= 1 || plan.snaps[2 * layer + 1] == 0) return;
    const long long row0 = plan.first[slot];
    const float* k = reinterpret_cast<const float*>(plan.snaps[2 * layer + 1]);
    const float* v = k + static_cast<long long>(rows) * key_heads * DK;
    const float* gate = v + static_cast<long long>(rows) * value_heads * dv;
    const float* beta = gate + static_cast<long long>(rows) * value_heads;
    float* state = reinterpret_cast<float*>(plan_desc(plan, slot)[plan_second(layer)]);
    gated_delta_tile_body<DK, 8, 16>(nullptr, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                     gate + row0 * value_heads, beta + row0 * value_heads, state, nullptr, n, key_heads,
                                     value_heads, dv, nullptr, 0);
}

}  // namespace rocm
}  // namespace tf

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_128(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<128>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_128(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<128>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_16(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<16>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_16(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
               int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<16>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

// The template kernels launches.zig resolves.
template __global__ void tf::rocm::gated_delta_tile<128, 8, 16>(const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
    const float* __restrict__ beta, float* __restrict__ state, float* __restrict__ y, int length, int key_heads,
    int value_heads, int dv, float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_tile<16, 8, 16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<128>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
