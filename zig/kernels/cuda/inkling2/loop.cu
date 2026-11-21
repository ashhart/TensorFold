// Inkling 2 Mini's loop: cross-pass message mixes and rank-r gates; a C2 stand-in with #4337's arithmetic, fp32 sums.
#include <cstdint>
#include <cuda_bf16.h>

namespace ink2l {

constexpr int kMaxBlocks = 8;    // finished blocks a mix reads beside the partial state
constexpr int kMaxQueries = 4;   // queries of one mix: a message's q, k, v and mlp, or an input's 1 or 3
constexpr int kMaxGates = 3;     // gates one launch runs: the attention's q, k and v inputs
constexpr int kMaxRank = 128;    // gate rank (config.max_gate_rank)
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The block's sum of v in one order: warp shuffles, then the warps' partials in warp order; every thread gets it.
__device__ __forceinline__ float blockSum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warpSum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < kWarps; ++i) t += sh[i];
    return t;
}

}  // namespace ink2l

using namespace ink2l;

// An attention-residual mix without the norm: sources are the blocks then the partial; queries attend over them.
struct LoopMixArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, H] each
    const void* partial;                 // [rows, H]: fp32 when partial_f32, else bf16
    const float* query[kMaxQueries];     // fp32 [H] each
    float* out;                          // fp32 [nq][rows, H], planes out_stride elements apart
    long long out_stride;
    int n_blocks, partial_f32, nq, rows, H;
    float eps, scale;  // eps: the sources' rms; scale: H^-0.5 on the scores
};

// Per row: each source's 1/rms, each query's softmax over (q . source) / rms * scale, then the fp32 mixes.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_mix(const LoopMixArgs a) {
    __shared__ float sh[kWarps];
    __shared__ float inv[kMaxBlocks + 1];
    __shared__ float coef[kMaxQueries][kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int H = a.H, S = a.n_blocks + 1;
    auto src = [&](int s, int i) -> float {
        if (s < a.n_blocks) return bf(a.blocks[s][row * H + i]);
        if (a.partial_f32) return static_cast<const float*>(a.partial)[row * H + i];
        return bf(static_cast<const uint16_t*>(a.partial)[row * H + i]);
    };
    for (int s = 0; s < S; ++s) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < H; i += kThreads) {
            const float v = src(s, i);
            acc = fmaf(v, v, acc);
        }
        const float t = blockSum(acc, sh);
        if (threadIdx.x == 0) inv[s] = rsqrtf(t / static_cast<float>(H) + a.eps);
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float score[kMaxBlocks + 1];
        for (int s = 0; s < S; ++s) {
            float acc = 0.0f;
            for (int i = threadIdx.x; i < H; i += kThreads) acc = fmaf(a.query[q][i], src(s, i), acc);
            score[s] = blockSum(acc, sh) * inv[s] * a.scale;
        }
        if (threadIdx.x == 0) {
            float m = -INFINITY, l = 0.0f;
            for (int s = 0; s < S; ++s) m = fmaxf(m, score[s]);
            for (int s = 0; s < S; ++s) {
                score[s] = expf(score[s] - m);
                l += score[s];
            }
            for (int s = 0; s < S; ++s) coef[q][s] = (score[s] / l) * inv[s];
        }
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float* dst = a.out + q * a.out_stride + row * H;
        for (int i = threadIdx.x; i < H; i += kThreads) {
            float u = 0.0f;
            for (int s = 0; s < S; ++s) u = fmaf(coef[q][s], src(s, i), u);
            dst[i] = u;
        }
    }
}

// The gates of one launch: gate g reads state h[g] (fp32 [rows, H]) and message plane slot[g] of msg.
struct LoopGateArgs {
    const float* h[kMaxGates];
    const float* msg;            // fp32 [planes][rows, H]
    long long msg_stride;        // elements between the message planes
    int slot[kMaxGates];
    const uint16_t* down_w[kMaxGates];   // bf16 [rank][2 H]: [state, message]
    const uint16_t* forget[kMaxGates];   // bf16 [H][rank]
    const uint16_t* gain[kMaxGates];     // bf16 [H][rank]
    const uint16_t* norm[kMaxGates];     // bf16 [H]
    uint16_t* down;              // bf16 [gates][rows][rank]
    uint16_t* out;               // bf16 [gates][rows, H], planes out_stride elements apart
    long long out_stride;
    int rows, H, rank;
    float eps_unit, eps;  // eps_unit: the unweighted norms of state and message; eps: the gated output's norm
    float* ys;            // fp32 [gates][rows, H]: the gated rows before their norm
};

// down[g][row] = bf16(W_down [unit(h), unit(m)]), gate y, row x; block z: rank rows [8z, 8z + 8), a warp each.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_down(const LoopGateArgs a) {
    extern __shared__ float xs[];  // [2 H]: the normed state, then the normed message, each rounded to bf16
    __shared__ float sh[kWarps];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H;
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    float sh_acc = 0.0f, sm_acc = 0.0f;
    for (int i = threadIdx.x; i < H; i += kThreads) {
        sh_acc = fmaf(h[i], h[i], sh_acc);
        sm_acc = fmaf(m[i], m[i], sm_acc);
    }
    const float ih = rsqrtf(blockSum(sh_acc, sh) / static_cast<float>(H) + a.eps_unit);
    const float im = rsqrtf(blockSum(sm_acc, sh) / static_cast<float>(H) + a.eps_unit);
    for (int i = threadIdx.x; i < H; i += kThreads) {
        xs[i] = bf(tobf(h[i] * ih));
        xs[H + i] = bf(tobf(m[i] * im));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.z * kWarps + w;
    if (r >= a.rank) return;
    const uint16_t* wr = a.down_w[g] + static_cast<size_t>(r) * 2 * H;
    float acc = 0.0f;
    for (int i = lane; i < 2 * H; i += 32) acc = fmaf(bf(wr[i]), xs[i], acc);
    acc = warpSum(acc);
    if (lane == 0) a.down[(static_cast<size_t>(g) * a.rows + row) * a.rank + r] = tobf(acc);
}

// ys[g][row][i] = (1 - tanh(Wf d)) h + tanh(Wg d) m, d the gate's bf16 down row: kLin outputs a block (z).
constexpr int kLin = 128;
extern "C" __global__ void __launch_bounds__(kLin) tf_ink2_loop_gate_lin(const LoopGateArgs a) {
    __shared__ float dn[kMaxRank];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H, R = a.rank;
    if (threadIdx.x < R) dn[threadIdx.x] = bf(a.down[(static_cast<size_t>(g) * a.rows + row) * R + threadIdx.x]);
    if (threadIdx.x + kLin < R) dn[threadIdx.x + kLin] = bf(a.down[(static_cast<size_t>(g) * a.rows + row) * R + threadIdx.x + kLin]);
    __syncthreads();
    const int i = blockIdx.z * kLin + threadIdx.x;
    if (i >= H) return;
    const uint16_t* wf = a.forget[g] + static_cast<size_t>(i) * R;
    const uint16_t* wg = a.gain[g] + static_cast<size_t>(i) * R;
    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < R; ++r) {
        f = fmaf(bf(wf[r]), dn[r], f);
        gn = fmaf(bf(wg[r]), dn[r], gn);
    }
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    a.ys[(static_cast<size_t>(g) * a.rows + row) * H + i] = (1.0f - tanhf(f)) * h[i] + tanhf(gn) * m[i];
}

// out[g][row] = norm * rms(ys[g][row]): the squares summed in the order a 256-thread block strides them.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_mix(const LoopGateArgs a) {
    __shared__ float sh[kWarps];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H;
    const float* ys = a.ys + (static_cast<size_t>(g) * a.rows + row) * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += kThreads) ss = fmaf(ys[i], ys[i], ss);
    const float inv = rsqrtf(blockSum(ss, sh) / static_cast<float>(H) + a.eps);
    uint16_t* dst = a.out + g * a.out_stride + row * H;
    for (int i = threadIdx.x; i < H; i += kThreads) dst[i] = tobf(bf(a.norm[g][i]) * (ys[i] * inv));
}
