// SPDX-License-Identifier: Apache-2.0

// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Chunked Gated DeltaNet for long prompts after FLA's chunk_gated_delta_rule, in five kernels: prep, kt, wy, h, o.

// fp16 operands and fp32 accumulation through v_dot2_f32_f16; the state and every sum stay fp32.

// Shared-memory rows are 16-byte aligned and four banks apart: a 16-byte read of eight consecutive rows is one pass.

#include <hip/hip_runtime.h>


namespace {

using half_t = _Float16;
using half2_t = _Float16 __attribute__((ext_vector_type(2)));

constexpr int kBT = 64;          // chunk
constexpr int kD = 128;          // head size, keys and values
constexpr int kThreads = 256;
constexpr int kBV = 16;          // value rows of a state block
constexpr int kLd = kD + 8;      // a 128-wide fp16 row in shared memory
constexpr int kTLd = kBT + 8;    // a 64-wide one
constexpr int kTile = kBT * kD;  // a chunk's tile, in values

struct alignas(16) Quad {
    half2_t p[4];
};

__device__ __forceinline__ float dot2(half2_t a, half2_t b, float c) { return __builtin_amdgcn_fdot2(a, b, c, false); }
__device__ __forceinline__ Quad ld_quad(const half_t* p) { return *reinterpret_cast<const Quad*>(p); }
__device__ __forceinline__ Quad zero_quad() {
    Quad z;
    for (int i = 0; i < 4; ++i) z.p[i] = half2_t{static_cast<half_t>(0), static_cast<half_t>(0)};
    return z;
}

// A barrier for data that moved through shared memory only: the global stores in flight need not land first.
__device__ __forceinline__ void lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// x from the lane whose index differs in bit 0, or in bit 1 (within a quad).
__device__ __forceinline__ float quad_xor1(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0xB1, 0xF, 0xF, true));
}
__device__ __forceinline__ float quad_xor2(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0x4E, 0xF, 0xF, true));
}

// acc[r][c] += (row r of a) . (row c of b) over the 8-wide groups from g_first by g_stride below g_end, for c < NC.
template <int NC>
__device__ __forceinline__ void tile_acc(float (&acc)[4][4], const half_t* a, int lda, const half_t* b, int b_step,
                                         int g_first, int g_end, int g_stride) {
#pragma unroll 2
    for (int g = g_first; g < g_end; g += g_stride) {
        Quad x[4], y[NC];
#pragma unroll
        for (int r = 0; r < 4; ++r) x[r] = ld_quad(a + r * lda + 8 * g);
#pragma unroll
        for (int c = 0; c < NC; ++c) y[c] = ld_quad(b + c * b_step + 8 * g);
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < NC; ++c)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[r][c] = dot2(x[r].p[i], y[c].p[i], acc[r][c]);
    }
}

// The same over every group with `nc` (1 to 4, the same for the whole wave) columns.
__device__ __forceinline__ void tile_acc_wave(float (&acc)[4][4], const half_t* a, int lda, const half_t* b, int b_step,
                                              int nc) {
    switch (nc) {
        case 1: tile_acc<1>(acc, a, lda, b, b_step, 0, kD / 8, 1); break;
        case 2: tile_acc<2>(acc, a, lda, b, b_step, 0, kD / 8, 1); break;
        case 3: tile_acc<3>(acc, a, lda, b, b_step, 0, kD / 8, 1); break;
        default: tile_acc<4>(acc, a, lda, b, b_step, 0, kD / 8, 1); break;
    }
}

// A chunk tile as the 16-byte pieces a thread fetches, rows from `n` on zero, stored into shared rows of kLd.
struct Tile {
    Quad piece[4];

    __device__ __forceinline__ void fetch(const half_t* src, int n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = threadIdx.x + j * kThreads;
            piece[j] = (idx >> 4) < n ? ld_quad(src + idx * 8) : zero_quad();
        }
    }

    __device__ __forceinline__ void store(half_t* dst) const {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = threadIdx.x + j * kThreads;
            *reinterpret_cast<Quad*>(&dst[(idx >> 4) * kLd + 8 * (idx & 15)]) = piece[j];
        }
    }
};

}  // namespace

// q, k (key heads) and v (value heads) to fp16 tiles [chunk][head][64][128].
extern "C" __global__ void __launch_bounds__(kThreads)
tf_gdn_prep(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
            half_t* __restrict__ qh, half_t* __restrict__ kh, half_t* __restrict__ vh, int length, int key_heads,
            int value_heads) {
    const long long id = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long nk = static_cast<long long>(length) * key_heads * (kD / 4);
    const long long nv = static_cast<long long>(length) * value_heads * (kD / 4);
    if (id < nk + nv) {
        const bool is_v = id >= nk;
        const long long at = is_v ? id - nk : id;
        const int heads = is_v ? value_heads : key_heads;
        const int d4 = static_cast<int>(at & 31);
        const long long row = at >> 5;  // t * heads + head
        const int head = static_cast<int>(row % heads);
        const long long t = row / heads;
        const long long out = (((t >> 6) * heads + head) * kBT + (t & 63)) * kD + 4 * d4;
        auto put = [&](const float* src, half_t* dst) {
            const float4 a = reinterpret_cast<const float4*>(src)[at];
            half2_t* o = reinterpret_cast<half2_t*>(dst + out);
            o[0] = half2_t{static_cast<half_t>(a.x), static_cast<half_t>(a.y)};
            o[1] = half2_t{static_cast<half_t>(a.z), static_cast<half_t>(a.w)};
        };
        if (is_v) {
            put(v, vh);
        } else {
            put(q, qh);
            put(k, kh);
        }
    }
}

// kt[chunk][key head][key][token]: each chunk's keys transposed, tokens past the end zero.
extern "C" __global__ void __launch_bounds__(kThreads)
tf_gdn_kt(const half_t* __restrict__ kh, half_t* __restrict__ kt, int length, int key_heads) {
    __shared__ alignas(16) half_t sT[kD * kTLd];
    const int tid = threadIdx.x;
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int n = min(kBT, length - chunk * kBT);
    const half_t* tile = kh + (static_cast<long long>(chunk) * key_heads + head) * kTile;
    for (int idx = tid; idx < kBT * (kD / 8); idx += kThreads) {
        const int t = idx & 63, cg = idx >> 6;
        Quad x = zero_quad();
        if (t < n) x = ld_quad(tile + t * kD + 8 * cg);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sT[(8 * cg + 2 * i) * kTLd + t] = x.p[i][0];
            sT[(8 * cg + 2 * i + 1) * kTLd + t] = x.p[i][1];
        }
    }
    lds_sync();
    half_t* out = kt + (static_cast<long long>(chunk) * key_heads + head) * kD * kBT;
    for (int idx = tid; idx < kD * (kBT / 8); idx += kThreads) {
        const int key = idx >> 3, c8 = idx & 7;
        *reinterpret_cast<Quad*>(&out[key * kBT + 8 * c8]) = ld_quad(&sT[key * kTLd + 8 * c8]);
    }
}

extern "C" __global__ void __launch_bounds__(kThreads)
tf_gdn_wy(const half_t* __restrict__ kh, const half_t* __restrict__ vh, const float* __restrict__ beta,
          const float* __restrict__ gate, float* __restrict__ gc, half_t* __restrict__ w, half_t* __restrict__ u, int length, int key_heads,
          int value_heads) {
    __shared__ alignas(16) half_t sK[kBT * kLd];
    __shared__ alignas(16) float sA[kBT * kBT];
    __shared__ float sGc[kBT];
    __shared__ float sBeta[kBT];
    __shared__ float sBe[kBT];
    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int head = blockIdx.y;
    const int chunk = blockIdx.x;
    const int t0 = chunk * kBT;
    const int n = min(kBT, length - t0);
    const int key_head = head / (value_heads / key_heads);
    const long long vtile = (static_cast<long long>(chunk) * value_heads + head) * kTile;

    Tile rows;
    rows.fetch(kh + (static_cast<long long>(chunk) * key_heads + key_head) * kTile, n);
    float rhs[kBT];
    const bool is_w = tid < kD;
    const int col = tid & (kD - 1);
    if (!is_w) {
#pragma unroll
        for (int i = 0; i < kBT; ++i) rhs[i] = i < n ? static_cast<float>(vh[vtile + i * kD + col]) : 0.f;
    }
    rows.store(sK);
    if (tid < 32) {
        // the chunk's running sum of log gate (a gate of 0 clamped), two tokens a lane
        const long long at0 = static_cast<long long>(t0 + tid) * value_heads + head;
        const long long at1 = at0 + 32LL * value_heads;
        float a = tid < n ? logf(fmaxf(gate[at0], 1e-30f)) : 0.f;
        float b = tid + 32 < n ? logf(fmaxf(gate[at1], 1e-30f)) : 0.f;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const float xa = __shfl_up(a, d, 32);
            const float xb = __shfl_up(b, d, 32);
            if (tid >= d) {
                a += xa;
                b += xb;
            }
        }
        b += __shfl(a, 31, 32);
        sGc[tid] = a;
        sGc[tid + 32] = b;
        if (tid < n) gc[at0] = a;
        if (tid + 32 < n) gc[at1] = b;
    } else if (tid < 96) {
        const int i = tid - 32;
        sBeta[i] = i < n ? beta[static_cast<long long>(t0 + i) * value_heads + head] : 0.f;
    }
    lds_sync();
    if (tid < kBT) sBe[tid] = sBeta[tid] * __expf(sGc[tid]);

    // A: each thread a 4 x 4 tile of rows 4 ti .. and columns tj + 16 c, the columns past the wave's last row skipped
    {
        const int ti = tid >> 4, tj = tid & 15;
        const int nc = (8 * wave + 7) / 16 + 1;
        float acc[4][4] = {};
        tile_acc_wave(acc, sK + ti * 4 * kLd, kLd, sK + tj * kLd, 16 * kLd, nc);
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int i = ti * 4 + r, j = tj + 16 * c;
                sA[i * kBT + j] = i > j ? sBeta[i] * __expf(sGc[i] - sGc[j]) * acc[r][c] : 0.f;
            }
    }
    lds_sync();
    if (is_w) {
#pragma unroll
        for (int i = 0; i < kBT; ++i) rhs[i] = i < n ? sBe[i] * static_cast<float>(sK[i * kLd + col]) : 0.f;
    } else {
#pragma unroll
        for (int i = 0; i < kBT; ++i) rhs[i] *= sBeta[i];
    }

    // (I + A) x = rhs down the rows, a column a thread: the first 128 threads w, the rest u
    half_t* ocol = (is_w ? w : u) + vtile + col;
#pragma unroll
    for (int i = 0; i < kBT; ++i) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int m = 0;
#pragma unroll
        for (; m + 4 <= i; m += 4) {
            const float4 a = *reinterpret_cast<const float4*>(&sA[i * kBT + m]);
            s0 = fmaf(a.x, rhs[m], s0);
            s1 = fmaf(a.y, rhs[m + 1], s1);
            s2 = fmaf(a.z, rhs[m + 2], s2);
            s3 = fmaf(a.w, rhs[m + 3], s3);
        }
#pragma unroll
        for (; m < i; ++m) s0 = fmaf(sA[i * kBT + m], rhs[m], s0);
        rhs[i] -= (s0 + s1) + (s2 + s3);
        if (i < n) ocol[i * kD] = static_cast<half_t>(rhs[i]);
    }
}

// vt[chunk][head][value][token] gets v_new = u - w H^T; hbuf[chunk][head][value][key] each chunk's start state (fp16).
extern "C" __global__ void __launch_bounds__(kThreads)
tf_gdn_h(const half_t* __restrict__ kt, const half_t* __restrict__ w, const half_t* __restrict__ u,
         half_t* __restrict__ vt, const float* __restrict__ gc, float* __restrict__ state, half_t* __restrict__ hbuf,
         int length, int key_heads, int value_heads) {
    __shared__ alignas(16) half_t sH[kBV * kLd];
    __shared__ alignas(16) half_t sW[kBT * kLd];
    __shared__ alignas(16) half_t sKT[kD * kTLd];
    __shared__ alignas(16) half_t sVg[kBV * kTLd];
    __shared__ float sDecay[kBT];
    const int tid = threadIdx.x;
    const int v0 = blockIdx.x * kBV;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const int chunks = (length + kBT - 1) / kBT;
    // the state: lane pairs share a (4 value rows, 4 keys) tile and both hold it
    const int vq = tid >> 6, kg = (tid >> 1) & 31, half_id = tid & 1;
    // the correction: a (4 tokens, 4 values) tile split over four lanes by key group
    const int ks = tid & 3, ti = tid >> 4, vj = (tid >> 2) & 3;

    float h[4][4];
    float* st = state + (static_cast<long long>(head) * kD + v0 + 4 * vq) * kD + kg;
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) h[r][c] = st[r * kD + 32 * c];

    Tile wt;
    Quad ktp[4];
    half_t up[4];
    float g_next = 0.f, d_next = 0.f;
    const int t_mine = 4 * ti + ks;
    auto fetch_u = [&](int chunk) {
        const int n = min(kBT, length - chunk * kBT);
        const half_t* src = u + (static_cast<long long>(chunk) * value_heads + head) * kTile + t_mine * kD + v0 + vj;
#pragma unroll
        for (int c = 0; c < 4; ++c) up[c] = t_mine < n ? src[4 * c] : static_cast<half_t>(0);
    };
    auto fetch_kt = [&](int chunk) {
        const half_t* src = kt + (static_cast<long long>(chunk) * key_heads + key_head) * kD * kBT;
#pragma unroll
        for (int j = 0; j < 4; ++j) ktp[j] = ld_quad(src + (tid + j * kThreads) * 8);
    };
    auto fetch_gc = [&](int chunk) {
        const int t0 = chunk * kBT;
        const int n = min(kBT, length - t0);
        g_next = gc[static_cast<long long>(t0 + n - 1) * value_heads + head];
        d_next = tid < n ? gc[static_cast<long long>(t0 + tid) * value_heads + head] : 0.f;
    };
    auto fetch_w = [&](int chunk) {
        wt.fetch(w + (static_cast<long long>(chunk) * value_heads + head) * kTile, min(kBT, length - chunk * kBT));
    };
    fetch_w(0);
    fetch_kt(0);
    fetch_u(0);
    fetch_gc(0);

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int n = min(kBT, length - chunk * kBT);
        const float g_last = g_next;
        // the start state: to shared memory for the correction, and out for o
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c)
                if ((r >> 1) == half_id) sH[(4 * vq + r) * kLd + kg + 32 * c] = static_cast<half_t>(h[r][c]);
        wt.store(sW);
        if (tid < kBT) sDecay[tid] = tid < n ? __expf(g_last - d_next) : 0.f;
        if (chunk + 1 < chunks) {
            fetch_w(chunk + 1);
            fetch_gc(chunk + 1);
        }
        lds_sync();

        // the keys transposed for the update, the start state out
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = tid + j * kThreads;
            *reinterpret_cast<Quad*>(&sKT[(idx >> 3) * kTLd + 8 * (idx & 7)]) = ktp[j];
        }
        if (chunk + 1 < chunks) fetch_kt(chunk + 1);
        {
            const int row = tid >> 4, cg = tid & 15;
            *reinterpret_cast<Quad*>(
                &hbuf[((static_cast<long long>(chunk) * value_heads + head) * kD + v0 + row) * kD + 8 * cg]) =
                ld_quad(&sH[row * kLd + 8 * cg]);
        }

        // v_new[t][v] = u - sum_k w[t][k] H[v][k]: 4 x 4 tiles, the key groups split over four lanes
        {
            float acc[4][4] = {};
            tile_acc<4>(acc, sW + ti * 4 * kLd, kLd, sH + vj * kLd, 4 * kLd, ks, kD / 8, 4);
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[r][c] += quad_xor1(acc[r][c]);
                    acc[r][c] += quad_xor2(acc[r][c]);
                }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float mine = ks == 0 ? acc[0][c] : ks == 1 ? acc[1][c] : ks == 2 ? acc[2][c] : acc[3][c];
                const float vn = t_mine < n ? static_cast<float>(up[c]) - mine : 0.f;
                const int v = vj + 4 * c;
                vt[((static_cast<long long>(chunk) * value_heads + head) * kD + v0 + v) * kBT + t_mine] =
                    static_cast<half_t>(vn);
                sVg[v * kTLd + t_mine] = static_cast<half_t>(vn * sDecay[t_mine]);
            }
        }
        if (chunk + 1 < chunks) fetch_u(chunk + 1);
        lds_sync();

        // H = exp(g_last) H + sum_t vg[t][v] kT[key][t]: the tokens split over a lane pair
        {
            float acc[4][4] = {};
            tile_acc<4>(acc, sVg + 4 * vq * kTLd, kTLd, sKT + kg * kTLd, 32 * kTLd, 4 * half_id, 4 * half_id + 4, 1);
            const float decay = __expf(g_last);
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) h[r][c] = h[r][c] * decay + (acc[r][c] + quad_xor1(acc[r][c]));
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if ((r >> 1) == half_id) st[r * kD + 32 * c] = h[r][c];
}

// A block a (chunk, head): the scores once, then each half of the values (the next half's tiles fetched meanwhile).
extern "C" __global__ void __launch_bounds__(kThreads)
tf_gdn_o(const half_t* __restrict__ qh, const half_t* __restrict__ kh, const half_t* __restrict__ vt,
         const half_t* __restrict__ hbuf, const float* __restrict__ gc, float* __restrict__ y, int length,
         int key_heads, int value_heads) {
    __shared__ alignas(16) half_t sQ[kBT * kLd];
    __shared__ alignas(16) half_t sK[kBT * kLd];  // the keys, then the start state's 64 value rows
    __shared__ alignas(16) half_t sS[kBT * kTLd];
    __shared__ alignas(16) half_t sVT[kBT * kTLd];
    __shared__ float sGc[kBT];
    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int t0 = chunk * kBT;
    const int n = min(kBT, length - t0);
    const int key_head = head / (value_heads / key_heads);
    const int ti = tid >> 4, tj = tid & 15;

    const half_t* hsrc = hbuf + (static_cast<long long>(chunk) * value_heads + head) * kD * kD;
    const half_t* vsrc = vt + (static_cast<long long>(chunk) * value_heads + head) * kD * kBT;
    Tile tq, tk, th;
    Quad pv[2];
    auto fetch_half = [&](int half) {
#pragma unroll
        for (int j = 0; j < 4; ++j) th.piece[j] = ld_quad(hsrc + half * 64 * kD + (tid + j * kThreads) * 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) pv[j] = ld_quad(vsrc + half * 64 * kBT + (tid + j * kThreads) * 8);
    };
    const long long ktile = (static_cast<long long>(chunk) * key_heads + key_head) * kTile;
    tq.fetch(qh + ktile, n);
    tk.fetch(kh + ktile, n);
    fetch_half(0);
    if (tid < kBT) sGc[tid] = tid < n ? gc[static_cast<long long>(t0 + tid) * value_heads + head] : 0.f;
    tq.store(sQ);
    tk.store(sK);
    lds_sync();

    // the masked, decayed scores q_t.k_j: a 4 x 4 tile of rows 4 ti .. and columns tj + 16 c
    {
        const int nc = (8 * wave + 7) / 16 + 1;
        float acc[4][4] = {};
        tile_acc_wave(acc, sQ + ti * 4 * kLd, kLd, sK + tj * kLd, 16 * kLd, nc);
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int i = ti * 4 + r, j = tj + 16 * c;
                sS[i * kTLd + j] = static_cast<half_t>(j <= i ? acc[r][c] * __expf(sGc[i] - sGc[j]) : 0.f);
            }
    }
    lds_sync();

    for (int half = 0; half < 2; ++half) {
        // the start state over the keys (reusing the key tile) and the new values by token, then the next half's
        th.store(sK);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + j * kThreads;
            *reinterpret_cast<Quad*>(&sVT[(idx >> 3) * kTLd + 8 * (idx & 7)]) = pv[j];
        }
        if (half == 0) fetch_half(1);
        lds_sync();

        float inter[4][4] = {};
        float intra[4][4] = {};
        tile_acc<4>(inter, sQ + ti * 4 * kLd, kLd, sK + tj * kLd, 16 * kLd, 0, kD / 8, 1);
        tile_acc<4>(intra, sS + ti * 4 * kTLd, kTLd, sVT + tj * kTLd, 16 * kTLd, 0, ti / 2 + 1, 1);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int t = ti * 4 + r;
            if (t >= n) continue;
            const float e = __expf(sGc[t]);
            float* row = y + (static_cast<long long>(t0 + t) * value_heads + head) * kD + half * 64 + tj;
#pragma unroll
            for (int c = 0; c < 4; ++c) row[16 * c] = e * inter[r][c] + intra[r][c];
        }
        if (half == 0) lds_sync();
    }
}
