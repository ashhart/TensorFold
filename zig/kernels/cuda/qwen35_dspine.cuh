// DSpine's code bridge and exact-sender table (cuda_dspine_frame.zig), rounding as Tencent's dspine_mlx port.
#pragma once

namespace tf_q35d {

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

// A top-k column back to a token id: columns past `split` are the tail span, `shift` ids further.
__device__ __forceinline__ int column_id(unsigned c, int split, int shift) { return split > 0 && (int)c >= split ? (int)c + shift : (int)c; }

}  // namespace tf_q35d

// RAW = bf16(H + ADD) (ADD may be null); RMS = sqrt(max(mean RAW^2, tiny)); XN = bf16(RAW / (RMS + beps)). Grid (rows).
extern "C" __global__ void __launch_bounds__(256) tf_q35d_bridge_read(const __nv_bfloat16* H, const __nv_bfloat16* ADD,
        __nv_bfloat16* RAW, __nv_bfloat16* XN, float* RMS, int D, float beps) {
    __shared__ float red[8];
    const size_t row = blockIdx.x;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = bf(H[row * D + i]);
        if (ADD != nullptr) x = rbf(x + bf(ADD[row * D + i]));
        RAW[row * D + i] = __float2bfloat16_rn(x);
        ss = fmaf(x, x, ss);
    }
    const float rms = sqrtf(fmaxf(tf_common::block_sum_serial(ss, red) / (float)D, 1.17549435e-38f));
    if (threadIdx.x == 0) RMS[row] = rms;
    for (int i = threadIdx.x; i < D; i += blockDim.x) XN[row * D + i] = __float2bfloat16_rn(bf(RAW[row * D + i]) / (rms + beps));
}

// Codes C wide: rows with row % every == 0 are bf16(sqrt_c * CB[IDS[row]]), the rest bf16(sqrt_c * P / (|P| + beps)).
extern "C" __global__ void __launch_bounds__(256) tf_q35d_bridge_code(const __nv_bfloat16* P, const __nv_bfloat16* CB,
        const unsigned* IDS, __nv_bfloat16* OUT, int C, int every, float sqrt_c, float beps) {
    __shared__ float red[8];
    const size_t row = blockIdx.x;
    if (row % every == 0) {
        const __nv_bfloat16* src = CB + (size_t)IDS[row] * C;
        for (int i = threadIdx.x; i < C; i += blockDim.x) OUT[row * C + i] = __float2bfloat16_rn(sqrt_c * bf(src[i]));
        return;
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) ss = fmaf(bf(P[row * C + i]), bf(P[row * C + i]), ss);
    const float n = sqrtf(tf_common::block_sum_serial(ss, red)) + beps;
    for (int i = threadIdx.x; i < C; i += blockDim.x) OUT[row * C + i] = __float2bfloat16_rn(sqrt_c * bf(P[row * C + i]) / n);
}

// Row t's message: 0 at a block's anchor, else bf16(sigmoid(Q[t] + S[t-1] + GB) * V[t-1]); SV rows are [S | V].
extern "C" __global__ void __launch_bounds__(256) tf_q35d_bridge_message(const __nv_bfloat16* Q, const __nv_bfloat16* SV,
        const __nv_bfloat16* GB, __nv_bfloat16* OUT, int R, int block_rows) {
    const size_t t = blockIdx.x;
    const bool anchor = t % block_rows == 0;
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        float m = 0.0f;
        if (!anchor) {
            const float g = sigmoid(bf(Q[t * R + i]) + bf(SV[(t - 1) * 2 * R + i]) + bf(GB[i]));
            m = g * bf(SV[(t - 1) * 2 * R + R + i]);
        }
        OUT[t * R + i] = __float2bfloat16_rn(m);
    }
}

// RES = RAW + RMS * DELTA (RAW at an anchor), XN = its norm times W; U set: anchor-free XN rows, then U = RAW * W.
extern "C" __global__ void __launch_bounds__(256) tf_q35d_bridge_write(const __nv_bfloat16* RAW, const __nv_bfloat16* DELTA,
        const float* RMS, const __nv_bfloat16* W, __nv_bfloat16* RES, __nv_bfloat16* XN, __nv_bfloat16* U, int D,
        int block_rows, int masked, float eps) {
    __shared__ float red[8];
    const size_t t = blockIdx.x;
    const bool anchor = t % block_rows == 0;
    if (U != nullptr && anchor) return;
    const float rms = RMS[t];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float raw = bf(RAW[t * D + i]);
        const float out = anchor ? raw : rbf(raw + rms * bf(DELTA[t * D + i]));
        RES[t * D + i] = __float2bfloat16_rn(out);
        ss = fmaf(out, out, ss);
    }
    const float inv = rsqrtf(tf_common::block_sum_serial(ss, red) / (float)D + eps);
    const size_t dst = U == nullptr ? t : (t / block_rows) * (block_rows - 1) + t % block_rows - 1;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        XN[dst * D + i] = __float2bfloat16_rn(rbf(bf(RES[t * D + i]) * inv) * bf(W[i]));
        if (U != nullptr) U[((size_t)masked + dst) * D + i] = __float2bfloat16_rn(bf(RAW[t * D + i]) * bf(W[i]));
    }
}

// Each head row's K senders: its stream's anchor for the block's first row, else the row before's candidates (ids).
extern "C" __global__ void tf_q35d_senders(const unsigned* WORDS, const unsigned* IDS, unsigned* SIDS, int K, int block_rows,
        int lp_words, int split, int shift) {
    const int row = blockIdx.x, j = threadIdx.x;
    if (j >= K) return;
    const int stream = row / (block_rows - 1), r = row % (block_rows - 1);
    SIDS[row * K + j] = r == 0 ? IDS[stream * block_rows] : (unsigned)column_id(WORDS[(size_t)(row - 1) * lp_words + 2 + j], split, shift);
}

// Sender p = (row, j') of a head row: bf16(sigmoid(Q[row's block row] + S[p] + GB) * V[p]); SV rows are [S | V].
extern "C" __global__ void __launch_bounds__(256) tf_q35d_sender_message(const __nv_bfloat16* Q, const __nv_bfloat16* SV,
        const __nv_bfloat16* GB, __nv_bfloat16* OUT, int R, int K, int block_rows) {
    const size_t p = blockIdx.x;
    const int row = (int)(p / K), stream = row / (block_rows - 1);
    const size_t t = (size_t)stream * block_rows + 1 + row % (block_rows - 1);
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        const float g = sigmoid(bf(Q[t * R + i]) + bf(SV[p * 2 * R + i]) + bf(GB[i]));
        OUT[p * R + i] = __float2bfloat16_rn(g * bf(SV[p * 2 * R + R + i]));
    }
}

// OUT[p = (row, j')][j] = (base_j + rms * M[p] . TG[id_j]) / rms(RAW + rms * WM[p]); base: LOGITS row masked + row.
extern "C" __global__ void __launch_bounds__(256) tf_q35d_table(const __nv_bfloat16* RAW, const float* RMS, const __nv_bfloat16* M,
        const __nv_bfloat16* WM, const __nv_bfloat16* TG, const __nv_bfloat16* LOGITS, const unsigned* WORDS, float* OUT, int D,
        int R, int K, int block_rows, int masked, int cols, int lp_words, int split, int shift, float eps) {
    __shared__ float red[8];
    __shared__ float dots[32];
    const size_t p = blockIdx.x;
    const int row = (int)(p / K), stream = row / (block_rows - 1);
    const size_t t = (size_t)stream * block_rows + 1 + row % (block_rows - 1);
    const float rms = RMS[t];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float y = bf(RAW[t * D + i]) + rms * bf(WM[p * D + i]);
        ss = fmaf(y, y, ss);
    }
    const float inv = rsqrtf(tf_common::block_sum_serial(ss, red) / (float)D + eps);
    const unsigned* w = WORDS + (size_t)row * lp_words + 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int j = warp; j < K; j += warps) {
        const __nv_bfloat16* tg = TG + (size_t)column_id(w[j], split, shift) * R;
        float s = 0.0f;
        for (int k = lane; k < R; k += 32) s = fmaf(bf(M[p * R + k]), bf(tg[k]), s);
        s = warp_sum(s);
        if (lane == 0) dots[j] = s;
    }
    __syncthreads();
    if (threadIdx.x < K) {
        const float base = bf(LOGITS[((size_t)masked + row) * cols + w[threadIdx.x]]);
        OUT[p * K + threadIdx.x] = (base + rms * dots[threadIdx.x]) * inv;
    }
}

// OUT (V, R) = IN (R, V) transposed, bf16; grid (V / 32, R / 32) of 32 x 8 threads.
extern "C" __global__ void tf_q35d_transpose(const __nv_bfloat16* IN, __nv_bfloat16* OUT, int R, int V) {
    __shared__ __nv_bfloat16 tile[32][33];
    const int v0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int r = r0 + y, v = v0 + threadIdx.x;
        if (r < R && v < V) tile[y][threadIdx.x] = IN[(size_t)r * V + v];
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int v = v0 + y, r = r0 + threadIdx.x;
        if (r < R && v < V) OUT[(size_t)v * R + r] = tile[threadIdx.x][y];
    }
}
