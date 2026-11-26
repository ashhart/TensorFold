// The LiLiCorr head/lattice kernels and the 16f row-batched GEMVs: this image's second responsibility,
// split from qwen35_draft.cu for the lean rule. Included at the same spot the text sat at, so the
// translation unit's preprocessed body is unchanged (same symbols, same image name: qwen35_draft).
// K-rms-f32 kernel: grid (1), block (1024) — the anchor norm (f32 in, f32 out; the anchor's
// rms stays f32 in the reference). W is the f32-staged per-channel scale.
// Args: (const float* X, const float* W, float* OUT, int d, float eps)
extern "C" __global__ void tf_lilicorr_rms_f32(const float* __restrict__ X, const float* __restrict__ W,
                                                float* __restrict__ OUT, int d, float eps) {
    const int tid = threadIdx.x;

    __shared__ float red[32];

    float local = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) local += X[i] * X[i];

    const float ssq = tf_lilicorr_block_sum(local, red);
    const float inv = 1.0f / sqrtf(ssq / (float)d + eps);

    for (int i = tid; i < d; i += blockDim.x) OUT[i] = X[i] * inv * W[i];
}

// K-widen kernel: grid (ceil(n/1024)), block (1024) — bf16 bits widened to f32 (the factor
// kernel's A_IN is f32; the layer chain's rms output is bf16).
// Args: (const __nv_bfloat16* IN, float* OUT, int n)
extern "C" __global__ void tf_lilicorr_widen(const __nv_bfloat16* __restrict__ IN, float* __restrict__ OUT, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OUT[i] = bf(IN[i]);
}

// K-narrow kernel: grid (ceil(n/1024)), block (1024) — f32 narrowed to bf16 bits, round-to-
// nearest-even (the anchor_out_head's input bits).
// Args: (const float* IN, __nv_bfloat16* OUT, int n)
extern "C" __global__ void tf_lilicorr_narrow(const float* __restrict__ IN, __nv_bfloat16* __restrict__ OUT, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OUT[i] = __float2bfloat16_rn(IN[i]);
}

// Factor block kernel: computes the factor hidden representation
// Grid (56 rows), block (1024 = one thread per output channel t).
// Thread t of row r computes:
// v = dot(A_IN[r], W + t*3072) + dot(ANCHOR, W + t*3072 + 1024) + dot_prod + B[t]
// where dot_prod = sum_i A_IN[r*1024+i] * ANCHOR[i] * W[t*3072 + 2048 + i]
// then OUT[r*1024+t] = bf16_bits(silu(v))
// Args: (const float* A_IN, const float* ANCHOR, const float* W, const float* B,
// __nv_bfloat16* OUT, int rows_unused_keep_shape, int hd).
extern "C" __global__ void tf_lilicorr_factor(const float* __restrict__ A_IN, const float* __restrict__ ANCHOR,
                                              const float* __restrict__ W, const float* __restrict__ B,
                                              __nv_bfloat16* __restrict__ OUT, int rows_unused_keep_shape, int hd) {
    const int t = threadIdx.x;
    if (t >= hd) return;

    const int r = blockIdx.x;
    if (r >= rows_unused_keep_shape) return;

    // Compute the main dot product
    float v = B[t];
    const float* w_row = W + t * 3 * hd;

    // First part: A_IN[r] * W[t]
    for (int i = 0; i < hd; i++) {
        v += A_IN[r * hd + i] * w_row[i];
    }

    // Second part: ANCHOR * W[t + 1024]
    for (int i = 0; i < hd; i++) {
        v += ANCHOR[i] * w_row[hd + i];
    }

    // Third part: dot_prod = sum_i A_IN[r*1024+i] * ANCHOR[i] * W[t*3072 + 2048 + i]
    const float* w_cross = w_row + 2 * hd;
    for (int i = 0; i < hd; i++) {
        v += A_IN[r * hd + i] * ANCHOR[i] * w_cross[i];
    }

    // Apply silu activation
    const float silu_val = v / (1.0f + expf(-v));

    // Convert to bf16
    OUT[r * hd + t] = __float2bfloat16_rn(silu_val);
}

// Extended token projection kernel with stride loop for both fused edge head and anchor head
// Grid (rows), block (1024) - identical behavior at out_dim<=1024
// Per-thread output channels with a stride loop over out_dim (block 1024, any out_dim):
// OUT[r, t] = bf16widen(EMB[r]) . W[t] + B[t] — the S2 projection at any output width.
extern "C" __global__ void tf_token_proj_extended(const __nv_bfloat16* __restrict__ EMB, const float* __restrict__ W,
                                                  const float* __restrict__ B, float* __restrict__ OUT,
                                                  int in_dim, int out_dim) {
    const int r = blockIdx.x;
    for (int t = threadIdx.x; t < out_dim; t += blockDim.x) {
        const float* w = W + (size_t)t * in_dim;
        const __nv_bfloat16* row = EMB + (size_t)r * in_dim;
        float acc = 0.f;
        for (int i = 0; i < in_dim; ++i) acc += bf(row[i]) * w[i];
        OUT[(size_t)r * out_dim + t] = acc + B[t];
    }
}

// L2 normalization kernel for edges and anchor
// Grid (57), block (1024). Block 56 = row r of the fused edges [56, 2048]:
// threads sum squares of their strided elements PER HALF (first 1024 / last 1024 are separate vectors),
// then normalize both halves in place with x / max(sqrtf(sum), eps).
// Block 56 (blockIdx.x == 56) = the anchor_out row [1024]: one half only, same math.
// In-place L2 normalization with an eps floor: blocks 0..rows-1 = edges rows (stride 2048, TWO
// independent 1024 halves); block rows = the anchor tail (one 1024 vector at V + rows*stride).
extern "C" __global__ void tf_lilicorr_l2norm(float* V, int rows, int stride, int tail_width, float eps) {
    const int idx = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float red[32];
    const bool edge_row = bid < rows;
    // edge rows: two halves of `stride` f32 each, normalized separately (width = stride/2, x2).
    // the anchor tail: exactly `tail_width` f32 — the S4-era kernel used `stride` here and
    // over-ran the tail's hd elements into whatever sat behind it (reads AND the `*= inv`
    // write; on the live path that was the PREVIOUS cycle's start/pair scores, so the tail's
    // inv carried their stale ssq — a real normalization defect, not just an over-read).
    const int width = edge_row ? stride / 2 : tail_width;
    float* base = V + (size_t)(edge_row ? bid : rows) * stride;
    const int nvecs = edge_row ? 2 : 1;
    for (int v = 0; v < nvecs; ++v) {
        float* dst = base + (size_t)v * width;
        float local = 0.f;
        for (int i = idx; i < width; i += blockDim.x) local += dst[i] * dst[i];
        const float total = tf_lilicorr_block_sum(local, red);
        const float inv = 1.0f / fmaxf(sqrtf(total), eps);
        for (int i = idx; i < width; i += blockDim.x) dst[i] *= inv;
        __syncthreads();
    }
}

// Dot product kernel for start and pair scores
// Grid (8 + 6*8*8 = 392), block (256, shared-mem reduce).
// Block b < 8: start[b] = dot(ANCHOR_OUT, IN_VEC + b*1024) over 1024 dims.
// Block b >= 8: decode k = b - 8; s = k / 64; i = (k / 8) % 8; j = k % 8;
// pair[k] = dot(OUT_VEC + (s*8+i)*1024, IN_VEC + ((s+1)*8+j)*1024).
// Dot products: blocks 0..7 = start[b] = dot(ANCHOR_OUT, IN_VEC + b*1024); blocks 8.. = pair
// scores decoded from k (s = k/64, i = (k/8)%8, j = k%8): dot(OUT_VEC + (s*8+i)*1024, IN_VEC + ...).
// ANCHOR_OUT/OUT_VEC/IN_VEC are BASE pointers — the caller passes the in-half offset in IN_VEC.
// Dot products: blocks 0..7 = start[b] = dot(ANCHOR_OUT, IN_VEC + b*stride); blocks 8.. = pair
// scores decoded from k (s = k/64, i = (k/8)%8, j = k%8): dot(OUT_VEC + (s*8+i)*stride, IN_VEC + ...).
// OUT_VEC/IN_VEC are VIEWS INTO the interleaved edges [56, 2048] (out-half at +0, in-half at +1024):
// every row index moves by `stride`, never 1024 (the S4 writer bug — packed-stride indexing read
// the wrong row entirely and the gate refused with pair 6.3e-1).
extern "C" __global__ void tf_lilicorr_dots(const float* __restrict__ ANCHOR_OUT, const float* __restrict__ OUT_VEC,
                                            const float* __restrict__ IN_VEC, float* __restrict__ START, float* __restrict__ PAIR,
                                            int stride) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float red[32];
    float local = 0.f;
    if (b < 8) {
        for (int i = tid; i < 1024; i += blockDim.x) local += ANCHOR_OUT[i] * IN_VEC[b * stride + i];
        const float total = tf_lilicorr_block_sum(local, red);
        if (tid == 0) START[b] = total;
    } else {
        const int k = b - 8;
        const int s = k / 64;
        const int i = (k / 8) % 8;
        const int j = k % 8;
        for (int t = tid; t < 1024; t += blockDim.x) local += OUT_VEC[(s * 8 + i) * stride + t] * IN_VEC[((s + 1) * 8 + j) * stride + t];
        const float total = tf_lilicorr_block_sum(local, red);
        if (tid == 0) PAIR[k] = total;
    }
}

// Lattice attention kernel: SDPA per head over the whole lattice (56 rows, 8 heads).
// Grid (56*8) — blockIdx.x = qi*8+h (query qi, head h); block (128 threads).
// Phase 1: threads 0..55 compute score[ki] = (q[qi,h] . k[ki,h]) * scale + bias[h, qi, ki],
// into __shared__ float p[64]. q at QKV + qi*3072 + h*128; k at QKV + ki*3072 + 1024 + h*128;
// dot over 128 dims. scale = 1.0f / sqrtf(128.0f). Use expf (precise), never __expf.
// Phase 2 (thread 0): softmax over the 56 scores (max-subtract, exp, sum, divide each by sum),
// write back to shared.
// Phase 3: threads 0..127: att[qi*1024 + h*128 + t] = sum over ki of p[ki] * QKV[ki*3072 + 2048 + h*128 + t].
// All math f32. __syncthreads() between phases.
extern "C" __global__ void tf_lilicorr_lat_attn(const float* __restrict__ qkv, const float* __restrict__ bias, float* __restrict__ out,
                                                int lattice, int hd, int head_dim) {
    const int qi = blockIdx.x / 8;
    const int h = blockIdx.x % 8;

    // Shared memory for attention scores
    __shared__ float p[56];

    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute attention scores
    for (int ki = threadIdx.x; ki < lattice; ki += blockDim.x) {
        float acc = 0.0f;
        const int q_offset = qi * 3 * hd + h * head_dim;
        const int k_offset = ki * 3 * hd + hd + h * head_dim;

        for (int t = 0; t < head_dim; t++) {
            acc += qkv[q_offset + t] * qkv[k_offset + t];
        }
        const float score = acc * scale + bias[(h * lattice + qi) * lattice + ki];
        p[ki] = score;
    }

    __syncthreads();

    // Phase 2: softmax
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int i = 0; i < lattice; i++) {
            if (p[i] > mx) mx = p[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < lattice; i++) {
            p[i] = expf(p[i] - mx);
            sum += p[i];
        }

        for (int i = 0; i < lattice; i++) {
            p[i] /= sum;
        }
    }

    __syncthreads();

    // Phase 3: compute output
    if (threadIdx.x < head_dim) {
        const int t = threadIdx.x;
        float acc = 0.0f;
        const int v_offset = t;

        for (int ki = 0; ki < lattice; ki++) {
            const int v_offset_ki = ki * 3 * hd + 2 * hd + h * head_dim + t;
            acc += p[ki] * qkv[v_offset_ki];
        }

        const int out_offset = qi * hd + h * head_dim + t;
        out[out_offset] = acc;
    }
}

// K-rms kernel: grid (rows), block (1024). Per row: block_sum of squares over d elements (bf16 IN, widened in-kernel),
// inv = 1/sqrtf(ssq/d + eps), out bf16: OUT[r*d+i] = __float2bfloat16_rn(bf(IN[r*d+i]) * inv * bf(W[i]))
// Args: (const __nv_bfloat16* IN, const float* W, __nv_bfloat16* OUT, int d, float eps)
extern "C" __global__ void tf_lilicorr_rms(const __nv_bfloat16* __restrict__ IN, const float* __restrict__ W,
                                            __nv_bfloat16* __restrict__ OUT, int d, float eps) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float red[32];

    // Per row: compute block sum of squares
    float local = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        const float x = bf(IN[r * d + i]);
        local += x * x;
    }

    const float ssq = tf_lilicorr_block_sum(local, red);
    const float inv = 1.0f / sqrtf(ssq / (float)d + eps);

    // Element-wise output
    for (int i = tid; i < d; i += blockDim.x) {
        const float x = bf(IN[r * d + i]) * inv * W[i];
        OUT[r * d + i] = __float2bfloat16_rn(x);
    }
}

// K-lin-f32 kernel: EXACTLY tf_token_proj_extended's shape but the input is f32 (the attention's raw output feeds out_proj uncast).
// Grid (rows), block (1024), stride loop over out_dim.
// Args: (const float* IN, const float* W, const float* B, float* OUT, int in_dim, int out_dim)
extern "C" __global__ void tf_lilicorr_lin_f32(const float* __restrict__ IN, const float* __restrict__ W,
                                                const float* __restrict__ B, float* __restrict__ OUT,
                                                int in_dim, int out_dim) {
    const int r = blockIdx.x;
    for (int t = threadIdx.x; t < out_dim; t += blockDim.x) {
        const float* w = W + (size_t)t * in_dim;
        const float* row = IN + (size_t)r * in_dim;
        float acc = 0.f;
        for (int i = 0; i < in_dim; ++i) {
            acc += row[i] * w[i];
        }
        OUT[(size_t)r * out_dim + t] = acc + B[t];
    }
}

// K-silu kernel: grid (56), block (1024), elementwise stride loop over n = rows*d:
// OUT[i] = __float2bfloat16_rn(x / (1.0f + expf(-x))) on f32 IN.
// Args: (const float* IN, __nv_bfloat16* OUT, int n)
extern "C" __global__ void tf_lilicorr_silu_bf16(const float* __restrict__ IN, __nv_bfloat16* __restrict__ OUT, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float x = IN[i];
    const float silu_val = x / (1.0f + expf(-x));
    OUT[i] = __float2bfloat16_rn(silu_val);
}

// K-add kernel: grid (56), block (1024), stride loop: the residual —
// OUT[i] = __float2bfloat16_rn(bf(A[i]) + bf(B[i])) where A is bf16 (the lattice) and B is f32 (the projection output, cast FIRST).
// Args: (const __nv_bfloat16* A, const float* B, __nv_bfloat16* OUT, int n)
extern "C" __global__ void tf_lilicorr_add_bf16(const __nv_bfloat16* __restrict__ A, const float* __restrict__ B,
                                                 __nv_bfloat16* __restrict__ OUT, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float a = bf(A[i]);
    const float b = rbf(B[i]); // the reference casts the projection output to bf16 BEFORE the add
    OUT[i] = __float2bfloat16_rn(a + b);
}

// K-lattice kernel: grid (56), block (1024): the assembly —
// OUT[r*1024+i] = __float2bfloat16_rn(TS[r*1024+i] + PS[i] + FM[r*1024+i] + SLOT[(r/8)*1024+i] + RANK[(r%8)*1024+i])
// (TS/FM/PS/SLOT/RANK all f32 — the host sums the f32 embeddings and stores ONE bf16; the S5b
// device chain mirrors that exactly, so SLOT/RANK are the f32-staged dhw tensors).
// Args: (const float* TS, const float* PS, const float* FM, const float* SLOT, const float* RANK, __nv_bfloat16* OUT)
extern "C" __global__ void tf_lilicorr_lattice(const float* __restrict__ TS, const float* __restrict__ PS,
                                                const float* __restrict__ FM, const float* __restrict__ SLOT,
                                                const float* __restrict__ RANK, __nv_bfloat16* __restrict__ OUT) {
    const int r = blockIdx.x;
    const int i = threadIdx.x;

    if (i >= 1024) return;

    const int idx = r * 1024 + i;
    const float ts = TS[idx];
    const float ps = PS[i];
    const float fm = FM[idx];
    const float slot = SLOT[(r / 8) * 1024 + i];
    const float rank = RANK[(r % 8) * 1024 + i];

    OUT[idx] = __float2bfloat16_rn(ts + ps + fm + slot + rank);
}

// Factor block kernel: computes the factor hidden representation
// Grid (56 rows), block (1024 = one thread per output channel t).
// Thread t of row r computes:
// v = dot(A_IN[r], W + t*3072) + dot(ANCHOR, W + t*3072 + 1024) + dot_prod + B[t]
// where dot_prod = sum_i A_IN[r*1024+i] * ANCHOR[i] * W[t*3072 + 2048 + i]
// then OUT[r*1024+t] = bf16_bits(silu(v))
// Args: (const float* A_IN, const float* ANCHOR, const float* W, const float* B,
// __nv_bfloat16* OUT, int rows_unused_keep_shape, int hd).

// ===== 16f: row-batched tiled GEMVs — W is read ONCE per call (the old kernels re-read the
// whole W once per ROW-BLOCK, uncoalesced 16KB-strided: 56 x 16 MB = the measured 3.7 ms).
// Same math at f32-accumulate tolerance; the open-time fixture A/Bs gate exactness.
// W layout [out][in] (t-major rows), X [rows][in], OUT [rows][out] f32, + B per output col.
// grid.x = ceil(out_dim / TN), block = 256. CONTRACT: rows <= TF_GEMV_MAX_ROWS (64)
// (enforced by the caller; the smem tiles are sized for it and the accumulators hold 8).
#define TF_GEMV_TN 32
#define TF_GEMV_TK 32
#define TF_GEMV_MAX_ROWS 64

// NOTE: the launch sites pin block=256; the strides below must match blockDim.x.
extern "C" __global__ void tf_gemv_bf16_rows(const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                                             const float* __restrict__ B, float* __restrict__ OUT,
                                             int in_dim, int out_dim, int rows) {
    __shared__ float Ws[TF_GEMV_TN][TF_GEMV_TK + 1];
    __shared__ __nv_bfloat16 Es[TF_GEMV_MAX_ROWS][TF_GEMV_TK + 1];
    const int t0 = blockIdx.x * TF_GEMV_TN;
    const int tid = threadIdx.x;
    float acc[8]; int rr[8]; int aa[8]; int n = 0;   // 8 = ceil(TF_GEMV_MAX_ROWS*TN/256): the true per-thread count at the rows cap
    for (int idx = tid; idx < rows * TF_GEMV_TN; idx += 256) {
        if (n < 8) { rr[n] = idx / TF_GEMV_TN; aa[n] = idx % TF_GEMV_TN; acc[n] = 0.f; ++n; }
    }
    for (int k0 = 0; k0 < in_dim; k0 += TF_GEMV_TK) {
        for (int i = tid; i < TF_GEMV_TN * TF_GEMV_TK; i += 256) {
            const int a = i / TF_GEMV_TK, kk = i % TF_GEMV_TK;
            Ws[a][kk] = ((t0 + a < out_dim) && (k0 + kk < in_dim)) ? W[(size_t)(t0 + a) * in_dim + k0 + kk] : 0.f;
        }
        for (int i = tid; i < rows * TF_GEMV_TK; i += 256) {
            const int r = i / TF_GEMV_TK, kk = i % TF_GEMV_TK;
            Es[r][kk] = (k0 + kk < in_dim) ? X[(size_t)r * in_dim + k0 + kk] : __float2bfloat16_rn(0.f);
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            float s = 0.f;
            #pragma unroll
            for (int kk = 0; kk < TF_GEMV_TK; ++kk) s += bf(Es[rr[j]][kk]) * Ws[aa[j]][kk];
            acc[j] += s;
        }
        __syncthreads();
    }
    for (int j = 0; j < n; ++j) if (t0 + aa[j] < out_dim) OUT[(size_t)rr[j] * out_dim + t0 + aa[j]] = acc[j] + B[t0 + aa[j]];
}

extern "C" __global__ void tf_gemv_f32_rows(const float* __restrict__ X, const float* __restrict__ W,
                                            const float* __restrict__ B, float* __restrict__ OUT,
                                            int in_dim, int out_dim, int rows) {
    __shared__ float Ws[TF_GEMV_TN][TF_GEMV_TK + 1];
    __shared__ float Es[TF_GEMV_MAX_ROWS][TF_GEMV_TK + 1];
    const int t0 = blockIdx.x * TF_GEMV_TN;
    const int tid = threadIdx.x;
    float acc[8]; int rr[8]; int aa[8]; int n = 0;   // 8 = ceil(TF_GEMV_MAX_ROWS*TN/256): the true per-thread count at the rows cap
    for (int idx = tid; idx < rows * TF_GEMV_TN; idx += 256) {
        if (n < 8) { rr[n] = idx / TF_GEMV_TN; aa[n] = idx % TF_GEMV_TN; acc[n] = 0.f; ++n; }
    }
    for (int k0 = 0; k0 < in_dim; k0 += TF_GEMV_TK) {
        for (int i = tid; i < TF_GEMV_TN * TF_GEMV_TK; i += 256) {
            const int a = i / TF_GEMV_TK, kk = i % TF_GEMV_TK;
            Ws[a][kk] = ((t0 + a < out_dim) && (k0 + kk < in_dim)) ? W[(size_t)(t0 + a) * in_dim + k0 + kk] : 0.f;
        }
        for (int i = tid; i < rows * TF_GEMV_TK; i += 256) {
            const int r = i / TF_GEMV_TK, kk = i % TF_GEMV_TK;
            Es[r][kk] = (k0 + kk < in_dim) ? X[(size_t)r * in_dim + k0 + kk] : 0.f;
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            float s = 0.f;
            #pragma unroll
            for (int kk = 0; kk < TF_GEMV_TK; ++kk) s += Es[rr[j]][kk] * Ws[aa[j]][kk];
            acc[j] += s;
        }
        __syncthreads();
    }
    for (int j = 0; j < n; ++j) if (t0 + aa[j] < out_dim) OUT[(size_t)rr[j] * out_dim + t0 + aa[j]] = acc[j] + B[t0 + aa[j]];
}

// ===== 16f: the factor block as ONE kernel with the same tiled discipline (same signature and
// math as tf_lilicorr_factor; grid.x = ceil(hd/8), block = 256 = 8 t-cols x 32 i-lanes).
// Thread (a, lane) owns output col t = blockIdx.x*8 + a and accumulates v[r] for ALL rows;
// W is read once, coalesced along i. v[] is fully unrolled so it stays in registers.
#define TF_FACT_ROWS 64

extern "C" __global__ void tf_lilicorr_factor_rows(const float* __restrict__ A_IN, const float* __restrict__ ANCHOR,
                                                   const float* __restrict__ W, const float* __restrict__ B,
                                                   __nv_bfloat16* __restrict__ OUT, int rows, int hd) {
    __shared__ float Ain[TF_FACT_ROWS][33];
    __shared__ float Anch[33];
    const int a = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int t = blockIdx.x * 8 + a;
    // TAIL-WARP FIX (A1 fuzz catch, 11/25): NO early return — the tail warps of the last
    // block (t >= hd when hd % 8 != 0) are also strided LOAD WORKERS for the shared Ain
    // array; exiting them left whole Ain rows (r = warp id) uninitialized and the factor
    // outputs for those rows were garbage. Inactive warps now stay for loads + barriers
    // and only skip the B-carry, the accumulate, and the OUT write.
    const bool active = t < hd;
    float v[TF_FACT_ROWS];
    #pragma unroll
    for (int r = 0; r < TF_FACT_ROWS; ++r) v[r] = (r < rows && lane == 0 && active) ? B[t] : 0.f;
    const float* w = W + (size_t)t * 3 * hd;
    for (int i0 = 0; i0 < hd; i0 += 32) {
        for (int i = threadIdx.x; i < rows * 32; i += 256) {
            const int r = i / 32, kk = i % 32;
            Ain[r][kk] = (i0 + kk < hd) ? A_IN[(size_t)r * hd + i0 + kk] : 0.f;
        }
        if (threadIdx.x < 32) Anch[threadIdx.x] = (i0 + threadIdx.x < hd) ? ANCHOR[i0 + threadIdx.x] : 0.f;
        __syncthreads();
        if (active) {
            const int ii = i0 + lane;
            const bool ok = ii < hd;
            const float w1 = ok ? w[ii] : 0.f;
            const float w2 = ok ? w[hd + ii] : 0.f;
            const float w3 = ok ? w[2 * hd + ii] : 0.f;
            const float an = Anch[lane];
            #pragma unroll
            for (int r = 0; r < TF_FACT_ROWS; ++r) {
                if (r < rows) { const float air = Ain[r][lane]; v[r] += air * w1 + an * w2 + air * an * w3; }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TF_FACT_ROWS; ++r) {
        if (r < rows && active) {
            float x = v[r];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) x += __shfl_down_sync(0xffffffffu, x, off);
            if (lane == 0) OUT[(size_t)r * hd + t] = __float2bfloat16_rn(x / (1.0f + expf(-x)));
        }
    }
}
