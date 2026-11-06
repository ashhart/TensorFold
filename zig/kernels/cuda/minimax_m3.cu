// MiniMax-M3.1-Flash's glue on CUDA: the Metal family's glue.metal step for step (bf16 activations, fp32 sums).

#include "minimax_m3.cuh"

namespace tf_mm3 {
// The token ids' bf16 embeddings as the fp32 residual rows (bf16 values).
extern "C" __global__ void tf_mm3_embed(const int* __restrict__ ids, const bf16* __restrict__ table,
                                        float* __restrict__ resid, int rows, int hidden) {
  const int r = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || i >= hidden) return;
  resid[(size_t)r * hidden + i] = ld(table + (size_t)ids[r] * hidden + i);
}

// Gemma's RMSNorm of a residual row (bf16 values in fp32) scaled by 1 + w, rounded once: one block a row.
__device__ __forceinline__ void norm_row(const float* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out,
                                         int width, float eps, float* part) {
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s = fmaf(x[i], x[i], s);
  const float total = block_sum(s, part);
  const float inv = rsqrtf(total / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) st(out + i, x[i] * inv * (1.0f + ld(w + i)));
}

extern "C" __global__ void __launch_bounds__(1024) tf_mm3_norm(const float* __restrict__ resid, const bf16* __restrict__ w,
                                       bf16* __restrict__ out, int rows, int hidden, float eps) {
  __shared__ float part[32];
  const int r = blockIdx.x;
  if (r >= rows) return;
  norm_row(resid + (size_t)r * hidden, w, out + (size_t)r * hidden, hidden, eps, part);
}

// R = bf16(R + X) with X the ranks' summed partials (slots 2: bf16(bf16(scale * routed) + bf16(shared))); w: next norm
extern "C" __global__ void __launch_bounds__(1024) tf_mm3_add_norm(float* __restrict__ resid, const float* __restrict__ parts, int world,
                                           int slots, float scale, const bf16* __restrict__ w, bf16* __restrict__ out,
                                           int rows, int hidden, float eps) {
  __shared__ float part[32];
  const int r = blockIdx.x;
  if (r >= rows) return;
  float* x = resid + (size_t)r * hidden;
  const size_t rank_stride = (size_t)rows * slots * hidden;
  const float* p = parts + (size_t)r * slots * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float a = 0.0f, b = 0.0f;
    for (int q = 0; q < world; ++q) {
      a += p[(size_t)q * rank_stride + i];
      if (slots == 2) b += p[(size_t)q * rank_stride + hidden + i];
    }
    const float y = slots == 2 ? bf(bf(a * scale) + bf(b)) : bf(a);
    x[i] = bf(x[i] + y);
  }
  if (w == nullptr) return;
  __syncthreads();
  norm_row(x, w, out + (size_t)r * hidden, hidden, eps, part);
}

// One projection's output rows for prep: its first row and the elements between rows.
struct Rows {
  const bf16* x;
  int stride;
  __device__ __forceinline__ const bf16* at(int row, int head) const { return x + (size_t)row * stride + (size_t)head * 128; }
};

// q | k | v | index q | index k heads Gemma-normed and NeoX-rotated (bf16 cos, sin); k, v, index k cached at pos + row

extern "C" __global__ void tf_mm3_prep(const Rows XQ, const Rows XK, const Rows XV, const Rows XI, const bf16* __restrict__ qw,
                                       const bf16* __restrict__ kw, const bf16* __restrict__ iqw,
                                       const bf16* __restrict__ ikw, bf16* __restrict__ Q, bf16* __restrict__ IQ,
                                       bf16* __restrict__ KC, bf16* __restrict__ VC, bf16* __restrict__ IK, int rows,
                                       int pos, int hq, int hk, int hi, int rotary, float eps, float theta) {
  constexpr int D = 128;
  const int unit = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int units = hq + hk + hi + (IK != nullptr ? 1 : 0);  // no IK: keys and values only (a drafter's context)
  if (unit >= units || row >= rows) return;
  const bf16* src;
  const bf16* w;
  bf16* y;
  const size_t slot = (size_t)pos + row;
  const bool key = unit >= hq && unit < hq + hk;
  int head;
  if (unit < hq) {
    head = unit;
    src = XQ.at(row, head);
    w = qw;
    y = Q + ((size_t)row * hq + head) * D;
  } else if (key) {
    head = unit - hq;
    src = XK.at(row, head);
    w = kw;
    y = KC + (slot * hk + head) * D;
  } else if (unit < hq + hk + hi) {
    head = unit - hq - hk;
    src = XI.at(row, head);
    w = iqw;
    y = IQ + ((size_t)row * hi + head) * D;
  } else {
    head = 0;
    src = XI.at(row, hi);
    w = ikw;
    y = IK + slot * D;
  }
  float v[4];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = ld(src + lane + 32 * j);
    sum = fmaf(v[j], v[j], sum);
  }
  const float inv = rsqrtf(warp_sum(sum) / (float)D + eps);
  __shared__ float nrm_all[8][D];  // a head a warp, 8 warps a block (the launcher's 256 threads)
  float* nrm = nrm_all[threadIdx.x >> 5];
#pragma unroll
  for (int j = 0; j < 4; ++j) nrm[lane + 32 * j] = bf(v[j] * inv * (1.0f + ld(w + lane + 32 * j)));
  __syncwarp();
  const int half_r = rotary / 2;
  const float position = (float)(pos + row);
  float out[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = lane + 32 * j;
    if (d >= rotary) {
      out[j] = nrm[d];
      continue;
    }
    const int i = d % half_r;
    // theta^(2i/rotary) rounded once to fp32 (Metal's precise::pow, torch's float power), then its reciprocal
    const float freq = 1.0f / (float)pow((double)theta, (double)((float)(2 * i) / (float)rotary));
    const float angle = position * freq;
    const float c = bf(cosf(angle)), s = bf(sinf(angle));
    const float rot = d < half_r ? -nrm[d + half_r] : nrm[d - half_r];
    out[j] = bf(bf(nrm[d] * c) + bf(rot * s));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) st(y + lane + 32 * j, out[j]);
  if (!key) return;
  const bf16* vs = XV.at(row, head);
  bf16* vd = VC + (slot * hk + head) * D;
#pragma unroll
  for (int j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// SwiGLU-OAI of gate and up rows in bf16 steps: (clamp(up) + 1) * g * sigmoid(g * alpha), g = min(gate, limit).
extern "C" __global__ void tf_mm3_swiglu_oai(const bf16* __restrict__ G, const bf16* __restrict__ U, bf16* __restrict__ A,
                                             int rows, int width, int stride, float alpha, float limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i >= width || r >= rows) return;
  const size_t at = (size_t)r * stride + i;
  // torch.clamp's bounds: a NaN stays a NaN (fminf and fmaxf would hand back the limit)
  const float gv = ld(G + at), uv = ld(U + at);
  const float g = gv > limit ? limit : gv;
  const float u = uv > limit ? limit : (uv < -limit ? -limit : uv);
  const float s = bf(sigmoid(bf(g * alpha)));
  const float glu = bf(g * s);
  st(A + (size_t)r * width + i, bf(u + 1.0f) * glu);
}

// The router's fp32 logits: L[r, e] = sum_d float(X[r, d]) * W[e, d]; a warp an (expert, row).
extern "C" __global__ void tf_mm3_router(const bf16* __restrict__ X, const float* __restrict__ W,
                                         float* __restrict__ L, int rows, int width, int experts) {
  const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), r = blockIdx.y;
  const int lane = threadIdx.x & 31;
  if (e >= experts || r >= rows) return;
  const bf16* x = X + (size_t)r * width;
  const float* w = W + (size_t)e * width;
  float acc = 0.0f;
  for (int d = lane * 4; d < width; d += 128) {
    const float4 wv = *reinterpret_cast<const float4*>(w + d);
    acc = fmaf(wv.x, ld(x + d), acc);
    acc = fmaf(wv.y, ld(x + d + 1), acc);
    acc = fmaf(wv.z, ld(x + d + 2), acc);
    acc = fmaf(wv.w, ld(x + d + 3), acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) L[(size_t)r * experts + e] = acc;
}

// A row's top-k of sigmoid + bias (ties to the lower id), weights score / their sum, in ascending id; a warp a row
extern "C" __global__ void tf_mm3_route(const float* __restrict__ L, const float* __restrict__ bias,
                                        int* __restrict__ PICK, float* __restrict__ WTS, int rows, int experts,
                                        int topk) {
  constexpr int PER = 8;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows) return;
  float sc[PER], ch[PER];
  unsigned taken = 0;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int id = j * 32 + lane;
    if (id < experts) {
      sc[j] = sigmoid(L[(size_t)row * experts + id]);
      ch[j] = sc[j] + bias[id];
    } else {
      sc[j] = ch[j] = 0.0f;
      taken |= 1u << j;
    }
  }
  float w[8];
  int ids[8];
  for (int k = 0; k < topk; ++k) {
    float best = -CUDART_INF_F, bw = 0.0f;
    int bid = experts;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int id = j * 32 + lane;
      if (!(taken & (1u << j)) && (ch[j] > best || (ch[j] == best && id < bid))) {
        best = ch[j];
        bw = sc[j];
        bid = id;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, off), ow = __shfl_xor_sync(0xffffffffu, bw, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bid, off);
      if (ob > best || (ob == best && oi < bid)) {
        best = ob;
        bw = ow;
        bid = oi;
      }
    }
    w[k] = bw;
    ids[k] = bid;
    if (bid % 32 == lane) taken |= 1u << (bid / 32);
  }
  if (lane != 0) return;
  float total = 0.0f;
  for (int k = 0; k < topk; ++k) total += w[k];
  for (int k = 0; k < topk; ++k) w[k] = w[k] / total;
  for (int j = 1; j < topk; ++j)
    for (int m = j; m > 0 && ids[m] < ids[m - 1]; --m) {
      const int ti = ids[m];
      ids[m] = ids[m - 1];
      ids[m - 1] = ti;
      const float tw = w[m];
      w[m] = w[m - 1];
      w[m - 1] = tw;
    }
  for (int k = 0; k < topk; ++k) {
    PICK[row * topk + k] = ids[k];
    WTS[row * topk + k] = w[k];
  }
}

// A rank's MoE partial into its gather slot [rows][2][hidden] fp32: its picks' weighted down rows, its shared expert's
extern "C" __global__ void tf_mm3_moe_partial(const float* __restrict__ EY, const float* __restrict__ WTS,
                                              const float* __restrict__ SY, float* __restrict__ OUT, int rows,
                                              int topk, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i >= hidden || r >= rows) return;
  float acc = 0.0f;
  for (int k = 0; k < topk; ++k) {
    const size_t at = ((size_t)r * topk + k) * hidden + i;
    acc = fmaf(EY[at], WTS[r * topk + k], acc);
  }
  OUT[((size_t)r * 2) * hidden + i] = acc;
  OUT[((size_t)r * 2 + 1) * hidden + i] = SY[(size_t)r * hidden + i];
}

// A row's greedy pick over this rank's vocabulary slice: the largest value and its lowest id (plus the slice's base).
extern "C" __global__ void __launch_bounds__(1024) tf_mm3_top1(const bf16* __restrict__ L, int rows, int n, int base, float* __restrict__ V,
                                       int* __restrict__ I) {
  __shared__ float vals[32];
  __shared__ int ids[32];
  const int r = blockIdx.x;
  if (r >= rows) return;
  const bf16* l = L + (size_t)r * n;
  float best = -CUDART_INF_F;
  int bid = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = ld(l + i);
    if (v > best || (v == best && i < bid)) {
      best = v;
      bid = i;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, off);
    const int oi = __shfl_xor_sync(0xffffffffu, bid, off);
    if (ob > best || (ob == best && oi < bid)) {
      best = ob;
      bid = oi;
    }
  }
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  if (lane == 0) {
    vals[w] = best;
    ids[w] = bid;
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  for (int i = 1; i < warps; ++i)
    if (vals[i] > best || (vals[i] == best && ids[i] < bid)) {
      best = vals[i];
      bid = ids[i];
    }
  V[r] = best;
  I[r] = bid + base;
}

// DSpark's Markov scores of a row: S[v] = logits[v] + bf16(w2[v] . w1[prev]) over this rank's vocabulary, a warp a v
extern "C" __global__ void tf_mm3_markov(const bf16* __restrict__ logits, const bf16* __restrict__ w2,
                                         const bf16* __restrict__ w1, const int* __restrict__ prev_at,
                                         float* __restrict__ S, int n) {
  const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (v >= n) return;
  const int prev = *prev_at;  // the pick before (the anchor first), left on the device by tf_mm3_pick_merge
  const bf16* a = w2 + (size_t)v * 256;
  const bf16* b = w1 + (size_t)prev * 256;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc = fmaf(ld(a + lane + 32 * i), ld(b + lane + 32 * i), acc);
  acc = warp_sum(acc);
  if (lane == 0) S[v] = ld(logits + v) + bf(acc);
}

// A row's greedy pick of fp32 scores (this rank's `n`, ids offset by `base`): one block.
extern "C" __global__ void __launch_bounds__(1024) tf_mm3_top1f(const float* __restrict__ L, int n, int base,
                                                               float* __restrict__ V, int* __restrict__ I) {
  __shared__ float vals[32];
  __shared__ int ids[32];
  float best = -CUDART_INF_F;
  int bid = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = L[i];
    if (v > best || (v == best && i < bid)) {
      best = v;
      bid = i;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, off);
    const int oi = __shfl_xor_sync(0xffffffffu, bid, off);
    if (ob > best || (ob == best && oi < bid)) {
      best = ob;
      bid = oi;
    }
  }
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  if (lane == 0) {
    vals[w] = best;
    ids[w] = bid;
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  for (int i = 1; i < warps; ++i)
    if (vals[i] > best || (vals[i] == best && ids[i] < bid)) {
      best = vals[i];
      bid = ids[i];
    }
  V[0] = best;
  I[0] = bid + base;
}

// The ranks' gathered (value, id) picks to one, the best (the low id on a tie), into walk[j + 1]
extern "C" __global__ void tf_mm3_pick_merge(const unsigned char* __restrict__ picks, int world, int stride,
                                             int* __restrict__ walk, int j) {
  if (threadIdx.x != 0) return;
  float best = -CUDART_INF_F;
  int id = 0;
  for (int q = 0; q < world; ++q) {
    const float v = *reinterpret_cast<const float*>(picks + (size_t)q * stride);
    const int k = *reinterpret_cast<const int*>(picks + (size_t)q * stride + 4);
    if (v > best || (v == best && k < id)) {
      best = v;
      id = k;
    }
  }
  walk[j + 1] = id;
}

// fp32 pixels [P][K] . the fp32 conv weight [N][K]^T, each output one fma chain in k order, rounded to bf16 once
extern "C" __global__ void __launch_bounds__(256) tf_mm3_patch_embed(const float* __restrict__ X, const float* __restrict__ W,
                                                                     bf16* __restrict__ Y, int P, int N, int K) {
  __shared__ float xs[16][65], ws[16][65];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int i = threadIdx.x; i < 16 * 64; i += 256) {
      const int kk = i & 15, r = i >> 4, gk = k0 + kk;
      xs[kk][r] = (row0 + r < P && gk < K) ? X[(size_t)(row0 + r) * K + gk] : 0.0f;
      ws[kk][r] = (col0 + r < N && gk < K) ? W[(size_t)(col0 + r) * K + gk] : 0.0f;
    }
    __syncthreads();
    const int steps = min(16, K - k0);
    for (int kk = 0; kk < steps; ++kk) {
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(xs[kk][ty * 4 + a], ws[kk][tx * 4 + b], acc[a][b]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int r = row0 + ty * 4 + a, c = col0 + tx * 4 + b;
      if (r < P && c < N) st(Y + (size_t)r * N + c, acc[a][b]);
    }
}

// The tower's axial RoPE on q, k of qkv [P][3][16][80] in place: pairs (i, i + 39), i < 39, at [P][39] cos and sin
extern "C" __global__ void tf_mm3_vit_rope(bf16* __restrict__ qkv, const float* __restrict__ cs,
                                           const float* __restrict__ sn, int P) {
  const int p = blockIdx.x, h = blockIdx.y, w = blockIdx.z;  // w: 0 q, 1 k
  const int i = threadIdx.x;
  if (p >= P || i >= 39) return;
  bf16* x = qkv + ((size_t)p * 3 + w) * 1280 + h * 80;
  const float c = cs[p * 39 + i], s = sn[p * 39 + i];
  const float a = ld(x + i), b = ld(x + i + 39);
  st(x + i, a * c - b * s);
  st(x + i + 39, b * c + a * s);
}

// Image rows (bf16 [n][hidden]) over the embedded residual rows (fp32) they replace.
extern "C" __global__ void tf_mm3_splice(const bf16* __restrict__ rows, float* __restrict__ resid, int n, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i >= hidden || r >= n) return;
  resid[(size_t)r * hidden + i] = ld(rows + (size_t)r * hidden + i);
}

// A step's header and ids put into the link buffer by a kernel (no host upload to wait for), as GLM's put_words.
struct Words { unsigned w[96]; };
extern "C" __global__ void tf_mm3_put_words(unsigned* __restrict__ dst, unsigned n, Words words) {
  const unsigned i = threadIdx.x;
  if (i < n) dst[i] = words.w[i];
}

// The residual after a tapped layer (bf16 values in fp32) into a drafter's feature rows [rows][taps * hidden] bf16.
extern "C" __global__ void tf_mm3_tap(const float* __restrict__ resid, bf16* __restrict__ taps, int rows, int hidden,
                                      int ntaps, int tap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i >= hidden || r >= rows) return;
  st(taps + ((size_t)r * ntaps + tap) * hidden + i, resid[(size_t)r * hidden + i]);
}

}  // namespace tf_mm3
