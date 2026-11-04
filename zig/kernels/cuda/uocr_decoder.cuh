// Unlimited-OCR-2's decoder steps on CUDA (uocr/decoder.metal's).
#pragma once
#include "uocr_common.cuh"

// Token rows: x[i] = embed[ids[i]] (1280 wide), image tokens' rows left for their pictures. Grid (rows, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_embed(const unsigned* __restrict__ ids, const bf16* __restrict__ embed,
                                                                bf16* __restrict__ x, unsigned image_token) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  const unsigned id = ids[blockIdx.x];
  if (c >= 1280 || id == image_token) return;
  x[(size_t)blockIdx.x * 1280 + c] = embed[(size_t)id * 1280 + c];
}

// DeepseekV2RMSNorm on bf16 rows: x * rsqrt(mean(x^2) + eps) in fp32, rounded, then times the weight in bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_rms_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                   bf16* __restrict__ y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = ld(xr + i);
    s += v * v;
  }
  s = block_sum(s, part);
  const float r = 1.0f / sqrtf(s / (float)width + eps);
  bf16* yr = y + (size_t)blockIdx.x * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to(ld(w + i) * bf(ld(xr + i) * r));
}

// The page2d angles for one position: 63 token frequencies, then the page parity (page mod 2) at frequency 1.
__device__ __forceinline__ float angle(const float* inv_tok, unsigned tok, unsigned page, int i) {
  const int j = i % 64;
  return j < 63 ? (float)tok * inv_tok[j] : (float)(page % 2);
}

// page2d rotary in place on 128-wide heads at (tok, page): bf16 products and sum, bf16 cos and sin.
extern "C" __global__ void __launch_bounds__(128) tf_uocr_rope(bf16* __restrict__ x, const unsigned* __restrict__ tok,
                                                               const unsigned* __restrict__ page,
                                                               const float* __restrict__ inv_tok, int stride) {
  const int d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
  bf16* h = x + (size_t)row * stride + head * 128;
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), s = bf(sinf(ang));
  const float rh = d < 64 ? -ld(h + d + 64) : ld(h + d - 64);
  const bf16 out = to(bf(ld(h + d) * c) + bf(rh * s));
  __syncthreads();
  h[d] = out;
}

// The router's index q or k: Gemma RMSNorm clamped, the page2d rotary in bf16, then fp32 rows of 128.
extern "C" __global__ void __launch_bounds__(128) tf_uocr_router_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                      const unsigned* __restrict__ tok,
                                                                      const unsigned* __restrict__ page,
                                                                      const float* __restrict__ inv_tok, float* __restrict__ y) {
  __shared__ float part[33];
  __shared__ bf16 normed[128];
  const int row = blockIdx.x, d = threadIdx.x;
  float v = ld(x + (size_t)row * 128 + d);
  if (!isfinite(v)) v = 0.0f;
  const float s = block_sum(v * v, part);
  float n = clampf(v * (1.0f / sqrtf(s / 128.0f + 1e-6f)), -UD_CLAMP, UD_CLAMP);
  n = clampf(n * (1.0f + ld(w + d)), -UD_CLAMP, UD_CLAMP);
  normed[d] = to(n);
  __syncthreads();
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), sn = bf(sinf(ang));
  const float rh = d < 64 ? -ld(normed + d + 64) : ld(normed + d - 64);
  float out = bf(bf(ld(normed + d) * c) + bf(rh * sn));
  if (!isfinite(out)) out = 0.0f;
  y[(size_t)row * 128 + d] = out;
}

// A complete block's summary key: the softmax of its keys against head_cls, then their weighted sum in fp32.
extern "C" __global__ void __launch_bounds__(32) tf_uocr_block_keys(const float* __restrict__ idx_k, const bf16* __restrict__ head_cls,
                                                                    float* __restrict__ kbar, int first, float scale) {
  __shared__ float wts[32];
  const int c = first + blockIdx.x, j = threadIdx.x;
  const float* k = idx_k + ((size_t)c * 32 + j) * 128;
  float a = 0.0f;
  for (int d = 0; d < 128; d++) a += k[d] * ld(head_cls + d);
  a = clampf(a * scale, -UD_CLAMP, UD_CLAMP);
  const float m = warp_max(a);
  const float e = expf(a - m);
  const float total = fmaxf(warp_sum(e), 1e-6f);
  wts[j] = e / total;
  __syncwarp();
  for (int d = j; d < 128; d += 32) {
    float s = 0.0f;
    for (int t = 0; t < 32; t++) s += wts[t] * idx_k[((size_t)c * 32 + t) * 128 + d];
    kbar[(size_t)c * 128 + d] = isfinite(s) ? s : 0.0f;
  }
}

// One query's routed blocks: page-aware candidates, the top 16 by score, entmax-1.5, the first 12 with their bias.
#define UR_THREADS 256
extern "C" __global__ void __launch_bounds__(UR_THREADS) tf_uocr_route(const float* __restrict__ idx_q, const float* __restrict__ kbar,
                                                                       const int* __restrict__ bmin, const int* __restrict__ bmax,
                                                                       const int* __restrict__ qpage, const unsigned* __restrict__ qpos,
                                                                       int* __restrict__ sel, float* __restrict__ bias, int nblk,
                                                                       int complete, float scale, float idx_scale, float floor_,
                                                                       int round_bias) {
  __shared__ float x[4096];
  __shared__ unsigned char cand[4096];
  __shared__ float iq[128];
  __shared__ float xv[16];
  __shared__ int xi[16];
  const int q = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (t < 128) iq[t] = idx_q[(size_t)q * 128 + t];
  const int qp = qpage[q];
  const int last_remote = (int)(qpos[q] / 32) - 4;
  bool any_page = false;
  if (qp >= 0)
    for (int c = t; c < nblk; c += UR_THREADS) any_page = any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
  any_page = __syncthreads_or(any_page);
  bool any_cand = false;
  for (int c = t; c < nblk; c += UR_THREADS) {
    const bool remote = c <= last_remote && c < complete;
    const bool allow = !any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
    float z = UD_NEG;
    if (remote) {
      const float4* kb = reinterpret_cast<const float4*>(kbar + (size_t)c * 128);
      float s = 0.0f;
#pragma unroll 8
      for (int d = 0; d < 32; d++) {
        const float4 k4 = kb[d];
        s += iq[4 * d] * k4.x;
        s += iq[4 * d + 1] * k4.y;
        s += iq[4 * d + 2] * k4.z;
        s += iq[4 * d + 3] * k4.w;
      }
      z = clampf(s * scale, -UD_CLAMP, UD_CLAMP);
    }
    any_cand = any_cand || (remote && allow);
    cand[c] = (unsigned char)(remote ? (allow ? 2 : 1) : 0);
    x[c] = z;
  }
  any_cand = __syncthreads_or(any_cand);
  for (int c = t; c < nblk; c += UR_THREADS) {
    const bool is = any_cand ? cand[c] == 2 : cand[c] >= 1;
    cand[c] = (unsigned char)is;
    float v = is ? x[c] * idx_scale : UD_ENTMAX_NEG;
    if (!isfinite(v)) v = UD_ENTMAX_NEG;
    x[c] = v;
  }
  __syncthreads();
// The top M, ties to the lower block: each warp's own top M by shuffles, then warp 0 merges the 8 lists.
  const int M = max(1, min(16, nblk));
  __shared__ float lv[UR_THREADS / 32][16];
  __shared__ int li[UR_THREADS / 32][16];
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int c = t; c < nblk; c += UR_THREADS)
      if (x[c] != -INFINITY && (x[c] > best || (x[c] == best && c < at))) {
        best = x[c];
        at = c;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      if (ov > best || (ov == best && oi < at)) {
        best = ov;
        at = oi;
      }
    }
    if (at != 0x7fffffff && (at & 31) == lane) x[at] = -INFINITY;  // its owner: c mod 32 is the lane
    if (lane == 0) {
      lv[w][r] = best;
      li[w][r] = at;
    }
    __syncwarp();
  }
  __syncthreads();
  if (w != 0) return;
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = 0x7fffffff, slot = -1;
    for (int e = lane; e < (UR_THREADS / 32) * M; e += 32) {
      const float v = lv[e / M][e % M];
      const int i = li[e / M][e % M];
      if (i != 0x7fffffff && (v > best || (v == best && i < at))) {
        best = v;
        at = i;
        slot = e;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      const int os = __shfl_xor_sync(0xffffffffu, slot, o);
      if (ov > best || (ov == best && oi < at)) {
        best = ov;
        at = oi;
        slot = os;
      }
    }
    if (lane == 0) {
      xv[r] = best;
      xi[r] = at;
      li[slot / M][slot % M] = 0x7fffffff;
    }
    __syncwarp();
  }
  if (t != 0) return;
  // entmax-1.5 over the M values (already in descending order)
  const float mx = xv[0];
  float X[16], tau[16];
  float cs = 0.0f, cs2 = 0.0f;
  int support = 0;
  for (int k = 0; k < M; k++) {
    X[k] = (xv[k] - mx) / 2.0f;
    cs += X[k];
    cs2 += X[k] * X[k];
    const float rho = (float)(k + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - sqrtf(delta);
    support += tau[k] <= X[k] ? 1 : 0;
  }
  const float tau_star = tau[max(support, 1) - 1];
  const int K = max(1, min(12, M));
  float logp[12];
  float total = 0.0f;
  int count = 0;
  for (int s = 0; s < K; s++) {
    const float d = fmaxf(X[s] - tau_star, 0.0f);
    logp[s] = logf(fmaxf(d * d, floor_));
    if (cand[xi[s]]) {
      total += logp[s];
      count++;
    }
  }
  const float mu = total / (float)max(count, 1);
  for (int s = 0; s < 12; s++) {
    const bool valid = s < K && cand[xi[s]];
    sel[q * 12 + s] = valid ? xi[s] : -1;
    const float b = valid ? logp[s] - mu : 0.0f;
    bias[q * 12 + s] = round_bias ? bf(b) : b;
  }
}

// One query row and head: softmax over the 128-key window and the routed blocks' keys, fp32, out rounded to bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_attend(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                                 const bf16* __restrict__ vc, const int* __restrict__ sel,
                                                                 const float* __restrict__ bias, const unsigned* __restrict__ qpos,
                                                                 bf16* __restrict__ o, int width, float scale, int local) {
  __shared__ float score[512];
  __shared__ unsigned key[512];
  __shared__ unsigned seg_first[13], seg_start[14];
  __shared__ float seg_bias[13];
  __shared__ float part_max[8], part_sum[8];
  __shared__ float4 part_acc[8][32];
  const int row = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  const unsigned pos = qpos[row];
  const unsigned lo = pos + 1 > (unsigned)local ? pos + 1 - local : 0;
  if (t == 0) {
    unsigned n = 0;
    for (int s = 0; s <= 12; s++) {
      unsigned first = 0, last = 0;
      float b = 0.0f;
      if (s < 12) {
        const int c = sel[row * 12 + s];
        if (c >= 0) {
          first = (unsigned)c * 32;
          last = max(first, min(first + 32, lo));
          b = bias[row * 12 + s];
        }
      } else {
        first = lo;
        last = pos + 1;
      }
      seg_first[s] = first;
      seg_start[s] = n;
      seg_bias[s] = b;
      n += last - first;
    }
    seg_start[13] = n;
  }
  __syncthreads();
  const unsigned n = seg_start[13];
  const bf16* qr = q + (size_t)row * width + h * 128 + lane * 4;
  const float4 qv = make_float4(ld(qr), ld(qr + 1), ld(qr + 2), ld(qr + 3));
  float m = -INFINITY;
  unsigned s = 0;
  for (unsigned i = sg; i < n; i += 8) {
    while (i >= seg_start[s + 1]) s++;
    const unsigned j = seg_first[s] + (i - seg_start[s]);
    const bf16* kr = kc + (size_t)j * width + h * 128 + lane * 4;
    const float dot = qv.x * ld(kr) + qv.y * ld(kr + 1) + qv.z * ld(kr + 2) + qv.w * ld(kr + 3);
    const float sc = warp_sum(dot) * scale + seg_bias[s];
    if (lane == 0) {
      score[i] = sc;
      key[i] = j;
    }
    m = fmaxf(m, sc);
  }
  if (lane == 0) part_max[sg] = m;
  __syncthreads();
  m = part_max[0];
  for (int g = 1; g < 8; g++) m = fmaxf(m, part_max[g]);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float total = 0.0f;
  for (unsigned i = sg; i < n; i += 8) {
    const float e = expf(score[i] - m);
    total += e;
    const bf16* vr = vc + (size_t)key[i] * width + h * 128 + lane * 4;
    acc.x += e * ld(vr);
    acc.y += e * ld(vr + 1);
    acc.z += e * ld(vr + 2);
    acc.w += e * ld(vr + 3);
  }
  part_acc[sg][lane] = acc;
  if (lane == 0) part_sum[sg] = total;
  __syncthreads();
  if (sg != 0) return;
  float4 sum = part_acc[0][lane];
  float tot = part_sum[0];
  for (int g = 1; g < 8; g++) {
    sum.x += part_acc[g][lane].x;
    sum.y += part_acc[g][lane].y;
    sum.z += part_acc[g][lane].z;
    sum.w += part_acc[g][lane].w;
    tot += part_sum[g];
  }
  bf16* out = o + (size_t)row * width + h * 128 + lane * 4;
  out[0] = to(sum.x / tot);
  out[1] = to(sum.y / tot);
  out[2] = to(sum.z / tot);
  out[3] = to(sum.w / tot);
}

// y = bf16(bf16(silu(g)) * u): the MLP's act_fn(gate) * up in bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_silu_mul(const bf16* __restrict__ g, const bf16* __restrict__ u,
                                                                   bf16* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(g + i);
  y[i] = to(bf(v / (1.0f + expf(-v))) * ld(u + i));
}

// The MoE gate for one row: fp32 logits, softmax, the 6 largest (ties to the lower expert), not renormalized.
extern "C" __global__ void __launch_bounds__(64) tf_uocr_moe_gate(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                  unsigned* __restrict__ ids, float* __restrict__ wts) {
  __shared__ float part[33];
  __shared__ float score[64];
  const int row = blockIdx.x, e = threadIdx.x;
  const bf16* xr = x + (size_t)row * 1280;
  const bf16* wr = w + (size_t)e * 1280;
  float s = 0.0f;
  for (int d = 0; d < 1280; d++) s += ld(xr + d) * ld(wr + d);
  float m = warp_max(s);
  if ((e & 31) == 0) part[e >> 5] = m;
  __syncthreads();
  m = fmaxf(part[0], part[1]);
  __syncthreads();
  const float ex = expf(s - m);
  const float total = block_sum(ex, part);
  score[e] = ex / total;
  __syncthreads();
  if (e != 0) return;
  bool taken[64];
  for (int i = 0; i < 64; i++) taken[i] = false;
  for (int k = 0; k < 6; k++) {
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < 64; i++) {
      if (!taken[i] && score[i] > bv) {
        bv = score[i];
        best = i;
      }
    }
    taken[best] = true;
    ids[row * 6 + k] = best;
    wts[row * 6 + k] = bv;
  }
}

// Gathered rows: y[i] = x[rows[i]] (1280 wide). Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_gather(const bf16* __restrict__ x, const unsigned* __restrict__ rows,
                                                                 bf16* __restrict__ y) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  y[(size_t)blockIdx.x * 1280 + c] = x[(size_t)rows[blockIdx.x] * 1280 + c];
}

// An expert's rows back to their slots: slots[dest[i]] = y[i]. Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_scatter(const bf16* __restrict__ y, const unsigned* __restrict__ dest,
                                                                  bf16* __restrict__ slots) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  slots[(size_t)dest[blockIdx.x] * 1280 + c] = y[(size_t)blockIdx.x * 1280 + c];
}

// The routed sum in gate order rounded to bf16, plus the shared expert, rounded again.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_moe_combine(const bf16* __restrict__ slots, const float* __restrict__ wts,
                                                                      const bf16* __restrict__ shared, bf16* __restrict__ y) {
  const int c = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (c >= 1280) return;
  float s = 0.0f;
  for (int k = 0; k < 6; k++) s += ld(slots + ((size_t)row * 6 + k) * 1280 + c) * wts[row * 6 + k];
  const size_t i = (size_t)row * 1280 + c;
  y[i] = to(bf(s) + ld(shared + i));
}

// One token's projection y[n] = bf16(W[n, :] . x) with fp32 sums: 4 rows a warp, 8 warps a block; k a multiple of 8.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_matvec(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                                 bf16* __restrict__ y, int n, int k) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < k; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      if (row0 + r >= n) break;
      float wv[8];
      load8(W + (size_t)(row0 + r) * k + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += xv[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += xv[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0 && row0 + r < n) y[row0 + r] = to(s);
  }
}

// An expert's three projections, [F, 1280], [F, 1280] and [1280, F] as stored.
struct UdExpert {
  const bf16* gate;
  const bf16* up;
  const bf16* down;
};

// One token's experts, first half: act = bf16(bf16(silu(bf16(gate . x))) * bf16(up . x)), 4 rows a warp.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_gu(const bf16* __restrict__ x, const UdExpert* __restrict__ tab,
                                                                    const unsigned* __restrict__ ids, bf16* __restrict__ act,
                                                                    int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= ffn) return;
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}, u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < 1280; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wg[8], wu[8];
      load8(e.gate + (size_t)(row0 + r) * 1280 + i, wg);
      load8(e.up + (size_t)(row0 + r) * 1280 + i, wu);
      float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        g0 += xv[j] * wg[j];
        u0 += xv[j] * wu[j];
      }
#pragma unroll
      for (int j = 4; j < 8; j++) {
        g1 += xv[j] * wg[j];
        u1 += xv[j] * wu[j];
      }
      g[r] += g0 + g1;
      u[r] += u0 + u1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float gs = bf(warp_sum(g[r])), us = bf(warp_sum(u[r]));
    if (lane == 0) act[(size_t)s * ffn + row0 + r] = to(bf(gs / (1.0f + expf(-gs))) * us);
  }
}

// The second half: slots[s][i] = bf16(down_i . act[s]) for slot s's expert, 4 rows a warp.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_down(const bf16* __restrict__ act, const UdExpert* __restrict__ tab,
                                                                      const unsigned* __restrict__ ids, bf16* __restrict__ slots,
                                                                      int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= 1280) return;
  const bf16* a = act + (size_t)s * ffn;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < ffn; i += 256) {
    float av[8];
    load8(a + i, av);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wv[8];
      load8(e.down + (size_t)(row0 + r) * ffn + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += av[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += av[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float v = warp_sum(acc[r]);
    if (lane == 0) slots[(size_t)s * 1280 + row0 + r] = to(v);
  }
}

// The greedy pick on the device: the first maximum of bf16 logits, EOS skipped when `bar_eos`.
extern "C" __global__ void __launch_bounds__(1024) tf_uocr_argmax(const bf16* __restrict__ logits, int n, int bar_eos,
                                                                  unsigned* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (bar_eos && i == 1) continue;
    const float v = ld(logits + i);
    if (v > best) {
      best = v;
      at = i;
    }
  }
  // the warp's best, ties to the lower id
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) {
    bv[w] = best;
    bi[w] = at;
  }
  __syncthreads();
  if (w != 0) return;
  best = bv[lane];
  at = bi[lane];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at == 0x7fffffff ? 0 : at);
}
