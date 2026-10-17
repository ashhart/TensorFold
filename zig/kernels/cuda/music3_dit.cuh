// music3 DiT kernels: the condition, the windows, LayerNorm, flash attention, GLU and the Euler step.
#pragma once
#include "music3_common.cuh"

// ------------------------------------------------------------------------------------------------ DiT

// The condition mix of frame states [T, 8, 4096] (bf16) in fp16: sum_l w_l h_l (fp32 sum, fp16 out) times scale (fp16).
extern "C" __global__ void m3_cond_mix(const bf16* __restrict__ states, const float* __restrict__ w8, float scale, __half* __restrict__ out, int T) {
  const size_t n = (size_t)T * 4096;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / 4096, d = i % 4096;
    float s = 0.f;
#pragma unroll
    for (int l = 0; l < 8; ++l) s = fmaf(rh(b2f(states[(t * 8 + l) * 4096 + d])), w8[l], s);
    out[i] = f2h(rh(rh(s) * scale));
  }
}

// im2col for the k=3, pad 1 condition conv: col [T, 3, 4096] from x [T, 4096] (fp16; zero outside).
extern "C" __global__ void m3_im2col3(const __half* __restrict__ x, __half* __restrict__ col, int T, int C) {
  const size_t n = (size_t)T * 3 * C;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / (3 * C), rem = i % (3 * C);
    const int k = (int)(rem / C), c = (int)(rem % C);
    const long src = (long)t + k - 1;
    col[i] = (src >= 0 && src < T) ? x[(size_t)src * C + c] : f2h(0.f);
  }
}

// Window inputs [S, Lw, 2304] = [x, 0, condition] (branch 1 reads the null condition, the bias); short rows 0.
extern "C" __global__ void m3_dit_input(const float* __restrict__ x, const __half* __restrict__ cond, const __half* __restrict__ null_row, const int* __restrict__ wins, int nwin, int Lw, __half* __restrict__ full) {
  const int s = blockIdx.y, j = blockIdx.x;
  const int w = s % nwin, branch = s / nwin;
  const int start = wins[2 * w], len = wins[2 * w + 1];
  __half* row = full + ((size_t)s * Lw + j) * 2304;
  for (int c = threadIdx.x; c < 2304; c += blockDim.x) {
    float v = 0.f;
    if (j < len) {
      const int t = start + j;
      if (c < 128)
        v = x[(size_t)t * 128 + c];
      else if (c >= 256)
        v = branch == 0 ? h2f(cond[(size_t)t * 2048 + c - 256]) : h2f(null_row[c - 256]);
    }
    row[c] = f2h(v);
  }
}

// a = round(a + b) in fp16.
extern "C" __global__ void m3_add_h(__half* __restrict__ a, const __half* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] = f2h(h2f(a[i]) + h2f(b[i]));
}

// The timestep embedding input: [cos(2 pi t f), sin(2 pi t f)] (fp32, then fp16), f = features [128].
extern "C" __global__ void m3_time_features(const __half* __restrict__ f, float t, __half* __restrict__ out) {
  const int i = threadIdx.x;  // 128
  const float a = 2.f * 3.14159265358979323846f * t * h2f(f[i]);
  out[i] = f2h(cosf(a));
  out[128 + i] = f2h(sinf(a));
}

// y = silu(x) (fp16) for the timestep MLP.
extern "C" __global__ void m3_silu_h(const __half* __restrict__ x, __half* __restrict__ y, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = h2f(x[i]);
    y[i] = f2h(v / (1.f + expf(-v)));
  }
}

// One fp16 row through a small Linear: y [N] = round(W [N, K] x + b) (fp32 sum), a warp an output.
extern "C" __global__ void m3_linear_h1(const __half* __restrict__ w, const __half* __restrict__ b, const __half* __restrict__ x, int K, int N, __half* __restrict__ y) {
  const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (n >= N) return;
  float s = 0.f;
  for (int k = lane; k < K; k += 32) s = fmaf(h2f(w[(size_t)n * K + k]), h2f(x[k]), s);
  s = warp_sum(s);
  if (lane == 0) y[n] = f2h(s + (b ? h2f(b[n]) : 0.f));
}

// dst [n, C] = src [idx[i], C] (fp16 rows: the condition's nearest-neighbour stretch to the latent timeline).
extern "C" __global__ void m3_gather_rows_h(const __half* __restrict__ src, const int* __restrict__ idx, __half* __restrict__ dst, int C) {
  const size_t i = blockIdx.x;
  for (int c = threadIdx.x; c < C; c += blockDim.x) dst[i * C + c] = src[(size_t)idx[i] * C + c];
}

// Row 0 of each sequence of h [S, Lseq, 2048] = the timestep embedding.
extern "C" __global__ void m3_put_time(const __half* __restrict__ te, __half* __restrict__ h, int Lseq) {
  const int s = blockIdx.x;
  for (int c = threadIdx.x; c < 2048; c += blockDim.x) h[(size_t)s * Lseq * 2048 + c] = te[c];
}

// LayerNorm (eps 1e-5) of fp16 rows of 2048 in fp32; `bias` first adds the previous block's ff_out bias into x.
extern "C" __global__ void m3_layernorm_h(__half* __restrict__ x, const __half* __restrict__ bias, const __half* __restrict__ g, const __half* __restrict__ b, __half* __restrict__ y) {
  __shared__ float sh[32];
  __half* xr = x + (size_t)blockIdx.x * 2048;
  float v[8];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int c = threadIdx.x + 256 * j;
    v[j] = h2f(xr[c]);
    if (bias) {
      v[j] = rh(v[j] + h2f(bias[c]));
      xr[c] = f2h(v[j]);
    }
    s += v[j];
  }
  const float mean = block_sum(s, sh) / 2048.f;
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float d = v[j] - mean;
    q = fmaf(d, d, q);
  }
  const float var = block_sum(q, sh) / 2048.f;
  const float rs = 1.0f / sqrtf(var + 1e-5f);
  __half* yr = y + (size_t)blockIdx.x * 2048;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int c = threadIdx.x + 256 * j;
    yr[c] = f2h((v[j] - mean) * rs * h2f(g[c]) + h2f(b[c]));
  }
}

// h [rows, 2048] = round(h + bias) (the last block's ff_out bias).
extern "C" __global__ void m3_bias_h(__half* __restrict__ h, const __half* __restrict__ bias, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) h[i] = f2h(h2f(h[i]) + h2f(bias[i % 2048]));
}

// q, k with rope on dims 0..31 and v, from qkv rows into head-major [S, 32, Lseq, 64] (the unfused path).
extern "C" __global__ void m3_dit_rope(const __half* __restrict__ qkv, const float* __restrict__ cs, int Lseq, __half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v) {
  const int j = blockIdx.x, s = blockIdx.y;
  const __half* row = qkv + ((size_t)s * Lseq + j) * 6144;
  for (int e = threadIdx.x; e < 2048; e += blockDim.x) {
    const int h = e >> 6, d = e & 63;
    const size_t o = (((size_t)s * 32 + h) * Lseq + j) * 64 + d;
    v[o] = row[4096 + e];
    float qv = h2f(row[e]), kv = h2f(row[2048 + e]);
    if (d < 32) {
      const int i = d & 15;
      const float c = cs[(j * 16 + i) * 2], sn = cs[(j * 16 + i) * 2 + 1];
      const int pe = d < 16 ? e + 16 : e - 16;
      const float qo = h2f(row[pe]), ko = h2f(row[2048 + pe]);
      if (d < 16) {
        qv = qv * c - qo * sn;
        kv = kv * c - ko * sn;
      } else {
        qv = qv * c + qo * sn;
        kv = kv * c + ko * sn;
      }
    }
    q[o] = f2h(qv);
    k[o] = f2h(kv);
  }
}

// Softmax of fp32 score rows over each sequence's key length, fp16 probabilities (the unfused attention path).
extern "C" __global__ void m3_softmax_rows(const float* __restrict__ sc, int ld, int Lseq, const int* __restrict__ lens, int s0, __half* __restrict__ p) {
  __shared__ float sh[32];
  const int j = blockIdx.x, bh = blockIdx.y;
  const int s = s0 + bh / 32;
  const int L = lens[s];
  const float* row = sc + ((size_t)bh * Lseq + j) * ld;
  __half* pr = p + ((size_t)bh * Lseq + j) * ld;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < L; i += blockDim.x) m = fmaxf(m, row[i]);
  m = block_max(m, sh);
  float sum = 0.f;
  for (int i = threadIdx.x; i < L; i += blockDim.x) sum += __expf(row[i] - m);
  sum = block_sum(sum, sh);
  const float inv = 1.f / sum;
  for (int i = threadIdx.x; i < ld; i += blockDim.x) pr[i] = f2h(i < L ? __expf(row[i] - m) * inv : 0.f);
}

// mma.sync m16n8k16 (fp16 in, fp32 accumulate) and ldmatrix for the flash attention below.
#define FA_PAD 72
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&h);
}

// A qkv row's element d with rope on dims 0..31 (split halves, cs [Lseq, 16, 2]), rounded to fp16 as the reference.
__device__ __forceinline__ float rope_at(const __half* row, int d, const float* cs, int j) {
  const float x = h2f(row[d]);
  if ((d & 63) >= 32) return x;
  const int i = d & 15;
  const float c = cs[(j * 16 + i) * 2], sn = cs[(j * 16 + i) * 2 + 1];
  return (d & 63) < 16 ? rh(x * c - h2f(row[d + 16]) * sn) : rh(x * c + h2f(row[d - 16]) * sn);
}

// rope on a k row segment (8 dims, seg 0..3) from it and its partner seg ^ 2 at position j, rounded to fp16.
__device__ __forceinline__ uint4 rope_seg(uint4 a, uint4 p, int seg, const float* cs, int j) {
  const __half2* ah = reinterpret_cast<const __half2*>(&a);
  const __half2* ph = reinterpret_cast<const __half2*>(&p);
  uint4 o;
  __half2* oh = reinterpret_cast<__half2*>(&o);
  const float sign = seg < 2 ? -1.f : 1.f;
  const int i0 = (seg & 1) * 8;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 x = __half22float2(ah[e]), y = __half22float2(ph[e]);
    const float* c0 = cs + (j * 16 + i0 + 2 * e) * 2;
    oh[e] = __floats2half2_rn(x.x * c0[0] + sign * y.x * c0[1], x.y * c0[2] + sign * y.y * c0[3]);
  }
  return o;
}

// Flash attention from qkv rows, rope as q and k load: 128 queries a CTA, keys 64 a block, mma.sync, online softmax.
#define FA_ROWS 128
extern "C" __global__ void __launch_bounds__(256) m3_flash_attn(const __half* __restrict__ qkv, const float* __restrict__ cs, int Lseq, const int* __restrict__ lens, float scale, __half* __restrict__ out) {
  __shared__ __align__(16) __half ks[64 * FA_PAD];
  __shared__ __align__(16) __half vs[64 * FA_PAD];
  const int bh = blockIdx.y, s = bh >> 5, h = bh & 31;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int L = lens[s];
  const __half* rows = qkv + (size_t)s * Lseq * 6144;
  const int r0 = blockIdx.x * FA_ROWS + warp * 16;
  unsigned qa[4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = r0 + g + (j & 1) * 8, col = kk * 16 + t * 2 + (j >> 1) * 8;
      if (row < Lseq) {
        const __half* rp = rows + (size_t)row * 6144 + h * 64;
        qa[kk][j] = pack_h2(rope_at(rp, col, cs, row), rope_at(rp, col + 1, cs, row));
      } else
        qa[kk][j] = 0u;
    }
  }
  float o[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float c = scale * 1.4426950408889634f;  // exp2 domain
  for (int kb = 0; kb < L; kb += 64) {
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 8; i += 256) {
      const int row = i >> 3, seg = i & 7;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (kb + row < Lseq) {
        const __half* rp = rows + (size_t)(kb + row) * 6144 + h * 64;
        vv = *reinterpret_cast<const uint4*>(rp + 4096 + seg * 8);
        kv = *reinterpret_cast<const uint4*>(rp + 2048 + seg * 8);
        if (seg < 4) kv = rope_seg(kv, *reinterpret_cast<const uint4*>(rp + 2048 + (seg ^ 2) * 8), seg, cs, kb + row);
      }
      *reinterpret_cast<uint4*>(ks + row * FA_PAD + seg * 8) = kv;
      *reinterpret_cast<uint4*>(vs + row * FA_PAD + seg * 8) = vv;
    }
    __syncthreads();
    float sc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        unsigned b[4];
        ldsm4(b, ks + (nt * 8 + (lane & 7)) * FA_PAD + half * 32 + (lane >> 3) * 8);
        mma16816(sc[nt], qa[half * 2], b[0], b[1]);
        mma16816(sc[nt], qa[half * 2 + 1], b[2], b[3]);
      }
    }
    // mask keys past the sequence's length, then the online softmax (rows g and g + 8)
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = kb + nt * 8 + t * 2 + (j & 1);
        sc[nt][j] = key < L ? sc[nt][j] * c : -INFINITY;
      }
      mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float cr0 = exp2f(m0 - mx0), cr1 = exp2f(m1 - mx1);
    m0 = mx0;
    m1 = mx1;
    l0 *= cr0;
    l1 *= cr1;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o[i][0] *= cr0;
      o[i][1] *= cr0;
      o[i][2] *= cr1;
      o[i][3] *= cr1;
    }
    unsigned pa[4][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float p0 = exp2f(sc[nt][0] - m0), p1 = exp2f(sc[nt][1] - m0), p2 = exp2f(sc[nt][2] - m1), p3 = exp2f(sc[nt][3] - m1);
      l0 += p0 + p1;
      l1 += p2 + p3;
      pa[nt >> 1][(nt & 1) * 2] = pack_h2(p0, p1);
      pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        unsigned b[4];
        ldsm4t(b, vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FA_PAD + np * 16 + (lane >> 4) * 8);
        mma16816(o[np * 2], pa[kk], b[0], b[1]);
        mma16816(o[np * 2 + 1], pa[kk], b[2], b[3]);
      }
    }
  }
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int ra = r0 + g, rb = r0 + g + 8;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int d = h * 64 + nt * 8 + t * 2;
    if (ra < Lseq) *reinterpret_cast<__half2*>(out + ((size_t)s * Lseq + ra) * 2048 + d) = __floats2half2_rn(o[nt][0] * i0, o[nt][1] * i0);
    if (rb < Lseq) *reinterpret_cast<__half2*>(out + ((size_t)s * Lseq + rb) * 2048 + d) = __floats2half2_rn(o[nt][2] * i1, o[nt][3] * i1);
  }
}

// Head-major attention out [S, 32, Lseq, 64] back to token rows [S, Lseq, 2048]. grid (Lseq, S), block 256.
extern "C" __global__ void m3_heads_to_rows(const __half* __restrict__ o, int Lseq, __half* __restrict__ out) {
  const int j = blockIdx.x, s = blockIdx.y;
  for (int e = threadIdx.x; e < 2048; e += blockDim.x) {
    const int h = e >> 6, d = e & 63;
    out[((size_t)s * Lseq + j) * 2048 + e] = o[(((size_t)s * 32 + h) * Lseq + j) * 64 + d];
  }
}

// GLU: round(value * round(silu(gate))) of the ff_in rows plus their bias (rounded to fp16 first), 8 a thread.
extern "C" __global__ void m3_glu_h(const __half* __restrict__ x, const __half* __restrict__ bias, __half* __restrict__ y, size_t rows) {
  const size_t n = rows * 1024;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / 1024, c = (i % 1024) * 8;
    const uint4 va = *reinterpret_cast<const uint4*>(x + r * 16384 + c);
    const uint4 ga = *reinterpret_cast<const uint4*>(x + r * 16384 + 8192 + c);
    const uint4 vb = *reinterpret_cast<const uint4*>(bias + c);
    const uint4 gb = *reinterpret_cast<const uint4*>(bias + 8192 + c);
    const __half2* vh = reinterpret_cast<const __half2*>(&va);
    const __half2* gh = reinterpret_cast<const __half2*>(&ga);
    const __half2* vbh = reinterpret_cast<const __half2*>(&vb);
    const __half2* gbh = reinterpret_cast<const __half2*>(&gb);
    uint4 o;
    __half2* oh = reinterpret_cast<__half2*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 vf = __half22float2(vh[j]), gf = __half22float2(gh[j]), vbf = __half22float2(vbh[j]), gbf = __half22float2(gbh[j]);
      const float v0 = rh(vf.x + vbf.x), v1 = rh(vf.y + vbf.y), g0 = rh(gf.x + gbf.x), g1 = rh(gf.y + gbf.y);
      oh[j] = __floats2half2_rn(v0 * rh(g0 / (1.f + expf(-g0))), v1 * rh(g1 / (1.f + expf(-g1))));
    }
    *reinterpret_cast<uint4*>(y + r * 8192 + c) = o;
  }
}

// A batch of windows' guided velocity -(o + post) summed into vsum / vcnt, windows in order (no atomics).
extern "C" __global__ void m3_dit_gather(const __half* __restrict__ o, const __half* __restrict__ post, const int* __restrict__ wins, int nwin, int Lw, float cfg, int L, float* __restrict__ vsum, float* __restrict__ vcnt) {
  const size_t n = (size_t)L * 128;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / 128), c = (int)(i % 128);
    float sum = vsum[i];
    float cnt = vcnt[i];
    for (int w = 0; w < nwin; ++w) {
      const int start = wins[2 * w], len = wins[2 * w + 1];
      if (t < start || t >= start + len) continue;
      const int j = t - start;
      const size_t rc = ((size_t)w * Lw + j) * 128 + c, ru = ((size_t)(nwin + w) * Lw + j) * 128 + c;
      const float vc = -rh(h2f(o[rc]) + h2f(post[rc]));
      const float vu = -rh(h2f(o[ru]) + h2f(post[ru]));
      sum += vu + (vc - vu) * cfg;
      cnt += 1.f;
    }
    vsum[i] = sum;
    vcnt[i] = cnt;
  }
}

// Euler: x += (vsum / vcnt) (sigma_next - sigma); the sums cleared for the next step.
extern "C" __global__ void m3_euler(float* __restrict__ x, float* __restrict__ vsum, float* __restrict__ vcnt, float dt, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    x[i] = fmaf(vsum[i] / vcnt[i], dt, x[i]);
    vsum[i] = 0.f;
    vcnt[i] = 0.f;
  }
}

// Gaussian noise [n] from (seed, i): Philox normals, two a counter.
extern "C" __global__ void m3_noise(float* __restrict__ x, unsigned long long seed, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; 2 * i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float2 g = Philox::normal2(seed, i, 0x6e6f697365ull);
    x[2 * i] = g.x;
    if (2 * i + 1 < n) x[2 * i + 1] = g.y;
  }
}
