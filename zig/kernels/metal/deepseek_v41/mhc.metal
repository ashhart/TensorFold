// Reference: DeepSeek-V4.1-Flash inference/model.py and kernel.py, revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
#include <metal_stdlib>
using namespace metal;
inline float bf(ushort x) { return as_type<float>(uint(x)<<16); }
inline ushort rb(float x) {
  uint u=as_type<uint>(x);
  if ((u&0x7fffffff)>0x7f800000) return ushort((u>>16)|0x40);
  return ushort((u+0x7fff+((u>>16)&1))>>16);
}
// Torch CUDA materializes FP32 squares before MeanOps accumulation.
#pragma clang fp contract(off)
struct Rms { uint rows,width,stream_stride; float eps; };
static_assert(sizeof(Rms) == 16, "Rms must mirror mhc.zig's Rms");
kernel void dsv41_hc_rms(const device ushort* x [[buffer(0)]], device float* inv [[buffer(1)]],
                         constant Rms& d [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  // Fixed single-row CUDA tree: four vector accumulators, stride 512, independent of prefill chunks.
  threadgroup float sums[512];
  float4 accum=0.0f;
  for (uint i=4*t;i<4*d.width;i+=4*512) {
    float4 a=float4(bf(x[ulong(row)*d.stream_stride+i]),
                    bf(x[ulong(row)*d.stream_stride+i+1]),
                    bf(x[ulong(row)*d.stream_stride+i+2]),
                    bf(x[ulong(row)*d.stream_stride+i+3]));
    float4 square=a*a;
    accum+=square;
  }
  float s=((accum.x+accum.y)+accum.z)+accum.w;
  sums[t]=s;
  for (uint offset=256;offset>=32;offset>>=1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t<offset) {
      s+=sums[t+offset];
      sums[t]=s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg==0) {
    for (uint offset=16;offset>0;offset>>=1) s+=simd_shuffle_down(s,offset);
    // MeanOps projects with a rounded FP32 factor, not a division of the sum.
    float factor=1.0f/float(4*d.width);
    if (lane==0) inv[row]=rsqrt(s*factor+d.eps);
  }
}
#pragma clang fp contract(on)
struct Split { uint rows,mix_stride,coefficient_stride,comb_stride; float eps; };
static_assert(sizeof(Split) == 20, "Split must mirror mhc.zig's Split");
kernel void dsv41_hc_split(const device float* mixes [[buffer(0)]], const device float* inv [[buffer(1)]],
                           const device float* scale [[buffer(2)]], const device float* base [[buffer(3)]],
                           device float* pre [[buffer(4)]], device float* post [[buffer(5)]],
                           device float* comb [[buffer(6)]], constant Split& d [[buffer(7)]],
                           uint row [[thread_position_in_grid]]) {
  if (row>=d.rows) return;
  float m[16];
  for (uint j=0;j<4;j++) {
    float a=fma(mixes[ulong(row)*d.mix_stride+j]*inv[row],scale[0],base[j]);
    float b=fma(mixes[ulong(row)*d.mix_stride+4+j]*inv[row],scale[1],base[4+j]);
    pre[ulong(row)*d.coefficient_stride+j]=1.0f/(1.0f+exp(-a))+d.eps;
    post[ulong(row)*d.coefficient_stride+j]=2.0f/(1.0f+exp(-b));
    for (uint k=0;k<4;k++) {
      uint q=4*j+k;
      m[q]=fma(mixes[ulong(row)*d.mix_stride+8+q]*inv[row],scale[2],base[8+q]);
    }
  }
  // Initial row softmax + eps, then a column normalization. This counts as iteration one.
  for (uint j=0;j<4;j++) {
    float mx=max(max(m[4*j],m[4*j+1]),max(m[4*j+2],m[4*j+3]));
    for (uint k=0;k<4;k++) m[4*j+k]=exp(m[4*j+k]-mx);
    // Pinned TileLang HC row/column sums use FP32 butterfly pairs.
    float s=(m[4*j]+m[4*j+2])+(m[4*j+1]+m[4*j+3]);
    for (uint k=0;k<4;k++) m[4*j+k]=m[4*j+k]/s+d.eps;
  }
  for (uint it=0;it<20;it++) {
    if (it>0) {
      for (uint j=0;j<4;j++) {
        float s=((m[4*j]+m[4*j+2])+(m[4*j+1]+m[4*j+3]))+d.eps;
        for (uint k=0;k<4;k++) m[4*j+k]/=s;
      }
    }
    for (uint k=0;k<4;k++) {
      float s=((m[k]+m[8+k])+(m[4+k]+m[12+k]))+d.eps;
      for (uint j=0;j<4;j++) m[4*j+k]/=s;
    }
  }
  for (uint q=0;q<16;q++) comb[ulong(row)*d.comb_stride+q]=m[q];
}
// PyTorch materializes these FP32 products before reduction and branch addition.
#pragma clang fp contract(off)
struct Collapse { uint rows,width,stream_stride,y_stride,pre_stride; };
static_assert(sizeof(Collapse) == 20, "Collapse must mirror mhc.zig's Collapse");
kernel void dsv41_hc_collapse(const device ushort* x [[buffer(0)]], const device float* previous_pre [[buffer(1)]],
                              device ushort* y [[buffer(2)]], constant Collapse& d [[buffer(3)]],
                              uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  float s=0;
  for (uint c=0;c<4;c++) s+=previous_pre[ulong(i.y)*d.pre_stride+c]*bf(x[ulong(i.y)*d.stream_stride+ulong(c)*d.width+i.x]);
  y[ulong(i.y)*d.y_stride+i.x]=rb(s);
}
struct Post { uint rows,width,branch_stride,residual_stride,y_stride,coefficient_stride,comb_stride; };
static_assert(sizeof(Post) == 28, "Post must mirror mhc.zig's Post");
kernel void dsv41_hc_post(const device ushort* branch [[buffer(0)]], const device ushort* residual [[buffer(1)]],
                          const device float* post [[buffer(2)]], const device float* comb [[buffer(3)]],
                          device ushort* y [[buffer(4)]], constant Post& d [[buffer(5)]],
                          uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  float r[4];
  for (uint j=0;j<4;j++) r[j]=bf(residual[ulong(i.y)*d.residual_stride+ulong(j)*d.width+i.x]);
  float b=bf(branch[ulong(i.y)*d.branch_stride+i.x]);
  for (uint k=0;k<4;k++) {
    float s=0;
    for (uint j=0;j<4;j++) s+=comb[ulong(i.y)*d.comb_stride+4*j+k]*r[j];
    y[ulong(i.y)*d.y_stride+ulong(k)*d.width+i.x]=rb(post[ulong(i.y)*d.coefficient_stride+k]*b+s);
  }
}
#pragma clang fp contract(on)
struct Expand { uint rows,width,x_stride,stream_stride,pre_stride; };
static_assert(sizeof(Expand) == 20, "Expand must mirror mhc.zig's Expand");
kernel void dsv41_hc_expand(const device ushort* x [[buffer(0)]], device ushort* stream [[buffer(1)]],
                            device float* pre [[buffer(2)]], constant Expand& d [[buffer(3)]],
                            uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  for (uint c=0;c<4;c++) stream[ulong(i.y)*d.stream_stride+ulong(c)*d.width+i.x]=x[ulong(i.y)*d.x_stride+i.x];
  if (i.x==0) for (uint c=0;c<4;c++) pre[ulong(i.y)*d.pre_stride+c]=c==0?1.0f:0.0f;
}

struct MixDense { uint rows, k, n, x_stride; };
static_assert(sizeof(MixDense) == 16, "MixDense must mirror mhc.zig's MixDense");
// One threadgroup a column: the 24 mix outputs read the whole 20480-wide projection once, coalesced.
kernel void dsv41_mix_dense(const device ushort* x [[buffer(0)]],
                            const device float* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant MixDense& d [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[512];
  const uint j = tg.x, row = tg.y;
  if (j >= d.n || row >= d.rows) return;
  const device ushort* xr = x + ulong(row) * d.x_stride;
  const device float* wr = w + ulong(j) * d.k;
  float acc = 0.0f;
  for (uint k = t; k < d.k; k += 512) acc += bf(xr[k]) * wr[k];
  sums[t] = acc;
  for (uint offset = 256; offset >= 32; offset >>= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < offset) { acc += sums[t + offset]; sums[t] = acc; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    for (uint offset = 16; offset > 0; offset >>= 1) acc += simd_shuffle_down(acc, offset);
    if (lane == 0) y[ulong(row) * d.n + j] = acc;
  }
}

struct ColDense { uint rows, k, n, x_stride, bf16_w; };
static_assert(sizeof(ColDense) == 20, "ColDense must mirror mhc.zig's ColDense");
// One threadgroup a column, for the narrow projections decode uses: out[row][j] = x[row,:] . w[j,:].
kernel void dsv41_dense_col(const device ushort* x [[buffer(0)]],
                            const device ushort* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant ColDense& d [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[512];
  const uint j = tg.x, row = tg.y;
  if (j >= d.n || row >= d.rows) return;
  const device ushort* xr = x + ulong(row) * d.x_stride;
  const device ushort* wr = w + ulong(j) * d.k;
  float acc = 0.0f;
  for (uint k = t; k < d.k; k += 512)
    acc += bf(xr[k]) * (d.bf16_w ? bf(wr[k]) : float(as_type<half>(wr[k])));
  sums[t] = acc;
  for (uint offset = 256; offset >= 32; offset >>= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < offset) { acc += sums[t + offset]; sums[t] = acc; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    for (uint offset = 16; offset > 0; offset >>= 1) acc += simd_shuffle_down(acc, offset);
    if (lane == 0) y[ulong(row) * d.n + j] = acc;
  }
}

struct GateTree { uint rows, k, n, x_stride, bf16_w; };
static_assert(sizeof(GateTree) == 20, "GateTree must mirror mhc.zig's GateTree");
// A bit-exact tiled twin of dsv41_dense_col. The shipped kernel spends one 512-thread threadgroup on one output
// cell and rereads the activation row once a column (14 GB of x a layer at prompt rows). This kernel computes a
// 4x4 cell tile per 128-thread threadgroup, stages a 512-k chunk of four activation rows and four weight rows in
// threadgroup memory, and reproduces the shipped sum exactly: each lane owns the sixteen partials
// t = lane + 32*i, so the four threadgroup tree levels over 512 become register adds (V[i] += V[i+8], then
// V[i+4], V[i+2], V[i+1]), after which every lane holds the shipped sums[t] for t = lane, and the last five
// levels are the shipped simd_shuffle_down over lanes 16, 8, 4, 2, 1. No barrier is in the reduction.
kernel void dsv41_gate_tree(const device ushort* x [[buffer(0)]],
                            const device ushort* w [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant GateTree& d [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row0 = tg.y * 4u, col0 = tg.x * 4u;
  if (row0 >= d.rows || col0 >= d.n) return;
  const uint nrow = min(4u, d.rows - row0), ncol = min(4u, d.n - col0);
  threadgroup ushort xs[4][512];
  threadgroup ushort ws[4][512];
  float V[4][16];
  for (uint r = 0; r < 4u; ++r)
    for (uint i = 0; i < 16u; ++i) V[r][i] = 0.0f;
  for (uint c = 0; c < d.k; c += 512u) {
    for (uint idx = t; idx < 2048u; idx += 128u) {
      const uint r = idx >> 9u, col = idx & 511u, kk = c + col;
      xs[r][col] = (r < nrow and kk < d.k) ? x[ulong(row0 + r) * d.x_stride + kk] : ushort(0);
      ws[r][col] = (r < ncol and kk < d.k) ? w[ulong(col0 + r) * d.k + kk] : ushort(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < ncol) {
      for (uint r = 0; r < nrow; ++r) {
        for (uint i = 0; i < 16u; ++i) {
          const uint p = lane + 32u * i, kk = c + p;
          if (kk < d.k) {
            const float xv = bf(xs[r][p]);
            const float wv = d.bf16_w ? bf(ws[sg][p]) : float(as_type<half>(ws[sg][p]));
            V[r][i] += xv * wv;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg < ncol) {
    for (uint r = 0; r < nrow; ++r) {
      for (uint h = 8u; h >= 1u; h >>= 1u)
        for (uint i = 0; i < h; ++i) V[r][i] += V[r][i + h];
      float acc = V[r][0];
      acc += simd_shuffle_down(acc, 16);
      acc += simd_shuffle_down(acc, 8);
      acc += simd_shuffle_down(acc, 4);
      acc += simd_shuffle_down(acc, 2);
      acc += simd_shuffle_down(acc, 1);
      if (lane == 0) y[ulong(row0 + r) * d.n + col0 + sg] = acc;
    }
  }
}

struct GateTree4 { uint rows, k, n, x_stride, bf16_w; };
static_assert(sizeof(GateTree4) == 20, "GateTree4 must mirror mhc.zig's GateTree4");
// The vectorized bit-exact gate: one warp a 2x2 cell tile, no threadgroup memory and no barrier. Lane l owns the
// sixteen partials t = 16*l + i, contiguous, so a lane reads one 32-byte run of a row per 512-k chunk (two uint4
// loads) instead of sixteen 2-byte gathers. The shipped reduction order is reproduced exactly: the 512-partial
// tree levels 256/128/64/32 are simd_shuffle_xor 16/8/4/2 over the sixteen registers, the level-16 step is
// simd_shuffle_xor 1, and the last four levels are the register adds V[i] += V[i+8], V[i+4], V[i+2], V[i+1] --
// the same pairs, in the same order, as dsv41_dense_col's threadgroup tree followed by its warp tree.
inline float tf_bf_h(ushort x) { return as_type<float>(uint(x) << 16); }
inline float tf_w_h(ushort x, uint bf16_w) { return bf16_w ? as_type<float>(uint(x) << 16) : float(as_type<half>(x)); }
inline ushort tf_h4(const uint4& v, uint j) {
  const uint w = (j >> 1) == 0u ? v.x : (j >> 1) == 1u ? v.y : (j >> 1) == 2u ? v.z : v.w;
  return ushort((j & 1u) ? (w >> 16) : (w & 0xffffu));
}
// ROUND-349: the same bit-exact gate on a wider threadgroup with the x and w tiles staged in threadgroup
// memory. The shipped gate is a warp a 2x2 cell tile, so the whole x (37 MB) is re-read by each of the 192
// column pairs and the whole w (3.93 MB) by each of the 1,810 row pairs: 14.2 GB a layer, 2.46 TB/s of L2 --
// the box's L2 rate, while the arithmetic is 1.23 TMAC/s and the ALU floor is 0.55 ms. A threadgroup that covers
// eight rows and four columns, with the tiles staged once, cuts that traffic 2.7x.
//
// The order is untouched: a lane still owns the sixteen consecutive partials t = 16*lane + i of every 512-k
// chunk, still accumulates the chunks in order, and the tree is the same (simd_shuffle_xor 16/8/4/2 over the
// sixteen registers, then shuffle_xor 1, then the register adds V[i] += V[i+8], V[i+4], V[i+2], V[i+1]). So the
// gate's logits are bit-identical and the router's picks cannot move.
#define TF_GATE_R 8u
#define TF_GATE_C 4u
#define TF_GATE_BK 512u
#define TF_GATE_WARPS 8u
kernel void dsv41_gate_tree8(const device ushort* x [[buffer(0)]],
                             const device ushort* w [[buffer(1)]],
                             device float* y [[buffer(2)]],
                             constant GateTree4& d [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 xt4[TF_GATE_R * (TF_GATE_BK / 8u)];
  threadgroup uint4 wt4[TF_GATE_C * (TF_GATE_BK / 8u)];
  const uint row0 = tg.y * TF_GATE_R, col0 = tg.x * TF_GATE_C;
  if (row0 >= d.rows || col0 >= d.n) return;
  // Eight warps: four row pairs by two column pairs, each warp the shipped 2x2 cell tile.
  const uint wr = (sg / 2u) * 2u, wc = (sg % 2u) * 2u;
  const uint r0 = row0 + wr, c0 = col0 + wc;
  const uint nr = min(2u, d.rows - r0), nc = min(2u, d.n - c0);
  float V[4][16];
  for (uint c4 = 0; c4 < 4u; ++c4)
    for (uint i = 0; i < 16u; ++i) V[c4][i] = 0.0f;
  const uint base = 16u * lane;  // this lane's first t in every 512-k chunk
  for (uint c = 0; c < d.k; c += TF_GATE_BK) {
    const uint avail = d.k - c < TF_GATE_BK ? d.k - c : TF_GATE_BK;
    // Stage the threadgroup's tiles: eight rows of x and four columns of w, once a chunk. Sixteen bytes a
    // thread (eight ushorts), so the global reads coalesce; a scalar copy here is eight times the traffic.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      constexpr uint XV = TF_GATE_BK / 8u;   // the uint4 slots a row
      for (uint i = tid; i < TF_GATE_R * XV; i += TF_GATE_WARPS * 32u) {
        const uint r = i / XV, cc = (i % XV) * 8u;
        uint4 v = uint4(0u);
        if (r < d.rows - row0) {
          const device ushort* p = x + (row0 + r) * d.x_stride + c + cc;
          if (cc + 8u <= avail) {
            v = *(const device uint4*)p;
          } else {
            ushort t8[8];
            for (uint q = 0; q < 8u; ++q) t8[q] = (cc + q < avail) ? p[q] : 0;
            v = uint4(t8[0] | (uint(t8[1]) << 16), t8[2] | (uint(t8[3]) << 16), t8[4] | (uint(t8[5]) << 16), t8[6] | (uint(t8[7]) << 16));
          }
        }
        xt4[i] = v;
      }
      for (uint i = tid; i < TF_GATE_C * XV; i += TF_GATE_WARPS * 32u) {
        const uint cc = i / XV, kk = (i % XV) * 8u;
        uint4 v = uint4(0u);
        if (cc < d.n - col0) {
          const device ushort* p = w + (col0 + cc) * d.k + c + kk;
          if (kk + 8u <= avail) {
            v = *(const device uint4*)p;
          } else {
            ushort t8[8];
            for (uint q = 0; q < 8u; ++q) t8[q] = (kk + q < avail) ? p[q] : 0;
            v = uint4(t8[0] | (uint(t8[1]) << 16), t8[2] | (uint(t8[3]) << 16), t8[4] | (uint(t8[5]) << 16), t8[6] | (uint(t8[7]) << 16));
          }
        }
        wt4[i] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[2][16];
    float wv[2][16];
    const bool live = base < avail;
    const bool full = base + 16u <= avail;
    for (uint r = 0; r < 2u; ++r) {
      if (r < nr and live) {
        const threadgroup ushort* p = reinterpret_cast<const threadgroup ushort*>(xt4 + (wr + r) * (TF_GATE_BK / 8u)) + base;
        const uint4 a = *(const threadgroup uint4*)p, b = *(const threadgroup uint4*)(p + 8);
        for (uint i = 0; i < 16u; ++i) xv[r][i] = (full or base + i < avail) ? tf_bf_h(tf_h4(i < 8u ? a : b, i & 7u)) : 0.0f;
      } else {
        for (uint i = 0; i < 16u; ++i) xv[r][i] = 0.0f;
      }
    }
    for (uint j = 0; j < 2u; ++j) {
      if (j < nc and live) {
        const threadgroup ushort* p = reinterpret_cast<const threadgroup ushort*>(wt4 + (wc + j) * (TF_GATE_BK / 8u)) + base;
        const uint4 a = *(const threadgroup uint4*)p, b = *(const threadgroup uint4*)(p + 8);
        for (uint i = 0; i < 16u; ++i) wv[j][i] = (full or base + i < avail) ? tf_w_h(tf_h4(i < 8u ? a : b, i & 7u), d.bf16_w) : 0.0f;
      } else {
        for (uint i = 0; i < 16u; ++i) wv[j][i] = 0.0f;
      }
    }
    for (uint r = 0; r < 2u; ++r)
      for (uint j = 0; j < 2u; ++j)
        for (uint i = 0; i < 16u; ++i) V[r * 2u + j][i] += xv[r][i] * wv[j][i];
  }
  for (uint r = 0; r < 2u; ++r) {
    if (r >= nr) continue;
    for (uint j = 0; j < 2u; ++j) {
      if (j >= nc) continue;
      float* v = V[r * 2u + j];
      for (uint i = 0; i < 16u; ++i) {
        v[i] += simd_shuffle_xor(v[i], 16);
        v[i] += simd_shuffle_xor(v[i], 8);
        v[i] += simd_shuffle_xor(v[i], 4);
        v[i] += simd_shuffle_xor(v[i], 2);
        v[i] += simd_shuffle_xor(v[i], 1);
      }
      for (uint h = 8u; h >= 1u; h >>= 1u)
        for (uint i = 0; i < h; ++i) v[i] += v[i + h];
      if (lane == 0) y[ulong(r0 + r) * d.n + c0 + j] = v[0];
    }
  }
}

kernel void dsv41_gate_tree4(const device ushort* x [[buffer(0)]],
                             const device ushort* w [[buffer(1)]],
                             device float* y [[buffer(2)]],
                             constant GateTree4& d [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = tg.y * 2u, col0 = tg.x * 2u;
  if (row0 >= d.rows || col0 >= d.n) return;
  const uint nr = min(2u, d.rows - row0), nc = min(2u, d.n - col0);
  float V[4][16];
  for (uint c4 = 0; c4 < 4u; ++c4)
    for (uint i = 0; i < 16u; ++i) V[c4][i] = 0.0f;
  const uint base = 16u * lane;  // this lane's first t in every 512-k chunk
  for (uint c = 0; c < d.k; c += 512u) {
    const uint avail = d.k - c < 512u ? d.k - c : 512u;
    float xv[2][16];
    float wv[2][16];
    const bool live = base < avail;
    const bool full = base + 16u <= avail;
    for (uint r = 0; r < 2u; ++r) {
      if (r < nr and live) {
        const device ushort* p = x + ulong(row0 + r) * d.x_stride + c + base;
        const uint4 a = *(const device uint4*)p, b = *(const device uint4*)(p + 8);
        for (uint i = 0; i < 16u; ++i) xv[r][i] = (full or base + i < avail) ? tf_bf_h(tf_h4(i < 8u ? a : b, i & 7u)) : 0.0f;
      } else {
        for (uint i = 0; i < 16u; ++i) xv[r][i] = 0.0f;
      }
    }
    for (uint j = 0; j < 2u; ++j) {
      if (j < nc and live) {
        const device ushort* p = w + ulong(col0 + j) * d.k + c + base;
        const uint4 a = *(const device uint4*)p, b = *(const device uint4*)(p + 8);
        for (uint i = 0; i < 16u; ++i) wv[j][i] = (full or base + i < avail) ? tf_w_h(tf_h4(i < 8u ? a : b, i & 7u), d.bf16_w) : 0.0f;
      } else {
        for (uint i = 0; i < 16u; ++i) wv[j][i] = 0.0f;
      }
    }
    for (uint r = 0; r < 2u; ++r)
      for (uint j = 0; j < 2u; ++j)
        for (uint i = 0; i < 16u; ++i) V[r * 2u + j][i] += xv[r][i] * wv[j][i];
  }
  for (uint r = 0; r < 2u; ++r) {
    if (r >= nr) continue;
    for (uint j = 0; j < 2u; ++j) {
      if (j >= nc) continue;
      float* v = V[r * 2u + j];
      for (uint i = 0; i < 16u; ++i) {
        v[i] += simd_shuffle_xor(v[i], 16);
        v[i] += simd_shuffle_xor(v[i], 8);
        v[i] += simd_shuffle_xor(v[i], 4);
        v[i] += simd_shuffle_xor(v[i], 2);
        v[i] += simd_shuffle_xor(v[i], 1);
      }
      for (uint h = 8u; h >= 1u; h >>= 1u)
        for (uint i = 0; i < h; ++i) v[i] += v[i + h];
      if (lane == 0) y[ulong(row0 + r) * d.n + col0 + j] = v[0];
    }
  }
}
