#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// 1-16 bf16 rows share each group's weight read; every row's sums and split-K order hold at every width.

#ifndef TF_PRECOMPUTE_SUMS
#define TF_PRECOMPUTE_SUMS 0
#endif

constant constexpr int NT = 32, GS = TF_GROUP, KG = TF_K / GS, NF = NT / 16;
constant constexpr int WP = GS * TF_BITS / 32;

struct tf_raw { uint2 w[WP / 2]; uint4 s[NF]; }; // a lane's packed code words and its scale/bias quads for one group

inline tf_raw tf_lane_fetch(const device uint* W, const device uint4* SB, int tile, int g, int g_end, int n0, int lane,
                            short fn) {
  tf_raw raw;
  const bool ok = g < g_end;
  const device uint2* src = (const device uint2*)(W + (((size_t)tile * KG + (ok ? g : 0)) * NT + lane) * WP);
#if TF_BITS != 4
  for (int i = 0; i < WP / 2; i++) raw.w[i] = ok && n0 + lane < TF_N ? src[i] : uint2(0);
#else
  for (int i = 0; i < WP / 2; i++) raw.w[i] = uint2(0);
#endif
  // Four lanes load the metadata quads and broadcast them; all 32 lanes fetch, even for inactive rows.
  const ushort leader = ushort((fn & 8) | ((fn >> 2) & 1));
  for (int f = 0; f < NF; f++) {
    const uint4 sb = lane == int(leader) && ok && n0 + f * 16 + fn < TF_N
      ? SB[((size_t)(ok ? g : 0) * TF_N + n0 + f * 16 + fn) / 4] : uint4(0);
    raw.s[f] = simd_shuffle(sb, leader);
  }
  return raw;
}

inline float tf_lane_xsum(const device bfloat* X, int m, int g, int M) {
  float acc = 0.0f;
  if (m < M) for (int i = 0; i < GS; i++) acc += float(X[(size_t)m * TF_K + g * GS + i]);
  return acc;
}

#if TF_BITS == 4
#define TF_STAGE_CODES(raw)
#define TF_GROUP_PRODUCT(g) \
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b( \
      (device uchar*)W + ((size_t)tile * KG + (g)) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT)); \
    auto a = tA.slice((g) * GS, 0); \
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(a, b, P);
#else
inline void tf_stage_codes(threadgroup uint* stage, thread const tf_raw& raw, int lane) {
    uint words[WP + 1];
    for (int i = 0; i < WP / 2; i++) { words[2*i] = raw.w[i].x; words[2*i+1] = raw.w[i].y; }
    words[WP] = 0u;
    for (int c = 0; c < GS / 4; c++) {
      uint word = 0u;
      for (int j = 0; j < 4; j++) {
        const int bit = (4*c + j) * TF_BITS, i = bit >> 5, sh = bit & 31;
        uint code = words[i] >> sh;
        if (sh + TF_BITS > 32) code |= words[i+1] << (32-sh);
        word |= (code & ((1u << TF_BITS)-1u)) << (8*j);
      }
      stage[lane * (GS / 4) + c] = word;
    }
}
#define TF_STAGE_CODES(raw) tf_stage_codes(stage, raw, lane);
#define TF_GROUP_PRODUCT(g) \
    simdgroup_barrier(mem_flags::mem_threadgroup); \
    auto a = tA.slice((g) * GS, 0); \
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(a, b, P); \
    simdgroup_barrier(mem_flags::mem_threadgroup);
#endif

#if TF_PRECOMPUTE_SUMS
#define TF_SUM_VALUE(row, g) XS[(g) * 16 + (row)]
#else
#define TF_SUM_VALUE(row, g) tf_lane_xsum(X, row, g, M)
#endif

// Each group loads packed codes or stages widened codes, performs the tensor product, then applies the scale/bias FMAs.
#define TF_LANE_GROUP(g, raw)                                                                                  \
  {                                                                                                            \
    TF_STAGE_CODES(raw) \
    float s[NF][4], bb[NF][4];                                                                                 \
    for (int f = 0; f < NF; f++) {                                                                             \
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(raw.s[f]);                                              \
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }               \
    }                                                                                                          \
    raw = tf_lane_fetch(W, sbv, tile, (g) + TF_PF, g_end, n0, lane, fn);                                       \
    TF_GROUP_PRODUCT(g) \
    const float xs0 = TF_SUM_VALUE(fm, g); \
    const float xs1 = TF_SUM_VALUE(fm + 8, g); \
    for (int f = 0; f < NF; f++)                                                                               \
      for (int r = 0; r < 2; r++)                                                                              \
        for (int j = 0; j < 4; j++) {                                                                          \
          const int i = f * 8 + r * 4 + j;                                                                     \
          C[i] = fma(s[f][j], P[i], fma(bb[f][j], r ? xs1 : xs0, C[i]));                                       \
        }                                                                                                      \
  }

[[kernel]] void tf_lane(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* SBt [[buffer(2)]], const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]],
#if TF_PRECOMPUTE_SUMS
    const device float* XS [[buffer(5)]],
#endif
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]],
    uint tgx [[threadgroup_position_in_grid]]) {
  const int lane = int(lanei), sg = int(sgi);
  const short qid = short(lane) >> 2;
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
  const int M = mdims[0];
  const int tile = tf_tile(int(tgx));
  const int n0 = tile * NT;
  const int g_begin = TF_G0 + (sg * TF_GN) / TF_SK, g_end = TF_G0 + ((sg + 1) * TF_GN) / TF_SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(TF_K, M));
#if TF_BITS != 4
  threadgroup uint stage_all[TF_SK * NT * (GS / 4)];
  threadgroup uint* stage = stage_all + sg * NT * (GS / 4);
  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, NT));
#endif
  const device uint4* sbv = (const device uint4*)SBt;
  float C[NF * 8];
  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
  tf_raw r0 = tf_lane_fetch(W, sbv, tile, g_begin, g_end, n0, lane, fn);
#if TF_PF == 2
  tf_raw r1 = tf_lane_fetch(W, sbv, tile, g_begin + 1, g_end, n0, lane, fn);
  for (int g = g_begin; g < g_end; g += 2) {
    TF_LANE_GROUP(g, r0)
    if (g + 1 < g_end) TF_LANE_GROUP(g + 1, r1)
  }
#else
  for (int g = g_begin; g < g_end; g++) TF_LANE_GROUP(g, r0)
#endif
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * NF * 8 * 32];
  if (TF_SK > 1) {
    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
      for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
  }
  if (sg == 0)
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        const int nn = n0 + f * 16 + fn;
        if (m < M && nn < TF_N)
          for (int j = 0; j < 4; j++) Y[(size_t)m * TF_N + nn + j] = static_cast<TF_OUT>(C[f * 8 + r * 4 + j]);
      }
}

#if TF_PRECOMPUTE_SUMS
kernel void tf_lane_sums(const device bfloat* X [[buffer(0)]],constant int& rows [[buffer(1)]],device float* XS [[buffer(2)]],constant int& stride [[buffer(3)]],uint index [[thread_position_in_grid]]) {
  const int row=int(index)%stride,group=int(index)/stride;
  if(group>=KG)return;
  XS[index]=tf_lane_xsum(X,row,group,rows);
}
#endif

#if TF_BITS == 4
kernel void tf_lane_pair(const device bfloat* X [[buffer(0)]],const device uint* W [[buffer(1)]],const device bfloat* SB [[buffer(2)]],const device int* mdims [[buffer(3)]],device TF_OUT* Y [[buffer(4)]],
#if TF_PRECOMPUTE_SUMS
const device float* XS [[buffer(5)]],
#endif
uint group_id [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint thread_id [[thread_position_in_threadgroup]],uint tile_id [[threadgroup_position_in_grid]]) {
  const uint sg=group_id/2,tip=thread_id-sg*64;
  const int M=mdims[0],tile=tf_tile(int(tile_id)),n0=tile*NT;
  const int first=TF_G0+(int(sg)*TF_GN)/TF_SK,last=TF_G0+((int(sg)+1)*TF_GN)/TF_SK;
  constexpr auto descriptor=matmul2d_descriptor(16,NT,GS,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<2>> op;
  tensor<device bfloat,dextents<int32_t,2>,tensor_inline> a((device bfloat*)X,dextents<int32_t,2>(TF_K,M));
  tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline> b0((device uchar*)W,dextents<int32_t,2>(GS,NT));
  auto a0=a.slice(0,0);
  auto p=op.template get_destination_cooperative_tensor<decltype(a0),decltype(b0),float>();
  short column[8],row[8];float value[8];
  for(int i=0;i<8;i++){auto index=p.get_multidimensional_index(i);column[i]=index[0];row[i]=index[1];value[i]=0.0f;}
  for(int group=first;group<last;group++) {
    tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline> b((device uchar*)W+((size_t)tile*KG+group)*(NT*GS/2),dextents<int32_t,2>(GS,NT));
    auto x=a.slice(group*GS,0);op.run(x,b,p);
    for(int i=0;i<8;i++) {
      const int n=n0+column[i];
      const float scale=float(SB[(size_t(group)*TF_N+n)*2]),bias=float(SB[(size_t(group)*TF_N+n)*2+1]);
#if TF_PRECOMPUTE_SUMS
      const float sum=XS[group*16+row[i]];
#else
      const float sum=tf_lane_xsum(X,row[i],group,M);
#endif
      value[i]=fma(scale,p[i],fma(bias,sum,value[i]));
    }
  }
  threadgroup float partial[(TF_SK>1?TF_SK-1:1)*8*64];
  if(TF_SK>1) {
    if(sg>0) for(int i=0;i<8;i++) partial[((sg-1)*8+i)*64+tip]=value[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) for(int part=1;part<TF_SK;part++)for(int i=0;i<8;i++)value[i]+=partial[((part-1)*8+i)*64+tip];
  }
  if(sg==0)for(int i=0;i<8;i++)if(row[i]<M)Y[size_t(row[i])*TF_N+n0+column[i]]=TF_OUT(value[i]);
}
#endif

#if TF_BITS == 4 && defined(TF_REG)
struct tf_reg_load { uint4 w0, w1, q0, q1; float x0, x1; }; // a lane's codes, its two metadata quads and its two row sums for one group

inline tf_reg_load tf_reg_fetch(const device uint4* wl, const device uint4* sbv, const device float* XS, int g, int n, int m, int stride) {
  return {wl[(size_t)g * 64], wl[(size_t)g * 64 + 32], sbv[((size_t)g * TF_N + n) / 4], sbv[((size_t)g * TF_N + n + 16) / 4], XS[g * stride + m], XS[g * stride + m + 8]};
}

// tf_lane's arithmetic with codes unpacked straight into tensor-op registers: every word equals tf_lane_pair's.
[[kernel]] void tf_lane_reg(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], const device bfloat* SB [[buffer(2)]],
    const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]], const device float* XS [[buffer(5)]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int r0 = int(tg.y) * 16, rows = mdims[0] - r0;
  if (rows <= 0) return;
  X += (size_t)r0 * TF_K;
  Y += (size_t)r0 * TF_N;
  XS += r0;
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1], tile = tf_tile(int(tg.x)), n0 = tile * NT;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const int first = TF_G0 + (ks * TF_GN) / TF_SK, last = TF_G0 + ((ks + 1) * TF_GN) / TF_SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(TF_K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(cb), float>();
  thread half2* b2 = (thread half2*)&cb[0];
  const device uint4* wl = W + (size_t)tile * KG * 64 + lane;
  const device uint4* sbv = (const device uint4*)SB;
  const half2 bias = half2(1024.0h);
  float C[16];
  for (int i = 0; i < 16; i++) C[i] = 0.0f;
  tf_reg_load next = {};
  if (first < last) next = tf_reg_fetch(wl, sbv, XS, first, n0 + nb, mb, stride);
  for (int g = first; g < last; g++) {
    const tf_reg_load cur = next;
    for (int w = 0; w < 8; w++) {
      // 1024 + code as fp16 bits, codes (0,4) (1,5) (2,6) (3,7) of a word; subtracting 1024 is exact
      const uint word = w < 4 ? cur.w0[w] : cur.w1[w - 4], lo = word & 0x0F0F0F0Fu, hi = (word >> 4) & 0x0F0F0F0Fu;
      const int e = 16 * (w & 1) + 2 * (w >> 1);
      b2[e] = as_type<half2>((lo & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 1] = as_type<half2>((hi & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 8] = as_type<half2>(((lo >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 9] = as_type<half2>(((hi >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
    }
    if (g + 1 < last) next = tf_reg_fetch(wl, sbv, XS, g + 1, n0 + nb, mb, stride);
    auto x = a.slice(g * GS, 0);
    op.run(x, cb, P);
    for (int j = 0; j < 4; j++) {
      const float s0 = as_type<float>(cur.q0[j] << 16), b0 = as_type<float>(cur.q0[j] & 0xFFFF0000u);
      const float s1 = as_type<float>(cur.q1[j] << 16), b1 = as_type<float>(cur.q1[j] & 0xFFFF0000u);
      C[j] = fma(s0, P[j], fma(b0, cur.x0, C[j]));
      C[4 + j] = fma(s0, P[4 + j], fma(b0, cur.x1, C[4 + j]));
      C[8 + j] = fma(s1, P[8 + j], fma(b1, cur.x0, C[8 + j]));
      C[12 + j] = fma(s1, P[12 + j], fma(b1, cur.x1, C[12 + j]));
    }
  }
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * 16 * 32];
  if (TF_SK > 1) {
    if (ks > 0) for (int i = 0; i < 16; i++) part[((ks - 1) * 16 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < 16; i++) C[i] += part[((s2 - 1) * 16 + i) * 32 + lane];
  }
  if (ks == 0)
    for (int i = 0; i < 16; i++) {
      const int m = mb + 8 * ((i >> 2) & 1), n = n0 + nb + 16 * (i >> 3) + (i & 3);
      if (m < M && n < TF_N) Y[(size_t)m * TF_N + n] = static_cast<TF_OUT>(C[i]);
    }
}
#endif
