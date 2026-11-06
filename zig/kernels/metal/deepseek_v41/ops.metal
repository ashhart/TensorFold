// Math: deepseek-ai/DeepSeek-V4.1-Flash inference/model.py, revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
#include <metal_stdlib>
using namespace metal;
inline float bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort rb(float x) {
  uint u = as_type<uint>(x);
  if ((u & 0x7fffffff) > 0x7f800000) return ushort((u >> 16) | 0x40);
  return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline float rd(const device uchar* p, ulong i, uint dtype) {
  if (dtype==1) return ((const device float*)p)[i];
  if (dtype==2) return float(((const device half*)p)[i]);
  return bf(((const device ushort*)p)[i]);
}
struct Rms { uint rows, width, x_stride, y_stride; float eps; uint weight_f32; };
static_assert(sizeof(Rms) == 24, "Rms must mirror ops.zig's RmsArgs");
#pragma clang fp contract(off)
kernel void dsv41_rms(const device ushort* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                      device ushort* y [[buffer(2)]], constant Rms& d [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  float s = 0;
  for (uint i = t; i < d.width; i += 256) { float a = bf(x[ulong(row)*d.x_stride+i]); s += a*a; }
  s = simd_sum(s);
  if (lane == 0) sums[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    const float factor = 1.0f / float(d.width);
    if (lane == 0) sums[0] = rsqrt(total * factor + d.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < d.width; i += 256)
    y[ulong(row)*d.y_stride+i] = rb((bf(x[ulong(row)*d.x_stride+i])*sums[0])*rd(w,i,d.weight_f32));
}
#pragma clang fp contract(on)
// Each SIMD group emits four columns; compensated dots do not depend on token count.

// Precision: 0=BF16, 1=FP32, 2=FP16; FP16 weights never round through BF16.
inline void dense_add(thread float& high, thread float& low, float value) {
  float next = high + value;
  float borrowed = next - high;
  low += (high - (next - borrowed)) + (value - borrowed);
  high = next;
}
struct Dense { uint rows, k, n, groups, x_stride, w_stride, y_stride, x_dtype, w_dtype, y_dtype; };
static_assert(sizeof(Dense) == 40, "Dense must mirror ops.zig's Dense");
kernel void dsv41_dense_col1(const device uchar* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                             device uchar* y [[buffer(2)]], constant Dense& d [[buffer(3)]],
                             uint3 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint col = g.x*4 + sg;
  if (col >= d.n) return;
  float a = 0, residual = 0;
  for (uint k = lane; k < d.k; k += 32) {
    float v = rd(x, ulong(g.y)*d.x_stride+ulong(g.z)*d.k+k, d.x_dtype);
    float weight = rd(w, (ulong(g.z)*d.n+col)*d.w_stride+k, d.w_dtype);
    float product = v * weight;
    dense_add(a, residual, product);
    if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v, weight, -product);
  }
  for (uint offset = 16; offset != 0; offset >>= 1) {
    float other = simd_shuffle_xor(a, offset);
    float other_residual = simd_shuffle_xor(residual, offset);
    dense_add(a, residual, other);
    residual += other_residual;
  }
  float v = a + residual;
  if (lane == 0) {
    ulong i = ulong(g.y)*d.y_stride+ulong(g.z)*d.n+col;
    if (d.y_dtype==1) ((device float*)y)[i]=v;
    else if (d.y_dtype==2) ((device half*)y)[i]=half(v);
    else ((device ushort*)y)[i]=rb(v);
  }
}

// The one-column dot with the weight dtype resolved at compile time and four k-steps of loads hoisted. The
// dense_add sequence is the shipped one (lane, lane+32, lane+64, lane+96, ...), so the output is bit-identical;
// only the loads move ahead of the adds. The caller picks k % 128 == 0.
inline float dsv41_hf(ushort u) { return float(as_type<half>(u)); }
#define DSV41_DENSE_COL1_SPEC(NAME, LOADW) \
kernel void NAME(const device uchar* x [[buffer(0)]], const device uchar* w [[buffer(1)]], \
                 device uchar* y [[buffer(2)]], constant Dense& d [[buffer(3)]], \
                 uint3 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
  const uint col = g.x*4 + sg; \
  if (col >= d.n) return; \
  const device ushort* xr = (const device ushort*)x + ulong(g.y)*d.x_stride; \
  const device ushort* wr = (const device ushort*)w + (ulong(g.z)*d.n+col)*d.w_stride; \
  float a = 0, residual = 0; \
  uint k = lane; \
  for (; k + 96 < d.k; k += 128) { \
    float v0 = bf(xr[k]);    float w0 = LOADW(wr[k]); \
    float v1 = bf(xr[k+32]); float w1 = LOADW(wr[k+32]); \
    float v2 = bf(xr[k+64]); float w2 = LOADW(wr[k+64]); \
    float v3 = bf(xr[k+96]); float w3 = LOADW(wr[k+96]); \
    float p0 = v0*w0; dense_add(a, residual, p0); if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v0, w0, -p0); \
    float p1 = v1*w1; dense_add(a, residual, p1); if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v1, w1, -p1); \
    float p2 = v2*w2; dense_add(a, residual, p2); if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v2, w2, -p2); \
    float p3 = v3*w3; dense_add(a, residual, p3); if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v3, w3, -p3); \
  } \
  for (; k < d.k; k += 32) { \
    float v = bf(xr[k]); \
    float weight = LOADW(wr[k]); \
    float product = v * weight; \
    dense_add(a, residual, product); \
    if (d.x_dtype == 1 || d.w_dtype == 1) residual += fma(v, weight, -product); \
  } \
  for (uint offset = 16; offset != 0; offset >>= 1) { \
    float other = simd_shuffle_xor(a, offset); \
    float other_residual = simd_shuffle_xor(residual, offset); \
    dense_add(a, residual, other); \
    residual += other_residual; \
  } \
  float v = a + residual; \
  if (lane == 0) { \
    ulong i = ulong(g.y)*d.y_stride+ulong(g.z)*d.n+col; \
    if (d.y_dtype==1) ((device float*)y)[i]=v; \
    else if (d.y_dtype==2) ((device half*)y)[i]=half(v); \
    else ((device ushort*)y)[i]=rb(v); \
  } \
}
DSV41_DENSE_COL1_SPEC(dsv41_dense_col1_bf16, bf)
DSV41_DENSE_COL1_SPEC(dsv41_dense_col1_f16, dsv41_hf)
kernel void dsv41_dense(const device uchar* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                        device uchar* y [[buffer(2)]], constant Dense& d [[buffer(3)]],
                        uint3 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  uint col = g.x*16 + sg*4;
  float a[4] = {0,0,0,0}, residual[4] = {0,0,0,0};
  for (uint k = lane; k < d.k; k += 32) {
    float v = rd(x,ulong(g.y)*d.x_stride+ulong(g.z)*d.k+k,d.x_dtype);
    for (uint j=0;j<4;j++) if (col+j<d.n) {
      float weight = rd(w,(ulong(g.z)*d.n+col+j)*d.w_stride+k,d.w_dtype);
      float product = v * weight;
      dense_add(a[j], residual[j], product);
      if (d.x_dtype == 1 || d.w_dtype == 1) residual[j] += fma(v, weight, -product);
    }
  }
  for (uint j=0;j<4;j++) {
    for (uint offset = 16; offset != 0; offset >>= 1) {
      float other = simd_shuffle_xor(a[j], offset);
      float other_residual = simd_shuffle_xor(residual[j], offset);
      dense_add(a[j], residual[j], other);
      residual[j] += other_residual;
    }
    float v = a[j] + residual[j];
    if (lane==0 && col+j<d.n) {
      ulong i = ulong(g.y)*d.y_stride+ulong(g.z)*d.n+col+j;
      if (d.y_dtype==1) ((device float*)y)[i]=v;
      else if (d.y_dtype==2) ((device half*)y)[i]=half(v);
      else ((device ushort*)y)[i]=rb(v);
    }
  }
}
struct Embedding { uint rows,width,table_stride,y_stride,copies; };
static_assert(sizeof(Embedding) == 20, "Embedding must mirror ops.zig's Embedding");
kernel void dsv41_embedding(const device uint* ids [[buffer(0)]], const device ushort* table [[buffer(1)]],
                            device ushort* y [[buffer(2)]], constant Embedding& d [[buffer(3)]],
                            uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  ushort v=table[ulong(ids[i.y])*d.table_stride+i.x];
  for (uint c=0;c<d.copies;c++) y[ulong(i.y)*d.y_stride+ulong(c)*d.width+i.x]=v;
}
struct Activation { uint rows,width,input_stride,output_stride,weighted,input_f32; };
static_assert(sizeof(Activation) == 24, "Activation must mirror ops.zig's Activation");
kernel void dsv41_swiglu(const device uchar* gate [[buffer(0)]], const device uchar* up [[buffer(1)]],
                         const device float* weights [[buffer(2)]], device ushort* y [[buffer(3)]],
                         constant Activation& d [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  ulong k=ulong(i.y)*d.input_stride+i.x;
  float a=min(rd(gate,k,d.input_f32),10.0f), b=clamp(rd(up,k,d.input_f32),-10.0f,10.0f);
  float v=(a/(1.0f+exp(-a)))*b;
  if (d.weighted) v=weights[i.y]*v;
  y[ulong(i.y)*d.output_stride+i.x]=rb(v);
}
// The drafter's conditioning: the mean of the hyper-connection's four stream copies, one row into a slot.
struct StreamMean { uint rows, width, stream_stride, out_stride, slot, row_base, ring; };
static_assert(sizeof(StreamMean) == 28, "StreamMean must mirror ops.zig's StreamMean");
kernel void dsv41_stream_mean(const device ushort* stream [[buffer(0)]], device ushort* out [[buffer(1)]],
                              constant StreamMean& d [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= d.width || i.y >= d.rows) return;
  const ulong s = ulong(i.y) * d.stream_stride + i.x;
  const float mean = (bf(stream[s]) + bf(stream[s + d.width]) + bf(stream[s + 2 * ulong(d.width)]) + bf(stream[s + 3 * ulong(d.width)])) * 0.25f;
  // The drafter's conditioning reads the three target layers as one 15,360-wide row: a layer a third.
  const ulong row = (ulong(d.row_base) + i.y) % ulong(d.ring);
  out[row * ulong(3u * d.out_stride) + ulong(d.slot) * ulong(d.out_stride) + i.x] = rb(mean);
}
// The drafter's input: a projected hidden state plus a token's embedding, both bf16 rows.
struct Add { uint rows, width, a_stride, b_stride, y_stride; };
static_assert(sizeof(Add) == 20, "Add must mirror ops.zig's Add");
kernel void dsv41_add(const device ushort* a [[buffer(0)]], const device ushort* b [[buffer(1)]],
                      device ushort* y [[buffer(2)]], constant Add& d [[buffer(3)]],
                      uint2 i [[thread_position_in_grid]]) {
  if (i.x >= d.width || i.y >= d.rows) return;
  y[ulong(i.y) * d.y_stride + i.x] = rb(bf(a[ulong(i.y) * d.a_stride + i.x]) + bf(b[ulong(i.y) * d.b_stride + i.x]));
}
struct Combine { uint rows,width,routed_row_stride,slot_stride,shared_stride,y_stride,input_f32,id_stride; };
static_assert(sizeof(Combine) == 32, "Combine must mirror ops.zig's Combine");
kernel void dsv41_combine(const device uchar* routed [[buffer(0)]], const device uchar* shared [[buffer(1)]],
                          const device uint* ids [[buffer(2)]], device ushort* y [[buffer(3)]],
                          constant Combine& d [[buffer(4)]],
                          uint2 i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint slots[6];
  if (t==0) {
    for (uint a=0;a<d.id_stride;a++) slots[a]=a;
    for (uint a=1;a<d.id_stride;a++) {
      uint slot=slots[a], j=a;
      while (j>0 && ids[ulong(i.y)*d.id_stride+slots[j-1]]>ids[ulong(i.y)*d.id_stride+slot]) {
        slots[j]=slots[j-1]; j--;
      }
      slots[j]=slot;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i.x>=d.width || i.y>=d.rows) return;
  for (uint slot=0;slot<d.id_stride;slot++) if (ids[ulong(i.y)*d.id_stride+slot]>=384) {
    y[ulong(i.y)*d.y_stride+i.x]=rb(NAN);
    return;
  }
  float s=0;
  for (uint rank=0;rank<d.id_stride;rank++)
    s+=rd(routed,ulong(i.y)*d.routed_row_stride+ulong(slots[rank])*d.slot_stride+i.x,d.input_f32);
  s+=rd(shared,ulong(i.y)*d.shared_stride+i.x,d.input_f32);
  y[ulong(i.y)*d.y_stride+i.x]=rb(s);
}

struct Copy { uint rows, width, x_stride, y_stride; };
static_assert(sizeof(Copy) == 16, "Copy must mirror ops.zig's Copy");
// A row-for-row copy of BF16 rows: the drafter taps a layer's hidden state out of a buffer the forward reuses.
kernel void dsv41_copy(const device ushort* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                       constant Copy& d [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= d.width || i.y >= d.rows) return;
  y[ulong(i.y) * d.y_stride + i.x] = x[ulong(i.y) * d.x_stride + i.x];
}

// The draft chain's Markov step (the port's ds_markov): row `step`'s logits plus the rank-256 bias
// dot(embed[token before], head[vocab]); each threadgroup's best (value, id) into PART.
struct MarkovArgs { int vocab; int rank; int step; int pad; };
static_assert(sizeof(MarkovArgs) == 16, "MarkovArgs must mirror ops.zig's Markov");
kernel void dsv41_markov(const device float* LOGITS [[buffer(0)]], const device ushort* EMB [[buffer(1)]],
                        const device half* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                        device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = bf(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab) {
    const device half* h = HEAD + size_t(v) * a.rank;
    float bias = 0.0f;
    // A.pad = 1 turns the Markov bigram bias OFF (the arm: the head-s own argmax).
    if (a.pad == 0) for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j]);
    val = LOGITS[size_t(a.step) * a.vocab + v] + bias;
    id = v;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) { val = ov; id = oi; }
  }
  if (lane == 0) { bv[sg] = val; bi[sg] = id; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++) if (bv[k] > m || (bv[k] == m && bi[k] < mi)) { m = bv[k]; mi = bi[k]; }
    PART[tg] = float2(m, as_type<float>(mi));
  }
}

// The step's token: the best of the threadgroups' bests into IDS[step + 1].
kernel void dsv41_markov_pick(const device float2* PART [[buffer(0)]], device uint* IDS [[buffer(1)]],
                              constant int2& n_step [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  float m = -INFINITY;
  int mi = 0x7fffffff;
  for (int k = 0; k < n_step.x; k++) {
    const float v = PART[k].x;
    const int i = as_type<int>(PART[k].y);
    if (v > m || (v == m && i < mi)) { m = v; mi = i; }
  }
  IDS[n_step.y + 1] = uint(mi);
}

// F16 -> BF16, round to nearest even: the bridge the tensor-unit GEMM needs when a pack stores a gate F16
// (gemm_nax binds A, B and D to one element type). One element a thread.
kernel void dsv41_f16_to_bf16(const device ushort *x [[buffer(0)]], device ushort *y [[buffer(1)]],
                              constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const uint u = as_type<uint>(float(as_type<half>(x[i])));
  y[i] = ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// BF16 -> F16, the exact widening of a bf16 significand into f16's 11 bits (f16's narrower exponent is the only
// place a value can move). The f16 router-gate GEMM needs both operands one element type, and the pack's gate is
// f16: converting the activation keeps the weight's own 10 bits instead of rounding the weight to bf16.
kernel void dsv41_bf16_to_f16(const device ushort *x [[buffer(0)]], device ushort *y [[buffer(1)]],
                              constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  y[i] = as_type<ushort>(half(as_type<float>(uint(x[i]) << 16)));
}

kernel void dsv41_dump_f32(const device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                           constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = x[i];
}
kernel void dsv41_dump_bf16(const device ushort *x [[buffer(0)]], device float *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = as_type<float>(uint(x[i]) << 16);
}
