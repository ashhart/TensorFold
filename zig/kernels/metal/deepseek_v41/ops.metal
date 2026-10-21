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
kernel void dsv41_embedding(const device uint* ids [[buffer(0)]], const device ushort* table [[buffer(1)]],
                            device ushort* y [[buffer(2)]], constant Embedding& d [[buffer(3)]],
                            uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  ushort v=table[ulong(ids[i.y])*d.table_stride+i.x];
  for (uint c=0;c<d.copies;c++) y[ulong(i.y)*d.y_stride+ulong(c)*d.width+i.x]=v;
}
struct Activation { uint rows,width,input_stride,output_stride,weighted,input_f32; };
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
struct Combine { uint rows,width,routed_row_stride,slot_stride,shared_stride,y_stride,input_f32,id_stride; };
kernel void dsv41_combine(const device uchar* routed [[buffer(0)]], const device uchar* shared [[buffer(1)]],
                          const device uint* ids [[buffer(2)]], device ushort* y [[buffer(3)]],
                          constant Combine& d [[buffer(4)]],
                          uint2 i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint slots[6];
  if (t==0) {
    for (uint a=0;a<6;a++) slots[a]=a;
    for (uint a=1;a<6;a++) {
      uint slot=slots[a], j=a;
      while (j>0 && ids[ulong(i.y)*d.id_stride+slots[j-1]]>ids[ulong(i.y)*d.id_stride+slot]) {
        slots[j]=slots[j-1]; j--;
      }
      slots[j]=slot;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i.x>=d.width || i.y>=d.rows) return;
  for (uint slot=0;slot<6;slot++) if (ids[ulong(i.y)*d.id_stride+slot]>=384) {
    y[ulong(i.y)*d.y_stride+i.x]=rb(NAN);
    return;
  }
  float s=0;
  for (uint rank=0;rank<6;rank++)
    s+=rd(routed,ulong(i.y)*d.routed_row_stride+ulong(slots[rank])*d.slot_stride+i.x,d.input_f32);
  s+=rd(shared,ulong(i.y)*d.shared_stride+i.x,d.input_f32);
  y[ulong(i.y)*d.y_stride+i.x]=rb(s);
}
