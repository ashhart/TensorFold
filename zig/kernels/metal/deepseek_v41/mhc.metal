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
kernel void dsv41_hc_collapse(const device ushort* x [[buffer(0)]], const device float* previous_pre [[buffer(1)]],
                              device ushort* y [[buffer(2)]], constant Collapse& d [[buffer(3)]],
                              uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  float s=0;
  for (uint c=0;c<4;c++) s+=previous_pre[ulong(i.y)*d.pre_stride+c]*bf(x[ulong(i.y)*d.stream_stride+ulong(c)*d.width+i.x]);
  y[ulong(i.y)*d.y_stride+i.x]=rb(s);
}
struct Post { uint rows,width,branch_stride,residual_stride,y_stride,coefficient_stride,comb_stride; };
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
kernel void dsv41_hc_expand(const device ushort* x [[buffer(0)]], device ushort* stream [[buffer(1)]],
                            device float* pre [[buffer(2)]], constant Expand& d [[buffer(3)]],
                            uint2 i [[thread_position_in_grid]]) {
  if (i.x>=d.width || i.y>=d.rows) return;
  for (uint c=0;c<4;c++) stream[ulong(i.y)*d.stream_stride+ulong(c)*d.width+i.x]=x[ulong(i.y)*d.x_stride+i.x];
  if (i.x==0) for (uint c=0;c<4;c++) pre[ulong(i.y)*d.pre_stride+c]=c==0?1.0f:0.0f;
}
