// Reference: DeepSeek-V4.1-Flash inference/model.py Gate, revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
#include <metal_stdlib>
using namespace metal;
inline float lp(float x) { float u=1.0f+x; return u==1.0f?x:x*(log(u)/(u-1.0f)); }
inline float score(float x) { return sqrt(x>20.0f?x:lp(exp(x))); }
struct Shape { uint rows,logit_stride,output_stride; float temperature; uint bias_dtype,merged_stride,merged_extra; };
kernel void dsv41_route(const device float* logits [[buffer(0)]], const device uchar* bias [[buffer(1)]],
                        device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                        constant Shape& d [[buffer(4)]], device uint* merged [[buffer(5)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scores[384], select[384], best[4], picked[6];
  threadgroup uint best_id[4], winner;
  for (uint e=t;e<384;e+=128) {
    float b;
    if (d.bias_dtype==1) b=((const device float*)bias)[e];
    else if (d.bias_dtype==2) b=float(((const device half*)bias)[e]);
    else b=as_type<float>(uint(((const device ushort*)bias)[e])<<16);
    scores[e]=score(logits[ulong(row)*d.logit_stride+e]/d.temperature);
    select[e]=scores[e]+b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Every expert participates in every rank selection; ties use the lower expert id deterministically.
  for (uint slot=0;slot<6;slot++) {
    float v=-INFINITY;
    uint id=0xffffffffu;
    for (uint e=t;e<384;e+=128) if (select[e]>v || (select[e]==v && e<id)) { v=select[e]; id=e; }
    float mx=simd_max(v);
    uint ix=simd_min(v==mx?id:0xffffffffu);
    if (lane==0) { best[sg]=mx; best_id[sg]=ix; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) {
      float b=lane<4?best[lane]:-INFINITY;
      float m=simd_max(b);
      uint q=simd_min(lane<4 && b==m?best_id[lane]:0xffffffffu);
      if (lane==0) {
        winner=q;
        ids[ulong(row)*d.output_stride+slot]=q;
        // The same six IDs again at merged_stride, for products that append a shared expert's row.
        if (d.merged_stride) merged[ulong(row)*d.merged_stride+slot]=q;
        picked[slot]=q<384?scores[q]:NAN;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t==0 && winner<384) select[winner]=-INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (d.merged_stride && t==0) merged[ulong(row)*d.merged_stride+6]=d.merged_extra;
  if (t<6) {
    // Source CUDA reduces contiguous six-score rows with four lanes, then shuffles by 2 and 1.
    float left=(picked[0]+picked[4])+picked[2];
    float right=(picked[1]+picked[5])+picked[3];
    float s=left+right;
    weights[ulong(row)*d.output_stride+t]=(picked[t]/(s+1e-20f))*1.5f;
  }
}
