// Reference: DeepSeek-V4.1-Flash inference/model.py Gate, revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
#include <metal_stdlib>
using namespace metal;
inline float lp(float x) { float u=1.0f+x; return u==1.0f?x:x*(log(u)/(u-1.0f)); }
inline float score(float x) { return sqrt(x>20.0f?x:lp(exp(x))); }
struct Shape { uint rows,logit_stride,output_stride; float temperature; uint bias_dtype; uint fused; uint logit_dtype; uint experts; uint topk; uint moe_up; };
static_assert(sizeof(Shape) == 40, "Shape must mirror route.zig's Args");
kernel void dsv41_route(const device float* logits [[buffer(0)]], const device uchar* bias [[buffer(1)]],
                        device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                        constant Shape& d [[buffer(4)]], device uint* fids [[buffer(5)]],
                        device uint* fsrc [[buffer(6)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scores[384], select[384], best[4], picked[6];
  threadgroup uint best_id[4], winner;
  for (uint e=t;e<d.experts;e+=128) {
    float b;
    if (d.bias_dtype==1) b=((const device float*)bias)[e];
    else if (d.bias_dtype==2) b=float(((const device half*)bias)[e]);
    else b=as_type<float>(uint(((const device ushort*)bias)[e])<<16);
    const ulong at=ulong(row)*d.logit_stride+e;
    const float lg=d.logit_dtype==1?as_type<float>(uint(((const device ushort*)logits)[at])<<16):logits[at];
    scores[e]=score(lg/d.temperature);
    select[e]=scores[e]+b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Every expert participates in every rank selection; ties use the lower expert id deterministically.
  for (uint slot=0;slot<d.topk;slot++) {
    float v=-INFINITY;
    uint id=0xffffffffu;
    // THE SELECTION SCANS THE BLOCK'S OWN EXPERT COUNT, so the drafter's 128 never ranks stale slots past its table.
    for (uint e=t;e<d.experts;e+=128) if (select[e]>v || (select[e]==v && e<id)) { v=select[e]; id=e; }
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
        // The fused gate+up dispatch: rows 0..6T-1 are every token's gate picks and 6T..12T-1 the same tokens' up.
        if (d.fused!=0) {
          fids[ulong(row)*d.output_stride+slot]=q;
          fids[ulong(d.rows+row)*d.output_stride+slot]=q+d.fused;
          fsrc[ulong(row)*d.output_stride+slot]=row;
          fsrc[ulong(d.rows+row)*d.output_stride+slot]=row;
          // The shared expert's two blocks after the routed twelve picks a token (gate rows then up rows).
          if (d.moe_up!=0 && row==0 && slot==0) {
            const uint tail = 2u*d.rows*d.output_stride;
            for (uint t2=0;t2<d.rows;++t2) {
              fids[tail+t2]=2u*d.fused;
              fsrc[tail+t2]=t2;
              fids[tail+d.rows+t2]=2u*d.fused+1u;
              fsrc[tail+d.rows+t2]=t2;
            }
          }
        }
        picked[slot]=q<d.experts?scores[q]:NAN;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t==0 && winner<d.experts) select[winner]=-INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t<d.topk) {
    // Source CUDA reduces contiguous six-score rows with four lanes, then shuffles by 2 and 1.
    float left, right;
    if (d.topk==6) { left=(picked[0]+picked[4])+picked[2]; right=(picked[1]+picked[5])+picked[3]; }
    else { left=picked[0]; right=(picked[1]+picked[2]); }
    float s=left+right;
    weights[ulong(row)*d.output_stride+t]=(picked[t]/(s+1e-20f))*1.5f;
  }
}
