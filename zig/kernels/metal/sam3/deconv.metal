#include <metal_stdlib>
using namespace metal;

struct SamConv { uint h,w,c,n,k,oh,ow,stride,pad,first,rows; };

[[max_total_threads_per_threadgroup(256)]]
kernel void im_sam_deconv(const device float *x, const device float *w, const device float *b, device float *y,
                         constant SamConv &p, uint i [[thread_position_in_grid]]) {
    if (i>=p.h*p.w*p.n) return;
    const uint o=i%p.n, pixel=i/p.n;
    float4 sum=float4(b[o]);
    for(uint c=0;c<p.c;c++) {
        const float4 weight=float4(*(const device packed_float4*)(w+(c*p.n+o)*4));
        sum+=x[pixel*p.c+c]*weight;
    }
    const uint dest=((pixel/p.w*2)*p.ow+pixel%p.w*2)*p.n+o;
    y[dest]=sum.x;
    y[dest+p.n]=sum.y;
    y[dest+p.ow*p.n]=sum.z;
    y[dest+(p.ow+1)*p.n]=sum.w;
}
