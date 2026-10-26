//! The CUDA probability stores and complete merge body execute on CPU; matrix products use a scalar host instrument.
const std = @import("std");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 CUDA attention source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_cuda_attention_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const text = @embedFile("../kernels/cuda/flashnext_attn.cu");
    if (std.mem.indexOf(u8, text, "constexpr int HD = 256;") == null or std.mem.indexOf(u8, text, "constexpr int TK = 32;") == null) return error.CudaGeometry;
    const start = std.mem.indexOf(u8, text, "                const float p = s[jt][e]") orelse return error.CudaProbability;
    const last = std.mem.indexOfPos(u8, text, start, "ps[g + 8 * h]") orelse return error.CudaProbability;
    const end = (std.mem.indexOfScalarPos(u8, text, last, ';') orelse return error.CudaProbability) + 1;
    const store = text[start..end];
    if (std.mem.count(u8, store, "__float2bfloat16_rn(p)") != 1 or std.mem.indexOf(u8, store, "rows[h] = __fadd_rn(rows[h], p);") == null) return error.CudaProbability;
    var merge = try edit.function(text, "void __launch_bounds__(256) fn_attn_merge(");
    merge = try edit.once(a, merge, "void __launch_bounds__(256)", "void");
    merge = try edit.once(a, merge, "__global__ ", "");
    return std.mem.concat(a, u8, &.{
        \\#include "hc_native_support.h"
        \\namespace ca {
        \\constexpr int HD=256,JT=1;static uint3 blockIdx,threadIdx;
        \\inline float tf_exp(float x){return expf(x); }
        \\
        ,
        \\static bfloat probability(float score,float maximum,float& sum){
        \\ const int jt=0,e=0,h=0,g=0,warp=0,t=0;const float s[1][4]={{score}},next_m[2]={maximum};
        \\ float rows[2]={};bfloat ps[16][32];
        \\
        ,
        store,
        "\n sum=rows[0];return ps[0][0];}\n",
        merge,
        \\static void run(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){
        \\ if(count<1||count>32)std::abort();float scores[32],sum=0,maximum=-INFINITY; bfloat weights[32];
        \\ for(int i=0;i<count;++i){float dot=0;for(int d=0;d<256;++d)dot=fmaf(float(reinterpret_cast<const bfloat*>(q)[d]),float(reinterpret_cast<const bfloat*>(k)[ids[i]*256+d]),dot);scores[i]=dot*0.0625f;maximum=fmaxf(maximum,scores[i]);}
        \\ for(int i=0;i<count;++i){float term;weights[i]=probability(scores[i],maximum,term);sum+=term;}
        \\ if(control)for(int i=0;i<count;++i)weights[i]=bfloat(expf(scores[i]-maximum)/sum);
        \\ float po[256];for(int d=0;d<256;++d){po[d]=0;for(int i=0;i<count;++i)po[d]=fmaf(float(weights[i]),float(reinterpret_cast<const bfloat*>(v)[ids[i]*256+d]),po[d]);}
        \\ const float pm=maximum,pl=control?1.0f:sum;const int pos=count-1;
        \\ blockIdx={0,0,0};for(int d=0;d<256;++d){threadIdx={uint(d),0,0};fn_attn_merge(po,&pm,&pl,&pos,reinterpret_cast<bfloat*>(output),nullptr,nullptr,1,1,1,512,1,0);}
        \\}
        \\}
        \\extern "C" void tf_q4_cuda_node_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){ca::run(q,k,v,ids,count,output,control);}
        \\
        ,
    });
}
