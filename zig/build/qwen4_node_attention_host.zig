//! The tiny fixture executes the generated Metal attention bodies on CPU, supplementing the real device gate.
const std = @import("std");
const gen = @import("../src/families/qwen4/metal_attention_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 tiny attention source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_node_attention.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const all = try gen.source(a, .{ .heads = 1, .kv = 1, .index_heads = 1, .index_dim = 128, .rotary = 32, .ratio = 2, .top = 4 });
    var text = try std.mem.concat(a, u8, &.{ try edit.function(all, "void qwen4_attn_parts("), "\n", try edit.function(all, "void qwen4_attn_merge(") });
    for ([_][2][]const u8{
        .{ "const device ", "const " },           .{ "device ", "" },                                      .{ "const constant ", "const " },               .{ "constant ", "const " },
        .{ "threadgroup float", "static float" }, .{ "void qwen4_attn_parts(", "Task qwen4_attn_parts(" }, .{ "simd_sum(sc)", "(co_await simd_sum(sc))" }, .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
    }) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
    while (std.mem.indexOf(u8, text, "[[")) |at| {
        const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.QsaAttribute) + 2;
        text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
    }
    return std.mem.concat(a, u8, &.{
        "#define TF_METAL_STRICT_BFLOAT 1\n#include \"hc_native_support.h\"\n", text,
        \\extern "C" void tf_q4_node_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output) {
        \\  const int ks[3]={1,1,count},is[2]={1,count},sparse=1;const float scale=0.0625f;
        \\  float po[16*256],pm[16*2];
        \\  for(uint p=0;p<16;++p){Task tasks[256];for(uint t=0;t<256;++t)
        \\    tasks[t]=qwen4_attn_parts(reinterpret_cast<const bfloat*>(q),reinterpret_cast<const bfloat*>(k),ks,reinterpret_cast<const bfloat*>(v),ids,is,&count,&sparse,&scale,po,pm,t/32,t%32,{t,0,0},{0,0,p});run_tasks(tasks);}
        \\  for(uint d=0;d<256;++d)qwen4_attn_merge(po,pm,nullptr,reinterpret_cast<bfloat*>(output),{d,0,0},{0,0,0});
        \\}
        \\
        ,
    });
}
