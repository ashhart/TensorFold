//! Execute the generated split attention shader bodies under CPU SIMD and barrier instruments.
const std = @import("std");
const gen = @import("../src/families/qwen4/metal_attention_source.zig");
const ctl = @import("../src/families/qwen4/qsa_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 attention source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_qsa_attn_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const production = try gen.source(a, .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 });
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#include \"hc_native_support.h\"\n");
    for (std.enums.values(ctl.Control), 0..) |control, i| {
        const all = try ctl.attention(a, production, control);
        var text = try std.mem.concat(a, u8, &.{ try edit.function(all, "void qwen4_attn_parts("), "\n", try edit.function(all, "void qwen4_attn_merge(") });
        for ([_][2][]const u8{
            .{ "const device ", "const " },           .{ "device ", "" },                                      .{ "const constant ", "const " },               .{ "constant ", "const " },
            .{ "threadgroup float", "static float" }, .{ "void qwen4_attn_parts(", "Task qwen4_attn_parts(" }, .{ "simd_sum(sc)", "(co_await simd_sum(sc))" }, .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
        }) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.QsaAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        try out.writer.print("namespace qa{d} {{\n{s}\n", .{ i, text });
        try out.writer.print(
            \\static void run(const unsigned char* const* in,float* po,float* pm,unsigned short* output,int rows,int keys,unsigned repeat) {{
            \\const auto* q=reinterpret_cast<const bfloat*>(in[0]);const auto* k=reinterpret_cast<const bfloat*>(in[1]);const auto* v=reinterpret_cast<const bfloat*>(in[2]);
            \\const auto* ids=reinterpret_cast<const int32_t*>(in[3]);const auto* nk=reinterpret_cast<const int32_t*>(in[4]);const auto* spr=reinterpret_cast<const int32_t*>(in[5]);
            \\const int ks[3]={{1,4,keys}},is[2]={{rows,keys}};const float scale=0.0625f;
            \\for(uint r=0;r<uint(rows);++r)for(uint h=0;h<24;++h)for(uint p=0;p<16;++p){{Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_parts(q,k,ks,v,ids,is,nk,spr,&scale,po,pm,t/32,t%32,{{t,0,0}},{{h,r,p}});run_tasks(tasks);}}
            \\for(uint r=0;r<uint(rows);++r)for(uint h=0;h<24;++h)for(uint d=0;d<256;++d)qwen4_attn_merge(po,pm,nullptr,reinterpret_cast<bfloat*>(output),{s}{{d,0,0}},{{0,h,r}});
            \\}}
            \\}}
            \\
        , .{if (control == .repeat) "repeat," else ""});
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_qsa_attn_cpu(int control,const unsigned char* const* in,float* po,float* pm,unsigned short* out,int rows,int keys,unsigned repeat){switch(control){\n");
    for (0..5) |i| try out.writer.print("case {d}:qa{d}::run(in,po,pm,out,rows,keys,repeat);return;\n", .{ i, i });
    try out.writer.writeAll("default:std::abort();}}\n");
    return out.toOwnedSlice();
}
