//! A CPU-only coroutine instrument compiles the actual generated recurrence and its controlled source variants.
const std = @import("std");
const gen = @import("../src/families/flashnext/hybrid_sources.zig");
const ctl = @import("../src/families/qwen4/gdn_control_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 GDN source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_gdn_native_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#define TF_METAL_STRICT_BFLOAT 1\n#include \"hc_native_support.h\"\n");
    try out.writer.writeAll(@import("../src/families/qwen4/metal_control.zig").declaration);
    for ([_]bool{ false, true }, 0..) |sigmoid, gate| for (std.enums.values(ctl.Control), 0..) |control, index| {
        const production = try gen.Gdn.metal(a, @embedFile("../kernels/metal/decode/fn_gdn.metal"), @embedFile("../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal"), 1, 2, sigmoid);
        var text = try ctl.metal(a, production, control);
        const replacements = [_][2][]const u8{
            .{ "#include <metal_stdlib>", "" },                                                   .{ "using namespace metal;", "" },
            .{ "constant constexpr ", "constexpr " },                                             .{ "const constant ", "const " },
            .{ "constant ", "const " },                                                           .{ "const device ", "const " },
            .{ "device ", "" },                                                                   .{ "threadgroup bfloat xq", "static bfloat xq" },
            .{ "threadgroup float inv", "static float inv" },                                     .{ "threadgroup bfloat*", "bfloat*" },
            .{ "void qwen4_gdn_chain(", "Task qwen4_gdn_chain(" },                                .{ "simd_sum(ss)", "(co_await simd_sum(ss))" },
            .{ "simd_sum(kv[j])", "(co_await simd_sum(kv[j]))" },                                 .{ "simd_sum(out[j])", "(co_await simd_sum(out[j]))" },
            .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
        };
        for (replacements) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.GdnAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        try out.writer.print("namespace g{d}c{d} {{\n{s}\n", .{ gate, index, text });
        try out.writer.print(
            \\static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {{const Qwen4Control control_params{{repeat}};
            \\  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
            \\  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
            \\  const auto* cw=reinterpret_cast<const float*>(input[2]);
            \\  const auto* s=reinterpret_cast<const float*>(input[3]);
            \\  const auto* alog=reinterpret_cast<const float*>(input[4]);
            \\  const auto* dt=reinterpret_cast<const float*>(input[5]);
            \\  const auto* nw=reinterpret_cast<const float*>(input[6]);
            \\  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
            \\  const ulong stride=0;
            \\  auto* tail=reinterpret_cast<bfloat*>(tail_out);
            \\  for(uint head=0;head<2;++head) {{
            \\    Task tasks[1024];
            \\    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,{s}t,t%32,t/32,head);
            \\    run_tasks(tasks,1024);
            \\  }}
            \\}}
            \\}}
            \\
        , .{if (control == .repeat) "control_params," else ""});
    };
    try out.writer.writeAll("extern \"C\" void tf_q4_gdn_native_cpu(int gate,int control,const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,unsigned repeat) {\n  switch(gate*5+control) {\n");
    for (0..2) |gate| for (0..5) |control| try out.writer.print("  case {d}:g{d}c{d}::run(input,out,state,tail_out,rows,flags,repeat);return;\n", .{ gate * 5 + control, gate, control });
    try out.writer.writeAll("  default:std::abort();}\n}\n");
    return out.toOwnedSlice();
}
