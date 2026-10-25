//! CPU instruments compile generated PLE source and execute its native scalar and collective operations only.
const std = @import("std");
const cfg = @import("../src/families/qwen4/config.zig");
const sources = @import("../src/families/qwen4/metal_ple.zig");
const controls = @import("../src/families/qwen4/ple_control_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 PLE source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_ple_native_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#include \"hc_native_support.h\"\n");
    for ([_]usize{ 256, 5120 }, [_]usize{ 2, 4 }, 0..) |h, s, geometry| {
        var problem: cfg.Problem = .{};
        var c = try cfg.parse(a, @embedFile("../src/families/qwen4/fixtures/tiny-dense.json"), &problem);
        defer c.deinit();
        c.hidden = h;
        c.streams = s;
        c.ple_conv = 4;
        c.ngram = 3;
        const production = try sources.source(a, &c);
        for (std.enums.values(controls.Control), 0..) |control, index| {
            var text = try controls.metal(a, production, control);
            const edits = [_][2][]const u8{
                .{ "#include <metal_stdlib>", "" }, .{ "using namespace metal;", "" }, .{ "const constant ", "const " },                 .{ "constant ", "const " },                          .{ "constexpr ", "constexpr " },
                .{ "const device ", "const " },     .{ "device ", "" },                .{ "threadgroup float red", "static float red" }, .{ "void qwen4_ple_gate(", "Task qwen4_ple_gate(" }, .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
            };
            for (edits) |e| text = try std.mem.replaceOwned(u8, a, text, e[0], e[1]);
            for ([_][]const u8{ "sk", "sq", "dot", "sc" }) |v| text = try std.mem.replaceOwned(u8, a, text, try std.fmt.allocPrint(a, "simd_sum({s})", .{v}), try std.fmt.allocPrint(a, "(co_await simd_sum({s}))", .{v}));
            while (std.mem.indexOf(u8, text, "[[")) |at| {
                const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.PleAttribute) + 2;
                text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
            }
            try out.writer.print("namespace p{d}c{d} {{\n{s}\n", .{ geometry, index, text });
            try out.writer.print(
                \\static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {{
                \\ const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
                \\ const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
                \\ const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
                \\ auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
                \\ const int w={d};const float eps=1e-6f;
                \\ for(int r=0;r<rows;++r)for(uint s=0;s<{d};++s) {{Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,{s}gated,normed,t/32,t%32,{{t,0,0}},{{s,uint(r),0}});run_tasks(tasks);}}
                \\ auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
                \\ for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{s}{{d,uint(r),0}});
                \\ for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
                \\}}
                \\}}
                \\
            , .{ h * s, s, if (control == .repeat) "repeat," else "", if (control == .repeat) "repeat," else "" });
        }
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_ple_native_cpu(int geometry,int control,const unsigned char* const* input,unsigned char* gated,unsigned char* normed,unsigned char* hidden,unsigned char* tail,int rows,unsigned repeat){switch(geometry*5+control){\n");
    for (0..2) |geometry| for (0..5) |control| try out.writer.print("case {d}:p{d}c{d}::run(input,gated,normed,hidden,tail,rows,repeat);return;\n", .{ geometry * 5 + control, geometry, control });
    try out.writer.writeAll("default:std::abort();}}\n");
    return out.toOwnedSlice();
}
