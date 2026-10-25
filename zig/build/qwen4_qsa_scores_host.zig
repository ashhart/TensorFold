//! The native QSA score shader executes under an owned CPU SIMD instrument.
const std = @import("std");
const gen = @import("../src/families/qwen4/metal_attention_source.zig");
const ctl = @import("../src/families/qwen4/qsa_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 score source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_qsa_scores_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const production = try gen.source(a, .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 });
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#include \"hc_native_support.h\"\n");
    for (std.enums.values(ctl.Control), 0..) |control, i| {
        var text = try edit.function(try ctl.scores(a, production, control), "void qwen4_idx_scores(");
        for ([_][2][]const u8{
            .{ "const device ", "const " },                          .{ "device ", "" },                               .{ "const constant ", "const " }, .{ "constant ", "const " },
            .{ "void qwen4_idx_scores(", "Task qwen4_idx_scores(" }, .{ "simd_sum(dot)", "(co_await simd_sum(dot))" },
        }) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.QsaAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        try out.writer.print("namespace qs{d} {{\n{s}\nstatic void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {{\n  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){{Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,{s}t/32,t%32,{{x,y,0}});run_tasks(tasks);}}\n}}\n}}\n", .{ i, text, if (control == .repeat) "repeat," else "" });
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_qsa_scores_cpu(int control,const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat){switch(control){\n");
    for (0..5) |i| try out.writer.print("case {d}:qs{d}::run(query,pooled,complete,out,rows,blocks,repeat);return;\n", .{ i, i });
    try out.writer.writeAll("default:std::abort();}}\n");
    return out.toOwnedSlice();
}
