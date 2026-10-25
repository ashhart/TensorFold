//! Execute the actual selection shader with CPU integer atomics, exact prefixes and block barriers.
const std = @import("std");
const gen = @import("../src/families/qwen4/metal_attention_source.zig");
const ctl = @import("../src/families/qwen4/qsa_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 selection source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_qsa_select_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const production = try gen.source(a, .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 });
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#include \"hc_native_support.h\"\n");
    for (std.enums.values(ctl.Control)[0..4], 0..) |control, i| {
        const all = try ctl.selection(a, production, control);
        const cut = try edit.function(all, "inline void cut_bin(");
        const brace = (std.mem.indexOfScalar(u8, cut, '{') orelse return error.QsaCut) + 1;
        var body = cut[brace .. cut.len - 1];
        body = try edit.once(a, body, "cut = prefix", "cut_t = prefix");
        body = try edit.once(a, body, "rest = need", "need_t = need");
        var text = try edit.function(all, "void qwen4_idx_select(");
        const call = "cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);";
        if (std.mem.count(u8, text, call) != 3) return error.QsaCut;
        text = try std.mem.replaceOwned(u8, a, text, call, try std.mem.concat(a, u8, &.{ "{\n", body, "\n}" }));
        text = try std.mem.concat(a, u8, &.{ "constexpr int RATIO=8;\n", try edit.function(all, "inline uint tf_key("), "\n", text });
        for ([_][2][]const u8{
            .{ "const device ", "const " },                       .{ "device ", "" },                                                                   .{ "const constant ", "const " },     .{ "constant ", "const " },
            .{ "threadgroup atomic_uint", "static atomic_uint" }, .{ "threadgroup uint", "static uint" },                                               .{ "threadgroup int", "static int" }, .{ "void qwen4_idx_select(", "Task qwen4_idx_select(" },
            .{ "return;", "co_return;" },                         .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
        }) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        for ([_][]const u8{ "mine", "cnt", "a", "n_above", "n_equal", "e" }) |v| text = try std.mem.replaceOwned(u8, a, text, try std.fmt.allocPrint(a, "simd_prefix_exclusive_sum({s})", .{v}), try std.fmt.allocPrint(a, "(co_await prefix_sum({s}))", .{v}));
        if (std.mem.indexOf(u8, text, "simd_prefix_exclusive_sum(") != null) return error.QsaPrefix;
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.QsaAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        try out.writer.print("namespace sl{d} {{\n{s}\nstatic void run(const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){{const int shape[2]={{rows,stride}};for(int r=0;r<rows;++r){{Task tasks[1024];for(uint t=0;t<1024;++t)tasks[t]=qwen4_idx_select(scores,shape,complete,ends,ids,{s}t/32,t%32,{{t,0,0}},{{uint(r),0,0}});run_tasks(tasks,1024);}}}}\n}}\n", .{ i, text, if (control == .repeat) "repeat," else "" });
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_qsa_select_cpu(int control,const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){switch(control){\n");
    for (0..4) |i| try out.writer.print("case {d}:sl{d}::run(scores,complete,ends,ids,rows,stride,repeat);return;\n", .{ i, i });
    try out.writer.writeAll("default:std::abort();}}\n");
    return out.toOwnedSlice();
}
