//! Compile the generated QSA frontend shaders on CPU, retaining their real reductions, barriers and rotary helper.
const std = @import("std");
const gen = @import("../src/families/qwen4/metal_attention_source.zig");
const ctl = @import("../src/families/qwen4/qsa_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 QSA frontend boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_qsa_norm_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const production = try gen.source(a, .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 });
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#define TF_METAL_STRICT_BFLOAT 1\n#include \"hc_native_support.h\"\n");
    try out.writer.writeAll(@import("../src/families/qwen4/metal_control.zig").declaration);
    for (std.enums.values(ctl.Control), 0..) |control, index| {
        const all = try ctl.frontend(a, production, control);
        var text = try std.mem.concat(a, u8, &.{
            "struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };\nconstexpr int RATIO=8;\n",
            try edit.function(all, "inline TF_RotaryScalar tf_rotary_scalar("),
            "\n",
            if (control == .nan or control == .repeat) try edit.function(all, "inline float qwen4_qsa_control(") else "",
            "\n",
            try edit.function(all, "void qwen4_attn_prep("),
            "\n",
            try edit.function(all, "void qwen4_idx_pool("),
            "\n",
        });
        const replacements = [_][2][]const u8{
            .{ "if (!live) return;", "if (!live) co_return;" },
            .{ "const constant ", "const " },
            .{ "constant ", "const " },
            .{ "const device ", "const " },
            .{ "device ", "" },
            .{ "threadgroup float", "static float" },
            .{ "void qwen4_attn_prep(", "Task qwen4_attn_prep(" },
            .{ "void qwen4_idx_pool(", "Task qwen4_idx_pool(" },
            .{ "simd_sum(x * x)", "(co_await simd_sum(x * x))" },
            .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
        };
        for (replacements) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.QsaAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        try out.writer.print("namespace qn{d} {{\n{s}\n", .{ index, text });
        try out.writer.print(
            \\static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {{const Qwen4Control control_params{{repeat}};
            \\ const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
            \\ if(pool) {{
            \\   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
            \\   for(int r=0;r<rows;++r){{Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),{s}t/32,t%32,{{t,0,0}},{{0,uint(r),0}});run_tasks(tasks,128);}}
            \\ }}else{{
            \\   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
            \\   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){{Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),{s}t/32,t%32,{{t,0,0}},{{0,h,uint(r)}});run_tasks(tasks,256);}}
            \\ }}
            \\}}
            \\}}
            \\
        , .{ if (control == .repeat) "control_params," else "", if (control == .repeat) "control_params," else "" });
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_qsa_norm_native_cpu(int pool,int control,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat){switch(control){\n");
    for (0..5) |i| try out.writer.print("case {d}:qn{d}::run(pool,input,qo,ko,io,rows,repeat);return;\n", .{ i, i });
    try out.writer.writeAll("default:std::abort();}}\n");
    return out.toOwnedSlice();
}
