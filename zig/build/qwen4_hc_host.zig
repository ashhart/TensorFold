//! Emit actual specialized HC shader bodies into a CPU-only coroutine/SIMD instrument at configuration time.
const std = @import("std");
const hc = @import("../src/families/qwen4/metal_hc.zig");
const ctl = @import("../src/families/qwen4/hc_control_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 HC source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_hc_native_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("#define TF_METAL_STRICT_BFLOAT 1\n#include \"hc_native_support.h\"\n");
    try out.writer.writeAll(@import("../src/families/qwen4/metal_control.zig").declaration);
    const original = try hc.source(a, .{ .hidden = 5120, .streams = 4 });
    for (std.enums.values(ctl.Control), 0..) |control, i| {
        var text = try ctl.source(a, original, control);
        text = try std.mem.replaceOwned(u8, a, text, "#include <metal_stdlib>", "");
        text = try std.mem.replaceOwned(u8, a, text, "using namespace metal;", "");
        const replacements = [_][2][]const u8{
            .{ "[[max_total_threads_per_threadgroup(256)]] [[kernel]]", "" },
            .{ "const device ", "const " },
            .{ "device ", "" },
            .{ "const constant ", "const " },
            .{ "constant ", "const " },
            .{ "threadgroup float", "static float" },
            .{ "void qwen4_hc_sums(", "Task qwen4_hc_sums(" },
            .{ "simd_sum(hv * hv)", "(co_await simd_sum(hv * hv))" },
            .{ "threadgroup_barrier(mem_flags::mem_threadgroup);", "co_await group_barrier();" },
        };
        for (replacements) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        while (std.mem.indexOf(u8, text, "[[")) |at| {
            const end = (std.mem.indexOfPos(u8, text, at, "]]") orelse return error.HcAttribute) + 2;
            text = try std.mem.concat(a, u8, &.{ text[0..at], text[end..] });
        }
        const repeat = if (control == .repeat) "control_params," else "";
        try out.writer.print("namespace c{d} {{\n{s}\n", .{ i, text });
        try out.writer.print(
            \\static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {{const Qwen4Control control_params{{repeat}};
            \\  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
            \\  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {{
            \\    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
            \\    for(uint t=0;t<256;++t) {{cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),{s}t/32,t%32,uint3{{t,0,0}},uint3{{uint(j),uint(r),0}});}}
            \\    run_tasks(tasks);
            \\  }}
            \\  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {{
            \\    const float eps=1e-6f;
            \\    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),{s}uint2{{e,uint(r)}});
            \\  }}
            \\  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
            \\    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),{s}uint2{{d,uint(r)}});
            \\  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
            \\    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),{s}i);
            \\}}
            \\}}
            \\
        , .{ repeat, repeat, repeat, repeat });
    }
    try out.writer.writeAll("extern \"C\" void tf_q4_hc_native_cpu(int op,int control,const unsigned char* const* input,void* output,int rows,int width,unsigned repeat) {\n  switch(control) {\n");
    for (0..5) |i| try out.writer.print("  case {d}:c{d}::run(op,input,output,rows,width,repeat);return;\n", .{ i, i });
    try out.writer.writeAll("  default:std::abort();}\n}\n");
    return out.toOwnedSlice();
}
