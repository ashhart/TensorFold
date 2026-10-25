//! Actual CUDA lexical source uses owned CPU collectives and the maintained IEEE scalar helper body.
const std = @import("std");
const gen = @import("../src/families/flashnext/hybrid_ple_source.zig");
const ctl = @import("../src/families/qwen4/ple_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 CUDA lexical boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_ple_cuda_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const production = try gen.cuda(a);
    const reduce = try edit.function(production, "float block_sum(");
    if (!std.mem.eql(u8, std.mem.trim(u8, reduce, "\n "), std.mem.trim(u8, @embedFile("../tests/qwen4/ple_block_sum.cuh"), "\n "))) return error.PleBlockReduction;
    var text = try ctl.cuda(a, production);
    text = try edit.once(a, text, reduce, "");
    for ([_][]const u8{ "kss", "qss", "dot", "gss" }) |v| text = try std.mem.replaceOwned(u8, a, text, try std.fmt.allocPrint(a, "block_sum({s}, scratch)", .{v}), try std.fmt.allocPrint(a, "(co_await cpu_block_sum({s}))", .{v}));
    var math = @embedFile("../kernels/cuda/glue_math.cuh").*;
    var helpers: []const u8 = &math;
    helpers = try edit.once(a, helpers, try edit.function(helpers, "float warp_sum("), "");
    const replacements = [_][2][]const u8{
        .{ "#pragma once", "" },                       .{ "#include <cuda_bf16.h>", "" },                                                                .{ "#include <cuda_fp16.h>", "" },                                                                .{ "#include \"glue_math.cuh\"", "" },
        .{ "__device__ __forceinline__ ", "inline " }, .{ "extern \"C\" __global__ void __launch_bounds__(512) qwen4_ple_gate", "Task qwen4_ple_gate" }, .{ "extern \"C\" __global__ void __launch_bounds__(512) qwen4_ple_conv", "void qwen4_ple_conv" }, .{ "__shared__ float", "static float" },
        .{ "blockIdx.", "cpu_block." },                .{ "threadIdx.x", "cpu_lane" },                                                                   .{ "blockDim.x", "512" },
    };
    for (replacements) |r| {
        text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        helpers = try std.mem.replaceOwned(u8, a, helpers, r[0], r[1]);
    }
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("#include \"hc_native_support.h\"\nnamespace pc {{\nstatic uint3 cpu_block;\n{s}\n{s}\n", .{ helpers, text });
    const dense_source = @embedFile("../kernels/cuda/qwen4_hc.cu");
    var dense: []const u8 = dense_source[(std.mem.indexOf(u8, dense_source, "template<int CONTROL>\n__device__ __forceinline__ void qwen4_dense_run(") orelse return error.DenseSource)..];
    dense = try std.mem.concat(a, u8, &.{ try edit.function(dense_source, "float hc_silu("), "\n", dense });
    for ([_][2][]const u8{
        .{ "__device__ __forceinline__ ", "inline " }, .{ "extern \"C\" __global__ void __launch_bounds__(256) ", "void " },
        .{ "tf_wo_bf(", "dense_bf(" },                 .{ "tf_wo_round(", "dense_round(" },
        .{ "blockIdx.", "cpu_block." },                .{ "threadIdx.x", "cpu_lane" },
        .{ "blockDim.x", "256" },
    }) |r| dense = try std.mem.replaceOwned(u8, a, dense, r[0], r[1]);
    try out.writer.writeAll("inline float dense_bf(unsigned short x){return __int_as_float(unsigned(x)<<16);}\ninline unsigned short dense_round(float x){return bfloat(x).bits;}\n");
    try out.writer.print("{s}\n", .{dense});
    try out.writer.writeAll("static void dense_run(int control,const unsigned short* gate,const unsigned short* up,unsigned short* out,int rows,int width,unsigned repeat){for(int r=0;r<rows;++r)for(int block=0;block<(width+255)/256;++block){cpu_block={uint(r),uint(block),0};for(cpu_lane=0;cpu_lane<256;++cpu_lane){switch(control){\n");
    for ([_][]const u8{ "qwen4_dense_act", "qwen4_dense_act_bad_nan", "qwen4_dense_act_bad_repeat", "qwen4_dense_act_bad_tail" }, 0..) |name, i| try out.writer.print("case {d}:{s}(gate,up,out,width,repeat);break;\n", .{ i, name });
    try out.writer.writeAll("default:std::abort();}}}}\n");
    try out.writer.writeAll(
        \\static void run(int control,const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,int D,int S,unsigned repeat) {
        \\ const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
        \\ const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
        \\ const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
        \\ auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
        \\ const int w=D*S;const float eps=1e-6f;auto* ss=static_cast<float*>(std::malloc(rows*S*4));if(!ss)std::abort();
        \\ for(int r=0;r<rows;++r){cpu_block={uint(r),0,0};Task tasks[512];for(int t=0;t<512;++t){switch(control){
        \\
    );
    for (0..5) |i| {
        const name = if (i == 0) "qwen4_ple_gate" else ctl.names[(i - 1) * 2];
        try out.writer.print("case {d}:tasks[t]={s}(kv,kv+w,h,ks,qs,gated,ss,eps,D,S{s});break;\n", .{ i, name, if (i == 0) "" else ",int(repeat)" });
    }
    try out.writer.writeAll("default:std::abort();}}run_tasks(tasks,512);}\n std::memcpy(hidden,h,rows*w*2);\n for(int r=0;r<rows;++r)for(int block=0;block<(w+511)/512;++block){cpu_block={uint(r),uint(block),0};for(cpu_lane=0;cpu_lane<512;++cpu_lane){switch(control){\n");
    for (0..5) |i| {
        const name = if (i == 0) "qwen4_ple_conv" else ctl.names[(i - 1) * 2 + 1];
        try out.writer.print("case {d}:{s}(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3{s});break;\n", .{ i, name, if (i == 0) "" else ",int(repeat)" });
    }
    try out.writer.writeAll("default:std::abort();}}}\n auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);std::free(ss);\n}\n}\nextern \"C\" void tf_q4_cuda_ple_native_cpu(int control,const unsigned char* const* input,unsigned char* gated,unsigned char* normed,unsigned char* hidden,unsigned char* tail,int rows,int hidden_size,int streams,unsigned repeat){pc::run(control,input,gated,normed,hidden,tail,rows,hidden_size,streams,repeat);}\n");
    try out.writer.writeAll("extern \"C\" void tf_q4_cuda_dense_native_cpu(int control,const unsigned short* gate,const unsigned short* up,unsigned short* out,int rows,int width,unsigned repeat){pc::dense_run(control,gate,up,out,rows,width,repeat);}\n");
    return out.toOwnedSlice();
}
