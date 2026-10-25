//! Compile the actual serving CUDA recurrence on CPU with its maintained update body and exact shuffle tree.
const std = @import("std");
const gen = @import("../src/families/flashnext/hybrid_sources.zig");
const ctl = @import("../src/families/qwen4/gdn_control_source.zig");
const edit = @import("../src/families/flashnext/geometry_source.zig");
pub fn wire(b: *std.Build, module: *std.Build.Module) void {
    const text = generate(b.allocator) catch @panic("Qwen4 CUDA recurrence source boundaries changed");
    module.addIncludePath(b.path("zig/tests/qwen4"));
    module.addCSourceFile(.{ .file = b.addWriteFiles().add("qwen4_gdn_cuda_host.cpp", text), .flags = &.{ "-std=c++20", "-fno-exceptions", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing" } });
}
pub fn generate(a: std.mem.Allocator) ![]const u8 {
    const serving = try gen.Gdn.cuda(a, @embedFile("../kernels/cuda/flashnext_gdn.cu"));
    var text = try ctl.cuda(a, serving);
    const update = try edit.function(text, "void update(");
    const start = (std.mem.indexOfScalar(u8, update, '{') orelse return error.GdnUpdate) + 1;
    var body = update[start .. update.len - 1];
    body = try std.mem.replaceOwned(u8, a, body, "vrow[", "vs[");
    text = try edit.once(a, text, update, "");
    const reduce = try edit.function(text, "float warp_sum(");
    text = try edit.once(a, text, reduce, "");
    const call = "update(s, kk, vs, warp, g, beta);";
    if (std.mem.count(u8, text, call) != 5) return error.GdnUpdate;
    text = try std.mem.replaceOwned(u8, a, text, call, try std.mem.concat(a, u8, &.{ "{\n", body, "\n}" }));
    for ([_][]const u8{ "ss", "kv", "o" }) |v| text = try std.mem.replaceOwned(u8, a, text, try std.fmt.allocPrint(a, "warp_sum({s})", .{v}), try std.fmt.allocPrint(a, "(co_await simd_sum({s}))", .{v}));
    if (std.mem.indexOf(u8, text, "warp_sum(") != null) return error.GdnReduction;
    const replacements = [_][2][]const u8{
        .{ "#include <cuda_bf16.h>", "" },                                     .{ "#include <cuda_runtime.h>", "" },          .{ "#include <cuda_fp16.h>", "" },
        .{ "extern \"C\" __global__ void __launch_bounds__(1024) ", "Task " }, .{ "__device__ __forceinline__ ", "inline " }, .{ "__shared__ float", "static float" },
        .{ "blockIdx.x", "cpu_block" },                                        .{ "threadIdx.x", "cpu_lane" },                .{ "__syncthreads();", "co_await group_barrier();" },
        .{ "__syncwarp();", "co_await warp_barrier();" },
    };
    for (replacements) |r| text = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("#include \"hc_native_support.h\"\nnamespace cu {{\nstatic int cpu_block;\n{s}\n", .{text});
    try out.writer.writeAll("static void run(int control,int gate,const unsigned char* const* input,unsigned char* out,unsigned char* state,int rows,int flags,unsigned repeat) {\n  float xs[16*2*4]; const float eps=1e-6f; const unsigned long long stride=2*128*128;\n  const auto* p=reinterpret_cast<const bfloat*>(input[0]);const auto* cs=reinterpret_cast<const bfloat*>(input[1]);\n  const auto* cw=reinterpret_cast<const float*>(input[2]);const auto* s=reinterpret_cast<const float*>(input[3]);\n  const auto* alog=reinterpret_cast<const float*>(input[4]);const auto* dt=reinterpret_cast<const float*>(input[5]);const auto* nw=reinterpret_cast<const float*>(input[6]);\n  for(int head=0;head<2;++head){cpu_block=head;Task tasks[1024];\n    for(int t=0;t<1024;++t) {switch(control){\n");
    const names = [_][]const u8{ "qwen4_gdn_chain", ctl.cuda_names[0], ctl.cuda_names[1], ctl.cuda_names[2], ctl.cuda_names[3] };
    for (names, 0..) |name, i| try out.writer.print("    case {d}:tasks[t]={s}(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride{s});break;\n", .{ i, name, if (i == 0) "" else ",int(repeat)" });
    try out.writer.writeAll("    default:std::abort();}}run_tasks(tasks,1024);}\n}\n}\nextern \"C\" void tf_q4_cuda_gdn_native_cpu(int gate,int control,const unsigned char* const* input,unsigned char* out,unsigned char* state,int rows,int flags,unsigned repeat){cu::run(control,gate,input,out,state,rows,flags,repeat);}\n");
    return out.toOwnedSlice();
}
