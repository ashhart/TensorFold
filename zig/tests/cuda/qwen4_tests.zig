//! Explicit synthetic device gate over the serving CUDA engine and its embedded native images.
const std = @import("std");
const C = @import("cuda");
const F = @import("qwen4").family;
const check = @import("check.zig");
const G = F.cuda_gpu.Gpu(C);
const B = G.Storage;
const D = F.cuda_draw.Drawer(C, @import("lanes"), B);
const Loader = struct {
    pub fn load(_: @This(), a: std.mem.Allocator, b: *B, c: *const F.Config, p: *F.Problem) !G {
        return G.load(a, b, c, p, .{ .projection = C.kernels.qwen4_projection, .hc_scalar = C.kernels.qwen4_hc, .hc = C.kernels.flashnext_hc, .gdn = C.kernels.qwen4_gdn, .attention = C.kernels.flashnext_attn, .batch = C.kernels.flashnext_batch, .ple = C.kernels.qwen4_ple }, .hf);
    }
};
fn read(_: *B, buffer: B.Buffer, out: []u8) !void {
    try buffer.download(0, out);
}
pub fn run(gpu: check.Gpu) !void {
    if (!C.kernels.available) return error.BuiltWithoutKernels;
    return execute(gpu);
}
pub fn execute(gpu: check.Gpu) !void {
    std.debug.print("QWEN4_DEVICE cuda sm_{d}\n", .{try gpu.ctx.capability()});
    var stream = try C.Stream.init(gpu.d, true);
    defer stream.deinit();
    var b: B = .{ .driver = gpu.d, .context = gpu.ctx, .stream = stream };
    var d = try D.load(&b, 32, 32, 2, C.kernels.sample, C.kernels.torch_argmax);
    var closed = false;
    defer if (!closed) d.close() catch {};
    try F.device_gate.Gate(G, D, @import("lanes"), read).runAll(gpu.gpa, gpu.io, &b, Loader{}, &d);
    try d.close();
    closed = true;
}
