//! The group-64 packed adjoints and Qwen gated pocket through the shared FP64 and finite-difference suite.
const cuda = @import("cuda");
const qwen = @import("qwen35");
const check = @import("check.zig");
const q4 = cuda.train_q4;
const to = qwen.train_ops;
const Kernels = struct { core: q4.Kernels, pocket: to.Kernels };
fn For(comptime omit_shared: bool) type {
    return struct {
        k: *const Kernels,
        s: cuda.Stream,
        pub const Pocket = struct { gate: u64, up: u64, down: u64, n: usize, act: u64 };
        pub fn moeBackPacked(o: @This(), e: q4.PackedExperts, router: u64, kept: q4.MoeKept, pk: Pocket, grad: u64, dx: u64, t: q4.MoeScratch, rows: usize, top_k: usize) !void {
            const ops: q4.Ops = .{ .k = &o.k.core, .s = o.s };
            try ops.expertsBackPacked(e, router, kept, grad, dx, t, rows, top_k);
            const pocket: to.Ops = .{ .k = &o.k.pocket, .s = o.s };
            try pocket.pocketBack(.{ .gate = pk.gate, .up = pk.up, .down = pk.down, .n = pk.n, .row_gate = kept.wts + top_k * 4, .gate_stride = top_k + 1, .a_shared = if (omit_shared) 0 else t.a + top_k * 4 }, kept.x, grad, dx, rows, e.dim);
            try ops.routeBack(router, kept, dx, t, rows, top_k, e.routed, e.dim);
        }
    };
}
pub fn run(gpu: check.Gpu) !void {
    return runImpl(gpu, false);
}
pub fn control(gpu: check.Gpu) !void {
    return runImpl(gpu, true);
}
fn runImpl(gpu: check.Gpu, comptime omit_shared: bool) !void {
    if (!cuda.kernels.available) return @import("moe_back_tests.zig").runQwen(gpu, void, {});
    var k: Kernels = .{ .core = try q4.Kernels.load(gpu.d), .pocket = undefined };
    defer k.core.deinit();
    k.pocket = try to.Kernels.load(gpu.d);
    defer k.pocket.deinit();
    try @import("moe_back_tests.zig").runQwen(gpu, For(omit_shared), &k);
}
