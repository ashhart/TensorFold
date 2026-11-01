//! The actual family adapter's packed and routing adjoints keep serving rounding enabled.
const std = @import("std");
const cuda = @import("cuda");
const q4 = cuda.train_q4;
const to = @import("cuda_train_ops.zig");
const Rec = struct {
    var modes: [2]c_int = @splat(-1);
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const at: usize = switch (@intFromPtr(f.?)) {
            105 => 0,
            108 => 1,
            else => return 0,
        };
        modes[at] = @as(*const c_int, @ptrCast(@alignCast(args.?[if (at == 0) 7 else 11].?))).*;
        return 0;
    }
};
const Layer = struct { mlp: struct { moe: struct { router: u64, experts: cuda.experts.Layer } } };
const Owner = struct { g: @import("cuda_geometry.zig").Geometry, stream: cuda.Stream, layers: [1]Layer };
const Trainer = struct {
    e: *Owner,
    mk: ?q4.Kernels,
    grad: u64 = 100,
    dy: u64 = 200,
    moe_scratch: q4.MoeScratch = .{ .logits = 1, .order = 2, .inv = 3, .offsets = 4, .tiles = 5, .gu = 6, .dact = 7, .dgu = 8, .dxp = 9, .a = 10, .dlog = 11 },
    pub fn ops(t: *@This()) to.Ops {
        return .{ .k = undefined, .s = t.e.stream };
    }
};
test "MoE learning: actual packed trainer enables serving rounds in SwiGLU and shared router" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Rec.launch;
    var owner: Owner = .{
        .g = .{ .hidden = 2048, .intermediate = 512, .layers = 40, .experts = 256, .top_k = 8, .vocab = 248320, .k_heads = 16, .v_heads = 32, .q_heads = 16, .kv_heads = 2, .eps = 1e-6, .tied = false, .context = 4096, .linear = @splat(false) },
        .stream = .{ .d = &d, .handle = null },
        .layers = .{.{ .mlp = .{ .moe = .{ .router = 50, .experts = .{ .format = .affine4, .up = 60, .down = 70, .width = 512, .dims = 2048, .experts = 257, .swiglu = true } } } }},
    };
    var t: Trainer = .{ .e = &owner, .mk = @as(q4.Kernels, undefined) };
    for (&t.mk.?.fq, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(100 + i) };
    for (&t.mk.?.ft) |*f| f.* = .{ .d = &d, .handle = @ptrFromInt(16) };
    Rec.modes = @splat(-1);
    try @import("cuda_moe_train.zig").backward(&t, 0, .{ .y = 1000, .ids = 2000, .wts = 3000 }, null, 3);
    try std.testing.expectEqual([2]c_int{ 1, 1 }, Rec.modes);
}
