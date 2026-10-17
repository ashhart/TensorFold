//! Public MoE backward interfaces reject unsupported array/tile geometry before any CUDA launch.
const std = @import("std");
const q = @import("train_q4.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig");
const t = std.testing;
var launches: usize = 0;
fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
    launches += 1;
    return 0;
}
const Rig = struct {
    d: Driver = undefined,
    k: q.Kernels = undefined,
    fn init(x: *Rig) void {
        x.d = .{ .lib = undefined, .api = undefined };
        x.d.api.cuLaunchKernel = launch;
        const f: Module.Function = .{ .d = &x.d, .handle = @ptrFromInt(16) };
        x.k.fq = @splat(f);
        x.k.ft = @splat(f);
        launches = 0;
    }
    fn ops(x: *Rig) q.Ops {
        return .{ .k = &x.k, .s = .{ .d = &x.d, .handle = null } };
    }
};
const kept: q.MoeKept = .{ .x = 1, .ids = 2, .wts = 3 };
const scratch: q.MoeScratch = .{ .logits = 10, .order = 20, .inv = 30, .offsets = 40, .tiles = 50, .gu = 60, .dact = 70, .dgu = 80, .dxp = 90, .a = 100, .dlog = 110 };
fn experts(routed: usize, width: usize, dim: usize) q.Experts {
    return .{ .gate = .{ 1, 2, 3 }, .up = .{ 4, 5, 6 }, .down = .{ 7, 8, 9 }, .shared_gate = .{ 11, 12, 13 }, .shared_up = .{ 14, 15, 16 }, .shared_down = .{ 17, 18, 19 }, .routed = routed, .width = width, .dim = dim };
}
fn rejected(top_k: usize, routed: usize) !void {
    var x: Rig = .{};
    x.init();
    const o = x.ops();
    const e = experts(routed, 64, 64);
    const p: q.PackedExperts = .{ .up = 1, .down = 2, .routed = routed, .width = 64, .dim = 64 };
    try t.expectError(error.UnsupportedShape, o.routeBack(5, kept, 6, scratch, 1, top_k, routed, 64));
    try t.expectError(error.UnsupportedShape, o.expertsBack(e, 5, kept, 6, 7, scratch, 1, top_k));
    try t.expectError(error.UnsupportedShape, o.expertsBackPacked(p, 5, kept, 6, 7, scratch, 1, top_k));
    try t.expectError(error.UnsupportedShape, o.moeBack(e, 5, kept, 6, 7, scratch, 1, top_k));
    try t.expectError(error.UnsupportedShape, o.moeBackPacked(p, 5, kept, 6, 7, scratch, 1, top_k));
    try t.expectEqual(@as(usize, 0), launches);
}
test "train launch bounds: top-k array bounds refuse every public MoE entry before launch" {
    try rejected(0, 32);
    try rejected(17, 32);
    try rejected(2, 1);
    try rejected(std.math.maxInt(usize), 32);
}
test "train launch bounds: routed plus shared sorter count is at most 1024 before launch" {
    try rejected(1, 1024);
    try rejected(1, 0);
    try rejected(1, std.math.maxInt(usize));
}
test "train launch bounds: unpacked and packed experts refuse unsupported 64-column tails before launch" {
    var x: Rig = .{};
    x.init();
    const o = x.ops();
    for ([_][2]usize{ .{ 96, 64 }, .{ 64, 96 }, .{ 0, 64 }, .{ 64, 0 } }) |dims| {
        const e = experts(32, dims[0], dims[1]);
        try t.expectError(error.UnsupportedShape, o.expertsBack(e, 5, kept, 6, 7, scratch, 1, 4));
        try t.expectError(error.UnsupportedShape, o.moeBack(e, 5, kept, 6, 7, scratch, 1, 4));
        const p: q.PackedExperts = .{ .up = 1, .down = 2, .routed = 32, .width = dims[0], .dim = dims[1] };
        try t.expectError(error.UnsupportedShape, o.expertsBackPacked(p, 5, kept, 6, 7, scratch, 1, 4));
        try t.expectError(error.UnsupportedShape, o.moeBackPacked(p, 5, kept, 6, 7, scratch, 1, 4));
    }
    try t.expectEqual(@as(usize, 0), launches);
}
test "train launch bounds: supported expert boundary still launches all forward-adjoint stages" {
    var x: Rig = .{};
    x.init();
    const o = x.ops();
    try o.moeBack(experts(1023, 64, 64), 5, kept, 6, 7, scratch, 1, 16);
    try t.expectEqual(@as(usize, 11), launches);
    launches = 0;
    try o.moeBackPacked(.{ .up = 1, .down = 2, .routed = 512, .width = 640, .dim = 2560 }, 5, kept, 6, 7, scratch, 17, 10);
    try t.expectEqual(@as(usize, 11), launches);
}
