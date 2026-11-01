//! Public MoE backward interfaces reject unsupported array/tile geometry before any CUDA launch.
const std = @import("std");
const q = @import("train_q4.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig");
const t = std.testing;
var launches: usize = 0;
var rounding: [2]c_int = @splat(-1);
var symbols: [32]usize = @splat(0);
fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
    symbols[launches] = @intFromPtr(f.?);
    if (@intFromPtr(f.?) == 105) rounding[0] = @as(*const c_int, @ptrCast(@alignCast(args.?[7].?))).*;
    if (@intFromPtr(f.?) == 108) rounding[1] = @as(*const c_int, @ptrCast(@alignCast(args.?[11].?))).*;
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
test "Quantized adjoints: top-k array bounds refuse every public MoE entry before launch" {
    try rejected(0, 32);
    try rejected(17, 32);
    try rejected(2, 1);
    try rejected(std.math.maxInt(usize), 32);
}
test "Quantized adjoints: routed plus shared sorter count is at most 1024 before launch" {
    try rejected(1, 1024);
    try rejected(1, 0);
    try rejected(1, std.math.maxInt(usize));
}
test "Quantized adjoints: unpacked and packed experts refuse unsupported 64-column tails before launch" {
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
test "Quantized adjoints: supported expert boundary still launches all forward-adjoint stages" {
    var x: Rig = .{};
    x.init();
    const o = x.ops();
    try o.moeBack(experts(1023, 64, 64), 5, kept, 6, 7, scratch, 1, 16);
    try t.expectEqual(@as(usize, 11), launches);
    launches = 0;
    try o.moeBackPacked(.{ .up = 1, .down = 2, .routed = 512, .width = 640, .dim = 2560 }, 5, kept, 6, 7, scratch, 17, 10);
    try t.expectEqual(@as(usize, 11), launches);
}

test "MoE packed: group 64 selects its two adjoints; group 32 keeps its original symbols" {
    for ([_]usize{ 32, 64 }) |group| {
        var x: Rig = .{};
        x.init();
        for (&x.k.fq, 0..) |*f, i| f.handle = @ptrFromInt(100 + i);
        try x.ops().moeBackPacked(.{ .up = 1, .down = 2, .routed = 256, .width = 512, .dim = 2048, .group = group }, 5, kept, 6, 7, scratch, 3, 8);
        try t.expectEqual(@as(usize, 11), launches);
        const nt: usize = if (group == 64) 110 else 106;
        const mm: usize = if (group == 64) 111 else 107;
        try t.expectEqual(nt, symbols[3]);
        try t.expectEqual(nt, symbols[4]);
        try t.expectEqual(mm, symbols[5]);
        try t.expectEqual(mm, symbols[7]);
        try t.expectEqual(mm, symbols[8]);
    }
}

test "MoE packed: serving rounding reaches both adjoints and the legacy default stays off" {
    for ([_]bool{ false, true }) |served| {
        var x: Rig = .{};
        x.init();
        for (&x.k.fq, 0..) |*f, i| f.handle = @ptrFromInt(100 + i);
        var inputs = kept;
        inputs.served_rounding = served;
        rounding = @splat(-1);
        try x.ops().moeBackPacked(.{ .up = 1, .down = 2, .routed = 256, .width = 512, .dim = 2048, .group = 64 }, 5, inputs, 6, 7, scratch, 3, 8);
        const want: c_int = @intFromBool(served);
        try t.expectEqual([2]c_int{ want, want }, rounding);
    }
    try t.expect(!kept.served_rounding);
}
