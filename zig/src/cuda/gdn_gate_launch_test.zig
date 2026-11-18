//! The real launcher sends the explicit gate after the unchanged shape and epsilon arguments.
const std = @import("std");
const gt = @import("gdn_train.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;
var sent: c_int = -1;
fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
    sent = @as(*const c_int, @ptrCast(@alignCast(args.?[8].?))).*;
    return 0;
}
test "GDN gate launch: default sigmoid and explicit SiLU retain their C argument" {
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = launch;
    var k: gt.Kernels = undefined;
    const f: Function = .{ .d = &d, .handle = @ptrFromInt(16) };
    k.fns = @splat(f);
    const ops: gt.Ops = .{ .k = &k, .s = .{ .d = &d, .handle = null } };
    const sc = gt.Scratch.carve(4096, 2, 16, 48, 5120);
    try ops.normBack(1, 2, sc, .init(2, 16, 48), 1e-6);
    try std.testing.expectEqual(@as(c_int, 0), sent);
    try ops.normBackGate(1, 2, sc, .init(2, 16, 48), 1e-6, .silu);
    try std.testing.expectEqual(@as(c_int, 1), sent);
}
