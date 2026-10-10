//! Plain HIP module launches; no CUDA cluster or programmatic-launch attributes.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
pub const Args = @import("args.zig").Args;
pub const Dim3 = struct { x: u32, y: u32 = 1, z: u32 = 1 };
pub const Config = struct {
    grid: Dim3,
    block: Dim3,
    shared: u32 = 0,
    /// hipModuleLaunchCooperativeKernel: every block resident at once, so blocks may wait on each other.
    cooperative: bool = false,

    pub fn validate(self: Config) runtime.Error!void {
        const g = self.grid;
        const b = self.block;
        if (g.x == 0 or g.y == 0 or g.z == 0 or b.x == 0 or b.y == 0 or b.z == 0)
            return error.Invalid;
        if (b.x > 1024 or b.y > 1024 or b.z > 1024 or @as(u64, b.x) * b.y * b.z > 1024)
            return error.Invalid;
        if (@as(u64, g.x) * b.x > std.math.maxInt(u32) or
            @as(u64, g.y) * b.y > std.math.maxInt(u32) or
            @as(u64, g.z) * b.z > std.math.maxInt(u32)) return error.Invalid;
    }
};

pub fn launch(f: Function, cfg: Config, stream: Stream, args: *Args) runtime.Error!void {
    try cfg.validate();
    if (f.r != stream.r) return error.Invalid;
    const g = cfg.grid;
    const b = cfg.block;
    if (cfg.cooperative) {
        try runtime.check(f.r.api.hipModuleLaunchCooperativeKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers()));
    } else {
        try runtime.check(f.r.api.hipModuleLaunchKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers(), null));
    }
}

test "launch geometry refuses zero, oversized blocks and dimension overflow" {
    try (Config{ .grid = .{ .x = 3 }, .block = .{ .x = 256 } }).validate();
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{ .x = 0 }, .block = .{ .x = 1 } }).validate());
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{ .x = 1 }, .block = .{ .x = 1024, .y = 2 } }).validate());
    try std.testing.expectError(error.Invalid, (Config{ .grid = .{ .x = std.math.maxInt(u32) }, .block = .{ .x = 2 } }).validate());
}

test "a cooperative launch takes HIP's cooperative entry point" {
    const Mock = struct {
        var which: u8 = 0;
        fn plain(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            which = 'p';
            return 0;
        }
        fn coop(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            which = 'c';
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.plain;
    r.api.hipModuleLaunchCooperativeKernel = Mock.coop;
    const f: Function = .{ .r = &r, .handle = @ptrFromInt(64) };
    const s: Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    var args: Args = .{};
    try launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, s, &args);
    try std.testing.expectEqual(@as(u8, 'p'), Mock.which);
    try launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .cooperative = true }, s, &args);
    try std.testing.expectEqual(@as(u8, 'c'), Mock.which);
}
