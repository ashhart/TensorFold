const std = @import("std");
const cuda = @import("cuda");
const batch = @import("cuda_batch.zig");
const load = @import("cuda_prompt_load.zig");
const exl3_load = @import("cuda_exl3_load.zig");

const FakeDriver = struct {
    var copies: usize = 0;

    fn copy(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        return 1;
    }

    fn errorName(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = null;
        return cuda.abi.success;
    }

    fn errorString(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = null;
        return cuda.abi.success;
    }

    fn make() cuda.Driver {
        var driver: cuda.Driver = undefined;
        driver.api.cuMemcpyDtoDAsync_v2 = copy;
        driver.api.cuGetErrorName = errorName;
        driver.api.cuGetErrorString = errorString;
        return driver;
    }
};

const MockTri = struct {
    own: ?*const @import("cuda_glue.zig").Glue = null,

    pub fn routerMany(_: MockTri, _: u64, _: u64, _: u64, _: usize, _: usize) !void {}
    pub fn topkMany(_: MockTri, _: u64, _: u64, _: u64, _: usize) !void {}
};

test "exl3: Keep dispatch rejects an unattached expert bank before legacy copy" {
    FakeDriver.copies = 0;
    var driver = FakeDriver.make();
    var host_bytes: [0]u8 align(16) = .{};
    var experts: exl3_load.Experts = .{
        .gpa = std.testing.allocator,
        .ops = undefined,
        .layer = undefined,
        .bank = .{ .d = &driver, .bytes = &host_bytes },
    };
    var layer: load.Layer = undefined;
    layer.router.ptr = 1;
    layer.exl3 = &experts;
    var scratch: batch.Scratch = undefined;
    scratch.mixed = 2;
    scratch.router_l = 3;
    scratch.picks = 4;
    scratch.wts = 5;
    var pass: batch.Pass(MockTri) = .{
        .gpa = std.testing.allocator,
        .io = undefined,
        .stream = .{ .d = &driver, .handle = null },
        .tri = .{},
        .k = undefined,
        .s = &scratch,
        .m = undefined,
        .seq = undefined,
        .hooks = .{},
        .rows = 1,
    };

    try std.testing.expectError(error.Exl3PoolNotAttached, pass.moeBlockKeep(&layer, .{ .moe_x = 9 }));
    try std.testing.expectEqual(@as(usize, 0), FakeDriver.copies);
}
