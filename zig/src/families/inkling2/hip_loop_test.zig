//! Inkling 2's HIP loop launches without a GPU: the grids and blocks a mock runtime records, and what is refused.
const std = @import("std");
const hip = @import("hip");
const k = @import("hip_loop.zig");
const t = std.testing;

const Mock = struct {
    var grids: [8][3]u32 = undefined;
    var blocks: [8]u32 = undefined;
    var which: [8]usize = undefined;
    var count: usize = 0;
    fn launch(f: hip.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: hip.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) hip.abi.Result {
        grids[count] = .{ gx, gy, gz };
        blocks[count] = bx;
        which[count] = @intFromPtr(f);
        count += 1;
        return 0;
    }
};

fn fun(r: *const hip.Runtime, i: usize) hip.Function {
    return .{ .r = r, .handle = @ptrFromInt(i) };
}

fn mocked(r: *hip.Runtime) k.Kernels {
    r.api.hipModuleLaunchKernel = Mock.launch;
    Mock.count = 0;
    var kern: k.Kernels = undefined;
    kern.mod = .{ .r = r, .handle = null };
    kern.math = fun(r, 1);
    return kern;
}

fn stream(r: *const hip.Runtime) hip.Stream {
    return .{ .r = r, .handle = @ptrFromInt(32) };
}

const E = error.Inkling2LoopShape;

test "inkling2 loop hip: the math probe launches 256 threads a block and refuses an empty call" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    try k.math(&kern, s, 0, 0, 1 << 26, .tanh);
    try k.math(&kern, s, 0, 0, 257, .device_rsqrt);
    try t.expectError(E, k.math(&kern, s, 0, 0, 0, .exp));
    try t.expectEqual(@as(usize, 2), Mock.count);
    try t.expectEqual([3]u32{ 1 << 18, 1, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 2, 1, 1 }, Mock.grids[1]);
    try t.expectEqual([2]u32{ 256, 256 }, Mock.blocks[0..2].*);
}
