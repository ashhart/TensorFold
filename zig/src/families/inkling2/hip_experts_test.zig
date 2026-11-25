//! Inkling 2's HIP expert launches without a GPU: the grids and blocks a mock runtime records, and what is refused.
const std = @import("std");
const hip = @import("hip");
const k = @import("hip_experts.zig");
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

fn mocked(r: *const hip.Runtime) k.Kernels {
    var kern: k.Kernels = undefined;
    kern.mod = .{ .r = r, .handle = null };
    kern.act_factors = fun(r, 1);
    return kern;
}

test "inkling2 experts hip: the factor proof covers all 65,536 bf16 inputs in 256 blocks of 256, and refuses none" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    Mock.count = 0;
    try k.actFactors(&kern, s, 0, 0, 0, k.factor_inputs, false);
    try k.actFactors(&kern, s, 0, 0, 0, 257, true);
    try t.expectError(error.Inkling2ExpertsShape, k.actFactors(&kern, s, 0, 0, 0, 0, false));
    try t.expectEqual(@as(usize, 2), Mock.count);
    try t.expectEqual([3]u32{ 256, 1, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 2, 1, 1 }, Mock.grids[1]);
    try t.expectEqual([2]u32{ 256, 256 }, Mock.blocks[0..2].*);
    try t.expectEqual([2]usize{ 1, 1 }, Mock.which[0..2].*);
}
