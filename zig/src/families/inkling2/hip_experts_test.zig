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

fn moe(r: *const hip.Runtime) k.Kernels {
    var kern = mocked(r);
    kern.nvfp4_down = fun(r, 4);
    kern.mxfp8_down = fun(r, 5);
    return kern;
}

const nv: k.Weights = .{ .format = .nvfp4, .codes = 0x1000, .scales = 0x2000, .amax = 0x3000, .den = 0x4000 };
const mx: k.Weights = .{ .format = .mxfp8, .codes = 0x1000, .scales = 0x2000 };
const E = error.Inkling2ExpertsShape;

fn stream(r: *hip.Runtime) hip.Stream {
    r.api.hipModuleLaunchKernel = Mock.launch;
    Mock.count = 0;
    return .{ .r = r, .handle = @ptrFromInt(32) };
}

test "inkling2 experts hip: down launches a block a 64 outputs and route, over each route's own activations" {
    var r: hip.Runtime = undefined;
    const s = stream(&r);
    const kern = moe(&r);
    try k.down(&kern, s, nv, 0, 0, 0, .{ .rows = 2, .n = k.hidden, .k = k.width });
    try k.down(&kern, s, mx, 0, 0, 0, .{ .rows = 1, .n = 1, .k = 32 });
    try t.expectEqual(@as(usize, 2), Mock.count);
    try t.expectEqual([2]usize{ 4, 5 }, Mock.which[0..2].*);
    try t.expectEqual([3]u32{ 36, 16, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 1, 8, 1 }, Mock.grids[1]);
    try t.expectEqual([2]u32{ 64, 64 }, Mock.blocks[0..2].*);
}

test "inkling2 experts hip: down refuses reductions past 2,304 or off the format's group, misaligned codes, empty calls" {
    var r: hip.Runtime = undefined;
    const s = stream(&r);
    const kern = moe(&r);
    try t.expectError(E, k.down(&kern, s, nv, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = k.max_k + 32 }));
    try t.expectError(E, k.down(&kern, s, nv, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = 24 })); // not whole 16-input groups
    try t.expectError(E, k.down(&kern, s, mx, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = 48 })); // not whole 32-input groups
    try t.expectError(E, k.down(&kern, s, mx, 0, 0, 0, .{ .rows = 0, .n = k.hidden, .k = k.width }));
    try t.expectError(E, k.down(&kern, s, mx, 0, 0, 0, .{ .rows = 1, .n = 0, .k = k.width }));
    try t.expectError(E, k.down(&kern, s, mx, 0, 0, 0, .{ .rows = 8192, .n = k.hidden, .k = k.width })); // 65,536 routes
    var odd = mx;
    odd.codes += 8; // MXFP8 rows load 16 bytes at a time
    try t.expectError(error.Inkling2ExpertsAlign, k.down(&kern, s, odd, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = k.width }));
    var odd_nv = nv;
    odd_nv.codes += 4; // NVFP4 rows load 8 bytes at a time
    try t.expectError(error.Inkling2ExpertsAlign, k.down(&kern, s, odd_nv, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = k.width }));
    var bare = nv;
    bare.amax = 0; // NVFP4 needs its amax and denominator
    try t.expectError(E, k.down(&kern, s, bare, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = k.width }));
    try t.expectEqual(@as(usize, 0), Mock.count);
    odd_nv.codes += 4; // eight bytes in: aligned again
    try k.down(&kern, s, odd_nv, 0, 0, 0, .{ .rows = 1, .n = k.hidden, .k = k.width });
    try t.expectEqual(@as(usize, 1), Mock.count);
}
