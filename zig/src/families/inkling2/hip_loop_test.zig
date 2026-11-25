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
    for (&kern.messages, 0..) |*f, i| f.* = fun(r, 10 + i);
    return kern;
}

fn stream(r: *const hip.Runtime) hip.Stream {
    return .{ .r = r, .handle = @ptrFromInt(32) };
}

const E = error.Inkling2LoopShape;

test "inkling2 loop hip: the query scale and eps are binary32 1/48 and float64 1e-6 narrowed" {
    try t.expectEqual(@as(u32, 0x3caaaaab), @as(u32, @bitCast(k.queryScale())));
    try t.expectEqual(@as(f32, @floatCast(@as(f64, 1e-6))), k.eps);
}

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

test "inkling2 loop hip: messages launch a block of 64 threads a row and message, each control its own kernel" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    const c: k.MessageCall = .{ .bank0 = 0x1000, .bank1 = 0x2000, .qm = 0x3000, .kv = 0x4000, .out = 0x5000, .rows = 1, .count = k.core_layers };
    try k.messages(&kern, s, c);
    var prompt = c;
    prompt.rows = k.max_rows;
    prompt.count = 1;
    try k.messagesAs(&kern, s, .signed_zero, prompt);
    prompt.rows = 5;
    try k.messagesAs(&kern, s, .rsqrt, prompt);
    try k.messagesAs(&kern, s, .serial_rho, prompt);
    try k.messagesAs(&kern, s, .butterfly_desc, prompt);
    try t.expectEqual(@as(usize, 5), Mock.count);
    try t.expectEqual([5]usize{ 10, 13, 11, 14, 15 }, Mock.which[0..5].*);
    try t.expectEqual([3]u32{ 1, 48, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 2048, 4, 1 }, Mock.grids[1]);
    try t.expectEqual([3]u32{ 5, 4, 1 }, Mock.grids[2]);
    try t.expectEqual([5]u32{ 64, 64, 64, 64, 64 }, Mock.blocks[0..5].*);
}

test "inkling2 loop hip: messages refuse no rows, more than 2,048, no or more than 12 core layers, and misaligned rows" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    const c: k.MessageCall = .{ .bank0 = 0x1000, .bank1 = 0x2000, .qm = 0x3000, .kv = 0x4000, .out = 0x5000, .rows = 1, .count = 1 };
    for ([_][2]u32{ .{ 0, 1 }, .{ k.max_rows + 1, 1 }, .{ 1, 0 }, .{ 1, k.core_layers + 1 } }) |bad| {
        var x = c;
        x.rows = bad[0];
        x.count = bad[1];
        try t.expectError(E, k.messages(&kern, s, x));
    }
    inline for (.{ "bank0", "bank1", "qm", "kv", "out" }) |field| {
        var x = c;
        @field(x, field) += 8; // rows load and store 16 bytes at a time
        try t.expectError(error.Inkling2LoopAlign, k.messages(&kern, s, x));
    }
    try t.expectEqual(@as(usize, 0), Mock.count);
}
