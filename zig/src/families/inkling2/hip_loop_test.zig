//! Inkling 2's HIP loop launches without a GPU: the grids and blocks a mock runtime records, and what is refused.
const std = @import("std");
const hip = @import("hip");
const k = @import("hip_loop.zig");
const t = std.testing;

const Mock = struct {
    var grids: [16][3]u32 = undefined;
    var blocks: [16]u32 = undefined;
    var rows: [16]u32 = undefined;
    var which: [16]usize = undefined;
    var count: usize = 0;
    fn launch(f: hip.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, _: c_uint, _: c_uint, _: hip.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) hip.abi.Result {
        grids[count] = .{ gx, gy, gz };
        blocks[count] = bx;
        rows[count] = by;
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
    for (&kern.gate_down, 0..) |*f, i| f.* = fun(r, 20 + i);
    for (&kern.gate_up, 0..) |*f, i| f.* = fun(r, 30 + i);
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

fn gate(base: u64) k.Gate {
    return .{ .h = base, .m = base + 0x100, .down = base + 0x200, .forget = base + 0x300, .gain = base + 0x400, .zeta = base + 0x500, .out = base };
}

test "inkling2 loop hip: gates launch down over (rows, 8 rank groups, gates), then up over (4-row groups, 36, gates)" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    try k.gates(&kern, s, &.{gate(0x10000)}, 1);
    try k.gatesAs(&kern, s, .fma, &.{ gate(0x10000), gate(0x20000), gate(0x30000) }, 9);
    try k.gatesAs(&kern, s, .tanhf, &.{ gate(0x10000), gate(0x20000) }, k.max_rows);
    try k.gatesAs(&kern, s, .unfused_sq, &.{gate(0x10000)}, 2);
    try t.expectEqual(@as(usize, 8), Mock.count);
    try t.expectEqual([8]usize{ 20, 30, 20, 32, 20, 31, 23, 30 }, Mock.which[0..8].*);
    try t.expectEqual([3]u32{ 1, 8, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 1, 36, 1 }, Mock.grids[1]);
    try t.expectEqual([3]u32{ 9, 8, 3 }, Mock.grids[2]);
    try t.expectEqual([3]u32{ 3, 36, 3 }, Mock.grids[3]);
    try t.expectEqual([3]u32{ 2048, 8, 2 }, Mock.grids[4]);
    try t.expectEqual([3]u32{ 512, 36, 2 }, Mock.grids[5]);
    try t.expectEqual([8]u32{ 256, 64, 256, 64, 256, 64, 256, 64 }, Mock.blocks[0..8].*);
    try t.expectEqual([8]u32{ 1, 4, 1, 4, 1, 4, 1, 4 }, Mock.rows[0..8].*); // up: 64 columns by 4 rows
}

test "inkling2 loop hip: each gate control takes its own down or up kernel and the served other half" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    for ([_]k.Gates{ .serial_rho, .butterfly_desc, .unfused_sq, .tanhf, .fma }) |which| try k.gatesAs(&kern, s, which, &.{gate(0x10000)}, 1);
    try t.expectEqual([10]usize{ 21, 30, 22, 30, 23, 30, 20, 31, 20, 32 }, Mock.which[0..10].*);
}

test "inkling2 loop hip: gates refuse none or more than 3, no rows or more than 2,048, and misaligned tensors" {
    var r: hip.Runtime = undefined;
    const kern = mocked(&r);
    const s = stream(&r);
    const g = gate(0x10000);
    try t.expectError(E, k.gates(&kern, s, &.{}, 1));
    try t.expectError(E, k.gates(&kern, s, &.{ g, g, g, g }, 1));
    try t.expectError(E, k.gates(&kern, s, &.{g}, 0));
    try t.expectError(E, k.gates(&kern, s, &.{g}, k.max_rows + 1));
    inline for (@typeInfo(k.Gate).@"struct".field_names) |name| {
        var x = g;
        @field(x, name) += 8; // rows, weights and zeta load and store 16 bytes at a time
        try t.expectError(error.Inkling2LoopAlign, k.gates(&kern, s, &.{ g, x }, 1));
    }
    try t.expectEqual(@as(usize, 0), Mock.count);
}
