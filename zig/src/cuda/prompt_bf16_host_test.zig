//! Observe real projection dispatch at the CUDA ABI without running a GPU.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const qmmf = @import("qmmf.zig");
const Prompt = @import("nvfp4.zig").Prompt;
const Linear = @import("qlinear.zig").Linear;

const Mock = struct {
    var handle: usize = 0;
    var grid: [3]u32 = undefined;
    var block: [3]u32 = undefined;
    var shared: u32 = 0;
    var args: [13]u64 = undefined;
    var calls: usize = 0;

    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        handle = @intFromPtr(f);
        grid = .{ gx, gy, gz };
        block = .{ bx, by, bz };
        shared = sm;
        const count: usize = if (handle >= 3) 11 else 13;
        for (0..count) |i| {
            const wide = i < (if (handle >= 3) @as(usize, 2) else 3) or i == (if (handle >= 3) @as(usize, 3) else 4) or (handle < 3 and i == 5);
            args[i] = if (wide) @as(*align(1) const u64, @ptrCast(params.?[i].?)).* else @as(*align(1) const u32, @ptrCast(params.?[i].?)).*;
        }
        calls += 1;
        return abi.success;
    }
};

fn function(d: *const Driver, id: usize) Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

fn lane(d: *const Driver) qmmf.Lane {
    const f = function(d, 1);
    return .{ .mod = .{ .d = d, .handle = null }, .fns = @splat(@splat(f)), .f32_fns = @splat(@splat(f)), .reduce = @splat(f), .half = @splat(@splat(@splat(f))), .clusters = false };
}

fn prompt(d: *const Driver) Prompt {
    const f = function(d, 2);
    return .{ .mod = .{ .d = d, .handle = null }, .fp4 = f, .fp8 = f, .fp4_f32 = f, .lean = f, .lean_f32 = f, .sms = 16, .bf16_serving = true, .bf16 = function(d, 3), .bf16_f32 = function(d, 4) };
}

test "BF16 prompt port: threshold, family opt-in, cached switch and decode keep the lane" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    const s: Stream = .{ .d = &d, .handle = null };
    const l = lane(&d);
    var p = prompt(&d);
    var linear: Linear = .{ .lane = &l, .nvfp4_prompt = &p, .bf16_prompt = true };
    const w: qmmf.Weight = .{ .mode = .bf16, .codes = 0x2000, .scales = 0, .n = 48, .k = 5120, .npad = 128 };
    for ([_]usize{ 256, 512, 768, 1023, 1024, 1025, 1536, 2048 }) |rows| {
        Mock.calls = 0;
        try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, rows);
        try std.testing.expectEqual(@as(usize, if (rows >= 1024) 3 else 1), Mock.handle);
        try std.testing.expectEqual(@as(usize, 1), Mock.calls);
    }
    try linear.decode(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, 1024);
    try std.testing.expectEqual(@as(usize, 1), Mock.handle);
    linear.bf16_prompt = false;
    try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, 1024);
    try std.testing.expectEqual(@as(usize, 1), Mock.handle);
    linear.bf16_prompt = true;
    p.bf16_serving = false;
    try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, 1024);
    try std.testing.expectEqual(@as(usize, 1), Mock.handle);
    try std.testing.expectEqual(@as(usize, 3), @intFromPtr(p.bf16.handle.?));
    try std.testing.expectEqual(@as(usize, 4), @intFromPtr(p.bf16_f32.handle.?));
    linear.nvfp4_prompt = null;
    try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, 1024);
    try std.testing.expectEqual(@as(usize, 1), Mock.handle);
}

test "BF16 prompt port: actual tile ABI retains stride, scale, tail and shape-fixed K slices" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    const s: Stream = .{ .d = &d, .handle = null };
    const p = prompt(&d);
    const l = lane(&d);
    const linear: Linear = .{ .lane = &l, .nvfp4_prompt = &p, .bf16_prompt = true };
    for ([_][2]u32{ .{ 3072, 512 }, .{ 640, 1024 }, .{ 96, 2048 }, .{ 48, 5120 }, .{ 70, 128 } }, [_]u32{ 1, 2, 4, 8, 1 }) |nk, sk| {
        const w: qmmf.Weight = .{ .mode = .bf16, .codes = 0x2200, .scales = 0, .scale = 0.125, .n = nk[0], .k = nk[1], .npad = @intCast(@import("nvfp4.zig").padded(nk[0])) };
        const stride = w.k + 64;
        Mock.calls = 0;
        try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1100, .ldx = stride }, 0x3300, 1025);
        try std.testing.expectEqual(@as(usize, 1), Mock.calls);
        try std.testing.expectEqual(@as(usize, 3), Mock.handle);
        try std.testing.expectEqual([3]u32{ 9 * ((w.n + 127) / 128), 1, 1 }, Mock.grid);
        try std.testing.expectEqual([3]u32{ 256, 1, 1 }, Mock.block);
        try std.testing.expectEqual(@as(u32, 98304), Mock.shared);
        const band = @max(1, @min(9, (12 << 20) / (128 * @as(usize, w.k) * 2)));
        try std.testing.expectEqualSlices(u64, &.{ 0x1100, 0x2200, @as(u32, @bitCast(w.scale)), 0x3300, 1025, w.n, w.k, sk, w.npad, stride, band }, Mock.args[0..11]);
        try p.matmulOut(s, 0x1100, stride, 1025, w, 0x3300, .f32);
        try std.testing.expectEqual(@as(usize, 4), Mock.handle);
    }
    const w: qmmf.Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 48, .k = 5120, .npad = 128 };
    try p.matmulOut(s, 0x1100, 0, 1, w, 0x3300, .bf16);
    try std.testing.expectEqual(@as(u64, w.k), Mock.args[9]);
}
