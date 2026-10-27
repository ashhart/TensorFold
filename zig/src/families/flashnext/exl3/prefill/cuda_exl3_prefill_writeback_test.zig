const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const dims = wt.dims;
const slots = wt.slots;
const streams = wt.streams;

const Call = struct { branch: u64, y: u64, weights: u64, mode: usize, topk: usize, slots: usize, rows: usize };

const Tri = struct {
    call: *Call,

    pub fn hcWriteback(self: Tri, _: u64, _: u64, _: u64, branch: u64, _: []const u8, _: u64, y: u64, _: []const u8, weights: u64, _: []const u8, _: usize, rows: usize, _: usize, _: usize, mode: usize, topk: usize, count: usize, _: usize) !void {
        self.call.* = .{ .branch = branch, .y = y, .weights = weights, .mode = mode, .topk = topk, .slots = count, .rows = rows };
    }
};

const Copies = struct {
    s: *batch.Scratch,
    calls: usize = 0,
    last: struct { dst: u64, src: u64, bytes: usize } = undefined,

    pub fn copy(self: *Copies, dst: u64, src: u64, bytes: usize) !void {
        self.calls += 1;
        self.last = .{ .dst = dst, .src = src, .bytes = bytes };
    }
};

pub fn run(comptime Deep: type, comptime layout: anytype, comptime view: anytype, comptime pending: anytype, comptime compact: anytype, comptime select: anytype) !void {
    var driver: cuda.Driver = undefined;
    var dp: Deep = undefined;
    dp.shared = null;
    dp.world = 2;
    dp.rows = 7;
    dp.lay = layout(dp.rows, dp.world);
    dp.dev = .{ .d = &driver, .ptr = 0x100000, .len = dp.lay.total };
    dp.reduced_branch = false;
    dp.unit_weights = .{ .d = &driver, .ptr = 0x900000, .len = dp.rows * 4 };
    try select(&dp, true);
    try std.testing.expect(dp.reduced_branch);
    var base: batch.Scratch = undefined;
    base.dummy = 0xdead;
    var call: Call = undefined;
    var pass: batch.Pass(Tri) = undefined;
    pass.pending_tp = null;
    pass.tri = .{ .call = &call };
    const sum = dp.dev.ptr + dp.lay.sum;
    const sub: usize = 3;
    for (0..3) |block| {
        const at = block * sub;
        const rows = @min(sub, dp.rows - at);
        var scratch = view(&base, &dp, at);
        pass.s = &scratch;
        pass.rows = rows;
        try std.testing.expectEqual(sum + at * dims * 4, scratch.moe_f32);
        pending(&pass, &dp, null, at);
        try std.testing.expect(!pass.pending_f32);
        pending(&pass, &dp, 4, at);
        try pass.writeback(0, rows);
        try std.testing.expectEqual(Call{ .branch = base.dummy, .y = sum + at * dims * 4, .weights = dp.unit_weights.?.ptr + at * 4, .mode = 2, .topk = 0, .slots = 1, .rows = rows }, call);
        try std.testing.expect(pass.pending == null and !pass.pending_f32 and pass.pending_f32_weights == null);
        if (rows > 1) {
            pending(&pass, &dp, 4, at);
            try pass.writeback(1, rows - 1);
            try std.testing.expectEqual(sum + (at + 1) * dims * 4, call.y);
            try std.testing.expectEqual(dp.unit_weights.?.ptr + (at + 1) * 4, call.weights);
        }
        var copies: Copies = .{ .s = &scratch };
        try compact(&copies, true, rows - 1);
        try std.testing.expectEqual(@as(usize, if (rows > 1) 4 else 0), copies.calls);
        if (rows > 1) {
            try std.testing.expectEqual(scratch.moe_f32, copies.last.dst);
            try std.testing.expectEqual(scratch.moe_f32 + (rows - 1) * dims * 4, copies.last.src);
            try std.testing.expectEqual(dims * 4, copies.last.bytes);
        }
    }
    var scratch = view(&base, &dp, 0);
    pass.s = &scratch;
    pass.rows = sub;
    try select(&dp, false);
    pending(&pass, &dp, 4, 0);
    try pass.writeback(0, sub);
    try std.testing.expectEqual(@as(usize, slots), call.slots);
    try std.testing.expectEqual(@as(usize, slots - 1), call.topk);
    try std.testing.expectEqual(scratch.y, call.y);
    pass.pending = 4;
    pass.pending_f32 = true;
    pass.pending_f32_weights = null;
    try pass.writeback(0, sub);
    try std.testing.expectEqual(@as(usize, 4), call.mode);
    try std.testing.expectEqual(scratch.moe_f32, call.branch);
    var copies: Copies = .{ .s = &scratch };
    try compact(&copies, false, 2);
    try std.testing.expectEqual(@as(usize, 5), copies.calls);
    try std.testing.expectEqual(scratch.wts, copies.last.dst);
    try std.testing.expectEqual(scratch.wts + 2 * slots * 4, copies.last.src);
    dp.world = 1;
    base.moe_f32 = 0xbeef;
    const single_rank = view(&base, &dp, sub);
    try std.testing.expectEqual(base.moe_f32, single_rank.moe_f32);
    try rounding();
}

fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn promote(value: u16) f32 {
    return @bitCast(@as(u32, value) << 16);
}

fn rounding() !void {
    const values = [_]f32{ 0.0, -0.0, 0x1p-126, -0x1p-126, 0x1p-133, -0x1p-133, 0x1.01p0, -0x1.01p0, 0x1.03p0, -0x1.03p0, 0x1p40, -0x1p40 };
    const hidden = [_]u16{ 0, 0x8000, 0x3b34, 0xbe16, 0x3f80, 0xbf80 };
    const inject = [_]u16{ 0, 0x8000, 0x3c9a, 0x3f80, 0xbf80 };
    for (values) |sum| {
        const direct = @mulAdd(f32, sum, 1, 0);
        var old = direct;
        for (1..slots) |slot| old = @mulAdd(f32, if (slot % 2 == 0) @as(f32, -7) else 13, 0, old);
        for (hidden) |bits| for (inject) |gate| {
            const want = bf16(@mulAdd(f32, promote(bf16(old)), promote(gate), promote(bits)));
            const got = bf16(@mulAdd(f32, promote(bf16(direct)), promote(gate), promote(bits)));
            try std.testing.expectEqual(want, got);
        };
    }
    const wrong = bf16(@mulAdd(f32, promote(bf16(-@as(f32, 0.0))), 1, -@as(f32, 0.0)));
    const correct = bf16(@mulAdd(f32, @mulAdd(f32, -0.0, 1, 0), 1, -@as(f32, 0.0)));
    try std.testing.expectEqual(@as(u16, 0x8000), wrong);
    try std.testing.expectEqual(@as(u16, 0), correct);
}
