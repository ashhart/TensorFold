//! The real tap plan/transaction on poisoned storage, with an independent old gather-and-absorb oracle.
const std = @import("std");
const t = std.testing;
const policy = @import("tap_prompt.zig");
const Buffer = struct {
    data: []u8,
    pub fn length(b: @This()) usize {
        return b.data.len;
    }
};
const Taps = struct {
    buffer: Buffer,
    capacity: u32,
    rows: u32,
    width: u32,
    count: usize,
    complete_ok: bool = true,
    pub fn complete(s: @This()) !void {
        if (!s.complete_ok) return error.IncompleteTaps;
    }
};
const Ring = struct { buf: Buffer, end: *u64, window: u32, stride: usize };
const Context = struct {
    taps: Taps,
    ring: Ring,
    calls: *usize,
    fail: bool = false,
    pub fn write(c: @This(), p: policy.Args) !void {
        if (c.fail) return error.PlantedGpuFailure;
        c.calls.* += 1;
        for (0..p.rows) |i| {
            const row = p.first + i;
            const slot: usize = @intCast((p.start + row) % p.window);
            for (0..p.planes) |plane| for (0..p.width) |column| {
                const src = ((plane * p.capacity + row) * p.width + column) * 2;
                const dst = slot * c.ring.stride + (plane * p.width + column) * 2;
                @memcpy(c.ring.buf.data[dst..][0..2], c.taps.buffer.data[src..][0..2]);
            };
        }
    }
};
fn oracle(out: []u8, source: []const u8, capacity: usize, width: usize, planes: usize, window: usize, start: u64, rows: usize) void {
    const stride = width * planes * 2;
    for (0..rows) |row| {
        const slot: usize = @intCast((start + row) % window);
        for (0..planes) |plane| @memcpy(out[slot * stride + plane * width * 2 ..][0 .. width * 2], source[(plane * capacity + row) * width * 2 ..][0 .. width * 2]);
    }
}
test "tap lean: bulk plan equals legacy bytes at tails wraps all planes and64-bit positions" {
    const a = t.allocator;
    for ([_]u32{ 1, 16, 17, 127, 128, 129, 512, 2048 }) |rows| for ([_]u32{ 16, 2048 }) |window| for ([_]usize{ 5, 8 }) |planes| for ([_]u64{ 0, 15, 2047, (1 << 33) + 2045 }) |start| {
        const stride = 3 * planes * 2;
        const source = try a.alloc(u8, rows * stride);
        defer a.free(source);
        for (source, 0..) |*v, i| v.* = @truncate(i * 37 + i / 11);
        const got = try a.alloc(u8, window * stride);
        defer a.free(got);
        @memset(got, 0xa5);
        const want = try a.dupe(u8, got);
        defer a.free(want);
        var end = start;
        var calls: usize = 0;
        try policy.absorb(Context{ .taps = .{ .buffer = .{ .data = source }, .capacity = rows, .rows = rows, .width = 3, .count = planes }, .ring = .{ .buf = .{ .data = got }, .end = &end, .window = window, .stride = stride }, .calls = &calls }, rows);
        oracle(want, source, rows, 3, planes, window, start, rows);
        try t.expectEqualSlices(u8, want, got);
        try t.expectEqual(start + rows, end);
        try t.expectEqual(@as(usize, 1), calls);
    };
}
const Direct = struct {
    source: [512 * 2]u8 = @splat(0),
    got: [2048 * 2]u8 = @splat(0xa5),
    want: [2048 * 2]u8 = @splat(0xa5),
    end: u64 = 0,
    calls: usize = 0,
    skipped: usize = 0,
    target_calls: usize = 0,
    invalid: bool = false,
    fail_after: ?usize = null,
    flag: bool = true,
    pub fn enabled(d: *@This()) bool {
        return d.flag;
    }
    pub fn chunk(_: *@This(), requested: usize) usize {
        return requested;
    }
    pub fn window(_: *@This()) u32 {
        return 2048;
    }
    pub fn invalidate(d: *@This()) void {
        d.invalid = true;
    }
    pub fn feed(d: *@This(), first: usize, count: usize, omit: bool, _: bool) !void {
        if (d.fail_after) |limit| if (d.target_calls == limit) return error.PlantedTargetFailure;
        d.target_calls += 1;
        for (d.source[0 .. count * 2], 0..) |*v, i| v.* = @truncate(first * 19 + i);
        oracle(&d.want, &d.source, count, 1, 1, 2048, first, count);
        if (omit) {
            @memset(&d.source, 0xde);
            d.end += count;
            d.skipped += 1;
        } else try policy.absorb(Context{ .taps = .{ .buffer = .{ .data = &d.source }, .capacity = @intCast(count), .rows = @intCast(count), .width = 1, .count = 1 }, .ring = .{ .buf = .{ .data = &d.got }, .end = &d.end, .window = 2048, .stride = 2 }, .calls = &d.calls }, @intCast(count));
        // A native observation at this end is live, so that caller retains its original stores.
        std.debug.assert(!policy.dead(d.end, d.end, 2048));
    }
};
test "tap lean: actual direct prefill loop poisons skipped planes and matches the entire legacy ring" {
    for ([_]usize{ 1, 127, 128, 129, 512 }) |cut| {
        var d: Direct = .{};
        try policy.prefill(&d, 8192, cut);
        try t.expectEqualSlices(u8, &d.want, &d.got);
        try t.expectEqual(@as(u64, 8192), d.end);
        try t.expect(d.skipped > 0 and !d.invalid);
        var old: Direct = .{ .flag = false };
        try policy.prefill(&old, 8192, cut);
        try t.expectEqualSlices(u8, &old.got, &d.got);
        try t.expectEqual(old.target_calls, d.target_calls);
        try t.expectEqual(@as(usize, 0), old.skipped);
    }
    try t.expect(!policy.dead(512, 2048, 2048));
    try t.expect(policy.dead(512, 2560, 2048));
    try t.expect(!policy.dead(512, 511, 2048));
    try t.expect(!policy.dead(512, 8192, 0));
}
test "tap lean: failed direct prefill after deferred stores invalidates the drafter without retry" {
    var d: Direct = .{ .fail_after = 3 };
    try t.expectError(error.PlantedTargetFailure, policy.prefill(&d, 8192, 512));
    try t.expect(d.invalid);
    try t.expectEqual(@as(usize, 3), d.target_calls);
    var old: Direct = .{ .flag = false, .fail_after = 3 };
    try t.expectError(error.PlantedTargetFailure, policy.prefill(&old, 8192, 512));
    try t.expect(!old.invalid);
}
test "tap lean: failed write incomplete taps bad bounds and overflow never publish an end" {
    var source: [32]u8 = @splat(0);
    var dst: [32]u8 = @splat(0xa5);
    var end: u64 = 7;
    var calls: usize = 0;
    var ctx: Context = .{ .taps = .{ .buffer = .{ .data = &source }, .capacity = 16, .rows = 16, .width = 1, .count = 1 }, .ring = .{ .buf = .{ .data = &dst }, .end = &end, .window = 16, .stride = 2 }, .calls = &calls, .fail = true };
    try t.expectError(error.PlantedGpuFailure, policy.absorb(ctx, 16));
    try t.expectEqual(@as(u64, 7), end);
    ctx.fail = false;
    ctx.taps.complete_ok = false;
    try t.expectError(error.IncompleteTaps, policy.absorb(ctx, 16));
    ctx.taps.complete_ok = true;
    try t.expectError(error.BadTapGather, policy.absorb(ctx, 17));
    ctx.ring.stride = 1;
    try t.expectError(error.BadTapGather, policy.absorb(ctx, 16));
    ctx.ring.stride = 2;
    end = std.math.maxInt(u64);
    try t.expectError(error.Overflow, policy.absorb(ctx, 1));
    try t.expectEqual(@as(usize, 0), calls);
    for (dst) |v| try t.expectEqual(@as(u8, 0xa5), v);
}
