//! Exact-capacity ownership, footprint and failure checks independent of CUDA or model allocation.
const std = @import("std");
const policy = @import("seq_pool.zig");
const Object = struct { alive: bool = true };
const Pool = policy.Pool(Object);

fn free(object: *Object) !void {
    try std.testing.expect(object.alive);
    object.alive = false;
}

test "default-off and invalid settings cannot silently enable retention" {
    try std.testing.expect(!try policy.parse(null));
    try std.testing.expect(!try policy.parse("0"));
    try std.testing.expect(try policy.parse("1"));
    for ([_][]const u8{ "", "2", "true", "-1" }) |value|
        try std.testing.expectError(error.InvalidSeqReuse, policy.parse(value));
    var p = Pool.init(false, 16);
    var object: Object = .{};
    try std.testing.expect(!p.retain(&object, 0, 158, 100));
    try std.testing.expect(p.take(158) == null);
    try std.testing.expect(object.alive);
}

test "take transfers exclusive ownership and does not round capacity or change the object" {
    var p = Pool.init(true, 2);
    var first: Object = .{};
    var second: Object = .{};
    try std.testing.expect(p.retain(&first, 0, 158, 100));
    try std.testing.expect(p.retain(&second, 0, 272, 120));
    try std.testing.expect(p.take(157) == null);
    try std.testing.expect(p.take(159) == null);
    const taken = p.take(158).?;
    try std.testing.expect(taken == &first and taken.alive);
    try std.testing.expect(p.take(158) == null);
    try std.testing.expectEqual(@as(usize, 1), p.stats.retained_slots);
    try std.testing.expectEqual(@as(usize, 120), p.stats.retained_bytes);
    try free(taken);
    try free(p.evict().?);
    try std.testing.expect(p.evict() == null);
    try std.testing.expect(!first.alive and !second.alive);
}

test "capacity miss must evict unused ownership before a fresh active slot fits" {
    var p = Pool.init(true, 2);
    var retained: Object = .{};
    var active: Object = .{};
    var replacement: Object = .{};
    try std.testing.expect(p.retain(&retained, 1, 158, 100));
    try std.testing.expect(!p.hasRoom(1));
    try std.testing.expect(p.take(272) == null);
    try free(p.evict().?);
    try std.testing.expect(p.hasRoom(1));
    // The caller can now own active and replacement, never active+retained+replacement.
    try std.testing.expect(!p.retain(&active, 2, 158, 100));
    try std.testing.expectEqual(@as(u64, 1), p.stats.evictions);
    try free(&active);
    try free(&replacement);
    try std.testing.expect(!retained.alive);
}

test "capacity miss drains every retained byte even with room for a fresh large allocation" {
    const Ledger = struct {
        freed: usize = 0,
        fn release(x: *@This(), object: *Object) void {
            std.debug.assert(object.alive);
            object.alive = false;
            x.freed += 1;
        }
    };
    var p = Pool.init(true, 16);
    var objects: [2]Object = @splat(.{});
    var ledger: Ledger = .{};
    try std.testing.expect(p.retain(&objects[0], 0, 158, 100));
    try std.testing.expect(p.retain(&objects[1], 0, 272, 120));
    try std.testing.expect(p.hasRoom(1));
    try std.testing.expect(p.take(16384) == null);
    p.beforeCreate(&ledger, Ledger.release);
    try std.testing.expectEqual(@as(usize, 2), ledger.freed);
    try std.testing.expectEqual(@as(usize, 0), p.stats.retained_bytes);
    try std.testing.expectEqual(@as(usize, 0), p.stats.retained_slots);
    for (objects) |object| try std.testing.expect(!object.alive);
}

test "total ownership byte cap short-capacity cap and configured stream bound are enforced" {
    var p = Pool.init(true, 20);
    var objects: [16]Object = @splat(.{});
    try std.testing.expectEqual(@as(usize, 16), p.limit);
    try std.testing.expect(!p.retain(&objects[0], 0, 513, 1));
    try std.testing.expect(!p.retain(&objects[0], 0, 512, policy.max_bytes + 1));
    try std.testing.expect(p.retain(&objects[0], 15, 512, policy.max_bytes));
    try std.testing.expect(!p.retain(&objects[1], 14, 158, 1));
    try free(p.evict().?);
    p = Pool.init(true, 2);
    try std.testing.expect(!p.retain(&objects[1], 2, 158, 1));
    try std.testing.expect(p.retain(&objects[1], 1, 158, 1));
    try std.testing.expect(!p.retain(&objects[2], 1, 158, 1));
    try free(p.evict().?);
    try std.testing.expectEqual(@as(usize, 0), p.stats.retained_bytes);
}

test "failure disables reuse and shutdown drains each detached object once" {
    var p = Pool.init(true, 3);
    var objects: [3]Object = @splat(.{});
    for (&objects, 0..) |*object, i| try std.testing.expect(p.retain(object, 0, 158, i + 1));
    p.fail();
    try std.testing.expect(p.stats.failed);
    try std.testing.expect(p.take(158) == null);
    try std.testing.expect(!p.retain(&objects[0], 0, 158, 1));
    while (p.evict()) |object| try free(object);
    for (objects) |object| try std.testing.expect(!object.alive);
    try std.testing.expectEqual(@as(usize, 0), p.stats.retained_slots);
    try std.testing.expectEqual(@as(usize, 0), p.stats.retained_bytes);
    try std.testing.expect(p.evict() == null);
}

const Stream = struct {
    finished: bool = true,
    reason: enum { none, stop, length, cancelled } = .length,
    cancelled: bool = false,
    cached: u32 = 14,
    prompt_len: u32 = 14,
    reuse_failed: bool = false,
    reuse: struct {
        saved: ?*anyopaque = null,
        at: u32 = 0,
        marks: []const u32 = &.{},
        hook: ?u8 = null,
    } = .{},
    pub fn isCancelled(s: *Stream) bool {
        return s.cancelled;
    }
};

test "only normally completed settled uncancelled proven full-N restored streams may transfer to the pool" {
    var s: Stream = .{};
    try std.testing.expect(policy.completed(&s, true, true, true, false));
    try std.testing.expect(!policy.completed(&s, false, true, true, false));
    try std.testing.expect(!policy.completed(&s, true, false, true, false));
    try std.testing.expect(!policy.completed(&s, true, true, false, false));
    try std.testing.expect(!policy.completed(&s, true, true, true, true));
    for (0..9) |case| {
        s = .{};
        switch (case) {
            0 => s.finished = false,
            1 => s.reason = .cancelled,
            2 => s.reason = .none,
            3 => s.cancelled = true,
            4 => s.cached = 1,
            5 => s.reuse_failed = true,
            6 => s.reuse.at = 1,
            7 => s.reuse.marks = &.{1},
            8 => s.reuse.hook = 1,
            else => unreachable,
        }
        try std.testing.expect(!policy.completed(&s, true, true, true, false));
    }
    s = .{};
    var sentinel: u8 = 0;
    s.reuse.saved = &sentinel;
    try std.testing.expect(!policy.completed(&s, true, true, true, false));
    s = .{ .reason = .stop };
    try std.testing.expect(policy.completed(&s, true, true, true, false));
}

test "independent pool flag requires the exact cache and remains strict" {
    try std.testing.expect(!try policy.compatible(null, false));
    try std.testing.expect(!try policy.compatible("0", true));
    try std.testing.expect(try policy.compatible("1", true));
    try std.testing.expectError(error.SeqReuseNeedsExactCache, policy.compatible("1", false));
    try std.testing.expectError(error.InvalidSeqReuse, policy.compatible("yes", true));
}
