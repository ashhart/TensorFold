//! Real page owners under the production growth transaction with an in-process rank agreement.

const std = @import("std");
const t = std.testing;
const sessions = @import("kv").sessions;
const growth = @import("pool_growth.zig");
const fams = [_]sessions.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 1 }};

const Pair = struct {
    votes: [2]std.atomic.Value(u8) = .{ .init(0), .init(0) },
    const Rank = struct {
        pair: *Pair,
        me: usize,
        slot: *sessions.Slot,
        err: ?anyerror = null,
    };
    fn all(ptr: *anyopaque, ok: bool) !bool {
        const r: *Rank = @ptrCast(@alignCast(ptr));
        r.pair.votes[r.me].store(if (ok) 1 else 2, .release);
        var spins: u64 = 0;
        while (r.pair.votes[1 - r.me].load(.acquire) == 0) : (spins += 1) {
            if (spins > 1 << 32) return error.Timeout;
            std.atomic.spinLoopHint();
        }
        return ok and r.pair.votes[1 - r.me].load(.acquire) == 1;
    }
    fn run(r: *Rank) void {
        growth.reserve(.{ .ctx = r, .all = all }, &.{.{ .slot = r.slot, .end = 256 }}) catch |err| {
            r.err = err;
        };
    }
};

test "pool growth: one rank out of pages rolls back both ranks before device table upload, then both retry" {
    var pools: [2]sessions.Pool = undefined;
    for (&pools, 0..) |*p, rank| p.* = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 2, @intCast(rank));
    defer for (&pools) |*p| p.deinit();
    const s0 = try pools[0].newSlot(512);
    const s1 = try pools[1].newSlot(512);
    const occupied = try pools[1].newSlot(512);
    try occupied.ensure(384);
    var pair: Pair = .{};
    var ranks = [_]Pair.Rank{ .{ .pair = &pair, .me = 0, .slot = s0 }, .{ .pair = &pair, .me = 1, .slot = s1 } };
    const a = try std.Thread.spawn(.{}, Pair.run, .{&ranks[0]});
    const b = try std.Thread.spawn(.{}, Pair.run, .{&ranks[1]});
    a.join();
    b.join();
    for (ranks) |r| {
        try t.expectEqual(@as(?anyerror, error.PoolExhausted), r.err);
        try t.expectEqual(@as(u32, 0), r.slot.len);
        try t.expectEqual(@as(u32, 0), r.slot.over);
    }
    try t.expectEqual(@as(u32, 32), pools[0].freePages());
    try t.expectEqual(@as(u32, 8), pools[1].freePages());
    _ = try occupied.releaseAll();
    pair = .{};
    for (&ranks) |*r| r.err = null;
    const c = try std.Thread.spawn(.{}, Pair.run, .{&ranks[0]});
    const d = try std.Thread.spawn(.{}, Pair.run, .{&ranks[1]});
    c.join();
    d.join();
    for (ranks) |r| {
        try t.expect(r.err == null);
        try t.expectEqual(@as(u32, 16), r.slot.len);
        _ = try r.slot.releaseAll();
    }
}

test "pool growth: batch refusal restores earlier tentative maps and preserves partial prefill ownership" {
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    const first = try pool.newSlot(512);
    const second = try pool.newSlot(512);
    try first.ensure(64);
    try second.ensure(64);
    const before = pool.freePages();
    const needs = [_]growth.Need{ .{ .slot = first, .end = 256 }, .{ .slot = second, .end = 512 } };
    try t.expectError(error.PoolExhausted, growth.reserve(null, &needs));
    try t.expectEqual(before, pool.freePages());
    try t.expectEqual(@as(u32, 4), first.len);
    try t.expectEqual(@as(u32, 4), second.len);
    _ = try first.releaseAll();
    _ = try second.releaseAll();
    try growth.reserve(null, &.{.{ .slot = first, .end = 512 }});
    _ = try first.releaseAll();
    try t.expectEqual(pool.npages, pool.available());
}

test "pool growth: a failed transport agreement propagates as fatal and retains the tentative owner" {
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    const slot = try pool.newSlot(512);
    const Fail = struct {
        fn all(_: *anyopaque, _: bool) anyerror!bool {
            return error.PeerFailed;
        }
    };
    var context: u8 = 0;
    try t.expectError(error.PeerFailed, growth.reserve(.{ .ctx = &context, .all = Fail.all }, &.{.{ .slot = slot, .end = 128 }}));
    try t.expectEqual(@as(u32, 8), slot.len);
    _ = try slot.releaseAll();
}

test "pool growth: an unadmitted caller cannot consume another active request's reservation" {
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    const active = try pool.newSlot(512);
    const other = try pool.newSlot(512);
    active.reserve(24);
    try t.expectError(error.PoolExhausted, growth.reserve(null, &.{.{ .slot = other, .end = 256 }}));
    try t.expectEqual(@as(u32, 0), other.len);
    try t.expect(other.quota == null);
    try t.expectEqual(@as(u32, 8), pool.available());
    try growth.reserve(null, &.{.{ .slot = active, .end = 384 }});
    _ = try active.releaseAll();
    try growth.reserve(null, &.{.{ .slot = other, .end = 512 }});
    _ = try other.releaseAll();
}

test "pool growth: a need within mapped pages skips the rank vote and leaves ownership, quota and overage as they were" {
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 2, 0);
    defer pool.deinit();
    const slot = try pool.newSlot(512);
    try slot.ensure(100);
    const Count = struct {
        n: u32 = 0,
        fn all(ptr: *anyopaque, ok: bool) anyerror!bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            c.n += 1;
            return ok;
        }
    };
    var votes: Count = .{};
    const agree: @import("lockstep.zig").Agree = .{ .ctx = &votes, .all = Count.all };
    const free = pool.freePages();
    for ([_]u64{ 1, 100, 112 }) |end| try growth.reserve(agree, &.{.{ .slot = slot, .end = end }});
    try t.expectEqual(@as(u32, 0), votes.n);
    try t.expectEqual(@as(u32, 7), slot.len);
    try t.expect(slot.quota == null);
    try t.expectEqual(@as(u32, 0), slot.over);
    try t.expectEqual(free, pool.freePages());
    try growth.reserve(agree, &.{.{ .slot = slot, .end = 113 }});
    try t.expectEqual(@as(u32, 1), votes.n);
    try t.expectEqual(@as(u32, 8), slot.len);
    _ = try slot.releaseAll();
}
