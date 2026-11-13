//! The shared pool's allocator on the host: fit, wait, refuse, reuse, sliding runs down, dirty blocks and the setting.
const std = @import("std");
const pool = @import("pool.zig");

const t = std.testing;
const B = pool.block;

fn tokens(blocks: u32) u64 {
    return pool.tokensOf(blocks);
}

test "pool: a run of n blocks holds n blocks less the slack, and blocksFor is its inverse" {
    try t.expectEqual(@as(u64, 0), pool.tokensOf(0));
    try t.expectEqual(@as(u64, B - pool.slack), pool.tokensOf(1));
    for ([_]u64{ 1, B - 2, B - 1, B, 4096, 262_144, 1_048_576 }) |n| {
        const b: u32 = @intCast(pool.blocksFor(n));
        try t.expect(pool.tokensOf(b) >= n);
        try t.expect(b == 1 or pool.tokensOf(b - 1) < n);
    }
    // a pool of 1,048,576 positions holds one 1,048,576-position request
    try t.expect(pool.tokensOf(@intCast(pool.blocksFor(1_048_576))) >= 1_048_576);
}

test "pool: four seats share the pool; a fifth run waits until one goes back, and reuses its blocks" {
    var p = try pool.Pool.init(t.allocator, 5, 16);
    defer p.deinit();
    for (0..4) |s| {
        const plan = try p.take(@intCast(s), tokens(4));
        try t.expectEqual(@as(u32, @intCast(4 * s)), plan.run.start);
        try t.expectEqual(@as(usize, 0), plan.moves.len);
        try t.expectEqual(@as(usize, 0), plan.zero.len); // a new pool is zeroed
    }
    try t.expectEqual(@as(u32, 0), p.free());
    try t.expectError(error.LaneMemory, p.take(4, 1));
    p.give(1, tokens(4));
    const plan = try p.take(4, tokens(4));
    try t.expectEqual(@as(u32, 4), plan.run.start);
    try t.expectEqual(@as(usize, 1), plan.zero.len); // the old request's rows are zeroed for the new one
    try t.expectEqual(pool.Run{ .start = 4, .blocks = 4 }, plan.zero[0]);
}

test "pool: a request larger than the pool is refused, one as large as it fits alone" {
    var p = try pool.Pool.init(t.allocator, 2, 8);
    defer p.deinit();
    try t.expectError(error.PastPool, p.take(0, tokens(8) + 1));
    try t.expectError(error.PastPool, p.take(0, 0));
    _ = try p.take(0, tokens(8));
    try t.expectError(error.LaneMemory, p.take(1, 1));
    try t.expectError(error.SeatHeld, p.take(0, 1));
    try t.expectError(error.SeatHeld, p.take(2, 1));
    p.give(0, 0);
    _ = try p.take(1, tokens(8));
}

test "pool: a long request ends, short ones take its blocks; the long one's turn comes when they end" {
    var p = try pool.Pool.init(t.allocator, 4, 16);
    defer p.deinit();
    _ = try p.take(0, tokens(14));
    _ = try p.take(1, tokens(2));
    try t.expectError(error.LaneMemory, p.take(2, tokens(3)));
    p.give(0, tokens(14));
    for ([_]u32{ 2, 3 }) |s| _ = try p.take(s, tokens(3));
    try t.expectError(error.LaneMemory, p.take(0, tokens(14)));
    for ([_]u32{ 1, 2, 3 }) |s| p.give(s, 5);
    const plan = try p.take(0, tokens(16));
    try t.expectEqual(@as(u32, 0), plan.run.start);
}

test "pool: when the free blocks are split, live runs slide down in start order and the run lands on top" {
    var p = try pool.Pool.init(t.allocator, 4, 12);
    defer p.deinit();
    _ = try p.take(0, tokens(3)); // [0, 3)
    _ = try p.take(1, tokens(3)); // [3, 6)
    _ = try p.take(2, tokens(3)); // [6, 9)
    _ = try p.take(3, tokens(3)); // [9, 12)
    p.give(0, tokens(3));
    p.give(2, tokens(3));
    // six free blocks, no run of four: seats 1 and 3 move to [0, 3) and [3, 6), the new run takes [6, 10)
    const plan = try p.take(0, tokens(4));
    try t.expectEqual(@as(usize, 2), plan.moves.len);
    try t.expectEqual(pool.Move{ .seat = 1, .from = 3, .to = 0, .blocks = 3 }, plan.moves[0]);
    try t.expectEqual(pool.Move{ .seat = 3, .from = 9, .to = 3, .blocks = 3 }, plan.moves[1]);
    try t.expectEqual(pool.Run{ .start = 6, .blocks = 4 }, plan.run);
    try t.expectEqual(pool.Run{ .start = 0, .blocks = 3 }, p.runs[1].?);
    try t.expectEqual(pool.Run{ .start = 3, .blocks = 3 }, p.runs[3].?);
    // every block the new run takes was written by an earlier request or vacated by a move: all zeroed
    try t.expectEqual(@as(usize, 1), plan.zero.len);
    try t.expectEqual(pool.Run{ .start = 6, .blocks = 4 }, plan.zero[0]);
    // the moves copy downwards in order: no move's destination overlaps a later move's source
    for (plan.moves, 0..) |m, i| for (plan.moves[i + 1 ..]) |later| try t.expect(m.to + m.blocks <= later.from);
    try t.expectEqual(@as(u32, 2), p.free());
}

test "pool: only blocks a request wrote are zeroed for the next, and a block is zeroed once" {
    var p = try pool.Pool.init(t.allocator, 2, 8);
    defer p.deinit();
    _ = try p.take(0, tokens(8));
    p.give(0, B + 10); // positions in blocks 0 and 1 (and the compressed row past them)
    var plan = try p.take(1, tokens(3));
    try t.expectEqual(@as(usize, 1), plan.zero.len);
    try t.expectEqual(pool.Run{ .start = 0, .blocks = 2 }, plan.zero[0]);
    p.give(1, 0); // wrote nothing: still clean
    plan = try p.take(0, tokens(8));
    try t.expectEqual(@as(usize, 0), plan.zero.len);
    p.give(0, 1);
    try t.expect(p.dirty[0] and !p.dirty[1]);
    p.give(0, 0); // no run: nothing happens
}

test "pool: a last position on a block's edge marks the block holding its compressed row" {
    var p = try pool.Pool.init(t.allocator, 1, 4);
    defer p.deinit();
    _ = try p.take(0, tokens(4));
    p.give(0, B - pool.slack); // the ratio-1 row B - 2 and the ratio-2 row (B - 2) / 2 are in block 0; slack may reach 1
    try t.expect(p.dirty[0] and p.dirty[1] and !p.dirty[2]);
}

test "pool: an interleaved pattern of takes and gives never loses a block and never overlaps two runs" {
    var p = try pool.Pool.init(t.allocator, 4, 64);
    defer p.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (0..2000) |_| {
        const s = r.uintLessThan(u32, 4);
        if (p.runs[s] != null) {
            p.give(s, r.uintLessThan(u64, tokens(p.runs[s].?.blocks) + 1));
        } else {
            const want = 1 + r.uintLessThan(u64, tokens(40));
            const before = p.free();
            if (p.take(s, want)) |plan| {
                try t.expect(pool.tokensOf(plan.run.blocks) >= want);
                try t.expectEqual(before - plan.run.blocks, p.free());
                for (plan.zero) |z| try t.expect(z.start >= plan.run.start and z.end() <= plan.run.end());
            } else |e| {
                try t.expectEqual(error.LaneMemory, e);
                try t.expect(before < pool.blocksFor(want)); // it waits only when too few blocks are free in all
            }
        }
        for (p.runs, 0..) |a, i| if (a) |x| for (p.runs[i + 1 ..]) |b| if (b) |y| try t.expect(x.end() <= y.start or y.end() <= x.start);
    }
}

test "pool: the setting is refused by name in words, unset keeps a window a seat, a small pool cuts the window" {
    var buf: [256]u8 = undefined;
    var why: []const u8 = "";
    try t.expectEqual(@as(?pool.Setting, null), try pool.setting(null, 4, 1_048_576, &why, &buf));
    try t.expectError(error.BadPoolTokens, pool.setting(0, 4, 1_048_576, &why, &buf));
    try t.expect(std.mem.indexOf(u8, why, pool.name) != null and std.mem.indexOf(u8, why, "less than one block") != null);
    try t.expectError(error.BadPoolTokens, pool.setting(255, 4, 1_048_576, &why, &buf));
    try t.expectError(error.BadPoolTokens, pool.setting(4 * 262_144 + 1, 4, 262_144, &why, &buf));
    try t.expect(std.mem.indexOf(u8, why, "--parallel 4 x --context 262144") != null);
    const s = (try pool.setting(1_048_576, 4, 1_048_576, &why, &buf)).?;
    try t.expectEqual(@as(u32, 1_048_576), s.window);
    try t.expect(pool.tokensOf(s.blocks) >= 1_048_576);
    const small = (try pool.setting(500_000, 4, 1_048_576, &why, &buf)).?;
    try t.expectEqual(@as(u32, 500_000), small.window);
    const exact = (try pool.setting(4 * 262_144, 4, 262_144, &why, &buf)).?;
    try t.expectEqual(@as(u32, 262_144), exact.window);
}

test "pool: init refuses an empty pool and frees what it made" {
    try t.expectError(error.EmptyPool, pool.Pool.init(t.allocator, 0, 4));
    try t.expectError(error.EmptyPool, pool.Pool.init(t.allocator, 2, 0));
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 3 });
    try t.expectError(error.OutOfMemory, pool.Pool.init(fail.allocator(), 2, 4)); // the leak check sees the errdefers
}
