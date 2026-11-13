//! The pool's moves applied to bytes as the device side applies them: live rows survive, a fresh run reads zeros.
const std = @import("std");
const pool = @import("pool.zig");
const Bufs = @import("pool_bufs.zig").Bufs;

const t = std.testing;
const B = pool.block;
/// Bytes a block takes in the simulated buffer.
const bpb = 3;

/// A pool and one buffer of `bpb` bytes a block, with each live seat's written blocks holding its tag.
const Sim = struct {
    p: pool.Pool,
    bytes: []u8,
    written: [8]u32 = @splat(0),

    fn init(seats: u32, blocks: u32) !Sim {
        const bytes = try t.allocator.alloc(u8, blocks * bpb);
        @memset(bytes, 0);
        return .{ .p = try pool.Pool.init(t.allocator, seats, blocks), .bytes = bytes };
    }

    fn deinit(s: *Sim) void {
        s.p.deinit();
        t.allocator.free(s.bytes);
    }

    fn span(first: u32, blocks: u32) [2]usize {
        return .{ first * bpb, blocks * bpb };
    }

    /// `take`, its moves and zeroes applied; the new run must read zeros; it then writes its first `w` blocks.
    fn take(s: *Sim, seat: u32, tokens: u64, w: u32) !pool.Plan {
        const plan = try s.p.take(seat, tokens);
        for (plan.moves) |m| {
            try t.expect(m.to < m.from);
            Bufs.move(s.bytes, span(m.from, m.blocks), span(m.to, m.blocks));
        }
        for (plan.zero) |z| Bufs.zero(s.bytes, span(z.start, z.blocks), .{ 0, 0 });
        const r = plan.run;
        for (s.bytes[r.start * bpb ..][0 .. r.blocks * bpb]) |b| try t.expectEqual(@as(u8, 0), b);
        const n = @min(w, r.blocks);
        @memset(s.bytes[r.start * bpb ..][0 .. n * bpb], tag(seat));
        s.written[seat] = n;
        return plan;
    }

    /// The seat's run goes back, its writes counted from its run's first block wherever the run now is.
    fn give(s: *Sim, seat: u32) void {
        const n = s.written[seat];
        s.p.give(seat, if (n == 0) 0 else @as(u64, n) * B - pool.slack);
        s.written[seat] = 0;
    }

    /// Every live seat's written blocks still hold its tag, and its unwritten ones zeros.
    fn check(s: *const Sim) !void {
        for (s.p.runs, 0..) |r, seat| if (r) |x| {
            for (0..x.blocks) |b| {
                const want: u8 = if (b < s.written[seat]) tag(@intCast(seat)) else 0;
                for (s.bytes[(x.start + b) * bpb ..][0..bpb]) |v| try t.expectEqual(want, v);
            }
        };
    }

    fn tag(seat: u32) u8 {
        return @intCast(seat + 1);
    }
};

test "pool moves: a run that overlaps its own destination slides down intact" {
    var s = try Sim.init(3, 12);
    defer s.deinit();
    _ = try s.take(0, pool.tokensOf(2), 2); // [0, 2)
    _ = try s.take(1, pool.tokensOf(6), 5); // [2, 8), five blocks written
    _ = try s.take(2, pool.tokensOf(2), 1); // [8, 10)
    s.give(0);
    // [0, 2) and [10, 12) free: seat 1 moves [2, 8) to [0, 6), overlapping itself, then seat 2 to [6, 8)
    const plan = try s.take(0, pool.tokensOf(3), 3);
    try t.expectEqual(@as(usize, 2), plan.moves.len);
    try t.expectEqual(pool.Move{ .seat = 1, .from = 2, .to = 0, .blocks = 6 }, plan.moves[0]);
    try t.expectEqual(pool.Run{ .start = 8, .blocks = 3 }, plan.run);
    try s.check();
}

test "pool moves: a moved seat's writes are counted from its new place when it goes back" {
    var s = try Sim.init(3, 10);
    defer s.deinit();
    _ = try s.take(0, pool.tokensOf(4), 4); // [0, 4)
    _ = try s.take(1, pool.tokensOf(4), 2); // [4, 8), two blocks written
    s.give(0);
    _ = try s.take(2, pool.tokensOf(5), 0); // seat 1 slides to [0, 4), the new run takes [4, 9)
    try t.expectEqual(@as(u32, 0), s.p.runs[1].?.start);
    s.give(2);
    s.give(1); // its writes reach blocks 0, 1 (and 2 for the compressed row past them): dirty there, not at [4, 6)
    try t.expect(s.p.dirty[0] and s.p.dirty[1] and s.p.dirty[2]);
    // [4, 8) was vacated by the move or written by seat 0: dirty; the next run reads zeros wherever it lands
    _ = try s.take(0, pool.tokensOf(10), 0);
}

test "pool moves: a refused take changes no run and no dirty block" {
    var s = try Sim.init(3, 8);
    defer s.deinit();
    _ = try s.take(0, pool.tokensOf(3), 3);
    _ = try s.take(1, pool.tokensOf(3), 1);
    s.give(0);
    const runs = s.p.runs[0..3].*;
    var dirty: [8]bool = undefined;
    @memcpy(&dirty, s.p.dirty);
    try t.expectError(error.LaneMemory, s.p.take(2, pool.tokensOf(6)));
    try t.expectError(error.PastPool, s.p.take(2, pool.tokensOf(8) + 1));
    try t.expectEqualSlices(?pool.Run, &runs, s.p.runs);
    try t.expectEqualSlices(bool, &dirty, s.p.dirty);
    try s.check();
}

test "pool moves: random takes and gives keep every live seat's bytes and give every new run zeros" {
    var s = try Sim.init(6, 48);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    var moves: usize = 0;
    for (0..4000) |_| {
        const seat = r.uintLessThan(u32, 6);
        if (s.p.runs[seat] != null) s.give(seat) else {
            const want = 1 + r.uintLessThan(u64, pool.tokensOf(20));
            if (s.take(seat, want, r.uintLessThan(u32, 21))) |plan| {
                moves += plan.moves.len;
            } else |e| try t.expectEqual(error.LaneMemory, e);
        }
        try s.check();
    }
    try t.expect(moves > 0); // the pattern slid runs down, not only first fits
}
