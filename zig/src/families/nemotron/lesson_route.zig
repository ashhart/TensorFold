//! How a pass's rows meet the lessons, for both backends' gate kernels (train.metal's and train.cu's Route): rows before
//! a stream's decision row carry no lesson, the decision row votes, and rows after it carry the lessons that won there.
const std = @import("std");

pub const max_slots = 64; // latch slots: a stream's replay id (a sequence's on CUDA), the trainer's the last
pub const max_blocks = @import("learn_shape.zig").max_blocks;
pub const trainer_slot = max_slots - 1;
pub const max_segments = 8;

/// A stream without a decision row: none of its rows carry a lesson.
pub const none: i64 = std.math.maxInt(i32);

/// No segments: no row carries a lesson.
pub const off: Route = .{};

/// A training example's decision row, and the lessons (a bit a block) acting on the rows after it.
pub const Turn = struct { decide: u32, bits: u32 = 0 };

/// A training pass over one example's `rows` rows: its latch is set before the pass (the trainer's slot).
pub fn trained(rows: usize, turn: Turn) Route {
    var r: Route = .{ .known = 1 };
    r.add(0, rows, 0, turn.decide, trainer_slot);
    return r;
}

pub const Route = extern struct {
    count: u32 = 0,
    known: u32 = 0, // 1: every latch was set before the pass, so rows after a decision row in it may act (training)
    pad: [2]u32 = .{ 0, 0 },
    seg: [max_segments][4]i32 = std.mem.zeroes([max_segments][4]i32),

    /// The pass's rows `first`..`first + rows`, a stream's rows from position `pos`, its decision row at `decide`.
    pub fn add(r: *Route, first: usize, rows: usize, pos: usize, decide: i64, slot: u32) void {
        std.debug.assert(r.count < max_segments);
        const at: i64 = if (decide == none) none else @as(i64, @intCast(first)) + decide - @as(i64, @intCast(pos));
        r.seg[r.count] = .{ @intCast(first), @intCast(rows), @intCast(std.math.clamp(at, -1, none)), @intCast(slot) };
        r.count += 1;
    }

    /// The pass's rows `first`..`first + rows`, a stream past its decision row: its latch alone decides.
    pub fn later(r: *Route, first: usize, rows: usize, slot: u32) void {
        r.add(first, rows, 1, 0, slot);
    }

    /// Segment s's decision row, when it lies in this pass (then the stream's latch is set after it).
    pub fn decision(r: *const Route, s: usize) ?usize {
        const g = r.seg[s];
        return if (g[2] >= g[0] and g[2] < g[0] + g[1]) @intCast(g[2]) else null;
    }
};

test "a stream's rows carry a lesson only after its decision row" {
    var r: Route = .{};
    r.add(0, 10, 100, 104, 3); // decision at position 104: pass row 4
    r.add(10, 2, 200, 150, 5); // decided in an earlier pass
    r.add(12, 3, 7, none, 6);
    try std.testing.expectEqual(@as(?usize, 4), r.decision(0));
    try std.testing.expectEqual(@as(?usize, null), r.decision(1));
    try std.testing.expect(r.seg[1][2] < r.seg[1][0]);
    try std.testing.expect(r.seg[2][2] >= r.seg[2][0] + r.seg[2][1]);
}
