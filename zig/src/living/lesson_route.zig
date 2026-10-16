//! How rows meet lessons in both gate kernels: the decision row scores each router, later rows carry those that clear.
const std = @import("std");

pub const max_slots = 64; // latch slots: a stream's replay id (a sequence's on CUDA), the trainer's the last
pub const max_blocks = @import("learn_shape.zig").max_blocks;
pub const trainer_slot = max_slots - 1;
pub const max_segments = 8;

/// A stream without a decision row: none of its rows carry a lesson.
pub const none: i64 = std.math.maxInt(i32);

/// No segments: no row carries a lesson.
pub const off: Route = .{};

/// Which lessons act on a reply, a bit a block.
pub const Acting = std.bit_set.Integer(max_blocks);

/// A training example's decision row and the lessons acting after it (`whole`: on every row, as weight edits do).
pub const Turn = struct { decide: u32, bits: Acting = .empty, whole: bool = false };

/// A slot's latch flags (`latch`: u32 [max_slots, max_blocks]) set to `bits`, as a training pass reads them.
pub fn flag(latch: []u32, slot: u32, bits: Acting) void {
    for (latch[slot * max_blocks ..][0..max_blocks], 0..) |*f, b| f.* = @intFromBool(bits.isSet(b));
}

/// A training pass over one example's `rows` rows: its latch is set before the pass (the trainer's slot).
pub fn trained(rows: usize, turn: Turn) Route {
    var r: Route = .{ .known = 1, .whole = @intFromBool(turn.whole) };
    r.add(0, rows, 0, turn.decide, trainer_slot);
    return r;
}

pub const Route = extern struct {
    count: u32 = 0,
    known: u32 = 0, // 1: every latch was set before the pass, so rows after a decision row in it may act (training)
    whole: u32 = 0, // 1: lessons edit the weights and act on every row (a training pass's flagged ones, else all)
    pad: u32 = 0,
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

/// A pass's rows the pocket acts on, segment by segment in pass order: runs of rows between untaught streams' rows.
pub const Runs = struct {
    first: usize = 0, // the open run's first row
    rows: usize = 0, // and its rows so far

    /// The run a segment of `rows` rows closes, if any: an untaught segment ends the open run and is left out.
    pub fn next(r: *Runs, rows: usize, untaught: bool) ?[2]usize {
        if (!untaught) {
            r.rows += rows;
            return null;
        }
        const closed = r.last();
        r.* = .{ .first = r.first + r.rows + rows };
        return closed;
    }

    /// The open run, once every segment is in.
    pub fn last(r: *const Runs) ?[2]usize {
        return if (r.rows > 0) .{ r.first, r.rows } else null;
    }
};

test "the pocket's runs leave out the rows of streams that read the model as shipped" {
    var r: Runs = .{};
    try std.testing.expectEqual(@as(?[2]usize, null), r.next(3, false));
    try std.testing.expectEqual(@as(?[2]usize, .{ 0, 3 }), r.next(2, true));
    try std.testing.expectEqual(@as(?[2]usize, null), r.next(4, true));
    try std.testing.expectEqual(@as(?[2]usize, null), r.next(1, false));
    try std.testing.expectEqual(@as(?[2]usize, null), r.next(2, false));
    try std.testing.expectEqual(@as(?[2]usize, .{ 9, 3 }), r.last());
    var all: Runs = .{};
    try std.testing.expectEqual(@as(?[2]usize, null), all.next(5, true));
    try std.testing.expectEqual(@as(?[2]usize, null), all.last());
}

/// The kept lessons (blocks below `blocks`) whose router scores reach what they need.
pub fn latched(score: []const f32, need: []const f32, blocks: usize) Acting {
    var bits: Acting = .empty;
    for (0..blocks) |b| if (need[b] > 0 and score[b] >= need[b]) bits.set(b);
    return bits;
}

test "a kept lesson acts on a reply only once its router score reaches what it needs" {
    var score: [max_blocks]f32 = @splat(0);
    var need: [max_blocks]f32 = @splat(0);
    score[0..4].* = .{ 2.5, 1.2, 0, 3 };
    need[0..2].* = .{ 2, 1.5 };
    const four = latched(&score, &need, 4);
    try std.testing.expect(four.isSet(0) and four.count() == 1);
    try std.testing.expectEqual(@as(usize, 0), latched(&score, &need, 0).count());
}
