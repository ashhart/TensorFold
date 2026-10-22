//! Prompt-cache marks are relative rows in one completed prompt call, with four physical destinations per layer.
const std = @import("std");
pub const limit = 4;
pub const Selection = struct { end: u32, rows: [limit]u32 = @splat(0), count: usize = 0 };

/// Keep four interior marks; a fifth becomes the chunk end and uses the ordinary completed-state save.
pub fn select(start: u32, end: u32, marks: []const u32) !Selection {
    if (end <= start) return error.BadPromptMarks;
    var next: [limit + 1]u32 = @splat(std.math.maxInt(u32));
    var count: usize = 0;
    for (marks) |mark| {
        if (mark <= start or mark >= end) continue;
        var at: usize = 0;
        while (at < count and next[at] < mark) : (at += 1) {}
        if ((at < count and next[at] == mark) or at == next.len) continue;
        const last = @min(count, next.len - 1);
        var i = last;
        while (i > at) : (i -= 1) next[i] = next[i - 1];
        next[at] = mark;
        count = @min(count + 1, next.len);
    }
    var out = Selection{ .end = if (count > limit) next[limit] else end, .count = @min(count, limit) };
    for (0..out.count) |i| out.rows[i] = next[i] - start;
    return out;
}

pub const Layout = struct { state: usize, conv: usize, total: usize };
pub fn layout(layers: usize, state_words: usize, conv_words: usize) !Layout {
    if (layers == 0 or state_words == 0 or conv_words == 0) return error.BadPromptMarks;
    const copies = try std.math.mul(usize, layers, limit);
    const state = try std.math.mul(usize, try std.math.mul(usize, state_words, 4), copies);
    const conv = try std.math.mul(usize, try std.math.mul(usize, conv_words, 2), copies);
    return .{ .state = state, .conv = conv, .total = try std.math.add(usize, 16, try std.math.add(usize, state, conv)) };
}

pub const Position = struct {
    slot: u32,
    start: u32,
    end: u32,
    rows: [limit]u32,
    ready: bool = false,

    pub fn init(slot: u32, start: u32, end: u32, rows: [limit]u32) !Position {
        if (end <= start) return error.BadPromptMarks;
        var previous: u32 = 0;
        var unused = false;
        for (rows) |row| {
            if (row == 0) {
                unused = true;
                continue;
            }
            if (unused or row <= previous or row > end - start) return error.BadPromptMarks;
            previous = row;
        }
        if (previous == 0) return error.BadPromptMarks;
        return .{ .slot = slot, .start = start, .end = end, .rows = rows };
    }

    pub fn index(p: Position, slot: u32, at: u32, stands: u32) ?usize {
        if (!p.ready or slot != p.slot or stands != p.end or at <= p.start or at > p.end) return null;
        for (p.rows, 0..) |row, i| if (row != 0 and at - p.start == row) return i;
        return null;
    }
};

test "four interior destinations and a fifth endpoint survive unordered, duplicate and outside marks" {
    const s = try select(10, 40, &.{ 39, 11, 0, 13, 12, 13, 18, 19, 41, 10, 40 });
    try std.testing.expectEqual(@as(u32, 19), s.end);
    try std.testing.expectEqual([limit]u32{ 1, 2, 3, 8 }, s.rows);
    try std.testing.expectEqual(@as(usize, 4), s.count);
    const empty = try select(10, 40, &.{ 0, 10, 40, 41 });
    try std.testing.expectEqual(@as(usize, 0), empty.count);
    try std.testing.expectEqual(@as(u32, 40), empty.end);
}

test "uncompleted or reused mark storage cannot publish a different position or slot" {
    var p = try Position.init(3, 10, 40, .{ 1, 2, 20, 0 });
    try std.testing.expectEqual(@as(?usize, null), p.index(3, 11, 40));
    p.ready = true;
    try std.testing.expectEqual(@as(?usize, 0), p.index(3, 11, 40));
    try std.testing.expectEqual(@as(?usize, null), p.index(2, 11, 40));
    try std.testing.expectEqual(@as(?usize, null), p.index(3, 11, 41));
    try std.testing.expectEqual(@as(?usize, null), p.index(3, 13, 40));
    try std.testing.expectError(error.BadPromptMarks, Position.init(0, 10, 40, .{ 2, 0, 3, 0 }));
    try std.testing.expectError(error.BadPromptMarks, Position.init(0, 10, 40, .{ 2, 2, 0, 0 }));
}

test "all four physical slots are charged even when just one mark is used, and products cannot wrap" {
    const l = try layout(3, 8, 6);
    try std.testing.expectEqual(@as(usize, 16 + 12 * (32 + 12)), l.total);
    try std.testing.expectError(error.Overflow, layout(std.math.maxInt(usize), 8, 6));
}
