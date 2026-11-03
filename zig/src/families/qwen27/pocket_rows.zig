//! Taught row runs for a shared pass; untaught streams have no pocket dispatch.
const std = @import("std");

pub const Selection = struct {
    pub const Range = struct { first: usize, count: usize };
    ranges: [128]Range = undefined,
    count: usize = 0,
    all: bool = true,

    pub fn windows(slots: []const u32, firsts: []const u32, spans: anytype, untaught: []const bool) !Selection {
        if (slots.len != firsts.len or slots.len != spans.len) return error.BadPocketRows;
        var selected: Selection = .{ .all = false };
        for (slots, firsts, spans) |slot, first, span| {
            if (slot >= untaught.len) return error.BadPocketRows;
            if (!untaught[slot]) try selected.add(first, span.parents.len);
        }
        return selected;
    }

    pub fn add(s: *Selection, first: usize, count: usize) !void {
        if (count == 0) return;
        _ = std.math.add(usize, first, count) catch return error.BadPocketRows;
        if (s.count > 0) {
            const last = &s.ranges[s.count - 1];
            if (last.first + last.count == first) {
                last.count += count;
                return;
            }
            if (last.first + last.count > first) return error.BadPocketRows;
        }
        if (s.count == s.ranges.len) return error.BadPocketRows;
        s.ranges[s.count] = .{ .first = first, .count = count };
        s.count += 1;
    }
};

test "mixed taught and untaught windows dispatch only taught runs, including a 2048-row prompt" {
    var s: Selection = .{ .all = false };
    try s.add(0, 3);
    try s.add(5, 2);
    try s.add(7, 4);
    try std.testing.expectEqual(@as(usize, 2), s.count);
    try std.testing.expectEqual(Selection.Range{ .first = 0, .count = 3 }, s.ranges[0]);
    try std.testing.expectEqual(Selection.Range{ .first = 5, .count = 6 }, s.ranges[1]);
    s = .{ .all = false };
    try std.testing.expectEqual(@as(usize, 0), s.count);
    try s.add(0, 2048);
    try std.testing.expectEqual(@as(usize, 2048), s.ranges[0].count);
    try std.testing.expectError(error.BadPocketRows, s.add(2047, 1));
}
