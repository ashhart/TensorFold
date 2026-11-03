//! The native dense branch has one optional owner; untaught row runs never enter the pocket kernel.
const std = @import("std");
pub const Run = struct { first: usize, rows: usize, untaught: bool = false };
pub const Range = struct { first: usize, rows: usize };
pub const Selection = struct {
    ranges: [128]Range = undefined,
    count: usize = 0,

    pub fn of(runs: []const Run, rows: usize) !Selection {
        var out: Selection = .{};
        var end: usize = 0;
        for (runs) |run| {
            if (run.rows == 0 or run.first != end or run.rows > rows -| end) return error.Qwen4LearningRows;
            end = try std.math.add(usize, end, run.rows);
            if (run.untaught) continue;
            if (out.count > 0 and out.ranges[out.count - 1].first + out.ranges[out.count - 1].rows == run.first) {
                out.ranges[out.count - 1].rows += run.rows;
            } else {
                if (out.count == out.ranges.len) return error.Qwen4LearningRows;
                out.ranges[out.count] = .{ .first = run.first, .rows = run.rows };
                out.count += 1;
            }
        }
        if (end != rows) return error.Qwen4LearningRows;
        return out;
    }
};

pub fn Hook(comptime G: type) type {
    const Tensor = @import("arena.zig").Arena(G.Storage).Tensor;
    return struct {
        ctx: *anyopaque,
        begin: *const fn (*anyopaque, []const Run, usize) anyerror!void,
        dense: *const fn (*anyopaque, *G, usize, Tensor, Tensor) anyerror!void,
    };
}

test "Qwen4 learning: mixed windows dispatch taught runs only and all-untaught does no work" {
    const got = try Selection.of(&.{ .{ .first = 0, .rows = 3 }, .{ .first = 3, .rows = 2, .untaught = true }, .{ .first = 5, .rows = 1 }, .{ .first = 6, .rows = 4 } }, 10);
    try std.testing.expectEqual(@as(usize, 2), got.count);
    try std.testing.expectEqualSlices(Range, &.{ .{ .first = 0, .rows = 3 }, .{ .first = 5, .rows = 5 } }, got.ranges[0..got.count]);
    const empty = try Selection.of(&.{.{ .first = 0, .rows = 16, .untaught = true }}, 16);
    try std.testing.expectEqual(@as(usize, 0), empty.count);
    try std.testing.expectError(error.Qwen4LearningRows, Selection.of(&.{.{ .first = 1, .rows = 1 }}, 2));
    try std.testing.expectError(error.Qwen4LearningRows, Selection.of(&.{.{ .first = 0, .rows = 3 }}, 2));
    try std.testing.expectError(error.Qwen4LearningRows, Selection.of(&.{.{ .first = 0, .rows = 1 }}, 2));
}
