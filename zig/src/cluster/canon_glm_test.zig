//! canon.partition over equal ranks cuts GLM-5.3-Flash's split axes as the Python family's split_units does.

const std = @import("std");
const canon = @import("canon.zig");

/// An axis in whole units (heads one by one; MLP widths and vocabulary in 64s), the Python family's [first, end).
const Case = struct { what: []const u8, units: u32, shares: []const [2]u32 };

// printed by release/2.7.0's split_units for TP=2 and TP=3
const cases = [_]Case{
    .{ .what = "KDA heads, TP=2", .units = 64, .shares = &.{ .{ 0, 32 }, .{ 32, 64 } } },
    .{ .what = "KDA heads, TP=3", .units = 64, .shares = &.{ .{ 0, 22 }, .{ 22, 43 }, .{ 43, 64 } } },
    .{ .what = "routed expert width / 64, TP=2", .units = 32, .shares = &.{ .{ 0, 16 }, .{ 16, 32 } } },
    .{ .what = "routed expert width / 64, TP=3", .units = 32, .shares = &.{ .{ 0, 11 }, .{ 11, 22 }, .{ 22, 32 } } },
    .{ .what = "dense MLP width / 64, TP=2", .units = 192, .shares = &.{ .{ 0, 96 }, .{ 96, 192 } } },
    .{ .what = "dense MLP width / 64, TP=3", .units = 192, .shares = &.{ .{ 0, 64 }, .{ 64, 128 }, .{ 128, 192 } } },
    .{ .what = "vocabulary / 64, TP=2", .units = 2420, .shares = &.{ .{ 0, 1210 }, .{ 1210, 2420 } } },
    .{ .what = "vocabulary / 64, TP=3", .units = 2420, .shares = &.{ .{ 0, 807 }, .{ 807, 1614 }, .{ 1614, 2420 } } },
};

test "equal ranks get the Python family's shares of every GLM axis: units / world, the remainder to the lowest ranks" {
    const ones = [_]u64{ 1, 1, 1 };
    for (cases) |c| {
        var out: [3]canon.Range = undefined;
        const world = c.shares.len;
        canon.partition(c.units, ones[0..world], out[0..world]);
        for (c.shares, out[0..world], 0..) |want, got, rank| {
            if (got.begin != want[0] or got.end != want[1]) {
                std.debug.print("{s}: rank {d} got [{d}, {d}), Python [{d}, {d})\n", .{ c.what, rank, got.begin, got.end, want[0], want[1] });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "a rank left empty: the Python family refuses the split, partition gives it nothing, so a share plan must check" {
    var out: [3]canon.Range = undefined;
    canon.partition(2, &.{ 1, 1, 1 }, &out);
    try std.testing.expectEqual(@as(u32, 0), out[2].len());
}
