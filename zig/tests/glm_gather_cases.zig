//! Routing distributions shared by the GPU gather gate and shader-free coverage checks.
const std = @import("std");
pub const experts = 35;
pub const pairs = 37;
pub const Kind = enum { sparse, fours, dense_tails };
pub const kinds = std.enums.values(Kind);

pub fn ids(kind: Kind) [pairs]i32 {
    var result: [pairs]i32 = undefined;
    for (&result, 0..) |*id, i| id.* = @intCast(switch (kind) {
        .sparse => (i * 17 + i / 3) % experts,
        .fours => if (i < 32) i / 4 else if (i < 36) @as(usize, 34) else 17,
        .dense_tails => if (i < 20) i / 5 else if (i < 30) @as(usize, 34) else if (i < 36) @as(usize, 16) else 17,
    });
    return result;
}

fn counts(kind: Kind) ![experts]usize {
    var rows: [experts]usize = @splat(0);
    for (ids(kind)) |id| {
        try std.testing.expect(id >= 0 and id < experts);
        rows[@intCast(id)] += 1;
    }
    return rows;
}

test "GLM gather fixture includes a valid fourth row, empty experts and partial tiles" {
    var fourth = false;
    var empty = false;
    var tail = false;
    for (kinds) |kind| for (try counts(kind)) |n| {
        fourth = fourth or n >= 4;
        empty = empty or n == 0;
        tail = tail or n % 4 != 0;
    };
    try std.testing.expect(fourth);
    try std.testing.expect(empty);
    try std.testing.expect(tail);
}

test "GLM gather fixture repeats an expert tile with valid rows and a partial final tile" {
    var repeated = false;
    var second_full = false;
    for (kinds) |kind| for (try counts(kind)) |n| {
        repeated = repeated or (n >= 5 and n % 4 != 0);
        second_full = second_full or n >= 8;
    };
    try std.testing.expect(repeated);
    try std.testing.expect(second_full);
}

test "GLM gather fixture pins sparse, full-four and repeated-tile expert populations" {
    const sparse = try counts(.sparse);
    try std.testing.expectEqual(@as(usize, 3), std.mem.max(usize, &sparse));
    const fours = try counts(.fours);
    for (0..8) |i| try std.testing.expectEqual(@as(usize, 4), fours[i]);
    try std.testing.expectEqual(@as(usize, 4), fours[34]);
    try std.testing.expectEqual(@as(usize, 1), fours[17]);
    const dense = try counts(.dense_tails);
    for (0..4) |i| try std.testing.expectEqual(@as(usize, 5), dense[i]);
    try std.testing.expectEqual(@as(usize, 6), dense[16]);
    try std.testing.expectEqual(@as(usize, 1), dense[17]);
    try std.testing.expectEqual(@as(usize, 10), dense[34]);
    for ([_][experts]usize{ sparse, fours, dense }) |rows| {
        var sum: usize = 0;
        var empty: usize = 0;
        for (rows) |n| {
            sum += n;
            empty += @intFromBool(n == 0);
        }
        try std.testing.expectEqual(@as(usize, pairs), sum);
        try std.testing.expect(empty > 0);
    }
}
