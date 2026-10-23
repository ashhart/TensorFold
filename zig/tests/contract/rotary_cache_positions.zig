//! Seed the diagnostic cache with positions consumed by each pooled row.
const std = @import("std");

pub fn pooled(allocator: std.mem.Allocator, first: usize, rows: usize, blocks: ?[]const i32) ![]i32 {
    if (blocks) |bpos| return allocator.dupe(i32, bpos[first * 3 ..][0 .. rows * 3]);
    const values = try allocator.alloc(i32, rows);
    for (values, 0..) |*v, i| v.* = @intCast(4 * (first + i));
    return values;
}

const a = std.testing.allocator;

test "rotary cache positions: long positioned windows seed all three consumed axes" {
    const positions = [_]i32{ 0, 1, 850, 262143, 262144, 1048575, 1048576, 1048639 };
    for ([_]usize{ 1, 4, 8, 16 }) |rows| {
        const first = 1048640 / 4 - rows;
        const blocks = try a.alloc(i32, (first + rows) * 3);
        defer a.free(blocks);
        @memset(blocks, -1);
        for (0..rows) |r| for (0..3) |axis| {
            blocks[(first + r) * 3 + axis] = positions[(r + (if (r % 2 == 0) 0 else axis)) % positions.len];
        };
        const values = try pooled(a, first, rows, blocks);
        defer a.free(values);
        try std.testing.expectEqual(rows * 3, values.len);
        for (0..rows) |r| for (0..3) |axis| {
            const want = positions[(r + (if (r % 2 == 0) 0 else axis)) % positions.len];
            try std.testing.expectEqual(want, values[r * 3 + axis]);
        };
        try std.testing.expectEqual(@as(i32, -1), blocks[first * 3 - 1]);
    }
}

test "rotary cache positions: positioned short rows retain distinct image axes and block offset" {
    var blocks: [11 * 3]i32 = @splat(-1);
    for (7..11) |b| for (0..3) |axis| {
        blocks[b * 3 + axis] = @intCast(4 * b + (if ((b - 7) % 2 == 0) 0 else axis * 3));
    };
    const values = try pooled(a, 7, 4, &blocks);
    defer a.free(values);
    try std.testing.expectEqualSlices(i32, &.{ 28, 28, 28, 32, 35, 38, 36, 36, 36, 40, 43, 46 }, values);
    try std.testing.expectEqual(@as(i32, -1), blocks[20]);
}

test "rotary cache positions: unpositioned pooling retains absolute block starts" {
    for ([_]usize{ 1, 4, 8, 16 }) |rows| for ([_]usize{ 7, 1048640 / 4 - rows }) |first| {
        const values = try pooled(a, first, rows, null);
        defer a.free(values);
        try std.testing.expectEqual(rows, values.len);
        for (values, 0..) |v, r| try std.testing.expectEqual(@as(i32, @intCast(4 * (first + r))), v);
    };
}
