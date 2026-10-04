const std = @import("std");
const c = @import("argmax_order");

fn select(words: []const u32) u64 {
    var chosen = c.tf_max_empty();
    for (words, 0..) |word, column| chosen = c.tf_max_pick(chosen, c.tf_max_bits(word, @intCast(column)));
    return chosen.column;
}

test "finite maxima and ties use their first column" {
    try std.testing.expectEqual(@as(u64, 1), select(&.{ 0xbf800000, 0x40000000, 0x3f800000, 0x40000000 }));
    try std.testing.expectEqual(@as(u64, 0), select(&.{ 0xff800000, 0xff800000 }));
    try std.testing.expectEqual(@as(u64, 1), select(&.{ 0x3f800000, 0x7f800000, 0x7f800000 }));
}

test "zeros have one numeric rank and do not lose subnormals" {
    try std.testing.expectEqual(@as(u64, 0), select(&.{ 0x80000000, 0x00000000 }));
    try std.testing.expectEqual(@as(u64, 1), select(&.{ 0x00000000, 0x00000001 }));
    try std.testing.expectEqual(@as(u64, 1), select(&.{ 0x80000001, 0x00000000 }));
}

test "NaNs outrank infinities and finite values with first NaN tie" {
    try std.testing.expectEqual(@as(u64, 1), select(&.{ 0x7f800000, 0xffc12345, 0x7fc00001, 0x3f800000 }));
    try std.testing.expectEqual(@as(u64, 0), select(&.{ 0x7f800001, 0x7fffffff }));
}

test "BF16 encodings retain the same ordering when widened by bits" {
    const words = [_]u16{ 0xbf80, 0x4000, 0x3f80, 0x4000 };
    var wide: [words.len]u32 = undefined;
    for (words, 0..) |word, i| wide[i] = @as(u32, word) << 16;
    try std.testing.expectEqual(@as(u64, 1), select(&wide));
}

test "partition and reduction order cannot change a chosen column" {
    const words = [_]u32{ 0xff800000, 0x3f800000, 0x7fc00001, 0x40000000, 0xff800001, 0x40000000, 1 };
    var left = c.tf_max_empty();
    var right = c.tf_max_empty();
    for (words, 0..) |word, i| {
        const candidate = c.tf_max_bits(word, @intCast(i));
        if (i % 2 == 0) left = c.tf_max_pick(left, candidate) else right = c.tf_max_pick(right, candidate);
    }
    try std.testing.expectEqual(select(&words), c.tf_max_pick(left, right).column);
    try std.testing.expectEqual(select(&words), c.tf_max_pick(right, left).column);
}

test "dimensions reject overflow and unsupported encodings before launch" {
    try std.testing.expect(c.tf_argmax_dimensions(0, 1, 1, 0));
    try std.testing.expect(c.tf_argmax_dimensions(16, 248320, 248336, 1));
    try std.testing.expect(!c.tf_argmax_dimensions(1, 0, 0, 0));
    try std.testing.expect(!c.tf_argmax_dimensions(1, 1, 1, 2));
    try std.testing.expect(!c.tf_argmax_dimensions(1, 9, 8, 0));
    try std.testing.expect(!c.tf_argmax_dimensions(2147483648, 1, 1, 0));
    try std.testing.expect(!c.tf_argmax_dimensions(3, 1, std.math.maxInt(i64), 1));
}

test "output indices cannot overlap input bytes or wrap an address" {
    try std.testing.expect(c.tf_argmax_disjoint(4096, 8192, 1024, 128));
    try std.testing.expect(!c.tf_argmax_disjoint(4096, 4104, 1024, 128));
    try std.testing.expect(!c.tf_argmax_disjoint(std.math.maxInt(usize) - 4, 8192, 8, 128));
}

test "raw-key selection agrees with IEEE comparisons on varied encodings" {
    var state: u32 = 317;
    var chosen = c.tf_max_empty();
    var expected: f32 = -std.math.inf(f32);
    var expected_column: u64 = 0;
    for (0..4096) |i| {
        state = state *% 1664525 +% 1013904223;
        const value: f32 = @bitCast(state);
        if (i == 0 or (!std.math.isNan(expected) and (std.math.isNan(value) or value > expected))) {
            expected = value;
            expected_column = @intCast(i);
        }
        chosen = c.tf_max_pick(chosen, c.tf_max_bits(state, @intCast(i)));
    }
    try std.testing.expectEqual(expected_column, chosen.column);
}
