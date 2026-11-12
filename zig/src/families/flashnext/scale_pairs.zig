//! A tensor's bf16 (scale, bias) pairs as one 16-bit index a group into a table of its distinct pairs.
const std = @import("std");

/// Distinct pairs a table holds at most: what a 16-bit index reaches.
pub const most = 1 << 16;
pub const slots = 1 << 18;

/// `groups` (scale, bias) pairs as indices into `table` (pairs in first-seen order); the count of distinct pairs.
pub fn index(scales: []align(1) const u16, biases: []align(1) const u16, out: []align(1) u16, table: *[most]u32, seen: *[slots]u32) !usize {
    @memset(seen, 0);
    var n: usize = 0;
    for (scales, biases, out) |s, b, *o| {
        const pair = @as(u32, b) << 16 | s;
        var h: usize = (pair *% 2654435761) >> 14;
        while (true) : (h = (h + 1) & (slots - 1)) {
            const at = seen[h];
            if (at == 0) {
                if (n == most) return error.TooManyPairs;
                table[n] = pair;
                n += 1;
                seen[h] = @intCast(n);
                o.* = @intCast(n - 1);
                break;
            }
            if (table[at - 1] == pair) {
                o.* = @intCast(at - 1);
                break;
            }
        }
    }
    return n;
}

test "pairs index in first-seen order and give back each group's scale and bias" {
    const s = [_]u16{ 7, 9, 7, 7, 1 };
    const b = [_]u16{ 3, 3, 3, 4, 0 };
    var out: [5]u16 = undefined;
    const table = try std.testing.allocator.create([most]u32);
    defer std.testing.allocator.destroy(table);
    const seen = try std.testing.allocator.create([slots]u32);
    defer std.testing.allocator.destroy(seen);
    try std.testing.expectEqual(@as(usize, 4), try index(&s, &b, &out, table, seen));
    try std.testing.expectEqualSlices(u16, &.{ 0, 1, 0, 2, 3 }, &out);
    for (s, b, out) |sv, bv, o| try std.testing.expectEqual(@as(u32, bv) << 16 | sv, table[o]);
}
