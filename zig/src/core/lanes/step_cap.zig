//! A capped prompt step's cut (the lane host's `max_rows`): extra cuts inside the family's cold chunk, on its row grid.
const std = @import("std");

/// A step from `at` to the cold cut `cold`: at most `cap` rows on `grid`, never past `cold`, so every cold cut lands.
pub fn end(at: usize, cold: usize, cap: u32, grid: usize) usize {
    std.debug.assert(grid > 0 and cold > at);
    if (cap == 0) return cold;
    const rows = @max(grid, @as(usize, cap) / grid * grid);
    return @min(cold, at + rows);
}

test "a cap cuts inside the cold chunk on the grid; 0 and a wide cap run to the cold cut" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 4096), end(0, 4096, 0, 128));
    try t.expectEqual(@as(usize, 512), end(0, 4096, 512, 128));
    try t.expectEqual(@as(usize, 4096), end(3584, 4096, 512, 128));
    try t.expectEqual(@as(usize, 4096), end(0, 4096, 8192, 128)); // never past the cold cut
    try t.expectEqual(@as(usize, 384), end(0, 4096, 500, 128)); // floored to the grid
    try t.expectEqual(@as(usize, 128), end(0, 4096, 64, 128)); // at least one grid step
    try t.expectEqual(@as(usize, 4096 + 100), end(4096, 4196, 512, 128)); // a short last chunk ends at the prompt
}

test "a capped pass hits every cold cut, whatever the caps between them" {
    const cold = [_]usize{ 4096, 8192, 9000 };
    const caps = [_]u32{ 512, 0, 128, 1000, 4096, 512 };
    var at: usize = 0;
    var c: usize = 0;
    var i: usize = 0;
    var hit: usize = 0;
    while (at < cold[cold.len - 1]) : (i += 1) {
        at = end(at, cold[c], caps[i % caps.len], 128);
        if (at == cold[c]) {
            hit += 1;
            c += 1;
        } else try std.testing.expect(at % 128 == 0);
    }
    try std.testing.expectEqual(cold.len, hit);
}
