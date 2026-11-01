//! The stream tile's columns a lane: the widest that fills the card, the narrowest at RDNA3.5's one and two rows.

const std = @import("std");
const affine = @import("affine.zig");
const Generation = @import("../caps.zig").Generation;

fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

// Output columns a block of the `c`th width: its waves' lane groups times a lane's outputs.
fn perBlock(c: usize, lpc_log2: u5, halves: c_int) c_int {
    return affine.stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(affine.stream_cbs[c], halves);
}

/// The row tile of `m` rows, an index into `stream_rows`: 1, 2, then 4 rows a block.
pub fn rowTile(m: c_int) usize {
    return if (m == 1) 0 else if (m == 2) 1 else 2;
}

/// A launch's columns a lane, an index into `stream_cbs`, and its output columns a block.
pub const Tile = struct { at: usize, per_block: c_int };

/// The tile for `m` rows of `n` outputs over `items`; a `pair` lane's columns are the gate and up halves of outputs.
pub fn pick(generation: Generation, m: c_int, lpc_log2: u5, bits: c_int, n: c_int, items: c_int, pair: bool) Tile {
    const ri = rowTile(m);
    // RDNA3.5's one- and two-row launches keep more loads in flight with the extra waves of the narrowest
    const narrowest = generation == .rdna3_5 and ri < 2;
    const halves: c_int = if (pair) 2 else 1;
    // the narrowest when no width fits the lane's registers
    var tile: Tile = .{ .at = 0, .per_block = perBlock(0, lpc_log2, halves) };
    var c: usize = affine.stream_cbs.len;
    while (c > 0) {
        c -= 1;
        const cb = affine.stream_cbs[c];
        // within the registers the lane's code words take
        if (cb * affine.stream_rows[ri] > 16 or cb * bits > affine.stream_code_words) continue;
        tile = .{ .at = c, .per_block = perBlock(c, lpc_log2, halves) };
        const waves = @as(u64, cdiv(n, tile.per_block)) * affine.stream_waves * @as(u64, @intCast(items)) * cdiv(m, affine.stream_rows[ri]);
        if (!narrowest and waves >= affine.streams_wanted) break;
    }
    return tile;
}

test "the widest columns that fill the card, and RDNA3.5's narrowest at one and two rows" {
    // 4-bit, K = 2048 (32 lanes a column): 12352 and 248320 columns
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 1, 5, 4, 12352, 1, false).at);
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 1, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 2, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 12352, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 2, 5, 4, 248320, 1, false).at);
}

test "three rows and more keep the widest that fills the card on every generation" {
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 4, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 4, 5, 4, 248320, 1, false).at);
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 3, 5, 4, 248320, 1, false).at);
}

test "a routed pair counts its gate and up as one column, over every item" {
    // 512 gate and up pairs over 40 items fill the card at 8 columns a lane, 4 outputs a wave
    try std.testing.expectEqual(Tile{ .at = 2, .per_block = 16 }, pick(.rdna3, 1, 5, 4, 512, 40, true));
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 4 }, pick(.rdna3_5, 1, 5, 4, 512, 40, true));
}

test "a width no column count fits falls back to the narrowest tile and its block width" {
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 8 }, pick(.rdna3, 1, 5, 32, 12352, 1, false));
    try std.testing.expectEqual(Tile{ .at = 0, .per_block = 4 }, pick(.rdna3_5, 1, 5, 32, 512, 9, true));
}

test "the row tile takes 1, 2, then 4 rows a block" {
    try std.testing.expectEqual(@as(usize, 0), rowTile(1));
    try std.testing.expectEqual(@as(usize, 1), rowTile(2));
    try std.testing.expectEqual(@as(usize, 2), rowTile(3));
    try std.testing.expectEqual(@as(usize, 2), rowTile(16));
}
