//! The stream tile's columns a lane: the widest that fills the card, the narrowest at RDNA3.5's one and two rows.

const std = @import("std");
const affine = @import("affine.zig");
const Generation = @import("../caps.zig").Generation;

fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

/// An index into `stream_cbs` for `m` rows of `n` columns over `items` (a pair's gate and up as one); sums ignore it.
pub fn pick(generation: Generation, m: c_int, lpc_log2: u5, bits: c_int, n: c_int, items: c_int, pair_div: c_int) usize {
    const ri: usize = if (m == 1) 0 else if (m == 2) 1 else 2;
    // RDNA3.5's one- and two-row launches keep more loads in flight with the extra waves of the narrowest
    const narrowest = generation == .rdna3_5 and ri < 2;
    var at: usize = 0;
    var c: usize = affine.stream_cbs.len;
    while (c > 0) {
        c -= 1;
        const cb = affine.stream_cbs[c];
        // within the registers the lane's code words take
        if (cb * affine.stream_rows[ri] > 16 or cb * bits > affine.stream_code_words) continue;
        at = c;
        const per_block: c_int = affine.stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(cb, pair_div);
        const waves = @as(u64, cdiv(n, per_block)) * affine.stream_waves * @as(u64, @intCast(items)) * cdiv(m, affine.stream_rows[ri]);
        if (!narrowest and waves >= affine.streams_wanted) break;
    }
    return at;
}

test "the widest columns that fill the card, and RDNA3.5's narrowest at one and two rows" {
    // 4-bit, K = 2048 (32 lanes a column): 12352 and 248320 columns
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 1, 5, 4, 12352, 1, 1));
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 1, 5, 4, 248320, 1, 1));
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 2, 5, 4, 248320, 1, 1));
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 12352, 1, 1));
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 248320, 1, 1));
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 2, 5, 4, 248320, 1, 1));
}

test "three rows and more keep the widest that fills the card on every generation" {
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3, 4, 5, 4, 248320, 1, 1));
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 4, 5, 4, 248320, 1, 1));
    try std.testing.expectEqual(@as(usize, 1), pick(.rdna3_5, 3, 5, 4, 248320, 1, 1));
}

test "a routed pair counts its gate and up as one column, over every item" {
    // 512 gate and up pairs over 40 items fill the card at 8 columns a lane
    try std.testing.expectEqual(@as(usize, 2), pick(.rdna3, 1, 5, 4, 512, 40, 2));
    try std.testing.expectEqual(@as(usize, 0), pick(.rdna3_5, 1, 5, 4, 512, 40, 2));
}
