//! The stream tile's launch: quant/mlx_tiles.hpp's StreamSides and where a round's row blocks lie on the grid.
const std = @import("std");
const hl = @import("../launch.zig");

/// quant/mlx_tiles.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
pub const StreamSides = extern struct {
    words: [4]u64 = @splat(0),
    scale: [4]u64 = @splat(0),
    bias: [4]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    n: [4]c_int = @splat(0),
    first: [5]c_int = @splat(0),
    count: c_int = 0,
    out_half: c_int = 0,
    pair_cols: c_int = 0,
    limit: f32 = 0,
};

/// The stream tile's grid; stream_tile reads a one-row grid over more than one row block as interleaved on x.
pub fn grid(interleave: bool, blocks: c_uint, ny: c_uint, items: c_int) hl.Dim3 {
    if (interleave and ny > 1) return .{ .x = blocks * ny, .y = 1, .z = @intCast(items) };
    return .{ .x = blocks, .y = ny, .z = @intCast(items) };
}

test "a round's row blocks interleave on x only when there are several and the launch asks" {
    const on = grid(true, 30, 4, 2);
    try std.testing.expect(on.x == 120 and on.y == 1 and on.z == 2);
    const off = grid(false, 30, 4, 2);
    try std.testing.expect(off.x == 30 and off.y == 4 and off.z == 2);
    const one = grid(true, 30, 1, 1);
    try std.testing.expect(one.x == 30 and one.y == 1 and one.z == 1);
}
