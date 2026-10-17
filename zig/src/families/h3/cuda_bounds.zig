//! The H3 CUDA library's checks on a host's inputs, before any cast or allocation; std only, tested on any host.
const std = @import("std");

/// A sequence-parallel world's ranks, as h3_xchg.c allows them.
pub const max_world = 8;

pub const World = struct { world: u32, rank: u32 };

/// The C ABI's signed world and rank, checked before either is cast: 2 to 8 ranks and 0 <= rank < world.
pub fn world(ranks: c_int, rank: c_int) error{BadWorld}!World {
    if (ranks < 2 or ranks > max_world or rank < 0 or rank >= ranks) return error.BadWorld;
    return .{ .world = @intCast(ranks), .rank = @intCast(rank) };
}

/// The C ABI's signed peer, checked before it is cast: another rank of the world.
pub fn peer(p: c_int, w: World) error{BadWorld}!u32 {
    if (p < 0 or p >= w.world or p == w.rank) return error.BadWorld;
    return @intCast(p);
}

test "world and peer: signed values are refused before they are cast" {
    try std.testing.expectEqual(World{ .world = 2, .rank = 1 }, try world(2, 1));
    try std.testing.expectEqual(World{ .world = 8, .rank = 0 }, try world(8, 0));
    for ([_][2]c_int{ .{ 2, -1 }, .{ -2, 0 }, .{ 0, 0 }, .{ 1, 0 }, .{ 9, 0 }, .{ 2, 2 }, .{ 3, 7 }, .{ std.math.minInt(c_int), 0 }, .{ std.math.maxInt(c_int), 1 } }) |c|
        try std.testing.expectError(error.BadWorld, world(c[0], c[1]));
    const w = try world(3, 1);
    try std.testing.expectEqual(@as(u32, 2), try peer(2, w));
    for ([_]c_int{ -1, 1, 3, std.math.minInt(c_int) }) |p| try std.testing.expectError(error.BadWorld, peer(p, w));
}
