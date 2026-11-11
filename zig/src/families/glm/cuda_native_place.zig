//! A GLM rank's place from its resolved launch (native/cuda_launch.zig), its weight share and NCCL's rendezvous.
const std = @import("std");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const ranks = @import("cuda_ranks.zig");

pub const Place = struct { rank: usize, world: usize };

/// This rank's place: `o.rank` of `o.tp`, as the launch resolved them (an empty `launch_from`: none was given).
pub fn place(o: anytype) !Place {
    if (o.launch_from.len == 0) return error.NoRank;
    return checked(.{ .rank = o.rank, .world = o.tp });
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
pub fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Bytes this rank's weights take at most: the checkpoint (TF_GLM_LAYERS: the tensors read) times its largest share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64, drafts: bool, rank: u32, world: u32) u64 {
    const p = checked(.{ .rank = rank, .world = world }) catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    const whole = @import("cuda_thin_bytes.zig").count(a, io, dir, total, ranks.environment("TF_GLM_LAYERS"), c.layers, drafts and c.mtp > 0);
    return @intCast(@as(u128, whole) * f[0] / f[1]);
}

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expectEqual(Place{ .rank = 1, .world = 3 }, try place(.{ .rank = @as(u32, 1), .tp = @as(u32, 3), .launch_from = "--tp" }));
    try std.testing.expectError(error.NoRank, place(.{ .rank = @as(u32, 0), .tp = @as(u32, 1), .launch_from = "" }));
}
