//! What a tensor-parallel group's ranks agree on at join: what each runs, and the field that differs.

const std = @import("std");

/// What a rank runs; every rank's must equal rank 0's (paths may differ, contents may not).
pub const Identity = extern struct {
    /// sha256 of the executable: the collective schedule is its code.
    program: [32]u8 = @splat(0),
    /// sha256 of the checkpoint's config.json.
    config: [32]u8 = @splat(0),
    /// sha256 of the weight index and each shard's size, header and sampled tensor bytes.
    weights: [32]u8 = @splat(0),
};

/// What a rank's join can differ in from rank 0's, in the order they are compared.
pub const Field = enum(u32) {
    layout,
    world,
    rank,
    gpu,
    rccl,
    program,
    config,
    weights,

    /// The refusal's words for it.
    pub fn what(f: Field) []const u8 {
        return switch (f) {
            .layout => "join handshake (another TensorFold build)",
            .world => "--tp (the ranks were started with different world sizes)",
            .rank => "--rank (outside the world, or given to two ranks)",
            .gpu => "GPU (a group needs one kind of GPU)",
            .rccl => "RCCL version",
            .program => "executable (run the same build on every rank)",
            .config => "config.json",
            .weights => "weights (the shards' headers or sampled tensor bytes)",
        };
    }
};

/// The rank whose join differs from rank 0's, and in what (filled when a join fails with error.RanksDiffer).
pub const Fault = struct { field: Field = .layout, rank: u32 = 0 };

test "the identity keeps its wire size" {
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Identity));
}
