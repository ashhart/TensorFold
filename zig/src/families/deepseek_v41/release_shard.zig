//! The release format's cuts, declared once on the mesh (fabric/mesh): output groups, expert width and ids, vocabulary.
const std = @import("std");
const mesh = @import("mesh");
const shard = mesh.shard;
const Config = @import("config.zig").Config;
const Share = @import("partition.zig").Share;

/// Expert-width elements a rank takes whole: two 32-wide blocks, the K step of the release's MMA tiles.
pub const expert_unit = 64;
/// Vocabulary rows a rank takes whole (the logits gather's blocks).
pub const vocab_unit = 128;

/// One rank's cuts of the release: tensor-parallel shares and the expert ids it holds.
pub const Cuts = struct {
    /// whole output groups: wq_b rows, wo_a groups, wo_b columns and the sinks follow them
    groups: Share,
    /// each expert's and the shared expert's width (w1 / w3 rows, w2 columns)
    width: Share,
    /// the expert ids whose weights this rank holds (every expert at expert degree 1)
    ids: Share,
    /// the head's and the embedding's rows
    vocab: Share,

    /// Rank `rank` of a tensor group of `world` (pipeline stages plan their own blocks; expert degree 1).
    pub fn of(c: *const Config, rank: u32, world: u32) !Cuts {
        return at(c, mesh.tensorGroup(rank, world), .{ .axis = .expert, .rank = 0, .world = 1, .first = 0, .stride = 1 }, c.experts);
    }

    /// The cuts of `tensor`'s rank, and of `experts`' rank over `count` experts.
    pub fn at(c: *const Config, tensor: mesh.Group, experts: mesh.Group, count: u32) !Cuts {
        return .{
            .groups = try span(.{ .kind = .column, .unit = 1 }, c.o_groups, tensor),
            .width = try span(.{ .kind = .column, .unit = expert_unit }, c.expert_width, tensor),
            .ids = try span(.{ .kind = .column, .unit = 1 }, count, experts),
            .vocab = try span(.{ .kind = .vocab, .unit = vocab_unit }, c.vocab, tensor),
        };
    }

    /// The rank's rows of a matrix laid out by output group (`width` elements, `groups` equal parts).
    pub fn byGroups(cut: Cuts, width: u32, groups: u32) Share {
        const per = width / groups;
        return .{ .lo = cut.groups.lo * per, .len = cut.groups.len * per };
    }
};

fn span(d: shard.Decl, extent: u32, g: mesh.Group) !Share {
    const r = try shard.range(d, extent, g);
    return .{ .lo = @intCast(r.lo), .len = @intCast(r.len()) };
}

test "the release's cuts on the mesh equal the family's own shares at worlds 1 to 4" {
    const partition = @import("partition.zig");
    var c: Config = .{};
    c.format = .release_fp8;
    for ([_]u32{ 1, 2, 4 }) |world| for (0..world) |r| {
        const rank: u32 = @intCast(r);
        const cut = try Cuts.of(&c, rank, world);
        try std.testing.expectEqualDeep(try partition.expertShare(&c, rank, world), cut.width);
        try std.testing.expectEqualDeep(try partition.share(c.vocab, rank, world), cut.vocab);
        try std.testing.expectEqualDeep(try partition.outputGroups(c.o_groups, rank, world), cut.groups);
        try std.testing.expectEqualDeep(try partition.outputShare(c.heads * c.head_dim, c.o_groups, rank, world), cut.byGroups(c.heads * c.head_dim, c.o_groups));
        try std.testing.expectEqualDeep(Share{ .lo = 0, .len = c.experts }, cut.ids);
    };
    // world 3 cuts the width (36 units) and the vocabulary evenly where it can, ties to the low ranks
    for (0..3) |r| {
        const cut = try Cuts.at(&c, mesh.tensorGroup(@intCast(r), 3), mesh.tensorGroup(0, 1), c.experts);
        try std.testing.expectEqual(@as(u32, 768), cut.width.len);
        try std.testing.expectEqualDeep(try partition.share(c.vocab, r, 3), cut.vocab);
    }
    // an expert group of 4 holds 96 experts a rank: the plan change expert parallelism needs, not a kernel change
    const ep = try Cuts.at(&c, mesh.tensorGroup(2, 4), .{ .axis = .expert, .rank = 2, .world = 4, .first = 0, .stride = 1 }, c.experts);
    try std.testing.expectEqualDeep(Share{ .lo = 192, .len = 96 }, ep.ids);
}
