//! One mesh of ranks (pipeline x tensor, experts in blocks of the tensor group), every group derived from it.
const std = @import("std");

pub const shard = @import("shard.zig");
pub const contract = @import("contract.zig");
pub const launch = @import("launch.zig");

pub const Axis = enum { tensor, pipeline, expert };

/// Degrees of the mesh: ranks = pipeline x tensor; expert divides tensor; data is reserved and stays 1.
pub const Degrees = struct {
    tensor: u32 = 1,
    pipeline: u32 = 1,
    expert: u32 = 1,
    data: u32 = 1,
};

/// One rank's group on an axis: `rank` of `world` members, member k at global rank first + k * stride.
pub const Group = struct {
    axis: Axis,
    rank: u32,
    world: u32,
    first: u32,
    stride: u32,

    /// The global rank of member `k`.
    pub fn member(g: Group, k: u32) u32 {
        return g.first + k * g.stride;
    }
};

pub const Error = error{ BadDegrees, BadRank };

pub const Mesh = struct {
    degrees: Degrees,
    rank: u32,

    /// Rank `rank` of the mesh `d` (pipeline-major, tensor fastest, as cluster/plan.zig numbers its stages).
    pub fn init(d: Degrees, rank: u32) Error!Mesh {
        if (d.tensor == 0 or d.pipeline == 0 or d.expert == 0 or d.data != 1) return error.BadDegrees;
        if (d.tensor % d.expert != 0) return error.BadDegrees;
        _ = std.math.mul(u32, d.pipeline, d.tensor) catch return error.BadDegrees;
        const m: Mesh = .{ .degrees = d, .rank = rank };
        if (rank >= m.ranks()) return error.BadRank;
        return m;
    }

    pub fn ranks(m: Mesh) u32 {
        return m.degrees.pipeline * m.degrees.tensor;
    }

    /// This rank's group on `axis`: its stage's ranks, its slot in every stage, or its slot in every expert block.
    pub fn group(m: Mesh, axis: Axis) Group {
        const t = m.degrees.tensor;
        const stage = m.rank / t;
        const local = m.rank % t;
        return switch (axis) {
            .tensor => .{ .axis = .tensor, .rank = local, .world = t, .first = stage * t, .stride = 1 },
            .pipeline => .{ .axis = .pipeline, .rank = stage, .world = m.degrees.pipeline, .first = local, .stride = t },
            .expert => blk: {
                const per = t / m.degrees.expert;
                break :blk .{ .axis = .expert, .rank = local / per, .world = m.degrees.expert, .first = stage * t + local % per, .stride = per };
            },
        };
    }
};

/// A tensor group alone (a family with no pipeline or expert axis): rank `rank` of `world`.
pub fn tensorGroup(rank: u32, world: u32) Group {
    return .{ .axis = .tensor, .rank = rank, .world = world, .first = 0, .stride = 1 };
}

test "groups follow cluster/plan.zig's stage numbering for every rank up to 16" {
    var tensor: u32 = 1;
    while (tensor <= 16) : (tensor += 1) {
        var pipeline: u32 = 1;
        while (pipeline * tensor <= 16) : (pipeline += 1) {
            const n = pipeline * tensor;
            for (0..n) |r| {
                const m = try Mesh.init(.{ .tensor = tensor, .pipeline = pipeline }, @intCast(r));
                // plan.zig: stage i holds ranks [i * tensor, i * tensor + tensor) (Stage.first, Stage.tensor)
                const stage: u32 = @intCast(r / tensor);
                const tg = m.group(.tensor);
                try std.testing.expectEqual(stage * tensor, tg.first);
                try std.testing.expectEqual(tensor, tg.world);
                try std.testing.expectEqual(@as(u32, @intCast(r)), tg.member(tg.rank));
                const pg = m.group(.pipeline);
                try std.testing.expectEqual(stage, pg.rank);
                try std.testing.expectEqual(pipeline, pg.world);
                try std.testing.expectEqual(@as(u32, @intCast(r)), pg.member(pg.rank));
                for (0..pipeline) |k| try std.testing.expectEqual(@as(u32, @intCast(r % tensor)), pg.member(@intCast(k)) % tensor);
            }
        }
    }
}

test "expert groups take one slot of each block, as Plan.bytesOn places experts" {
    // tensor 8, expert 4: blocks of 2 ranks; rank 5 is local 5, block 2, slot 1
    const m = try Mesh.init(.{ .tensor = 8, .pipeline = 2, .expert = 4 }, 8 + 5);
    const g = m.group(.expert);
    try std.testing.expectEqual(@as(u32, 2), g.rank);
    try std.testing.expectEqual(@as(u32, 4), g.world);
    try std.testing.expectEqual([4]u32{ 9, 11, 13, 15 }, .{ g.member(0), g.member(1), g.member(2), g.member(3) });
}

test "bad degrees and ranks refuse" {
    try std.testing.expectError(error.BadDegrees, Mesh.init(.{ .tensor = 0 }, 0));
    try std.testing.expectError(error.BadDegrees, Mesh.init(.{ .tensor = 4, .expert = 3 }, 0));
    try std.testing.expectError(error.BadDegrees, Mesh.init(.{ .tensor = 2, .data = 2 }, 0));
    try std.testing.expectError(error.BadDegrees, Mesh.init(.{ .tensor = 65536, .pipeline = 65536 }, 0));
    try std.testing.expectError(error.BadRank, Mesh.init(.{ .tensor = 2, .pipeline = 2 }, 4));
}

test {
    _ = shard;
    _ = contract;
    _ = launch;
}
