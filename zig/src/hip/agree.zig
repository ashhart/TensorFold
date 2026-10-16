//! What a tensor-parallel group's ranks agree on: what they run at join, each startup step, each round's graph vote.

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

/// A startup step's outcome on one rank: two values, or why it cannot go on (the first 96 bytes).
pub const Step = extern struct {
    refused: u32 = 0,
    len: u32 = 0,
    values: [2]u64 = .{ 0, 0 },
    why: [96]u8 = @splat(0),

    pub fn ok(values: [2]u64) Step {
        return .{ .values = values };
    }

    pub fn refuse(why: []const u8) Step {
        var s: Step = .{ .refused = 1 };
        s.len = @intCast(@min(why.len, s.why.len));
        @memcpy(s.why[0..s.len], why[0..s.len]);
        return s;
    }
};

/// Every rank's verdict on a step: the values combined, or the first rank (in rank order) that refused and why.
pub const Verdict = extern struct {
    refused: u32 = 0,
    rank: u32 = 0,
    values: [2]u64 = .{ 0, 0 },
    len: u32 = 0,
    pad: u32 = 0,
    why: [96]u8 = @splat(0),

    pub fn reason(v: *const Verdict) []const u8 {
        return v.why[0..@min(v.len, v.why.len)];
    }
};

/// How a step's values combine: rank 0's (what it was asked for), or the least over the ranks (what every rank fits).
pub const How = enum(u32) { first, least };

/// The verdict on `steps`, one per rank in rank order: the first refusal stands for every rank.
pub fn combine(steps: []const Step, how: How) Verdict {
    for (steps, 0..) |s, r| if (s.refused != 0) return .{
        .refused = 1,
        .rank = @intCast(r),
        .len = s.len,
        .why = s.why,
    };
    var v: Verdict = .{ .values = steps[0].values };
    if (how == .least) for (steps[1..]) |s| {
        v.values[0] = @min(v.values[0], s.values[0]);
        v.values[1] = @min(v.values[1], s.values[1]);
    };
    return v;
}

/// A rank's word on a round it recorded: every rank replays, every rank runs it eagerly, or every rank stops.
pub const Vote = enum(u64) { declined = 0, kept = 1, failed = 2 };

/// The round's outcome on every rank from each rank's vote: one failure stops all, one decline runs all eagerly.
pub fn tally(votes: []const u64) Vote {
    var all: Vote = .kept;
    for (votes) |v| {
        switch (v) {
            @backingInt(Vote.kept) => {},
            @backingInt(Vote.declined) => all = .declined,
            else => return .failed,
        }
    }
    return all;
}

test "a step's verdict: rank 0's values, the least of each, or the first refusal by rank" {
    const steps = [_]Step{ .ok(.{ 8, 30 }), .ok(.{ 6, 40 }), .ok(.{ 7, 20 }) };
    try std.testing.expectEqual([2]u64{ 8, 30 }, combine(&steps, .first).values);
    try std.testing.expectEqual([2]u64{ 6, 20 }, combine(&steps, .least).values);
    try std.testing.expectEqual(@as(u32, 0), combine(&steps, .least).refused);
    const two = [_]Step{ .ok(.{ 1, 1 }), .refuse("--parallel 4 needs 9.5 GiB"), .refuse("later") };
    const v = combine(&two, .least);
    try std.testing.expectEqual(@as(u32, 1), v.refused);
    try std.testing.expectEqual(@as(u32, 1), v.rank);
    try std.testing.expectEqualStrings("--parallel 4 needs 9.5 GiB", v.reason());
    const text: [200]u8 = @splat('x');
    const long = Step.refuse(&text);
    try std.testing.expectEqual(@as(u32, 96), long.len);
}

test "a round's votes: all kept replays, a decline runs eagerly, a failure or an unknown word stops every rank" {
    try std.testing.expectEqual(Vote.kept, tally(&.{ 1, 1, 1, 1 }));
    try std.testing.expectEqual(Vote.declined, tally(&.{ 1, 0, 1, 1 }));
    try std.testing.expectEqual(Vote.failed, tally(&.{ 0, 1, 2, 1 }));
    try std.testing.expectEqual(Vote.failed, tally(&.{ 1, 7 }));
}

test "the wire layouts keep their sizes" {
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Identity));
    try std.testing.expectEqual(@as(usize, 120), @sizeOf(Step));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Verdict));
}
