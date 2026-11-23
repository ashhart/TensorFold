//! How a round's drafting GLM streams group into DFlash2 passes: one pass at the deepest stream's block, eight at most.
const std = @import("std");
const st = @import("state.zig");
const StreamRounds = @import("metal_dflash_rounds.zig").StreamRounds;

/// A drafting stream's round to run: its rounds, committed end, anchor and the depth the core asked for.
pub const Job = struct {
    sr: *StreamRounds,
    position: u64,
    anchor: u32,
    depth: u32,
    pass: u32 = 0, // the drafts its pass runs for (0: its own depth): a shared pass takes its deepest stream's

    pub fn passDepth(j: Job) u32 {
        return if (j.pass > 0) j.pass else j.depth;
    }
};

/// One pass for every drafting stream of a round: each runs at the deepest one's block, its chain cut to its own depth.
pub fn sharePass(jobs: []Job) void {
    var deepest: u32 = 0;
    for (jobs) |j| deepest = @max(deepest, j.depth);
    if (jobs.len > 1) for (jobs) |*j| {
        j.pass = deepest;
    };
}

/// Each job's pass (one block size, at most eight jobs a pass, passes in order of their first job), and the count.
pub const Plan = struct { group: [st.max_rows]u8 = @splat(0), count: usize = 0 };
pub fn plan(blocks: []const u32) Plan {
    var p: Plan = .{};
    var block: [st.max_rows]u32 = undefined;
    var size: [st.max_rows]u8 = @splat(0);
    for (blocks, 0..) |b, i| {
        const g = for (0..p.count) |x| {
            if (block[x] == b and size[x] < 8) break x;
        } else blk: {
            block[p.count] = b;
            p.count += 1;
            break :blk p.count - 1;
        };
        p.group[i] = @intCast(g);
        size[g] += 1;
    }
    return p;
}

test "a round's drafting streams share one pass at the deepest one's depth, each chain cut to its own" {
    var jobs = [_]Job{ .{ .sr = undefined, .position = 9, .anchor = 1, .depth = 1 }, .{ .sr = undefined, .position = 9, .anchor = 1, .depth = 7 }, .{ .sr = undefined, .position = 9, .anchor = 1, .depth = 3 } };
    sharePass(&jobs);
    for (jobs) |j| try std.testing.expectEqual(@as(u32, 7), j.passDepth());
    try std.testing.expectEqual(@as(u32, 1), jobs[0].depth); // its held chain stops at its own
    var one = [_]Job{.{ .sr = undefined, .position = 9, .anchor = 1, .depth = 2 }};
    sharePass(&one);
    try std.testing.expectEqual(@as(u32, 2), one[0].passDepth()); // a lone stream keeps its own block
}

test "a round's passes: streams of one block size share one, at most eight, in the order of their first stream" {
    const p = plan(&.{ 8, 8, 4, 8, 4, 16 });
    try std.testing.expectEqual(@as(usize, 3), p.count);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 1, 0, 1, 2 }, p.group[0..6]);
    const tens: [10]u32 = @splat(8);
    const many = plan(&tens);
    try std.testing.expectEqual(@as(usize, 2), many.count);
    for (many.group[0..8]) |g| try std.testing.expectEqual(@as(u8, 0), g);
    try std.testing.expectEqual(@as(u8, 1), many.group[8]);
    try std.testing.expectEqual(@as(usize, 0), plan(&.{}).count);
}
