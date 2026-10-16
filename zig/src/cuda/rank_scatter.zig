//! A reduce-scatter by rows in rank order: an all-gather plus a rank-order sum's bits (not ncclReduceScatter's order).
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Stream = stream_.Stream;

/// Rank `rank`'s rows [lo, hi): whole rows, the first `rows % world` ranks one more (canon.partition's equal cut).
pub fn share(rows: usize, world: usize, rank: usize) [2]usize {
    const base = rows / world;
    const extra = rows % world;
    const lo = rank * base + @min(rank, extra);
    return .{ lo, lo + base + @intFromBool(rank < extra) };
}

/// Bytes of the receive buffer `run` needs: a slot of the largest share for every rank.
pub fn scratchBytes(rows: usize, world: usize, width: usize) usize {
    return world * ((rows + world - 1) / world) * width * 4;
}

pub const RankScatter = struct {
    mod: module.Module,
    sum: module.Function,

    pub fn load(d: *const driver.Driver) !RankScatter {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.rank_sum);
        errdefer mod.unload();
        return .{ .mod = mod, .sum = try mod.function("tf_rank_sum") };
    }

    pub fn unload(k: *RankScatter) void {
        k.mod.unload();
    }

    /// `partial` [rows][width] fp32 summed over the ranks into `out` [own rows][width]: a send/recv group, then a sum.
    pub fn run(
        k: *const RankScatter,
        lib: *const nccl.Library,
        comm: nccl.Comm,
        world: usize,
        rank: usize,
        s: Stream,
        partial: u64,
        rows: usize,
        width: usize,
        scratch: u64,
        out: u64,
    ) !void {
        const mine = share(rows, world, rank);
        const n = (mine[1] - mine[0]) * width;
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        for (0..world) |peer| {
            if (peer == rank) continue;
            const theirs = share(rows, world, peer);
            const count = (theirs[1] - theirs[0]) * width;
            const send = partial + theirs[0] * width * 4;
            const slot = scratch + peer * n * 4;
            const to: c_int = @intCast(peer);
            if (count > 0) try lib.check(lib.api.ncclSend(send, count, .f32, to, comm, s.handle), "ncclSend");
            if (n > 0) try lib.check(lib.api.ncclRecv(slot, n, .f32, to, comm, s.handle), "ncclRecv");
        }
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
        if (n > 0) try k.rankSum(s, scratch, partial + mine[0] * width * 4, world, rank, n, out);
    }

    /// out[i]: the ranks' terms added rank 0 first, `peers`' slot r (n apart) for every rank but `rank`, `own` for it.
    pub fn rankSum(
        k: *const RankScatter,
        s: Stream,
        peers: u64,
        own: u64,
        world: usize,
        rank: usize,
        n: usize,
        out: u64,
    ) !void {
        var a: launch_.Args = .{};
        for ([_]u64{ peers, own }) |v| a.add(v);
        a.add(@as(c_int, @intCast(world)));
        a.add(@as(c_int, @intCast(rank)));
        a.add(@as(i64, @intCast(n)));
        a.add(out);
        const blocks: u32 = @intCast(@min((n + 255) / 256, 4096));
        const cfg: launch_.Config = .{
            .grid = .{ .x = blocks, .y = 1, .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        };
        try launch_.launch(k.sum, cfg, s, &a);
    }
};

test "shares cover the rows in rank order, whole rows, the first rows % world ranks one more" {
    for ([_]usize{ 0, 1, 4, 64, 2047, 2048 }) |rows| for (1..6) |world| {
        var at: usize = 0;
        for (0..world) |r| {
            const s = share(rows, world, r);
            try std.testing.expectEqual(at, s[0]);
            try std.testing.expectEqual(rows / world + @intFromBool(r < rows % world), s[1] - s[0]);
            at = s[1];
        }
        try std.testing.expectEqual(rows, at);
    };
    try std.testing.expectEqual([2]usize{ 0, 683 }, share(2048, 3, 0));
    try std.testing.expectEqual([2]usize{ 1366, 2048 }, share(2048, 3, 2));
    try std.testing.expectEqual([2]usize{ 683, 1365 }, share(2047, 3, 1));
    try std.testing.expectEqual(@as(usize, 3 * 683 * 4096 * 4), scratchBytes(2048, 3, 4096));
}
