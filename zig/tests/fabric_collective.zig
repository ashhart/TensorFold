//! Four ranks on threads over the queued fake with Thunderbolt's limits (no READ, no atomics): collectives in lockstep.
const std = @import("std");
const fabric = @import("fabric");
const collective = fabric.collective;
const fake = fabric.fake;
const Rdma = fabric.Rdma;

const gpa = std.testing.allocator;
const n = 4;
const hidden = 7168;
const max_rows = 300;
const row_bytes = hidden * 2;
const slot = max_rows * row_bytes;
const channel_bytes = collective.Channel.bytes(n, slot);
const pipe_at = 2 * channel_bytes;
const pipe_bytes = collective.Pipe.bytes(64 << 10);
const bulk_at = pipe_at + n * pipe_bytes;
const bulk_flag = bulk_at - 64;
const window = bulk_at + (12 << 20);

/// A row's partial on one rank depends only on (rank, row id): never on the batch it rides in.
fn partialRow(rank: u32, id: u64, out: []u8) void {
    var prng: std.Random.DefaultPrng = .init(id * 131 + rank);
    const r = prng.random();
    for (0..hidden) |i| {
        const v = (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 6)));
        std.mem.writeInt(u16, out[2 * i ..][0..2], collective.toBf16(v), .little);
    }
}

/// Which row ids step `s` carries: 1 to 300 rows, a prompt chunk among decode rows.
fn rowsOf(s: u64, ids: []u64) []u64 {
    const counts = [_]usize{ 1, 7, 64, 300, 2, 129, 1, 33 };
    const k = counts[s % counts.len];
    for (ids[0..k], 0..) |*id, j| id.* = (s * 7919 + j * 104729) % 1000;
    return ids[0..k];
}

const Shared = struct {
    cluster: *fake.Cluster,
    outs: [n][]u8,
    failed: std.atomic.Value(bool) = .init(false),
};

fn allReduceRank(sh: *Shared, rank: u32) void {
    allReduceSteps(sh, rank) catch |err| {
        std.debug.print("rank {d}: {s}\n", .{ rank, @errorName(err) });
        sh.failed.store(true, .release);
    };
}

fn allReduceSteps(sh: *Shared, rank: u32) !void {
    const ep = sh.cluster.endpoint(rank);
    var ch = collective.Channel.init(ep, 0, slot);
    const partial = try gpa.alloc(u8, slot);
    defer gpa.free(partial);
    var ids: [max_rows]u64 = undefined;
    var scratch: [n][]const u8 = undefined;
    var at: usize = 0;
    for (0..16) |s| {
        const rows = rowsOf(s, &ids);
        for (rows, 0..) |id, j| partialRow(rank, id, partial[j * row_bytes ..][0..row_bytes]);
        const len = rows.len * row_bytes;
        try collective.allReduce(&ch, .bf16, partial[0..len], sh.outs[rank][at..][0..len], &scratch);
        at += len;
    }
}

fn spawnRanks(sh: *Shared, comptime f: anytype) !void {
    var stop: std.atomic.Value(bool) = .init(false);
    const pump = try std.Thread.spawn(.{}, fake.Cluster.pump, .{ sh.cluster, &stop });
    var threads: [n]std.Thread = undefined;
    for (&threads, 0..) |*t, r| t.* = try std.Thread.spawn(.{}, f, .{ sh, @as(u32, @intCast(r)) });
    for (threads) |t| t.join();
    stop.store(true, .release);
    pump.join();
    try std.testing.expect(!sh.failed.load(.acquire));
}

fn thunderbolt(c: *fake.Cluster) void {
    for (c.access) |*a| a.* = .{ .remote_write = true, .remote_read = false, .remote_atomic = false };
}

fn totalBytes() usize {
    var ids: [max_rows]u64 = undefined;
    var total: usize = 0;
    for (0..16) |s| total += rowsOf(s, &ids).len * row_bytes;
    return total;
}

test "all-reduce: every rank gets the same bits, each row equal to its solo reduction in rank order" {
    const total = totalBytes();
    var first: ?[]u8 = null;
    defer if (first) |f| gpa.free(f);
    for (0..2) |run| {
        const c = try fake.Cluster.init(gpa, n, window, .queued, null);
        defer c.deinit();
        thunderbolt(c);
        c.prng = .init(run + 11);
        var sh: Shared = .{ .cluster = c, .outs = undefined };
        for (&sh.outs) |*o| o.* = try gpa.alloc(u8, total);
        defer for (sh.outs) |o| gpa.free(o);
        try spawnRanks(&sh, allReduceRank);
        for (sh.outs[1..]) |o| try std.testing.expectEqualSlices(u8, sh.outs[0], o);
        var ids: [max_rows]u64 = undefined;
        var parts: [n][row_bytes]u8 = undefined;
        var solo: [row_bytes]u8 = undefined;
        var at: usize = 0;
        for (0..16) |s| {
            for (rowsOf(s, &ids)) |id| {
                for (&parts, 0..) |*p, r| partialRow(@intCast(r), id, p);
                const views = [n][]const u8{ &parts[0], &parts[1], &parts[2], &parts[3] };
                collective.reduce(.bf16, &views, &solo);
                try std.testing.expectEqualSlices(u8, &solo, sh.outs[0][at..][0..row_bytes]);
                try expectReference(&parts, sh.outs[0][at..][0..row_bytes]);
                at += row_bytes;
            }
        }
        if (first) |f| try std.testing.expectEqualSlices(u8, f, sh.outs[0]) else first = try gpa.dupe(u8, sh.outs[0]);
    }
}

/// An independent reference: ((p0 + p1) + p2) + p3 in fp32, then round to nearest even by hand.
fn expectReference(parts: *const [n][row_bytes]u8, got: []const u8) !void {
    for (0..hidden) |i| {
        var acc: f32 = 0;
        for (parts, 0..) |p, r| {
            const v: f32 = @bitCast(@as(u32, std.mem.readInt(u16, p[2 * i ..][0..2], .little)) << 16);
            acc = if (r == 0) v else acc + v;
        }
        const bits: u32 = @bitCast(acc);
        const low = bits & 0xFFFF;
        var high = bits >> 16;
        if (low > 0x8000 or (low == 0x8000 and high & 1 == 1)) high += 1;
        try std.testing.expectEqual(@as(u16, @intCast(high)), std.mem.readInt(u16, got[2 * i ..][0..2], .little));
    }
}

fn mixedRank(sh: *Shared, rank: u32) void {
    mixedSteps(sh, rank) catch |err| {
        std.debug.print("rank {d}: {s}\n", .{ rank, @errorName(err) });
        sh.failed.store(true, .release);
    };
}

/// Leader plans, all-to-all with uneven chunks, gather, then a pipeline 0 -> 1 -> 2 -> 3 and a bulk copy.
fn mixedSteps(sh: *Shared, rank: u32) !void {
    const ep = sh.cluster.endpoint(rank);
    var plan_ch = collective.Channel.init(ep, channel_bytes, 4096);
    var ch = collective.Channel.init(ep, 0, slot);
    var scratch: [n][]const u8 = undefined;
    var bufs: [n][64 * 64]u8 = undefined;
    for (0..40) |s| {
        var plan: [16]u8 = undefined;
        std.mem.writeInt(u64, plan[0..8], s, .little);
        std.mem.writeInt(u64, plan[8..16], s * 3 % 17, .little);
        const got = try collective.broadcast(&plan_ch, 0, if (rank == 0) &plan else "", &scratch);
        if (std.mem.readInt(u64, got[0..8], .little) != s) return error.PlanMismatch;
        var chunks: [n][]const u8 = undefined;
        for (&chunks, &bufs, 0..) |*c, *b, p| {
            const rows = (rank * 7 + p * 3 + s) % 50 % 9;
            for (b[0 .. rows * 64], 0..) |*x, i| x.* = @truncate(rank * 31 + p * 7 + s + i);
            c.* = b[0 .. rows * 64];
        }
        try collective.allToAll(&ch, &chunks, &scratch);
        for (scratch, 0..) |got_p, p| {
            const rows = (p * 7 + rank * 3 + s) % 50 % 9;
            if (got_p.len != rows * 64) return error.SizeMismatch;
            for (got_p, 0..) |x, i| if (x != @as(u8, @truncate(p * 31 + rank * 7 + s + i))) return error.ByteMismatch;
        }
        try collective.gather(&ch, 2, &plan, &scratch);
        if (rank == 2) for (scratch) |g| if (!std.mem.eql(u8, g, &plan)) return error.GatherMismatch;
    }
    var msg: [64 << 10]u8 = undefined;
    if (rank > 0) {
        var in = collective.Pipe.init(ep, pipe_at + rank * pipe_bytes, 64 << 10, rank - 1);
        for (0..200) |k| {
            const m = try in.recv();
            if (m.len != (k * 977) % (64 << 10) or (m.len > 0 and m[m.len - 1] != @as(u8, @truncate(k)))) return error.PipeMismatch;
            if (rank < n - 1) @memcpy(msg[0..m.len], m);
            try in.release();
            if (rank < n - 1) try forward(ep, rank, msg[0..m.len], k);
        }
    } else {
        for (0..200) |k| {
            const len = (k * 977) % (64 << 10);
            if (len > 0) msg[len - 1] = @truncate(k);
            try forward(ep, rank, msg[0..len], k);
        }
    }
    if (rank == 0) {
        const big = try gpa.alloc(u8, 9 << 20);
        defer gpa.free(big);
        for (big, 0..) |*b, i| b.* = @truncate(i * 13);
        try collective.bulk(ep, 1, bulk_at, big, 2 << 20, bulk_flag, 77);
    } else if (rank == 1) {
        while (ep.local(bulk_flag) != 77) std.atomic.spinLoopHint();
        for (ep.window()[bulk_at..][0 .. 9 << 20], 0..) |b, i| if (b != @as(u8, @truncate(i * 13))) return error.BulkMismatch;
    }
}

var pipes_out: [n]?collective.Pipe = @splat(null);

fn forward(ep: Rdma, rank: u32, m: []const u8, k: usize) !void {
    if (pipes_out[rank] == null) pipes_out[rank] = collective.Pipe.init(ep, pipe_at + (rank + 1) * pipe_bytes, 64 << 10, rank + 1);
    _ = k;
    try pipes_out[rank].?.send(m);
}

test "plans, uneven all-to-all, gather, a pipeline and a bulk copy stay in lockstep under random delivery" {
    const c = try fake.Cluster.init(gpa, n, window, .queued, null);
    defer c.deinit();
    thunderbolt(c);
    var sh: Shared = .{ .cluster = c, .outs = undefined };
    try spawnRanks(&sh, mixedRank);
    try std.testing.expectError(error.AccessDenied, c.endpoint(0).fetchAdd(1, 0, 1));
}

test "an all-reduce moves each partial once to each peer: (n - 1) x bytes per rank" {
    const c = try fake.Cluster.init(gpa, n, window, .immediate, null);
    defer c.deinit();
    var sh: Shared = .{ .cluster = c, .outs = undefined };
    const total = totalBytes();
    for (&sh.outs) |*o| o.* = try gpa.alloc(u8, total);
    defer for (sh.outs) |o| gpa.free(o);
    try spawnRanks(&sh, allReduceRank);
    for (0..n) |a| for (0..n) |b| {
        if (a != b) try std.testing.expectEqual(@as(u64, total), c.bytesWritten(@intCast(a), @intCast(b)));
    };
}
