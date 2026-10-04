//! Four ranks over the two-sided fake with random delivery: collectives exact, at most two messages buffered per pair.
const std = @import("std");
const fabric = @import("fabric");
const collective = fabric.collective;
const sr = fabric.sendrecv;
const Fake = fabric.sendrecv_fake.Cluster;

const gpa = std.testing.allocator;
const n = 4;
const hidden = 7168;
const row_bytes = hidden * 2;
const max_rows = 300;

fn partialRow(rank: u32, id: u64, out: []u8) void {
    var prng: std.Random.DefaultPrng = .init(id *% 1_000_003 +% rank);
    const r = prng.random();
    for (0..out.len / 2) |i| {
        const v = (r.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(r.intRangeAtMost(i32, -6, 6)));
        std.mem.writeInt(u16, out[2 * i ..][0..2], collective.toBf16(v), .little);
    }
}

fn rowsOf(s: u64) usize {
    const counts = [_]usize{ 1, 7, 64, 300, 2, 129, 1, 33 };
    return counts[s % counts.len];
}

const Shared = struct {
    cluster: *Fake,
    outs: [n][]u8,
    x: [n]sr.Exchange = undefined,
    failed: std.atomic.Value(bool) = .init(false),
};

fn runRank(sh: *Shared, r: u32) void {
    steps(sh, r) catch |err| {
        std.debug.print("rank {d}: {s}\n", .{ r, @errorName(err) });
        sh.failed.store(true, .release);
    };
}

/// Plans, all-reduces of 1-300 rows, uneven all-to-alls and gathers, interleaved, all on one Exchange.
fn steps(sh: *Shared, r: u32) !void {
    const x = &sh.x[r];
    const partial = try gpa.alloc(u8, max_rows * row_bytes);
    defer gpa.free(partial);
    var scratch: [n][]const u8 = undefined;
    var bufs: [n][9 * 64]u8 = undefined;
    var at: usize = 0;
    for (0..24) |s| {
        var plan: [8]u8 = undefined;
        std.mem.writeInt(u64, &plan, s, .little);
        const got = try collective.broadcast(x, 0, if (r == 0) &plan else "", &scratch);
        if (std.mem.readInt(u64, got[0..8], .little) != s) return error.PlanMismatch;
        const rows = rowsOf(s);
        for (0..rows) |j| partialRow(r, s * 1000 + j, partial[j * row_bytes ..][0..row_bytes]);
        const len = rows * row_bytes;
        try collective.allReduce(x, .bf16, partial[0..len], sh.outs[r][at..][0..len], &scratch);
        at += len;
        var chunks: [n][]const u8 = undefined;
        for (&chunks, &bufs, 0..) |*c, *b, p| {
            const k = (r * 7 + p * 3 + s) % 9;
            for (b[0 .. k * 64], 0..) |*v, i| v.* = @truncate(r * 31 + p * 7 + s + i);
            c.* = b[0 .. k * 64];
        }
        try collective.allToAll(x, &chunks, &scratch);
        for (scratch, 0..) |g, p| {
            const k = (p * 7 + r * 3 + s) % 9;
            if (g.len != k * 64) return error.SizeMismatch;
            for (g, 0..) |v, i| if (v != @as(u8, @truncate(p * 31 + r * 7 + s + i))) return error.ByteMismatch;
        }
        try collective.barrier(x, &scratch);
    }
}

fn totalBytes() usize {
    var t: usize = 0;
    for (0..24) |s| t += rowsOf(s) * row_bytes;
    return t;
}

test "two-sided collectives stay exact under random delivery with at most two messages buffered per pair" {
    var first: ?[]u8 = null;
    defer if (first) |f| gpa.free(f);
    for (0..2) |run| {
        const c = try Fake.init(gpa, n, true);
        defer c.deinit();
        c.prng = .init(run + 3);
        var sh: Shared = .{ .cluster = c, .outs = undefined };
        for (&sh.outs) |*o| o.* = try gpa.alloc(u8, totalBytes());
        defer for (sh.outs) |o| gpa.free(o);
        for (&sh.x, 0..) |*x, r| x.* = sr.Exchange.init(c.link(@intCast(r)));
        var stop: std.atomic.Value(bool) = .init(false);
        const pump = try std.Thread.spawn(.{}, Fake.pump, .{ c, &stop });
        var threads: [n]std.Thread = undefined;
        for (&threads, 0..) |*t, r| t.* = try std.Thread.spawn(.{}, runRank, .{ &sh, @as(u32, @intCast(r)) });
        for (threads) |t| t.join();
        stop.store(true, .release);
        pump.join();
        try std.testing.expect(!sh.failed.load(.acquire));
        try std.testing.expect(c.peak <= 2);
        for (sh.outs[1..]) |o| try std.testing.expectEqualSlices(u8, sh.outs[0], o);
        var parts: [n][row_bytes]u8 = undefined;
        var solo: [row_bytes]u8 = undefined;
        var at: usize = 0;
        for (0..24) |s| for (0..rowsOf(s)) |j| {
            for (&parts, 0..) |*p, r| partialRow(@intCast(r), s * 1000 + j, p);
            collective.reduce(.bf16, &.{ &parts[0], &parts[1], &parts[2], &parts[3] }, &solo);
            try std.testing.expectEqualSlices(u8, &solo, sh.outs[0][at..][0..row_bytes]);
            at += row_bytes;
        };
        if (first) |f| try std.testing.expectEqualSlices(u8, f, sh.outs[0]) else first = try gpa.dupe(u8, sh.outs[0]);
    }
}
