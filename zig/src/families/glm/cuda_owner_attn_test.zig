//! Owner-computed attention (cuda_owner_attn.run) on 2 and 3 host ranks, a thread each, NCCL's groups as barriers.
const std = @import("std");
const attn = @import("cuda_owner_attn.zig");

const max_rows = 64;
const max_ranks = 3;
const max_phases = 4;

fn mix(a: u64, b: u64) u64 {
    var h = std.hash.Wyhash.init(a);
    h.update(std.mem.asBytes(&b));
    return h.final();
}

/// A segment's latent rows: a function of the segment, so attending them anywhere gives the same output.
fn rowsOf(i: usize) u64 {
    return mix(0x1a7e, i);
}

/// What every rank sees: the segments, and each phase's messages from one rank to another, in posting order.
const World = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    world: usize,
    owners: []const ?u32,
    rows: []const usize,
    gen: [max_ranks]usize = @splat(0),
    mail: [max_ranks][max_ranks][max_phases]std.ArrayList([]u64) = @splat(@splat(@splat(.empty))), // [to][from][phase]
    gpa: std.mem.Allocator,
    failed: ?anyerror = null,

    fn deinit(w: *World) void {
        for (&w.mail) |*to| for (to) |*from| for (from) |*m| {
            for (m.items) |x| w.gpa.free(x);
            m.deinit(w.gpa);
        };
    }
};

/// One rank: its queries and outputs, each other rank's queries and the outputs made for it, rows a segment attended.
const Rank = struct {
    w: *World,
    rank: usize,
    qa: [max_rows]u64 = undefined,
    ol: [max_rows]u64 = @splat(0),
    pq: [max_ranks][max_rows]u64 = @splat(@splat(0)),
    po: [max_ranks][max_rows]u64 = @splat(@splat(0)),
    read: [max_rows]bool = @splat(false), // the segment rows whose latent rows this rank read
    out: std.ArrayList(struct { to: usize, rows: []u64 }) = .empty,
    want: std.ArrayList(struct { from: usize, buf: attn.Buf, seg: attn.Seg }) = .empty,

    fn buf(r: *Rank, b: attn.Buf, p: usize) *[max_rows]u64 {
        return switch (b) {
            .qa => &r.qa,
            .ol => &r.ol,
            .pq => &r.pq[p],
            .po => &r.po[p],
        };
    }
};

/// cuda_owner_attn.run's device: a host rank.
const Model = struct {
    r: *Rank,

    pub fn group(m: Model, e: attn.Edge) !void {
        const r = m.r;
        const w = r.w;
        if (e == .begin) {
            r.out.clearRetainingCapacity();
            r.want.clearRetainingCapacity();
            return;
        }
        const io = std.testing.io;
        w.mutex.lockUncancelable(io);
        defer w.mutex.unlock(io);
        const phase = w.gen[r.rank];
        for (r.out.items) |x| try w.mail[x.to][r.rank][phase].append(w.gpa, x.rows);
        r.out.clearRetainingCapacity();
        w.gen[r.rank] += 1;
        w.cond.broadcast(io);
        while (w.failed == null) {
            var behind = false;
            for (w.gen[0..w.world]) |g| behind = behind or g <= phase;
            if (!behind) break;
            w.cond.waitUncancelable(io, &w.mutex);
        }
        if (w.failed) |err| return err;
        var taken: [max_ranks]usize = @splat(0); // a group matches each pair's sends and receives in order
        for (r.want.items) |x| {
            const got = w.mail[r.rank][x.from][phase].items;
            if (taken[x.from] >= got.len) return error.UnpairedExchange;
            const rows = got[taken[x.from]];
            taken[x.from] += 1;
            if (rows.len != x.seg.rows) return error.UnpairedExchange;
            @memcpy(r.buf(x.buf, x.from)[x.seg.row0..][0..x.seg.rows], rows);
        }
        for (0..w.world) |p| if (taken[p] != w.mail[r.rank][p][phase].items.len) return error.UnpairedExchange;
    }

    pub fn move(m: Model, which: attn.Buf, p: usize, s: attn.Seg, dir: attn.Dir) !void {
        const r = m.r;
        if (p == r.rank) return error.SelfExchange;
        switch (dir) {
            .send => try r.out.append(r.w.gpa, .{ .to = p, .rows = try r.w.gpa.dupe(u64, r.buf(which, p)[s.row0..][0..s.rows]) }),
            .recv => try r.want.append(r.w.gpa, .{ .from = p, .buf = which, .seg = s }),
        }
    }

    pub fn attend(m: Model, s: attn.Seg, pass: attn.Pass) !void {
        const r = m.r;
        if (pass == .none) return; // the writes and token lists: no latent row read
        if (s.owner) |o| if (o != r.rank) return error.AttendedWithoutRows;
        for (s.row0..s.row0 + s.rows) |row| r.read[row] = true;
        const q, const o = switch (pass) {
            .own => .{ &r.qa, &r.ol },
            .peer => |p| .{ &r.pq[p], &r.po[p] },
            .none => unreachable,
        };
        for (s.row0..s.row0 + s.rows) |row| o[row] = mix(q[row], rowsOf(s.i));
    }
};

fn rankMain(r: *Rank, segs: []const attn.Seg) void {
    attn.run(Model, .{ .r = r }, segs, r.rank, r.w.world) catch |err| {
        const io = std.testing.io;
        r.w.mutex.lockUncancelable(io);
        defer r.w.mutex.unlock(io);
        if (r.w.failed == null) r.w.failed = err;
        r.w.cond.broadcast(io);
    };
}

/// `world` ranks over segments of `rows` rows owned as `owners`: each rank's outputs against its unsplit ones.
fn check(gpa: std.mem.Allocator, world: usize, owners: []const ?u32, rows: []const usize) !void {
    var w: World = .{ .world = world, .owners = owners, .rows = rows, .gpa = gpa };
    defer w.deinit();
    var ranks: [max_ranks]Rank = undefined;
    for (ranks[0..world], 0..) |*r, i| r.* = .{ .w = &w, .rank = i };
    defer for (ranks[0..world]) |*r| {
        r.out.deinit(gpa);
        r.want.deinit(gpa);
    };
    var segs: [16]attn.Seg = undefined;
    var total: usize = 0;
    for (rows, owners, 0..) |n, o, i| {
        segs[i] = .{ .row0 = total, .rows = n, .owner = o, .i = i };
        total += n;
    }
    for (ranks[0..world]) |*r| for (0..total) |row| {
        r.qa[row] = mix(r.rank + 1, row);
    };
    var threads: [max_ranks]std.Thread = undefined;
    for (threads[0..world], ranks[0..world]) |*th, *r| th.* = try std.Thread.spawn(.{}, rankMain, .{ r, segs[0..rows.len] });
    for (threads[0..world]) |th| th.join();
    if (w.failed) |err| return err;
    for (segs[0..rows.len]) |g| for (ranks[0..world]) |*r| for (g.row0..g.row0 + g.rows) |row| {
        if (r.ol[row] != mix(r.qa[row], rowsOf(g.i))) return error.WrongOutput;
        // a rank reads a segment's latent rows only where they are: every rank's unsplit, its owner's split
        if (r.read[row] != (g.owner == null or g.owner.? == r.rank)) return error.ReadElsewhere;
    };
}

const t = std.testing;

test "GLM KV split owner attention: every rank's outputs are its unsplit ones, at TP=2 and TP=3" {
    for ([_]usize{ 2, 3 }) |w| {
        try check(t.allocator, w, &.{0}, &.{8}); // a decode window on rank 0: the others give
        try check(t.allocator, w, &.{1}, &.{1});
        try check(t.allocator, w, &.{null}, &.{3}); // an unsplit stream: each rank attends its own heads
        try check(t.allocator, w, &.{ 0, 1, null, 1 }, &.{ 3, 8, 1, 5 });
        try check(t.allocator, w, &.{ 1, 1, 1 }, &.{ 2, 2, 2 });
    }
    try check(t.allocator, 3, &.{ 2, 0, 1, 2 }, &.{ 4, 4, 4, 4 });
}

test "GLM KV split owner attention: random shared windows of up to 16 segments on 2 and 3 ranks" {
    var prng = std.Random.DefaultPrng.init(0x0a11ce);
    const rnd = prng.random();
    for (0..200) |k| {
        const w: usize = 2 + k % 2;
        const n = rnd.intRangeAtMost(usize, 1, 16);
        var owners: [16]?u32 = undefined;
        var rows: [16]usize = undefined;
        var left: usize = max_rows;
        for (0..n) |i| {
            owners[i] = if (rnd.uintLessThan(u8, 5) == 0) null else rnd.uintLessThan(u32, @intCast(w));
            rows[i] = rnd.intRangeAtMost(usize, 1, @max(1, @min(8, left - (n - 1 - i))));
            left -= rows[i];
        }
        try check(t.allocator, w, owners[0..n], rows[0..n]);
    }
}

test "GLM KV split owner attention: roles, and a room's slots and bytes at TP=2 and TP=3" {
    try t.expectEqual(attn.Role.local, attn.role(null, 0));
    try t.expectEqual(attn.Role.take, attn.role(1, 1));
    try t.expectEqual(attn.Role.give, attn.role(1, 0));
    try t.expect(!attn.exchanges(&.{ .{ .row0 = 0, .rows = 1, .owner = null }, .{ .row0 = 1, .rows = 2, .owner = null } }));
    try t.expect(attn.exchanges(&.{ .{ .row0 = 0, .rows = 1, .owner = null }, .{ .row0 = 1, .rows = 2, .owner = 0 } }));
    const c: @import("config.zig").Config = .{};
    try t.expectEqual(@as(usize, 2 * 4096 * 32 * 512 * 2), try attn.Room.bytes(&c, 2, 4096)); // 256 MiB at TP=2
    try t.expectEqual(@as(usize, 2 * 2 * 4096 * 22 * 512 * 2), try attn.Room.bytes(&c, 3, 4096)); // TP=3: 22 heads a slot
    var r: attn.Room = .{ .buf = .{ .d = undefined, .ptr = 1 << 32, .len = 0 }, .rank = 1, .world = 3, .rows = 10, .lw = 512, .heads = @splat(0), .lib = undefined, .comm = undefined };
    r.heads[0..3].* = .{ 22, 21, 21 };
    const slot = 10 * 22 * 512 * 2;
    try t.expectEqual(@as(u64, 1 << 32), r.at(.pq, 0, 0));
    try t.expectEqual(@as(u64, (1 << 32) + 3 * 22 * 1024), r.at(.pq, 0, 3)); // rank 0's rows, 22 heads each
    try t.expectEqual(@as(u64, (1 << 32) + slot + 3 * 21 * 1024), r.at(.pq, 2, 3)); // rank 2's slot, 21 heads a row
    try t.expectEqual(@as(u64, (1 << 32) + 2 * slot), r.at(.po, 0, 0)); // the outputs after every slot of queries
    try t.expectEqual(@as(u64, (1 << 32) + 3 * slot + 21 * 1024), r.at(.po, 2, 1));
}

test "GLM KV split owner attention: a segment past the room is refused before any exchange, a whole window never" {
    const r: attn.Room = .{ .buf = .{ .d = undefined, .ptr = 1 << 32, .len = 0 }, .rank = 1, .world = 3, .rows = 10, .lw = 512, .heads = @splat(0), .lib = undefined, .comm = undefined };
    try r.holds(&.{ .{ .row0 = 0, .rows = 4, .owner = 0 }, .{ .row0 = 4, .rows = 6, .owner = 2 }, .{ .row0 = 10, .rows = 90, .owner = null } });
    try std.testing.expectError(error.KvSplitRoom, r.holds(&.{ .{ .row0 = 0, .rows = 4, .owner = 0 }, .{ .row0 = 4, .rows = 7, .owner = 2 } }));
    try std.testing.expectError(error.KvSplitRoom, r.holds(&.{.{ .row0 = 10, .rows = 1, .owner = 1 }}));
}

test "GLM KV split owner attention: each guard's mutant is caught" {
    defer attn.mutant = .none;
    inline for (.{ attn.Mutant.peer_skipped, attn.Mutant.output_into_qa }) |m| {
        attn.mutant = m;
        try t.expectError(error.WrongOutput, check(t.allocator, 2, &.{ 0, 1 }, &.{ 4, 4 }));
    }
    attn.mutant = .output_to_owner; // outputs sent back to the owner itself, not the rank whose heads they are
    try t.expectError(error.SelfExchange, check(t.allocator, 3, &.{ 0, 1 }, &.{ 4, 4 }));
}
