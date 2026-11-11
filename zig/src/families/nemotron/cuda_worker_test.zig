//! Nemotron TP jobs over a fake link: rank 0's real posts, rank 1's real loop, host memory standing in for devices.

const std = @import("std");
const worker = @import("cuda_worker.zig");
const Config = @import("config.zig").Config;
const latent = @import("cuda_latent.zig");
const fwd = @import("cuda_forward.zig");
const W = @import("cuda_weights.zig");

const t = std.testing;
const max_rows = 32;

/// One rank's device memory: an address is an offset into `mem`.
const Dev = struct {
    mem: []u8,

    pub fn fill32(d: *Dev, dst: u64, value: u32, count: usize) !void {
        for (0..count) |i| std.mem.writeInt(u32, d.mem[dst + i * 4 ..][0..4], value, .little);
    }

    pub fn copy(d: *Dev, dst: u64, src: u64, bytes: usize) !void {
        @memmove(d.mem[dst..][0..bytes], d.mem[src..][0..bytes]);
    }
};

/// Rank 0's side of the link: each send puts a copy of its bytes on the wire, in order.
const Sender = struct {
    dev: *Dev,
    wire: *std.ArrayList([]u8),

    pub fn send(s: Sender, ptr: u64, bytes: usize) !void {
        try s.wire.append(t.allocator, try t.allocator.dupe(u8, s.dev.mem[ptr..][0..bytes]));
    }

    /// Rank 1's reply, which this wire does not carry back.
    pub fn recv(_: Sender, _: u64, _: usize) !void {}
};

/// The ranks' liveness line: the name a rank stopped with, which every other rank hears and stops with.
const Line = struct {
    name: ?[]const u8 = null,

    pub fn stop(l: *Line, name: []const u8) void {
        l.name = name;
    }
};

/// What rank 1 ran for a job: its layer, rows, where its sections were, and their bytes.
const Seen = struct { layer: u32, rows: usize, at: [3]u64, bytes: [3][]u8 };

/// Rank 1's side: a receive takes the next message and must be its size, as an NCCL receive must.
const Follower = struct {
    dev: *Dev,
    wire: []const []u8,
    c: Config,
    line: ?*const Line = null,
    next: usize = 0,
    seen: std.ArrayList(Seen) = .empty,
    heard: ?[]const u8 = null,

    pub fn recv(f: *Follower, ptr: u64, bytes: usize) !void {
        if (f.next == f.wire.len) { // nothing more comes: rank 1 waits until its line says rank 0 stopped
            f.heard = if (f.line) |l| l.name else null;
            return if (f.heard != null) error.RankStopped else error.NothingSent;
        }
        const m = f.wire[f.next];
        f.next += 1;
        if (m.len != bytes) return error.SizeMismatch;
        @memcpy(f.dev.mem[ptr..][0..bytes], m);
    }

    pub fn read(f: *Follower, ptr: u64) ![worker.header_words]u32 {
        var h: [worker.header_words]u32 = undefined;
        for (&h, 0..) |*w, i| w.* = std.mem.readInt(u32, f.dev.mem[ptr + i * 4 ..][0..4], .little);
        return h;
    }

    pub fn run(f: *Follower, layer: u32, at: [3]u64, rows: usize) !void {
        const c = f.c;
        var bytes: [3][]u8 = undefined;
        for (&bytes, at, [3]usize{ rows * c.top_k * 4, rows * c.latent * 2, rows * c.hidden * 2 }) |*b, p, n| b.* = try t.allocator.dupe(u8, f.dev.mem[p..][0..n]);
        try f.seen.append(t.allocator, .{ .layer = layer, .rows = rows, .at = at, .bytes = bytes });
    }

    fn deinit(f: *Follower) void {
        for (f.seen.items) |s| for (s.bytes) |b| t.allocator.free(b);
        f.seen.deinit(t.allocator);
    }
};

/// Super's job geometry: 22 picks a row, a 1,024-wide latent, 4,096-wide hidden rows.
fn superConfig() Config {
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    return c;
}

/// Super's 88 blocks: 40 latent MoE layers (odd ones here) hold tables; rank 1 names each by its layer.
const layers = 88;
const moe_ids = blk: {
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*id, k| id.* = 2 * k + 1;
    break :blk ids;
};
const tables = blk: {
    var b: [layers]?u32 = @splat(null);
    for (moe_ids) |id| b[id] = id;
    break :blk b;
};

/// Rank 0 and rank 1 over one wire: rank 0 posts its jobs and its end, then rank 1 follows them all.
const Pair = struct {
    c: Config,
    sender: Dev,
    receiver: Dev,
    wire: std.ArrayList([]u8) = .empty,
    follower: Follower,

    const job: u64 = 4 << 20; // rank 0's job buffer, past its picks, latent and hidden rows
    const long: u64 = 1 << 20; // rank 1's long-job region, past its first message

    fn init(c: Config) !Pair {
        const send_bytes = job + worker.firstBytes(c) + worker.payloadBytes(c, max_rows);
        const sender = try t.allocator.alloc(u8, send_bytes);
        errdefer t.allocator.free(sender);
        const receiver = try t.allocator.alloc(u8, long + worker.payloadBytes(c, max_rows));
        return .{ .c = c, .sender = .{ .mem = sender }, .receiver = .{ .mem = receiver }, .follower = .{ .dev = undefined, .wire = &.{}, .c = c } };
    }

    fn deinit(p: *Pair) void {
        for (p.wire.items) |m| t.allocator.free(m);
        p.wire.deinit(t.allocator);
        p.follower.deinit();
        t.allocator.free(p.sender.mem);
        t.allocator.free(p.receiver.mem);
    }

    /// Rank 0's rows for a job: bytes that differ by job, section and place.
    fn from(p: *Pair, k: usize, rows: usize) [3]u64 {
        const c = p.c;
        const at = [3]u64{ 0, 1 << 20, 2 << 20 };
        for (at, [3]usize{ rows * c.top_k * 4, rows * c.latent * 2, rows * c.hidden * 2 }, 0..) |a, n, s| {
            for (p.sender.mem[a..][0..n], 0..) |*b, i| b.* = @truncate(k *% 131 +% s *% 17 +% i *% 7 +% i / 251);
        }
        return at;
    }

    fn post(p: *Pair, ids: []const u32, rows: usize, pass: ?worker.Pass) !void {
        const link: Sender = .{ .dev = &p.sender, .wire = &p.wire };
        for (ids, 0..) |id, k| try worker.post(&p.sender, link, p.c, pass, null, job, p.from(k, rows), id, rows);
    }

    fn follow(p: *Pair) !worker.Count {
        try worker.postEnd(&p.sender, Sender{ .dev = &p.sender, .wire = &p.wire }, p.c, job);
        p.follower = .{ .dev = &p.receiver, .wire = p.wire.items, .c = p.c };
        const n = try worker.follow(&p.follower, p.c, @as([]const ?u32, &tables), @as(?u32, latent.mtp_id), 0, long);
        try t.expectEqual(p.wire.items.len, p.follower.next); // every message taken, none left over
        return n;
    }
};

/// The 40 layers' jobs at `rows` rows, under a pass or a header each; rank 1's counts.
fn window(rows: usize, pass: bool) !worker.Count {
    var p = try Pair.init(superConfig());
    defer p.deinit();
    try p.post(&moe_ids, rows, if (pass) worker.Pass.of(&moe_ids).? else null);
    return p.follow();
}

test "Nemotron TP jobs: a window is one message a layer and one header read" {
    for ([_]usize{ 1, 4, max_rows }) |rows| {
        const long: usize = @intFromBool(rows > worker.small_rows);
        const one = try window(rows, false); // a header a job: two messages a layer past one row, a read each
        try t.expectEqual(worker.Count{ .jobs = 40, .messages = 40 * (1 + long) + 1, .reads = 41 }, one);
        const pass = try window(rows, true); // the pass's header rides with its first job, the end has its own
        try t.expectEqual(worker.Count{ .jobs = 40, .messages = 40 + long + 1, .reads = 2 }, pass);
    }
}

test "Nemotron TP jobs: rank 1 runs the same layers, rows, places and bytes under a pass" {
    for ([_]usize{ 1, 4, max_rows }) |rows| {
        var runs: [2]Pair = undefined;
        for (&runs, [2]bool{ false, true }) |*p, pass| {
            p.* = try Pair.init(superConfig());
            try p.post(&moe_ids, rows, if (pass) worker.Pass.of(&moe_ids).? else null);
            _ = try p.follow();
        }
        defer for (&runs) |*p| p.deinit();
        const one = runs[0].follower.seen.items;
        const pass = runs[1].follower.seen.items;
        try t.expectEqual(moe_ids.len, pass.len);
        for (one, pass, moe_ids, 0..) |a, b, id, k| {
            try t.expectEqual(id, b.layer);
            try t.expectEqual(a.layer, b.layer);
            try t.expectEqual(rows, b.rows);
            try t.expectEqual(a.at, b.at); // where a job of its size always landed
            var want = try Pair.init(superConfig());
            defer want.deinit();
            const src = want.from(k, rows);
            for (a.bytes, b.bytes, src) |x, y, s| {
                try t.expectEqualSlices(u8, want.sender.mem[s..][0..x.len], y);
                try t.expectEqualSlices(u8, x, y);
            }
        }
    }
}

/// The wire of one pass at `rows` rows over `ids`, then its end.
fn passWire(p: *Pair, ids: []const u32, rows: usize) !void {
    try p.post(ids, rows, worker.Pass.of(ids).?);
    try worker.postEnd(&p.sender, Sender{ .dev = &p.sender, .wire = &p.wire }, p.c, Pair.job);
}

fn headerOf(m: []const u8) [worker.header_words]u32 {
    var h: [worker.header_words]u32 = undefined;
    for (&h, 0..) |*w, i| w.* = std.mem.readInt(u32, m[i * 4 ..][0..4], .little);
    return h;
}

test "Nemotron TP jobs: a short or malformed header is refused" {
    var p = try Pair.init(superConfig());
    defer p.deinit();
    try passWire(&p, &moe_ids, 4);
    const good = headerOf(p.wire.items[0]);
    try t.expectEqual(@as(usize, 4), (try worker.parse(&good, layers)).pass.rows);
    try t.expectEqual(@as(worker.Header, .end), try worker.parse(&headerOf(p.wire.items[p.wire.items.len - 1]), layers));
    try t.expectError(error.BadJob, worker.parse(good[0 .. worker.header_words - 1], layers)); // short
    const edits = [_]struct { usize, u32 }{
        .{ 0, 0x4a4f4253 }, // a pass's words under a job's magic: kind 3 is no job's
        .{ 0, 0 }, // no magic
        .{ 1, 1 }, // a pass's magic on a job's kind
        .{ 2, 39 }, // one job fewer than its layers
        .{ 2, 41 }, // one more
        .{ 3, 0 }, // no rows
        .{ 3, 2 * 2048 + 1 }, // more rows than a pass holds
        .{ 4, good[4] | 1 }, // layer 0 added without its job
        .{ 6, good[6] | 1 << 24 }, // layer 88, past the model
    };
    for (edits) |e| {
        var h = good;
        h[e[0]] = e[1];
        try t.expectError(error.BadJob, worker.parse(&h, layers));
    }
    var none = good;
    none[2] = 0;
    for (none[4..7]) |*m| m.* = 0;
    try t.expectError(error.BadJob, worker.parse(&none, layers)); // no jobs at all
    try t.expectEqual(@as(?worker.Pass, null), worker.Pass.of(&.{ 3, 3 })); // ids must rise
    try t.expectEqual(@as(?worker.Pass, null), worker.Pass.of(&.{ 5, 1 }));
    try t.expectEqual(@as(?worker.Pass, null), worker.Pass.of(&.{96}));
    try t.expectEqual(@as(?worker.Pass, null), worker.Pass.of(&.{}));
    // a layer rank 1 holds no tables for (a Mamba block): refused before any of the pass's jobs is taken
    var q = try Pair.init(superConfig());
    defer q.deinit();
    try passWire(&q, &.{ 1, 2, 3 }, 1);
    q.follower = .{ .dev = &q.receiver, .wire = q.wire.items, .c = q.c };
    try t.expectError(error.BadJob, worker.follow(&q.follower, q.c, @as([]const ?u32, &tables), @as(?u32, null), 0, Pair.long));
    try t.expectEqual(@as(usize, 1), q.follower.next);
    try t.expectEqual(@as(usize, 0), q.follower.seen.items.len);
}

test "Nemotron TP jobs: prompt jobs and the MTP head keep a header each, as before passes" {
    const c = superConfig();
    for ([_]usize{ 1, 7, max_rows }) |rows| {
        var p = try Pair.init(c);
        defer p.deinit();
        const pass = worker.Pass.of(&moe_ids).?;
        try p.post(&.{3}, rows, null); // a prompt chunk's job
        try p.post(&.{latent.mtp_id}, rows, pass); // the MTP head's job beside a window's pass
        const long = rows > worker.small_rows;
        try t.expectEqual(@as(usize, if (long) 4 else 2), p.wire.items.len);
        for ([_]u32{ 3, latent.mtp_id }, 0..) |id, k| {
            const at = k * @as(usize, if (long) 2 else 1);
            const first = p.wire.items[at];
            try t.expectEqual(worker.firstBytes(c), first.len);
            try t.expectEqualSlices(u32, &.{ 0x4a4f4253, 1, id, @intCast(rows) }, headerOf(first)[0..4]);
            if (long) try t.expectEqual(worker.payloadBytes(c, rows), p.wire.items[at + 1].len);
            const rows_msg = if (long) p.wire.items[at + 1] else first[64..];
            var want = try Pair.init(c);
            defer want.deinit();
            const src = want.from(0, rows); // each post's first job
            try t.expectEqualSlices(u8, want.sender.mem[src[0]..][0 .. rows * c.top_k * 4], rows_msg[0 .. rows * c.top_k * 4]);
        }
        const n = try p.follow();
        try t.expectEqual(worker.Count{ .jobs = 2, .messages = if (long) 5 else 3, .reads = 3 }, n);
        try t.expectEqual(@as(u32, latent.mtp_id), p.follower.seen.items[1].layer);
    }
}

test "Nemotron TP jobs: a job outside the pass's layers takes its own header" {
    var p = try Pair.init(superConfig());
    defer p.deinit();
    const pass = worker.Pass.of(moe_ids[0..3]).?; // layers 1, 3 and 5
    try p.post(moe_ids[0..3], 4, pass);
    try p.post(&.{7}, 4, pass);
    const n = try p.follow();
    try t.expectEqual(worker.Count{ .jobs = 4, .messages = 3 + 1 + 2 + 1, .reads = 3 }, n); // a pass, a job, the end
    for (p.follower.seen.items, [_]u32{ 1, 3, 5, 7 }) |s, id| try t.expectEqual(id, s.layer);
}

test "Nemotron TP jobs: a window over two ranks takes its latent layers' pass, other forwards none" {
    var blocks: [layers]W.Block = undefined;
    for (&blocks, 0..) |*b, i| b.* = .{ .kind = if (i % 2 == 1) .moe else .mamba, .norm = 0 };
    for (moe_ids) |id| blocks[id].moe = .{ .router = 0, .bias = 0, .experts = undefined, .latent = .{ .into = undefined, .out = undefined, .t = undefined, .id = id } };
    blocks[79].moe.latent = null; // a MoE layer rank 0 runs alone sends no job
    const w: W.Weights = .{ .gpa = t.allocator, .config = superConfig(), .blocks = &blocks };
    const group: @import("cuda").tp_group.Group = undefined;
    var f: fwd.Forward = .{ .c = superConfig(), .w = &w, .b = undefined, .ops = undefined, .glue = undefined, .max_len = 0, .nch = 0, .sampled = false, .tp = &group };
    const segs = [_]fwd.Seg{.{ .b = undefined, .row0 = 0, .rows = 4, .sampled = false }};
    try t.expectEqual(worker.Pass.of(moe_ids[0..39]), f.routedWindow(&segs).pass);
    try t.expectEqual(@as(?worker.Pass, null), f.pass); // a prompt chunk's forward keeps a header a job
    f.tp = null;
    try t.expectEqual(@as(?worker.Pass, null), f.routedWindow(&segs).pass);
}

/// Rank 0 runs a window over the 40 layers at `rows` rows and fails before job `fail` goes out, after `back` replies.
fn failWindow(p: *Pair, line: *Line, rows: usize, fail: usize, back: usize) !void {
    const pass = worker.Pass.of(&moe_ids).?;
    var flight: worker.Flight = .{ .jobs = pass.jobs };
    const link: Sender = .{ .dev = &p.sender, .wire = &p.wire };
    for (moe_ids[0..fail], 0..) |id, k| {
        try worker.post(&p.sender, link, p.c, pass, &flight, Pair.job, p.from(k, rows), id, rows);
        if (k < back) try worker.collect(link, &flight, 0, 0);
    }
    worker.settle(flight, false, "CudaLaunchFailed", line);
    if (line.name == null) try worker.postEnd(&p.sender, link, p.c, Pair.job); // rank 0 carried on: its engine closes
    p.follower = .{ .dev = &p.receiver, .wire = p.wire.items, .c = p.c, .line = line };
}

test "Nemotron TP jobs: a window failing inside its jobs stops both ranks by its name, nothing misread" {
    for ([_]usize{ 1, 4 }) |rows| for ([_][2]usize{ .{ 1, 1 }, .{ 17, 17 }, .{ 39, 39 }, .{ 40, 39 } }) |at| {
        var p = try Pair.init(superConfig());
        defer p.deinit();
        var line: Line = .{};
        try failWindow(&p, &line, rows, at[0], at[1]);
        const got = worker.follow(&p.follower, p.c, @as([]const ?u32, &tables), @as(?u32, latent.mtp_id), 0, Pair.long);
        try t.expectError(error.RankStopped, got); // not a later message read as the window's rows
        try t.expectEqualStrings("CudaLaunchFailed", line.name.?);
        try t.expectEqualStrings("CudaLaunchFailed", p.follower.heard.?);
        try t.expectEqual(p.wire.items.len, p.follower.next); // rank 0 sent nothing after its failure
        try t.expectEqual(at[0], p.follower.seen.items.len);
        var want = try Pair.init(superConfig());
        defer want.deinit();
        for (p.follower.seen.items, moe_ids[0..at[0]], 0..) |s, id, k| {
            try t.expectEqual(id, s.layer);
            for (s.bytes, want.from(k, rows)) |got_bytes, src| try t.expectEqualSlices(u8, want.sender.mem[src..][0..got_bytes.len], got_bytes);
        }
    };
}

test "Nemotron TP jobs: a window failing before its first job, after its last reply or while captured stops no rank" {
    for ([_][2]usize{ .{ 0, 0 }, .{ 40, 40 } }) |at| {
        var p = try Pair.init(superConfig());
        defer p.deinit();
        var line: Line = .{};
        try failWindow(&p, &line, 4, at[0], at[1]);
        try t.expectEqual(@as(?[]const u8, null), line.name);
        const n = try worker.follow(&p.follower, p.c, @as([]const ?u32, &tables), @as(?u32, latent.mtp_id), 0, Pair.long);
        try t.expectEqual(at[0], n.jobs); // rank 1 is back at a header, which the end header fills
    }
    var line: Line = .{};
    worker.settle(.{ .jobs = 40, .sent = 3, .received = 2 }, true, "CudaLaunchFailed", &line); // a capture sent nothing
    try t.expectEqual(@as(?[]const u8, null), line.name);
    worker.settle(.{ .sent = 1 }, false, "CudaLaunchFailed", &line); // a header a job, its reply owed
    try t.expectEqualStrings("CudaLaunchFailed", line.name.?);
}
