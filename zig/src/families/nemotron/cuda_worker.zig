//! TP over hosts for a latent MoE: rank 0 runs the model and asks rank 1, which holds only its expert shares, a layer.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const Kind = @import("config.zig").Kind;
const kern = @import("cuda_kernels.zig");
const W = @import("cuda_weights.zig");
const latent = @import("cuda_latent.zig");
const identity = core.checkpoint_identity;
const fwd = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const Source = @import("cuda_source.zig").Source;
const Engine = @import("cuda_engine.zig").Engine;

/// Rows a job's first message carries after its header (a decode step's one row); more go in a second message.
pub const small_rows = 1;
const magic: u32 = 0x4a4f4253;
const pass_magic: u32 = 0x53534150; // a window's header: a build without window passes refuses it as a bad job
const header_bytes = 64;
pub const header_words = header_bytes / 4;
const max_layers = @import("config.zig").max_layers;
const mask_words = (max_layers + 31) / 32;
const Job = enum(u32) { rows = 1, end = 2, pass = 3 };

comptime {
    std.debug.assert(4 + mask_words <= header_words); // a pass's header: magic, kind, jobs, rows, then its layer mask
}

/// A window's jobs under one header: a latent MoE layer each, in rising layer order, all at its rows, none between.
pub const Pass = struct {
    mask: [mask_words]u32 = @splat(0),
    jobs: u32 = 0,
    first: u32 = 0,

    /// The pass over layer ids in the order rank 0 runs them; null for none, or for ids that do not rise.
    pub fn of(ids: []const u32) ?Pass {
        if (ids.len == 0) return null;
        var p: Pass = .{ .jobs = @intCast(ids.len), .first = ids[0] };
        for (ids, 0..) |id, k| {
            if (id >= max_layers or (k > 0 and id <= ids[k - 1])) return null;
            p.mask[id / 32] |= @as(u32, 1) << @intCast(id % 32);
        }
        return p;
    }

    /// The pass over a forward's latent MoE blocks, in block order.
    pub fn blocks(list: []const W.Block) ?Pass {
        var ids: [max_layers]u32 = undefined;
        var n: usize = 0;
        for (list) |b| {
            if (b.kind != .moe) continue;
            const lt = b.moe.latent orelse continue;
            if (n == ids.len) return null;
            ids[n] = lt.id;
            n += 1;
        }
        return of(ids[0..n]);
    }

    pub fn has(p: Pass, id: u32) bool {
        return id < max_layers and bit(p.mask, id);
    }

    fn header(p: Pass, rows: usize, out: *[4 + mask_words]u32) []const u32 {
        out.* = [4]u32{ pass_magic, @backingInt(Job.pass), p.jobs, @intCast(rows) } ++ p.mask;
        return out;
    }
};

fn bit(mask: [mask_words]u32, i: usize) bool {
    return mask[i / 32] >> @intCast(i % 32) & 1 != 0;
}

/// Byte offsets of a job's sections, `per` rows each, 256 apart: picks (int32), latent rows (bf16), hidden rows (bf16).
fn sections(c: Config, per: usize) [3]usize {
    const picks = std.mem.alignForward(usize, per * c.top_k * 4, 256); // rows stay 16-byte aligned for odd counts
    return .{ 0, picks, picks + std.mem.alignForward(usize, per * c.latent * 2, 256) };
}

/// Bytes of a job's sections for `rows` rows.
pub fn payloadBytes(c: Config, rows: usize) usize {
    return sections(c, rows)[2] + rows * c.hidden * 2;
}

/// Bytes of a job's first message: the header, then a small job's sections.
pub fn firstBytes(c: Config) usize {
    return header_bytes + payloadBytes(c, small_rows);
}

/// Rank 0's job buffer: the first message, then a long job's sections.
pub fn jobBytes(c: Config) usize {
    return firstBytes(c) + payloadBytes(c, state.pass_rows);
}

/// One peer's NCCL line on one stream.
const Link = struct {
    g: *const cuda.tp_group.Group,
    s: cuda.Stream,
    peer: usize,

    pub fn send(l: Link, ptr: u64, bytes: usize) !void {
        return l.g.send(l.s, ptr, bytes, l.peer);
    }

    pub fn recv(l: Link, ptr: u64, bytes: usize) !void {
        return l.g.recv(l.s, ptr, bytes, l.peer);
    }
};

/// Rank 0: layer `id`'s job of `rows` rows to rank 1, its picks, latent rows and hidden rows.
pub fn send(f: *const fwd.Forward, id: u32, rows: usize) !void {
    const tp = f.tp.?;
    if (tp.world != 2) return error.TensorParallelNotBuilt;
    const b = f.b;
    try post(f.ops, Link{ .g = tp, .s = f.ops.s, .peer = 1 }, f.c, f.pass, f.flight, b.job, .{ b.pick, b.lat, b.y }, id, rows);
}

/// A window's jobs on rank 0: rank 1 waits inside it from its first job out until its pass's (or a job's) last reply.
pub const Flight = struct {
    jobs: u32 = 0, // its pass's jobs (0: a header a job)
    sent: u32 = 0,
    received: u32 = 0,

    pub fn open(x: Flight) bool {
        return x.received < x.sent or (x.sent > 0 and x.received < x.jobs);
    }
};

/// A window failed with `name`: while rank 1 waits inside it, every rank stops through `stop`, unless it was captured.
pub fn settle(flight: Flight, captured: bool, name: []const u8, stop: anytype) void {
    if (flight.open() and !captured) stop.stop(name);
}

/// Rank 0's window failed with `name`: a capture sent nothing, else rank 1 may wait on its jobs (NCCL leaves it hung).
pub fn windowFailed(s: cuda.Stream, flight: Flight, name: []const u8) void {
    if (!flight.open()) return;
    settle(flight, (cuda.graph.captureStatus(s) catch .none) != .none, name, Stop{});
}

/// Every rank hears `name` on its liveness line, then this process ends (its stream may wait on rank 1).
const Stop = struct {
    fn stop(_: Stop, name: []const u8) noreturn {
        std.log.err("rank 0: a window failed ({s}) while rank 1 waited on its jobs; every rank stops", .{name});
        cuda.tp_watch.fail(name);
        cuda.tp_watch.stop();
    }
};

/// Rank 0's job through `job`'s buffer (`o` fills and copies, `link` sends): a pass's later jobs send their rows alone.
pub fn post(o: anytype, link: anytype, c: Config, pass: ?Pass, flight: ?*Flight, job: u64, from: [3]u64, id: u32, rows: usize) !void {
    const in_pass = if (pass) |p| p.has(id) else false;
    const later = in_pass and id != pass.?.first; // rank 1 has this job's size from the pass's header
    if (!later) {
        var out: [4 + mask_words]u32 = undefined;
        out[0..4].* = .{ magic, @backingInt(Job.rows), id, @intCast(rows) };
        const words: []const u32 = if (in_pass) pass.?.header(rows, &out) else out[0..4];
        for (words, 0..) |v, i| try o.fill32(job + i * 4, v, 1);
    }
    const small = rows <= small_rows;
    const base: u64 = if (small) job + header_bytes else job + firstBytes(c);
    const at = sections(c, if (small) small_rows else rows);
    try o.copy(base + at[0], from[0], rows * c.top_k * 4);
    try o.copy(base + at[1], from[1], rows * c.latent * 2);
    try o.copy(base + at[2], from[2], rows * c.hidden * 2);
    if (later) {
        try link.send(base, payloadBytes(c, if (small) small_rows else rows));
    } else {
        try link.send(job, firstBytes(c));
        if (!small) try link.send(base, payloadBytes(c, rows));
    }
    if (flight) |x| x.sent += 1; // out: a failed NCCL post stops every rank itself (nccl.zig)
}

/// Rank 0's end header through `job`'s buffer.
pub fn postEnd(o: anytype, link: anytype, c: Config, job: u64) !void {
    for ([4]u32{ magic, @backingInt(Job.end), 0, 0 }, 0..) |v, i| try o.fill32(job + i * 4, v, 1);
    try link.send(job, firstBytes(c));
}

/// Rank 0: rank 1's expert rows for its job, into rank 1's region of ymoe.
pub fn receive(f: *const fwd.Forward, rows: usize) !void {
    const region: u64 = latent.partFloats(f.c, state.pass_rows) * 4;
    try collect(Link{ .g = f.tp.?, .s = f.ops.s, .peer = 1 }, f.flight, f.b.ymoe + region, latent.partFloats(f.c, rows) * 4);
}

/// Rank 0: a job's reply from rank 1 into `ptr`, counted in `flight`.
pub fn collect(link: anytype, flight: ?*Flight, ptr: u64, bytes: usize) !void {
    try link.recv(ptr, bytes);
    if (flight) |x| x.received += 1;
}

/// Rank 0 as its engine closes: the end header, so rank 1 leaves its loop.
pub fn end(e: *Engine) !void {
    cuda.tp_watch.ended(); // the other ranks' lines close on purpose from here
    const o = e.ops();
    try postEnd(o, Link{ .g = &e.tp.?, .s = o.s, .peer = 1 }, e.c, e.b.job);
    try e.stream.synchronize();
}

/// What a header asks of rank 1: the end, one job, or a window pass's jobs in rising layer order.
pub const Header = union(enum) {
    end,
    one: struct { id: u32, rows: usize },
    pass: struct { rows: usize, mask: [mask_words]u32 },
};

/// A header rank 1 read, for a model of `layers` layers; a short or malformed one is a bad job.
pub fn parse(h: []const u32, layers: usize) !Header {
    if (h.len < header_words) return error.BadJob;
    const rows: usize = h[3];
    const sized = rows >= 1 and rows <= state.pass_rows;
    switch (h[0]) {
        magic => {
            if (h[1] == @backingInt(Job.end)) return .end;
            if (h[1] != @backingInt(Job.rows) or !sized) return error.BadJob;
            return .{ .one = .{ .id = h[2], .rows = rows } };
        },
        pass_magic => {
            if (h[1] != @backingInt(Job.pass) or !sized) return error.BadJob;
            const mask = h[4..][0..mask_words].*;
            var jobs: usize = 0;
            for (mask, 0..) |m, w| {
                const held: u32 = if (layers >= (w + 1) * 32) ~@as(u32, 0) else if (layers <= w * 32) 0 else (@as(u32, 1) << @intCast(layers - w * 32)) - 1;
                if (m & ~held != 0) return error.BadJob; // a layer past the model's
                jobs += @popCount(m);
            }
            if (jobs == 0 or jobs != h[2]) return error.BadJob;
            return .{ .pass = .{ .rows = rows, .mask = mask } };
        },
        else => return error.BadJob,
    }
}

/// Messages rank 1 took from rank 0 (the end's too), headers it read on the host, and the jobs they held.
pub const Count = struct { jobs: usize = 0, messages: usize = 0, reads: usize = 0 };

/// Rank 1's jobs until rank 0's end: `r` receives, reads a header on the host and runs a layer's job (tests fake it).
pub fn follow(r: anytype, c: Config, blocks: anytype, mtp: anytype, first: u64, long: u64) !Count {
    var n: Count = .{};
    while (true) {
        try r.recv(first, firstBytes(c));
        const h = try r.read(first);
        n.messages += 1;
        n.reads += 1;
        switch (try parse(&h, blocks.len)) {
            .end => return n,
            .one => |j| {
                const t = (if (j.id == latent.mtp_id) mtp else if (j.id < blocks.len) blocks[j.id] else null) orelse return error.BadJob;
                try take(r, c, t, j.rows, true, first, long, &n);
            },
            .pass => |p| {
                // every layer it names has tables here before any of its jobs is queued
                for (blocks, 0..) |t, i| if (bit(p.mask, i) and t == null) return error.BadJob;
                var headed = true;
                for (blocks, 0..) |t, i| if (bit(p.mask, i)) {
                    try take(r, c, t.?, p.rows, headed, first, long, &n);
                    headed = false;
                };
            },
        }
    }
}

/// One job's rows, where a job of its size always lands: a headed small job's came in the first message already.
fn take(r: anytype, c: Config, t: anytype, rows: usize, headed: bool, first: u64, long: u64, n: *Count) !void {
    const small = rows <= small_rows;
    const base = if (small) first + header_bytes else long;
    if (!small or !headed) {
        try r.recv(base, payloadBytes(c, if (small) small_rows else rows));
        n.messages += 1;
    }
    const s = sections(c, if (small) small_rows else rows);
    try r.run(t, .{ base + s[0], base + s[1], base + s[2] }, rows);
    n.jobs += 1;
}

/// Rank 1's device side of `follow`: NCCL from rank 0, a header read to the host, a layer's partials and their reply.
const Follower = struct {
    link: Link,
    ops: kern.Ops,
    scratch: latent.Scratch,
    c: Config,
    out: u64,

    pub fn recv(x: *const Follower, ptr: u64, bytes: usize) !void {
        return x.link.recv(ptr, bytes);
    }

    pub fn read(x: *const Follower, ptr: u64) ![header_words]u32 {
        var h: [header_words]u32 = undefined;
        try x.ops.s.synchronize();
        try x.ops.download(std.mem.sliceAsBytes(&h), ptr);
        try x.ops.s.synchronize();
        return h;
    }

    pub fn run(x: *const Follower, t: latent.Tables, at: [3]u64, rows: usize) !void {
        try latent.partials(x.ops, t, x.scratch, at[0], at[1], at[2], rows, x.c.ownShares(), x.c, x.out);
        try x.link.send(x.out, latent.partFloats(x.c, rows) * 4);
    }
};

/// Rank 1's tables: each MoE block's (by block index) and the MTP head's.
const Layers = struct { blocks: []?latent.Tables, mtp: ?latent.Tables = null };

fn loadLayers(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, c: Config, w: *W.Weights) !Layers {
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    var L: W.Loader = .{ .gpa = gpa, .ops = ops, .w = w, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(ops.k.d, 1 << 20) };
    defer {
        src.flush() catch {};
        ops.s.synchronize() catch {};
        L.scratch.free();
        L.host.deinit(gpa);
        for (L.held.items) |h| gpa.free(h);
        L.held.deinit(gpa);
    }
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    try src.add(&ck);
    var out: Layers = .{ .blocks = try gpa.alloc(?latent.Tables, c.layers) };
    errdefer gpa.free(out.blocks);
    var nm: [128]u8 = undefined;
    var pb: [128]u8 = undefined;
    for (c.kinds[0..c.layers], out.blocks, 0..) |kind, *t, i| {
        t.* = null;
        if (kind != Kind.moe) continue;
        const pre = try std.fmt.bufPrint(&pb, "{s}backbone.layers.{d}.mixer.", .{ c.prefix(), i });
        t.* = try latent.tables(&L, &ck, try std.fmt.bufPrint(&nm, "blocks.{d}.moe", .{i}), pre, c);
    }
    const mtp_pre = try std.fmt.bufPrint(&pb, "{s}mtp.layers.1.mixer.", .{c.prefix()});
    var gate: [160]u8 = undefined;
    if (ck.has(try std.fmt.bufPrint(&gate, "{s}gate.weight", .{mtp_pre}))) out.mtp = try latent.tables(&L, &ck, "mtp.moe", mtp_pre, c);
    try src.flush();
    try ops.s.synchronize();
    return out;
}

/// Rank r > 0: its expert shares of every MoE layer, computed for rank 0's jobs until rank 0 ends.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, precision: ?@import("config.zig").Precision) !void {
    const t0 = std.Io.Clock.awake.now(io);
    var c = try Config.read(gpa, io, dir);
    try latent.place(&c);
    var load_cache = core.checkpoint.cache.Load.begin(io, c.world > 1);
    defer load_cache.deinit();
    if (c.rank == 0 or c.latent == 0) return error.NotAnExpertRank;
    try latent.checkHeld(c);
    c.ck = try @import("cuda_latent_ck.zig").wanted(ctx, c, precision);
    try cuda.tp_watch.start(io, c.rank, c.world); // rank 0's line before NCCL: either rank's failure stops both
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(gpa, io, ctx, null);
    defer k.deinit();
    try k.loadModelopt(ctx);
    try k.loadLatent();
    if (c.ck) try k.loadCk(ctx);
    var g = try cuda.tp_group.Group.open(ctx.d, c.rank, c.world);
    defer g.close();
    try latent.agree(gpa, io, c, &g, ctx.d, stream, dir);
    const inventory = try @import("core").benchmark_group.cuda(gpa, io, ctx, &g, stream);
    defer if (inventory) |b| gpa.free(b);
    var content: ?*identity.Content = try identity.Content.start(gpa, io, dir, c.held != null); // compared at ready
    errdefer if (content) |x| x.abandon();
    const ops: kern.Ops = .{ .k = &k, .s = stream };
    var w: W.Weights = .{ .gpa = gpa, .config = c };
    defer w.deinit();
    const layers = try loadLayers(gpa, io, ops, dir, c, &w);
    defer gpa.free(layers.blocks);
    const R = state.pass_rows;
    const own = c.ownShares();
    const pairs = R * c.top_k * own;
    const sw = c.expert_width / c.shares;
    const ssw = c.shared_width / c.shares;
    const items = kern.maxItems(pairs, c.experts * own, 16);
    const ck4 = if (c.ck) @import("cuda_latent_ck.zig").rowsBytes(c, R) else 0; // NVFP4 rows (checkpoint math)
    const sizes = [_]usize{ firstBytes(c), payloadBytes(c, R), pairs * sw * 2, R * own * ssw * 2, R * own * 4, latent.partFloats(c, R) * 4, pairs * 4, items * 12, 8, pairs * 4, (pairs + 1023) / 1024 * c.experts * own * 4, ck4 };
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var arena = try cuda.DeviceBuffer.alloc(ctx.d, total);
    defer arena.free();
    var at: [sizes.len]u64 = undefined;
    var used: usize = 0;
    for (sizes, &at) |n, *p| {
        p.* = arena.ptr + used;
        used += std.mem.alignForward(usize, n, 256);
    }
    const first = at[0];
    const long = at[1];
    const out = at[5];
    const scratch: latent.Scratch = .{ .plan = .{ .members = at[6], .items = at[7], .counts = at[8], .rank = at[9], .hist = at[10] }, .act = at[2], .sact = at[3], .sids = at[4], .ck4 = if (c.ck) at[11] else 0 };
    try latent.initShareIds(k.latent.?, stream, scratch.sids, R, own);
    try stream.synchronize();
    var blocks: usize = 0;
    for (layers.blocks) |t| blocks += @intFromBool(t != null);
    std.log.info("rank {d} of {d}: {d} MoE layers{s} of share {d} ({s}) in {d:.1} s, {d:.2} GiB on the device; serving rank 0's jobs", .{ c.rank, c.world, blocks, if (layers.mtp != null) " and the MTP head's" else "", c.rank, if (c.ck) "the checkpoint's math" else "full precision", @import("cuda_engine.zig").seconds(io, t0), @as(f64, @floatFromInt(cuda.usage(false).device)) / (1 << 30) });
    const loaded = content.?;
    content = null;
    try load_cache.finishAfter(.{ io, try loaded.finish() }, cuda.tp_watch.ready);
    const follower: Follower = .{ .link = .{ .g = &g, .s = stream, .peer = 0 }, .ops = ops, .scratch = scratch, .c = c, .out = out };
    const n = try follow(&follower, c, layers.blocks, layers.mtp, first, long);
    cuda.tp_watch.ended();
    try stream.synchronize();
    std.log.info("rank {d}: rank 0 ended after {d} jobs in {d} messages and {d} header reads; leaving", .{ c.rank, n.jobs, n.messages, n.reads });
}

test "a job's sections and messages" {
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    try std.testing.expectEqual([3]usize{ 0, 1536, 1536 + 16 * 2048 }, sections(c, 16));
    try std.testing.expectEqual(@as(usize, 64 + 256 + 2048 + 8192), firstBytes(c)); // one row, not 16 rows of padding
    try std.testing.expectEqual(@as(usize, 0), sections(c, 1821)[1] % 256); // an odd job's rows start aligned
    try std.testing.expectEqual(@as(usize, 0), sections(c, 1821)[2] % 256);
}

test {
    _ = @import("cuda_worker_test.zig");
}
