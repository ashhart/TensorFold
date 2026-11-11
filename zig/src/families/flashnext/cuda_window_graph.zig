//! Drafted windows' passes, the draft head's pieces and rollback replays as CUDA graphs on one GPU, one key a graph.
const std = @import("std");
const cuda = @import("cuda");
const parts = @import("cuda_batch_state.zig");
const wt = @import("cuda_prompt_wt.zig");

// A key holds every host value its kernels take; uploads stay outside the graph (stageWindow); caches clear on realloc.

/// Graphs a cache holds at most; past it the cache starts over (finished sequences' graphs stay until then).
const max_graphs = 128;

var gens = std.atomic.Value(u64).init(1);

/// A sequence's or a draft's own number for the keys: a freed one's graphs never match a new one at its addresses.
pub fn nextGen() u64 {
    return gens.fetchAdd(1, .monotonic);
}

/// On by default (the same kernels, launched from a graph); TF_FLASHNEXT_WINDOW_GRAPH=0: every pass eagerly.
pub fn enabled() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_WINDOW_GRAPH", true) catch true;
}

/// attnRows' host values for rows ending at `ends` in a `cap`-token cache: dense chunks, the score grid (0: none).
pub const Geometry = struct {
    chunks: u32,
    score_grid: u32,

    pub fn at(ends: usize, cap: usize) Geometry {
        const ratio = wt.index_ratio;
        const nb = parts.scoreStride(cap);
        const scored = ends / ratio > wt.index_top;
        return .{
            .chunks = @intCast((@min(ends, wt.dense_keys) + 511) / 512),
            .score_grid = if (scored) @intCast((@min(nb, (ends + ratio - 1) / ratio) + 63) / 64) else 0,
        };
    }
};

/// A key's two hashes: `shape` decides which kernels launch (the graph's slot), `moving` holds every other host value.
const Hash = struct {
    shape: std.hash.Wyhash = .init(0),
    moving: std.hash.Wyhash = .init(0),

    fn fixed(h: *Hash, v: u64) void {
        h.shape.update(std.mem.asBytes(&v));
    }

    /// Values a held graph takes in place (buffers, lengths, the sequence's number); whether each is 0 stays a shape.
    fn values(h: *Hash, vs: []const u64) void {
        for (vs) |v| {
            h.moving.update(std.mem.asBytes(&v));
            h.fixed(@intFromBool(v != 0));
        }
    }

    /// The length's host values: the score grid on or off launches other kernels, its size and the chunks do not.
    fn geometry(h: *Hash, g: Geometry) void {
        h.values(&[_]u64{ g.chunks, g.score_grid });
    }

    fn keep(h: *Hash, k: ?*const parts.Keep) void {
        const kept = k orelse return h.fixed(0);
        h.fixed(1);
        h.values(std.mem.bytesAsSlice(u64, std.mem.asBytes(kept)));
    }
};

/// Every buffer a pass takes from a sequence's tables (conv windows, states, caches, tail, page table) as one value.
pub fn pieces(seq: anytype) u64 {
    const S = switch (@typeInfo(@TypeOf(seq))) {
        .pointer => |p| p.child,
        else => @TypeOf(seq),
    };
    var d: Digest = .{};
    inline for (.{ "conv", "state", "kc", "vc", "ikc", "pooled" }) |f| {
        for (@field(seq, f)) |b| d.add(b.ptr);
    }
    const pg = if (@hasField(S, "pg")) seq.pg else seq.pg();
    for ([_]u64{ seq.tail.ptr, pg.pt, @bitCast(pg.ps), @as(u32, @bitCast(pg.ptok)) }) |v| d.add(v);
    return d.final();
}

/// Values hashed in blocks (one update a block, not one a value).
const Digest = struct {
    h: std.hash.Wyhash = .init(0),
    buf: [64]u64 = undefined,
    n: usize = 0,

    fn add(d: *Digest, v: u64) void {
        if (d.n == d.buf.len) {
            d.h.update(std.mem.sliceAsBytes(&d.buf));
            d.n = 0;
        }
        d.buf[d.n] = v;
        d.n += 1;
    }

    fn final(d: *Digest) u64 {
        d.h.update(std.mem.sliceAsBytes(d.buf[0..d.n]));
        return d.h.final();
    }
};

/// A verify window's pass (Engine.forward): its rows and `hooks`' outputs, then the sequence's buffers and length.
pub const Key = struct {
    rows: u32,
    shape: u64,
    moving: u64,

    pub fn init(seq: anytype, pos0: usize, rows: usize, s: *const parts.Scratch, scores: u64, model: u64, hooks: parts.Hooks) Key {
        var h: Hash = .{};
        h.geometry(.at(pos0 + rows, seq.cap));
        h.values(&[_]u64{ seq.gen, seq.cap, pieces(seq), s.h, s.cap, scores, model, hooks.logits_out, hooks.streams });
        h.keep(hooks.keep);
        return .{ .rows = @intCast(rows), .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// One piece of the draft head's pass (forwardHead): its rows, whether it is the last and scores, then its buffers.
pub const HeadKey = struct {
    rows: u32,
    last: bool,
    scores: bool,
    shape: u64,
    moving: u64,

    pub fn init(d: anytype, rows: usize, last: bool, scores: bool, input: u64, s: *const parts.Scratch, score_rows: u64) HeadKey {
        var h: Hash = .{};
        h.geometry(.at(d.len + rows, d.cap));
        h.values(&[_]u64{ d.gen, d.cap, d.kc.ptr, input, d.out.ptr, d.dlogits.ptr, s.h, s.cap, score_rows });
        return .{ .rows = @intCast(rows), .last = last, .scores = scores, .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// Several streams' drafted windows in one pass (Engine.windows): each lane's rows and outputs, then its buffers.
pub const SharedKey = struct {
    lanes: u32,
    rows: u32,
    /// The round's attention in one launch (cuda_batch_lanes.useAttn), other kernels than the per-lane ones.
    attn: bool,
    shape: u64,
    moving: u64,

    pub fn init(lanes: []const parts.Lane, s: *const parts.Scratch, scores: u64, model: u64, out: u64, attn: bool) SharedKey {
        var h: Hash = .{};
        var rows: usize = 0;
        for (lanes) |l| {
            h.fixed(l.rows);
            h.geometry(.at(l.pos + l.rows, l.seq.cap));
            h.values(&[_]u64{ l.seq.gen, pieces(l.seq), l.seq.cap, l.streams, l.logits });
            if (l.logit_rows.len > 0) {
                h.fixed(l.logit_rows.len); // a copy a row: other launches than one lane copy
                h.values(l.logit_rows);
            }
            h.keep(l.keep);
            rows += l.rows;
        }
        h.values(&[_]u64{ s.h, s.cap, scores, model, out });
        return .{ .lanes = @intCast(lanes.len), .rows = @intCast(rows), .attn = attn, .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// A group of streams' draft-head pieces in one pass (Ops.headGroup): each lane's rows, then its buffers.
pub const HeadManyKey = struct {
    lanes: u32,
    rows: u32,
    scores: bool,
    shape: u64,
    moving: u64,

    pub fn init(ds: anytype, nexts: anytype, inputs: []const u64, total: usize, scores: bool, s: *const parts.Scratch, score_rows: u64) HeadManyKey {
        var h: Hash = .{};
        for (ds, nexts, inputs) |d, nx, input| {
            h.fixed(nx.len);
            h.geometry(.at(d.len + nx.len, d.cap));
            h.values(&[_]u64{ d.gen, d.kc.ptr, d.cap, input, d.out.ptr, d.dlogits.ptr });
        }
        h.values(&[_]u64{ s.h, s.cap, score_rows });
        return .{ .lanes = @intCast(ds.len), .rows = @intCast(total), .scores = scores, .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// A partial keep's replay over the DeltaNet layers (Ops.rollback): rows kept, then every buffer it touches.
pub const RollbackKey = struct {
    keep: u32,
    shape: u64,
    moving: u64,

    pub fn init(seq: anytype, d: anytype, keep: usize, s: *const parts.Scratch) RollbackKey {
        var h: Hash = .{};
        h.values(&[_]u64{ seq.gen, d.gen, seq.tail.ptr, d.saved.ptr, d.saved_tail, d.saved_nrow, s.h, s.cap });
        for (seq.conv) |b| h.values(&[_]u64{b.ptr});
        for (seq.state) |b| h.values(&[_]u64{b.ptr});
        h.keep(&d.keep);
        return .{ .keep = @intCast(keep), .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// A key without what moves between runs of one held graph (`moving`): the graph's slot.
pub fn slotOf(key: anytype) @TypeOf(key) {
    var slot = key;
    slot.moving = 0;
    return slot;
}

pub fn Cache(comptime K: type) type {
    return struct {
        const Self = @This();
        /// A held graph and the whole key it was last recorded for; the map holds it under its slot (slotOf).
        const Held = struct { key: K, exec: cuda.graph.Exec };
        map: std.AutoHashMapUnmanaged(K, Held) = .empty,
        /// Slots run once eagerly and are captured the second time (a one-off pass would never pay its capture back).
        seen: std.AutoHashMapUnmanaged(K, void) = .empty,
        captures: u64 = 0,
        replays: u64 = 0,
        /// Held graphs re-recorded at a new length, for a new sequence or on other buffers and updated in place.
        updates: u64 = 0,
        /// A capture failed: every later pass runs eagerly (logged once).
        off: bool = false,

        pub fn clear(c: *Self, gpa: std.mem.Allocator) void {
            var it = c.map.valueIterator();
            while (it.next()) |h| h.exec.deinit();
            c.map.clearAndFree(gpa);
            c.seen.clearAndFree(gpa);
        }

        /// `pass.run()` (kernels only: its uploads already queued) as `key`'s graph, captured first when new.
        pub fn run(c: *Self, gpa: std.mem.Allocator, stream: cuda.Stream, key: K, pass: anytype) !void {
            if (c.off) return pass.run();
            const slot = slotOf(key);
            if (c.map.getPtr(slot)) |h| {
                if (std.meta.eql(h.key, key)) {
                    c.replays += 1;
                    return h.exec.launchOn(stream);
                }
                // the same pass at another length, sequence or lane: recorded again, the held graph updated in place
                refresh(stream, pass, &h.exec) catch |err| return c.fail(err, pass);
                h.key = key;
                c.updates += 1;
                return h.exec.launchOn(stream);
            }
            if (c.map.count() >= max_graphs or c.seen.count() >= 4 * max_graphs) c.clear(gpa);
            if (!c.seen.contains(slot)) {
                try c.seen.put(gpa, slot, {});
                return pass.run();
            }
            var exec = capture(stream, pass) catch |err| return c.fail(err, pass);
            errdefer exec.deinit();
            try c.map.put(gpa, slot, .{ .key = key, .exec = exec });
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("{s} graphs: {d} captured, {d} updated, {d} replayed, {d} held", .{ @typeName(K), c.captures, c.updates, c.replays, c.map.count() });
            try exec.launchOn(stream);
        }

        /// The stream is out of capture again: this pass eagerly, and the ones after it.
        fn fail(c: *Self, err: anyerror, pass: anytype) !void {
            std.log.warn("{s} graphs off: capture failed ({s}), eager from here", .{ @typeName(K), @errorName(err) });
            c.off = true;
            return pass.run();
        }
    };
}

/// The engine's graph caches, one per kind of pass.
pub const Graphs = struct {
    window: Cache(Key) = .{},
    head: Cache(HeadKey) = .{},
    rollback: Cache(RollbackKey) = .{},
    shared: Cache(SharedKey) = .{},
    head_many: Cache(HeadManyKey) = .{},

    pub fn clear(g: *Graphs, gpa: std.mem.Allocator) void {
        g.window.clear(gpa);
        g.head.clear(gpa);
        g.rollback.clear(gpa);
        g.shared.clear(gpa);
        g.head_many.clear(gpa);
    }
};

/// The pass's kernels recorded into a graph, nothing launched; the stream left uncaptured.
fn record(stream: cuda.Stream, pass: anytype) !cuda.graph.Graph {
    try cuda.graph.beginCapture(stream, .thread_local);
    pass.run() catch |err| {
        if (cuda.graph.endCapture(stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    return cuda.graph.endCapture(stream);
}

/// The pass's kernels captured and instantiated (uploaded to the device), nothing launched.
fn capture(stream: cuda.Stream, pass: anytype) !cuda.graph.Exec {
    var graph = try record(stream, pass);
    defer graph.deinit();
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    return exec;
}

/// `exec` with the pass's values recorded now; a pass of other launches (another topology) gets a new executable.
fn refresh(stream: cuda.Stream, pass: anytype, exec: *cuda.graph.Exec) !void {
    var graph = try record(stream, pass);
    defer graph.deinit();
    if (try exec.update(graph) == .success) return;
    var fresh = try graph.instantiate();
    errdefer fresh.deinit();
    try fresh.upload(stream);
    exec.deinit();
    exec.* = fresh;
}

/// A window's uploads ahead of its pass: ids and positions, then its n-gram rows gathered and queued behind one fence.
pub fn stageWindow(p: anytype, ids: []const u32, pos0: usize) !void {
    const s = p.s;
    const n = ids.len;
    if (n == 0 or n > s.cap) return error.UnexpectedTensor;
    try s.ready();
    p.pos0 = pos0;
    const was = p.prepared;
    p.prepared = true;
    defer p.prepared = was;
    try p.stage(ids);
    const emb = s.hostEmb(n);
    const xs = s.hostXs(n);
    var gather: parts.Gather = .{ .ngram = p.m.ngram, .history = p.seq.history[0..], .ids = ids, .emb = emb, .xs = xs, .io = p.io, .lanes = p.lanes };
    gather.main();
    if (gather.failed) |err| return err;
    try p.upload(s.ple_emb, std.mem.sliceAsBytes(emb));
    try p.upload(s.ple_xs, std.mem.sliceAsBytes(xs));
    try s.uploaded();
    p.staged = true;
}

/// A pass over a window whose uploads stageWindow queued: Pass.chunk launches its kernels alone.
pub fn Chunk(comptime P: type) type {
    return struct {
        p: *P,
        ids: []const u32,
        pos0: usize,

        pub fn run(r: @This()) !void {
            return r.p.chunk(r.ids, r.pos0, null);
        }
    };
}

test {
    _ = @import("cuda_window_graph_test.zig");
}
