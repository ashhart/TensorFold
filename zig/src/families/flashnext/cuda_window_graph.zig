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
    const v = std.c.getenv("TF_FLASHNEXT_WINDOW_GRAPH") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
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

/// A verify window's pass (Engine.forward); `hooks` hashes the scores out, the streams and every pointer of its keep.
pub const Key = struct {
    gen: u64,
    rows: u32,
    geometry: Geometry,
    cap: u64,
    kc: u64,
    state: u64,
    scratch: u64,
    scratch_cap: u64,
    scores: u64,
    model: u64,
    hooks: u64,

    pub fn init(seq: anytype, pos0: usize, rows: usize, s: *const parts.Scratch, scores: u64, model: u64, hooks: parts.Hooks) Key {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&hooks.logits_out));
        h.update(std.mem.asBytes(&hooks.streams));
        if (hooks.keep) |k| h.update(std.mem.asBytes(k));
        return .{
            .gen = seq.gen,
            .rows = @intCast(rows),
            .geometry = .at(pos0 + rows, seq.cap),
            .cap = seq.cap,
            .kc = seq.kc[0].ptr,
            .state = seq.state[0].ptr,
            .scratch = s.h,
            .scratch_cap = s.cap,
            .scores = scores,
            .model = model,
            .hooks = h.final(),
        };
    }
};

/// One piece of the draft head's pass (forwardHead): its rows, where its input rows come from, whether it scores.
pub const HeadKey = struct {
    gen: u64,
    rows: u32,
    geometry: Geometry,
    scores: bool,
    cap: u64,
    kc: u64,
    input: u64,
    out: u64,
    dlogits: u64,
    scratch: u64,
    scratch_cap: u64,
    score_rows: u64,
};

/// Several streams' drafted windows in one pass (Engine.windows): each lane's buffers, rows and geometry, hashed.
pub const SharedKey = struct {
    lanes: u32,
    rows: u32,
    scratch: u64,
    scratch_cap: u64,
    scores: u64,
    model: u64,
    out: u64,
    each: u64,

    pub fn init(lanes: []const parts.Lane, s: *const parts.Scratch, scores: u64, model: u64, out: u64) SharedKey {
        var h = std.hash.Wyhash.init(0);
        var rows: usize = 0;
        for (lanes) |l| {
            const g: Geometry = .at(l.pos + l.rows, l.seq.cap);
            for ([_]u64{ l.seq.gen, l.seq.kc[0].ptr, l.seq.state[0].ptr, l.seq.cap, l.rows, g.chunks, g.score_grid, l.streams, l.logits }) |v| h.update(std.mem.asBytes(&v));
            if (l.keep) |k| h.update(std.mem.asBytes(k));
            rows += l.rows;
        }
        return .{ .lanes = @intCast(lanes.len), .rows = @intCast(rows), .scratch = s.h, .scratch_cap = s.cap, .scores = scores, .model = model, .out = out, .each = h.final() };
    }
};

/// A group of streams' draft-head pieces in one pass (Ops.headGroup): each lane's buffers, rows and geometry hashed.
pub const HeadManyKey = struct {
    lanes: u32,
    rows: u32,
    scores: bool,
    scratch: u64,
    scratch_cap: u64,
    score_rows: u64,
    each: u64,
};

/// A partial keep's replay over the DeltaNet layers (Ops.rollback): rows kept, scratch, every buffer it touches hashed.
pub const RollbackKey = struct {
    keep: u32,
    scratch: u64,
    scratch_cap: u64,
    bufs: u64,
};

pub fn Cache(comptime K: type) type {
    return struct {
        const Self = @This();
        map: std.AutoHashMapUnmanaged(K, cuda.graph.Exec) = .empty,
        /// Keys run once eagerly and are captured the second time (a one-off key would never pay its capture back).
        seen: std.AutoHashMapUnmanaged(K, void) = .empty,
        captures: u64 = 0,
        replays: u64 = 0,
        /// A capture failed: every later pass runs eagerly (logged once).
        off: bool = false,

        pub fn clear(c: *Self, gpa: std.mem.Allocator) void {
            var it = c.map.valueIterator();
            while (it.next()) |e| e.deinit();
            c.map.clearAndFree(gpa);
            c.seen.clearAndFree(gpa);
        }

        /// `pass.run()` (kernels only: its uploads already queued) as `key`'s graph, captured first when new.
        pub fn run(c: *Self, gpa: std.mem.Allocator, stream: cuda.Stream, key: K, pass: anytype) !void {
            if (c.off) return pass.run();
            if (c.map.get(key)) |exec| {
                c.replays += 1;
                return exec.launchOn(stream);
            }
            if (c.map.count() >= max_graphs or c.seen.count() >= 4 * max_graphs) c.clear(gpa);
            if (!c.seen.contains(key)) {
                try c.seen.put(gpa, key, {});
                return pass.run();
            }
            var exec = capture(stream, pass) catch |err| {
                // the stream is out of capture again: this pass eagerly, and the ones after it
                std.log.warn("{s} graphs off: capture failed ({s}), eager from here", .{ @typeName(K), @errorName(err) });
                c.off = true;
                return pass.run();
            };
            errdefer exec.deinit();
            try c.map.put(gpa, key, exec);
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("{s} graphs: {d} captured, {d} replayed, {d} held", .{ @typeName(K), c.captures, c.replays, c.map.count() });
            try exec.launchOn(stream);
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

/// The pass's kernels captured and instantiated (uploaded to the device), nothing launched; the stream left uncaptured.
fn capture(stream: cuda.Stream, pass: anytype) !cuda.graph.Exec {
    try cuda.graph.beginCapture(stream, .thread_local);
    pass.run() catch |err| {
        if (cuda.graph.endCapture(stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    return exec;
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
