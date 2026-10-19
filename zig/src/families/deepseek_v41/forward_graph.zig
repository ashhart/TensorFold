//! Decode windows through CUDA graphs (TF_DSV41_GRAPHS): keys, capture, replay, fingerprint.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const run = @import("run.zig");
const graphs = @import("graphs.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const ph = @import("phases.zig");
const fwd = @import("forward.zig");
const fwd_glue = @import("forward_glue.zig");
const Forward = fwd.Forward;
const glueFn = fwd_glue.glueFn;

/// Graphed: layer L's staged rows (copied before the launch) all-gathered, then forward.gather_cols' order.
pub fn engramGathered(f: *Forward, g: *Graphed, r: *run.Runner, c: *const calls.Call) !void {
    const L: u32 = @intCast(c.args[1].arg.i);
    const st = g.stageOf(L) orelse return error.NotEngram;
    const n = f.ids.len;
    if (f.gated) try f.gateCopy(r, L, st.dev.ptr, n);
    try f.comm.allGather(st.dev.ptr, f.gathered.?.ptr, n * st.cols, .bf16, r.stream.handle);
    return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, f.comm.world(), n, st.cols, try r.tensorAddr(c.args[0].arg.t));
}

pub fn activeGraph(f: *Forward) ?*Graphed {
    const g = f.graphed orelse return null;
    return if (g.active) g else null;
}

/// The graphs' state, made on the first window when TF_DSV41_GRAPHS is set (both ranks must set it alike).
pub fn graphState(f: *Forward) !?*Graphed {
    if (!f.graphs_read) {
        f.graphs_read = true;
        const s = try graphs.Settings.fromEnvOr(f.graphs_default);
        if (s.on and f.planned_rows > 0) {
            const g = try f.gpa.create(Graphed);
            errdefer f.gpa.destroy(g);
            try g.init(f, s);
            f.graphed = g;
            std.log.scoped(.dsv41).info("graphs: on, windows of up to {d} rows, context buckets of {d}", .{ g.rows_max, s.bucket });
        }
    }
    return f.graphed;
}

/// The window through its key's graph (captured on first use); false: not graphable here, run eagerly.
pub fn graphedWindow(f: *Forward, ids: []const u32) !bool {
    const g = (try f.graphState()) orelse return false;
    // a traced window is never graphed, and its program is not the key's: never emitted or cached here
    if (f.opts.trace) return false;
    const n: u32 = @intCast(ids.len);
    const limit: u64 = @intCast(f.opts.limit);
    if (n > g.rows_max or f.slot.pos + n > limit) return false;
    const key: graphs.Key = .{ .rows = n, .ctx = graphs.bucketOf(f.slot.pos + n - 1, g.settings.bucket, g.settings.grow) };
    const cs = try g.program(f, key);
    if (graphs.audit(cs) != null) return false;
    const r = f.runner;
    f.ids = ids;
    f.slot.pending = .{ .start = f.slot.pos, .n = n };
    // inputs: one copy of the ids and start, then each Engram layer's rows (hashed here, on the host)
    {
        const ts = ph.start(f.phases, .@"graph.stage");
        defer ts.stop();
        f.gated = try f.armEngram(&.{.{ .ids = ids, .tail = f.slot.tail[0..f.slot.tail_len] }}, n);
        if (!f.gated) f.issueEngram(ids, f.slot.tail[0..f.slot.tail_len]);
        try g.statics.stage(r.stream, ids, f.slot.pos);
        if (f.engram) |h| if (!f.gated) for (g.stages.items) |*st| {
            try f.fillEngramRows(h, st.layer, try st.begin(n));
            try st.commit(r.stream, n);
        };
    }
    g.active = true;
    defer g.active = false;
    r.glue = .{ .ctx = f, .run = glueFn };
    g.current = cs;
    const t0 = ph.nowNs();
    const fp = f.fingerprint(g);
    // TF_DSV41_ROUND_GRAPH=split: the head's graph, then the tail's, launched while the GPU runs the head
    const at = if (g.split) |rg| hooks.set.round_graph.?.splitAt(cs, rg.head) else null;
    var outcome: graphs.gc.Outcome = .replayed;
    if (at) |i| {
        g.span = .{ 0, i };
        const o1 = try g.cache.run(.{ .rows = key.rows, .ctx = key.ctx, .part = 1 }, r.stream, fp, .{ .ctx = f, .run = graphBody });
        g.span = .{ i, cs.len };
        outcome = try g.cache.run(.{ .rows = key.rows, .ctx = key.ctx, .part = 2 }, r.stream, fp, .{ .ctx = f, .run = graphBody });
        if (o1 == .captured) outcome = .captured;
    } else {
        g.span = .{ 0, cs.len };
        outcome = try g.cache.run(key, r.stream, fp, .{ .ctx = f, .run = graphBody });
    }
    if (f.phases) |x| x.add(if (outcome == .captured) .@"graph.capture" else .@"graph.replay", ph.nowNs() - t0);
    if (outcome == .captured and g.cache.stats.captured == 1)
        std.log.scoped(.dsv41).info("graphs: first capture ({d} rows, context bucket {d})", .{ key.rows, key.ctx });
    return true;
}

pub fn graphBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    _ = stream; // the runner's stream: the one the cache captures
    const g = f.graphed.?;
    try f.runner.window(g.current[g.span[0]..g.span[1]]);
}

/// The addresses a graphed window bakes in beyond the runner's bound roles: any change drops every graph.
pub fn fingerprint(f: *const Forward, g: *const Graphed) u64 {
    var words: [64]u64 = undefined;
    var m: usize = 0;
    for (f.runner.owned.items) |b| {
        if (m + 2 > words.len) break;
        words[m] = b.ptr;
        words[m + 1] = b.len;
        m += 2;
    }
    for ([_]?cuda.DeviceBuffer{ f.runner.scratch, f.send, f.gathered, g.statics.dev }) |b| if (b) |x| {
        if (m + 1 > words.len) break;
        words[m] = x.ptr;
        m += 1;
    };
    for (g.stages.items) |st| if (m < words.len) {
        words[m] = st.dev.ptr;
        m += 1;
    };
    return graphs.gc.fingerprintOf(words[0..m]);
}

/// The graphed path's state: the cache, window statics, Engram stages, programs by key (emitted at the bucket's end).
pub const Graphed = struct {
    gpa: std.mem.Allocator,
    settings: graphs.Settings,
    rows_max: u32,
    engine: graphs.gc.CudaEngine,
    agreement: graphs.Agreement,
    cache: graphs.Cache,
    statics: graphs.Statics,
    stages: std.ArrayList(graphs.EngramStage) = .empty,
    programs: std.AutoHashMapUnmanaged(graphs.Key, []const calls.Call) = .empty,
    current: []const calls.Call = &.{},
    /// the calls of `current` the graph body issues (TF_DSV41_ROUND_GRAPH's `split`: the head's, then the tail's)
    span: [2]usize = .{ 0, 0 },
    split: ?ft.RoundGraph = null,
    active: bool = false,

    fn init(g: *Graphed, f: *Forward, s: graphs.Settings) !void {
        const r = f.runner;
        const rows = @min(s.rows_max, f.planned_rows);
        g.* = .{
            .gpa = f.gpa,
            .settings = s,
            .rows_max = rows,
            .engine = .{ .d = r.d },
            .agreement = try graphs.Agreement.init(r.d, f.comm, r.stream),
            .cache = undefined,
            .statics = undefined,
        };
        errdefer g.agreement.deinit();
        g.statics = try graphs.Statics.init(r.d, rows);
        errdefer g.statics.deinit();
        const rg = try (try hooks.roundGraph()).current();
        if (rg.split) g.split = rg;
        // `split`: a window's head and tail graphs in one cache of twice the cap (a head is ~1/43 of a window)
        g.cache = graphs.Cache.init(f.gpa, g.engine.engine(), .{ .on = true, .max = if (rg.split) 2 * s.max else s.max, .hold = s.floor_hold });
        g.cache.agree = g.agreement.agree();
        g.cache.room = graphs.room;
        g.cache.meter = graphs.meter;
        g.cache.name = "decode windows";
        graphs.rule = s.room;
        // every buffer a graph bakes in, at the largest window: the exchanges' send, Engram's gathered rows
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        // at the planned rows, so an eager window of more rows than a graph takes moves nothing (the fingerprint)
        const most: usize = f.planned_rows;
        _ = try f.sendBuffer(r, 2 * most * D);
        if (f.engram) |h| {
            var la = std.heap.ArenaAllocator.init(f.gpa);
            defer la.deinit();
            const cols = f.engramCols(h);
            for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) {
                try g.stages.append(f.gpa, try graphs.EngramStage.init(r.d, L, rows, cols));
            };
            const bytes = W * 2 * most * cols;
            if (f.gathered == null or f.gathered.?.len < bytes) {
                if (f.gathered) |*b| b.free();
                f.gathered = try cuda.DeviceBuffer.alloc(r.d, bytes);
            }
        }
    }

    pub fn drain(g: *Graphed) !void {
        try g.agreement.stream.synchronize();
        try g.statics.drain();
        for (g.stages.items) |*st| try st.drain();
    }
    pub fn close(g: *Graphed) !void {
        try g.drain();
        g.deinitAfterDrain();
    }
    pub fn deinitAfterDrain(g: *Graphed) void {
        const cs = g.cache.stats;
        std.log.scoped(.dsv41).info("graphs: {d} captured, {d} replayed, {d} eager, {d} keys held, {d} evicted, {d} dropped{s}", .{ cs.captured, cs.replayed, cs.eager, g.cache.count(), cs.evicted, cs.dropped, if (g.cache.failed) " (off after a failed capture)" else "" });
        g.cache.deinit();
        for (g.stages.items) |*st| st.deinit();
        g.stages.deinit(g.gpa);
        g.programs.deinit(g.gpa);
        g.statics.deinit();
        g.agreement.deinit();
    }

    fn stageOf(g: *Graphed, L: u32) ?*graphs.EngramStage {
        for (g.stages.items) |*st| if (st.layer == L) return st;
        return null;
    }

    /// The key's calls, emitted once (forward's arena) at the start that puts the window's end at its bucket's end.
    fn program(g: *Graphed, f: *Forward, key: graphs.Key) ![]const calls.Call {
        if (g.programs.get(key)) |cs| return cs;
        const a = f.arena.allocator();
        const start = graphs.emitStart(key.rows, key.ctx, g.settings, @intCast(f.opts.limit));
        const cs = try block.emit(a, f.cfg, f.widths, f.opts, try f.backbone(a), key.rows, start, true);
        try g.programs.put(g.gpa, key, cs);
        return cs;
    }
};
