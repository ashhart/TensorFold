//! Opt-in CUDA graphs of a DFlash2 hold and live absorb: the eager launches, captured, live arguments set each round.
const std = @import("std");
const cuda = @import("cuda");
const platform = @import("sys");
const trace = @import("cuda_launch_trace.zig");
const graph = cuda.graph;

/// TF_DFLASH_GRAPH=1 replays a DFlash2 hold and live absorb from CUDA graphs; off by default.
pub fn enabled() bool {
    return platform.environment.read(bool, "TF_DFLASH_GRAPH", false) catch false;
}

/// TF_QWEN_GRAPH_STATS=1 records hold outcomes once at drafter startup; off by default.
pub fn statsEnabled() bool {
    return platform.environment.read(bool, "TF_QWEN_GRAPH_STATS", false) catch false;
}

const HoldKey = struct { streams: u8, narrow: bool, projection_role: u8, width_mask: u128 };
const Hold = struct { exec: graph.Exec, spanned: ?u64 };

const Count = struct { captured: usize = 0, replayed: usize = 0, eager: usize = 0, key_miss: usize = 0, eviction: usize = 0, width_not_graphable: usize = 0 };
const Stats = struct {
    on: bool = false,
    counts: std.AutoHashMapUnmanaged(HoldKey, Count) = .empty,

    fn note(s: *Stats, gpa: std.mem.Allocator, key: HoldKey, comptime field: []const u8) !void {
        if (!s.on) return;
        const entry = try s.counts.getOrPut(gpa, key);
        if (!entry.found_existing) entry.value_ptr.* = .{};
        @field(entry.value_ptr, field) += 1;
    }
};
const Absorb = struct { source: graph.Graph, exec: graph.Exec, gather: graph.Node, ropes: []graph.Node, rings: []graph.Node };

pub const Graphs = struct {
    on: bool = false,
    stats: Stats = .{},
    holds: std.AutoHashMapUnmanaged(HoldKey, Hold) = .empty,
    absorbs: std.AutoHashMapUnmanaged(u32, Absorb) = .empty,
    /// Drops every graph: their captured addresses die with the scratch they name.
    pub fn clear(g: *Graphs, gpa: std.mem.Allocator) void {
        var holds = g.holds.valueIterator();
        while (holds.next()) |h| h.exec.deinit();
        g.holds.clearRetainingCapacity();
        var absorbs = g.absorbs.valueIterator();
        while (absorbs.next()) |a| dropAbsorb(gpa, a);
        g.absorbs.clearRetainingCapacity();
    }

    pub fn deinit(g: *Graphs, gpa: std.mem.Allocator) void {
        if (g.stats.on) {
            std.log.info("qwen draft graphs: shutdown statistics", .{});
            var it = g.stats.counts.iterator();
            while (it.next()) |entry| {
                var line: [512]u8 = undefined;
                var w = std.Io.Writer.fixed(&line);
                writeCount(&w, entry.key_ptr.*, entry.value_ptr.*) catch continue;
                const rendered = w.buffered();
                std.log.info("{s}", .{rendered[0 .. rendered.len - 1]});
            }
        }
        g.clear(gpa);
        g.holds.deinit(gpa);
        g.absorbs.deinit(gpa);
        g.stats.counts.deinit(gpa);
    }
};

fn dropAbsorb(gpa: std.mem.Allocator, a: *Absorb) void {
    a.exec.deinit();
    a.source.deinit();
    gpa.free(a.ropes);
    gpa.free(a.rings);
}

fn live(d: anytype) bool {
    return enabled() and d.graphs.on and trace.active == null and d.e.learning == null and !d.e.pockets.any();
}

fn tapsArgs(d: anytype, ring: u64, start: u64) struct { u64, c_int, c_longlong, c_int, u64 } {
    return .{ ring, @as(c_int, @intCast(d.e.tapRows())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(d.c.n_taps * d.c.hidden)), d.taps_in };
}

fn tapRows(d: anytype, rows: usize, rs: []const u32) @TypeOf(d.*).TapRows {
    var tr: @TypeOf(d.*).TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
    for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
    return tr;
}

fn rowsArgs(d: anytype, ring: u64, start: u64, rows: usize, rs: []const u32) struct { u64, c_int, c_longlong, @TypeOf(d.*).TapRows, c_int, u64 } {
    return .{ ring, @as(c_int, @intCast(d.e.tapRows())), @as(c_longlong, @intCast(start)), tapRows(d, rows, rs), @as(c_int, @intCast(d.c.n_taps * d.c.hidden)), d.taps_in };
}

fn ropeArgs(d: anytype, w: u64, start: u64) struct { u64, u64, u64, c_int, c_int, c_longlong, @TypeOf(d.c.eps), @TypeOf(d.c.theta) } {
    const heads = d.c.kv_heads;
    return .{ d.kabs, w, d.kn, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * d.c.head_dim)), @as(c_longlong, @intCast(start)), d.c.eps, d.c.theta };
}

fn ringArgs(d: anytype, ctx: anytype, layer: usize, start: u64) struct { u64, u64, u64, u64, c_int, c_longlong, c_int } {
    return .{ d.kn, d.vabs, d.ringK(ctx, layer), d.ringV(ctx, layer), @as(c_int, @intCast(d.c.kvWidth())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(d.c.ringRows())) };
}

/// One chunk of an absorb: the launches in their eager order.
pub fn absorbRows(d: anytype, ctx: anytype, ring: u64, start: u64, rows: usize, rs: ?[]const u32) !void {
    const e = d.e;
    const c = d.c;
    if (rs) |list| {
        try e.run(d.fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, rowsArgs(d, ring, start, rows, list));
    } else try e.run(d.fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, tapsArgs(d, ring, start));
    try d.proj(d.w.fc, d.taps_in, d.th, rows);
    try d.norm(d.th, d.w.hidden_norm, d.thn, rows);
    for (d.w.layers, 0..) |l, i| {
        if (e.affine.pdl) {
            try d.projManySums(&.{ l.k, l.v }, d.thn, &.{ d.kabs, d.vabs }, rows, false);
            try e.run(d.fns.norm_rope, .{ rows * c.kv_heads, 1, 1 }, .{ 32, 1, 1 }, ropeArgs(d, l.k_norm, start));
        } else {
            try d.projAgain(l.k, d.thn, d.kabs, rows);
            try e.run(d.fns.norm_rope, .{ rows * c.kv_heads, 1, 1 }, .{ 32, 1, 1 }, ropeArgs(d, l.k_norm, start));
            try d.projAgain(l.v, d.thn, d.vabs, rows);
        }
        try e.run(d.fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, ringArgs(d, ctx, i, start));
    }
}

fn set(exec: graph.Exec, node: graph.Node, f: cuda.Function, grid: [3]u32, block: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    const cfg: cuda.Config = .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block, .y = 1, .z = 1 } };
    try exec.setKernel(node, f, cfg, &a);
}

fn capture(d: anytype, ctx: anytype, ring: u64, start: u64, rows: usize, rs: ?[]const u32) !?Absorb {
    const stream = d.e.stream;
    try graph.beginCapture(stream, .thread_local);
    absorbRows(d, ctx, ring, start, rows, rs) catch |err| {
        var broken = graph.endCapture(stream) catch return err;
        broken.deinit();
        return err;
    };
    var source = try graph.endCapture(stream);
    errdefer source.deinit();
    const layers = d.w.layers.len;
    const all = try d.gpa.alloc(graph.Node, try source.nodeCount());
    defer d.gpa.free(all);
    const ropes = try d.gpa.alloc(graph.Node, layers);
    errdefer d.gpa.free(ropes);
    const rings = try d.gpa.alloc(graph.Node, layers);
    errdefer d.gpa.free(rings);
    const want = if (rs != null) d.fns.gather_rows.handle else d.fns.gather_taps.handle;
    var gather: ?graph.Node = null;
    var nr: usize = 0;
    var np: usize = 0;
    for (try source.nodes(all)) |node| {
        var p: cuda.abi.KernelNodeParams = undefined;
        try d.e.d.check(d.e.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
        if (p.func == want) {
            if (gather != null) return error.BadDraftGraph;
            gather = node;
        } else if (p.func == d.fns.norm_rope.handle) {
            if (nr == layers) return error.BadDraftGraph;
            ropes[nr] = node;
            nr += 1;
        } else if (p.func == d.fns.ring_put.handle) {
            if (np == layers) return error.BadDraftGraph;
            rings[np] = node;
            np += 1;
        }
    }
    if (gather == null or nr != layers or np != layers) return error.BadDraftGraph;
    const exec = source.instantiate() catch |err| {
        if (err != error.GraphNotPermitted) return err;
        source.deinit();
        d.gpa.free(ropes);
        d.gpa.free(rings);
        return null;
    };
    return .{ .source = source, .exec = exec, .gather = gather.?, .ropes = ropes, .rings = rings };
}

/// Runs one live chunk from its graph; false: run it eagerly (off, a prompt chunk, or a graph the driver refused).
pub fn absorb(d: anytype, ctx: anytype, ring: u64, start: u64, rows: usize, rs: ?[]const u32, prompt: bool) !bool {
    if (prompt or rows > 16 or !live(d)) return false;
    const key: u32 = @intCast(rows | @as(usize, @intFromBool(rs != null)) << 8);
    const gop = try d.graphs.absorbs.getOrPut(d.gpa, key);
    if (!gop.found_existing) {
        errdefer _ = d.graphs.absorbs.remove(key);
        gop.value_ptr.* = (try capture(d, ctx, ring, start, rows, rs)) orelse {
            _ = d.graphs.absorbs.remove(key);
            d.graphs.on = false;
            return false;
        };
    }
    const a = gop.value_ptr;
    const r: u32 = @intCast(rows);
    if (rs) |list| {
        try set(a.exec, a.gather, d.fns.gather_rows, .{ r, 1, 1 }, 256, rowsArgs(d, ring, start, rows, list));
    } else try set(a.exec, a.gather, d.fns.gather_taps, .{ r, 1, 1 }, 256, tapsArgs(d, ring, start));
    for (d.w.layers, a.ropes, a.rings, 0..) |l, rope, put, i| {
        try set(a.exec, rope, d.fns.norm_rope, .{ r * @as(u32, @intCast(d.c.kv_heads)), 1, 1 }, 32, ropeArgs(d, l.k_norm, start));
        try set(a.exec, put, d.fns.ring_put, .{ r, 1, 1 }, 256, ringArgs(d, ctx, i, start));
    }
    try a.exec.launchOn(d.e.stream);
    return true;
}
/// The launches and captured host-copy destinations vary with every stream's width class.
fn holdKey(streams: usize, contexts: anytype, narrow: bool, projection_role: u8) !HoldKey {
    if (streams != contexts.len or streams > 128) return error.BadDraftGraph;
    var width_mask: u128 = 0;
    for (contexts, 0..) |ctx, i| {
        if (ctx.wide) width_mask |= @as(u128, 1) << @intCast(i);
    }
    return .{ .streams = @intCast(streams), .narrow = narrow, .projection_role = projection_role, .width_mask = width_mask };
}

/// Eight resident variants per stream count bound graph memory; a ninth evicts one.
fn boundHolds(g: *Graphs, streams: u8) ?HoldKey {
    var count: usize = 0;
    var evict: ?HoldKey = null;
    var it = g.holds.iterator();
    while (it.next()) |entry| if (entry.key_ptr.streams == streams) {
        count += 1;
        if (evict == null) evict = entry.key_ptr.*;
    };
    if (count < 8) return null;
    const key = evict.?;
    g.holds.getPtr(key).?.exec.deinit();
    _ = g.holds.remove(key);
    return key;
}

/// A hold frame's kernels from their graph, keyed by what changes them; returns the packed logits when spanned.
pub fn hold(d: anytype, streams: usize, contexts: anytype, narrow: bool, comptime body: anytype) !?u64 {
    if (!live(d)) {
        if (d.graphs.stats.on) try d.graphs.stats.note(d.gpa, try holdKey(streams, contexts, narrow, @backingInt(d.projection_role)), "eager");
        return body(d, streams, contexts);
    }
    const key = try holdKey(streams, contexts, narrow, @backingInt(d.projection_role));
    if (d.graphs.holds.get(key)) |h| {
        try d.graphs.stats.note(d.gpa, key, "replayed");
        try h.exec.launchOn(d.e.stream);
        return h.spanned;
    }
    try d.graphs.stats.note(d.gpa, key, "key_miss");
    const stream = d.e.stream;
    try graph.beginCapture(stream, .thread_local);
    const spanned = body(d, streams, contexts) catch |err| {
        var broken = graph.endCapture(stream) catch return err;
        broken.deinit();
        return err;
    };
    var source = try graph.endCapture(stream);
    defer source.deinit();
    var exec = source.instantiate() catch |err| {
        if (err != error.GraphNotPermitted) return err;
        try d.graphs.stats.note(d.gpa, key, "eager");
        d.graphs.on = false;
        return body(d, streams, contexts);
    };
    if (boundHolds(&d.graphs, key.streams)) |old| try d.graphs.stats.note(d.gpa, old, "eviction");
    try d.graphs.stats.note(d.gpa, key, "captured");
    d.graphs.holds.put(d.gpa, key, .{ .exec = exec, .spanned = spanned }) catch |err| {
        exec.deinit();
        return err;
    };
    try exec.launchOn(stream); // on failure the cache still owns the graph
    return spanned;
}

fn writeCount(w: anytype, key: HoldKey, c: Count) !void {
    try w.print("qwen draft graphs: streams={d} narrow={} role={d} width_mask={x:0>32} captured={d} replayed={d} eager={d} key_miss={d} eviction={d} width_not_graphable={d}\n", .{ key.streams, key.narrow, key.projection_role, key.width_mask, c.captured, c.replayed, c.eager, c.key_miss, c.eviction, c.width_not_graphable });
}

pub fn writeStats(g: *const Graphs, w: anytype) !void {
    var it = g.stats.counts.iterator();
    while (it.next()) |entry| try writeCount(w, entry.key_ptr.*, entry.value_ptr.*);
}
