//! Flash Next on one HIP device: the kernels, a stream, the rank's weights, the host n-gram tables and the scratch.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const serve = @import("hip_serve");
const config = @import("../config.zig");
const model = @import("model.zig");
const load = @import("load.zig");
const tables = @import("tables.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");
const index = @import("../index.zig");

pub const Options = struct {
    device: c_int = 0,
    rank: usize = 0,
    world: usize = 1,
    group: ?hip.collective.Group = null,
    policy: hip.Policy = .{},
    /// Positions a stream holds at most (its prompt, reply and a window's rows).
    capacity: usize = 4096,
    /// Prompt rows a forward takes at most: a multiple of 64, so a prompt's bits do not depend on where it is cut.
    chunk: usize = 512,
};

/// Bytes of the pinned staging a forward's uploads and n-gram rows go through.
const staging_bytes = 64 << 20;

/// A round's shape: each stream's cache slot and rows; one graph serves every later round of it.
pub const Key = struct {
    count: u32 = 0,
    slots: [forward.max_segs]u32 = @splat(0),
    rows: [forward.max_segs]u8 = @splat(0),
};

/// What a round's launches leave: the forward's rows, the logits and where the arena stands after them.
pub const Done = struct { out: forward.Out, logits: hip.ops.Tensor, used: usize };

/// A round's replay: graphs cut at its collectives, which run between them (RCCL calls record slowly in a graph).
pub const Graphs = hip.graph_cache.CacheOf(Key, Done, hip.graph_split.Split);

/// What a round does with its shape's graph (rank 0's choice, which every rank makes too).
pub const Pick = enum(u32) { eager, replay, capture };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: hip.Runtime,
    ctx: hip.Context,
    lib: hip.Launcher,
    stream: hip.Stream,
    o: Options,
    cfg: config.Config,
    ck: core.checkpoint.Checkpoint,
    /// The n-gram tables' row views and other loader leftovers that live as long as the engine.
    keep: std.heap.ArenaAllocator,
    weights: load.Weights,
    tables: tables.Tables,
    host: forward.Host,
    qk: [2]hip.DeviceBuffer,
    arena: hip.Arena,
    drawer: serve.draw.Drawer,
    comm: ?hip.collective.Comm = null,
    act: model.Kind,
    eos: i64,
    /// Rounds replay their shape's graph unless the policy says `graphs=off`.
    graphed: bool,
    graphs: Graphs,
    chosen: ?*Graphs.Entry = null,
    /// Streams' caches at the full capacity, kept between streams so their graphs stay valid (null: not made yet).
    pool: std.ArrayList(?cache.Caches) = .empty,

    /// The checkpoint `dir` on device `o.device`, this rank's share loaded; the scratch sized for `o.chunk` rows.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.o = o;
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const caps = try hip.Device.kernelCaps(&e.driver, o.device);
        e.lib = try hip.Launcher.load(&e.driver, o.policy.choices(), hip.kernels.images);
        errdefer e.lib.unload();
        e.act = if (caps.act == .f16) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        e.comm = null;
        if (o.world > 1) e.comm = try (o.group orelse return error.NoGroup).open();
        errdefer if (e.comm) |c| c.close();
        e.cfg = try config.Config.read(gpa, io, dir);
        errdefer e.cfg.deinit();
        e.ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        errdefer e.ck.close();
        e.keep = .init(gpa);
        errdefer e.keep.deinit();
        const rank: ?core.quant.Rank = if (o.world > 1) .{ .rank = o.rank, .world = o.world } else null;
        e.weights = try load.load(gpa, &e.driver, &e.ck, &e.cfg, rank, e.act);
        errdefer e.weights.deinit();
        e.weights.model.tp = e.comm;
        const spelling = try spellingOf(e.keep.allocator(), io, dir);
        e.tables = try tables.Tables.init(e.keep.allocator(), &e.ck, &e.cfg, e.weights.model.ple.layer, spelling);
        const c = mixers.qkConstants(e.cfg.key_dim);
        var w: [2]hip.DeviceBuffer = undefined;
        for (&w, [_]f32{ c.q, c.k }) |*b, v| {
            const vals = try gpa.alloc(f32, e.cfg.key_dim);
            defer gpa.free(vals);
            @memset(vals, v);
            b.* = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(vals));
        }
        e.qk = w;
        errdefer for (&e.qk) |*b| b.free();
        e.weights.model.qk = .{ .q = e.qk[0].base(), .k = e.qk[1].base(), .eps = c.eps };
        e.host = .{ .gpa = gpa, .d = &e.driver, .staging = try hip.HostBuffer.alloc(&e.driver, staging_bytes), .tables = &e.tables };
        errdefer e.host.staging.free();
        e.eos = e.tables.hasher.ref.eos;
        e.arena = try hip.Arena.init(&e.driver, try e.scratchBytes());
        errdefer e.arena.deinit();
        e.drawer = try serve.draw.Drawer.init(gpa, &e.driver, if (e.act == .f16) .f16 else .bf16, e.cfg.vocab, cache.max_window * forward.max_segs);
        e.graphed = o.policy.graphs != .off;
        e.graphs = Graphs.init(gpa);
        e.chosen = null;
        e.pool = .empty;
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.graphs.captured > 0 and e.o.rank == 0) std.log.info("graphs: {d} captured ({d:.0} ms each), {d} of {d} rounds replayed", .{ e.graphs.captured, @as(f64, @floatFromInt(e.graphs.capture_ns)) / @as(f64, @floatFromInt(e.graphs.captured)) / 1e6, e.graphs.replayed, e.graphs.rounds });
        e.graphs.deinit();
        for (e.pool.items) |*c| if (c.*) |*x| x.deinit(e.gpa);
        e.pool.deinit(e.gpa);
        e.drawer.deinit();
        e.arena.deinit();
        e.host.staging.free();
        for (&e.qk) |*b| b.free();
        e.weights.deinit();
        e.keep.deinit();
        e.ck.close();
        e.cfg.deinit();
        if (e.comm) |c| c.close();
        e.stream.deinit();
        e.lib.unload();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn model_(e: *const Engine) *const model.Model {
        return &e.weights.model;
    }

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .l = &e.lib, .bf16 = e.act == .bf16, .stream = e.stream.handle, .arena = arena };
    }

    /// The layer kinds as the caches lay them out: true for a DeltaNet layer.
    pub fn kinds(e: *const Engine, out: []bool) ![]const bool {
        for (out[0..e.cfg.layers], 0..) |*k, i| k.* = (try e.cfg.kind(i)) == .linear_attention;
        return out[0..e.cfg.layers];
    }

    /// The least of `mine` over the ranks, the same on every rank (the streams every rank's memory holds).
    pub fn least(e: *Engine, mine: u64) !u64 {
        const c = e.comm orelse return mine;
        const o = e.ops(&e.arena);
        const at = e.arena.mark();
        defer e.arena.release(at);
        const send = try o.arena.of(i64, 1 + c.world);
        const word: i64 = @intCast(mine);
        try hip.runtime.check(e.driver.api.hipMemcpyHtoDAsync(@ptrFromInt(send), @ptrCast(@constCast(&word)), 8, e.stream.handle));
        try c.allGather(send, send + 8, 1, .i64, e.stream.handle);
        var all: [64]i64 = undefined;
        try hip.runtime.check(e.driver.api.hipMemcpyDtoHAsync(std.mem.sliceAsBytes(&all).ptr, @ptrFromInt(send + 8), 8 * c.world, e.stream.handle));
        try e.stream.synchronize();
        var out = mine;
        for (all[0..c.world]) |v| out = @min(out, @as(u64, @intCast(v)));
        return out;
    }

    /// A stream's caches for `cap` positions.
    pub fn caches(e: *Engine, cap: usize) !cache.Caches {
        var buf: [config.max_layers]bool = undefined;
        return cache.Caches.init(e.gpa, &e.driver, e.weights.model.dims, try e.kinds(&buf), cap, e.act, e.eos);
    }

    /// The scratch the largest forward takes: a prompt chunk at the capacity's end, counted on a dry run.
    fn scratchBytes(e: *Engine) !usize {
        var counting = hip.Arena.counting();
        var o = e.ops(&counting);
        o.stream = hip.counting;
        var c = try e.caches(@max(e.o.capacity, e.o.chunk) + cache.max_window);
        defer c.deinit(e.gpa);
        const tokens = try e.gpa.alloc(u32, e.o.chunk);
        defer e.gpa.free(tokens);
        @memset(tokens, 0);
        const before = e.host.used;
        defer e.host.used = before;
        const segs = [_]forward.Seg{.{ .c = &c, .tokens = tokens, .pos = e.o.capacity -| e.o.chunk }};
        const out = try forward.run(o, &e.weights.model, &e.host, &segs, false, try forward.stage(&e.weights.model, &e.host, &segs, false));
        _ = try e.logits(o, out.mixed, e.o.chunk);
        return counting.used + (16 << 20);
    }

    /// The output head over `rows` final rows: whole logits rows (act dtype), joined across ranks in rank order.
    pub fn logits(e: *Engine, o: hip.ops.Ops, mixed: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        const m = &e.weights.model;
        const y = try o.project(mixed, m.head, rows, false);
        const c = m.tp orelse return y;
        const width = m.head.n;
        const size = y.kind.size();
        const parts = try o.arena.take(c.world * rows * width * size);
        try hip.reduce.gather(o, c, y, parts, rows * width);
        const out = try o.arena.take(c.world * rows * width * size);
        if (o.stream == hip.counting) return .{ .ptr = out, .kind = y.kind };
        for (0..rows) |row| for (0..c.world) |r| {
            const to = out + (row * c.world + r) * width * size;
            const from = parts + (r * rows + row) * width * size;
            try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(to), @ptrFromInt(from), width * size, o.stream));
        };
        return .{ .ptr = out, .kind = y.kind };
    }

    /// The first `kept` rows of a window from `c.len` become the stream's: DeltaNet states, n-gram tail and history.
    pub fn keepWindow(e: *Engine, c: *cache.Caches, tokens: []const u32, kept: usize) !void {
        if (kept == 0 or kept > tokens.len) return error.BadKeep;
        const l = c.layout;
        const s = e.stream.handle;
        if (kept < tokens.len) for (c.linear, 0..) |linear, i| {
            if (!linear) continue;
            const from = c.snapshot(i, kept - 1);
            const to = c.linearAt(i);
            try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(to.conv), @ptrFromInt(from.conv), l.conv_bytes, s));
            try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(to.state), @ptrFromInt(from.state), l.state_bytes, s));
        };
        const row = l.tail_bytes / l.tail_rows;
        try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(c.tail()), @ptrFromInt(c.cin.base() + kept * row), l.tail_bytes, s));
        c.history = @import("ngram.zig").after(c.history, tokens[0..kept]);
        c.len += kept;
        c.pooled = @min(c.pooled, c.len / e.cfg.index_ratio);
    }

    /// A prompt's `tokens` from `c.len` into `c`, kept: the final rows of the last of them.
    pub fn prompt(e: *Engine, c: *cache.Caches, tokens: []const u32) !forward.Out {
        // the staging is reused: the last forward's uploads must have left it
        try e.stream.synchronize();
        e.host.used = 0;
        e.arena.reset();
        const segs = [_]forward.Seg{.{ .c = c, .tokens = tokens, .pos = c.len }};
        const staged = try forward.stage(&e.weights.model, &e.host, &segs, false);
        const out = try forward.run(e.ops(&e.arena), &e.weights.model, &e.host, &segs, false, staged);
        c.len += tokens.len;
        return out;
    }

    /// The graph choice for a round of windows `segs`, from its shape's history; every rank makes it in the same order.
    pub fn choose(e: *Engine, segs: []const forward.Seg) !Pick {
        e.chosen = null;
        if (!e.graphed) return .eager;
        const before = e.graphs.entries.items.len;
        const entry = try e.graphs.find(try keyOf(e, segs));
        e.chosen = entry;
        // a shape is recorded the second time it is met: a stream's first rounds of each width run eagerly
        if (e.graphs.entries.items.len > before) return .eager;
        return switch (entry.state) {
            .ready => .replay,
            .seen => .capture,
            .failed => .eager,
        };
    }

    /// A round of windows and its logits as `pick` says (rank 0's, after every rank's `choose`).
    pub fn round(e: *Engine, segs: []const forward.Seg, pick: Pick) !Done {
        defer for (segs) |g| {
            g.c.pooled = (g.pos + g.tokens.len) / e.cfg.index_ratio;
        };
        try e.stream.synchronize();
        e.host.used = 0;
        e.arena.reset();
        const staged = try forward.stage(&e.weights.model, &e.host, segs, true);
        const began = nowNs(e.io);
        defer e.graphs.submit_ns += @intCast(nowNs(e.io) - began);
        e.graphs.rounds += 1;
        const ctx: Body = .{ .e = e, .segs = segs, .staged = staged };
        const entry = e.chosen orelse return body(ctx);
        e.chosen = null;
        switch (pick) {
            .eager => return body(ctx),
            .replay => {
                if (entry.state != .ready) return error.GraphsDisagree;
                e.arena.used = entry.out.used;
                try entry.exec.?.launchOn(e.stream);
                e.graphs.replayed += 1;
                return entry.out;
            },
            .capture => {
                const at = e.arena.mark();
                const kept = try e.record(entry, ctx);
                e.graphs.capture_ns += @intCast(nowNs(e.io) - began);
                if (!(try e.agreed(kept))) {
                    e.graphs.revoke(entry);
                    e.arena.release(at);
                    return body(ctx);
                }
                try entry.exec.?.launchOn(e.stream);
                return entry.out;
            },
        }
    }

    /// Records a round into `entry` (nothing runs): whether this rank kept it.
    fn record(e: *Engine, entry: *Graphs.Entry, ctx: Body) !bool {
        var rec: hip.graph_split.Recorder = .{ .gpa = e.gpa, .stream = e.stream, .inner = e.comm };
        rec.begin() catch return false;
        const m = &e.weights.model;
        const tp = m.tp;
        m.tp = rec.comm();
        const got = body(ctx);
        m.tp = tp;
        var split = (try rec.finish()) orelse return false;
        const out = got catch |err| {
            split.deinit();
            return if (err == error.OutOfDeviceMemory) err else false;
        };
        if (e.o.policy.graph_fail >= 0 and @as(usize, @intCast(e.o.policy.graph_fail)) == e.o.rank) {
            split.deinit();
            return false;
        }
        e.graphs.hold(entry, split, out);
        return true;
    }

    /// Slot `slot`'s caches, fresh, for a new stream (made at the capacity the first time).
    pub fn lease(e: *Engine, slot: u32) !cache.Caches {
        while (e.pool.items.len <= slot) try e.pool.append(e.gpa, null);
        var c = if (e.pool.items[slot]) |held| held else try e.caches(e.o.capacity);
        e.pool.items[slot] = null;
        errdefer c.deinit(e.gpa);
        try e.stream.synchronize();
        try c.reset(e.eos);
        c.id = slot;
        return c;
    }

    /// A finished stream's caches back to their slot.
    pub fn unlease(e: *Engine, c: cache.Caches) void {
        e.pool.items[c.id] = c;
    }

    /// Whether every rank kept its capture (one word each, gathered).
    fn agreed(e: *Engine, yes: bool) !bool {
        return try e.least(@intFromBool(yes)) == 1;
    }
};

const Body = struct { e: *Engine, segs: []const forward.Seg, staged: forward.Staged };

/// A round's launches: the staged inputs' copy, the forward and the logits.
fn body(ctx: Body) anyerror!Done {
    const e = ctx.e;
    const o = e.ops(&e.arena);
    const out = try forward.run(o, &e.weights.model, &e.host, ctx.segs, true, ctx.staged);
    var rows: usize = 0;
    for (ctx.segs) |g| rows += g.tokens.len;
    const logits = try e.logits(o, out.mixed, rows);
    return .{ .out = out, .logits = logits, .used = e.arena.used };
}

fn keyOf(e: *const Engine, segs: []const forward.Seg) !Key {
    if (segs.len > forward.max_segs) return error.TooManyStreams;
    var k: Key = .{ .count = @intCast(segs.len) };
    for (segs, 0..) |g, i| {
        if (g.c.layout.cap != e.o.capacity) return error.CachesNotPooled;
        k.slots[i] = g.c.id;
        k.rows[i] = @intCast(g.tokens.len);
    }
    return k;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

/// The index's spelling of its n-gram shards (`shard_` or oMLX's `shards.`).
fn spellingOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const map = doc.object.get("weight_map") orelse return error.InvalidFlashIndex;
    return index.ngramSpelling(map.object);
}
