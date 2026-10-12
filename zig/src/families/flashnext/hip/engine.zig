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
    policy: hip.Policy = .{},
    /// Positions a stream holds at most (its prompt, reply and a window's rows).
    capacity: usize = 4096,
    /// Prompt rows a forward takes at most: a multiple of 64, so a prompt's bits do not depend on where it is cut.
    chunk: usize = 512,
};

/// Bytes of the pinned staging a forward's uploads and n-gram rows go through.
const staging_bytes = 64 << 20;

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
    act: model.Kind,
    eos: i64,

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
        e.cfg = try config.Config.read(gpa, io, dir);
        errdefer e.cfg.deinit();
        e.ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        errdefer e.ck.close();
        e.keep = .init(gpa);
        errdefer e.keep.deinit();
        e.weights = try load.load(gpa, &e.driver, &e.ck, &e.cfg, e.act);
        errdefer e.weights.deinit();
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
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.drawer.deinit();
        e.arena.deinit();
        e.host.staging.free();
        for (&e.qk) |*b| b.free();
        e.weights.deinit();
        e.keep.deinit();
        e.ck.close();
        e.cfg.deinit();
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
        const out = try forward.run(o, &e.weights.model, &e.host, &.{.{ .c = &c, .tokens = tokens, .pos = e.o.capacity -| e.o.chunk }}, false);
        _ = try e.logits(o, out.mixed, e.o.chunk);
        return counting.used + (16 << 20);
    }

    /// The output head over `rows` final rows: whole logits rows (act dtype).
    pub fn logits(e: *Engine, o: hip.ops.Ops, mixed: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        return o.project(mixed, e.weights.model.head, rows, false);
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
        const out = try e.run(c, tokens, c.len, false);
        c.len += tokens.len;
        return out;
    }

    /// `tokens` from `pos` through the forward into `c`; a prompt's rows, or a window's (kept later).
    pub fn run(e: *Engine, c: *cache.Caches, tokens: []const u32, pos: usize, window: bool) !forward.Out {
        // the staging is reused: the last forward's uploads must have left it
        try e.stream.synchronize();
        e.host.used = 0;
        e.arena.reset();
        return e.runMany(&.{.{ .c = c, .tokens = tokens, .pos = pos }}, window);
    }

    /// Several streams' rows in one forward (every segment from its own caches): windows of a shared round.
    pub fn runMany(e: *Engine, segs: []const forward.Seg, window: bool) !forward.Out {
        try e.stream.synchronize();
        e.host.used = 0;
        e.arena.reset();
        return forward.run(e.ops(&e.arena), &e.weights.model, &e.host, segs, window);
    }
};

/// The index's spelling of its n-gram shards (`shard_` or oMLX's `shards.`).
fn spellingOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const map = doc.object.get("weight_map") orelse return error.InvalidFlashIndex;
    return index.ngramSpelling(map.object);
}
