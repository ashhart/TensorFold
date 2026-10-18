//! The DFlash2 drafter on CUDA behind core/lanes/drafter.zig: a tap ring a stream, 7 drafts as a chain or a tree.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const weights = @import("cuda_draft_weights.zig");
const tree = @import("cuda_draft_tree.zig");

const E = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const dr = lanes.drafter;
const Stream = lanes.Stream;

/// Rows an absorb runs at once (the target's prompt chunks are at most this).
const max_rows = engine.max_rows;
/// The logprob row's words (sample.cu's LP_WORDS) and its best-token capacity.
const lp_words = 3 + 2 * 20;
/// A hold's rows: the reference's 16-row blocks (dflash2.block), past the 8 the checkpoint was trained with.
const lp_top = 20;

pub const Config = @import("cuda_draft_config.zig").Config;
pub const frame = @import("cuda_draft_frame.zig");
const Fns = struct { gather_taps: cuda.Function, gather_rows: cuda.Function, dyn_conv: cuda.Function, norm_rope: cuda.Function, ring_put: cuda.Function, attn: cuda.Function, dyn_conv_batch: cuda.Function, norm_rope_batch: cuda.Function, mask_rows: cuda.Function, attn_partial: cuda.Function, attn_merge: cuda.Function };

/// qwen35_draft.cu's TapRows.
const TapRows = extern struct { n: c_int, rows: [engine.most_window]c_int };

/// A stream's context: its layers' key and value rings over [begin, end), and the drafts it holds (a chain or tree).
pub const Ctx = struct {
    ring: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    held: [engine.most_window]u32 = undefined,
    held_n: usize = 0,
    // a grown tree's parents (-1: the pending token) and nodes' log-chances under them; 0 nodes: a chain
    parents: [engine.most_window]i32 = undefined,
    own: [engine.most_window]f64 = undefined,
    grown: usize = 0,
};

pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *E,
    c: Config,
    w: weights.Weights,
    mod: cuda.Module,
    fns: Fns,
    scratch: DeviceBuffer,
    selection_base: usize = 0,
    segments: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    po: u64 = 0,
    at: usize = 0,
    // absorb rows (max_rows) and block rows (block), all bf16 unless said
    taps_in: u64,
    th: u64,
    thn: u64,
    kabs: u64,
    kn: u64,
    vabs: u64,
    xs: u64, // 64-group sums for the projections (f32)
    h: u64,
    h2: u64,
    x: u64,
    xin: u64,
    dyn: u64,
    qraw: u64,
    q: u64,
    kraw: u64,
    kb: u64,
    vb: u64,
    att: u64,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    logits: u64,
    words: u64, // u32 logprob words a row
    parts: u64, // topkRows' slice candidates
    hp: u64,
    ids: u64, // u32 the block's ids
    staged: cuda.HostBuffer, // pinned: the ids up, the words and projected rows down
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    /// Contexts between streams, allocated and run once (`reserve`), as the driver stalls a buffer's first kernel.
    pool: std.ArrayList(*Ctx) = .empty,
    /// Grown trees' nodes offered and landed, by own chance: the nodes' chances the round loop prices.
    landing: tree.Landing = .{},
    /// One level of a hold (a hold drafts every level at once; timed at load, over the block's levels).
    step_ms: f64 = 1.0,
    /// The target's window ms by rows (cuda_native.zig times them), for the round loop's prices.
    costs: [engine.most_window]lanes.config.Cost = undefined,
    n_costs: usize = 0,

    const words_at = frame.words_at;
    const hp_at = frame.hp_at;
    const stagedLen = frame.stagedLen;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8) !*Draft {
        const c = try Config.read(gpa, io, dir, e.g);
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .c = c, .w = undefined, .mod = undefined, .fns = undefined, .scratch = undefined, .taps_in = 0, .th = 0, .thn = 0, .kabs = 0, .kn = 0, .vabs = 0, .xs = 0, .h = 0, .h2 = 0, .x = 0, .xin = 0, .dyn = 0, .qraw = 0, .q = 0, .kraw = 0, .kb = 0, .vb = 0, .att = 0, .o = 0, .gate = 0, .up = 0, .act = 0, .logits = 0, .words = 0, .parts = 0, .hp = 0, .ids = 0, .staged = undefined };
        d.mod = try cuda.Module.load(e.d, cuda.kernels.qwen35_draft);
        errdefer d.mod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(d.fns, name) = try d.mod.function("tf_q35d_" ++ name);
        d.w = try weights.load(gpa, io, e, dir, .{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .vocab = c.vocab, .q_width = c.qWidth(), .kv_width = c.kvWidth(), .head_dim = c.head_dim, .dynamic = c.dynamic(), .rank = c.rank, .taps = c.n_taps });
        errdefer d.w.deinit(gpa);
        d.scratch = try DeviceBuffer.alloc(e.d, try d.scratchBytes());
        errdefer d.scratch.free();
        try d.scratch.fill8(0, null);
        d.at = 0;
        _ = try d.layout(true);
        d.staged = try cuda.HostBuffer.alloc(e.d, try frame.stagedBytes(c.rank));
        errdefer d.staged.free();
        // the first context, and a hold timed on it (the quickest of several: the first ones pay the driver's stalls)
        const ctx = try d.create();
        errdefer d.drop(ctx);
        try d.absorb(ctx, d.taps_in, 0, 1, null);
        var quickest: f64 = std.math.inf(f64);
        for (0..8) |_| {
            ctx.end = 1;
            const t0 = std.Io.Clock.awake.now(io);
            try d.hold(ctx, 0, 1, c.block - 1, null);
            quickest = @min(quickest, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6);
        }
        d.step_ms = quickest / @as(f64, @floatFromInt(c.block - 1));
        ctx.* = .{ .ring = ctx.ring };
        try d.pool.append(gpa, ctx);
        return d;
    }

    /// The drafter's taps (the target's layers it reads).
    pub fn taps(d: *const Draft) []const u32 {
        return d.c.taps[0..d.c.n_taps];
    }

    /// The target's measured window cost, published after the complete reset-cache width sweep.
    pub fn setCost(d: *Draft, rows: u32, ms: f64) void {
        if (platform.c.getenv("TF_TIMING") != null) std.debug.print("[timing] window {d} rows {d:.2} ms (a hold level {d:.2})\n", .{ rows, ms, d.step_ms });
        if (d.n_costs < d.costs.len) {
            d.costs[d.n_costs] = .{ .width = rows, .ms = ms };
            d.n_costs += 1;
        }
    }

    pub fn deinit(d: *Draft) void {
        var it = d.streams.valueIterator();
        while (it.next()) |ctx| d.drop(ctx.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |ctx| d.drop(ctx);
        d.pool.deinit(d.gpa);
        d.staged.free();
        d.scratch.free();
        d.w.deinit(d.gpa);
        d.mod.unload();
        d.gpa.destroy(d);
    }

    fn scratchBytes(d: *Draft) !usize {
        d.at = 0;
        return d.layout(false);
    }

    fn bytes(factors: anytype) !usize {
        var value: usize = 1;
        inline for (factors) |factor| value = try std.math.mul(usize, value, factor);
        return value;
    }

    /// The scratch rows, measured (nothing assigned) or assigned from the scratch buffer.
    pub fn layout(d: *Draft, assign: bool) !usize {
        const c = d.c;
        const B = c.block * frame.max_streams;
        const R = @max(max_rows, B);
        const sizes = .{
            .{ "taps_in", try bytes(.{ R, c.n_taps, c.hidden, 2 }) },                   .{ "th", try bytes(.{ R, c.hidden, 2 }) },              .{ "thn", try bytes(.{ R, c.hidden, 2 }) },                         .{ "kabs", try bytes(.{ R, c.kvWidth(), 2 }) },
            .{ "kn", try bytes(.{ R, c.kvWidth(), 2 }) },                               .{ "vabs", try bytes(.{ R, c.kvWidth(), 2 }) },         .{ "xs", try bytes(.{ R, c.n_taps, (c.hidden / 64), 4 }) },         .{ "h", try bytes(.{ B, c.hidden, 2 }) },
            .{ "h2", try bytes(.{ B, c.hidden, 2 }) },                                  .{ "x", try bytes(.{ B, c.hidden, 2 }) },               .{ "xin", try bytes(.{ B, c.hidden, 2 }) },                         .{ "dyn", try bytes(.{ B, c.dynamic(), 2 }) },
            .{ "qraw", try bytes(.{ B, c.qWidth(), 2 }) },                              .{ "q", try bytes(.{ B, c.qWidth(), 2 }) },             .{ "kraw", try bytes(.{ B, c.kvWidth(), 2 }) },                     .{ "kb", try bytes(.{ B, c.kvWidth(), 2 }) },
            .{ "vb", try bytes(.{ B, c.kvWidth(), 2 }) },                               .{ "att", try bytes(.{ B, c.qWidth(), 2 }) },           .{ "o", try bytes(.{ B, c.hidden, 2 }) },                           .{ "gate", try bytes(.{ B, c.intermediate, 2 }) },
            .{ "up", try bytes(.{ B, c.intermediate, 2 }) },                            .{ "act", try bytes(.{ B, c.intermediate, 2 }) },       .{ "logits", try bytes(.{ B, c.vocab, 2 }) },                       .{ "words", try bytes(.{ B, lp_words, 4 }) },
            .{ "segments", try bytes(.{ frame.max_streams, @sizeOf(frame.Segment) }) }, .{ "pm", try bytes(.{ B, c.heads, frame.chunks, 4 }) }, .{ "pl", try bytes(.{ B, c.heads, frame.chunks, 4 }) },             .{ "po", try bytes(.{ B, c.heads, frame.chunks, c.head_dim, 4 }) },
            .{ "hp", try bytes(.{ B, c.rank, 2 }) },                                    .{ "ids", try bytes(.{ B, 4 }) },                       .{ "parts", cuda.sample.Draws.topkPartsBytes(c.vocab, c.topk, B) },
        };
        var total: usize = 0;
        inline for (sizes) |s| {
            const at = (try std.math.add(usize, total, 255)) & ~@as(usize, 255);
            total = try std.math.add(usize, at, s[1]);
            if (assign) @field(d, s[0]) = d.scratch.ptr + at;
        }
        return total;
    }

    fn drop(d: *Draft, ctx: *Ctx) void {
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    /// A new context, its ring touched once (an absorb, then a draft), so no request pays the driver's stall.
    fn create(d: *Draft) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try DeviceBuffer.alloc(d.e.d, d.c.ringBytes()) };
        errdefer ctx.ring.free();
        try ctx.ring.fill8(0, null);
        try d.absorb(ctx, d.taps_in, 0, 1, null); // any tap rows do: the drafter's own zeroed staging
        try d.hold(ctx, 0, 1, d.c.block - 1, null);
        ctx.* = .{ .ring = ctx.ring };
        return ctx;
    }

    /// `n` contexts ready in the pool, for the streams the server admits.
    pub fn reserve(d: *Draft, n: usize) !void {
        try d.pool.ensureTotalCapacity(d.gpa, n);
        while (d.pool.items.len < n) d.pool.appendAssumeCapacity(try d.create());
    }

    fn ringK(d: *Draft, ctx: *Ctx, layer: usize) u64 {
        return ctx.ring.ptr + layer * 2 * d.c.ringRows() * d.c.kvWidth() * 2;
    }

    fn ringV(d: *Draft, ctx: *Ctx, layer: usize) u64 {
        return d.ringK(ctx, layer) + d.c.ringRows() * d.c.kvWidth() * 2;
    }

    /// out (rows, n) = x (rows, k) through `l`, as the target's decode tiles or prompt GEMM, with the drafter's sums.
    pub fn proj(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
        return d.projSums(l, x, out, rows, true);
    }

    /// proj over the x of an earlier proj or norm (nothing between writing x or xs): its group sums are in xs.
    pub fn projAgain(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
        return d.projSums(l, x, out, rows, false);
    }

    pub fn projSums(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize, sums: bool) !void {
        const e = d.e;
        if (rows > 16) return e.affine.prompt(e.stream, x, l, out, rows);
        if (sums) try e.run(e.fns.sums, .{ rows, (l.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
        try e.affine.decode(e.stream, x, d.xs, l, out, rows);
    }

    /// out = the normed h, and its group sums into xs for a projAgain over out.
    pub fn norm(d: *Draft, h: u64, w: u64, out: u64, rows: usize) !void {
        const e = d.e;
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, @as(u64, 0), w, out, d.c.eps, @as(c_int, @intCast(d.c.hidden)), @as(u64, 0), d.xs });
    }

    fn dynConv(d: *Draft, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, branch: usize) !void {
        try d.e.run(d.fns.dyn_conv, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ x, dyn, base, res, out, @as(c_int, @intCast(d.c.hidden)), @as(c_int, @intCast(d.c.group)), @as(c_int, @intCast(branch)) });
    }

    fn normRope(d: *Draft, x: u64, w: u64, out: u64, rows: usize, heads: usize, pos0: u64) !void {
        try d.e.run(d.fns.norm_rope, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * d.c.head_dim)), @as(c_longlong, @intCast(pos0)), d.c.eps, d.c.theta });
    }

    /// Target rows [start, start + n) from the tap ring into the context; with `rows`, ring rows start + rows[i].
    fn absorb(d: *Draft, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32) !void {
        const c = d.c;
        var start = start_;
        var n = n_;
        if (rows_) |rs| if (rs.len != n or n > engine.most_window) return error.BadDraftRows;
        if (start < ctx.end) ctx.end = start;
        if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
        if (n > c.window) {
            start += n - c.window;
            n = c.window;
            ctx.begin = start;
        }
        ctx.end = start;
        const e = d.e;
        while (n > 0) {
            const rows = @min(n, max_rows);
            if (rows_) |rs| {
                var tr: TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
                for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
                try e.run(d.fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(c.n_taps * c.hidden)), d.taps_in });
            } else try e.run(d.fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.n_taps * c.hidden)), d.taps_in });
            try d.proj(d.w.fc, d.taps_in, d.th, rows);
            try d.norm(d.th, d.w.hidden_norm, d.thn, rows);
            for (d.w.layers, 0..) |l, i| {
                try d.projAgain(l.k, d.thn, d.kabs, rows);
                try d.normRope(d.kabs, l.k_norm, d.kn, rows, c.kv_heads, start);
                try d.projAgain(l.v, d.thn, d.vabs, rows);
                try e.run(d.fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn, d.vabs, d.ringK(ctx, i), d.ringV(ctx, i), @as(c_int, @intCast(c.kvWidth())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.ringRows())) });
            }
            start += rows;
            n -= rows;
            ctx.end = start;
        }
        if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
    }

    /// A warmup or standalone hold goes through the same batched frame as shared lane holds.
    fn hold(d: *Draft, ctx: *Ctx, pending: u32, pos0: u64, depth: usize, shape: ?*const lanes.shape.Shape) !void {
        const item: dr.Hold = .{ .stream = undefined, .pending = pending, .position = pos0 + 1, .depth = @intCast(depth), .lanes = shape };
        try frame.forward(d, &.{item}, &.{ctx});
    }

    fn candidates(d: *const Draft, rows: usize) tree.Candidates {
        const base = d.selection_base;
        return .{ .words = d.staged.bytes[words_at + base * lp_words * 4 ..][0 .. rows * lp_words * 4], .projected = d.staged.bytes[hp_at + base * d.c.rank * 2 ..][0 .. rows * d.c.rank * 2], .topk = d.c.topk, .rank = d.c.rank, .predecessor = d.w.predecessor, .successor = d.w.successor };
    }

    pub fn select(d: *Draft, ctx: *Ctx, pending: u32, rows: usize) void {
        tree.chain(d.candidates(rows), pending, ctx.held[0..rows]);
    }

    pub fn selectLanes(d: *Draft, ctx: *Ctx, pending: u32, rows: usize, shape: *const lanes.shape.Shape) !void {
        if (shape.parents.len > ctx.held.len) return error.TooManyLanes;
        try tree.lanesOf(d.candidates(rows), pending, rows, shape, ctx.held[0..shape.parents.len]);
        ctx.held_n = shape.parents.len;
    }

    pub fn selectHold(d: *Draft, ctx: *Ctx, it: dr.Hold) !void {
        const rows = d.c.block - 1;
        ctx.grown = 0;
        if (it.lanes) |shape| return d.selectLanes(ctx, it.pending, rows, shape);
        const limit = if (d.c.block == 16) tree.nodes() else 0;
        if (limit > 0) {
            const n = @min(it.depth, limit);
            ctx.grown = tree.grow(d.candidates(rows), it.pending, rows, n, &ctx.held, &ctx.parents, &ctx.own);
            ctx.held_n = ctx.grown;
        } else {
            d.select(ctx, it.pending, rows);
            ctx.held_n = @min(it.depth, rows);
        }
    }

    // the Drafter contract

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .tree = treeFn, .chances = chancesFn } };
    }

    fn cast(p: *anyopaque) *Draft {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        const d = cast(p);
        return d.c.taps[0..d.c.n_taps];
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = cast(p);
        // TF_NO_TREES=1: chains only (for comparing)
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = true, .stream_bytes = d.c.ringBytes(), .trees = platform.c.getenv("TF_NO_TREES") == null, .window_costs = d.costs[0..d.n_costs], .lane_costs = d.costs[0..d.n_costs], .tree_nodes = @intCast(if (d.c.block == 16) tree.nodes() else 0) };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        const ctx = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, ctx) catch d.drop(ctx);
        ctx.* = .{ .ring = ctx.ring };
        try d.streams.put(d.gpa, s, ctx);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("absorb", d.io, t0);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
            if (it.offered > 0 and ctx.grown > 0) {
                var prefix: [engine.most_window]u32 = undefined;
                for (prefix[0..@min(it.follow.len, prefix.len)], 0..) |*x, i| x.* = @intCast(i);
                const kept = it.rows orelse prefix[0..@min(it.follow.len, prefix.len)];
                d.landing.observeTree(it.stream.sampling != null, ctx.parents[0..ctx.grown], ctx.own[0..ctx.grown], it.offered, kept);
            }
            ctx.grown = 0;
            // the target's features: its ring, `offset` the first row's position (cuda_native.zig's featuresFn)
            if (it.rows) |rs| {
                try d.absorb(ctx, it.features.buffer, it.features.offset, rs.len, rs);
            } else try d.absorb(ctx, it.features.buffer, it.features.offset, it.features.rows, null);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("hold", d.io, t0);
        if (items.len == 0) return;
        if (items.len > frame.max_streams) return error.BadDraftFrame;
        var contexts: [frame.max_streams]*Ctx = undefined;
        for (items, contexts[0..items.len]) |it, *ctx| ctx.* = d.streams.get(it.stream) orelse return error.UnknownStream;
        try frame.forward(d, items, contexts[0..items.len]);
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(p);
        for (streams, out) |s, o| {
            const ctx = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > ctx.held_n) return error.TooFewDrafts;
            @memcpy(o, ctx.held[0..o.len]);
        }
    }

    fn treeFn(p: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const d = cast(p);
        const ctx = d.streams.get(s) orelse return error.UnknownStream;
        if (ctx.grown == 0) return null;
        const tokens = try gpa.dupe(u32, ctx.held[0..ctx.grown]);
        errdefer gpa.free(tokens);
        return .{ .count = @intCast(ctx.grown), .tokens = tokens, .parents = try gpa.dupe(i32, ctx.parents[0..ctx.grown]) };
    }

    /// A grown tree's nodes' chances: each its landing chance under its parent times its parent's.
    fn chancesFn(p: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const d = cast(p);
        const ctx = d.streams.get(s) orelse return error.UnknownStream;
        if (ctx.grown == 0 or out.len > ctx.grown) return false;
        const sampled = s.sampling != null;
        for (out, ctx.parents[0..out.len], ctx.own[0..out.len]) |*o, parent, own| o.* = d.landing.chance(sampled, own) * (if (parent >= 0) out[@intCast(parent)] else 1);
        return true;
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }
};

/// Acceptance by depth until a stream has its own (the reference's chat text: measured, see the commit).
const prior = [_]f64{ 0.75, 0.65, 0.58, 0.52, 0.47, 0.43, 0.4, 0.37, 0.35, 0.33, 0.31, 0.29, 0.28, 0.27, 0.26 };

/// TF_TIMING=1: each call's wall time on stderr.
pub fn timing(what: []const u8, io: std.Io, t0: std.Io.Timestamp) void {
    if (platform.c.getenv("TF_TIMING") == null) return;
    const now = std.Io.Clock.awake.now(io);
    const ns: f64 = @floatFromInt(t0.durationTo(now).toNanoseconds());
    const at: f64 = @floatFromInt(@mod(now.toNanoseconds(), 1_000_000_000_000));
    std.debug.print("[timing] {s} {d:.2} ms at {d:.1}\n", .{ what, ns / 1e6, at / 1e6 });
}

test {
    _ = tree;
}

test "a drafter config with a zero divisor is refused, and any admitted rank's selector rows fit the staging area" {
    const gpa = std.testing.allocator;
    const target: engine.Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    const text =
        \\{"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,"selector_rank":256,
        \\ "selector_top_k":16,"target_layer_ids":[5,19,33,47,61]},"head_dim":128,"hidden_size":5120,"intermediate_size":17408,
        \\ "num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\ "rope_parameters":{"rope_theta":10000000},"sliding_window":2048,"vocab_size":248320}
    ;
    try std.testing.expectEqual(@as(usize, 256), (try Config.parse(gpa, text, target)).rank);
    for ([_][2][]const u8{ .{ "\"conv_group_size\":16", "\"conv_group_size\":0" }, .{ "\"num_key_value_heads\":8", "\"num_key_value_heads\":0" } }) |r| {
        const bad = try std.mem.replaceOwned(u8, gpa, text, r[0], r[1]);
        defer gpa.free(bad);
        try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, bad, target));
    }
    const wide = try std.mem.replaceOwned(u8, gpa, text, "\"selector_rank\":256", "\"selector_rank\":1024");
    defer gpa.free(wide);
    const c = try Config.parse(gpa, wide, target);
    try std.testing.expect(Draft.stagedLen(c.rank) >= Draft.hp_at + (c.block - 1) * c.rank * 2);
}
