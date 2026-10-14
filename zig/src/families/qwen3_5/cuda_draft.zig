//! The DFlash2 drafter on CUDA behind the lane core's Drafter contract (core/lanes/drafter.zig): it absorbs the
//! target's tapped rows into a context ring a stream, then drafts a block of 7 tokens after the pending one, the
//! target's own embedding and head at its ends and a codebook selector tracing one path through the top 16 a row.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const weights = @import("cuda_draft_weights.zig");

const E = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const dr = lanes.drafter;
const Stream = lanes.Stream;

/// Rows an absorb runs at once (the target's prompt chunks are at most this).
const max_rows = engine.max_rows;
/// The logprob row's words (sample.cu's LP_WORDS) and its best-token capacity.
const lp_words = 3 + 2 * 20;
const lp_top = 20;

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    window: usize,
    group: usize,
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [8]u32,
    n_taps: usize,

    fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// The ring's rows: the window and a block more.
    fn ringRows(c: Config) usize {
        return c.window + 16;
    }
    fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: engine.Geometry) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = (o.get("dflash_config") orelse return error.BadDraftConfig);
        if (d != .object) return error.BadDraftConfig;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(d.object, "block_size"),
            .mask = @intCast(try int(d.object, "mask_token_id")),
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = try int(d.object, "selector_rank"),
            .topk = try int(d.object, "selector_top_k"),
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num((o.get("rope_parameters") orelse return error.BadDraftConfig).object, "rope_theta")),
            .taps = undefined,
            .n_taps = 0,
        };
        const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > c.taps.len) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
            c.taps[i] = @intCast(t.integer);
        }
        c.n_taps = taps.array.items.len;
        // what the kernels fix: the head width, two-tap convolutions, a block of 8, the selector's top-k within a row
        if (c.head_dim != 128 or try int(d.object, "conv_kernel_size") != 2 or c.block != 8 or c.topk > lp_top or c.mask >= c.vocab) return error.UnsupportedDraftConfig;
        if (c.hidden != target.hidden or c.vocab != target.vocab or c.hidden % 256 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.layers > 8) return error.UnsupportedDraftConfig;
        if (c.window < c.block or c.window > engine.tap_rows - max_rows) return error.UnsupportedDraftConfig;
        return c;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

const Fns = struct { gather_taps: cuda.Function, dyn_conv: cuda.Function, norm_rope: cuda.Function, ring_put: cuda.Function, attn: cuda.Function };

/// A stream's context: its layers' key and value rings over positions [begin, end), and the drafts it holds.
const Ctx = struct {
    ring: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    held: [8]u32 = undefined,
    held_n: usize = 0,
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
    picks: u64, // u32 zeros: the logprob rows' pick (unused)
    hp: u64,
    ids: u64, // u32 the block's ids
    staged: cuda.HostBuffer, // pinned: the ids up, the words and projected rows down
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    /// Contexts between streams: allocated and run through once (`reserve`), since the driver stalls a buffer's
    /// first kernel for seconds.
    pool: std.ArrayList(*Ctx) = .empty,
    step_ms: f64 = 4.0,

    const words_at = 64;
    const hp_at = words_at + 16 * lp_words * 4;
    const staged_len = hp_at + 16 * 256 * 2;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8) !*Draft {
        const c = try Config.read(gpa, io, dir, e.g);
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .c = c, .w = undefined, .mod = undefined, .fns = undefined, .scratch = undefined, .taps_in = 0, .th = 0, .thn = 0, .kabs = 0, .kn = 0, .vabs = 0, .xs = 0, .h = 0, .h2 = 0, .x = 0, .xin = 0, .dyn = 0, .qraw = 0, .q = 0, .kraw = 0, .kb = 0, .vb = 0, .att = 0, .o = 0, .gate = 0, .up = 0, .act = 0, .logits = 0, .words = 0, .picks = 0, .hp = 0, .ids = 0, .staged = undefined };
        d.mod = try cuda.Module.load(e.d, cuda.kernels.qwen35_draft);
        errdefer d.mod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(d.fns, name) = try d.mod.function("tf_q35d_" ++ name);
        d.w = try weights.load(gpa, io, e, dir, .{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .vocab = c.vocab, .q_width = c.qWidth(), .kv_width = c.kvWidth(), .head_dim = c.head_dim, .dynamic = c.dynamic(), .rank = c.rank, .taps = c.n_taps });
        errdefer d.w.deinit(gpa);
        d.scratch = try DeviceBuffer.alloc(e.d, d.scratchBytes());
        errdefer d.scratch.free();
        try d.scratch.fill8(0, null);
        d.at = 0;
        _ = try d.layout(true);
        d.staged = try cuda.HostBuffer.alloc(e.d, staged_len);
        errdefer d.staged.free();
        try d.picksZero();
        return d;
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

    fn scratchBytes(d: *Draft) usize {
        d.at = 0;
        return d.layout(false) catch unreachable;
    }

    /// The scratch rows, measured (nothing assigned) or assigned from the scratch buffer.
    fn layout(d: *Draft, assign: bool) !usize {
        const c = d.c;
        const B = c.block;
        const R = max_rows;
        const sizes = .{
            .{ "taps_in", R * c.n_taps * c.hidden * 2 }, .{ "th", R * c.hidden * 2 },        .{ "thn", R * c.hidden * 2 },                  .{ "kabs", R * c.kvWidth() * 2 },
            .{ "kn", R * c.kvWidth() * 2 },              .{ "vabs", R * c.kvWidth() * 2 },   .{ "xs", R * (c.n_taps * c.hidden / 64) * 4 }, .{ "h", B * c.hidden * 2 },
            .{ "h2", B * c.hidden * 2 },                 .{ "x", B * c.hidden * 2 },         .{ "xin", B * c.hidden * 2 },                  .{ "dyn", B * c.dynamic() * 2 },
            .{ "qraw", B * c.qWidth() * 2 },             .{ "q", B * c.qWidth() * 2 },       .{ "kraw", B * c.kvWidth() * 2 },              .{ "kb", B * c.kvWidth() * 2 },
            .{ "vb", B * c.kvWidth() * 2 },              .{ "att", B * c.qWidth() * 2 },     .{ "o", B * c.hidden * 2 },                    .{ "gate", B * c.intermediate * 2 },
            .{ "up", B * c.intermediate * 2 },           .{ "act", B * c.intermediate * 2 }, .{ "logits", B * c.vocab * 2 },                .{ "words", B * lp_words * 4 },
            .{ "picks", B * 4 },                         .{ "hp", B * c.rank * 2 },          .{ "ids", B * 4 },
        };
        var total: usize = 0;
        inline for (sizes) |s| {
            const at = engine.aligned(total);
            total = at + s[1];
            if (assign) @field(d, s[0]) = d.scratch.ptr + at;
        }
        return total;
    }

    fn picksZero(d: *Draft) !void {
        const zero: [64]u8 = @splat(0);
        try d.e.d.check(d.e.d.api.cuMemcpyHtoD_v2(d.picks, &zero, 64), "cuMemcpyHtoD");
    }

    fn drop(d: *Draft, ctx: *Ctx) void {
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    /// A new context, its ring touched by the kernels once: a row absorbed from the first tap ring at hand, then
    /// a block drafted over it, so no request pays the driver's stall.
    fn create(d: *Draft) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try DeviceBuffer.alloc(d.e.d, d.c.ringBytes()) };
        errdefer ctx.ring.free();
        try ctx.ring.fill8(0, null);
        try d.absorb(ctx, d.taps_in, 0, 1); // any tap rows do: the drafter's own zeroed staging
        try d.hold(ctx, 0, 1, d.c.block - 1);
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
    fn proj(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
        const e = d.e;
        if (rows > 16) return e.affine.prompt(e.stream, x, l, out, rows);
        try e.run(e.fns.sums, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
        try e.affine.decode(e.stream, x, d.xs, l, out, rows);
    }

    fn norm(d: *Draft, h: u64, w: u64, out: u64, rows: usize) !void {
        const e = d.e;
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, @as(u64, 0), w, out, d.c.eps, @as(c_int, @intCast(d.c.hidden)), @as(u64, 0) });
    }

    fn dynConv(d: *Draft, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, branch: usize) !void {
        try d.e.run(d.fns.dyn_conv, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ x, dyn, base, res, out, @as(c_int, @intCast(d.c.hidden)), @as(c_int, @intCast(d.c.group)), @as(c_int, @intCast(branch)) });
    }

    fn normRope(d: *Draft, x: u64, w: u64, out: u64, rows: usize, heads: usize, pos0: u64) !void {
        try d.e.run(d.fns.norm_rope, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * d.c.head_dim)), @as(c_longlong, @intCast(pos0)), d.c.eps, d.c.theta });
    }

    /// Target rows at positions [start, start + n) from the stream's tap ring into its context (replacing what was
    /// held at or past `start`); rows older than the window are skipped.
    fn absorb(d: *Draft, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize) !void {
        const c = d.c;
        var start = start_;
        var n = n_;
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
            try e.run(d.fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.n_taps * c.hidden)), d.taps_in });
            try d.proj(d.w.fc, d.taps_in, d.th, rows);
            try d.norm(d.th, d.w.hidden_norm, d.thn, rows);
            for (d.w.layers, 0..) |l, i| {
                try d.proj(l.k, d.thn, d.kabs, rows);
                try d.normRope(d.kabs, l.k_norm, d.kn, rows, c.kv_heads, start);
                try d.proj(l.v, d.thn, d.vabs, rows);
                try e.run(d.fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn, d.vabs, d.ringK(ctx, i), d.ringV(ctx, i), @as(c_int, @intCast(c.kvWidth())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.ringRows())) });
            }
            start += rows;
            n -= rows;
            ctx.end = start;
        }
        if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
    }

    /// The block [pending, mask x 7] at positions pos0.. over the stream's context: `depth` drafts held.
    fn hold(d: *Draft, ctx: *Ctx, pending: u32, pos0: u64, depth: usize) !void {
        const c = d.c;
        const e = d.e;
        const B = c.block;
        if (ctx.end != pos0) return error.DraftContextBehind;
        const ids = std.mem.bytesAsSlice(u32, d.staged.bytes[0 .. B * 4]);
        ids[0] = pending;
        @memset(ids[1..B], c.mask);
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, ids.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
        try e.run(e.fns.embed, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.ids, e.embed_w, e.embed_s, e.embed_b, d.h, @as(c_int, @intCast(c.hidden)), @as(u64, 0) });
        for (d.w.layers, 0..) |l, i| {
            try d.norm(d.h, l.input_norm, d.x, B);
            try d.proj(l.attn_conv, d.x, d.dyn, B);
            try d.dynConv(d.x, d.dyn, l.attn_base, 0, d.xin, B, 0);
            try d.proj(l.q, d.xin, d.qraw, B);
            try d.proj(l.k, d.xin, d.kraw, B);
            try d.proj(l.v, d.xin, d.vb, B);
            try d.normRope(d.qraw, l.q_norm, d.q, B, c.heads, pos0);
            try d.normRope(d.kraw, l.k_norm, d.kb, B, c.kv_heads, pos0);
            try e.run(d.fns.attn, .{ B, c.heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.kb, d.vb, d.ringK(ctx, i), d.ringV(ctx, i), @as(c_int, @intCast(B)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.ringRows())), @as(c_longlong, @intCast(ctx.begin)), @as(c_longlong, @intCast(pos0)), @as(c_int, @intCast(c.window)), d.att });
            try d.proj(l.o, d.att, d.o, B);
            try d.dynConv(d.o, d.dyn, l.attn_base, d.h, d.h2, B, 1);
            try d.norm(d.h2, l.post_norm, d.x, B);
            try d.proj(l.mlp_conv, d.x, d.dyn, B);
            try d.dynConv(d.x, d.dyn, l.mlp_base, 0, d.xin, B, 0);
            try d.proj(l.gate, d.xin, d.gate, B);
            try d.proj(l.up, d.xin, d.up, B);
            try e.run(e.fns.mlp_act, .{ (B * c.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)) });
            try d.proj(l.down, d.act, d.o, B);
            try d.dynConv(d.o, d.dyn, l.mlp_base, d.h2, d.h, B, 1);
        }
        // the mask rows' logits through the target's head, their best tokens and their selector projections
        const rows = B - 1;
        try d.norm(d.h + c.hidden * 2, d.w.final_norm, d.x, rows);
        try d.proj(e.head, d.x, d.logits, rows);
        try e.draws.logprobRows(e.stream, d.logits, c.vocab, d.picks, @intCast(c.topk), false, d.words, rows);
        try d.proj(d.w.selector, d.x, d.hp, rows);
        const st = d.staged.bytes;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + hp_at, d.hp, rows * c.rank * 2, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        d.select(ctx, pending, rows);
        ctx.held_n = @min(depth, rows);
    }

    /// The path through the rows' candidates: each row's best of unary logit + predecessor . projected . successor.
    fn select(d: *Draft, ctx: *Ctx, pending: u32, rows: usize) void {
        const c = d.c;
        const words = std.mem.bytesAsSlice(u32, d.staged.bytes[words_at..][0 .. rows * lp_words * 4]);
        const hp = std.mem.bytesAsSlice(u16, d.staged.bytes[hp_at..][0 .. rows * c.rank * 2]);
        var pred = pending;
        for (0..rows) |r| {
            const w = words[r * lp_words ..][0..lp_words];
            var best: f32 = -std.math.inf(f32);
            var pick: u32 = w[2];
            for (0..c.topk) |j| {
                const cand = w[2 + j];
                const unary: f32 = @bitCast(w[2 + lp_top + j]);
                var edge: f32 = 0;
                for (0..c.rank) |k| edge += bf16(d.w.predecessor[pred * c.rank + k]) * bf16(hp[r * c.rank + k]) * bf16(d.w.successor[cand * c.rank + k]);
                const score = unary + edge;
                if (score > best) {
                    best = score;
                    pick = cand;
                }
            }
            ctx.held[r] = pick;
            pred = pick;
        }
    }

    // the Drafter contract

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
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
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = false, .stream_bytes = d.c.ringBytes() };
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
            // the target's features: its ring, `offset` the first row's position (cuda_native.zig's featuresFn)
            try d.absorb(ctx, it.features.buffer, it.features.offset, it.features.rows);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("hold", d.io, t0);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            try d.hold(ctx, it.pending, it.position - 1, it.depth);
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(p);
        for (streams, out) |s, o| {
            const ctx = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > ctx.held_n) return error.TooFewDrafts;
            @memcpy(o, ctx.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }
};

/// Acceptance by depth until a stream has its own (the reference's chat text: measured, see the commit).
const prior = [_]f64{ 0.75, 0.65, 0.58, 0.52, 0.47, 0.43, 0.4 };

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// TF_TIMING=1: each call's wall time on stderr.
pub fn timing(what: []const u8, io: std.Io, t0: std.Io.Timestamp) void {
    if (std.c.getenv("TF_TIMING") == null) return;
    const now = std.Io.Clock.awake.now(io);
    const ns: f64 = @floatFromInt(t0.durationTo(now).toNanoseconds());
    const at: f64 = @floatFromInt(@mod(now.toNanoseconds(), 1_000_000_000_000));
    std.debug.print("[timing] {s} {d:.2} ms at {d:.1}\n", .{ what, ns / 1e6, at / 1e6 });
}
