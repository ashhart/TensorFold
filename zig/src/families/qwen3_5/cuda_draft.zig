//! The DFlash2 drafter on CUDA behind core/lanes/drafter.zig: a tap ring a stream, 7 drafts as a chain or a tree.
const std = @import("std");
const platform = @import("sys");
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
        return parse(gpa, bytes, target);
    }

    fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: engine.Geometry) !Config {
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
            .mask = @intCast(try whole(d.object, "mask_token_id")),
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

/// A size, count or divisor: positive, so a zero can't reach a remainder or an empty allocation.
fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

/// A token id: any whole number.
fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
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

const Fns = struct { gather_taps: cuda.Function, gather_rows: cuda.Function, dyn_conv: cuda.Function, norm_rope: cuda.Function, ring_put: cuda.Function, attn: cuda.Function };

/// qwen35_draft.cu's TapRows.
const TapRows = extern struct { n: c_int, rows: [engine.most_window]c_int };

/// A stream's context: its layers' key and value rings over [begin, end), and the drafts it holds (a chain or tree).
const Ctx = struct {
    ring: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    held: [engine.most_window]u32 = undefined,
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
    /// Contexts between streams, allocated and run once (`reserve`), as the driver stalls a buffer's first kernel.
    pool: std.ArrayList(*Ctx) = .empty,
    /// One level of a hold (a hold drafts every level at once; timed at load, over the block's levels).
    step_ms: f64 = 1.0,
    /// The target's window ms by rows (cuda_native.zig times them), for the round loop's prices.
    costs: [engine.most_window]lanes.config.Cost = undefined,
    n_costs: usize = 0,

    const words_at = 64;
    const hp_at = words_at + 16 * lp_words * 4;
    /// The host staging area: the words then 16 rows of `rank` selector projections (the block's mask rows fit).
    fn stagedLen(rank: usize) usize {
        return hp_at + 16 * rank * 2;
    }

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
        d.staged = try cuda.HostBuffer.alloc(e.d, stagedLen(c.rank));
        errdefer d.staged.free();
        try d.picksZero();
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

    /// The target's window of `rows` rows takes `ms` (told by cuda_native.zig before the facts are read).
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

    /// The block [pending, mask x 7] at pos0.. over the context: `depth` drafts as a chain, or the `lanes` tree.
    fn hold(d: *Draft, ctx: *Ctx, pending: u32, pos0: u64, depth: usize, lanes_: ?*const lanes.shape.Shape) !void {
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
        if (lanes_) |l| {
            try d.selectLanes(ctx, pending, rows, l);
            if (platform.c.getenv("TF_TIMING") != null) std.debug.print("[timing] lanes {d} levels {d}\n", .{ l.parents.len, l.levels() });
        } else {
            d.select(ctx, pending, rows);
            ctx.held_n = @min(depth, rows);
            if (platform.c.getenv("TF_TIMING") != null) std.debug.print("[timing] chain {d}\n", .{ctx.held_n});
        }
    }

    /// Row `r`'s `rank`th candidate after `pred`: unary logit + pred . projected . successor, ties to the earlier one.
    fn choice(d: *const Draft, r: usize, pred: u32, rank: usize) u32 {
        const c = d.c;
        const w = std.mem.bytesAsSlice(u32, d.staged.bytes[words_at..][r * lp_words * 4 ..][0 .. lp_words * 4]);
        const hp = std.mem.bytesAsSlice(u16, d.staged.bytes[hp_at..][r * c.rank * 2 ..][0 .. c.rank * 2]);
        var scores: [lp_top]f32 = undefined;
        for (0..c.topk) |j| {
            const cand = w[2 + j];
            const unary: f32 = @bitCast(w[2 + lp_top + j]);
            var edge: f32 = 0;
            for (0..c.rank) |k| edge += bf16(d.w.predecessor[pred * c.rank + k]) * bf16(hp[k]) * bf16(d.w.successor[cand * c.rank + k]);
            scores[j] = unary + edge;
        }
        var taken: [lp_top]bool = @splat(false);
        var pick: usize = 0;
        for (0..rank + 1) |_| {
            var best: f32 = -std.math.inf(f32);
            pick = 0;
            for (0..c.topk) |j| if (!taken[j] and scores[j] > best) {
                best = scores[j];
                pick = j;
            };
            taken[pick] = true;
        }
        return w[2 + pick];
    }

    /// The path through the rows' candidates: each row's best after the one before.
    fn select(d: *Draft, ctx: *Ctx, pending: u32, rows: usize) void {
        var pred = pending;
        for (0..rows) |r| {
            ctx.held[r] = d.choice(r, pred, 0);
            pred = ctx.held[r];
        }
    }

    /// The tree's lanes in order: lane i the rank-r candidate at its depth's row after its parent lane's token.
    fn selectLanes(d: *Draft, ctx: *Ctx, pending: u32, rows: usize, l: *const lanes.shape.Shape) !void {
        if (l.parents.len > ctx.held.len) return error.TooManyLanes;
        for (l.parents, l.depths, l.ranks, 0..) |par, depth, rank, i| {
            if (depth >= rows or rank >= d.c.topk or par >= @as(i32, @intCast(i))) return error.BadLanes;
            const pred = if (par < 0) pending else ctx.held[@intCast(par)];
            ctx.held[i] = d.choice(depth, pred, rank);
        }
        ctx.held_n = l.parents.len;
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
        // TF_NO_TREES=1: chains only (for comparing)
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = false, .stream_bytes = d.c.ringBytes(), .trees = platform.c.getenv("TF_NO_TREES") == null, .window_costs = d.costs[0..d.n_costs], .lane_costs = d.costs[0..d.n_costs] };
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
            if (it.rows) |rs| {
                try d.absorb(ctx, it.features.buffer, it.features.offset, rs.len, rs);
            } else try d.absorb(ctx, it.features.buffer, it.features.offset, it.features.rows, null);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("hold", d.io, t0);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            try d.hold(ctx, it.pending, it.position - 1, it.depth, it.lanes);
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
    if (platform.c.getenv("TF_TIMING") == null) return;
    const now = std.Io.Clock.awake.now(io);
    const ns: f64 = @floatFromInt(t0.durationTo(now).toNanoseconds());
    const at: f64 = @floatFromInt(@mod(now.toNanoseconds(), 1_000_000_000_000));
    std.debug.print("[timing] {s} {d:.2} ms at {d:.1}\n", .{ what, ns / 1e6, at / 1e6 });
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
