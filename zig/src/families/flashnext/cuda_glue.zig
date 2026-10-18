//! Flash Next's own glue kernels (flashnext_hc.cu, flashnext_attn.cu) with the core's shared rows: the default path.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");

/// The attention tiles' head width and query heads a KV head (flashnext_attn.cu's HD and QR).
const tile_head_dim = 256;
const tile_heads = 16;
/// fn_attn_chunks' staged rows: the query heads, `buffers` 32-key tiles of keys and values (rows padded by 8), bf16 p.
fn chunkSmem(buffers: u32) u32 {
    return (tile_heads + buffers * 2 * 32) * (tile_head_dim + 8) * 2 + tile_heads * (32 + 8) * 2;
}
/// The one-buffer form: a tile's keys, then its values, in the same rows of shared memory.
fn oneSmem() u32 {
    return (tile_heads + 32) * (tile_head_dim + 8) * 2 + tile_heads * (32 + 8) * 2;
}
/// Rows up to which attention reads the next tile ahead (fewer blocks, so latency, not occupancy, sets the time).
const ahead_rows = 16;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The native kernels, by their extern "C" names.
const Fns = struct {
    wb0: cuda.Function,
    wb_moe: cuda.Function,
    wb_slices: cuda.Function,
    normalize: cuda.Function,
    dense_mix: cuda.Function,
    shared_swiglu: cuda.Function,
    shared_scatter: cuda.Function,
    down: cuda.Function,
    reduce_act: cuda.Function,
    upmix: cuda.Function,
    upmix_up: cuda.Function,
    ple_gate: cuda.Function,
    ple_conv: cuda.Function,
    add_streams: cuda.Function,
    prep: cuda.Function,
    prep_media: cuda.Function,
    pool_media: cuda.Function,
    gate: cuda.Function,
    pool: cuda.Function,
    scores: cuda.Function,
    chunks: cuda.Function,
    chunks_ahead: cuda.Function,
    chunks_one: cuda.Function,
    fused: cuda.Function,
    merge: cuda.Function,
};

/// The captured entry points our glue replaces, by Triton name (TF_FLASHNEXT_OWNED lists them).
pub const entries = [_][]const u8{ "_embed", "_rmsnorm", "_add_streams", "_attn_prep", "_attn_gate", "_chunks", "_merge", "_pool", "_scores", "_hc_writeback", "_qmm_hcdown", "_hc_reduce_act", "_qmm_upmix", "_ple_gate", "_ple_conv", "_router", "_topk_rows" };

/// The bit of entry `name` in Glue.mixed.
pub fn entryBit(comptime name: []const u8) u32 {
    inline for (entries, 0..) |e, i| if (comptime std.mem.eql(u8, e, name)) return 1 << i;
    @compileError("no entry point " ++ name);
}

/// With a captured set loaded, the entries TF_FLASHNEXT_OWNED names ("all", or Triton names) run ours: an A/B switch.
fn mixedFromEnv() u32 {
    const v = std.mem.span(std.c.getenv("TF_FLASHNEXT_OWNED") orelse return 0);
    if (std.mem.eql(u8, v, "all")) return (1 << entries.len) - 1;
    var bits: u32 = 0;
    var it = std.mem.tokenizeAny(u8, v, ", ");
    while (it.next()) |name| for (entries, 0..) |e, i| if (std.mem.eql(u8, e, name)) {
        bits |= @as(u32, 1) << @intCast(i);
    };
    return bits;
}

pub const Glue = struct {
    /// Entries ours run even with a captured set loaded (TF_FLASHNEXT_OWNED, read once at load).
    mixed: u32 = 0,
    hc: cuda.Module,
    attn: cuda.Module,
    emb: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    route: cuda.route_rows.RouteRows,
    f: Fns,

    pub fn load(d: *const cuda.Driver) !Glue {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var g: Glue = undefined;
        g.mixed = mixedFromEnv();
        g.hc = try cuda.Module.load(d, cuda.kernels.flashnext_hc);
        errdefer g.hc.unload();
        g.attn = try cuda.Module.load(d, cuda.kernels.flashnext_attn);
        errdefer g.attn.unload();
        g.emb = try cuda.embed_rows.EmbedRows.load(d);
        errdefer g.emb.unload();
        g.norms = try cuda.norms.Norms.load(d);
        errdefer g.norms.unload();
        g.route = try cuda.route_rows.RouteRows.load(d);
        errdefer g.route.unload();
        g.f = .{
            .wb0 = try g.hc.function("fn_hc_writeback0"),
            .wb_moe = try g.hc.function("fn_hc_writeback_moe"),
            .wb_slices = try g.hc.function("fn_hc_writeback_slices"),
            .normalize = try g.hc.function("fn_hc_normalize"),
            .dense_mix = try g.hc.function("fn_hc_dense_mix"),
            .shared_swiglu = try g.hc.function("fn_shared_swiglu"),
            .shared_scatter = try g.hc.function("fn_shared_scatter"),
            .down = try g.hc.function("fn_hc_down"),
            .reduce_act = try g.hc.function("fn_hc_reduce_act"),
            .upmix = try g.hc.function("fn_hc_upmix"),
            .upmix_up = try g.hc.function("fn_hc_upmix_up"),
            .ple_gate = try g.hc.function("fn_ple_gate"),
            .ple_conv = try g.hc.function("fn_ple_conv"),
            .add_streams = try g.hc.function("fn_add_streams"),
            .prep = try g.attn.function("fn_attn_prep"),
            .prep_media = try g.attn.function("fn_attn_prep_media"),
            .pool_media = try g.attn.function("fn_qsa_pool_media"),
            .gate = try g.attn.function("fn_attn_gate"),
            .pool = try g.attn.function("fn_qsa_pool"),
            .scores = try g.attn.function("fn_qsa_scores"),
            .chunks = try g.attn.function("fn_attn_chunks"),
            .chunks_ahead = try g.attn.function("fn_attn_chunks_ahead"),
            .chunks_one = try g.attn.function("fn_attn_chunks_one"),
            .fused = try g.attn.function("fn_attn_fused"),
            .merge = try g.attn.function("fn_attn_merge"),
        };
        try g.f.chunks.allowDynamicShared(chunkSmem(1));
        try g.f.chunks_ahead.allowDynamicShared(chunkSmem(2));
        try g.f.chunks_one.allowDynamicShared(oneSmem());
        try g.f.fused.allowDynamicShared(oneSmem());
        return g;
    }

    pub fn deinit(g: *Glue) void {
        g.route.unload();
        g.norms.unload();
        g.emb.unload();
        g.attn.unload();
        g.hc.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared };
        try cuda.launch.launch(f, cfg, s, a);
    }

    fn ptrs(a: *cuda.Args, xs: []const u64) void {
        for (xs) |x| a.add(x);
    }

    fn ints(a: *cuda.Args, xs: []const usize) void {
        for (xs) |x| a.add(int(x));
    }

    /// Token rows of the MLX 4-bit group-32 table into `copies` streams each (the core's embed rows).
    pub fn embed(g: *const Glue, s: cuda.Stream, ids: u64, w: u64, sc: u64, b: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        try g.emb.affine4Groups(s, ids, w, sc, b, out, rows, dims, 32, copies);
    }

    pub fn embedDense(g: *const Glue, s: cuda.Stream, ids: u64, w: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        try g.emb.bf16Copies(s, ids, w, out, rows, dims, copies);
    }

    /// One RMS norm over each row's `dims` features (rows `x_stride` apart), bf16 out and 32-group sums (core norms).
    pub fn rmsnorm(g: *const Glue, s: cuda.Stream, x: u64, w: u64, out: u64, xs: u64, eps: f32, x_stride: usize, rows: usize, dims: usize, group: usize) !void {
        if (group != dims) return error.RowShape; // the head norms each row whole
        try g.norms.rmsSums(s, x, x_stride, w, out, xs, rows, dims, 32, eps);
    }

    pub fn addStreams(g: *const Glue, s: cuda.Stream, e: u64, hs: u64, out: u64, rows: usize, dims: usize, streams: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ e, hs, out });
        ints(&a, &.{ dims, streams });
        try go(g.f.add_streams, s, .{ rows, cdiv(dims, 256), 1 }, 256, 0, &a);
    }

    pub fn attnGate(g: *const Glue, s: cuda.Stream, o: u64, proj: u64, out: u64, xs: u64, rows: usize, pair_width: usize, q_heads: usize, head_dim: usize) !void {
        if (head_dim != tile_head_dim) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ o, proj, out, xs });
        ints(&a, &.{ pair_width, q_heads });
        try go(g.f.gate, s, .{ rows, q_heads, 1 }, tile_head_dim, 0, &a);
    }

    /// Text positions, a bf16 cache: the model's heads (cuda_prompt_wt.zig) normed, roped and stored.
    pub fn attnPrep(g: *const Glue, s: cuda.Stream, proj: u64, pos: u64, q_scale: u64, k_scale: u64, i_scale: u64, inv: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, eps: f32, rows: usize) !void {
        if (wt.head_dim != tile_head_dim) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ proj, pos, q_scale, k_scale, i_scale, inv, q, kc, vc, iq, ikc });
        a.add(eps);
        ints(&a, &.{ wt.proj_n, wt.q_heads, wt.kv_heads, wt.index_heads, wt.index_dim, wt.half });
        try go(g.f.prep, s, .{ rows, wt.q_heads + wt.kv_heads + wt.index_heads + 1, 1 }, tile_head_dim, 0, &a);
    }

    /// Attention over 512-key chunks; `qsa` the sparse form (NCH 5, rows' key lists), else dense (NCH 3).
    pub fn attnChunks(g: *const Glue, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos: u64, po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, rows: usize, kv: usize, chunks: usize, qsa: bool) !void {
        const group = wt.q_heads / kv;
        if (group > tile_heads) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ q, kc, vc, pos, po, pm, pl, ids, nk, sparse });
        ints(&a, &.{ wt.q_heads, kv, group, 512, if (qsa) wt.nch else wt.dense_nch });
        a.add(@as(f32, 0.0625));
        ints(&a, &.{ wt.index_idw, @intFromBool(qsa) });
        const ahead = rows <= ahead_rows;
        try go(if (ahead) g.f.chunks_ahead else g.f.chunks_one, s, .{ rows, kv, chunks }, 128, if (ahead) chunkSmem(2) else oneSmem(), &a);
    }

    /// Prompt rows' attention with every chunk merged in the program (fn_attn_fused): out bf16, attnChunks then
    pub fn attnFused(g: *const Glue, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos: u64, out: u64, ids: u64, nk: u64, sparse: u64, rows: usize, kv: usize) !void {
        const group = wt.q_heads / kv;
        if (group > tile_heads) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ q, kc, vc, pos, out, ids, nk, sparse });
        ints(&a, &.{ wt.q_heads, kv, group, 512 });
        a.add(@as(f32, 0.0625));
        ints(&a, &.{ wt.index_idw, 1 });
        try go(g.f.fused, s, .{ rows, kv, 1 }, 128, oneSmem(), &a);
    }

    pub fn attnMerge(g: *const Glue, s: cuda.Stream, po: u64, pm: u64, pl: u64, pos: u64, out: u64, nk: u64, sparse: u64, rows: usize, kv: usize, qsa: bool) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ po, pm, pl, pos, out, nk, sparse });
        ints(&a, &.{ wt.q_heads, kv, wt.q_heads / kv, 512, if (qsa) wt.nch else wt.dense_nch, @intFromBool(qsa) });
        try go(g.f.merge, s, .{ rows, wt.q_heads, 1 }, tile_head_dim, 0, &a);
    }

    pub fn qsaPool(g: *const Glue, s: cuda.Stream, ikc: u64, pooled: u64, pos: u64, k_scale: u64, inv: u64, eps: f32, rows: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ ikc, pooled, pos, k_scale, inv });
        a.add(eps);
        ints(&a, &.{ rows, wt.index_dim, wt.half, wt.index_ratio });
        try go(g.f.pool, s, .{ rows / wt.index_ratio + 2, 1, 1 }, wt.index_dim, 0, &a);
    }

    pub fn qsaScores(g: *const Glue, s: cuda.Stream, iq: u64, pooled: u64, pos: u64, sc: u64, nb: usize, rows: usize, blocks: usize) !void {
        if (wt.index_dim != 128 or wt.index_heads > 8) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ iq, pooled, pos, sc });
        ints(&a, &.{ nb, wt.index_heads, wt.index_ratio, wt.index_top });
        try go(g.f.scores, s, .{ rows, cdiv(blocks, 64), 1 }, 256, 0, &a);
    }

    pub fn hcNormalize(g: *const Glue, s: cuda.Stream, h: u64, pss: u64, scale: u64, out: u64, eps: f32, rows: usize, dims: usize, streams: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ h, pss, scale, out }) |v| args.add(v);
        args.add(eps);
        args.add(int(dims));
        args.add(int(streams));
        try cuda.launch.launch(g.f.normalize, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(streams) }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn hcDenseMix(g: *const Glue, s: cuda.Stream, up: u64, norm: u64, mixed: u64, sums: u64, rows: usize, dims: usize, streams: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ up, norm, mixed, sums }) |v| args.add(v);
        args.add(int(dims));
        args.add(int(streams));
        try cuda.launch.launch(g.f.dense_mix, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(dims / 32) }, .block = .{ .x = 32 } }, s, &args);
    }

    pub fn sharedSwiGLU(g: *const Glue, s: cuda.Stream, gate: u64, up: u64, act: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ gate, up, act });
        ints(&a, &.{width});
        try go(g.f.shared_swiglu, s, .{ rows, cdiv(width, 256), 1 }, 256, 0, &a);
    }

    pub fn sharedScatter(g: *const Glue, s: cuda.Stream, shared: u64, out: u64, rows: usize, dims: usize, slots: usize) !void {
        if (slots == 0) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ shared, out });
        ints(&a, &.{ dims, slots });
        try go(g.f.shared_scatter, s, .{ rows, cdiv(dims, 256), 1 }, 256, 0, &a);
    }

    pub fn hcReduceAct(g: *const Glue, s: cuda.Stream, part: u64, act: u64, xs: u64, inj: u64, sk: usize, rows: usize, streams: usize, low: usize, ndn: usize, has_inj: usize) !void {
        const cols = low + has_inj * streams;
        if (cols > 1024 or sk > 32) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ part, act, xs, inj });
        ints(&a, &.{ sk, rows, streams, low, has_inj, ndn });
        try go(g.f.reduce_act, s, .{ rows, 1, 1 }, @intCast(cdiv(@max(cols, low / 32), 32) * 32), 0, &a);
    }

    pub fn hcUpmix(g: *const Glue, s: cuda.Stream, act: u64, xs: u64, w: u64, sc: u64, b: u64, normed: u64, mixed: u64, xs_mixed: u64, rows: usize, n: usize, k: usize, dims: usize, streams: usize) !void {
        if (streams != 4 or dims % 32 != 0 or k % 32 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ act, xs, w, sc, b, normed, mixed, xs_mixed });
        ints(&a, &.{ rows, n, k, dims, streams });
        const shared: u32 = @intCast(16 * (k + 8) * 2 + streams * 16 * 32 * 4);
        try go(g.f.upmix, s, .{ cdiv(rows, 16), dims / 32, 1 }, 256, shared, &a);
    }

    /// hcUpmix that also keeps the up rows `up` [rows, n] bf16 the mix read, the same mixed bits.
    pub fn hcUpmixUp(g: *const Glue, s: cuda.Stream, act: u64, xs: u64, w: u64, sc: u64, b: u64, normed: u64, mixed: u64, xs_mixed: u64, up: u64, rows: usize, n: usize, k: usize, dims: usize, streams: usize) !void {
        if (streams != 4 or dims % 32 != 0 or k % 32 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ act, xs, w, sc, b, normed, mixed, xs_mixed, up });
        ints(&a, &.{ rows, n, k, dims, streams });
        const shared: u32 = @intCast(16 * (k + 8) * 2 + streams * 16 * 32 * 4);
        try go(g.f.upmix_up, s, .{ cdiv(rows, 16), dims / 32, 1 }, 256, shared, &a);
    }

    /// The normed streams times the down rows, each K slice's fp32 sums to `part` [sk, rows, n] (sk > 1).
    pub fn hcDown(g: *const Glue, s: cuda.Stream, h: u64, pss: u64, scale: u64, normed: u64, w: u64, sc: u64, b: u64, out: u64, part: u64, eps: f32, rows: usize, n: usize, k: usize, dims: usize, nc: usize, streams: usize, sk: usize) !void {
        _ = out;
        const per = k / 32 / sk;
        if (sk < 2 or k % (32 * sk) != 0 or dims % (per * 32) != 0) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ h, pss, scale, normed, w, sc, b, part });
        a.add(int(rows));
        a.add(eps);
        ints(&a, &.{ n, k, dims, nc, streams, sk });
        const shared: u32 = @intCast(16 * (per * 32 + 8) * 2 + 16 * per * 4);
        try go(g.f.down, s, .{ cdiv(rows, 16), cdiv(n, 64), sk }, 128, shared, &a);
    }

    /// Mode 0: the streams' sums; 2: the MoE slots (fp32 y and weights) written back; 4: `world` fp32 K slices.
    pub fn hcWriteback(g: *const Glue, s: cuda.Stream, h: u64, hout: u64, pss: u64, branch: u64, inj: u64, y: u64, wts: u64, rs: usize, rows: usize, dims: usize, streams: usize, mode: usize, topk: usize, slots: usize, world: usize) !void {
        if (dims % 256 != 0 or streams > 8) return error.RowShape;
        var a: cuda.Args = .{};
        const grid: [3]usize = .{ rows, dims / 256, 1 };
        switch (mode) {
            0 => {
                ptrs(&a, &.{ h, pss });
                ints(&a, &.{ dims, streams });
                try go(g.f.wb0, s, grid, 256, 0, &a);
            },
            2 => {
                ptrs(&a, &.{ h, hout, pss, inj, y, wts });
                ints(&a, &.{ dims, streams, topk, slots });
                try go(g.f.wb_moe, s, grid, 256, 0, &a);
            },
            4 => {
                ptrs(&a, &.{ h, hout, pss, branch, inj });
                a.add(@as(i64, @intCast(rs)));
                ints(&a, &.{ dims, streams, world });
                try go(g.f.wb_slices, s, grid, 256, 0, &a);
            },
            else => return error.RowShape,
        }
    }

    pub fn pleGate(g: *const Glue, s: cuda.Stream, keys: u64, vals: u64, h: u64, norm_key: u64, norm_query: u64, gated: u64, pss: u64, eps: f32, rows: usize, dims: usize, streams: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ keys, vals, h, norm_key, norm_query, gated, pss });
        a.add(eps);
        ints(&a, &.{ dims, streams });
        try go(g.f.ple_gate, s, .{ rows, 1, 1 }, 512, 0, &a);
    }

    pub fn pleConv(g: *const Glue, s: cuda.Stream, gated: u64, pss: u64, norm_conv: u64, tail: u64, conv_w: u64, h: u64, hout: u64, nrow: u64, eps: f32, rows: usize, dims: usize, streams: usize, taps: usize, dilation: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ gated, pss, norm_conv, tail, conv_w, h, hout, nrow });
        a.add(eps);
        ints(&a, &.{ dims, streams, taps, dilation });
        try go(g.f.ple_conv, s, .{ rows, cdiv(streams * dims, 512), 1 }, 512, 0, &a);
    }

    /// Router logits [rows, experts + 1] fp32: the experts' rows, then the shared expert's gate row (core route rows).
    pub fn routerMany(g: *const Glue, s: cuda.Stream, x: u64, w: u64, out: u64, x_stride: usize, rows: usize) !void {
        try g.route.logits(s, x, x_stride, w, out, rows, wt.dims, wt.experts_n + 1);
    }

    /// Each row's ten experts and the shared one: picks and weights [rows, 11] (core route rows).
    pub fn topkMany(g: *const Glue, s: cuda.Stream, logits: u64, pick: u64, wts: u64, rows: usize) !void {
        try g.route.topk(s, logits, wt.experts_n + 1, wt.experts_n, wt.slots - 1, pick, wts, wt.slots, rows, true);
    }
};

test "every entry point has its own bit" {
    try std.testing.expectEqual(@as(u32, 1), entryBit("_embed"));
    try std.testing.expectEqual(@as(u32, 1) << 16, entryBit("_topk_rows"));
}

test "the attention tiles hold the model's heads" {
    try std.testing.expect(wt.head_dim == tile_head_dim and wt.q_heads / wt.kv_heads <= tile_heads);
}
