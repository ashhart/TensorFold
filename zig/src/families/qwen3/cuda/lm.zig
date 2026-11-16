//! The Qwen3 dense prompt pass on CUDA's shared rows: norms, cuBLASLt projections, QK norm with RoPE, GQA chunks.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("../config.zig");
const gemm = @import("gemm.zig");
const store = @import("store.zig");
const weights = @import("weights.zig");

const Driver = cuda.Driver;

/// nemotron_attention.cu's chunk: keys a partial covers, and the most chunk blocks a row launches.
pub const chunk_keys = 512;
pub const chunk_blocks = 64;

/// Rows one attention launch takes: its partials grow with rows x chunks, so a long input runs in row blocks.
pub const attn_rows = 512;

const Attn = struct {
    mod: cuda.Module,
    chunk: cuda.Function,
    chunk_pf: cuda.Function,
    merge: cuda.Function,

    fn load(d: *const Driver) !Attn {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.nemotron_attention);
        errdefer mod.unload();
        return .{ .mod = mod, .chunk = try mod.function("tf_nemo_attn_chunk"), .chunk_pf = try mod.function("tf_nemo_attn_chunk_pf"), .merge = try mod.function("tf_nemo_attn_merge") };
    }
};

/// The pass's scratch in one block: each section's offset, `total` bytes, and the addresses `at` makes of them.
pub const Scratch = struct {
    ids: usize = 0,
    pos: usize = 0,
    cos: usize = 0,
    sin: usize = 0,
    x: usize = 0,
    normed: usize = 0,
    attn: usize = 0,
    q: usize = 0,
    k: usize = 0,
    v: usize = 0,
    kb: usize = 0,
    gate: usize = 0,
    up: usize = 0,
    z: usize = 0,
    kc: usize = 0,
    vc: usize = 0,
    po: usize = 0,
    pm: usize = 0,
    pl: usize = 0,
    xs: usize = 0,
    rows: usize = 0,
    total: usize = 0,

    /// The sections one pass over `window` tokens holds, pooling `inputs` rows at most.
    pub fn of(s: config.Spec, window: usize, inputs: usize) Scratch {
        const cap = std.mem.alignForward(usize, window, chunk_keys);
        const hd = s.head_dim;
        const qw = s.heads * hd;
        const kvw = s.kv_heads * hd;
        const chunks = cap / chunk_keys;
        var cursor: usize = 0;
        return .{
            .ids = cut(&cursor, 4 * window),
            .pos = cut(&cursor, 4 * cap),
            .cos = cut(&cursor, 4 * cap * hd),
            .sin = cut(&cursor, 4 * cap * hd),
            .x = cut(&cursor, 2 * window * s.hidden),
            .normed = cut(&cursor, 2 * window * s.hidden),
            .attn = cut(&cursor, 2 * window * qw),
            .q = cut(&cursor, 2 * window * qw),
            .k = cut(&cursor, 2 * window * kvw),
            .v = cut(&cursor, 2 * window * kvw),
            // a row block's k rows, then its v rows, as the kernel writes them beside the cache
            .kb = cut(&cursor, 4 * attn_rows * kvw),
            .gate = cut(&cursor, 2 * window * s.inter),
            .up = cut(&cursor, 2 * window * s.inter),
            .z = cut(&cursor, 2 * window * s.hidden),
            .kc = cut(&cursor, 2 * cap * kvw),
            .vc = cut(&cursor, 2 * cap * kvw),
            .po = cut(&cursor, 4 * attn_rows * chunks * s.heads * hd),
            .pm = cut(&cursor, 4 * attn_rows * chunks * s.heads),
            .pl = cut(&cursor, 4 * attn_rows * chunks * s.heads),
            .xs = cut(&cursor, 4 * attn_rows * qw / 64),
            .rows = cut(&cursor, 2 * inputs * s.hidden),
            .total = store.up(cursor),
        };
    }

    /// The block's addresses.
    pub fn at(sc: Scratch, base: u64) Addrs {
        var a: Addrs = .{};
        inline for (@typeInfo(Addrs).@"struct".field_names) |name| @field(a, name) = base + @field(sc, name);
        return a;
    }
};

/// The scratch block's addresses, in the sections' order.
pub const Addrs = struct {
    ids: u64 = 0,
    pos: u64 = 0,
    cos: u64 = 0,
    sin: u64 = 0,
    x: u64 = 0,
    normed: u64 = 0,
    attn: u64 = 0,
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    kb: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    z: u64 = 0,
    kc: u64 = 0,
    vc: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    xs: u64 = 0,
    rows: u64 = 0,
};

/// `n` bytes at the block's cursor, a section unit in.
fn cut(cursor: *usize, n: usize) usize {
    cursor.* = store.up(cursor.*);
    defer cursor.* += n;
    return cursor.*;
}

/// Which of the CUDA kernels' geometry limits a spec breaks: the attention's heads, the row kernels' widths, or none.
pub const Geometry = enum { ok, heads, widths };

/// The attention takes head_dim 128 and whole groups of up to 16 q heads a kv head; the row kernels, widths of 64s.
pub fn geometry(spec: config.Spec) Geometry {
    const grp = if (spec.kv_heads == 0) 0 else spec.heads / spec.kv_heads;
    if (spec.head_dim != 128 or spec.kv_heads == 0 or spec.heads % spec.kv_heads != 0 or grp > 16) return .heads;
    if (spec.hidden % 64 != 0 or spec.inter % 64 != 0) return .widths;
    return .ok;
}

test "the geometry limits: the attention's head size and groups, and the row kernels' multiples of 64" {
    const ok: config.Spec = .{ .layers = 28, .hidden = 4096, .heads = 32, .kv_heads = 8, .head_dim = 128, .inter = 12288, .vocab = 151669, .eps = 1e-6, .theta = 1e6, .max_positions = 40960, .tied = false };
    try std.testing.expectEqual(Geometry.ok, geometry(ok));
    var s = ok;
    s.head_dim = 64;
    try std.testing.expectEqual(Geometry.heads, geometry(s));
    s = ok;
    s.kv_heads = 0;
    try std.testing.expectEqual(Geometry.heads, geometry(s));
    s = ok;
    s.heads = 30;
    try std.testing.expectEqual(Geometry.heads, geometry(s)); // not a whole number of groups
    s = ok;
    s.heads = 136;
    try std.testing.expectEqual(Geometry.heads, geometry(s)); // 17 q heads a kv head
    s.heads = 128;
    try std.testing.expectEqual(Geometry.ok, geometry(s)); // 16 is the most
    s = ok;
    s.hidden = 4000;
    try std.testing.expectEqual(Geometry.widths, geometry(s));
    s = ok;
    s.inter = 12000;
    try std.testing.expectEqual(Geometry.widths, geometry(s));
}

test "the published Qwen3 sizes, embedding and reranker alike, are within the geometry limits" {
    const sizes = [_][3]usize{ .{ 1024, 3072, 16 }, .{ 2560, 9728, 32 }, .{ 4096, 12288, 32 } }; // hidden, inter, q heads
    for (sizes) |z| {
        const spec: config.Spec = .{ .layers = 28, .hidden = z[0], .heads = z[2], .kv_heads = 8, .head_dim = 128, .inter = z[1], .vocab = 151669, .eps = 1e-6, .theta = 1e6, .max_positions = 40960, .tied = z[0] < 4096 };
        try std.testing.expectEqual(Geometry.ok, geometry(spec));
    }
}

/// The shared operators one pass runs, over one stream.
pub const Lm = struct {
    d: *const Driver,
    s: cuda.Stream,
    g: gemm.Gemms,
    nm: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    emb: cuda.embed_rows.EmbedRows,
    attn: Attn,

    /// The kernels for `s`'s device; the geometry is the attention kernels' (head dim 128, to 16 q heads a kv head).
    pub fn init(gpa: std.mem.Allocator, d: *const Driver, s: cuda.Stream, spec: config.Spec) !Lm {
        switch (geometry(spec)) {
            .ok => {},
            .heads => {
                std.log.err("qwen3: {d} heads on {d} kv heads at head_dim {d}: the CUDA attention takes head_dim 128 and to 16 q heads a kv head", .{ spec.heads, spec.kv_heads, spec.head_dim });
                return error.UnsupportedGeometry;
            },
            .widths => {
                std.log.err("qwen3: hidden {d} and intermediate {d}: the CUDA row kernels take multiples of 64", .{ spec.hidden, spec.inter });
                return error.UnsupportedGeometry;
            },
        }
        var lm: Lm = .{ .d = d, .s = s, .g = undefined, .nm = undefined, .qk = undefined, .act = undefined, .emb = undefined, .attn = undefined };
        lm.g = try gemm.Gemms.init(gpa, d);
        errdefer lm.g.deinit();
        lm.nm = try cuda.norms.Norms.load(d);
        errdefer lm.nm.unload();
        lm.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer lm.qk.unload();
        lm.act = try cuda.act_rows.ActRows.load(d);
        errdefer lm.act.unload();
        lm.emb = try cuda.embed_rows.EmbedRows.load(d);
        errdefer lm.emb.unload();
        lm.attn = try Attn.load(d);
        return lm;
    }

    /// The pass's (n, k) plans, built where `lm` stays: each plan keeps a pointer to its Gemms' library.
    pub fn prepare(lm: *Lm, spec: config.Spec) !void {
        const qw = spec.heads * spec.head_dim;
        const kvw = spec.kv_heads * spec.head_dim;
        // one algorithm per (n, k) at every row count: a batch's rows keep the bits they have alone
        try lm.g.prepare(&.{ .{ qw, spec.hidden }, .{ kvw, spec.hidden }, .{ spec.hidden, qw }, .{ spec.inter, spec.hidden }, .{ spec.hidden, spec.inter } });
    }

    pub fn deinit(lm: *Lm) void {
        lm.attn.mod.unload();
        lm.emb.unload();
        lm.act.unload();
        lm.qk.unload();
        lm.nm.unload();
        lm.g.deinit();
    }

    /// Every input's final-normed last row into `rows`: after a shared `trunk` of rows, inputs end at `ends`.
    pub fn forward(lm: *Lm, m: *const weights.Model, a: Addrs, cap: usize, trunk: usize, ends: []const u32) !void {
        const s = m.spec;
        try lm.emb.bf16(lm.s, a.ids, m.embed, a.x, ends[ends.len - 1], s.hidden);
        for (m.layers, 0..) |*l, i| try layerRows(lm, m, l, a, cap, trunk, ends, ends[ends.len - 1], i == 0);
        for (ends, 0..) |e, i| {
            // the last layer's residual add and the final norm, for the one row an input pools
            const row = (e - 1) * s.hidden * 2;
            try lm.nm.addRmsH(lm.s, a.z + row, a.x + row, m.final_norm, .f32, a.x + row, a.rows + i * s.hidden * 2, null, 1, s.hidden, @floatCast(s.eps));
        }
    }

    /// The chunk partials of a row block's `q`, over every chunk below its last key.
    fn chunks(lm: *Lm, s: config.Spec, q: u64, a: Addrs, pos: u64, rows: usize, nch: usize, scale: f32) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, a.kc, a.vc, pos, a.po, a.pm, a.pl }) |x| args.add(x);
        for ([_]usize{ s.heads * s.head_dim, s.heads, s.kv_heads, s.heads / s.kv_heads, chunk_keys, nch }) |x| args.add(@as(c_int, @intCast(x)));
        args.add(scale);
        args.add(@as(u64, 0));
        args.add(@as(i64, 0));
        const grid: cuda.Dim3 = .{ .x = @intCast(rows), .y = @intCast(s.kv_heads), .z = @intCast(@min(nch, chunk_blocks)) };
        try cuda.launch.launch(if (rows > 1) lm.attn.chunk_pf else lm.attn.chunk, .{ .grid = grid, .block = .{ .x = 256 } }, lm.s, &args);
    }

    /// The chunks' partials merged into `out`: one row block over its own cache.
    fn merge(lm: *Lm, s: config.Spec, out: u64, a: Addrs, pos: u64, rows: usize, nch: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ a.po, a.pm, a.pl, pos, out, a.xs }) |x| args.add(x);
        for ([_]usize{ s.heads, s.kv_heads, s.heads / s.kv_heads, chunk_keys, nch }) |x| args.add(@as(c_int, @intCast(x)));
        try cuda.launch.launch(lm.attn.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(s.kv_heads), .z = 1 }, .block = .{ .x = 256 } }, lm.s, &args);
    }
};

/// One decoder layer over every row; `first` is layer 0, whose residual is the embedding rows themselves.
fn layerRows(lm: *Lm, m: *const weights.Model, l: *const weights.Layer, a: Addrs, cap: usize, trunk: usize, ends: []const u32, total: usize, first: bool) !void {
    const s = m.spec;
    const qw = s.heads * s.head_dim;
    const kvw = s.kv_heads * s.head_dim;
    const eps: f32 = @floatCast(s.eps);
    try lm.nm.addRmsH(lm.s, a.x, if (first) null else a.z, l.input_norm, .f32, a.x, a.normed, null, total, s.hidden, eps);
    // the projected q goes to a.q, and the attention's rotated rows to a.attn, which the o_proj then reads
    try lm.g.run(lm.s, a.normed, l.q, a.q, total, qw, s.hidden);
    try lm.g.run(lm.s, a.normed, l.k, a.k, total, kvw, s.hidden);
    try lm.g.run(lm.s, a.normed, l.v, a.v, total, kvw, s.hidden);
    // the trunk's keys stay in the cache's first rows; each input after it attends to them and its own, from there
    if (trunk > 0) try attention(lm, m, l, a, cap, 0, trunk, 0);
    var begin: usize = trunk;
    for (ends) |e| {
        defer begin = e;
        try attention(lm, m, l, a, cap, begin, e, trunk);
    }
    try lm.g.run(lm.s, a.attn, l.o, a.z, total, s.hidden, qw);
    try lm.nm.addRmsH(lm.s, a.z, a.x, l.post_norm, .f32, a.x, a.normed, null, total, s.hidden, eps);
    try lm.g.run(lm.s, a.normed, l.gate, a.gate, total, s.inter, s.hidden);
    try lm.g.run(lm.s, a.normed, l.up, a.up, total, s.inter, s.hidden);
    // in place into gate: each thread reads its 8 values of gate and up before it writes them
    try lm.act.swiglu(lm.s, a.gate, s.inter, a.up, s.inter, a.gate, null, total, s.inter);
    try lm.g.run(lm.s, a.gate, l.down, a.z, total, s.hidden, s.inter);
}

/// Rows [begin, end) standing at positions from `at`: QK norm, RoPE and the cache write, then causal attention.
fn attention(lm: *Lm, m: *const weights.Model, l: *const weights.Layer, a: Addrs, cap: usize, begin: usize, end: usize, at: usize) !void {
    const s = m.spec;
    const hd = s.head_dim;
    const qw = s.heads * hd;
    const kvw = s.kv_heads * hd;
    const nch = cap / chunk_keys;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
    const heads: cuda.qk_rows.Heads = .{ .h = s.heads, .hk = s.kv_heads, .dim = hd };
    const shape: cuda.qk_rows.Shape = .{ .norms = l.q_norm, .norm_k = l.k_norm, .cos = a.cos, .sin = a.sin, .rot = hd, .eps = @floatCast(s.eps) };
    const len = end - begin;
    var p: usize = 0;
    while (p < len) : (p += attn_rows) {
        const rows = @min(attn_rows, len - p);
        const k = (begin + p) * kvw * 2;
        const out = a.attn + (begin + p) * qw * 2;
        const ld: i64 = @intCast(qw);
        const kld: i64 = @intCast(kvw);
        const hd64: i64 = @intCast(hd);
        const src: cuda.qk_rows.Src = .{ .q = a.q + (begin + p) * qw * 2, .q_ld = ld, .q_hs = hd64, .k = a.k + k, .k_ld = kld, .k_hs = hd64, .v = a.v + k, .v_ld = kld, .v_hs = hd64 };
        const dst: cuda.qk_rows.Out = .{ .q = out, .k = a.kb, .v = a.kb + attn_rows * kvw * 2, .kc = a.kc, .vc = a.vc, .len = cap };
        // a block's first row stands at position at + p: the position table is the meta the chunks read, shifted too
        const pos = a.pos + 4 * (at + p);
        try lm.qk.rowsFrom(lm.s, src, heads, shape, pos, null, dst, rows);
        try lm.chunks(s, dst.q, a, pos, rows, nch, scale);
        try lm.merge(s, out, a, pos, rows, nch);
    }
}
