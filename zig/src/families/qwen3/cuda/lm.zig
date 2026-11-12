//! The Qwen3 dense prompt pass on CUDA's shared rows: norms, cuBLASLt projections, QK norm with RoPE, GQA chunks.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("../config.zig");
const gemm = @import("gemm.zig");
const store = @import("store.zig");
const weights = @import("weights.zig");

const Driver = cuda.Driver;
const Allocator = std.mem.Allocator;

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

/// The position table and the RoPE table for `cap` positions, uploaded into the block at `a`.
pub fn tables(gpa: Allocator, d: *const Driver, a: Addrs, spec: config.Spec, cap: usize) !void {
    try positions(gpa, d, a, cap);
    try rope(gpa, d, a, spec, cap);
}

/// Positions 0 to cap - 1: each row's rotation angle, and the first position a row block's chunks read.
fn positions(gpa: Allocator, d: *const Driver, a: Addrs, cap: usize) !void {
    const pos = try gpa.alloc(i32, cap);
    defer gpa.free(pos);
    for (pos, 0..) |*x, i| x.* = @intCast(i);
    try store.view(d, a.pos, cap * 4).upload(0, std.mem.sliceAsBytes(pos));
}

/// The RoPE table [cap, head_dim] of fp32 cosines and sines, each half of a head rotating by the same angle.
fn rope(gpa: Allocator, d: *const Driver, a: Addrs, spec: config.Spec, cap: usize) !void {
    const hd = spec.head_dim;
    const theta: f64 = spec.theta;
    const cs = try gpa.alloc(f32, cap * hd);
    defer gpa.free(cs);
    const sn = try gpa.alloc(f32, cap * hd);
    defer gpa.free(sn);
    var i: usize = 0;
    while (i < hd / 2) : (i += 1) {
        const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd)));
        var p: usize = 0;
        while (p < cap) : (p += 1) {
            const angle = @as(f64, @floatFromInt(p)) * inv;
            const c: f32 = @floatCast(@cos(angle));
            const s: f32 = @floatCast(@sin(angle));
            for ([_]usize{ i, i + hd / 2 }) |col| {
                cs[p * hd + col] = c;
                sn[p * hd + col] = s;
            }
        }
    }
    try store.view(d, a.cos, cs.len * 4).upload(0, std.mem.sliceAsBytes(cs));
    try store.view(d, a.sin, sn.len * 4).upload(0, std.mem.sliceAsBytes(sn));
}

/// `n` bytes at the block's cursor, a section unit in.
fn cut(cursor: *usize, n: usize) usize {
    cursor.* = store.up(cursor.*);
    defer cursor.* += n;
    return cursor.*;
}

/// Rows from elsewhere over the embedded ones: `n` rows of hidden at `ptr`, standing at positions [at, at + n).
pub const Splice = struct {
    ptr: u64,
    at: usize,
    n: usize,

    /// The rows standing in a pass over positions [pos, pos + rows): row `src` on, `len` of them, from pass row `dst`.
    pub fn within(sp: Splice, pos: usize, rows: usize) ?struct { src: usize, dst: usize, len: usize } {
        const lo = @max(sp.at, pos);
        const hi = @min(sp.at + sp.n, pos + rows);
        return if (lo < hi) .{ .src = lo - sp.at, .dst = lo - pos, .len = hi - lo } else null;
    }
};

/// Every layer's k and v caches of `cap` positions, in place of the scratch's one: layer i's at kc + i * stride.
pub const Cache = struct { kc: u64, vc: u64, stride: usize };

/// The final norm over the pass's last `rows` rows, then the tied head's logits [rows, vocab] into `logits`.
pub const Head = struct { rows: usize, logits: u64 };

/// A stage's rows (lm_embed, lm0_norm, lm0_attn, lm0_o, lm0_post, lm0_mlp, lm_layer{i}, lm_norm), after its launch.
pub const Tap = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void,
};

/// What a resumed or spliced pass adds; each field is read only when set, so `.{}` is the embeddings' call.
pub const Extra = struct {
    /// The input's first position: the pass continues a `cache` holding [0, start), one input only.
    start: usize = 0,
    cache: ?Cache = null,
    splice: ?Splice = null,
    head: ?Head = null,
    tap: ?Tap = null,
};

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

    /// Every input's final-normed last row into the block's `rows`, inputs independent and ending at `ends`.
    pub fn forward(lm: *Lm, m: *const weights.Model, a: Addrs, cap: usize, ends: []const u32) !void {
        return lm.forwardFrom(m, a, cap, ends, .{});
    }

    /// forward with `x`'s start, caches, splice, head and taps; refused before any launch when they do not fit.
    pub fn forwardFrom(lm: *Lm, m: *const weights.Model, a: Addrs, cap: usize, ends: []const u32, x: Extra) !void {
        const s = m.spec;
        const total = ends[ends.len - 1];
        if ((x.start > 0 or x.head != null) and (ends.len != 1 or x.cache == null)) return error.InvalidBatch;
        if (x.start + total > cap) return error.ContextFull;
        if (x.head) |h| if (h.rows == 0 or h.rows > total) return error.InvalidBatch;
        try lm.emb.bf16(lm.s, a.ids, m.embed, a.x, total, s.hidden);
        if (x.splice) |sp| if (sp.within(x.start, total)) |in| {
            const row = s.hidden * 2;
            try store.view(lm.d, a.x + in.dst * row, in.len * row).copyFrom(0, sp.ptr + in.src * row, in.len * row, lm.s.handle);
        };
        if (x.tap) |t| try t.put(t.ctx, "lm_embed", a.x, total, s.hidden);
        for (m.layers, 0..) |*l, i| try layerRows(lm, m, l, a, cap, ends, total, i, x);
        if (x.head) |h| return lm.head(m, a, total, h, x.tap);
        for (ends, 0..) |e, i| {
            // the last layer's residual add and the final norm, for the one row an input pools
            const row = (e - 1) * s.hidden * 2;
            try lm.nm.addRmsH(lm.s, a.z + row, a.x + row, m.final_norm, .f32, a.x + row, a.rows + i * s.hidden * 2, null, 1, s.hidden, @floatCast(s.eps));
        }
    }

    /// The final norm over the head's rows (every row when tapped), then their logits by the tied embedding table.
    fn head(lm: *Lm, m: *const weights.Model, a: Addrs, total: usize, h: Head, tap: ?Tap) !void {
        const s = m.spec;
        const rows = if (tap == null) h.rows else total;
        const at = (total - rows) * s.hidden * 2;
        try lm.nm.addRmsH(lm.s, a.z + at, a.x + at, m.final_norm, .f32, a.x + at, a.normed + at, null, rows, s.hidden, @floatCast(s.eps));
        if (tap) |t| {
            var name: [24]u8 = undefined;
            try t.put(t.ctx, try std.fmt.bufPrint(&name, "lm_layer{d}", .{m.layers.len - 1}), a.x, total, s.hidden);
            try t.put(t.ctx, "lm_norm", a.normed, total, s.hidden);
        }
        try lm.g.run(lm.s, a.normed + (total - h.rows) * s.hidden * 2, m.embed, h.logits, h.rows, s.vocab, s.hidden);
    }

    /// The chunk partials of a row block's `q`, over every chunk below its last key.
    fn chunks(lm: *Lm, s: config.Spec, q: u64, kc: u64, vc: u64, a: Addrs, pos: u64, rows: usize, nch: usize, scale: f32) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, pos, a.po, a.pm, a.pl }) |x| args.add(x);
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

/// Decoder layer `li` over every row; layer 0's residual is the embedding rows themselves.
fn layerRows(lm: *Lm, m: *const weights.Model, l: *const weights.Layer, a: Addrs, cap: usize, ends: []const u32, total: usize, li: usize, x: Extra) !void {
    const s = m.spec;
    const qw = s.heads * s.head_dim;
    const kvw = s.kv_heads * s.head_dim;
    const eps: f32 = @floatCast(s.eps);
    const first = li == 0;
    try lm.nm.addRmsH(lm.s, a.x, if (first) null else a.z, l.input_norm, .f32, a.x, a.normed, null, total, s.hidden, eps);
    const tap0 = if (first) x.tap else null; // layer 0's steps, as a case taps them
    if (x.tap) |t| if (!first) {
        var name: [24]u8 = undefined;
        try t.put(t.ctx, try std.fmt.bufPrint(&name, "lm_layer{d}", .{li - 1}), a.x, total, s.hidden);
    };
    if (tap0) |t| try t.put(t.ctx, "lm0_norm", a.normed, total, s.hidden);
    // the projected q goes to a.q, and the attention's rotated rows to a.attn, which the o_proj then reads
    try lm.g.run(lm.s, a.normed, l.q, a.q, total, qw, s.hidden);
    try lm.g.run(lm.s, a.normed, l.k, a.k, total, kvw, s.hidden);
    try lm.g.run(lm.s, a.normed, l.v, a.v, total, kvw, s.hidden);
    var begin: usize = 0;
    for (ends) |e| {
        defer begin = e;
        try attention(lm, m, l, a, cap, begin, e, li, x);
    }
    if (tap0) |t| try t.put(t.ctx, "lm0_attn", a.attn, total, qw);
    try lm.g.run(lm.s, a.attn, l.o, a.z, total, s.hidden, qw);
    if (tap0) |t| try t.put(t.ctx, "lm0_o", a.z, total, s.hidden);
    try lm.nm.addRmsH(lm.s, a.z, a.x, l.post_norm, .f32, a.x, a.normed, null, total, s.hidden, eps);
    if (tap0) |t| try t.put(t.ctx, "lm0_post", a.normed, total, s.hidden);
    try lm.g.run(lm.s, a.normed, l.gate, a.gate, total, s.inter, s.hidden);
    try lm.g.run(lm.s, a.normed, l.up, a.up, total, s.inter, s.hidden);
    // in place into gate: each thread reads its 8 values of gate and up before it writes them
    try lm.act.swiglu(lm.s, a.gate, s.inter, a.up, s.inter, a.gate, null, total, s.inter);
    try lm.g.run(lm.s, a.gate, l.down, a.z, total, s.hidden, s.inter);
    if (tap0) |t| try t.put(t.ctx, "lm0_mlp", a.z, total, s.hidden);
}

/// One input's rows [begin, end) at positions from x.start: QK norm, RoPE, the cache write, then causal attention.
fn attention(lm: *Lm, m: *const weights.Model, l: *const weights.Layer, a: Addrs, cap: usize, begin: usize, end: usize, li: usize, x: Extra) !void {
    const s = m.spec;
    const hd = s.head_dim;
    const qw = s.heads * hd;
    const kvw = s.kv_heads * hd;
    const nch = cap / chunk_keys;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
    const heads: cuda.qk_rows.Heads = .{ .h = s.heads, .hk = s.kv_heads, .dim = hd };
    const shape: cuda.qk_rows.Shape = .{ .norms = l.q_norm, .norm_k = l.k_norm, .cos = a.cos, .sin = a.sin, .rot = hd, .eps = @floatCast(s.eps) };
    const len = end - begin;
    // layer li's own caches when the pass keeps them, else the scratch's one, rewritten by every layer
    const kc, const vc = if (x.cache) |c| .{ c.kc + li * c.stride, c.vc + li * c.stride } else .{ a.kc, a.vc };
    var p: usize = 0;
    while (p < len) : (p += attn_rows) {
        const rows = @min(attn_rows, len - p);
        const k = (begin + p) * kvw * 2;
        const out = a.attn + (begin + p) * qw * 2;
        const ld: i64 = @intCast(qw);
        const kld: i64 = @intCast(kvw);
        const hd64: i64 = @intCast(hd);
        const src: cuda.qk_rows.Src = .{ .q = a.q + (begin + p) * qw * 2, .q_ld = ld, .q_hs = hd64, .k = a.k + k, .k_ld = kld, .k_hs = hd64, .v = a.v + k, .v_ld = kld, .v_hs = hd64 };
        const dst: cuda.qk_rows.Out = .{ .q = out, .k = a.kb, .v = a.kb + attn_rows * kvw * 2, .kc = kc, .vc = vc, .len = cap };
        // a block's first row stands at position start + p: the position table is the meta the chunks read, shifted too
        const pos = a.pos + 4 * (x.start + p);
        try lm.qk.rowsFrom(lm.s, src, heads, shape, pos, null, dst, rows);
        try lm.chunks(s, dst.q, kc, vc, a, pos, rows, nch, scale);
        try lm.merge(s, out, a, pos, rows, nch);
    }
}
