//! Nemotron's glue on CUDA: our own kernels (nemotron_*.cu), or the captured Triton set where one is loaded.

const std = @import("std");
const cuda = @import("cuda");
const triton = @import("cuda_triton.zig");

pub const Shape = triton.Tri.Shape;
pub const Attn = triton.Tri.Attn;
pub const Keyed = triton.Keyed;

/// Keys an attention chunk holds, at fixed absolute positions (Python's CHUNK).
pub const chunk_keys = 512;

/// Each attention block takes every chunk_blocks-th chunk, bounding idle blocks at long windows.
pub const chunk_blocks = 64;

/// The native kernels, by their extern "C" names.
pub const Fns = struct {
    embed16: cuda.Function,
    add_moe_norm: cuda.Function,
    concat_norms: cuda.Function,
    group_rmsnorm: cuda.Function,
    router: cuda.Function,
    router_rows: cuda.Function,
    topk: cuda.Function,
    conv: cuda.Function,
    conv_rows: cuda.Function,
    conv_commit: cuda.Function,
    scan: cuda.Function,
    conv_segs: cuda.Function,
    scan_segs: cuda.Function,
    kv_write: cuda.Function,
    attn_chunk: cuda.Function,
    attn_chunk_pf: cuda.Function,
    attn_merge: cuda.Function,
    kv_write_segs: cuda.Function,
    attn_chunk_segs: cuda.Function,
    attn_merge_segs: cuda.Function,
    keyed: cuda.Function,

    /// The modules of nemotron_norms, _route, _mamba, _attention and _keyed, in that order.
    pub fn resolve(m: []const cuda.Module) !Fns {
        return .{
            .embed16 = try m[0].function("tf_nemo_embed16"),
            .add_moe_norm = try m[0].function("tf_nemo_add_moe_norm"),
            .concat_norms = try m[0].function("tf_nemo_concat_norms"),
            .group_rmsnorm = try m[0].function("tf_nemo_group_rmsnorm"),
            .router = try m[1].function("tf_nemo_router"),
            .router_rows = try m[1].function("tf_nemo_router_rows"),
            .topk = try m[1].function("tf_nemo_topk"),
            .conv = try m[2].function("tf_nemo_conv"),
            .conv_rows = try m[2].function("tf_nemo_conv_rows"),
            .conv_commit = try m[2].function("tf_nemo_conv_commit"),
            .scan = try m[2].function("tf_nemo_scan"),
            .conv_segs = try m[2].function("tf_nemo_conv_segs"),
            .scan_segs = try m[2].function("tf_nemo_scan_segs"),
            .kv_write = try m[3].function("tf_nemo_kv_write"),
            .attn_chunk = try m[3].function("tf_nemo_attn_chunk"),
            .attn_chunk_pf = try m[3].function("tf_nemo_attn_chunk_pf"),
            .attn_merge = try m[3].function("tf_nemo_attn_merge"),
            .kv_write_segs = try m[3].function("tf_nemo_kv_write_segs"),
            .attn_chunk_segs = try m[3].function("tf_nemo_attn_chunk_segs"),
            .attn_merge_segs = try m[3].function("tf_nemo_attn_merge_segs"),
            .keyed = try m[4].function("tf_nemo_keyed_greedy"),
        };
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// A stream's Mamba buffers in a shared window, at one layer: its projection rows and its own state.
pub const MambaSeg = struct { proj: u64, base: u64, raw: u64, xc: u64, dt: u64, state: u64, meta: u64, y: u64, rows: usize };

/// Streams one conv or scan launch takes (nemotron_mamba.cu's MAXSEG).
pub const max_segs = 8;

const ConvSegs = extern struct { p: [max_segs]u64, base: [max_segs]u64, raw: [max_segs]u64, xc: [max_segs]u64, meta: [max_segs]u64, r: [max_segs]c_int };
const ScanSegs = extern struct { p: [max_segs]u64, xc: [max_segs]u64, dt: [max_segs]u64, s: [max_segs]u64, meta: [max_segs]u64, y: [max_segs]u64, r: [max_segs]c_int };

/// A stream's attention in a shared window, at one layer: its caches, meta row and first row of the shared rows.
pub const AttnSeg = struct { kc: u64, vc: u64, meta: u64, row0: usize, rows: usize };

const AttnSegs = extern struct { kc: [max_segs]u64, vc: [max_segs]u64, meta: [max_segs]u64, row0: [max_segs]c_int, n: c_int };

/// The router's K slices and the shared bytes its block stages: 16 rows and 16 experts of one slice, padded a word.
pub fn routerShape(d: usize) struct { sk: usize, shared: u32 } {
    var sk = triton.divisor(d / 64, 6);
    while (staged(d, sk) > 48 << 10 and (d / 64) % (sk * 2) == 0) sk *= 2; // wide rows (Bolt's): within 48 KiB, no opt-in
    return .{ .sk = sk, .shared = @intCast(staged(d, sk)) };
}

fn staged(d: usize, sk: usize) usize {
    return 2 * 16 * (d / sk / 2 + 1) * 4;
}

/// The most K slices routerShape takes for a row of `d`: the router's scratch is that many [rows, E] fp32 sums.
pub const max_router_slices = 8;

/// Every launch of a window or prompt chunk's glue, on one stream.
pub const Glue = struct {
    set: ?*const cuda.aot.Set, // a captured Triton set (GB10's qualified one), else our own kernels
    f: *const Fns,
    s: cuda.Stream,
    emb: *const cuda.embed_rows.EmbedRows, // the shared token rows (embed_rows.cu)
    norms: *const cuda.norms.Norms, // the shared RMS norm rows (norms.cu)

    fn tri(g: Glue) ?triton.Tri {
        return if (g.set) |x| .{ .set = x, .s = g.s } else null;
    }

    fn go(g: Glue, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared };
        try cuda.launch.launch(f, cfg, g.s, a);
    }

    fn ptrs(a: *cuda.Args, xs: []const u64) void {
        for (xs) |x| a.add(x);
    }

    fn ints(a: *cuda.Args, xs: []const usize) void {
        for (xs) |x| a.add(int(x));
    }

    /// MLX 4-bit token rows, dequantized.
    pub fn embed(g: Glue, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        if (g.tri()) |t| return t.embed(ids, w, s, b, out, rows, d);
        try g.emb.affine4(g.s, ids, w, s, b, out, rows, d);
    }

    /// Rows of a bf16 table [n, d] (ours with or without a captured set, whose embed reads MLX words).
    pub fn embed16(g: Glue, ids: u64, w: u64, out: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ ids, w, out });
        a.add(int(d));
        try g.go(g.f.embed16, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// h = x + r (or x itself without r), y = rmsnorm(h) * w, xs = y's 64-group sums.
    pub fn addRmsnorm(g: Glue, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        if (g.tri()) |t| return t.addRmsnorm(x, r, w, h, y, xs, rows, d, eps);
        try g.norms.addRmsH(g.s, x, r, w, .bf16, h, y, xs, rows, d, eps);
    }

    /// The routed and shared expert sums in slot order, added to h, then its RMSNorm.
    pub fn addMoeNorm(g: Glue, h: u64, y: u64, y_f32: bool, wt: u64, w: u64, hn: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32, top_k: usize, slots: usize) !void {
        if (g.tri()) |t| return t.addMoeNorm(h, y, y_f32, wt, w, hn, out, xs, rows, d, eps, top_k, slots);
        if (d > 4096) return error.RowTooWide;
        if (slots > 8) return error.TooManySlots; // the kernel's MAXS
        var a: cuda.Args = .{};
        ptrs(&a, &.{ h, y });
        a.add(@as(c_int, @intFromBool(y_f32)));
        ptrs(&a, &.{ wt, w, hn, out, xs });
        a.add(eps);
        ints(&a, &.{ d, top_k, slots });
        try g.go(g.f.add_moe_norm, .{ rows, 1, 1 }, 1024, 0, &a);
    }

    /// fp32 router logits over K slices, then each row's top-k and the shared halves' slots (`slots` past top-k).
    pub fn route(g: Glue, x: u64, w: u64, bias: u64, part: u64, ids: u64, wts: u64, rows: usize, d: usize, e: usize, top_k: usize, slots: usize, scaling: f32, norm: bool) !void {
        if (g.tri()) |t| return t.route(x, w, bias, part, ids, wts, rows, d, e, top_k, scaling, norm);
        if (e > 512 or top_k > 16 or slots < top_k or slots > top_k + 2) return error.RouterTooWide; // the top-k kernel's MAXC
        const shape = routerShape(d);
        if (shape.sk > 8) return error.RouterTooWide; // topk's MAXSK
        var a: cuda.Args = .{};
        ptrs(&a, &.{ x, w, part });
        ints(&a, &.{ rows, d, e, shape.sk });
        // decode: a block a (16 experts, slice) stages the rows and its experts, same bits
        if (rows <= 16 and d % (8 * shape.sk) == 0 and (x | w) % 16 == 0) {
            const row_shared: u32 = @intCast((rows + 16) * (d / shape.sk / 8 + 1) * 16);
            try g.go(g.f.router_rows, .{ cdiv(e, 16), shape.sk, 1 }, 256, row_shared, &a);
        } else {
            try g.go(g.f.router, .{ cdiv(rows, 16), cdiv(e, 16), shape.sk }, 256, shape.shared, &a);
        }
        var b: cuda.Args = .{};
        ptrs(&b, &.{ part, bias, ids, wts });
        b.add(int(rows));
        b.add(scaling);
        ints(&b, &.{ e, shape.sk, top_k, slots });
        b.add(@as(c_int, @intFromBool(norm)));
        try g.go(g.f.topk, .{ rows, 1, 1 }, 32, 0, &b);
    }

    /// The Mamba conv of a window: the last window's kept rows replay from RAW, then the window's rows.
    pub fn conv(g: Glue, proj: u64, base: u64, raw: u64, xc: u64, cw: u64, cb: u64, meta: u64, rows: usize, m: Shape) !void {
        if (g.tri()) |t| return t.conv(proj, base, raw, xc, cw, cb, meta, rows, m);
        var a: cuda.Args = .{};
        ptrs(&a, &.{ proj, base, raw, xc, cw, cb, meta });
        ints(&a, &.{ rows, m.proj, m.xd, m.cd, m.rmax });
        try g.go(g.f.conv, .{ cdiv(m.cd, 256), 1, 1 }, 256, 0, &a);
    }

    /// The Mamba scan of a window: the state lags a window; kept rows replay first in the same loop body.
    pub fn scan(g: Glue, proj: u64, xc: u64, dt: u64, state: u64, a_: u64, dsk: u64, dtb: u64, meta: u64, y: u64, rows: usize, lo: f32, hi: f32, m: Shape) !void {
        if (g.tri()) |t| return t.scan(proj, xc, dt, state, a_, dsk, dtb, meta, y, rows, lo, hi, m);
        if (m.state != 128 or m.dh % 32 != 0) return error.ScanShape;
        if (m.rmax > 16 or rows > m.rmax) return error.ScanShape; // the kernel steps at most 32 kept and window rows
        var a: cuda.Args = .{};
        ptrs(&a, &.{ proj, xc, dt, state, a_, dsk, dtb, meta, y });
        a.add(int(rows));
        a.add(lo);
        a.add(hi);
        ints(&a, &.{ m.proj, m.xd, m.cd, m.xd + m.cd, m.heads, m.dh, m.groups, m.rmax });
        try g.go(g.f.scan, .{ m.heads, m.dh / 32, 1 }, 128, 0, &a);
    }

    /// Every stream's conv, then every stream's scan, of a shared window: one launch each, or a stream at a time.
    pub fn mambaSegs(g: Glue, segs: []const MambaSeg, cw: u64, cb: u64, a_: u64, dsk: u64, dtb: u64, lo: f32, hi: f32, m: Shape) !void {
        if (g.set != null or segs.len < 2 or segs.len > max_segs) {
            for (segs) |s| {
                try g.conv(s.proj, s.base, s.raw, s.xc, cw, cb, s.meta, s.rows, m);
                try g.scan(s.proj, s.xc, s.dt, s.state, a_, dsk, dtb, s.meta, s.y, s.rows, lo, hi, m);
            }
            return;
        }
        if (m.state != 128 or m.dh % 32 != 0) return error.ScanShape;
        var cv = std.mem.zeroes(ConvSegs);
        var sc = std.mem.zeroes(ScanSegs);
        for (segs, 0..) |s, k| {
            cv.p[k] = s.proj;
            cv.base[k] = s.base;
            cv.raw[k] = s.raw;
            cv.xc[k] = s.xc;
            cv.meta[k] = s.meta;
            cv.r[k] = int(s.rows);
            sc.p[k] = s.proj;
            sc.xc[k] = s.xc;
            sc.dt[k] = s.dt;
            sc.s[k] = s.state;
            sc.meta[k] = s.meta;
            sc.y[k] = s.y;
            sc.r[k] = int(s.rows);
        }
        var a: cuda.Args = .{};
        a.add(cv);
        ptrs(&a, &.{ cw, cb });
        ints(&a, &.{ m.proj, m.xd, m.cd, m.rmax });
        try g.go(g.f.conv_segs, .{ cdiv(m.cd, 256), segs.len, 1 }, 256, 0, &a);
        var b: cuda.Args = .{};
        b.add(sc);
        ptrs(&b, &.{ a_, dsk, dtb });
        b.add(lo);
        b.add(hi);
        ints(&b, &.{ m.proj, m.xd, m.cd, m.xd + m.cd, m.heads, m.dh, m.groups, m.rmax });
        try g.go(g.f.scan_segs, .{ m.heads, m.dh / 32, segs.len }, 128, 0, &b);
    }

    /// Weight * RMSNorm over groups of xd / groups, and the 64-input group sums.
    pub fn groupRmsnorm(g: Glue, x: u64, w: u64, out: u64, xs: u64, rows: usize, xd: usize, groups: usize, eps: f32) !void {
        if (g.tri()) |t| return t.groupRmsnorm(x, w, out, xs, rows, xd, groups, eps);
        if (xd / groups > 1024) return error.RowTooWide;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ x, w, out, xs });
        a.add(eps);
        ints(&a, &.{ xd, xd / groups });
        try g.go(g.f.group_rmsnorm, .{ rows, groups, 1 }, 128, 0, &a);
    }

    /// A prompt chunk's conv rows in parallel, then BASE takes its last three inputs.
    pub fn convRows(g: Glue, proj: u64, base: u64, xc: u64, cw: u64, cb: u64, rows: usize, m: Shape) !void {
        if (g.tri()) |t| return t.convRows(proj, base, xc, cw, cb, rows, m);
        var a: cuda.Args = .{};
        ptrs(&a, &.{ proj, base, xc, cw, cb });
        ints(&a, &.{ rows, m.proj, m.xd, m.cd });
        try g.go(g.f.conv_rows, .{ rows, cdiv(m.cd, 256), 1 }, 256, 0, &a);
        var b: cuda.Args = .{};
        ptrs(&b, &.{ proj, base });
        ints(&b, &.{ rows, m.proj, m.xd, m.cd });
        try g.go(g.f.conv_commit, .{ cdiv(m.cd, 256), 1, 1 }, 256, 0, &b);
    }

    /// Each row's keys and values into the caches at meta's position.
    pub fn kvWrite(g: Glue, qkv: u64, kc: u64, vc: u64, meta: u64, rows: usize, at: Attn) !void {
        if (g.tri()) |t| return t.kvWrite(qkv, kc, vc, meta, rows, at);
        var a: cuda.Args = .{};
        ptrs(&a, &.{ qkv, kc, vc, meta });
        ints(&a, &.{ at.nqkv, at.heads * at.dim, at.kv_heads * at.dim });
        try g.go(g.f.kv_write, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// 512-key chunks at absolute positions, merged in order (a row's bits ignore its window).
    pub fn attention(g: Glue, qkv: u64, kc: u64, vc: u64, meta: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, at: Attn) !void {
        if (g.tri()) |t| return t.attention(qkv, kc, vc, meta, po, pm, pl, out, xs, rows, at);
        const grp = at.heads / at.kv_heads;
        if (at.dim != 128 or grp > 16) return error.AttentionShape;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(at.dim), -0.5));
        var a: cuda.Args = .{};
        ptrs(&a, &.{ qkv, kc, vc, meta, po, pm, pl });
        ints(&a, &.{ at.nqkv, at.heads, at.kv_heads, grp, chunk_keys, at.nch });
        a.add(scale);
        // several rows: each block loads its next key tile while it computes this one (one row gains nothing)
        try g.go(if (rows > 1) g.f.attn_chunk_pf else g.f.attn_chunk, .{ rows, at.kv_heads, @min(at.nch, chunk_blocks) }, 256, 0, &a);
        var b: cuda.Args = .{};
        ptrs(&b, &.{ po, pm, pl, meta, out, xs });
        ints(&b, &.{ at.heads, at.kv_heads, grp, chunk_keys, at.nch });
        try g.go(g.f.attn_merge, .{ rows, at.kv_heads, 1 }, 256, 0, &b);
    }

    /// Every stream's keys, values and attention over the shared rows: one launch each, or a stream at a time.
    pub fn attentionSegs(g: Glue, segs: []const AttnSeg, qkv: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, at: Attn) !void {
        var rows: usize = 0;
        for (segs) |s| rows += s.rows;
        if (g.set != null or segs.len < 2 or segs.len > max_segs) {
            const qd: u64 = at.heads * at.dim;
            for (segs) |s| {
                const r0: u64 = s.row0;
                const q = qkv + r0 * at.nqkv * 2;
                try g.kvWrite(q, s.kc, s.vc, s.meta, s.rows, at);
                try g.attention(q, s.kc, s.vc, s.meta, po + r0 * at.nch * qd * 4, pm + r0 * at.nch * at.heads * 4, pl + r0 * at.nch * at.heads * 4, out + r0 * qd * 2, xs + r0 * (qd / 64) * 4, s.rows, at);
            }
            return;
        }
        const grp = at.heads / at.kv_heads;
        if (at.dim != 128 or grp > 16) return error.AttentionShape;
        var t = std.mem.zeroes(AttnSegs);
        for (segs, 0..) |s, k| {
            t.kc[k] = s.kc;
            t.vc[k] = s.vc;
            t.meta[k] = s.meta;
            t.row0[k] = int(s.row0);
        }
        t.n = int(segs.len);
        var w: cuda.Args = .{};
        w.add(t);
        w.add(qkv);
        ints(&w, &.{ at.nqkv, at.heads * at.dim, at.kv_heads * at.dim });
        try g.go(g.f.kv_write_segs, .{ rows, 1, 1 }, 256, 0, &w);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(at.dim), -0.5));
        var a: cuda.Args = .{};
        a.add(t);
        ptrs(&a, &.{ qkv, po, pm, pl });
        ints(&a, &.{ at.nqkv, at.heads, at.kv_heads, grp, chunk_keys, at.nch });
        a.add(scale);
        try g.go(g.f.attn_chunk_segs, .{ rows, at.kv_heads, @min(at.nch, chunk_blocks) }, 256, 0, &a);
        var b: cuda.Args = .{};
        b.add(t);
        ptrs(&b, &.{ po, pm, pl, out, xs });
        ints(&b, &.{ at.heads, at.kv_heads, grp, chunk_keys, at.nch });
        try g.go(g.f.attn_merge_segs, .{ rows, at.kv_heads, 1 }, 256, 0, &b);
    }

    /// The MTP input [rmsnorm(e) * enorm | rmsnorm(h) * hnorm] and its group sums.
    pub fn concatNorms(g: Glue, e: u64, h: u64, we: u64, wh: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        if (g.tri()) |t| return t.concatNorms(e, h, we, wh, out, xs, rows, d, eps);
        if (d > 4096) return error.RowTooWide;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ e, h, we, wh, out, xs });
        a.add(eps);
        a.add(int(d));
        try g.go(g.f.concat_norms, .{ rows, 2, 1 }, 256, 0, &a);
    }

    /// A draw over candidate rows; our kernel takes the greedy rule (the head's draw), with or without its share.
    pub fn keyed(g: Glue, vals: u64, ids: u64, meta: u64, out: u64, seed: u64, fp: u64, prob: ?u64, offset: usize, rows: usize, count: usize, k: Keyed) !void {
        if (g.tri()) |t| return t.keyed(vals, ids, meta, out, seed, fp, prob, offset, rows, count, k);
        if (!k.greedy or count > 32) return error.KeyedRule;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ vals, ids, out, prob orelse 0 });
        ints(&a, &.{ count, @min(k.k, count) });
        try g.go(g.f.keyed, .{ rows, 1, 1 }, 32, 0, &a);
    }
};

test "the router stages one slice of 16 rows and 16 experts" {
    const s = routerShape(2688);
    try std.testing.expectEqual(@as(usize, 6), s.sk);
    try std.testing.expectEqual(@as(u32, 2 * 16 * 225 * 4), s.shared);
    // Bolt's 4096-wide rows: four slices would stage 66 KiB, eight stay under 48
    const b = routerShape(4096);
    try std.testing.expectEqual(@as(usize, 8), b.sk);
    try std.testing.expect(b.shared <= 48 << 10 and b.sk <= max_router_slices);
}
