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
    embed: cuda.Function,
    embed16: cuda.Function,
    add_rmsnorm: cuda.Function,
    add_moe_norm: cuda.Function,
    concat_norms: cuda.Function,
    group_rmsnorm: cuda.Function,
    router: cuda.Function,
    topk: cuda.Function,
    conv: cuda.Function,
    conv_rows: cuda.Function,
    conv_commit: cuda.Function,
    scan: cuda.Function,
    kv_write: cuda.Function,
    attn_chunk: cuda.Function,
    attn_merge: cuda.Function,
    keyed: cuda.Function,

    /// The modules of nemotron_norms, _route, _mamba, _attention and _keyed, in that order.
    pub fn resolve(m: []const cuda.Module) !Fns {
        return .{
            .embed = try m[0].function("tf_nemo_embed"),
            .embed16 = try m[0].function("tf_nemo_embed16"),
            .add_rmsnorm = try m[0].function("tf_nemo_add_rmsnorm"),
            .add_moe_norm = try m[0].function("tf_nemo_add_moe_norm"),
            .concat_norms = try m[0].function("tf_nemo_concat_norms"),
            .group_rmsnorm = try m[0].function("tf_nemo_group_rmsnorm"),
            .router = try m[1].function("tf_nemo_router"),
            .topk = try m[1].function("tf_nemo_topk"),
            .conv = try m[2].function("tf_nemo_conv"),
            .conv_rows = try m[2].function("tf_nemo_conv_rows"),
            .conv_commit = try m[2].function("tf_nemo_conv_commit"),
            .scan = try m[2].function("tf_nemo_scan"),
            .kv_write = try m[3].function("tf_nemo_kv_write"),
            .attn_chunk = try m[3].function("tf_nemo_attn_chunk"),
            .attn_merge = try m[3].function("tf_nemo_attn_merge"),
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

/// The router's K slices and the shared bytes its block stages: 16 rows and 16 experts of one slice, padded a word.
pub fn routerShape(d: usize) struct { sk: usize, shared: u32 } {
    const sk = triton.divisor(d / 64, 6);
    return .{ .sk = sk, .shared = @intCast(2 * 16 * (d / sk / 2 + 1) * 4) };
}

/// Every launch of a window or prompt chunk's glue, on one stream.
pub const Glue = struct {
    set: ?*const cuda.aot.Set, // a captured Triton set (GB10's qualified one), else our own kernels
    f: *const Fns,
    s: cuda.Stream,

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
        var a: cuda.Args = .{};
        ptrs(&a, &.{ ids, w, s, b, out });
        a.add(int(d));
        try g.go(g.f.embed, .{ rows, 1, 1 }, 256, 0, &a);
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
        if (d > 4096) return error.RowTooWide;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ x, r orelse 0, w, h, y, xs });
        a.add(eps);
        a.add(int(d));
        try g.go(g.f.add_rmsnorm, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// The routed and shared expert sums in slot order, added to h, then its RMSNorm.
    pub fn addMoeNorm(g: Glue, h: u64, y: u64, y_f32: bool, wt: u64, w: u64, hn: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32, top_k: usize, slots: usize) !void {
        if (g.tri()) |t| return t.addMoeNorm(h, y, y_f32, wt, w, hn, out, xs, rows, d, eps, top_k, slots);
        if (d > 4096) return error.RowTooWide;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ h, y });
        a.add(@as(c_int, @intFromBool(y_f32)));
        ptrs(&a, &.{ wt, w, hn, out, xs });
        a.add(eps);
        ints(&a, &.{ d, top_k, slots });
        try g.go(g.f.add_moe_norm, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// fp32 router logits over K slices, then each row's top-k and the shared halves' slots.
    pub fn route(g: Glue, x: u64, w: u64, bias: u64, part: u64, ids: u64, wts: u64, rows: usize, d: usize, e: usize, top_k: usize, scaling: f32, norm: bool) !void {
        if (g.tri()) |t| return t.route(x, w, bias, part, ids, wts, rows, d, e, top_k, scaling, norm);
        if (e > 256 or top_k + 2 > 8) return error.RouterTooWide;
        const shape = routerShape(d);
        var a: cuda.Args = .{};
        ptrs(&a, &.{ x, w, part });
        ints(&a, &.{ rows, d, e, shape.sk });
        try g.go(g.f.router, .{ cdiv(rows, 16), cdiv(e, 16), shape.sk }, 256, shape.shared, &a);
        var b: cuda.Args = .{};
        ptrs(&b, &.{ part, bias, ids, wts });
        b.add(int(rows));
        b.add(scaling);
        ints(&b, &.{ e, shape.sk, top_k, top_k + 2 });
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
        var a: cuda.Args = .{};
        ptrs(&a, &.{ proj, xc, dt, state, a_, dsk, dtb, meta, y });
        a.add(int(rows));
        a.add(lo);
        a.add(hi);
        ints(&a, &.{ m.proj, m.xd, m.cd, m.xd + m.cd, m.heads, m.dh, m.groups, m.rmax });
        try g.go(g.f.scan, .{ m.heads, m.dh / 32, 1 }, 128, 0, &a);
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
        try g.go(g.f.attn_chunk, .{ rows, at.kv_heads, @min(at.nch, chunk_blocks) }, 256, 0, &a);
        var b: cuda.Args = .{};
        ptrs(&b, &.{ po, pm, pl, meta, out, xs });
        ints(&b, &.{ at.heads, at.kv_heads, grp, chunk_keys, at.nch });
        try g.go(g.f.attn_merge, .{ rows, at.kv_heads, 1 }, 256, 0, &b);
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
}
