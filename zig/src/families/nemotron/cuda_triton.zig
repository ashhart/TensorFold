//! Nemotron's Triton kernels from the captured cubins: each launch has the Python wrapper's grid and constexprs.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

/// glue._divisor: the largest K-slice count up to `want` dividing `n`.
pub fn divisor(n: usize, want: usize) usize {
    var g = @min(want, n);
    while (g > 1) : (g -= 1) if (n % g == 0) return g;
    return 1;
}

pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,

    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.run(t.s, name, grid, args, consts);
    }

    pub fn embed(t: Tri, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_embed", .{ u(rows), u(d / 64), 1 }, &.{ p("IDS", "*i32", ids), p("Wt", "*i32", w), p("S", "*bf16", s), p("B", "*bf16", b), p("OUT", "*bf16", out) }, &.{ci("D", d)});
    }

    /// base.add_rmsnorm: h = x + r (or x itself without r), y = rmsnorm(h) * w, xs = y's 64-group sums.
    pub fn addRmsnorm(t: Tri, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.run("_add_rmsnorm", .{ u(rows), 1, 1 }, &.{ p("X", "*bf16", x), p("R", "*bf16", r orelse x), p("W", "*bf16", w), p("H", "*bf16", if (r != null) h else x), p("Y", "*bf16", y), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)), ci("HAS_R", @intFromBool(r != null)) });
    }

    /// glue.add_moe_norm: the routed and shared expert sums in slot order, added to h, then its RMSNorm.
    pub fn addMoeNorm(t: Tri, h: u64, y: u64, y_f32: bool, wt: u64, w: u64, hn: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32, top_k: usize, slots: usize) !void {
        try t.run("_add_moe_norm", .{ u(rows), 1, 1 }, &.{ p("H", "*bf16", h), p("Y", if (y_f32) "*fp32" else "*bf16", y), p("WT", "*fp32", wt), p("W", "*bf16", w), p("HN", "*bf16", hn), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)), ci("NR", top_k), ci("NS", slots) });
    }

    /// glue.route: fp32 router partials over K slices, then each row's top-k and the shared halves' slots.
    pub fn route(t: Tri, x: u64, w: u64, bias: u64, part: u64, ids: u64, wts: u64, rows: usize, d: usize, e: usize, top_k: usize, scaling: f32, norm: bool) !void {
        const sk = divisor(d / 64, 6);
        try t.run("_router", .{ cdiv(rows, 16), cdiv(e, 16), u(sk) }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("PART", "*fp32", part), int("R", rows) }, &.{ ci("D", d), ci("E", e), ci("SK", sk), ci("BE", 16) });
        try t.run("_topk", .{ u(rows), 1, 1 }, &.{ p("PART", "*fp32", part), p("BIAS", "*fp32", bias), p("IDX", "*i32", ids), p("WT", "*fp32", wts), int("R", rows), aot.float("scaling", scaling) }, &.{ ci("E", e), ci("EP", pow2(e)), ci("SK", sk), ci("TOPK", top_k), ci("NS", top_k + 2), ci("NORM", @intFromBool(norm)) });
    }

    pub const Shape = struct { proj: usize, xd: usize, cd: usize, heads: usize, dh: usize, groups: usize, state: usize, rmax: usize };

    /// mamba.conv: the last window's kept rows replay from RAW, then the window's rows.
    pub fn conv(t: Tri, proj: u64, base: u64, raw: u64, xc: u64, cw: u64, cb: u64, meta: u64, rows: usize, m: Shape) !void {
        try t.run("_conv", .{ cdiv(m.cd, 256), 1, 1 }, &.{ p("P", "*bf16", proj), p("BASE", "*bf16", base), p("RAW", "*bf16", raw), p("XC", "*bf16", xc), p("CW", "*fp32", cw), p("CB", "*fp32", cb), p("META", "*i32", meta), int("R", rows) }, &.{ ci("PROJ", m.proj), ci("XOFF", m.xd), ci("CD", m.cd), ci("RMAX", m.rmax), ci("BC", 256) });
    }

    /// mamba.scan: the state lags a window; kept rows replay first in the same loop body.
    pub fn scan(t: Tri, proj: u64, xc: u64, dt: u64, state: u64, a: u64, dsk: u64, dtb: u64, meta: u64, y: u64, rows: usize, lo: f32, hi: f32, m: Shape) !void {
        try t.run("_scan", .{ u(m.heads), cdiv(m.dh, 32), 1 }, &.{ p("P", "*bf16", proj), p("XC", "*bf16", xc), p("DT", "*fp32", dt), p("S", "*fp32", state), p("A", "*fp32", a), p("DSK", "*fp32", dsk), p("DTB", "*fp32", dtb), p("META", "*i32", meta), p("Y", "*bf16", y), int("R", rows), aot.float("lo", lo), aot.float("hi", hi) }, &.{ ci("PROJ", m.proj), ci("XD", m.xd), ci("CD", m.cd), ci("DTOFF", m.xd + m.cd), ci("H", m.heads), ci("DH", m.dh), ci("NG", m.groups), ci("DS", m.state), ci("RMAX", m.rmax), ci("BD", 32) });
    }

    pub fn groupRmsnorm(t: Tri, x: u64, w: u64, out: u64, xs: u64, rows: usize, xd: usize, groups: usize, eps: f32) !void {
        try t.run("_group_rmsnorm", .{ u(rows), u(groups), 1 }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("XD", xd), ci("GS", xd / groups) });
    }

    /// mamba.conv_rows then commit_conv_rows: a chunk's rows in parallel, then BASE takes its last three inputs.
    pub fn convRows(t: Tri, proj: u64, base: u64, xc: u64, cw: u64, cb: u64, rows: usize, m: Shape) !void {
        try t.run("_conv_rows", .{ cdiv(rows, 16), cdiv(m.cd, 128), 1 }, &.{ p("P", "*bf16", proj), p("BASE", "*bf16", base), p("XC", "*bf16", xc), p("CW", "*fp32", cw), p("CB", "*fp32", cb), int("R", rows) }, &.{ ci("PROJ", m.proj), ci("XOFF", m.xd), ci("CD", m.cd), ci("BR", 16), ci("BC", 128) });
        try t.run("_conv_commit", .{ cdiv(m.cd, 128), 1, 1 }, &.{ p("P", "*bf16", proj), p("BASE", "*bf16", base), int("R", rows) }, &.{ ci("PROJ", m.proj), ci("XOFF", m.xd), ci("CD", m.cd), ci("BC", 128) });
    }

    pub const Attn = struct { nqkv: usize, heads: usize, kv_heads: usize, dim: usize, nch: usize };

    pub fn kvWrite(t: Tri, qkv: u64, kc: u64, vc: u64, meta: u64, rows: usize, a: Attn) !void {
        try t.run("_kv_write", .{ u(rows), 1, 1 }, &.{ p("QKV", "*bf16", qkv), p("KC", "*bf16", kc), p("VC", "*bf16", vc), p("META", "*i32", meta), int("R", rows) }, &.{ ci("NQKV", a.nqkv), ci("QD", a.heads * a.dim), ci("KVD", a.kv_heads * a.dim) });
    }

    /// attention.attention: 512-key chunks at absolute positions, merged in order (a row's bits ignore its window).
    pub fn attention(t: Tri, qkv: u64, kc: u64, vc: u64, meta: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, a: Attn) !void {
        const g = a.heads / a.kv_heads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(a.dim), -0.5));
        try t.run("_chunk", .{ u(rows), u(a.kv_heads), u(a.nch) }, &.{ p("QKV", "*bf16", qkv), p("KC", "*bf16", kc), p("VC", "*bf16", vc), p("META", "*i32", meta), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl) }, &.{ ci("NQKV", a.nqkv), ci("H", a.heads), ci("HK", a.kv_heads), ci("D", a.dim), ci("G", g), ci("CH", 512), ci("NCH", a.nch), aot.cf("SCALE", scale) });
        try t.run("_merge", .{ u(rows), u(a.kv_heads), 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("META", "*i32", meta), p("OUT", "*bf16", out), p("XS", "*fp32", xs) }, &.{ ci("H", a.heads), ci("HK", a.kv_heads), ci("D", a.dim), ci("G", g), ci("CH", 512), ci("NCH", a.nch) });
    }

    /// glue.concat_norms: the MTP input [rmsnorm(e) * enorm | rmsnorm(h) * hnorm] and its group sums.
    pub fn concatNorms(t: Tri, e: u64, h: u64, we: u64, wh: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.run("_concat_norms", .{ u(rows), 2, 1 }, &.{ p("E", "*bf16", e), p("Hd", "*bf16", h), p("WE", "*bf16", we), p("WH", "*bf16", wh), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)) });
    }

    /// sampler._keyed, one program a row: row r draws position META[0] + r + 1 + offset; without `prob` PROB is OUT, as Python passes it.
    pub fn keyed(t: Tri, vals: u64, ids: u64, meta: u64, out: u64, seed: u64, fp: u64, prob: ?u64, offset: usize, rows: usize, count: usize, k: Keyed) !void {
        const share = if (prob) |x| p("PROB", "*fp32", x) else p("PROB", "*i32", out);
        try t.run("_keyed", .{ u(rows), 1, 1 }, &.{ p("VALS", "*fp32", vals), p("IDS", "*i64", ids), p("META", "*i32", meta), p("OUT", "*i32", out), p("SEED", "*i64", seed), p("FP", "*fp64", fp), share, aot.word("c1", 0x9E3779B97F4A7C15), aot.word("c2", 0xD1B54A32D192ED03), aot.word("m1", 0xBF58476D1CE4E5B9), aot.word("m2", 0x94D049BB133111EB), int("offset", offset) }, &.{ ci("C", count), ci("CP", pow2(count)), ci("K", k.k), ci("CUT", @intFromBool(k.cut)), ci("GREEDY", @intFromBool(k.greedy)), ci("WRITE_PROB", @intFromBool(prob != null)), ci("MINP", @intFromBool(k.minp)), ci("CONF_T", @intFromBool(k.conf_t)) });
    }
};

/// The rule a _keyed launch compiles in: top_k, the top-p cut, greedy rank 0, the min-p cut, a draft's share at the request's temperature.
pub const Keyed = struct { k: usize, cut: bool = false, greedy: bool = false, minp: bool = false, conf_t: bool = false };

test "router K slices" {
    try std.testing.expectEqual(@as(usize, 6), divisor(42, 6));
    try std.testing.expectEqual(@as(usize, 4), divisor(28, 6));
}
