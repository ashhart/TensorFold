//! Music 3's Metal launches: each wrapper binds its buffers and argument struct as zig/kernels/metal/music3 declares.
const std = @import("std");
const gpu = @import("gpu.zig");
const wts = @import("../weights.zig");

const Gpu = gpu.Gpu;
const H = 4096;

fn u(x: usize) u32 {
    return @intCast(x);
}

// -------------------------------------------------------------------------------------------- AR rows

pub fn embed(g: *Gpu, ids: u64, table: u64, out: u64, rows: usize) void {
    g.go(.m3_embed, .{ rows, 1, 1 }, 256, &.{ ids, table, out }, u(H));
}

pub fn embedCode(g: *Gpu, code: u64, offset: usize, table: u64, out: u64, rows: usize) void {
    const A = extern struct { offset: i32, rows: u32, dim: u32 };
    g.go(.m3_embed_code, .{ 1, 1, 1 }, 256, &.{ code, table, out }, A{ .offset = @intCast(offset), .rows = u(rows), .dim = H });
}

pub fn frameEmbed(g: *Gpu, codes: u64, audio: u64, extra: u64, out: u64, rows: usize) void {
    const A = extern struct { rows: u32, dim: u32, scale: f32 };
    g.go(.m3_frame_embed, .{ 16, 1, 1 }, 256, &.{ codes, audio, extra, out }, A{ .rows = u(rows), .dim = H, .scale = @floatCast(1.0 / @sqrt(8.0)) });
}

pub fn rmsnorm(g: *Gpu, x: u64, xs: usize, w: u64, y: u64, ys: usize, rows: usize, eps: f32) void {
    const A = extern struct { xs: u32, ys: u32, dim: u32, eps: f32 };
    g.go(.m3_rmsnorm, .{ rows, 1, 1 }, 256, &.{ x, w, y }, A{ .xs = u(xs), .ys = u(ys), .dim = H, .eps = eps });
}

pub fn add(g: *Gpu, x: u64, b: u64, n: usize) void {
    g.flat(.m3_add, n, &.{ x, b }, u(n));
}

pub fn swiglu(g: *Gpu, gu: u64, y: u64, width: usize, rows: usize) void {
    const A = extern struct { width: u32, rows: u32 };
    g.flat(.m3_swiglu, rows * width, &.{ gu, y }, A{ .width = u(width), .rows = u(rows) });
}

/// bf16 rows to int8 with a scale a row (group 0) or a scale a group of `group` inputs.
pub fn quant(g: *Gpu, src: u64, n: usize, k: usize, group: usize, q: u64, sc: u64) void {
    g.go(.m3_quant, .{ n, 1, 1 }, 256, &.{ src, q, sc }, [2]u32{ u(k), u(group) });
}

pub fn dequant(g: *Gpu, m: wts.Mat, out: u64) void {
    g.go(.m3_dequant, .{ (m.k + 255) / 256, m.n, 1 }, 256, &.{ m.w, m.sc, out }, [2]u32{ u(m.k), u(m.group) });
}

pub const gemv_sg = 2; // M3_GEMV_SG in ar.metal

pub const Gemv = struct {
    xs: usize = 0, // x row stride (0: K)
    glu: bool = false, // gate | up rows: write the SwiGLU of rows n and N / 2 + n at n
    y: u64 = 0,
    ys: usize = 0, // 0: N (N / 2 with glu)
    yf: u64 = 0,
    res: u64 = 0,
    res_div: usize = 1,
};

/// y[m] = W x[m] for m < rows (2 or 4): bf16, int8 a scale a row or a scale a group of 128, by the Mat.
pub fn gemv(g: *Gpu, w: wts.Mat, x: u64, rows: usize, o: Gemv) void {
    std.debug.assert(rows == 2 or rows == 4);
    std.debug.assert(w.group == 0 or w.group == 128);
    const four = rows == 4;
    const kind: usize = if (w.group != 0) 2 else if (w.sc != 0) 1 else 0;
    const table = [2][3][2]gpu.P{
        .{ .{ .m3_gemv_bf16_2, .m3_gemv_bf16_4 }, .{ .m3_gemv_i8_2, .m3_gemv_i8_4 }, .{ .m3_gemv_g8_2, .m3_gemv_g8_4 } },
        .{ .{ .m3_glu_bf16_2, .m3_glu_bf16_4 }, .{ .m3_glu_i8_2, .m3_glu_i8_4 }, .{ .m3_glu_g8_2, .m3_glu_g8_4 } },
    };
    const outs = if (o.glu) w.n / 2 else w.n;
    const A = extern struct { xs: u32, K: u32, N: u32, ys: u32, res_div: u32, has_y: u32, has_yf: u32, has_res: u32 };
    const a: A = .{ .xs = u(if (o.xs != 0) o.xs else w.k), .K = u(w.k), .N = u(w.n), .ys = u(if (o.ys != 0) o.ys else outs), .res_div = u(o.res_div), .has_y = @intFromBool(o.y != 0), .has_yf = @intFromBool(o.yf != 0), .has_res = @intFromBool(o.res != 0) };
    // gemv_sg simdgroups a group: a SwiGLU pair each, else two outputs each
    const per: usize = if (o.glu) gemv_sg else 2 * gemv_sg;
    g.go(table[@intFromBool(o.glu)][kind][@intFromBool(four)], .{ (outs + per - 1) / per, 1, 1 }, 32 * gemv_sg, &.{ w.w, w.sc, x, o.y, o.yf, o.res }, a);
}

/// Prompt rows: y [t, N] = x [t, K] W^T in bf16 (fp32 sums); int8 weights dequantized to `tmp` first.
pub fn linearRows(g: *Gpu, w: wts.Mat, x: u64, y: u64, t: usize, tmp: u64) void {
    var wp = w.w;
    if (w.sc != 0) {
        dequant(g, w, tmp);
        wp = tmp;
    }
    linear(g, .m3_linear_b, x, wp, 0, y, .{ .m = t, .n = w.n, .k = w.k });
}

pub fn qkRope(g: *Gpu, qkv: u64, qn: u64, kn: u64, inv_freq: u64, pos: u64, t: usize, rows: usize, q: u64, kc: u64, vc: u64, ctx: usize) void {
    const A = extern struct { t: u32, ctx: u32, eps: f32 };
    g.go(.m3_qk_rope, .{ rows, 48, 1 }, 128, &.{ qkv, qn, kn, inv_freq, pos, q, kc, vc }, A{ .t = u(t), .ctx = u(ctx), .eps = 1e-6 });
}

pub fn attention(g: *Gpu, q: u64, kc: u64, vc: u64, pos: u64, t: usize, rows: usize, ctx: usize, splits: usize, ml: u64, acc: u64, out: u64) void {
    const A = extern struct { t: u32, ctx: u32, splits: u32, scale: f32 };
    g.go(.m3_attn_part, .{ 8, rows, splits }, 128, &.{ q, kc, vc, pos, ml, acc }, A{ .t = u(t), .ctx = u(ctx), .splits = u(splits), .scale = @floatCast(1.0 / @sqrt(128.0)) });
    g.go(.m3_attn_merge, .{ rows, 32, 1 }, 128, &.{ ml, acc, out }, u(splits));
}

pub fn depthAttention(g: *Gpu, qkv: u64, p0: usize, rows: usize, kc: u64, vc: u64, out: u64) void {
    const A = extern struct { p0: u32, B: u32, scale: f32 };
    const a: A = .{ .p0 = u(p0), .B = 2, .scale = 1.0 / 16.0 };
    g.go(.m3_depth_kv, .{ rows, 16, 1 }, 256, &.{ qkv, kc, vc }, a);
    g.go(.m3_depth_attn, .{ rows, 16, 1 }, 256, &.{ qkv, kc, vc, out }, a);
}

pub const Draw = struct { cfg: f32, top_k: i32, seed: u64, step: u64, slot: i32, pick: u64, code: u64 = 0, tokens: u64 = 0, count: u64 = 0 };

pub const SampleArgs = extern struct { seed: u64, V: u32, cfg: f32, top_k: i32, rank_cond: i32, slot: i32, has_code: i32, has_tokens: i32, pad: i32 = 0 };

pub fn sample(g: *Gpu, logits: u64, v: usize, rank_cond: bool, d: Draw) void {
    const a: SampleArgs = .{ .seed = d.seed, .V = u(v), .cfg = d.cfg, .top_k = d.top_k, .rank_cond = @intFromBool(rank_cond), .slot = d.slot, .has_code = @intFromBool(d.code != 0), .has_tokens = @intFromBool(d.tokens != 0) };
    g.go(.m3_sample, .{ 1, 1, 1 }, 1024, &.{ logits, d.step, d.pick, d.code, d.tokens, d.count }, a);
}

pub fn frameDone(g: *Gpu, pos: u64, step: u64, count: u64) void {
    g.go(.m3_frame_done, .{ 1, 1, 1 }, 32, &.{ pos, step, count }, {});
}

pub fn statePart(g: *Gpu, src: u64, states: u64, count: u64, part: usize) void {
    g.go(.m3_state_part, .{ 1, 1, 1 }, 256, &.{ src, states, count }, u(part));
}

pub fn forceCodes(g: *Gpu, forced: u64, count: u64, codes: u64, tokens: u64) void {
    g.go(.m3_force_codes, .{ 1, 1, 1 }, 32, &.{ forced, count, codes, tokens }, {});
}

// -------------------------------------------------------------------------------------------- GEMMs

pub const Shape = struct {
    m: usize,
    n: usize,
    k: usize,
    ldx: usize = 0, // 0: k
    ldy: usize = 0, // 0: n
    batch: usize = 1,
    sx: usize = 0, // batch strides in elements
    sy: usize = 0,
    accumulate: bool = false,
    cs: u64 = 0, // m3_linear_rope_h: the rope table [Lseq, 16, 2]
    lseq: usize = 0, // m3_linear_rope_h: rows a sequence
};

/// Y[b] [m, n] = X[b] [m, k] W [n, k]^T (+ bias [n]) (+ Y[b]) on the tensor units; `p` picks the type and epilogue.
pub fn linear(g: *Gpu, p: gpu.P, x: u64, w: u64, bias: u64, y: u64, s: Shape) void {
    const a: LinearArgs = .{ .M = @intCast(s.m), .N = @intCast(s.n), .K = @intCast(s.k), .ldx = @intCast(if (s.ldx != 0) s.ldx else s.k), .ldy = @intCast(if (s.ldy != 0) s.ldy else s.n), .has_bias = @intFromBool(bias != 0), .accumulate = @intFromBool(s.accumulate), .pad = @intCast(s.lseq), .sx = @intCast(s.sx), .sy = @intCast(s.sy) };
    g.go(p, .{ (s.n + 127) / 128, (s.m + 63) / 64, s.batch }, 256, &.{ x, w, bias, y, s.cs }, a);
}

const LinearArgs = extern struct { M: i32, N: i32, K: i32, ldx: i32, ldy: i32, has_bias: i32, accumulate: i32, pad: i32 = 0, sx: i64, sy: i64 };

/// fp32 Y [m, n] (+)= X W^T from bf16 halves of both (x = xh + xl at xh, xl; w likewise), no bias.
pub fn linearSplit(g: *Gpu, xh: u64, xl: u64, wh: u64, wl: u64, y: u64, s: Shape) void {
    const a: LinearArgs = .{ .M = @intCast(s.m), .N = @intCast(s.n), .K = @intCast(s.k), .ldx = @intCast(if (s.ldx != 0) s.ldx else s.k), .ldy = @intCast(if (s.ldy != 0) s.ldy else s.n), .has_bias = 0, .accumulate = @intFromBool(s.accumulate), .sx = @intCast(s.sx), .sy = @intCast(s.sy) };
    g.go(.m3_linear_s, .{ (s.n + 127) / 128, (s.m + 63) / 64, s.batch }, 256, &.{ xh, xl, wh, wl, y }, a);
}

/// fp32 x [n] into bf16 halves hi + lo.
pub fn split(g: *Gpu, x: u64, hi: u64, lo: u64, n: usize) void {
    g.flat(.m3_split, n, &.{ x, hi, lo }, u(n));
}

test "argument structs match the kernels' layouts" {
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(SampleArgs));
}
