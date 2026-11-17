//! The codec decoder's Metal launches: codec.metal by kernel, each with the threads of zonos3_codec.cu's launch.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;

fn i(x: usize) i32 {
    return @intCast(x);
}

fn l(x: usize) i64 {
    return @intCast(x);
}

/// The kernel's threadgroup copy of an input row holds this many values.
pub const max_k = 5120;

/// y [rows, n] = x [rows, kdim] W^T (+ bias when not 0), all fp32.
pub fn linear(g: *Gpu, w: u64, bias: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, rows: usize) void {
    const A = extern struct { xs: i32, K: i32, N: i32, ys: i32, has_bias: i32 };
    std.debug.assert(kdim <= max_k);
    g.go(.zc_linear, .{ (n + 7) / 8, rows, 1 }, 256, &.{ w, x, y, if (bias != 0) bias else w }, A{ .xs = i(xs), .K = i(kdim), .N = i(n), .ys = i(ys), .has_bias = @intFromBool(bias != 0) });
}

pub fn layernorm(g: *Gpu, x: u64, w: u64, b: u64, y: u64, d: usize, eps: f32, rows: usize) void {
    const A = extern struct { D: i32, eps: f32 };
    g.go(.zc_layernorm, .{ rows, 1, 1 }, 256, &.{ x, w, b, y }, A{ .D = i(d), .eps = eps });
}

pub fn attnPrep(g: *Gpu, qkv: u64, d: usize, pos0: usize, kc: u64, vc: u64, rows: usize) void {
    const A = extern struct { D: i32, pos0: i32 };
    g.go(.zc_attn_prep, .{ rows, d / 64, 1 }, 32, &.{ qkv, kc, vc }, A{ .D = i(d), .pos0 = i(pos0) });
}

pub fn attention(g: *Gpu, qkv: u64, d: usize, kc: u64, vc: u64, pos0: usize, ctx: usize, out: u64, rows: usize) void {
    const A = extern struct { D: i32, pos0: i32, ctx: i32 };
    g.go(.zc_attention, .{ rows, d / 64, 1 }, 128, &.{ qkv, kc, vc, out }, A{ .D = i(d), .pos0 = i(pos0), .ctx = i(ctx) });
}

/// x += scale * y per channel; scale == 0 is a plain add.
pub fn addScaled(g: *Gpu, x: u64, y: u64, scale: u64, d: usize, rows: usize) void {
    const A = extern struct { D: i32, has_scale: i32 };
    g.go(.zc_add_scaled, .{ (d + 255) / 256, rows, 1 }, 256, &.{ x, y, if (scale != 0) scale else y }, A{ .D = i(d), .has_scale = @intFromBool(scale != 0) });
}

pub fn gelu(g: *Gpu, x: u64, n: usize) void {
    const A = extern struct { n: u32 };
    g.go(.zc_gelu, .{ (n + 255) / 256, 1, 1 }, 256, &.{x}, A{ .n = @intCast(n) });
}

/// in [rows / 2][c] to out [rows][c / 2].
pub fn patchUp(g: *Gpu, in: u64, out: u64, c: usize, rows: usize) void {
    const A = extern struct { C: i32 };
    g.go(.zc_patch_up, .{ (c / 2 + 255) / 256, rows, 1 }, 256, &.{ in, out }, A{ .C = i(c) });
}

pub fn dwconv(g: *Gpu, x: u64, t: usize, w: u64, b: u64, hist: u64, out: u64, c: usize) void {
    const A = extern struct { T: i32, C: i32 };
    g.go(.zc_dwconv, .{ (c + 255) / 256, 1, 1 }, 256, &.{ x, w, b, hist, out }, A{ .T = i(t), .C = i(c) });
}

pub fn gather(g: *Gpu, codes: u64, col: usize, table: u64, out: u64, d: usize, t: usize) void {
    const A = extern struct { col: i32, D: i32 };
    g.go(.zc_gather, .{ (d + 255) / 256, t, 1 }, 256, &.{ codes, table, out }, A{ .col = i(col), .D = i(d) });
}

pub fn latent(g: *Gpu, codes: u64, tables: u64, z: u64, d: usize, t: usize) void {
    const A = extern struct { D: i32 };
    g.go(.zc_latent, .{ (d + 255) / 256, t, 1 }, 256, &.{ codes, tables, z }, A{ .D = i(d) });
}

pub fn spec(g: *Gpu, y: u64, half: usize, rows: usize) void {
    const A = extern struct { half: i32 };
    g.go(.zc_spec, .{ (half + 255) / 256, rows, 1 }, 256, &.{y}, A{ .half = i(half) });
}

pub fn ola(g: *Gpu, frames: u64, win2: u64, f: usize, n: usize, hop: usize, n0: usize, count: usize, out: u64) void {
    const A = extern struct { F: i32, N: i32, hop: i32, pad: i32 = 0, n0: i64, count: i64 };
    g.go(.zc_ola, .{ (count + 255) / 256, 1, 1 }, 256, &.{ frames, win2, out }, A{ .F = i(f), .N = i(n), .hop = i(hop), .n0 = l(n0), .count = l(count) });
}

/// Branch rows `stride` samples apart, read as zero outside [0, total).
pub fn mix(g: *Gpu, branches: u64, stride: usize, total: usize, w: u64, n0: usize, count: usize, out: u64) void {
    const A = extern struct { stride: i64, total: i64, n0: i64, count: i64 };
    g.go(.zc_mix, .{ (count + 255) / 256, 1, 1 }, 256, &.{ branches, w, out }, A{ .stride = l(stride), .total = l(total), .n0 = l(n0), .count = l(count) });
}

pub fn resample(g: *Gpu, x: u64, total: usize, kernel: u64, m0: usize, count: usize, out: u64) void {
    const A = extern struct { total: i64, m0: i64, count: i64 };
    g.go(.zc_resample, .{ (count + 255) / 256, 1, 1 }, 256, &.{ x, kernel, out }, A{ .total = l(total), .m0 = l(m0), .count = l(count) });
}
