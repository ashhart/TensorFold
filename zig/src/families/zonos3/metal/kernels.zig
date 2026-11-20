//! ZONOS3 Mini's Metal launches: backbone.metal by kernel, each with the threads of zonos3.cu's launch of that name.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;

fn i(x: usize) i32 {
    return @intCast(x);
}

pub fn embed(g: *Gpu, rows: u64, tables: u64, audio_rows: usize, out: u64, d: usize, t: usize) void {
    const A = extern struct { audio_rows: i32, D: i32 };
    g.go(.z3_embed, .{ t, 1, 1 }, 256, &.{ rows, tables, out }, A{ .audio_rows = i(audio_rows), .D = i(d) });
}

/// w == 0: the weightless norm.
pub fn rmsnorm(g: *Gpu, x: u64, xs: usize, w: u64, y: u64, ys: usize, d: usize, eps: f32, t: usize) void {
    const A = extern struct { xs: i32, ys: i32, D: i32, has_w: i32, eps: f32 };
    g.go(.z3_rmsnorm, .{ t, 1, 1 }, 256, &.{ x, if (w != 0) w else x, y }, A{ .xs = i(xs), .ys = i(ys), .D = i(d), .has_w = @intFromBool(w != 0), .eps = eps });
}

pub fn attnres(g: *Gpu, bank: u64, n_src: usize, cap: usize, query: u64, out: u64, d: usize, eps: f32, t: usize) void {
    const A = extern struct { S: i32, cap: i32, D: i32, eps: f32 };
    std.debug.assert(n_src <= 64);
    g.go(.z3_attnres, .{ t, 1, 1 }, 256, &.{ bank, query, out }, A{ .S = i(n_src), .cap = i(cap), .D = i(d), .eps = eps });
}

pub const Linear = struct { bias: u64 = 0, picks: u64 = 0, pick_stride: usize = 0, xrep: usize = 1 };

/// The longest input row the kernel's threadgroup copy holds (32 KB of threadgroup memory).
pub const max_k = 8192;

/// y [rows, n] = x [rows / xrep, kdim] W^T: bf16 weights, fp32 rows xs / ys elements apart.
pub fn linear(g: *Gpu, w: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, rows: usize, o: Linear) void {
    const A = extern struct { xs: i32, xrep: i32, K: i32, N: i32, ys: i32, has_bias: i32, has_picks: i32, pad: i32 = 0, pick_stride: i64 };
    std.debug.assert(kdim % 8 == 0 and kdim <= max_k);
    const a: A = .{ .xs = i(xs), .xrep = i(o.xrep), .K = i(kdim), .N = i(n), .ys = i(ys), .has_bias = @intFromBool(o.bias != 0), .has_picks = @intFromBool(o.picks != 0), .pick_stride = @intCast(o.pick_stride) };
    g.goShared(.z3_linear, .{ (n + 7) / 8, rows, 1 }, 256, &.{ w, x, y, if (o.bias != 0) o.bias else w, if (o.picks != 0) o.picks else x }, a, kdim * 4);
}

pub fn convSilu(g: *Gpu, proj: u64, ps: usize, t: usize, w: u64, state: u64, out: u64, c: usize) void {
    const A = extern struct { ps: i32, T: i32, C: i32 };
    g.go(.z3_conv_silu, .{ (c + 255) / 256, 1, 1 }, 256, &.{ proj, w, state, out }, A{ .ps = i(ps), .T = i(t), .C = i(c) });
}

pub fn gdnPrep(g: *Gpu, conv: u64, proj: u64, ps: usize, a_off: usize, a_log: u64, dt_bias: u64, gates: u64, t: usize) void {
    const A = extern struct { ps: i32, a_off: i32 };
    g.go(.z3_gdn_prep, .{ t, 4, 1 }, 128, &.{ conv, proj, a_log, dt_bias, gates }, A{ .ps = i(ps), .a_off = i(a_off) });
}

pub fn gdn(g: *Gpu, conv: u64, gates: u64, t: usize, state: u64, o: u64) void {
    const A = extern struct { T: i32 };
    g.go(.z3_gdn, .{ 4, 1, 1 }, 128, &.{ conv, gates, state, o }, A{ .T = i(t) });
}

pub fn gatedNorm(g: *Gpu, o: u64, proj: u64, ps: usize, z_off: usize, w: u64, y: u64, t: usize) void {
    const A = extern struct { ps: i32, z_off: i32 };
    g.go(.z3_gated_norm, .{ t, 4, 1 }, 128, &.{ o, proj, w, y }, A{ .ps = i(ps), .z_off = i(z_off) });
}

pub fn attnPrep(g: *Gpu, q: u64, kv: u64, temp: u64, pos0: usize, kc: u64, vc: u64, theta: f32, t: usize) void {
    const A = extern struct { pos0: i32, theta: f32 };
    g.go(.z3_attn_prep, .{ t, 1, 1 }, 512, &.{ q, kv, temp, kc, vc }, A{ .pos0 = i(pos0), .theta = theta });
}

pub fn attention(g: *Gpu, q: u64, kc: u64, vc: u64, gate: u64, pos0: usize, out: u64, t: usize) void {
    const A = extern struct { pos0: i32 };
    g.go(.z3_attention, .{ t, 16, 1 }, 256, &.{ q, kc, vc, gate, out }, A{ .pos0 = i(pos0) });
}

/// out [rows, n] = in[a_off ..] * silu(in[b_off ..]).
pub fn mulSilu(g: *Gpu, in: u64, stride: usize, a_off: usize, b_off: usize, n: usize, out: u64, rows: usize) void {
    const A = extern struct { stride: i32, a_off: i32, b_off: i32, n: i32 };
    g.go(.z3_mul_silu, .{ (n + 255) / 256, rows, 1 }, 256, &.{ in, out }, A{ .stride = i(stride), .a_off = i(a_off), .b_off = i(b_off), .n = i(n) });
}

pub fn route(g: *Gpu, logits: u64, bias: u64, e: usize, picks: u64, w: u64, t: usize) void {
    const A = extern struct { E: i32 };
    g.go(.z3_route, .{ t, 1, 1 }, 32, &.{ logits, bias, picks, w }, A{ .E = i(e) });
}

pub fn combine(g: *Gpu, down: u64, w: u64, n: usize, out: u64, t: usize) void {
    const A = extern struct { n: i32 };
    g.go(.z3_combine, .{ (n + 255) / 256, t, 1 }, 256, &.{ down, w, out }, A{ .n = i(n) });
}

pub fn softcap(g: *Gpu, x: u64, n: usize, cap: f32) void {
    const A = extern struct { n: u32, cap: f32 };
    g.go(.z3_softcap, .{ (n + 255) / 256, 1, 1 }, 256, &.{x}, A{ .n = @intCast(n), .cap = cap });
}
